import os
import re
import shutil
import datetime
import subprocess


def _resolve_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/usr/bin/ffmpeg"):
        if os.path.exists(candidate):
            return candidate
    return "ffmpeg"
import requests
import torch
import torchaudio
import av
import folder_paths


# ── ElevenLabs Batch SFX ──────────────────────────────────────────────────────

class ElevenLabsBatchSFX:
    """
    Parses #-delimited SFX prompts from Gemini, calls ElevenLabs for each,
    saves individual MP3s, and outputs a manifest (filepath|start_sec per line)
    so MergeVideoWithSFX can place every clip at the right timestamp.
    """

    _TIME_RE = re.compile(r"\[\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*s?\]")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts_string":   ("STRING", {"forceInput": True}),
                "api_key":          ("STRING", {"default": "", "multiline": False}),
                "default_duration": ("FLOAT",  {"default": 2.0, "min": 0.5, "max": 22.0, "step": 0.5}),
                "prompt_influence": ("FLOAT",  {"default": 0.3, "min": 0.0, "max": 1.0,  "step": 0.05}),
                "output_format":    (["mp3_44100_192", "mp3_44100_128", "pcm_44100"], {"default": "mp3_44100_192"}),
                "filename_prefix":  ("STRING", {"default": "sfx"}),
                "delimiter":        ("STRING", {"default": "#"}),
            }
        }

    RETURN_TYPES  = ("STRING", "STRING")
    RETURN_NAMES  = ("output_log", "sfx_manifest")
    FUNCTION      = "generate_all"
    OUTPUT_NODE   = True
    CATEGORY      = "audio/ElevenLabs"

    def _parse_prompt(self, raw: str, default_duration: float):
        """Returns (clean_text, duration_sec, start_sec)."""
        m = self._TIME_RE.search(raw)
        if m:
            start    = float(m.group(1))
            end      = float(m.group(2))
            duration = round(max(0.5, min(22.0, end - start)), 2)
            clean    = self._TIME_RE.sub("", raw).strip().lstrip(":").strip()
            return clean, duration, start
        return raw.strip(), default_duration, 0.0

    def generate_all(self, prompts_string, api_key, default_duration,
                     prompt_influence, output_format, filename_prefix, delimiter):

        raw_prompts = [p.strip() for p in prompts_string.split(delimiter) if p.strip()]
        if not raw_prompts:
            msg = "No prompts found — check Gemini output uses '#' as separator."
            print(f"[ElevenLabsBatchSFX] {msg}")
            return (msg, "")

        ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = os.path.join(folder_paths.get_output_directory(), "audio", f"{filename_prefix}_{ts}")
        os.makedirs(out_dir, exist_ok=True)

        logs             = [f"Found {len(raw_prompts)} prompt(s) → {out_dir}"]
        manifest_entries = []          # list of (start_sec, filepath)

        for i, raw in enumerate(raw_prompts):
            clean, duration, start_sec = self._parse_prompt(raw, default_duration)
            print(f"[ElevenLabsBatchSFX] [{i+1}/{len(raw_prompts)}] "
                  f"t={start_sec}s dur={duration}s | {clean[:80]}")
            try:
                resp = requests.post(
                    "https://api.elevenlabs.io/v1/sound-generation",
                    params  = {"output_format": output_format},
                    headers = {"xi-api-key": api_key, "Content-Type": "application/json"},
                    json    = {"text": clean, "duration_seconds": duration,
                               "prompt_influence": prompt_influence},
                    timeout = 60,
                )
                if resp.status_code == 200:
                    ext      = "mp3" if "mp3" in output_format else "wav"
                    filename = f"{filename_prefix}_{i+1:03d}.{ext}"
                    filepath = os.path.join(out_dir, filename)
                    with open(filepath, "wb") as f:
                        f.write(resp.content)
                    manifest_entries.append((start_sec, filepath))
                    log = f"[{i+1}] OK  t={start_sec}s  dur={duration}s → {filename}"
                else:
                    log = f"[{i+1}] HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:
                log = f"[{i+1}] Exception: {exc}"

            print(f"[ElevenLabsBatchSFX] {log}")
            logs.append(log)

        # manifest: one "filepath|start_sec" per line
        manifest = "\n".join(f"{fp}|{st}" for st, fp in manifest_entries)
        return ("\n".join(logs), manifest)


# ── Merge Video with SFX ──────────────────────────────────────────────────────

class MergeVideoWithSFX:
    """
    Places every SFX clip at its recorded timestamp on a silent audio timeline,
    then combines the mixed audio track with the original video via ffmpeg.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sfx_manifest":         ("STRING",  {"forceInput": True}),
                "keep_original_audio":  ("BOOLEAN", {"default": False}),
                "sfx_volume":           ("FLOAT",   {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "output_filename":      ("STRING",  {"default": "merged_output"}),
            },
            "optional": {
                # Connect a LoadVideo (or any VIDEO) output here for dynamic input.
                "video":                ("VIDEO",),
                # Fallback: filename in ComfyUI's input folder, or absolute path.
                "video_filename":       ("STRING",  {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION     = "merge"
    OUTPUT_NODE  = True
    CATEGORY     = "audio/ElevenLabs"

    # ── helpers ────────────────────────────────────────────────────────────────

    def _video_duration(self, video_path: str) -> float | None:
        try:
            with av.open(video_path) as c:
                if c.duration:
                    return c.duration / 1_000_000   # microseconds → seconds
                for s in c.streams.video:
                    if s.duration and s.time_base:
                        return float(s.duration * s.time_base)
        except Exception as e:
            print(f"[MergeVideoWithSFX] Could not read duration: {e}")
        return None

    def _video_has_audio(self, video_path: str) -> bool:
        try:
            with av.open(video_path) as c:
                return len(c.streams.audio) > 0
        except Exception:
            return False

    def _mix_sfx_timeline(self, manifest: str, video_duration: float | None,
                           sfx_volume: float, sample_rate: int = 44100) -> torch.Tensor | None:
        """
        Builds a stereo float32 tensor with every SFX placed at its timestamp.
        """
        entries = []
        for line in manifest.strip().splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                filepath, start_str = parts
                try:
                    entries.append((float(start_str), filepath))
                except ValueError:
                    pass

        if not entries:
            return None

        # Determine total length
        if video_duration:
            total_samples = int(video_duration * sample_rate)
        else:
            # fallback: longest SFX end + 2 s buffer
            max_end = 0.0
            for start, fp in entries:
                try:
                    info    = torchaudio.info(fp)
                    max_end = max(max_end, start + info.num_frames / info.sample_rate)
                except Exception:
                    max_end = max(max_end, start + 5.0)
            total_samples = int((max_end + 2) * sample_rate)

        mixed = torch.zeros(2, total_samples)

        for start_sec, filepath in entries:
            try:
                waveform, sr = torchaudio.load(filepath)
                if sr != sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
                # Normalise to stereo
                if waveform.shape[0] == 1:
                    waveform = waveform.repeat(2, 1)
                elif waveform.shape[0] > 2:
                    waveform = waveform[:2]

                waveform     *= sfx_volume
                start_sample  = int(start_sec * sample_rate)
                clip_len      = min(waveform.shape[1], total_samples - start_sample)
                if clip_len > 0 and start_sample < total_samples:
                    mixed[:, start_sample:start_sample + clip_len] += waveform[:, :clip_len]
            except Exception as e:
                print(f"[MergeVideoWithSFX] Warning – skipping {filepath}: {e}")

        # Prevent clipping
        peak = mixed.abs().max()
        if peak > 1.0:
            mixed /= peak

        return mixed

    # ── main ───────────────────────────────────────────────────────────────────

    def merge(self, sfx_manifest, keep_original_audio, sfx_volume,
              output_filename, video=None, video_filename=""):

        # ── 1. Locate video ───────────────────────────────────────────────────
        video_path = None

        # Preferred: VIDEO input from upstream node (e.g. LoadVideo)
        if video is not None:
            try:
                src = video.get_stream_source()
                if isinstance(src, str) and os.path.exists(src):
                    video_path = src
            except Exception as e:
                print(f"[MergeVideoWithSFX] Could not read VIDEO input: {e}")
            if video_path is None:
                # Not a file-backed VIDEO (e.g. in-memory BytesIO) — dump bytes to temp mp4.
                # Avoid video.save_to() which requires torchcodec.
                tmp_dir = folder_paths.get_temp_directory()
                os.makedirs(tmp_dir, exist_ok=True)
                video_path = os.path.join(
                    tmp_dir, f"merge_src_{datetime.datetime.now().strftime('%H%M%S%f')}.mp4"
                )
                try:
                    src = video.get_stream_source()
                    if hasattr(src, "read"):
                        src.seek(0)
                        with open(video_path, "wb") as f:
                            f.write(src.read())
                    else:
                        return (f"Error: unsupported VIDEO source type: {type(src).__name__}",)
                except Exception as e:
                    return (f"Error: could not materialize VIDEO input: {e}",)

        # Fallback: filename string (absolute path or inside ComfyUI/input)
        if video_path is None and video_filename:
            candidate = video_filename
            if not os.path.isabs(candidate):
                in_dir = os.path.join(folder_paths.get_input_directory(), candidate)
                candidate = in_dir if os.path.exists(in_dir) else candidate
            if os.path.exists(candidate):
                video_path = candidate

        if video_path is None:
            return ("Error: no video provided (connect VIDEO input or set video_filename)",)

        video_dur = self._video_duration(video_path)
        print(f"[MergeVideoWithSFX] Video: {video_path}  duration: {video_dur}s")

        # ── 2. Mix SFX onto timeline ──────────────────────────────────────────
        mixed = self._mix_sfx_timeline(sfx_manifest, video_dur, sfx_volume)
        if mixed is None:
            return ("Error: no valid entries in sfx_manifest",)

        # ── 3. Save mixed audio as temp WAV ───────────────────────────────────
        tmp_dir   = folder_paths.get_temp_directory()
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_audio = os.path.join(tmp_dir, f"sfx_mix_{datetime.datetime.now().strftime('%H%M%S%f')}.wav")
        # Write WAV via stdlib to avoid torchaudio.save (requires torchcodec in newer versions)
        import wave
        pcm = (mixed.clamp(-1.0, 1.0) * 32767.0).to(torch.int16).t().contiguous().numpy().tobytes()
        with wave.open(tmp_audio, "wb") as wf:
            wf.setnchannels(mixed.shape[0])
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(pcm)
        print(f"[MergeVideoWithSFX] Mixed audio → {tmp_audio}")

        # ── 4. Combine video + audio with ffmpeg ──────────────────────────────
        out_dir  = os.path.join(folder_paths.get_output_directory(), "video")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{output_filename}.mp4")

        has_audio = self._video_has_audio(video_path)

        if keep_original_audio and has_audio:
            # Mix original audio + SFX
            cmd = [
                _resolve_ffmpeg(), "-y",
                "-i", video_path,
                "-i", tmp_audio,
                "-filter_complex",
                "[0:a][1:a]amix=inputs=2:duration=first:weights=1 1",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                out_path,
            ]
        else:
            # Replace audio entirely with SFX track
            cmd = [
                _resolve_ffmpeg(), "-y",
                "-i", video_path,
                "-i", tmp_audio,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k",
                "-shortest",
                out_path,
            ]

        print(f"[MergeVideoWithSFX] Running ffmpeg…")
        result = subprocess.run(cmd, capture_output=True, text=True)

        # ── 5. Cleanup temp file ──────────────────────────────────────────────
        try:
            os.remove(tmp_audio)
        except Exception:
            pass

        if result.returncode != 0:
            err = (result.stderr or "")[-600:]
            print(f"[MergeVideoWithSFX] ffmpeg error:\n{err}")
            return (f"ffmpeg error: {err}",)

        print(f"[MergeVideoWithSFX] Done → {out_path}")
        return (out_path,)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "ElevenLabsBatchSFX":  ElevenLabsBatchSFX,
    "MergeVideoWithSFX":   MergeVideoWithSFX,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ElevenLabsBatchSFX":  "ElevenLabs Batch SFX (Dynamic)",
    "MergeVideoWithSFX":   "Merge Video with SFX",
}
