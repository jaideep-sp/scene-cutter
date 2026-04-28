import subprocess
import glob
import hashlib
import json
import os
import sys
from pathlib import Path


def extract_audio(video_path: str, work_dir: str) -> str:
    """Extract full-quality stereo WAV for demucs stem separation."""
    raw_wav = os.path.join(work_dir, "raw_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "2",          # stereo — demucs is trained on stereo
        "-ar", "44100",      # 44.1 kHz — demucs native sample rate
        "-vn",
        raw_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return raw_wav


def _video_fingerprint(video_path: str) -> str:
    """SHA-256 of the first 4 MB of the video file — fast, stable identity check."""
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()


def separate_stems(raw_wav: str, work_dir: str, video_path: str = "") -> dict:
    """Run demucs two-stem separation (vocals / no_vocals) on the WAV file."""
    stem_meta = os.path.join(work_dir, ".stem_source.json")
    vocals_matches    = glob.glob(os.path.join(work_dir, "**", "vocals.wav"),    recursive=True)
    no_vocals_matches = glob.glob(os.path.join(work_dir, "**", "no_vocals.wav"), recursive=True)

    current_fp = _video_fingerprint(video_path) if video_path else ""

    stems_valid = False
    if vocals_matches and no_vocals_matches and os.path.exists(stem_meta):
        with open(stem_meta) as f:
            saved = json.load(f)
        stems_valid = saved.get("fingerprint") == current_fp

    if stems_valid:
        print("       stems already exist for this video, skipping demucs.")
        return {
            "raw":       raw_wav,
            "vocals":    vocals_matches[0],
            "no_vocals": no_vocals_matches[0],
        }

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",
        "--out", work_dir,
        raw_wav,
    ]
    subprocess.run(cmd, check=True)

    vocals_matches    = glob.glob(os.path.join(work_dir, "**", "vocals.wav"),    recursive=True)
    no_vocals_matches = glob.glob(os.path.join(work_dir, "**", "no_vocals.wav"), recursive=True)

    if not vocals_matches or not no_vocals_matches:
        raise FileNotFoundError(
            f"Demucs output not found under {work_dir}. "
            "Found: " + str(glob.glob(os.path.join(work_dir, "**", "*.wav"), recursive=True))
        )

    with open(stem_meta, "w") as f:
        json.dump({"fingerprint": current_fp, "source": video_path}, f)

    return {
        "raw":       raw_wav,
        "vocals":    vocals_matches[0],
        "no_vocals": no_vocals_matches[0],
    }


def extract_and_separate(video_path: str, work_dir: str) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    raw_wav = extract_audio(video_path, work_dir)
    return separate_stems(raw_wav, work_dir, video_path=video_path)
