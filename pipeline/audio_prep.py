import subprocess
import glob
import os
import sys
from pathlib import Path


def extract_audio(video_path: str, work_dir: str) -> str:
    """Extract mono 16 kHz WAV from input video."""
    raw_wav = os.path.join(work_dir, "raw_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1",          # mono
        "-ar", "16000",      # 16 kHz
        "-vn",               # no video
        raw_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return raw_wav


def separate_stems(raw_wav: str, work_dir: str) -> dict:
    """Run demucs two-stem separation (vocals / no_vocals) on the WAV file."""
    # Skip separation if stems already exist from a previous run.
    vocals_matches    = glob.glob(os.path.join(work_dir, "**", "vocals.wav"),    recursive=True)
    no_vocals_matches = glob.glob(os.path.join(work_dir, "**", "no_vocals.wav"), recursive=True)

    if vocals_matches and no_vocals_matches:
        print("       stems already exist, skipping demucs.")
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

    return {
        "raw":       raw_wav,
        "vocals":    vocals_matches[0],
        "no_vocals": no_vocals_matches[0],
    }


def extract_and_separate(video_path: str, work_dir: str) -> dict:
    os.makedirs(work_dir, exist_ok=True)
    raw_wav = extract_audio(video_path, work_dir)
    return separate_stems(raw_wav, work_dir)
