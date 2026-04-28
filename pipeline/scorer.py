import os
import json
import time
import numpy as np
import config
from concurrent.futures import ThreadPoolExecutor, as_completed
from pipeline.audio_prep import extract_and_separate
from pipeline.speech     import get_vad_segments
from pipeline.windows    import build_unsafe_zones, build_safe_windows
from pipeline.video      import get_scene_boundaries, score_motion_at
from pipeline.music      import preload_audio, score_music_at_preloaded
from pipeline.semantics  import load_clip_model, precompute_text_features, score_clip_neutrality


def _cache_path(work_dir: str, name: str) -> str:
    return os.path.join(work_dir, f".cache_{name}.json")


def _load_cache(work_dir: str, name: str):
    path = _cache_path(work_dir, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def _save_cache(work_dir: str, name: str, data) -> None:
    with open(_cache_path(work_dir, name), "w") as f:
        json.dump(data, f)


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def _step(n: int, total: int, msg: str):
    print(f"\n[{n}/{total}] {msg}", flush=True)


def _done(elapsed: float):
    print(f"       done in {elapsed:.1f}s", flush=True)


def _find_window(timestamp: float, safe_windows: list[dict]):
    for w in safe_windows:
        if w["start"] <= timestamp <= w["end"]:
            return w
    return None


def _score_one(
    timestamp: float,
    safe_windows: list[dict],
    scene_boundaries: list[float],
    no_vocals_y: np.ndarray,
    no_vocals_sr: int,
    video_path: str,
    clip_model,
    clip_preprocess,
    neutral_feats,
    active_feats,
    fps: float,
) -> dict:
    window = _find_window(timestamp, safe_windows)

    if window is None:
        return {
            "timestamp": timestamp,
            "frame":     int(timestamp * fps),
            "score":     -999.0,
            "signals":   {"label": "speech_nearby"},
        }

    silence_dur     = min(window["duration"], 5.0)
    center_distance = abs(timestamp - window["center"]) / max(window["duration"] / 2.0, 1e-6)
    center_distance = min(center_distance, 1.0)

    motion          = score_motion_at(video_path, timestamp)
    music           = score_music_at_preloaded(no_vocals_y, no_vocals_sr, timestamp)
    clip_neutrality = score_clip_neutrality(
        video_path, timestamp,
        clip_model, clip_preprocess,
        neutral_feats, active_feats,
        config.DEVICE,
    )

    boundary_dist = min(abs(timestamp - b) for b in scene_boundaries) if scene_boundaries else 999.0
    near_boundary = boundary_dist <= 2.0

    score  = 0.0
    score += min(silence_dur, 5.0)   * 6
    score += (1.0 - center_distance) * 20
    score -= min(motion, 4.375)      * 8
    if music["peak_rms"] > config.MUSIC_RMS_THRESHOLD:
        score -= music["peak_rms"]   * 25
    if music["beat_strength"] > config.BEAT_STRENGTH_CAP:
        score -= music["beat_strength"] * 3
    score += clip_neutrality         * 15
    if near_boundary:
        score += 10.0

    return {
        "timestamp": timestamp,
        "frame":     int(timestamp * fps),
        "score":     round(score, 2),
        "signals": {
            "label":           "ok",
            "silence":         round(window["duration"], 2),
            "motion":          round(motion, 3),
            "music_rms":       round(music["rms"], 3),
            "peak_rms":        round(music["peak_rms"], 3),
            "beat_strength":   round(music["beat_strength"], 3),
            "clip_neutrality": round(clip_neutrality, 3),
            "boundary_dist":   round(boundary_dist, 2),
            "near_boundary":   near_boundary,
        },
    }


def _sample_candidates(safe_windows: list[dict]) -> list[float]:
    candidates = []
    n = config.WINDOW_SAMPLES
    for w in safe_windows:
        if n == 1:
            candidates.append(w["center"])
        else:
            step = w["duration"] / (n + 1)
            for i in range(1, n + 1):
                candidates.append(w["start"] + step * i)
    return candidates


def _video_cache_key(video_path: str) -> str:
    """First 8 chars of the SHA-256 of the first 4 MB — short, stable ID for this file."""
    import hashlib
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()[:8]


def find_best_cuts(video_path: str, work_dir: str) -> list[dict]:
    TOTAL = 6
    t0 = time.time()

    _step(1, TOTAL, "Extracting audio and separating stems (demucs)…")
    t = time.time()
    stems = extract_and_separate(video_path, work_dir)
    _done(time.time() - t)
    print(f"       vocals    → {stems['vocals']}")
    print(f"       no_vocals → {stems['no_vocals']}")

    _step(2, TOTAL, "Running SileroVAD (language-agnostic voice detection)…")
    t = time.time()
    vad_segs = get_vad_segments(stems["vocals"])
    _done(time.time() - t)
    print(f"       detected {len(vad_segs)} speech segment(s)")
    total_speech = sum(s["end"] - s["start"] for s in vad_segs)
    print(f"       total speech duration: {total_speech:.1f}s")

    import ffmpeg
    probe    = ffmpeg.probe(video_path)
    duration = float(probe["format"]["duration"])
    fps      = float(next(
        s["r_frame_rate"].split("/")[0] for s in probe["streams"] if s["codec_type"] == "video"
    ))
    print(f"       video duration: {_fmt_ts(duration)}  fps: {fps:.2f}")

    _step(3, TOTAL, "Building safe windows…")
    t = time.time()
    unsafe  = build_unsafe_zones([], vad_segs, duration)
    windows = build_safe_windows(unsafe, duration)
    _done(time.time() - t)
    print(f"       unsafe zones : {len(unsafe)}")
    print(f"       safe windows : {len(windows)}")

    if not windows:
        print("WARNING: No safe windows found. Try reducing SPEECH_BUFFER in config.py.")
        return []

    total_safe = sum(w["duration"] for w in windows)
    print(f"       total safe duration: {total_safe:.1f}s  ({100 * total_safe / duration:.1f}% of video)")

    _step(4, TOTAL, "Detecting scene boundaries (OpenCV)…")
    t = time.time()
    boundaries = get_scene_boundaries(video_path)
    _done(time.time() - t)
    print(f"       found {len(boundaries)} scene boundary/boundaries")

    _step(5, TOTAL, "Loading OpenCLIP and pre-encoding text prompts…")
    t = time.time()
    clip_model, clip_preprocess, clip_tokenizer = load_clip_model(config.DEVICE)
    neutral_feats, active_feats = precompute_text_features(clip_model, clip_tokenizer, config.DEVICE)
    _done(time.time() - t)
    print(f"       text features pre-encoded (will reuse across all candidates)")

    _step(6, TOTAL, "Preloading no_vocals stem and scoring candidates…")
    t = time.time()

    candidates   = _sample_candidates(windows)
    vid_key      = _video_cache_key(video_path)
    cache_name   = f"scored_{vid_key}_ws{config.WINDOW_SAMPLES}"
    scored       = _load_cache(work_dir, cache_name)

    if scored is not None:
        print(f"       loaded {len(scored)} scored candidates from cache (.cache_{cache_name}.json)")
    else:
        nv_y, nv_sr = preload_audio(stems["no_vocals"])
        workers     = min(os.cpu_count() or 4, len(candidates))
        print(f"       {len(candidates)} candidates across {len(windows)} windows")
        print(f"       parallelising across {workers} threads…")

        scored  = []
        done_n  = 0
        futures = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for ts in candidates:
                f = pool.submit(
                    _score_one,
                    ts, windows, boundaries,
                    nv_y, nv_sr, video_path,
                    clip_model, clip_preprocess,
                    neutral_feats, active_feats,
                    fps,
                )
                futures[f] = ts

            for f in as_completed(futures):
                result  = f.result()
                done_n += 1
                scored.append(result)
                print(
                    f"       [{done_n:>3}/{len(candidates)}] "
                    f"{_fmt_ts(result['timestamp'])}  "
                    f"frame={result['frame']}  "
                    f"score={result['score']}",
                    flush=True,
                )

        _save_cache(work_dir, cache_name, scored)
        _done(time.time() - t)

    scored.sort(key=lambda x: x["score"], reverse=True)

    selected = []
    for item in scored:
        if item["score"] <= -999:
            continue
        too_close = any(
            abs(item["timestamp"] - s["timestamp"]) < config.MIN_CUT_SPACING
            for s in selected
        )
        if not too_close:
            selected.append(item)
        # N_CUTS=0 means return all valid candidates
        if config.N_CUTS > 0 and len(selected) >= config.N_CUTS:
            break

    # Sort final selection by timestamp for logical clip ordering
    selected.sort(key=lambda x: x["timestamp"])

    print(f"\n   Total pipeline time: {time.time() - t0:.1f}s")
    return selected
