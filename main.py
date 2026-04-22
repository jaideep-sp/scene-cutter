#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import config


def fmt_timestamp(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m):02d}:{s:05.2f}"


def print_results(cuts: list[dict], n_cuts: int) -> None:
    print()
    print("=" * 60)
    limit = f"capped at {n_cuts}" if n_cuts > 0 else "all valid"
    print(f"{len(cuts)} CUT POINTS  ({limit}, min spacing {config.MIN_CUT_SPACING}s)")
    print("=" * 60)

    for i, cut in enumerate(cuts, 1):
        ts  = cut["timestamp"]
        sc  = cut["score"]
        sig = cut["signals"]

        boundary_line = (
            f"    boundary_dist={sig['boundary_dist']}s  *** near scene boundary ***"
            if sig.get("near_boundary")
            else f"    boundary_dist={sig['boundary_dist']}s"
        )

        print(f"\n#{i}  {fmt_timestamp(ts)}  frame={cut['frame']}  score={sc}")
        print(
            f"    silence={sig['silence']}s  "
            f"motion={sig['motion']}  "
            f"music_rms={sig['music_rms']}  "
            f"clip={sig['clip_neutrality']}"
        )
        print(boundary_line)


def write_csv(cuts: list[dict], csv_path: str, video_path: str, pipeline_seconds: float) -> None:
    import csv
    import ffmpeg
    probe    = ffmpeg.probe(video_path)
    duration = float(probe["format"]["duration"])
    fps      = float(next(
        s["r_frame_rate"].split("/")[0] for s in probe["streams"] if s["codec_type"] == "video"
    ))
    timestamps = [0.0] + sorted(c["timestamp"] for c in cuts) + [duration]

    # Build segment duration lookup keyed by cut timestamp
    seg_dur = {}
    for i in range(len(timestamps) - 1):
        seg_dur[round(timestamps[i + 1], 3)] = round(timestamps[i + 1] - timestamps[i], 3)
    seg_dur[round(timestamps[0], 3)] = round(timestamps[1] - timestamps[0], 3)

    fields = [
        "rank", "timecode", "timestamp_s", "frame",
        "segment_start_s", "segment_end_s", "segment_duration_s",
        "score",
        "silence_s", "motion", "music_rms", "peak_rms",
        "beat_strength", "clip_neutrality",
        "boundary_dist_s", "near_boundary",
        "pipeline_time_s",
    ]

    sorted_cuts = sorted(cuts, key=lambda x: x["timestamp"])
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, cut in enumerate(sorted_cuts):
            ts   = round(cut["timestamp"], 3)
            prev = timestamps[timestamps.index(ts) - 1] if ts in timestamps else 0.0
            w.writerow({
                "rank":                cut["rank"],
                "timecode":            cut["timecode"],
                "timestamp_s":         ts,
                "frame":               cut["frame"],
                "segment_start_s":     round(timestamps[timestamps.index(ts) - 1], 3) if ts in timestamps else "",
                "segment_end_s":       ts,
                "segment_duration_s":  seg_dur.get(ts, ""),
                "score":               cut["score"],
                "silence_s":           cut["signals"]["silence"],
                "motion":              cut["signals"]["motion"],
                "music_rms":           cut["signals"]["music_rms"],
                "peak_rms":            cut["signals"]["peak_rms"],
                "beat_strength":       cut["signals"]["beat_strength"],
                "clip_neutrality":     cut["signals"]["clip_neutrality"],
                "boundary_dist_s":     cut["signals"]["boundary_dist"],
                "near_boundary":       cut["signals"]["near_boundary"],
                "pipeline_time_s":     pipeline_seconds,
            })
    print(f"CSV written  → {csv_path}")


def write_json(cuts: list[dict], json_path: str) -> None:
    payload = [
        {
            "rank":           i + 1,
            "timestamp":      round(c["timestamp"], 3),
            "timecode":       fmt_timestamp(c["timestamp"]),
            "frame":          c["frame"],
            "score":          c["score"],
            "signals":        c["signals"],
        }
        for i, c in enumerate(cuts)
    ]
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON written → {json_path}")


def split_video(video_path: str, cuts: list[dict], clips_dir: str) -> None:
    """Split video at each cut point using ffmpeg stream-copy (fast, no re-encode)."""
    os.makedirs(clips_dir, exist_ok=True)

    # Build segment boundaries: 0 → cut1 → cut2 → … → end
    import ffmpeg
    probe    = ffmpeg.probe(video_path)
    duration = float(probe["format"]["duration"])

    timestamps = [0.0] + sorted(c["timestamp"] for c in cuts) + [duration]

    print(f"\nSplitting into {len(timestamps) - 1} segment(s) → {clips_dir}")

    for i in range(len(timestamps) - 1):
        t_start = timestamps[i]
        t_end   = timestamps[i + 1]
        tc      = fmt_timestamp(t_start).replace(":", "-")
        out     = os.path.join(clips_dir, f"segment_{i + 1:03d}_{tc}.mp4")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t_start),
            "-to", str(t_end),
            "-i", video_path,
            "-c:v", "libx264",     # re-encode to H.264 for universal compatibility
            "-preset", "fast",
            "-crf", "18",          # high quality (0=lossless, 51=worst)
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            out,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        seg_dur = t_end - t_start
        print(f"  segment {i + 1:03d}  {fmt_timestamp(t_start)} → {fmt_timestamp(t_end)}  "
              f"({seg_dur:.1f}s)  → {os.path.basename(out)}")

    print("Done splitting.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find safe cut points in an MP4 (no speech, action, or dramatic music)."
    )
    parser.add_argument("input", help="Path to input MP4 file")
    parser.add_argument(
        "--out",
        default="./cut_analysis",
        metavar="DIR",
        help="Work directory for intermediates and output (default: ./cut_analysis)",
    )
    parser.add_argument(
        "--n-cuts",
        type=int,
        default=None,
        metavar="INT",
        help="Max cut points to return (default: 0 = all valid cuts). "
             "Cuts are always spaced at least MIN_CUT_SPACING seconds apart.",
    )
    parser.add_argument(
        "--profile",
        choices=["drama", "action", "documentary", "animation"],
        default=None,
        help="Content-type preset that overrides threshold defaults",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split the input video at each cut point using ffmpeg (stream-copy, no re-encode). "
             "Segments saved to <out>/clips/",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Error: input file not found: {args.input}")

    if args.profile:
        config.apply_profile(args.profile)
        print(f"Profile '{args.profile}' applied.")
    if args.n_cuts is not None:
        config.N_CUTS = args.n_cuts

    from pipeline.scorer import find_best_cuts
    import time

    work_dir = os.path.abspath(args.out)
    os.makedirs(work_dir, exist_ok=True)

    t0   = time.time()
    cuts = find_best_cuts(args.input, work_dir)
    pipeline_seconds = round(time.time() - t0, 1)

    if not cuts:
        print("No cut points found. See warnings above.")
        sys.exit(1)

    print_results(cuts, config.N_CUTS)

    # Enrich cuts with rank/timecode for CSV
    enriched = [
        {**c, "rank": i + 1, "timecode": fmt_timestamp(c["timestamp"])}
        for i, c in enumerate(cuts)
    ]

    json_path = os.path.join(work_dir, "cut_points.json")
    write_json(enriched, json_path)

    csv_path = os.path.join(work_dir, "cut_points.csv")
    write_csv(enriched, csv_path, args.input, pipeline_seconds)

    if args.split:
        clips_dir = os.path.join(work_dir, "clips")
        split_video(args.input, cuts, clips_dir)


if __name__ == "__main__":
    main()
