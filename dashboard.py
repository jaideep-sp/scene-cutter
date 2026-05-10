#!/usr/bin/env python3
"""Streamlit dashboard — pipeline monitor, cut-point inspector, clip browser."""
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

st.set_page_config(page_title="Scene Cutter", layout="wide", page_icon="🎬")

# ── session-state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "logs":          [],
    "cuts":          [],
    "cpu_history":   [],
    "selected_clip": None,
    "running":       False,
    "done":          False,
    "work_dir":      "",
    "video_path":    "",
    "exit_code":     None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── constants ─────────────────────────────────────────────────────────────────
PHASES = [
    ("Extract audio / demucs",       "Extracting audio"),
    ("SileroVAD speech detection",   "Running SileroVAD"),
    ("Build safe windows",           "Building safe windows"),
    ("Scene boundary detection",     "Detecting scene boundaries"),
    ("Load OpenCLIP",                "Loading OpenCLIP"),
    ("Score candidates",             "Preloading no_vocals"),
]
VENV_PYTHON = str(Path(__file__).parent / ".venv" / "bin" / "python")
PYTHON      = VENV_PYTHON if Path(VENV_PYTHON).exists() else sys.executable

# ── helpers ───────────────────────────────────────────────────────────────────
def _completed_phases(logs: list[str]) -> int:
    return sum(1 for ln in logs if "done in" in ln)


def _active_phase(logs: list[str]) -> int:
    for ln in reversed(logs):
        for i, (_, keyword) in enumerate(PHASES):
            if keyword.lower() in ln.lower():
                return i
    return -1


def _thumbnail(src: str, ts: float, out: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-ss", str(ts), "-i", src, "-vframes", "1", "-q:v", "3", out],
        capture_output=True,
    )
    return r.returncode == 0


def _probe_duration(path: str) -> float:
    try:
        import ffmpeg
        info = ffmpeg.probe(path)
        return float(info["format"]["duration"])
    except Exception:
        return 0.0


def _fmt(s: float) -> str:
    m, sec = divmod(s, 60)
    return f"{int(m):02d}:{sec:05.2f}"


def _read_video(path: str, max_mb: int = 250) -> bytes | None:
    size_mb = os.path.getsize(path) / 1e6
    if size_mb > max_mb:
        return None
    with open(path, "rb") as f:
        return f.read()


def _load_clips(clips_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(clips_dir, "segment_*.mp4")))


def _csv_path() -> str:
    return os.path.join(st.session_state.work_dir, "cut_points.csv")


def _json_path() -> str:
    return os.path.join(st.session_state.work_dir, "cut_points.json")


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 Scene Cutter")
    st.markdown("---")

    video_input = st.text_input(
        "Video file path",
        placeholder="/path/to/episode.mp4",
        value=st.session_state.video_path,
    )
    work_input = st.text_input(
        "Output directory",
        value=st.session_state.work_dir or "./cut_analysis",
    )
    profile = st.selectbox(
        "Content profile",
        ["(none)", "drama", "action", "documentary", "animation"],
    )
    n_cuts   = st.number_input("Max cuts (0 = all valid)", min_value=0, value=0, step=1)
    do_split = st.checkbox("Split into clips", value=True)

    st.markdown("---")
    run_btn = st.button(
        "▶ Run Pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.running,
    )

    if st.session_state.running:
        st.warning("⏳ Pipeline running…")
    elif st.session_state.done:
        code = st.session_state.exit_code
        if code == 0:
            st.success(f"✅ Done — {len(st.session_state.cuts)} cut point(s)")
        else:
            st.error(f"❌ Exited with code {code}")

    # load existing results without re-running
    if st.sidebar.button("↩ Load existing results", use_container_width=True):
        wd = os.path.abspath(work_input)
        st.session_state.work_dir   = wd
        jp = os.path.join(wd, "cut_points.json")
        if os.path.exists(jp):
            with open(jp) as f:
                data = json.load(f)
                if isinstance(data, dict):
                    st.session_state.cuts = data.get("cut_points", [])
                    if data.get("video"):
                        st.session_state.video_path = data.get("video")
                else:
                    st.session_state.cuts = data
            st.session_state.done = True
            st.session_state.exit_code = 0
            st.rerun()
        else:
            st.warning("No cut_points.json found in that directory.")

# ── run pipeline ──────────────────────────────────────────────────────────────
if run_btn and video_input:
    abs_work = os.path.abspath(work_input)
    st.session_state.update({
        "video_path":  video_input,
        "work_dir":    abs_work,
        "logs":        [],
        "cuts":        [],
        "cpu_history": [],
        "running":     True,
        "done":        False,
        "exit_code":   None,
    })

    cmd = [PYTHON, "main.py", video_input, "--out", abs_work]
    if profile != "(none)":
        cmd += ["--profile", profile]
    if n_cuts > 0:
        cmd += ["--n-cuts", str(n_cuts)]
    if do_split:
        cmd.append("--split")

    st.info(f"Running: `{' '.join(cmd)}`")
    
    # Placeholders for live updates
    cpu_placeholder = st.empty()
    log_placeholder = st.empty()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(__file__).parent),
    )
    
    # Set to non-blocking read
    import fcntl
    fd = proc.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    while proc.poll() is None:
        # 1. Update CPU
        cpu = psutil.cpu_percent(interval=None)
        st.session_state.cpu_history.append(cpu)
        if len(st.session_state.cpu_history) > 120: # keep 2 mins
            st.session_state.cpu_history.pop(0)
        
        with cpu_placeholder.container():
            st.caption(f"Live CPU Usage: {cpu}%")
            st.area_chart(st.session_state.cpu_history, height=150)

        # 2. Update Logs
        try:
            line = proc.stdout.readline()
            if line:
                st.session_state.logs.append(line.rstrip())
                log_placeholder.code("\n".join(st.session_state.logs[-40:]), language=None)
        except BlockingIOError:
            pass
            
        time.sleep(0.1)

    proc.wait()

    st.session_state.running   = False
    st.session_state.done      = True
    st.session_state.exit_code = proc.returncode

    jp = os.path.join(abs_work, "cut_points.json")
    if os.path.exists(jp):
        with open(jp) as f:
            st.session_state.cuts = json.load(f)

    st.rerun()

# ── tabs ──────────────────────────────────────────────────────────────────────
tab_pipe, tab_cuts, tab_clips, tab_json = st.tabs(
    ["📋 Pipeline", "✂️ Cut Points", "🎞️ Clips", "📄 JSON / CSV"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Pipeline
# ─────────────────────────────────────────────────────────────────────────────
with tab_pipe:
    logs     = st.session_state.logs
    done_n   = _completed_phases(logs)
    active_i = _active_phase(logs)

    st.subheader("Phase tracker")
    cols = st.columns(len(PHASES))
    for i, (col, (label, _)) in enumerate(zip(cols, PHASES)):
        if i < done_n:
            col.success(f"✅ {label}")
        elif i == active_i and st.session_state.running:
            col.warning(f"⏳ {label}")
        else:
            col.info(f"⬜ {label}")

    if logs:
        total_done = sum(
            float(ln.split("done in")[1].split("s")[0].strip())
            for ln in logs
            if "done in" in ln
        )
        st.caption(f"Total elapsed so far: {total_done:.1f}s")

    st.markdown("---")
    st.subheader("Full log")
    if logs:
        # highlight phase headers
        colored = "\n".join(
            f">>> {ln}" if ln.startswith("[") else ln
            for ln in logs
        )
        st.code(colored, language=None)
        st.download_button(
            "⬇ Download log",
            "\n".join(logs),
            file_name="pipeline.log",
            mime="text/plain",
            key="dl_log"
        )
    else:
        st.info("Run the pipeline to see live logs here.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Cut Points
# ─────────────────────────────────────────────────────────────────────────────
with tab_cuts:
    cuts = st.session_state.cuts
    vp   = st.session_state.video_path
    wd   = st.session_state.work_dir

    if not cuts:
        st.info("No results yet. Run the pipeline or load existing results.")
        st.stop()

    # ── input video info ──────────────────────────────────────────────────────
    with st.expander("📁 Input video", expanded=True):
        if vp and os.path.exists(vp):
            dur      = _probe_duration(vp)
            size_mb  = os.path.getsize(vp) / 1e6
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Duration", _fmt(dur))
            col2.metric("File size", f"{size_mb:.0f} MB")
            col3.metric("Cut points", len(cuts))
            col4.metric("Clips", len(cuts) + 1)
            st.code(vp, language=None)
        else:
            st.warning("Video file not found at the stored path.")

    # ── cut-point table ───────────────────────────────────────────────────────
    st.subheader(f"{len(cuts)} cut point(s)")
    rows = []
    for c in cuts:
        sig = c.get("signals", {})
        rows.append({
            "Rank":         c["rank"],
            "Timecode":     c["timecode"],
            "Timestamp s":  c["timestamp"],
            "Score":        c["score"],
            "Silence s":    sig.get("silence", ""),
            "Motion":       sig.get("motion", ""),
            "Music RMS":    sig.get("music_rms", ""),
            "Beat":         sig.get("beat_strength", ""),
            "CLIP neutral": sig.get("clip_neutrality", ""),
            "Bndry dist s": sig.get("boundary_dist", ""),
            "Near bndry":   sig.get("near_boundary", ""),
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # score bar chart
    st.bar_chart(df.set_index("Timecode")["Score"])

    # CSV download
    cp = _csv_path()
    if os.path.exists(cp):
        with open(cp, "rb") as f:
            st.download_button("⬇ Download CSV", f, file_name="cut_points.csv", mime="text/csv", key="dl_csv_tab2")

    # ── frame previews ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Frame at each cut point")

    if not (vp and os.path.exists(vp)):
        st.warning("Source video not accessible — thumbnails unavailable.")
    else:
        thumb_dir = os.path.join(wd, "thumbnails")
        os.makedirs(thumb_dir, exist_ok=True)

        COLS = 5
        for row_start in range(0, len(cuts), COLS):
            chunk = cuts[row_start: row_start + COLS]
            cols  = st.columns(len(chunk))
            for col, cut in zip(cols, chunk):
                sig   = cut.get("signals", {})
                thumb = os.path.join(thumb_dir, f"cut_{cut['rank']:03d}.jpg")
                if not os.path.exists(thumb):
                    _thumbnail(vp, cut["timestamp"], thumb)
                with col:
                    if os.path.exists(thumb):
                        st.image(thumb, use_container_width=True)
                    st.caption(
                        f"**#{cut['rank']}** {cut['timecode']}\n"
                        f"score {cut['score']}  silence {sig.get('silence', '?')}s\n"
                        f"motion {sig.get('motion', '?')}  "
                        f"{'📍near boundary' if sig.get('near_boundary') else ''}"
                    )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Clips
# ─────────────────────────────────────────────────────────────────────────────
with tab_clips:
    wd        = st.session_state.work_dir
    clips_dir = os.path.join(wd, "clips") if wd else ""
    clips     = _load_clips(clips_dir) if clips_dir else []

    if not clips:
        st.info("No clips found. Run with 'Split into clips' enabled.")
        st.stop()

    st.markdown(f"### {len(clips)} clip(s) in `{clips_dir}`")

    # ── clip player ───────────────────────────────────────────────────────────
    names = [os.path.basename(c) for c in clips]
    
    # Initialize or validate selection
    if st.session_state.selected_clip not in names:
        st.session_state.selected_clip = names[0]
        
    current_idx = names.index(st.session_state.selected_clip)
    
    selected = st.selectbox(
        "Select clip to play", 
        names, 
        index=current_idx,
        key="clip_selector_dropdown"
    )
    
    # If dropdown changes, update state
    if selected != st.session_state.selected_clip:
        st.session_state.selected_clip = selected
        st.rerun()

    if st.session_state.selected_clip:
        clip_path = os.path.join(clips_dir, st.session_state.selected_clip)
        size_mb   = os.path.getsize(clip_path) / 1e6
        dur       = _probe_duration(clip_path)

        c1, c2, c3 = st.columns(3)
        c1.metric("Duration", _fmt(dur))
        c2.metric("File size", f"{size_mb:.1f} MB")
        c3.metric("Clip #", names.index(st.session_state.selected_clip) + 1)

        data = _read_video(clip_path)
        if data:
            st.video(data)
        else:
            st.warning(f"Clip is {size_mb:.0f} MB — too large to embed. Open directly:\n`{clip_path}`")

    # ── thumbnail gallery ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("All clips (Click 'Play' to watch)")

    thumb_dir = os.path.join(wd, "thumbnails")
    os.makedirs(thumb_dir, exist_ok=True)

    COLS = 4
    for row_start in range(0, len(clips), COLS):
        chunk = clips[row_start: row_start + COLS]
        cols  = st.columns(len(chunk))
        for col, clip_path in zip(cols, chunk):
            name    = os.path.basename(clip_path)
            thumb   = os.path.join(thumb_dir, name.replace(".mp4", "_thumb.jpg"))
            size_mb = os.path.getsize(clip_path) / 1e6
            dur     = _probe_duration(clip_path)
            if not os.path.exists(thumb):
                _thumbnail(clip_path, min(0.5, dur / 2), thumb)
            with col:
                if os.path.exists(thumb):
                    st.image(thumb, use_container_width=True)
                
                parts = name.replace(".mp4", "").split("_")
                tc    = parts[-1].replace("-", ":") if len(parts) >= 3 else "?"
                label = f"**{parts[1] if len(parts) >= 2 else name}**  {tc}"
                st.caption(f"{label}\n{_fmt(dur)}  ·  {size_mb:.1f} MB")
                
                # Highlight if currently selected
                is_selected = (name == st.session_state.selected_clip)
                btn_type = "primary" if is_selected else "secondary"
                if st.button(f"▶ Play {parts[1] if len(parts) >= 2 else ''}", key=f"btn_{name}", type=btn_type, use_container_width=True):
                    st.session_state.selected_clip = name
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — JSON / CSV
# ─────────────────────────────────────────────────────────────────────────────
with tab_json:
    cuts = st.session_state.cuts
    wd   = st.session_state.work_dir

    if not cuts:
        st.info("No results yet.")
        st.stop()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("cut_points.json")
        st.json(cuts)
        jp = _json_path()
        if os.path.exists(jp):
            with open(jp, "rb") as f:
                st.download_button(
                    "⬇ Download JSON", f,
                    file_name="cut_points.json",
                    mime="application/json",
                    key="dl_json"
                )

    with col_right:
        st.subheader("Per-cut signal breakdown")
        selected_rank = st.selectbox(
            "Inspect cut #",
            [c["rank"] for c in cuts],
            format_func=lambda r: f"#{r}  {next(c['timecode'] for c in cuts if c['rank'] == r)}",
        )
        cut = next(c for c in cuts if c["rank"] == selected_rank)
        sig = cut.get("signals", {})

        st.metric("Score",           cut["score"])
        st.metric("Timecode",        cut["timecode"])
        st.metric("Frame",           cut["frame"])
        st.metric("Silence window",  f"{sig.get('silence', '?')} s")
        st.metric("Motion (flow)",   sig.get("motion", "?"))
        st.metric("Music RMS",       sig.get("music_rms", "?"))
        st.metric("Peak RMS",        sig.get("peak_rms", "?"))
        st.metric("Beat strength",   sig.get("beat_strength", "?"))
        st.metric("CLIP neutrality", sig.get("clip_neutrality", "?"))
        st.metric("Boundary dist",   f"{sig.get('boundary_dist', '?')} s")
        st.metric("Near boundary",   "Yes" if sig.get("near_boundary") else "No")

        # score breakdown bar chart
        score_parts = {
            "Silence (+)":    min(sig.get("silence", 0), 5.0) * 6,
            "Center (+)":     20.0,          # simplified; actual depends on position
            "Motion (-)":     -min(sig.get("motion", 0), 4.375) * 8,
            "Music RMS (-)":  -(sig.get("peak_rms", 0) * 25 if sig.get("peak_rms", 0) > 0.09 else 0),
            "CLIP (+)":       sig.get("clip_neutrality", 0) * 15,
            "Boundary (+)":   10.0 if sig.get("near_boundary") else 0,
        }
        st.markdown("**Score breakdown (approx)**")
        st.bar_chart(pd.DataFrame.from_dict(score_parts, orient="index", columns=["pts"]))

        cp = _csv_path()
        if os.path.exists(cp):
            st.markdown("---")
            with open(cp, "rb") as f:
                st.download_button(
                    "⬇ Download CSV", f,
                    file_name="cut_points.csv",
                    mime="text/csv",
                    key="dl_csv_tab4"
                )
