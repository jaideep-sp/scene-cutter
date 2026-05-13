# Scene Cutter — Intelligent Video Cut-Point Detection

Fully local, open-source CLI that finds the best moments to cut a video — no speech, no significant motion, no dramatic music. Works with any language. No API keys required.

---

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Run on a video (returns all valid cut points)
python main.py input.mp4

# 3. Run and also split the video at each cut point
python main.py input.mp4 --split

# 4. Limit to top 10 cuts, use drama profile
python main.py input.mp4 --n-cuts 10 --profile drama --split
```

---

## Visual Dashboard (Streamlit)

Scene Cutter includes a web-based dashboard for real-time monitoring, cut-point inspection, and playing generated clips.

```bash
# Start the dashboard
streamlit run dashboard.py
```

**Features:**
- **Live Monitor:** Watch the pipeline phases and CPU usage in real-time.
- **Inspector:** Browse every cut point with auto-generated frame thumbnails.
- **Player:** Watch the split video segments directly in your browser.
- **Exporter:** Download results as JSON or CSV.

---

## Output Files

| File | Description |
|---|---|
| `cut_analysis/cut_points.json` | All cut points with full scores and signals |
| `cut_analysis/cut_points.csv` | Same data as CSV (Excel / Sheets compatible) |
| `cut_analysis/clips/` | Video segments split at each cut point (requires `--split`) |

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--out DIR` | `./cut_analysis` | Work directory for intermediates and output |
| `--n-cuts INT` | `0` (all valid) | Cap number of cut points returned |
| `--profile` | none | Preset: `drama`, `action`, `documentary`, `animation` |
| `--split` | off | Split video at cut points using ffmpeg stream-copy |

---

## Configuration (`config.py`)

| Parameter | Default | Effect |
|---|---|---|
| `SPEECH_BUFFER` | `2.0s` | Padding added around each detected speech segment |
| `MIN_SILENCE_GAP` | `1.5s` | Minimum silence window to consider as cut candidate |
| `MIN_CUT_SPACING` | `10.0s` | Minimum distance between two selected cut points |
| `MOTION_THRESHOLD` | `2.5` | Optical flow magnitude above which a frame is penalised |
| `MUSIC_RMS_THRESHOLD` | `0.09` | Peak RMS above which music intensity is penalised |
| `BEAT_STRENGTH_CAP` | `1.5` | Onset strength above which a musical accent is avoided |
| `WINDOW_SAMPLES` | `3` | Candidate points sampled per silence window |
| `N_CUTS` | `0` | 0 = return all valid; positive int caps the result |

### Content Profiles

```bash
python main.py video.mp4 --profile drama       # wider speech buffer, stricter music gate
python main.py video.mp4 --profile action      # tolerant of motion, relaxed music gate
python main.py video.mp4 --profile documentary # higher whisper confidence, tighter silence
python main.py video.mp4 --profile animation   # relaxed music threshold for score-heavy content
```

---

## Pipeline Overview

```
MP4
 └─ ffmpeg ──────────────────► raw_audio.wav (16 kHz mono)
      └─ demucs ─────────────► vocals.wav  +  no_vocals.wav
           ├─ SileroVAD ─────► speech segments  (language-agnostic)
           ├─ librosa ───────► RMS + beat strength on no_vocals stem
           ├─ PySceneDetect ─► scene boundary timestamps
           ├─ OpenCV ────────► optical flow motion score per frame
           └─ OpenCLIP ──────► visual neutrality score per frame
                └─ scorer ───► ranked, spaced cut candidates
```

---

## System Requirements

- Python 3.11+
- ffmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)
- ~4 GB RAM minimum; 8 GB+ recommended for ViT-H-14 CLIP model
- GPU optional but significantly faster (see performance table in TECHNICAL.md)

---

## Troubleshooting

**"No safe windows found"** — Reduce `SPEECH_BUFFER` to `1.0` in `config.py`.

**Cuts land mid-dialogue** — Increase `SPEECH_BUFFER` to `3.0` and `MIN_SILENCE_GAP` to `2.0`.

**Too slow on CPU** — Reduce `WINDOW_SAMPLES` to `1` or use a GPU. See `TECHNICAL.md` for benchmarks.

**QuickTime can't play clips** — Source was AV1-encoded (YouTube). Use VLC or IINA to play, or re-encode: `ffmpeg -i input.mp4 -c:v libx264 -crf 18 output.mp4`.
