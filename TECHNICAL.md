# Scene Cutter — Technical Documentation

**Version:** 1.1 (Phase 1 Upgraded)
**Purpose:** Confluence / internal technical reference
**Audience:** Engineering team evaluating this POC for production

---

## 1. Problem Statement

The original requirement was to use **PySceneDetect alone** to identify cut points in long-form drama content. After evaluation, PySceneDetect in isolation proved insufficient for the following reasons:

| Limitation | Impact |
|---|---|
| Detects visual scene *changes*, not safe *gaps* | A scene boundary mid-dialogue is a terrible cut point |
| No awareness of speech | Cuts land on active dialogue |
| No awareness of music intensity | Cuts land on dramatic music stings |
| No awareness of motion | Cuts land mid-action |
| Threshold tuning is global | Content-adaptive decisions are not possible |

PySceneDetect's `ContentDetector` computes HSV histogram differences between consecutive frames. It answers *"did the shot change here?"*, not *"is this a safe moment to cut?"*. These are fundamentally different questions.

**The solution:** PySceneDetect is integrated as both a **Smart Sampling** source and a scoring signal. The pipeline builds a multi-signal scoring model that combines audio, motion, and visual semantics.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: MP4 video                                               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼─────────────┐
              │  PHASE 1 — Audio Prep    │  ffmpeg + demucs
              │  Extract & separate stems│
              └────────────┬─────────────┘
                           │
         ┌─────────────────┼─────────────────────┐
         ▼                                        ▼
┌────────────────┐                    ┌────────────────────┐
│ PHASE 2        │                    │ PHASE 2 (parallel) │
│ SileroVAD      │                    │ no_vocals.wav      │
│ Speech segments│                    │ (music analysis)   │
└───────┬────────┘                    └────────┬───────────┘
        │                                      │
        ▼                                      │
┌────────────────┐                             │
│ PHASE 3        │                             │
│ Safe Windows   │                             │
│ (silence gaps) │                             │
└───────┬────────┘                             │
        │                                      │
        ▼                                      ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 4 — Multi-signal Scoring (parallelised, 10+ threads)  │
│                                                              │
│  Per candidate timestamp (Smart + Exploratory):              │
│  ├─ PySceneDetect   → exact boundary alignment bonus          │
│  ├─ OpenCV          → peak local optical flow penalty         │
│  ├─ librosa         → HPSS percussive + spectral flux penalty │
│  └─ OpenCLIP        → batched temporal visual neutrality      │
└───────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  PHASE 5 — Output          │
              │  JSON + CSV + Snippets      │
              └────────────────────────────┘
```

---

## 3. Phase-by-Phase Breakdown

### Phase 1 — Audio Extraction & Stem Separation

**File:** `pipeline/audio_prep.py`

#### Step 1a — Audio Extraction
**Tool:** `ffmpeg` (system binary, called via `subprocess`)

```
ffmpeg -y -i input.mp4 -ac 1 -ar 16000 -vn raw_audio.wav
```

**Output:** `cut_analysis/raw_audio.wav`

#### Step 1b — Stem Separation
**Tool:** [demucs](https://github.com/facebookresearch/demucs) `htdemucs` model (Meta AI)

**Outputs:**
- `vocals.wav` — isolated voice track → fed to speech detection
- `no_vocals.wav` — isolated music/ambience track → fed to music scoring

---

### Phase 2 — Speech Detection

**File:** `pipeline/speech.py`
**Tool:** [SileroVAD](https://github.com/snakers4/silero-models) v6.2 (Language-agnostic)

---

### Phase 3 — Safe Window Construction

**File:** `pipeline/windows.py`

Gaps between speech segments are buffered by `SPEECH_BUFFER` (default 1.2s) to create safe windows.

---

### Phase 4 — High-Precision Candidate Sampling

**File:** `pipeline/scorer.py`

To ensure frame-perfect cuts, the pipeline now uses a dual-sampling strategy:

1.  **Exploratory Samples:** `WINDOW_SAMPLES` (default 6) evenly spaced points per window.
2.  **Smart Samples:** Every **PySceneDetect** boundary that falls within a safe window is added as an exact candidate. This ensures the tool evaluates the "perfect" visual cut point.

---

### Phase 5 — Multi-Signal Scoring

#### 5a — Motion Score (Peak Detection)
**Tool:** OpenCV `calcOpticalFlowFarneback`

**Optimization:** Frames are downsampled to 320px for 4x speed.
**Logic:** Instead of mean motion, it calculates the **95th percentile (peak)** motion. This catches local movement (e.g., a hand wave) that global averages miss.

#### 5b — Music Score (Percussive Awareness)
**Tool:** [librosa]

**Logic:** Uses **HPSS (Harmonic-Percussive Source Separation)** to isolate dramatic "stings" (drums, sharp accents) from steady background music. Also monitors **Spectral Flux** for sudden mood shifts.

#### 5c — Visual Semantics (Temporal CLIP)
**Tool:** [OpenCLIP] `ViT-L-14` trained on **DataComp-1B**

**Upgrades:**
1.  **Temporal Window:** Checks 5 frames (±0.5s) per candidate.
2.  **Batch Inference:** Processes all 5 frames in a single GPU/CPU pass for speed.
3.  **Expert Prompts:** Detects mouths open, emotional reaction shots, and on-screen text.
4.  **Pessimistic Scoring:** Returns the *minimum* neutrality found in the window (if one frame is active, the whole window is risky).

---

## 4. Output Formats

### JSON / CSV
Full signal breakdown for every cut.

### Video Snippets (Review Mode)
Generated via `--snippets`. 5-second context clips (4s before, 1s after) with a **red border highlight** at the exact cut frame.

---

## 5. Model Inventory (Upgraded)

| Model | Version | Size | Source | Purpose |
|---|---|---|---|---|
| demucs `htdemucs` | 4.0.1 | ~320 MB | Meta AI | Vocal/music separation |
| SileroVAD | 6.2.1 | ~5 MB | Silero | Speech detection |
| **OpenCLIP ViT-L-14** | **DataComp-1B** | **~1.5 GB** | **LAION** | **High-precision visual scoring** |
| PySceneDetect | 0.6.7 | — | PyPI | Scene boundary source |

---

## 6. Performance Benchmarks (Phase 1)

Tested on: MacBook Pro M-series (CPU-only)

| Phase | Time (40m Video) |
|---|---|
| Audio/Demucs | ~13 min (cached: 2s) |
| CLIP Scoring (Batched) | ~4 min |
| Motion (Downsampled) | ~2 min |
| **Total Pipeline** | **~20 min** |

---

## 7. Comparison Automation

**File:** `compare_cuts.py`
Automates the validation of results against Adobe Premiere Pro's XML/JSON exports. Groups matching frames and isolates "missed" scene changes for engineering analysis.
