# Scene Cutter — Technical Documentation

**Version:** 1.0 (POC)
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

**The solution:** PySceneDetect is retained as one signal among several. A cut point near a scene boundary gets a scoring bonus (+10), but the primary gate is silence detection. The pipeline builds a multi-signal scoring model that combines audio, motion, and visual semantics.

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
│  PHASE 4 — Multi-signal Scoring (parallelised, 10 threads)   │
│                                                              │
│  Per candidate timestamp:                                    │
│  ├─ PySceneDetect   → scene boundary proximity bonus         │
│  ├─ OpenCV          → optical flow motion penalty            │
│  ├─ librosa         → music RMS + beat strength penalty      │
│  └─ OpenCLIP        → visual neutrality score                │
└───────────────────────────┬──────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │  PHASE 5 — Output          │
              │  JSON + CSV + video splits  │
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

| Parameter | Value | Reason |
|---|---|---|
| `-ac 1` | mono | SileroVAD and Whisper require mono |
| `-ar 16000` | 16 kHz | Standard for speech models |
| `-vn` | no video | Audio-only extraction |

**Output:** `cut_analysis/raw_audio.wav`

#### Step 1b — Stem Separation
**Tool:** [demucs](https://github.com/facebookresearch/demucs) `htdemucs` model (Meta AI)

```
python -m demucs --two-stems=vocals --out <work_dir> raw_audio.wav
```

| Parameter | Value | Reason |
|---|---|---|
| `--two-stems=vocals` | vocals / no_vocals | Only need 2 stems, faster than 4-stem |
| Model | `htdemucs` | Default; Hybrid Transformer, best quality |

**Outputs:**
- `vocals.wav` — isolated voice track → fed to speech detection
- `no_vocals.wav` — isolated music/ambience track → fed to music scoring

**Why demucs?** Without stem separation, speech detection runs on mixed audio (voice + music), causing false positives where music is mistaken for speech. Separating first gives clean inputs to downstream models.

**Caching:** If stems already exist in the work directory, demucs is skipped entirely on re-runs.

---

### Phase 2 — Speech Detection

**File:** `pipeline/speech.py`

**Tool:** [SileroVAD](https://github.com/snakers4/silero-models) v6.2

```python
model = load_silero_vad()
timestamps = get_speech_timestamps(
    audio_tensor,
    model,
    threshold=0.45,
    min_speech_duration_ms=200,
    min_silence_duration_ms=800,
    return_seconds=False,
)
```

| Parameter | Value | Reason |
|---|---|---|
| `threshold` | `0.45` | Speech confidence gate; lower = more sensitive |
| `min_speech_duration_ms` | `200ms` | Ignore sub-200ms voice bursts (breath, clicks) |
| `min_silence_duration_ms` | `800ms` | Merge speech segments separated by <800ms gaps |

**Why SileroVAD instead of Whisper?**

Originally the pipeline used Whisper (`faster-whisper medium`) for word-level timestamps. It was removed because:

1. Whisper took **~40 minutes on CPU** for a 40-min video
2. Word-level precision is washed out by the `SPEECH_BUFFER = 2.0s` padding anyway
3. Whisper requires language identification; SileroVAD is **language-agnostic** — works on Urdu, Hindi, Punjabi, Sindhi, Pashto, or any regional language without configuration

SileroVAD runs in **~11 seconds** on the same content with equivalent cut quality.

**Language support:** SileroVAD detects voice as an acoustic pattern, independent of language. It will correctly detect speech in any language including unsupported regional languages where Whisper would fail.

**Output:** List of `{start, end}` dicts in seconds.

---

### Phase 3 — Safe Window Construction

**File:** `pipeline/windows.py`

Two-step process:

#### Step 3a — Build Unsafe Zones
Each speech segment gets padded by `SPEECH_BUFFER` (default 2.0s) on both sides and merged into non-overlapping unsafe zones.

```
speech:    [===]       [=====]      [==]
buffered:  [=======]   [=========]  [======]
merged:    [=======]   [=========]  [======]
safe:            [====]         [==]
```

#### Step 3b — Filter Safe Windows
Gaps between unsafe zones shorter than `MIN_SILENCE_GAP` (1.5s) are discarded. Remaining gaps become safe windows with a computed center point.

**For a 40-min drama (sample result):**
- 399 speech segments detected
- 65 unsafe zones after merging
- 43 safe windows covering 420s (17.6% of video)

---

### Phase 4 — Scene Boundary Detection

**File:** `pipeline/video.py`

**Tool:** [PySceneDetect](https://www.scenedetect.com/) `ContentDetector`

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

scenes.add_detector(ContentDetector(threshold=27.0))
scenes.detect_scenes(video)
```

| Parameter | Value | Reason |
|---|---|---|
| `threshold` | `27.0` | HSV histogram delta above which a scene change is declared |

**How it works:** For each consecutive frame pair, computes the mean absolute difference of HSV histograms. If the delta exceeds the threshold, a scene boundary is recorded.

**Why it's retained as a bonus signal only:** For a 40-min drama, 320 boundaries were detected (~1 every 7 seconds). This is noisy. Rather than gating on it, cuts near a boundary receive a `+10` score bonus. The signal rewards cuts that coincide with natural edit points without penalising others.

**Time cost:** ~4-5 minutes (reads every frame of the video).

---

### Phase 5 — Multi-Signal Scoring

**File:** `pipeline/scorer.py`, `pipeline/video.py`, `pipeline/music.py`, `pipeline/semantics.py`

#### Candidate Sampling
Each safe window is sampled at `WINDOW_SAMPLES` (default 3) evenly spaced points. For 43 windows × 3 samples = **129 candidates**.

#### Parallelisation
Candidates are scored in parallel using `ThreadPoolExecutor` with `os.cpu_count()` threads. PyTorch inference and OpenCV operations release the GIL, making threading effective here.

#### Per-Candidate Scoring

**5a — Motion Score**
**Tool:** OpenCV `calcOpticalFlowFarneback`

Reads ±0.5s of frames around the timestamp, computes dense optical flow between consecutive frames, returns mean flow magnitude.

```python
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, gray, None,
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
```

| Parameter | Value | Reason |
|---|---|---|
| `pyr_scale` | `0.5` | Image pyramid scale; 0.5 = half size per level |
| `levels` | `3` | Pyramid levels; more = captures larger motions |
| `winsize` | `15` | Averaging window; larger = smoother but less detail |
| `iterations` | `3` | Per-level iterations |

**Score contribution:** `-min(motion, 4.375) × 8` (max penalty: -35)

**5b — Music Score**
**Tool:** [librosa](https://librosa.org/)

Operates on the pre-loaded `no_vocals.wav` array (loaded once, reused across all candidates).

```python
rms   = librosa.feature.rms(y=segment)
onset = librosa.onset.onset_strength(y=segment, sr=sr)
```

**Score contribution:**
- If `peak_rms > MUSIC_RMS_THRESHOLD`: `-peak_rms × 25`
- If `beat_strength > BEAT_STRENGTH_CAP`: `-beat_strength × 3`

**5c — Visual Neutrality (CLIP)**
**Tool:** [OpenCLIP](https://github.com/mlfoundations/open_clip) `ViT-H-14` pretrained on LAION-2B

Text prompts are **pre-encoded once** before the scoring loop and reused as tensors:

```python
NEUTRAL_PROMPTS = [
    "a quiet empty room",
    "a calm outdoor scene with no people",
    "a neutral establishing shot",
    "black screen transition",
]
ACTIVE_PROMPTS = [
    "person talking intensely",
    "action scene with fast movement",
    "emotional dialogue between characters",
    "person crying or shouting",
]
```

Score = `mean(cosine_sim(frame, neutral_prompts)) - mean(cosine_sim(frame, active_prompts))`

Positive = visually neutral/safe. Negative = visually active/dramatic.

**Score contribution:** `clip_neutrality × 15`

**5d — Silence & Centering**
- `+min(silence_duration, 5.0) × 6` — longer silence = better (capped at +30)
- `+(1 - center_distance) × 20` — prefer center of silence window

**5e — Scene Boundary Bonus**
- `+10` if nearest scene boundary is within 2.0s

#### Full Scoring Formula

```
score = silence_reward        (max +30)
      + centering_reward      (max +20)
      + clip_neutrality × 15  (typically ±10)
      + boundary_bonus        (+10 if near boundary)
      - motion_penalty        (max -35)
      - music_rms_penalty     (variable)
      - beat_strength_penalty (variable)
```

Theoretical max score ≈ 85. Practical range for good cuts: 40–65.

#### Final Selection
Candidates are sorted by score descending. A spacing filter enforces `MIN_CUT_SPACING = 10s` between selected cuts. If `N_CUTS = 0` (default), all passing candidates are returned.

---

## 4. Output Formats

### JSON (`cut_points.json`)
```json
{
  "rank": 1,
  "timestamp": 8.624,
  "timecode": "00:08.62",
  "frame": 215,
  "score": 55.68,
  "signals": {
    "silence": 4.48,
    "motion": 0.145,
    "music_rms": 0.001,
    "peak_rms": 0.012,
    "beat_strength": 0.895,
    "clip_neutrality": -0.004,
    "boundary_dist": 0.62,
    "near_boundary": true
  }
}
```

### CSV (`cut_points.csv`)
Columns: `rank, timecode, timestamp_s, frame, segment_start_s, segment_end_s, segment_duration_s, score, silence_s, motion, music_rms, peak_rms, beat_strength, clip_neutrality, boundary_dist_s, near_boundary, pipeline_time_s`

### Video Clips (`clips/segment_NNN_MM-SS.mp4`)
Segments split at each cut point using ffmpeg stream-copy (`-c copy`). No re-encoding — instantaneous. Note: if source is AV1-encoded (e.g. YouTube), clips require a player with AV1 support (VLC, IINA). Re-encode with `-c:v libx264 -crf 18` for universal compatibility.

---

## 5. Model Inventory

| Model | Version | Size | Source | Purpose |
|---|---|---|---|---|
| demucs `htdemucs` | 4.0.1 | ~320 MB | Meta AI / PyPI | Vocal/music stem separation |
| SileroVAD | 6.2.1 | ~5 MB | Silero / PyPI | Language-agnostic speech detection |
| OpenCLIP `ViT-H-14` | LAION-2B | ~3.9 GB | LAION / HuggingFace | Visual frame neutrality scoring |
| PySceneDetect | 0.6.7 | — | PyPI | Scene boundary detection |

All models run **fully locally**. No external API calls at inference time. HuggingFace Hub is contacted only on first run to download model weights; subsequent runs are fully offline.

---

## 6. Performance Benchmarks

Tested on: MacBook Pro M-series (Apple Silicon, CPU-only, no MPS)
Content: 39m 48s Pakistani drama (AV1, 870 MB, 25fps)

### With Optimisations (current)

| Phase | Time |
|---|---|
| Demucs stem separation | ~13 min (cached on re-run: 2s) |
| SileroVAD | ~11s |
| Safe window construction | <1s |
| PySceneDetect | ~4 min |
| OpenCLIP load + text pre-encode | ~1.5 min |
| Scoring 129 candidates (10 threads) | ~6.5 min |
| ffmpeg split (48 segments) | ~30s |
| **Total (first run)** | **~25 min** |
| **Total (re-run, stems cached)** | **~12 min** |

### Before Optimisations (original)

| Issue | Impact |
|---|---|
| Whisper transcription (medium, CPU) | +40 min |
| Text prompts re-encoded per candidate | +~5 min for 301 candidates |
| Sequential scoring (no parallelism) | ~90 min for 301 candidates |
| No demucs caching | +13 min on every re-run |
| **Total** | **~150 min** |

**Optimisation summary: ~150 min → ~25 min (6× faster)**

### GPU Projection (if CUDA available)

| Phase | CPU | GPU (estimate) |
|---|---|---|
| Demucs | 13 min | ~2 min |
| OpenCLIP scoring | ~6.5 min | ~1 min |
| **Total** | ~25 min | **~5–7 min** |

---

## 7. Cost Analysis

This pipeline is **fully local** — no cloud API costs.

### One-time costs (first run only)
| Item | Size | Source |
|---|---|---|
| demucs model download | ~320 MB | HuggingFace |
| OpenCLIP ViT-H-14 download | ~3.9 GB | HuggingFace |
| SileroVAD | ~5 MB | Bundled in package |

### Per-run costs
- **Compute:** CPU/GPU electricity only
- **API cost:** $0
- **Storage:** ~300 MB per processed video (stems + clips)

### If migrated to cloud (reference only)
| Option | Estimated cost per 40-min video |
|---|---|
| AWS p3.2xlarge (V100, on-demand) | ~$0.40 |
| AWS g4dn.xlarge (T4, on-demand) | ~$0.15 |
| Local GPU workstation (amortised) | ~$0.01 |

---

## 8. Limitations

| Limitation | Detail | Mitigation |
|---|---|---|
| CPU speed | Full pipeline ~25 min on CPU | Use GPU; CUDA auto-detected |
| AV1 video playback | Clips from AV1 source won't play in QuickTime | Use VLC/IINA or re-encode |
| CLIP model size | ViT-H-14 requires ~4 GB RAM | Swap to `ViT-B-32` for lower memory (~0.6 GB), lower accuracy |
| Safe window coverage | Only 17.6% of a drama is safe to cut | Reduce `SPEECH_BUFFER`; drama content is dialogue-heavy by nature |
| Scene detection noise | 320 boundaries in 40 min = ~1 per 7s | Raise `ContentDetector(threshold=40.0)` for fewer, cleaner boundaries |
| No subtitle/caption output | Pipeline identifies *when* to cut, not *what* is said | Add Whisper back as optional `--transcribe` flag if needed |
| Thread safety of OpenCV | VideoCapture opened per thread | Each thread opens its own instance; safe but adds file-open overhead |
| No GPU on Apple Silicon MPS | MPS (Metal) is not used | Can be added: `config.DEVICE = "mps"` if torch MPS is available |

---

## 9. File Structure

```
scene_cutter/
├── main.py                  # CLI entry point
├── config.py                # All tuneable parameters and content profiles
├── requirements.txt         # Python dependencies
├── README.md                # Quick-start guide
├── TECHNICAL.md             # This document
└── pipeline/
    ├── audio_prep.py        # ffmpeg extraction + demucs separation
    ├── speech.py            # SileroVAD speech detection
    ├── windows.py           # Unsafe zone + safe window construction
    ├── video.py             # PySceneDetect boundaries + optical flow
    ├── music.py             # librosa RMS + beat strength
    └── semantics.py         # OpenCLIP visual neutrality scoring
```

---

## 10. Recommended Next Steps for Production

1. **MPS/CUDA support** — Add `device = "mps"` detection for Apple Silicon GPU acceleration
2. **Batch processing** — Wrap `find_best_cuts` in a loop to process a folder of episodes
3. **Score calibration** — Collect editor feedback on cut quality and retune scoring weights
4. **Lighter CLIP model** — Evaluate `ViT-B-32` vs `ViT-H-14` quality/speed tradeoff for production
5. **REST API wrapper** — Expose pipeline as a FastAPI endpoint for integration with video editing tools
6. **Confidence threshold on VAD** — Expose SileroVAD `threshold` parameter in `config.py` for per-language tuning
