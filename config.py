import torch

SPEECH_BUFFER       = 1.2   # reduced from 2.0 to find more gaps
MIN_SILENCE_GAP     = 1.0   # reduced from 1.5
MIN_CUT_SPACING     = 3.0   # allow more frequent cuts (was 5.0)
WHISPER_CONFIDENCE  = 0.55
MOTION_THRESHOLD    = 2.5
MUSIC_RMS_THRESHOLD = 0.09
BEAT_STRENGTH_CAP   = 1.5
N_CUTS              = 0
WINDOW_SAMPLES      = 6     # increased from 3 for better resolution
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

PROFILES = {
    "drama": {
        "SPEECH_BUFFER": 3.0,
        "MIN_SILENCE_GAP": 2.0,
        "MUSIC_RMS_THRESHOLD": 0.06,
    },
    "action": {
        "MOTION_THRESHOLD": 4.0,
        "MUSIC_RMS_THRESHOLD": 0.15,
    },
    "documentary": {
        "WHISPER_CONFIDENCE": 0.65,
        "MIN_SILENCE_GAP": 1.0,
    },
    "animation": {
        "MUSIC_RMS_THRESHOLD": 0.12,
    },
}


def apply_profile(profile_name):
    import sys
    import config as cfg
    if profile_name not in PROFILES:
        print(f"Unknown profile '{profile_name}'. Available: {', '.join(PROFILES)}", file=sys.stderr)
        return
    overrides = PROFILES[profile_name]
    for key, val in overrides.items():
        setattr(cfg, key, val)
