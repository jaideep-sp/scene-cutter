import numpy as np
import librosa


def score_music_at(no_vocals_path: str, timestamp: float, window: float = 1.0) -> dict:
    """Compute RMS and beat onset strength on no_vocals stem around a timestamp."""
    # Load at native sample rate — we don't need a fixed rate for music analysis.
    y, sr = librosa.load(no_vocals_path, sr=None, mono=True)

    duration = len(y) / sr
    t_start = max(0.0, timestamp - window)
    t_end   = min(duration, timestamp + window)

    s_start = int(t_start * sr)
    s_end   = int(t_end   * sr)
    segment = y[s_start:s_end]

    if len(segment) == 0:
        return {"rms": 0.0, "peak_rms": 0.0, "beat_strength": 0.0}

    rms    = librosa.feature.rms(y=segment)[0]
    onset  = librosa.onset.onset_strength(y=segment, sr=sr)

    return {
        "rms":          float(np.mean(rms)),
        "peak_rms":     float(np.max(rms)),
        "beat_strength": float(np.mean(onset)),
    }


def preload_audio(no_vocals_path: str):
    """Load the full no_vocals stem once and return (y, sr) for repeated queries."""
    y, sr = librosa.load(no_vocals_path, sr=None, mono=True)
    return y, sr


def score_music_at_preloaded(y: np.ndarray, sr: int, timestamp: float, window: float = 1.0) -> dict:
    """
    Analyzes music for dramatic cues using HPSS and Spectral Flux.
    - HPSS helps us ignore steady drones and focus on 'hits' (percussive).
    - Spectral Flux detects sudden changes in music texture/mood.
    - Uses Max values (pessimistic) to ensure no cut lands on a drum hit.
    """
    duration = len(y) / sr
    t_start  = max(0.0, timestamp - window)
    t_end    = min(duration, timestamp + window)

    s_start  = int(t_start * sr)
    s_end    = int(t_end   * sr)
    segment  = y[s_start:s_end]

    if len(segment) < 512: # Minimal length for FFT
        return {"rms": 0.0, "peak_rms": 0.0, "beat_strength": 0.0, "spectral_flux": 0.0}

    # 1. Harmonic-Percussive Source Separation (HPSS)
    # We focus on the 'percussive' part (the hits/stings)
    D = librosa.stft(segment)
    D_harmonic, D_percussive = librosa.decompose.hpss(D)
    
    # Calculate RMS of the percussive part
    y_percussive = librosa.istft(D_percussive)
    rms_percussive = librosa.feature.rms(y=y_percussive)[0]
    
    # 2. Spectral Flux (Onset Strength)
    # Measures sudden changes in the frequency spectrum
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    
    # 3. Overall RMS (Loudness)
    rms_full = librosa.feature.rms(y=segment)[0]

    return {
        "rms":           float(np.mean(rms_full)),
        "peak_rms":      float(np.max(rms_full)),
        "beat_strength": float(np.max(onset_env)), # Use MAX to catch the sting
        "percussive_rms": float(np.max(rms_percussive)),
        "spectral_flux": float(np.mean(onset_env))
    }
