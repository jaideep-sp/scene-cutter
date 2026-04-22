import torch
import config


def get_speech_segments(vocals_path: str) -> list[dict]:
    """Pass 1: faster-whisper word-level transcription on vocals stem."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if config.DEVICE == "cuda" else "int8"
    model = WhisperModel("medium", device=config.DEVICE, compute_type=compute_type)

    segments, _ = model.transcribe(
        vocals_path,
        word_timestamps=True,
        vad_filter=True,
    )

    results = []
    for segment in segments:
        if segment.words is None:
            continue
        for word in segment.words:
            if word.probability >= config.WHISPER_CONFIDENCE:
                results.append({
                    "start": word.start,
                    "end": word.end,
                    "text": word.word,
                    "prob": word.probability,
                })

    return results


def get_vad_segments(vocals_path: str) -> list[dict]:
    """Pass 2: SileroVAD speech activity detection on vocals stem."""
    import soundfile as sf
    from silero_vad import load_silero_vad, get_speech_timestamps

    model = load_silero_vad()

    audio, sr = sf.read(vocals_path, dtype="float32")
    # SileroVAD expects a 1-D torch tensor at 16 kHz or 8 kHz.
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_tensor = torch.from_numpy(audio)

    # Resample to 16 kHz if necessary.
    if sr != 16000:
        import torchaudio
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_tensor = resampler(audio_tensor.unsqueeze(0)).squeeze(0)
        sr = 16000

    timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=0.45,
        min_speech_duration_ms=200,
        min_silence_duration_ms=800,
        return_seconds=False,
    )

    return [
        {"start": t["start"] / sr, "end": t["end"] / sr}
        for t in timestamps
    ]
