import config


def _merge_intervals(intervals: list[tuple]) -> list[tuple]:
    """Merge a list of (start, end) intervals, returning a sorted, non-overlapping list."""
    if not intervals:
        return []
    sorted_ivs = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_ivs[0]]
    for start, end in sorted_ivs[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def build_unsafe_zones(
    # whisper_segs: list[dict],
    vad_segs: list[dict],
    duration: float,
) -> list[tuple]:
    """Apply SPEECH_BUFFER to all speech segments and merge into unsafe zones."""
    buf = config.SPEECH_BUFFER
    raw = []

    # for seg in whisper_segs:
    #     raw.append((
    #         max(0.0, seg["start"] - buf),
    #         min(duration, seg["end"]   + buf),
    #     ))

    for seg in vad_segs:
        raw.append((
            max(0.0, seg["start"] - buf),
            min(duration, seg["end"]   + buf),
        ))

    return _merge_intervals(raw)


def build_safe_windows(unsafe_zones: list[tuple], duration: float) -> list[dict]:
    """Return gaps between unsafe zones that are at least MIN_SILENCE_GAP long."""
    min_gap = config.MIN_SILENCE_GAP
    windows = []

    # Gap before the first unsafe zone.
    prev_end = 0.0
    for start, end in unsafe_zones:
        gap = start - prev_end
        if gap >= min_gap:
            windows.append(_make_window(prev_end, start))
        prev_end = end

    # Gap after the last unsafe zone.
    gap = duration - prev_end
    if gap >= min_gap:
        windows.append(_make_window(prev_end, duration))

    return windows


def _make_window(start: float, end: float) -> dict:
    return {
        "start":    start,
        "end":      end,
        "duration": end - start,
        "center":   (start + end) / 2.0,
    }
