import hashlib


def video_fingerprint(video_path: str, head_bytes: int = 4 * 1024 * 1024) -> str:
    """SHA-256 of the first `head_bytes` of a video file — fast, stable identity check."""
    h = hashlib.sha256()
    with open(video_path, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()
