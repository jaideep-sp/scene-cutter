import cv2
import numpy as np


def get_scene_boundaries(video_path: str, threshold: float = 27.0) -> list[float]:
    """Return scene-change timestamps (seconds) using PySceneDetect ContentDetector."""
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video  = open_video(video_path)
    scenes = SceneManager()
    scenes.add_detector(ContentDetector(threshold=threshold))
    scenes.detect_scenes(video)

    scene_list = scenes.get_scene_list()
    # Each element is (start_timecode, end_timecode); return the start of each scene
    # after the very first (index 0 is the beginning of the file).
    boundaries = [scene[0].get_seconds() for scene in scene_list[1:]]
    return boundaries


def score_motion_at(video_path: str, timestamp: float, half_window: float = 0.5) -> float:
    """
    Calculates the 95th percentile of optical flow magnitude in a ±half_window window.
    Downsamples frames to 320px width for a 4x speed boost while maintaining accuracy.
    Uses 'peak' motion rather than 'mean' to catch movement in small areas of the frame.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    t_start = max(0.0, timestamp - half_window)
    t_end   = timestamp + half_window

    start_frame = int(t_start * fps)
    end_frame   = int(t_end   * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    peak_magnitudes = []
    prev_gray  = None

    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 1. Downsample for speed (320px width is plenty for motion detection)
        h, w = frame.shape[:2]
        new_w = 320
        new_h = int(h * (new_w / w))
        small = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is not None:
            # 2. Dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # 3. Use 95th percentile to capture 'peak' local motion 
            # (e.g. a character moving their arm even if the background is still)
            peak_magnitudes.append(float(np.percentile(mag, 95)))
            
        prev_gray = gray

    cap.release()
    
    if not peak_magnitudes:
        return 0.0
        
    # Return the maximum peak found in the window (Pessimistic approach)
    return float(np.max(peak_magnitudes))
