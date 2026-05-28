import argparse
import subprocess
import os
import sys

def get_fps(video_path):
    """Gets the frames per second (fps) of a video using ffprobe."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-show_entries", "stream=r_frame_rate", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        video_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
        num, den = output.split('/')
        fps = float(num) / float(den)
        return fps
    except Exception as e:
        print(f"Error getting fps (is ffprobe installed?): {e}")
        sys.exit(1)

def run_cmd(cmd, desc):
    print(f"{desc}...")
    # Run and capture output in case of error
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"\n--- ERROR executing: {' '.join(cmd)} ---")
        print(result.stdout)
        print("----------------------------------------\n")
    return result.returncode == 0

def extract_frame_and_clips(video_path, frame_number, fps=None, output_dir=".", prefix=""):
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        sys.exit(1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if fps is None:
        print("Probing video for FPS...")
        fps = get_fps(video_path)
    
    print(f"Using FPS: {fps}")
    
    # Calculate exact time for the frame
    target_time = frame_number / fps
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Add prefix if provided
    name_stem = f"{prefix}_{base_name}" if prefix else base_name
    
    # 1. Extract the exact frame as an image
    out_image = os.path.join(output_dir, f"{name_stem}_frame_{frame_number}.jpg")
    cmd_image = [
        "ffmpeg", "-y", 
        "-ss", str(target_time), 
        "-i", video_path, 
        "-frames:v", "1", 
        "-q:v", "2", 
        out_image
    ]
    run_cmd(cmd_image, f"Extracting frame {frame_number} at {target_time:.3f}s to {out_image}")
    
    # 2. Extract a 5-second clip ending AT that frame
    start_time_5s = max(0, target_time - 5)
    
    # To guarantee the exact frame is included, we calculate exactly how many frames we need.
    num_frames_5s = int((target_time - start_time_5s) * fps) + 1
    rel_target_5s = target_time - start_time_5s

    box_filter_5s = (
        f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.8:thickness=40:"
        f"enable='between(t,{rel_target_5s - 0.2},{rel_target_5s + 0.2})'"
    )
    
    out_clip_5s = os.path.join(output_dir, f"{name_stem}_5s_ending_at_{frame_number}.mp4")
    cmd_clip_5s = [
        "ffmpeg", "-y", 
        "-ss", str(start_time_5s), 
        "-i", video_path, 
        "-frames:v", str(num_frames_5s),
        "-vf", box_filter_5s,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", 
        out_clip_5s
    ]
    run_cmd(cmd_clip_5s, f"Extracting {num_frames_5s} frames clip ending at frame {frame_number} to {out_clip_5s}")
    
    # 3. Extract a context clip (3 seconds before, 3 seconds after) with a visual highlight
    context_duration_before = 3
    context_duration_after = 3
    start_time_ctx = max(0, target_time - context_duration_before)
    
    num_frames_ctx = int((target_time - start_time_ctx + context_duration_after) * fps) + 1
    relative_target_time = target_time - start_time_ctx
    
    out_clip_ctx = os.path.join(output_dir, f"{name_stem}_context_highlight_{frame_number}.mp4")
    
    # Use drawbox to draw a thick red border around the target frame for 0.4 seconds
    # This avoids using drawtext which requires external fonts and libfreetype
    box_filter = (
        f"drawbox=x=0:y=0:w=iw:h=ih:color=red@0.8:thickness=40:"
        f"enable='between(t,{relative_target_time - 0.2},{relative_target_time + 0.2})'"
    )
    
    cmd_clip_ctx = [
        "ffmpeg", "-y", 
        "-ss", str(start_time_ctx), 
        "-i", video_path, 
        "-frames:v", str(num_frames_ctx),
        "-vf", box_filter,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", 
        out_clip_ctx
    ]
    run_cmd(cmd_clip_ctx, f"Extracting context clip with red border highlight to {out_clip_ctx}")
    
    print("\nExtraction complete! Generated files:")
    print(f" - Image: {out_image}")
    print(f" - 5s Clip: {out_clip_5s}")
    print(f" - Context Clip: {out_clip_ctx}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract a specific frame and surrounding context clips from a video.")
    parser.add_argument("video_path", help="Path to the input video file (e.g., .mp4)")
    parser.add_argument("frame_number", type=int, help="The target frame number to extract")
    parser.add_argument("--fps", type=float, help="Explicitly provide the video FPS. If omitted, it will be probed automatically.")
    parser.add_argument("-o", "--output_dir", default=".", help="Directory to save the extracted files (default: current directory)")
    
    args = parser.parse_args()
    
    extract_frame_and_clips(args.video_path, args.frame_number, args.fps, args.output_dir)
