import json
import os
import sys
from extract_frame_context import extract_frame_and_clips, get_fps

def compare_results(video_path, our_json, adobe_json, output_root="./comparison_results"):
    # Load our results
    with open(our_json, 'r') as f:
        our_cuts = json.load(f)
    
    # Load Adobe results
    with open(adobe_json, 'r') as f:
        adobe_data = json.load(f)
    
    # Adobe cuts are at the 'start' of each clipitem
    adobe_frames = sorted(list(set([int(item['start']) for item in adobe_data])))
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    print(f"Probing video for FPS...")
    fps = get_fps(video_path)
    
    matched_adobe_frames = set()
    
    # 1. Process matches
    for cut in our_cuts:
        our_frame = cut['frame']
        rank = cut['rank']
        
        # Find Adobe frames within +/- 50 of our_frame
        group_frames = [f for f in adobe_frames if abs(f - our_frame) <= 50]
        
        group_dir = os.path.join(output_root, f"group_rank_{rank:03d}_frame_{our_frame}")
        os.makedirs(group_dir, exist_ok=True)
        
        print(f"\nProcessing Group: our_frame {our_frame} (Rank {rank})")
        
        # Extract our frame
        print(f"  -> Extracting OUR frame: {our_frame}")
        extract_frame_and_clips(video_path, our_frame, fps, group_dir, prefix="OURS")
        
        # Extract matched Adobe frames
        for a_frame in group_frames:
            matched_adobe_frames.add(a_frame)
            if a_frame == our_frame:
                continue # Already extracted
            print(f"  -> Extracting matched ADOBE frame: {a_frame}")
            extract_frame_and_clips(video_path, a_frame, fps, group_dir, prefix="ADOBE")

    # 2. Process missed Adobe frames
    missed_dir = os.path.join(output_root, "missed_by_pipeline")
    os.makedirs(missed_dir, exist_ok=True)
    
    missed_frames = [f for f in adobe_frames if f not in matched_adobe_frames]
    
    print(f"\nProcessing {len(missed_frames)} missed Adobe frames...")
    for m_frame in missed_frames:
        # Only process if not at 0 (start of video)
        if m_frame == 0: continue
        
        print(f"  -> Extracting MISSED Adobe frame: {m_frame}")
        extract_frame_and_clips(video_path, m_frame, fps, missed_dir, prefix="ADOBE")

    print(f"\nAutomation complete! Results in: {output_root}")

if __name__ == "__main__":
    # Hardcoded for the requested instance, but could be argparse
    VIDEO = "/Users/jaideepsingh.phogat/Downloads/input13.mp4"
    OUR_JSON = "./cut_analysis/input13/cut_points.json"
    ADOBE_JSON = "clipitems_output.json"
    
    if not os.path.exists(VIDEO):
        print(f"Error: Video not found at {VIDEO}")
        sys.exit(1)
        
    compare_results(VIDEO, OUR_JSON, ADOBE_JSON)
