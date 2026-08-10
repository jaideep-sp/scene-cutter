#!/usr/bin/env python3
import os
import sys
import time
import uuid
import shutil
import subprocess
import logging
from pathlib import Path
from collections import defaultdict
from google.cloud import storage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("bucket_watcher")

# Configuration from environment variables
BUCKET_NAME = os.getenv("SCENE_CUTTER_BUCKET")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))
LOCAL_JOBS_DIR = Path(os.getenv("LOCAL_JOBS_DIR", "./jobs"))

# Pipeline configuration
STRATEGY = os.getenv("SCENE_CUTTER_STRATEGY", "visual_ai")
PROFILE = os.getenv("SCENE_CUTTER_PROFILE", "drama")
SPLIT_VIDEO = os.getenv("SCENE_CUTTER_SPLIT", "false").lower() in ("true", "1", "yes")
CREATE_SNIPPETS = os.getenv("SCENE_CUTTER_SNIPPETS", "false").lower() in ("true", "1", "yes")

# Supported video formats
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".ts", ".webm")


def get_gcs_client():
    try:
        return storage.Client()
    except Exception as e:
        logger.error(f"Failed to initialize GCS Client. Ensure service account is attached: {str(e)}")
        sys.exit(1)


def scan_bucket_for_jobs(bucket):
    """
    Scans the bucket, groups all blobs by their containing directory,
    and returns a list of (folder_prefix, video_blob) that need processing.
    """
    logger.info("Scanning bucket for new video assets...")
    
    # 1. Fetch all blobs from the bucket
    try:
        blobs = list(bucket.list_blobs())
    except Exception as e:
        logger.error(f"Error listing blobs in bucket: {str(e)}")
        return []

    # 2. Group blobs by folder prefix
    folders = defaultdict(list)
    for blob in blobs:
        parts = blob.name.split('/')
        if len(parts) > 1:
            folder_prefix = '/'.join(parts[:-1]) + '/'
            folders[folder_prefix].append(blob)
        else:
            # Root level files
            folders["/"].append(blob)

    # 3. Identify folders requiring processing
    unprocessed_jobs = []
    for folder_prefix, folder_blobs in folders.items():
        # Check if output cut_points.json already exists in this folder
        already_processed = any(blob.name.endswith("cut_points.json") for blob in folder_blobs)
        if already_processed:
            continue

        # Look for a video file in this folder
        video_blob = None
        for blob in folder_blobs:
            if blob.name.lower().endswith(VIDEO_EXTENSIONS):
                video_blob = blob
                break

        if video_blob:
            unprocessed_jobs.append((folder_prefix, video_blob))

    return unprocessed_jobs


def process_job(bucket, folder_prefix, video_blob):
    """Downloads, processes, uploads, and cleans up an individual asset."""
    job_id = str(uuid.uuid4())[:8]
    local_job_dir = LOCAL_JOBS_DIR / job_id
    local_out_dir = local_job_dir / "output"
    
    logger.info(f"Starting Job {job_id} for asset in '{folder_prefix}'")
    
    try:
        # Create directories
        local_job_dir.mkdir(parents=True, exist_ok=True)
        local_out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Download video file
        video_filename = os.path.basename(video_blob.name)
        local_video_path = local_job_dir / video_filename
        
        logger.info(f"Downloading video from GCS: gs://{BUCKET_NAME}/{video_blob.name} ...")
        start_time = time.time()
        video_blob.download_to_filename(str(local_video_path))
        download_duration = time.time() - start_time
        logger.info(f"Downloaded in {download_duration:.2f}s ({local_video_path.stat().st_size / 1024 / 1024:.2f} MB)")

        # 2. Build and run Scene Cutter command
        cmd = [
            sys.executable, "main.py",
            str(local_video_path),
            "--out", str(local_out_dir),
            "--strategy", STRATEGY,
        ]
        if PROFILE:
            cmd += ["--profile", PROFILE]
        if SPLIT_VIDEO:
            cmd.append("--split")
        if CREATE_SNIPPETS:
            cmd.append("--snippets")

        logger.info(f"Executing Scene Cutter pipeline command: {' '.join(cmd)}")
        pipeline_start = time.time()
        
        # Run subprocess and stream logs directly
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ""):
            clean_line = line.strip()
            if clean_line:
                logger.info(f"[pipeline] {clean_line}")
                
        process.wait()
        pipeline_duration = time.time() - pipeline_start

        if process.returncode != 0:
            raise RuntimeError(f"Scene Cutter pipeline exited with code {process.returncode}")
            
        logger.info(f"Pipeline executed successfully in {pipeline_duration:.2f}s")

        # 3. Upload output files back to the same folder on GCS
        logger.info(f"Uploading generated output files back to GCS directory '{folder_prefix}'...")
        upload_count = 0
        for root, _, files in os.walk(local_out_dir):
            for file in files:
                local_file_path = Path(root) / file
                # Determine GCS destination path relative to GCS folder prefix
                rel_path = os.path.relpath(local_file_path, local_out_dir)
                # If folder_prefix is root "/", do not prepend it
                dest_blob_name = rel_path if folder_prefix == "/" else f"{folder_prefix}{rel_path}"
                
                logger.info(f"Uploading output file to GCS: gs://{BUCKET_NAME}/{dest_blob_name}")
                dest_blob = bucket.blob(dest_blob_name)
                dest_blob.upload_from_filename(str(local_file_path))
                upload_count += 1
                
        logger.info(f"Successfully uploaded {upload_count} output file(s) for job {job_id}")

    except Exception as e:
        logger.error(f"Error processing job {job_id} for asset '{folder_prefix}': {str(e)}")
        # Optionally upload an error log file to the folder in GCS so users know it failed
        try:
            error_blob_name = f"{folder_prefix}error.log" if folder_prefix != "/" else "error.log"
            error_blob = bucket.blob(error_blob_name)
            error_blob.upload_from_string(f"Job {job_id} failed.\nError: {str(e)}")
            logger.info(f"Uploaded error log to gs://{BUCKET_NAME}/{error_blob_name}")
        except Exception as upload_err:
            logger.error(f"Could not upload error log to GCS: {str(upload_err)}")

    finally:
        # 4. Clean up the local folder to prevent disk exhaustion
        if local_job_dir.exists():
            logger.info(f"Cleaning up local job directory '{local_job_dir}' to free disk space")
            shutil.rmtree(local_job_dir)
            logger.info(f"Disk cleanup complete for Job {job_id}")


def main():
    if not BUCKET_NAME:
        logger.error("SCENE_CUTTER_BUCKET environment variable must be specified!")
        sys.exit(1)

    logger.info("==================================================")
    logger.info("Starting Scene Cutter Continuous Bucket Watcher")
    logger.info(f"Target Bucket:      {BUCKET_NAME}")
    logger.info(f"Polling Interval:   {POLL_INTERVAL} seconds")
    logger.info(f"Pipeline Strategy:  {STRATEGY}")
    logger.info(f"Content Profile:    {PROFILE}")
    logger.info(f"Split Videos:       {SPLIT_VIDEO}")
    logger.info(f"Create Snippets:    {CREATE_SNIPPETS}")
    logger.info("==================================================")

    # Initialize GCS client & bucket
    client = get_gcs_client()
    bucket = client.bucket(BUCKET_NAME)

    # Main continuous polling loop
    while True:
        try:
            # 1. Scan for new assets
            jobs = scan_bucket_for_jobs(bucket)
            if jobs:
                logger.info(f"Found {len(jobs)} folder(s) requiring processing.")
                for folder_prefix, video_blob in jobs:
                    process_job(bucket, folder_prefix, video_blob)
            else:
                logger.info("No new video assets found. Sitting tight.")
                
        except Exception as e:
            logger.error(f"Unexpected error in watcher main loop: {str(e)}")
            
        logger.info(f"Sleeping for {POLL_INTERVAL} seconds...")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
