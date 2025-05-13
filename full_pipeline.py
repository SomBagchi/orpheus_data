#!/usr/bin/env python3
"""
full_pipeline.py

This script provides functionality to retrieve episode IDs from a Spotify podcast.
"""

import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from tqdm import tqdm
import time
import json
import sys
from collections import defaultdict
import requests
from pydub import AudioSegment
from pathlib import Path
import uuid
import csv
import concurrent.futures
import logging

NUM_EPISODES = 10
RAW_DIR = "raw_clips"
API_BASE = "https://api.pyannote.ai/v1"
DIARISATION_DIR = "diarisation"
MAX_PAUSE_LENGTH = 1.0
MIN_BLOCK_DURATION = 3.0
PROCESSED_DIARISATION_DIR = "processed_diarisation"
FINAL_CLIPS_DIR = "final_clips"
CSV_FILE = "index.csv"
SHOW_NAME = "The Joe Rogan Experience"
MAX_CONCURRENT_JOBS = 15  # Balance between speed and API limits

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def authenticate():
    """
    Authenticate with the Spotify API.
    """
    load_dotenv()
    creds = SpotifyClientCredentials(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
    )
    sp = Spotify(client_credentials_manager=creds)
    return sp

def get_episodes(sp):
    """
    Retrieve episode IDs for a specified podcast show from Spotify.
    
    Args:
        sp: Spotify API object
        
    Returns:
        List of tuples containing (episode_id, episode_name, audio_preview_url)
    """
    # 1) Search for the show
    res = sp.search(
        q=SHOW_NAME,
        type="show",
        limit=1
    )
    show = res["shows"]["items"][0]
    show_id = show["id"]
    logger.info(f"Show ID for '{SHOW_NAME}': {show_id}")
    
    # 2) Get the num_episodes most recent episodes
    episodes = []
    limit = 50  # Spotify API typically allows max 50 per request
    offset = 0
    
    while True:
        page = sp.show_episodes(show_id, limit=limit, offset=offset, market="US")
        items = page["items"]
        if not items:
            break
    
        # Collect IDs, names, and preview URLs
        episodes.extend((ep["id"], ep["name"], ep.get("audio_preview_url")) for ep in items if ep is not None)
    
        # Advance offset; if fewer than `limit` came back, we're done
        offset += len(items)
        if len(items) < limit:
            break
    
        # Stop if we've collected enough episodes
        if len(episodes) >= NUM_EPISODES:
            episodes = episodes[:NUM_EPISODES]
            break
            
    return episodes

def download_previews(episodes):
    """
    Download previews for a batch of episodes.
    
    Args:
        sp: Spotify API object
        episodes: List of (episode_id, episode_name, audio_preview_url) tuples
    """
    # Create output directory if it doesn't exist
    os.makedirs(RAW_DIR, exist_ok=True)

    for episode in episodes:
        episode_id, _, preview_url = episode
        try:
            # Use the provided preview URL
            url = preview_url
                    
            # Download preview MP3
            resp = requests.get(url)
            resp.raise_for_status()
            
            filename = f"{RAW_DIR}/{episode_id}.mp3"
            with open(filename, 'wb') as f:
                f.write(resp.content)
                
            logger.info(f"Downloaded preview for episode {episode_id} to {filename}")

        except Exception as e:
            logger.error(f"Error downloading preview for episode {episode_id}: {e}")

def diarise_audio(episode):
    """
    Diarise an audio file.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    episode_id = episode[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(DIARISATION_DIR, exist_ok=True)

    load_dotenv()
    api_token = os.getenv("PYANNOTEAI_API_TOKEN")
    if not api_token:
        raise Exception("PYANNOTEAI_API_TOKEN not found in .env")

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    # 1. Create temporary storage location
    input_path = f"{RAW_DIR}/{episode_id}.mp3"
    media_uri = f"media://{os.path.basename(input_path)}"
    resp = requests.post(
        f"{API_BASE}/media/input",
        headers=headers,
        json={"url": media_uri}
    )
    resp.raise_for_status()
    presigned_url = resp.json()["url"]

    # 2. Upload the file bytes
    with open(input_path, "rb") as f:
        put_resp = requests.put(presigned_url, data=f)
        put_resp.raise_for_status()

    # 3. Submit diarization job (no webhook)
    job_resp = requests.post(
        f"{API_BASE}/diarize",
        headers=headers,
        json={"url": media_uri}
    )
    job_resp.raise_for_status()
    job = job_resp.json()
    job_id = job["jobId"]
    status = job["status"]
    logger.info(f"Job {job_id} submitted for episode {episode_id}, status={status}")

    # 4. Poll for completion
    while status not in ("succeeded", "failed", "canceled"):
        time.sleep(5)
        status_resp = requests.get(
            f"{API_BASE}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_token}"}
        )
        status_resp.raise_for_status()
        status = status_resp.json()["status"]
        logger.info(f"Polling job {job_id} for episode {episode_id}, status={status}")

    if status != "succeeded":
        raise Exception(f"Job ended with status: {status}")

    # 5. Fetch and write diarization output
    output = status_resp.json().get("output", {})
    output_path = f"{DIARISATION_DIR}/{episode_id}.json"
    with open(output_path, "w") as fo:
        json.dump(output, fo, indent=2)
    logger.info(f"Diarization results saved to {output_path} for episode {episode_id}")
    return episode_id

def diarise_audios_concurrently(episodes):
    """
    Diarise multiple audio files concurrently.
    
    Args:
        episodes: List of episode tuples
    
    Returns:
        List of episode IDs that were successfully processed
    """
    # Create output directory if it doesn't exist
    os.makedirs(DIARISATION_DIR, exist_ok=True)
    
    successful_episodes = []
    
    # Create a thread pool with limited concurrency to avoid API rate limiting
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        # Submit all diarisation jobs
        future_to_episode = {
            executor.submit(diarise_audio, episode): episode
            for episode in episodes
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_episode), 
                          total=len(future_to_episode), 
                          desc="Diarising episodes"):
            episode = future_to_episode[future]
            episode_id = episode[0]
            try:
                result = future.result()
                successful_episodes.append(episode)
                logger.info(f"Successfully diarised episode {episode_id}")
            except Exception as e:
                logger.error(f"Error diarising episode {episode_id}: {e}")
    
    return successful_episodes

def process_diarisation(episode):
    """
    Process a diarisation file.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    episode_id = episode[0]
    
    # Create output directory if it doesn't exist
    os.makedirs(PROCESSED_DIARISATION_DIR, exist_ok=True)

    # Load input file
    input_file = f"{DIARISATION_DIR}/{episode_id}.json"
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract diarization array
    diarization = data.get("diarization", [])
    if not diarization:
        raise Exception("No diarization data found")
    
    # Sort blocks by start time
    diarization.sort(key=lambda x: x["start"])
    
    # Step 1: Fuse consecutive blocks from the same speaker if gap < MAX_PAUSE_LENGTH
    fused_blocks = []
    current_block = None
    
    for block in diarization:
        if current_block is None:
            # First block
            current_block = block.copy()
        elif (block["speaker"] == current_block["speaker"] and 
              block["start"] - current_block["end"] <= MAX_PAUSE_LENGTH):
            # Same speaker with gap < MAX_PAUSE_LENGTH, fuse blocks
            current_block["end"] = block["end"]
        else:
            # Different speaker or gap too large, add current block and start new one
            fused_blocks.append(current_block)
            current_block = block.copy()
    
    # Add the last block
    if current_block is not None:
        fused_blocks.append(current_block)
    
    # Step 2: Calculate total speaking time per speaker
    speaking_times = defaultdict(float)
    for block in fused_blocks:
        speaking_times[block["speaker"]] += block["end"] - block["start"]
    
    # Find the speaker who spoke the most
    if not speaking_times:
        raise Exception("No speaking time data available")
        
    most_speaking_speaker = max(speaking_times.items(), key=lambda x: x[1])[0]
    
    # Step 3: Keep only blocks from the speaker who spoke the most
    filtered_blocks = [block for block in fused_blocks if block["speaker"] == most_speaking_speaker]
    
    # Step 4: Remove blocks shorter than 3 seconds
    filtered_blocks = [block for block in filtered_blocks if (block["end"] - block["start"]) >= MIN_BLOCK_DURATION]
    
    # Prepare output
    output_data = {
        "diarization": filtered_blocks
    }
    
    # Write to new file
    output_file = f"{PROCESSED_DIARISATION_DIR}/processed_{episode_id}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nProcessed diarisation written to {output_file}")

def clip_audio(episode):
    """
    Clip an audio file based on diarisation segments.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    episode_id, episode_name, _ = episode
    
    # Create output directory if it doesn't exist
    os.makedirs(FINAL_CLIPS_DIR, exist_ok=True)
    
    # Load diarisation data
    with open(f"{PROCESSED_DIARISATION_DIR}/processed_{episode_id}.json", 'r') as f:
        data = json.load(f)
    
    segments = data.get("diarization", [])
    if not segments:
        raise Exception("No diarisation segments found")
    
    # Load audio file
    audio = AudioSegment.from_file(f"{RAW_DIR}/{episode_id}.mp3")
    
    # Open CSV file in append mode
    with open(CSV_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for i, segment in enumerate(segments, 1):
            start_time = segment["start"] * 1000  # Convert to milliseconds
            end_time = segment["end"] * 1000
            
            # Calculate duration in seconds
            duration_seconds = (end_time - start_time) / 1000
            
            # Generate UUID for this clip
            clip_uuid = str(uuid.uuid4())
            
            # Extract segment
            clip = audio[start_time:end_time]
            
            # Create filename with UUID
            os.makedirs(f"{FINAL_CLIPS_DIR}/{episode_id}", exist_ok=True)
            output_file = f"{FINAL_CLIPS_DIR}/{episode_id}/{clip_uuid}.mp3"
            
            # Export clip
            clip.export(output_file, format="mp3")
            
            # Write to CSV
            csv_writer.writerow([clip_uuid, i, episode_id, episode_name, SHOW_NAME, duration_seconds])
            
            logger.info(f"Created: {output_file} ({duration_seconds:.2f} seconds)")
    
    logger.info(f"\nDone! Created {len(segments)} clips in {FINAL_CLIPS_DIR}/ directory")

if __name__ == "__main__":   
    # Ensure the directory exists for the CSV file
    csv_path = Path(CSV_FILE)
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['uuid', 'clip_number', 'episode_id', 'episode_name', 'show_name', 'duration_seconds'])
            logger.info(f"Created index CSV file at {CSV_FILE}")
    
    sp = authenticate()
    episodes = get_episodes(sp)
    download_previews(episodes)
    
    logger.info(f"Starting concurrent diarisation of {len(episodes)} episodes")
    successful_episodes = diarise_audios_concurrently(episodes)
    logger.info(f"Completed diarisation of {len(successful_episodes)}/{len(episodes)} episodes")
    
    # # Process all episodes using the concurrent approach
    # for episode in tqdm(successful_episodes):
    #     try:
    #         process_diarisation(episode)
    #         clip_audio(episode)
    #         logger.info(f"Successfully processed episode {episode[0]}")
    #     except Exception as e:
    #         logger.error(f"Error processing episode {episode[0]}: {e}")
