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

def get_episode_ids(sp):
    """
    Retrieve episode IDs for a specified podcast show from Spotify.
    
    Args:
        sp: Spotify API object
        
    Returns:
        List of episode IDs
    """
    # 1) Search for the show
    res = sp.search(
        q=SHOW_NAME,
        type="show",
        limit=1
    )
    show = res["shows"]["items"][0]
    show_id = show["id"]
    # print(f"Show ID for '{SHOW_NAME}':", show_id)
    
    # 2) Get the num_episodes most recent episodes
    episode_ids = []
    limit = 50  # Spotify API typically allows max 50 per request
    offset = 0
    
    while True:
        page = sp.show_episodes(show_id, limit=limit, offset=offset, market="US")
        items = page["items"]
        if not items:
            break
    
        # Collect IDs
        episode_ids.extend(ep["id"] for ep in items if ep is not None)
    
        # Advance offset; if fewer than `limit` came back, we're done
        offset += len(items)
        if len(items) < limit:
            break
    
        # Stop if we've collected enough episodes
        if len(episode_ids) >= NUM_EPISODES:
            episode_ids = episode_ids[:NUM_EPISODES]
            break
            
    return episode_ids

def download_preview(sp, episode_id):
    """
    Download previews for a list of episode IDs.
    
    Returns:
        str: The name of the episode
    """
    # Create output directory if it doesn't exist
    os.makedirs(RAW_DIR, exist_ok=True)

    try:
        # Get episode metadata
        ep = sp.episode(episode_id)
        url = ep.get("audio_preview_url")
        episode_name = ep.get("name", "Unknown Episode")
        
        if not url:
            print(f"No preview available for episode {episode_id}")
            
        # Download preview MP3
        resp = requests.get(url)
        resp.raise_for_status()
		
        os.makedirs(RAW_DIR, exist_ok=True)
        filename = f"{RAW_DIR}/{episode_id}.mp3"
        with open(filename, 'wb') as f:
            f.write(resp.content)

    except Exception as e:
        print(f"Error processing episode {episode_id}: {e}")
                
    # print(f"Downloaded preview for episode {episode_id} to {filename}")
    
    return episode_name

def diarise_audio(episode_id):
    """
    Diarise an audio file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(DIARISATION_DIR, exist_ok=True)

    load_dotenv()
    api_token = os.getenv("PYANNOTEAI_API_TOKEN")
    if not api_token:
        print("Error: PYANNOTEAI_API_TOKEN not found in .env")
        sys.exit(1)

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
    # print(f"Job {job_id} submitted, status={status}")

    # 4. Poll for completion
    while status not in ("succeeded", "failed", "canceled"):
        time.sleep(5)
        status_resp = requests.get(
            f"{API_BASE}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_token}"}
        )
        status_resp.raise_for_status()
        status = status_resp.json()["status"]
        # print(f"Polling job {job_id}, status={status}")

    if status != "succeeded":
        print(f"Job ended with status: {status}")
        sys.exit(1)

    # 5. Fetch and write diarization output
    output = status_resp.json().get("output", {})
    os.makedirs(DIARISATION_DIR, exist_ok=True)
    output_path = f"{DIARISATION_DIR}/{episode_id}.json"
    with open(output_path, "w") as fo:
        json.dump(output, fo, indent=2)
    # print(f"Diarization results saved to {output_path}")

def process_diarisation(episode_id):
    """
    Process a diarisation file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(PROCESSED_DIARISATION_DIR, exist_ok=True)

    # Load input file
    input_file = f"{DIARISATION_DIR}/{episode_id}.json"
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract diarization array
    diarization = data.get("diarization", [])
    if not diarization:
        print("No diarization data found")
        return
    
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
        print("No speaking time data available")
        return
        
    most_speaking_speaker = max(speaking_times.items(), key=lambda x: x[1])[0]
    
    # Print speaker stats
    # print("Speaker statistics after fusion:")	
    # for speaker, time in speaking_times.items():
    #     print(f"{speaker}: {time:.2f} seconds")
    # print(f"\nKeeping only blocks from {most_speaking_speaker} (spoke the most)")
    
    # Step 3: Keep only blocks from the speaker who spoke the most
    filtered_blocks = [block for block in fused_blocks if block["speaker"] == most_speaking_speaker]
    
    # Step 4: Remove blocks shorter than 3 seconds
    min_block_duration = 3.0
    block_count_before_duration_filter = len(filtered_blocks)
    filtered_blocks = [block for block in filtered_blocks if (block["end"] - block["start"]) >= min_block_duration]
    
    # Prepare output
    output_data = {
        "diarization": filtered_blocks
    }
    
    # Write to new file
    output_file = f"{PROCESSED_DIARISATION_DIR}/processed_{episode_id}.json"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # print(f"\nProcessed diarisation written to {output_file}")
    # print(f"Original block count: {len(diarization)}")
    # print(f"After fusion: {len(fused_blocks)}")
    # print(f"After keeping most speaking speaker: {block_count_before_duration_filter}")
    # print(f"Final block count (after removing blocks < {min_block_duration}s): {len(filtered_blocks)}")

def clip_audio(episode_id, episode_name):
    """
    Clip an audio file based on diarisation segments.
    
    Args:
        diarisation_file: Path to JSON diarisation file
        audio_file: Path to MP3 audio file
        output_dir: Directory to save the clips (created if it doesn't exist)
    """
    # Create output directory if it doesn't exist
    os.makedirs(FINAL_CLIPS_DIR, exist_ok=True)
    
    # Load diarisation data
    with open(f"{PROCESSED_DIARISATION_DIR}/processed_{episode_id}.json", 'r') as f:
        data = json.load(f)
    
    segments = data.get("diarization", [])
    if not segments:
        print("No diarisation segments found")
        return
    
    # Load audio file
    # print(f"Loading audio file: {RAW_DIR}/{episode_id}.mp3")
    audio = AudioSegment.from_file(f"{RAW_DIR}/{episode_id}.mp3")
    
    # Process each segment
    # print(f"Creating {len(segments)} clips...")
    
    # Open CSV file in append mode
    with open(CSV_FILE, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        for i, segment in enumerate(segments, 1):
            start_time = segment["start"] * 1000  # Convert to milliseconds
            end_time = segment["end"] * 1000
            speaker = segment["speaker"]
            
            # Generate UUID for this clip
            clip_uuid = str(uuid.uuid4())
            
            # Extract segment
            clip = audio[start_time:end_time]
            
            # Create filename with UUID
            output_file = f"{FINAL_CLIPS_DIR}/{clip_uuid}.mp3"
            
            # Export clip
            clip.export(output_file, format="mp3")
            
            # Write to CSV
            csv_writer.writerow([clip_uuid, i, episode_id, episode_name, SHOW_NAME])
            
            # print(f"Created: {output_file} ({(end_time-start_time)/1000:.2f} seconds)")
    
    # print(f"\nDone! Created {len(segments)} clips in {FINAL_CLIPS_DIR}/ directory")

def process_episode(sp, episode_id):
    episode_name = download_preview(sp, episode_id)
    diarise_audio(episode_id)
    process_diarisation(episode_id)
    clip_audio(episode_id, episode_name)

if __name__ == "__main__":
    sp = authenticate()
    episode_ids = get_episode_ids(sp)    
    # Ensure the directory exists for the CSV file
    csv_path = Path(CSV_FILE)
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV with headers if it doesn't exist
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['uuid', 'clip_number', 'episode_id', 'episode_name', 'show_name'])
            # print(f"Created index CSV file at {CSV_FILE}")
            
    # Process all episodes
    for episode_id in tqdm(episode_ids):
        # print(f"\nProcessing episode: {episode_id}")
        process_episode(sp, episode_id)
