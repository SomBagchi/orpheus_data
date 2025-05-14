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
import traceback
import whisperx
import torch
import re

NUM_EPISODES = 10
RAW_DIR = "raw_clips"
API_BASE = "https://api.pyannote.ai/v1"
DIARISATION_DIR = "diarisation"
MAX_PAUSE_LENGTH = 1.0
MIN_CLIP_DURATION = 3.0
PROCESSED_DIARISATION_DIR = "processed_diarisation"
FINAL_CLIPS_DIR = "final_clips"
CSV_FILE = "index.csv"
SHOW_NAME = "The Joe Rogan Experience"
PENULTIMATE_CLIPS_DIR = "penultimate_clips"
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
            traceback.print_exc()

def diarise_audio(episode):
    """
    Diarise an audio file and save the results.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    os.makedirs(DIARISATION_DIR, exist_ok=True)
    
    episode_id, _, _ = episode
    
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
    diarisation_data = status_resp.json().get("output", {})
    output_path = f"{DIARISATION_DIR}/{episode_id}.json"
    with open(output_path, "w") as fo:
        json.dump(diarisation_data, fo, indent=2)
    logger.info(f"Diarization results saved to {output_path} for episode {episode_id}")
    
    return diarisation_data
    
def process_diarisation(episode, diarisation_data):
    """
    Process the diarisation output and create clips.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    os.makedirs(PROCESSED_DIARISATION_DIR, exist_ok=True)
    
    episode_id, episode_name, _ = episode
    
	# Extract speaker segments
    segments = diarisation_data.get('diarization', [])
        
    # Count speaking time per speaker
    speaker_durations = defaultdict(float)
    for segment in segments:
        speaker = segment.get('speaker')
        if speaker:
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            duration = end - start
            speaker_durations[speaker] += duration
        
    # Find the maximal speaker
    if not speaker_durations:
        logger.error(f"No speakers found in diarisation for episode {episode_id}")
        return None
            
    maximal_speaker = max(speaker_durations, key=speaker_durations.get)
    logger.info(f"Maximal speaker for episode {episode_id} is {maximal_speaker} with {speaker_durations[maximal_speaker]:.2f} seconds")
        
    # Remove overlaps with other speakers
    maximal_segments = sorted([s for s in segments if s.get('speaker') == maximal_speaker], key=lambda x: x['start'])
    other_segments = sorted([s for s in segments if s.get('speaker') != maximal_speaker], key=lambda x: x['start'])
        
    if other_segments:
        combined_other_segments = [other_segments[0]]
        for other_seg in other_segments[1:]:
            if other_seg['start'] <= combined_other_segments[-1]['end']:
                combined_other_segments[-1]['end'] = max(combined_other_segments[-1]['end'], other_seg['end'])
            else:
                combined_other_segments.append(other_seg)
            
        result_segments = []
        j = 0 # combined_other_segments pointer
        n = len(combined_other_segments)
        for seg in maximal_segments:
            while j<n and combined_other_segments[j]['end'] <= seg['start']:
                j += 1
                
            current_start = seg['start']
            while j<n and combined_other_segments[j]['start'] < seg['end']:
                if combined_other_segments[j]['start'] > current_start:
                    result_segments.append({
                        'start': current_start,
                        'end': min(seg['end'], combined_other_segments[j]['start'])
                    })
                current_start = max(current_start, combined_other_segments[j]['end'])
                if current_start >= seg['end']:
                    break
                j += 1
                
            if current_start < seg['end']:
                result_segments.append({
                    'start': current_start,
                    'end': seg['end']
                })
    else:
        combined_other_segments = []
        result_segments = maximal_segments
        
    # Combine segments and mark their type (maximal or other)
    all_segments = []
    for segment in result_segments:
        segment['type'] = 'maximal'
        all_segments.append(segment)
    for segment in combined_other_segments:
        segment['type'] = 'other'
        all_segments.append(segment)
    all_segments.sort(key=lambda x: x['start'])
        
    # Fuse consecutive maximal segments with small gaps
    fused_segments = []
    i = 0
    while i < len(all_segments):
        current_segment = all_segments[i].copy()
        
        # If current segment is maximal, look ahead for potential fusion
        if current_segment['type'] == 'maximal':
            while i + 1 < len(all_segments):
                next_segment = all_segments[i + 1]
                    
                # Check if next segment is also maximal and gap is small enough
                if (next_segment['type'] == 'maximal' and 
                    next_segment['start'] - current_segment['end'] <= MAX_PAUSE_LENGTH):
                    # Fuse the segments
                    current_segment['end'] = next_segment['end']
                    i += 1  # Skip the next segment as we've fused it
                else:
                    break  # No more segments to fuse
            
        fused_segments.append(current_segment)
        i += 1
    
    # Filter fused segments to keep only 'maximal' type
    maximal_segments = []
    for segment in fused_segments:
        if segment['type'] == 'maximal':
            maximal_segments.append(segment)
    
    # Replace fused_segments with filtered maximal_segments
    fused_segments = maximal_segments
    maximal_count = sum(1 for segment in all_segments if segment['type'] == 'maximal')
    logger.info(f"Found {maximal_count} maximal segments in all_segments, fused to {len(fused_segments)} segments for episode {episode_id}")
    
    # Save processed diarisation
    processed_diarisation = {
        'episode_id': episode_id,
        'episode_name': episode_name,
        'maximal_speaker': maximal_speaker,
        'segments': fused_segments
    }
        
    processed_path = f"{PROCESSED_DIARISATION_DIR}/{episode_id}.json"
    with open(processed_path, 'w') as f:
        json.dump(processed_diarisation, f, indent=2)
        
    logger.info(f"Processed diarisation saved to {processed_path} for episode {episode_id}")
    
    return processed_diarisation

def create_penultimate_clips(episode, processed_diarisation):
    """
    Create penultimate clips from the processed diarisation.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    os.makedirs(PENULTIMATE_CLIPS_DIR, exist_ok=True)
    
    episode_id, episode_name, _ = episode
    
    input_path = f"{RAW_DIR}/{episode_id}.mp3"
    
	# 7. Create audio clips
    audio = AudioSegment.from_mp3(input_path)
        
	# Create output directory for this episode
    episode_clips_dir = f"{PENULTIMATE_CLIPS_DIR}/{episode_id}"
    os.makedirs(episode_clips_dir, exist_ok=True)
        
    fused_segments = processed_diarisation['segments']
    
	# Create a clip for each segment
    for i, segment in enumerate(fused_segments):
        start_ms = int(segment['start'] * 1000)  # Convert to milliseconds
        end_ms = int(segment['end'] * 1000)
            
		# Extract the audio clip
        clip = audio[start_ms:end_ms]
            
		# Generate filename
        clip_filename = f"{episode_id}_{i+1}.mp3"
        clip_path = f"{episode_clips_dir}/{clip_filename}"
            
        # Save the clip
        clip.export(clip_path, format="mp3")
            
        # Save metadata
        clip_meta = {
	        'episode_id': episode_id,
            'episode_name': episode_name,
            'clip_number': i+1,
            'start_time': segment['start'],
            'end_time': segment['end'],
            'duration': segment['end'] - segment['start']
        }
            
        meta_path = f"{episode_clips_dir}/{episode_id}_{i+1}.json"
        with open(meta_path, 'w') as f:
            json.dump(clip_meta, f, indent=2)
        
    logger.info(f"Created {len(fused_segments)} clips for episode {episode_id} in {episode_clips_dir}")

def transcribe_and_process_clips(episode):
    """
    Transcribe clips using WhisperX and process them according to punctuation rules.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    os.makedirs(FINAL_CLIPS_DIR, exist_ok=True)
    
    episode_id, episode_name, _ = episode
    
    # Ensure episode directory exists in final clips
    episode_final_dir = f"{FINAL_CLIPS_DIR}/{episode_id}"
    os.makedirs(episode_final_dir, exist_ok=True)
    
    # Load WhisperX model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = whisperx.load_model("base", device, compute_type=compute_type)
    
    # Get the penultimate clips directory for this episode
    episode_clips_dir = f"{PENULTIMATE_CLIPS_DIR}/{episode_id}"
    
    # Get all the MP3 files in the directory
    mp3_files = [f for f in os.listdir(episode_clips_dir) if f.endswith('.mp3')]
    
    # Process each clip
    for mp3_file in mp3_files:
        clip_path = f"{episode_clips_dir}/{mp3_file}"
        
        # Get metadata file path
        meta_file = mp3_file.replace('.mp3', '.json')
        meta_path = f"{episode_clips_dir}/{meta_file}"
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        clip_number = metadata['clip_number']
        original_start_time = metadata['start_time']
        original_end_time = metadata['end_time']
        original_duration = metadata['duration']
        
        # Skip if clip is too short
        if original_duration < MIN_CLIP_DURATION:
            logger.info(f"Skipping clip {mp3_file} as it's too short ({original_duration} seconds)")
            continue
        
        # Transcribe audio with WhisperX
        try:
            logger.info(f"Transcribing {mp3_file}")
            result = model.transcribe(clip_path)
            
            # Get the transcript
            segments = result["segments"]
            
            if not segments:
                logger.warning(f"No transcript generated for {mp3_file}")
                continue
                
            # Combine all segments into a full transcript
            full_transcript = " ".join([s["text"].strip() for s in segments])
            
            # New start and end times based on punctuation rules
            new_start_time = original_start_time
            new_end_time = original_end_time
            
            # Check if transcript starts with a capital letter
            if not full_transcript or not full_transcript[0].isupper():
                # Find the first punctuation mark
                punctuation_pattern = r'[,.!?;]'
                first_punct_match = re.search(punctuation_pattern, full_transcript)
                
                if first_punct_match and segments:
                    # Find which segment contains the punctuation
                    punct_pos = first_punct_match.start()
                    char_count = 0
                    
                    for segment in segments:
                        segment_text = segment["text"].strip()
                        if char_count + len(segment_text) > punct_pos:
                            # Found the segment with the punctuation
                            relative_pos = punct_pos - char_count
                            
                            # Estimate timestamp based on relative position in the segment
                            if len(segment_text) > 0:
                                segment_duration = segment["end"] - segment["start"]
                                relative_time = segment["start"] + (relative_pos / len(segment_text)) * segment_duration
                                new_start_time = original_start_time + relative_time
                            break
                        
                        char_count += len(segment_text) + 1  # +1 for the space we add when joining
            
            # Check if transcript ends with punctuation
            if full_transcript and not re.search(r'[,.!?;]$', full_transcript):
                # Find the last punctuation mark
                last_punct_match = None
                for match in re.finditer(r'[,.!?;]', full_transcript):
                    last_punct_match = match
                
                if last_punct_match and segments:
                    # Find which segment contains the last punctuation
                    punct_pos = last_punct_match.start()
                    char_count = 0
                    
                    for segment in segments:
                        segment_text = segment["text"].strip()
                        if char_count + len(segment_text) > punct_pos:
                            # Found the segment with the punctuation
                            relative_pos = punct_pos - char_count
                            
                            # Estimate timestamp based on relative position in the segment
                            if len(segment_text) > 0:
                                segment_duration = segment["end"] - segment["start"]
                                relative_time = segment["start"] + (relative_pos / len(segment_text)) * segment_duration
                                new_end_time = original_start_time + relative_time
                            break
                        
                        char_count += len(segment_text) + 1  # +1 for the space we add when joining
            
            # Check if the new clip duration is sufficient
            new_duration = new_end_time - new_start_time
            if new_duration < MIN_CLIP_DURATION:
                logger.info(f"Skipping clip {mp3_file} as the processed duration ({new_duration} seconds) is too short")
                continue
            
            # Load the audio and create the new clip
            audio = AudioSegment.from_mp3(clip_path)
            
            # Convert times to milliseconds for pydub
            new_start_ms = int((new_start_time - original_start_time) * 1000)
            new_end_ms = int((new_end_time - original_start_time) * 1000)
            
            # Ensure we don't go out of bounds
            new_start_ms = max(0, new_start_ms)
            new_end_ms = min(len(audio), new_end_ms)
            
            # Create the new clip
            new_clip = audio[new_start_ms:new_end_ms]
            
            # Generate UUID for the clip
            clip_uuid = str(uuid.uuid4())
            
            # Save the new clip
            new_clip_path = f"{episode_final_dir}/{clip_uuid}.mp3"
            new_clip.export(new_clip_path, format="mp3")
            
            # Add entry to CSV
            with open(CSV_FILE, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([
                    clip_uuid, 
                    clip_number, 
                    episode_id, 
                    episode_name, 
                    SHOW_NAME, 
                    new_duration,
                    full_transcript
                ])
            
            logger.info(f"Created final clip for {mp3_file} at {new_clip_path}")
            
        except Exception as e:
            logger.error(f"Error processing clip {mp3_file}: {e}")
            traceback.print_exc()

def process_audio(episode):
    """
    Diarise an audio file and process the results to create clips.
    
    Args:
        episode: Tuple of (episode_id, episode_name, audio_preview_url)
    """
    episode_id, episode_name, _ = episode
    
    # Create output directories
    os.makedirs(PENULTIMATE_CLIPS_DIR, exist_ok=True)

    diarisation_data = diarise_audio(episode)
    processed_diarisation = process_diarisation(episode, diarisation_data)
    create_penultimate_clips(episode, processed_diarisation)

def process_audios_concurrently(episodes):
    """
    Process multiple audio files concurrently.
    
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
            executor.submit(process_audio, episode): episode
            for episode in episodes
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_episode), 
                          total=len(future_to_episode), 
                          desc="Processing episodes"):
            episode = future_to_episode[future]
            episode_id = episode[0]
            try:
                result = future.result()
                successful_episodes.append(episode)
                logger.info(f"Successfully processed episode {episode_id}")
            except Exception as e:
                logger.error(f"Error processing: {episode_id}: {e}")
    
    return successful_episodes

def transcribe_and_process_clips_concurrently(episodes):
    """
    Transcribe and process clips concurrently.
    
    Args:
        episodes: List of episode tuples
    """
    # Create output directory if it doesn't exist
    os.makedirs(FINAL_CLIPS_DIR, exist_ok=True)
    
    # Create a thread pool with limited concurrency 
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS) as executor:
        # Submit all transcription jobs
        future_to_episode = {
            executor.submit(transcribe_and_process_clips, episode): episode
            for episode in episodes
        }
        
        # Process results as they complete
        for future in tqdm(concurrent.futures.as_completed(future_to_episode), 
                          total=len(future_to_episode), 
                          desc="Transcribing and processing episodes"):
            episode = future_to_episode[future]
            episode_id = episode[0]
            try:
                future.result()
                logger.info(f"Successfully transcribed and processed episode {episode_id}")
            except Exception as e:
                logger.error(f"Error transcribing and processing: {episode_id}: {e}")

if __name__ == "__main__":   
    # Ensure the directory exists for the CSV file
    csv_path = Path(CSV_FILE)
    if not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create CSV with headers if it doesn't exist, or check and update if it exists
    csv_headers = ['uuid', 'clip_number', 'episode_id', 'episode_name', 'show_name', 'duration_seconds', 'transcript']
    
    if not os.path.exists(CSV_FILE):
        # Create new CSV with headers
        with open(CSV_FILE, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_headers)
            logger.info(f"Created index CSV file at {CSV_FILE}")
    
    sp = authenticate()
    episodes = get_episodes(sp)
    download_previews(episodes)
    
    logger.info(f"Starting concurrent processing of {len(episodes)} episodes")
    successful_episodes = process_audios_concurrently(episodes)
    logger.info(f"Completed processing {len(successful_episodes)}/{len(episodes)} episodes")
    
    # Transcribe and process the clips
    logger.info(f"Starting transcription and processing of clips for {len(successful_episodes)} episodes")
    transcribe_and_process_clips_concurrently(successful_episodes)
    logger.info("Completed transcription and processing of clips")
