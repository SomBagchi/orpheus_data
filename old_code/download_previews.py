#!/usr/bin/env python3
"""
download_previews.py - Download preview MP3s for Spotify episodes

This script reads episode IDs from episode_ids.txt and downloads 
the preview MP3s to a 'clips' folder.
"""

import os
import re
import requests
import sys
from tqdm import tqdm
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

# Create clips directory if it doesn't exist
CLIPS_DIR = "raw_clips"
os.makedirs(CLIPS_DIR, exist_ok=True)

load_dotenv()

def download_episode_previews():
    # Set up Spotify API connection
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri="https://google.com",
        scope=None,
        open_browser=True
    )
    sp = Spotify(auth_manager=auth_manager)
    
    # Read episode IDs from file
    episode_ids = []
    with open("episode_ids.txt", "r") as f:
        for line in f:
            if match := re.search(r'• (\w+)', line):
                episode_ids.append(match.group(1))
    
    print(f"Found {len(episode_ids)} episode IDs to process")
    
    # Download preview MP3s
    successful_downloads = 0
    for episode_id in tqdm(episode_ids):
        try:
            # Get episode metadata
            ep = sp.episode(episode_id)
            url = ep.get("audio_preview_url")
            if not url:
                print(f"No preview available for episode {episode_id}")
                continue
                
            # Sanitize filename - replace special characters
            title = ep["name"]
            safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            
            # Download preview MP3
            resp = requests.get(url)
            resp.raise_for_status()
            
            filename = f"{CLIPS_DIR}/{episode_id}_{safe_title}.mp3"
            with open(filename, 'wb') as f:
                f.write(resp.content)
                
            successful_downloads += 1
            
        except Exception as e:
            print(f"Error processing episode {episode_id}: {e}")
    
    print(f"Successfully downloaded {successful_downloads} preview MP3s to {CLIPS_DIR}/ folder")

if __name__ == "__main__":
    download_episode_previews()
