#!/usr/bin/env python3
"""
play_preview.py

Usage:
    python play_preview.py <episode_id>

Example:
    python play_preview.py 7rbP33zQkM7TJJJJXXXXX
"""

import os
import sys
import requests
import subprocess
from spotipy import Spotify
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
from pydub import AudioSegment

# ─── Configuration ──────────────────────────────────────────────────────────────
# 1) Set these env vars (or hard-code the strings, though env vars are safer):
#    export SPOTIPY_CLIENT_ID="your-client-id"
#    export SPOTIPY_CLIENT_SECRET="your-client-secret"
# 2) In your Spotify Dashboard, add this exact URI under Redirect URIs:
#    http://localhost:8888/callback

REDIRECT_URI = "https://google.com"
SCOPE = None  # no special scopes needed for public preview URLs

load_dotenv()

def play_preview(episode_id: str):
    # set up OAuth flow
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        open_browser=True
    )
    sp = Spotify(auth_manager=auth_manager)

    # fetch episode metadata
    ep = sp.episode(episode_id)
    url = ep.get("audio_preview_url")
    if not url:
        print(f"No preview available for episode {episode_id}")
        return

    # download preview MP3
    resp = requests.get(url)
    resp.raise_for_status()
    tmpfile = "spotify_preview.mp3"
    with open(tmpfile, "wb") as f:
        f.write(resp.content)

    # play it using macOS afplay
    print(f'Playing 30 sec preview of "{ep["name"]}" …')
    try:
        # Get audio file info and print sample rate before playing
        song = AudioSegment.from_mp3(tmpfile)
        print(f"Sample Rate: {song.frame_rate} Hz")
        subprocess.run(["afplay", tmpfile], check=True)
    except subprocess.CalledProcessError:
        print("Error playing audio file")
    except FileNotFoundError:
        print("afplay command not found. Are you on macOS?")

    # clean up
    os.remove(tmpfile)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    play_preview(sys.argv[1])
