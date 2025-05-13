#!/usr/bin/env python3
"""
clipped_diarisation.py

Usage:
    python clipped_diarisation.py <diarisation_json> <audio_file.mp3> [output_dir]

Takes a processed diarisation JSON file and an MP3 file, and creates
multiple MP3 clips based on the diarisation segments.

If output_dir is not provided, clips will be saved in 'clips/' directory.
"""

import sys
import os
import json
from pathlib import Path
from pydub import AudioSegment

def clip_audio(diarisation_file, audio_file, output_dir="clipped_diarisation"):
    """
    Clip an audio file based on diarisation segments.
    
    Args:
        diarisation_file: Path to JSON diarisation file
        audio_file: Path to MP3 audio file
        output_dir: Directory to save the clips (created if it doesn't exist)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load diarisation data
    with open(diarisation_file, 'r') as f:
        data = json.load(f)
    
    segments = data.get("diarization", [])
    if not segments:
        print("No diarisation segments found")
        return
    
    # Load audio file
    print(f"Loading audio file: {audio_file}")
    audio = AudioSegment.from_file(audio_file)
    
    # Get base filename without extension
    audio_basename = Path(audio_file).stem
    
    # Process each segment
    print(f"Creating {len(segments)} clips...")
    
    for i, segment in enumerate(segments, 1):
        start_time = segment["start"] * 1000  # Convert to milliseconds
        end_time = segment["end"] * 1000
        speaker = segment["speaker"]
        
        # Extract segment
        clip = audio[start_time:end_time]
        
        # Create filename with segment information
        output_file = f"{output_dir}/{audio_basename}_{speaker}_seg{i:03d}_{start_time/1000:.2f}-{end_time/1000:.2f}.mp3"
        
        # Export clip
        clip.export(output_file, format="mp3")
        
        print(f"Created: {output_file} ({(end_time-start_time)/1000:.2f} seconds)")
    
    print(f"\nDone! Created {len(segments)} clips in {output_dir}/ directory")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    diarisation_file = sys.argv[1]
    audio_file = sys.argv[2]
    
    output_dir = "clips"
    if len(sys.argv) > 3:
        output_dir = sys.argv[3]
    
    clip_audio(diarisation_file, audio_file, output_dir)
