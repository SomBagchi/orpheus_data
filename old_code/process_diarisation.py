#!/usr/bin/env python3
"""
process_diarisation.py

Usage:
    python process_diarisation.py <input_json_file> [max_pause_length]

Processes a diarisation JSON file by:
1. Fusing consecutive blocks from the same speaker if gap < MAX_PAUSE_LENGTH
2. Keeping only blocks from the speaker who spoke the most
3. Writing result to a new file with prefix "processed_"

If max_pause_length is not provided, defaults to 1.0 second
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

# Default maximum pause length for merging blocks (in seconds)
DEFAULT_MAX_PAUSE_LENGTH = 1.0

def process_diarisation(input_file, max_pause_length=DEFAULT_MAX_PAUSE_LENGTH):
    """Process diarisation data according to the rules."""
    # Load input file
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
              block["start"] - current_block["end"] <= max_pause_length):
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
    print("Speaker statistics after fusion:")
    for speaker, time in speaking_times.items():
        print(f"{speaker}: {time:.2f} seconds")
    print(f"\nKeeping only blocks from {most_speaking_speaker} (spoke the most)")
    
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
    input_path = Path(input_file)
    output_file = input_path.parent / f"processed_{input_path.name}"
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nProcessed diarisation written to {output_file}")
    print(f"Original block count: {len(diarization)}")
    print(f"After fusion: {len(fused_blocks)}")
    print(f"After keeping most speaking speaker: {block_count_before_duration_filter}")
    print(f"Final block count (after removing blocks < {min_block_duration}s): {len(filtered_blocks)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    max_pause_length = DEFAULT_MAX_PAUSE_LENGTH
    if len(sys.argv) > 2:
        try:
            max_pause_length = float(sys.argv[2])
        except ValueError:
            print(f"Invalid max_pause_length: {sys.argv[2]}. Using default: {DEFAULT_MAX_PAUSE_LENGTH}")
    
    process_diarisation(input_file, max_pause_length)
