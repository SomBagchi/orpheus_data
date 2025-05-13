#!/usr/bin/env python3
"""
visualize_diarisation.py

Usage:
    python visualize_diarisation.py example.json

Creates a visual timeline of speaker segments from diarisation output.
"""

import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def visualize_diarisation(json_path):
    # Load diarisation data
    with open(json_path, 'r') as f:
        data_wrapper = json.load(f)
    
    # Extract the diarization data from the wrapper
    data = data_wrapper.get("diarization", [])
    
    if not data:
        print("No diarisation data found in the file.")
        return
    
    # Extract all speakers and find time range
    speakers = sorted(set(item["speaker"] for item in data))
    end_time = max(item["end"] for item in data)
    
    # Assign colors to speakers
    colors = plt.cm.get_cmap('tab10', len(speakers))
    speaker_colors = {speaker: colors(i) for i, speaker in enumerate(speakers)}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot speaker segments
    for i, speaker in enumerate(speakers):
        speaker_segments = [item for item in data if item["speaker"] == speaker]
        y_position = len(speakers) - i
        
        for segment in speaker_segments:
            start = segment["start"]
            duration = segment["end"] - segment["start"]
            rect = patches.Rectangle(
                (start, y_position - 0.4), 
                duration, 
                0.8, 
                linewidth=1, 
                edgecolor='black', 
                facecolor=speaker_colors[speaker],
                alpha=0.7
            )
            ax.add_patch(rect)
    
    # Set plot limits and labels
    ax.set_xlim(0, end_time)
    ax.set_ylim(0.5, len(speakers) + 0.5)
    ax.set_yticks(range(1, len(speakers) + 1))
    ax.set_yticklabels(speakers[::-1])
    
    # Format x-axis in seconds
    ax.set_xlabel('Time (seconds)')
    
    # Create ticks for every second
    second_ticks = np.arange(0, int(end_time) + 1, 1)
    ax.set_xticks(second_ticks)
    ax.set_xticklabels([str(int(t)) for t in second_ticks])
    
    # Add grid lines for each second
    ax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add title and labels
    input_name = Path(json_path).stem
    ax.set_title(f'Speaker Diarisation: {input_name}')
    ax.set_ylabel('Speakers')
    
    # Generate output filename
    output_path = Path(json_path).with_suffix('.png')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    
    # Show interactive plot
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    json_path = sys.argv[1]
    visualize_diarisation(json_path)
