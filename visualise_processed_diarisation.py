#!/usr/bin/env python3
"""
visualise_processed_diarisation.py

Usage:
    python visualise_processed_diarisation.py example.json

Creates a visual timeline of speaker segments from processed diarisation output.
"""

import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path

def visualize_processed_diarisation(json_path):
    # Load processed diarisation data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extract the segments and speaker info
    segments = data.get("segments", [])
    maximal_speaker = data.get("maximal_speaker", "Unknown")
    episode_id = data.get("episode_id", "Unknown")
    episode_name = data.get("episode_name", "Unknown")
    
    if not segments:
        print("No segment data found in the file.")
        return
    
    # Find time range
    end_time = max(item["end"] for item in segments)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Assign colors to segment types
    type_colors = {
        "maximal": "tab:blue"
    }
    
    # Plot segments
    for segment in segments:
        start = segment["start"]
        duration = segment["end"] - segment["start"]
        segment_type = segment.get("type", "maximal")
        
        rect = patches.Rectangle(
            (start, 0.6), 
            duration, 
            0.8, 
            linewidth=1, 
            edgecolor='black', 
            facecolor=type_colors.get(segment_type, "tab:gray"),
            alpha=0.7
        )
        ax.add_patch(rect)
    
    # Set plot limits and labels
    ax.set_xlim(0, end_time)
    ax.set_ylim(0, 2)
    ax.set_yticks([1])
    ax.set_yticklabels([maximal_speaker])
    
    # Format x-axis in seconds
    ax.set_xlabel('Time (seconds)')
    
    # Create ticks for every second
    second_ticks = np.arange(0, int(end_time) + 1, 1)
    ax.set_xticks(second_ticks)
    ax.set_xticklabels([str(int(t)) for t in second_ticks])
    
    # Add grid lines for each second
    ax.grid(axis='x', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add title and labels
    ax.set_title(f'Processed Diarisation: {episode_id} - {episode_name}')
    ax.set_ylabel('Speaker')
    
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
    visualize_processed_diarisation(json_path)
