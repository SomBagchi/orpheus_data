#!/bin/bash

CLIPS_DIR="orpheus_voice_cloning/final_clips"

# Check if the directory exists
if [ ! -d "$CLIPS_DIR" ]; then
    echo "Error: Directory $CLIPS_DIR not found."
    exit 1
fi

# Find folders with exactly 1 mp3 file
echo "Finding folders with only one mp3 file..."
single_mp3_folders=()

while IFS= read -r dir; do
    count=$(find "$dir" -maxdepth 1 -name "*.mp3" | wc -l)
    if [ "$count" -eq 1 ] && [ "$dir" != "$CLIPS_DIR" ]; then
        single_mp3_folders+=("$dir")
    fi
done < <(find "$CLIPS_DIR" -type d)

# Display results
total=${#single_mp3_folders[@]}
echo "Found $total folders with only one mp3 file."

# Preview the folders to be deleted
if [ $total -gt 0 ]; then
    echo "Folders that will be deleted:"
    for folder in "${single_mp3_folders[@]}"; do
        echo "  - $folder"
    done
    
    # Ask for confirmation
    read -p "Do you want to delete these folders? (y/n): " confirm
    
    if [[ "$confirm" == "y" || "$confirm" == "Y" ]]; then
        # Delete the folders
        echo "Deleting folders..."
        deleted=0
        for folder in "${single_mp3_folders[@]}"; do
            rm -rf "$folder"
            if [ $? -eq 0 ]; then
                deleted=$((deleted + 1))
                echo "Deleted: $folder"
            else
                echo "Failed to delete: $folder"
            fi
        done
        echo "Completed. Deleted $deleted out of $total folders."
    else
        echo "Operation cancelled. No folders were deleted."
    fi
else
    echo "No folders to delete."
fi 
