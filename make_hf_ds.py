from datasets import Dataset, Audio
import os
import pandas as pd
import librosa
import soundfile as sf
import time

# Start measuring total execution time
total_start_time = time.time()

# Path to the CSV file
csv_file = "orpheus_voice_cloning/data.csv"

# Load the CSV file
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} rows from {csv_file}")

all_rows = []
dataset_name = "orpheus_voice_cloning_dataset"

# Process each row in the CSV file
for i, row in df.iterrows():
    episode_id = row['episode_id']
    uuid = row['uuid']
    audio_file_name = f"orpheus_voice_cloning/final_clips/{episode_id}/{uuid}.mp3"
    
    # Check if audio file exists
    if not os.path.exists(audio_file_name):
        print(f"Audio file {audio_file_name} does not exist. Skipping...")
        continue
    
    try:
        # Load audio using librosa
        array, sr = librosa.load(audio_file_name, sr=None)
        
        # Create dataset row with all CSV columns plus audio
        dataset_row = {
            "audio": {
                "array": array,
                "sampling_rate": sr
            }
        }
        
        # Add all columns from the CSV
        for column in df.columns:
            dataset_row[column] = row[column]
        
        all_rows.append(dataset_row)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} files...")
            
        # # Break after processing 30 rows
        # if len(all_rows) >= 30:
        #     print(f"Processed first 30 files, stopping...")
        #     break
            
    except Exception as e:
        print(f"Error processing file {audio_file_name}: {e}")

# Create the dataset
if all_rows:
    ds = Dataset.from_list(all_rows)
    print(f"Created dataset with {len(ds)} entries")
    
    # Cast audio column to Audio type
    start_time = time.time()
    ds = ds.cast_column("audio", Audio())
    cast_time = time.time() - start_time
    print(f"Casting audio column took {cast_time:.2f} seconds")
    
    # Push to Hugging Face
    start_time = time.time()
    ds.push_to_hub("SomBagchi/orpheus_voice_cloning", private=True)
    push_time = time.time() - start_time
    print(f"Pushing to Hugging Face took {push_time:.2f} seconds")
    print("Dataset successfully pushed to Hugging Face!")
else:
    print("No valid data found to create dataset.")

# Print total execution time
total_time = time.time() - total_start_time
print(f"Total script execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
