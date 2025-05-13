#!/usr/bin/env python3
"""
diarisation.py

Usage:
    python diarisation.py <input_mp3> <output_json>

Example:
    python diarisation.py episode.mp3 diarization.json

This script:
  1. Loads your PYANNOTEAI_API_TOKEN from .env
  2. Uploads the local MP3 to pyannoteAI temporary storage
  3. Submits a diarization job without webhooks
  4. Polls until the job completes
  5. Writes the diarization JSON to the specified output file
"""

import os
import sys
import time
import json
import requests
from dotenv import load_dotenv

API_BASE = "https://api.pyannote.ai/v1"

def diarize_file(input_path: str, output_path: str):
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
    print(f"Job {job_id} submitted, status={status}")

    # 4. Poll for completion
    while status not in ("succeeded", "failed", "canceled"):
        time.sleep(5)
        status_resp = requests.get(
            f"{API_BASE}/jobs/{job_id}",
            headers={"Authorization": f"Bearer {api_token}"}
        )
        status_resp.raise_for_status()
        status = status_resp.json()["status"]
        print(f"Polling job {job_id}, status={status}")

    if status != "succeeded":
        print(f"Job ended with status: {status}")
        sys.exit(1)

    # 5. Fetch and write diarization output
    output = status_resp.json().get("output", {})
    with open(output_path, "w") as fo:
        json.dump(output, fo, indent=2)
    print(f"Diarization results saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    diarize_file(sys.argv[1], sys.argv[2])

