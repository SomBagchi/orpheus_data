import os
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
from tqdm import tqdm

NUM_EPISODES = 1000

load_dotenv()

# 1) Authenticate
creds = SpotifyClientCredentials(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIPY_CLIENT_SECRET")
)
sp = Spotify(client_credentials_manager=creds)

# 2) Search for the show
res = sp.search(
    q="The Joe Rogan Experience",
    type="show",
    limit=1
)
show = res["shows"]["items"][0]
show_id = show["id"]
print("Show ID:", show_id)

# 3) Get the NUM_EPISODES most recent episodes
episode_ids = []
limit = 50  # Spotify API typically allows max 50 per request
offset = 0

while True:
    page = sp.show_episodes(show_id, limit=limit, offset=offset, market="US")
    items = page["items"]
    if not items:
        break

    # Collect IDs
    episode_ids.extend(ep["id"] for ep in items)

    # Advance offset; if fewer than `limit` came back, we're done
    offset += len(items)
    if len(items) < limit:
        break

    # Stop if we've collected enough episodes
    if len(episode_ids) >= NUM_EPISODES:
        episode_ids = episode_ids[:NUM_EPISODES]
        break

# Write episode IDs to a file
with open("episode_ids.txt", "w") as f:
    f.write(f"{NUM_EPISODES} most recent episode IDs:\n")
    for eid in tqdm(episode_ids):
        f.write(f" • {eid}\n")

print(f"Episode IDs written to episode_ids.txt")
