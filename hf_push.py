from huggingface_hub import HfApi, Repository
import os
import shutil

# 1) Create the dataset repo (private=True makes it private)
api = HfApi()
# api.create_repo(
#     repo_id="SomBagchi/orpheus_voice_cloning",
#     repo_type="dataset",
#     private=True
# )

# 2) Clone it locally or pull if it already exists
repo_path = "orpheus_voice_cloning"
if os.path.exists(repo_path):
    repo = Repository(local_dir=repo_path)
    print(f"Repository already exists. Pulling latest changes...")
    repo.git_pull()
else:
    repo = Repository(
        local_dir=repo_path,
        clone_from="SomBagchi/orpheus_voice_cloning",
        repo_type="dataset"
    )

# 3) Copy your CSV and audio folder in
print("Copying files...")
shutil.copy("index.csv", f"{repo_path}/data.csv")
if os.path.exists(f"{repo_path}/final_clips"):
    print("Removing existing final_clips directory...")
    shutil.rmtree(f"{repo_path}/final_clips")
shutil.copytree("final_clips", f"{repo_path}/final_clips")

# 4) Track MP3s with LFS, commit and push
print("Adding, committing, and pushing changes...")
repo.git_add(auto_lfs_track=True)
repo.git_commit("Update CSV + final_clips")
repo.push_to_hub()
print("Done!")
