from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv()

# Destination for the zip files
DESTINATION = os.getenv("RAW_DATA_DIR", "./data/anime_hq")

# Download "skytnt/fbanimehq"
path = snapshot_download(repo_id="skytnt/fbanimehq", 
                         repo_type="dataset", 
                         local_dir=DESTINATION,
                         local_dir_use_symlinks=False) # Important for Windows!

print("Done!")