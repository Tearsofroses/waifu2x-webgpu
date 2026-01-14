import os
import zipfile
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Config
# Where the zip files are
SOURCE_DIR = os.getenv("RAW_DATA_DIR", "./data/anime_hq")
# Where to extract them
TARGET_DIR = os.getenv("DATASET_PATH", "./data/anime_hq/images")

def extract_and_clean():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 1. Find all zip files recursively
    zip_files = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))
    
    if not zip_files:
        print(f"No zip files found in {SOURCE_DIR}!")
        return

    print(f"Found {len(zip_files)} zip files. Extracting to {TARGET_DIR}...")
    print("WARNING: Zip files will be DELETED after extraction to save space.")
    
    # 2. Extract and Delete loop
    for zip_path in tqdm(zip_files):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(TARGET_DIR)
            
            # --- THE CHANGE: Delete the zip file after success ---
            os.remove(zip_path)
            
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is corrupt. Skipping (Not deleted).")
        except Exception as e:
            print(f"Failed to extract {zip_path}: {e}")

    print("Extraction & Cleanup Complete!")
    print(f"Images are ready in: {TARGET_DIR}")

if __name__ == "__main__":
    extract_and_clean()