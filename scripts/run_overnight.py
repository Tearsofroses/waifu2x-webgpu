import os
import subprocess
import time
import sys
from dotenv import load_dotenv

load_dotenv()
# --- CONFIG ---
SHUTDOWN_ON_FINISH = os.getenv("SHUTDOWN_ON_FINISH", "False").lower() == "true"

def run_script(script_name):
    print(f"\n\n{'='*40}")
    print(f"--- LAUNCHING {script_name} ---")
    print(f"{'='*40}\n")
    try:
        # sys.executable ensures we use the same python environment
        result = subprocess.run([sys.executable, script_name], check=True)
        print(f"--- SUCCESS: {script_name} Finished ---")
        return True
    except subprocess.CalledProcessError:
        print(f"!!! FAILURE: {script_name} Crashed !!!")
        return False

def shutdown_pc():
    print("All tasks complete. Shutting down in 60 seconds...")
    print("Press CTRL+C to cancel.")
    time.sleep(60) 
    if os.name == 'nt':
        os.system("shutdown /s /t 1")

if __name__ == "__main__":
    # Task 1: GAN Training (Sharpness)
    # Ensure train_gan.py is set to use PRETRAINED_BASE_MODEL = "checkpoints/model_epoch_29.pth"
    gan_success = run_script("train_gan.py")
    
    # Cool down period between massive jobs
    if gan_success:
        print("Cooling down GPU for 5 minutes...")
        time.sleep(300)

    # Task 2: Denoise Training (Cleaning)
    # Ensure train_denoise.py is set to use PRETRAINED_PATH = "checkpoints/model_epoch_29.pth"
    denoise_success = run_script("train_denoise.py")

    if denoise_success:
        print("Cooling down GPU for 5 minutes...")
        time.sleep(300)

    fourx_success = run_script("train_4x.py")

    if fourx_success:
        print("All training tasks completed successfully!")
        print("Cooling down GPU for 5 minutes...")
        time.sleep(300)

    # Shutdown Logic
    if SHUTDOWN_ON_FINISH:
        shutdown_pc()