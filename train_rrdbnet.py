import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time  # <--- Added for temperature control
from torch.cuda.amp import GradScaler, autocast # Import the magic tools
from dotenv import load_dotenv

# Import from our src folder
from src.model import SRResNet
from src.dataset import AnimeDataset
from src.loss import TotalLoss # Using the new loss we made
from src.rrdbnet import RRDBNet
from src.degradations import degradation_pipeline

# --- CONFIGURATION ---
DATA_PATH = os.getenv("DATASET_PATH", "./data/anime_hq/images")
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "./checkpoints_rrdbnet")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
TOTAL_EPOCHS = int(os.getenv("EPOCHS", 30))
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-5
RESUME_EPOCH = 0
COOL_DOWN = 0.01


def train():
    print(f"--- Starting Training on {DEVICE} ---")
    
    # 1. Setup Data
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Cannot find data at {DATA_PATH}")
        return

    # num_workers=0 is standard for Windows
    train_dataset = AnimeDataset(high_res_dir=DATA_PATH, crop_size=96, scale_factor=2)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # 2. Setup Model
    # Note: We are using the NEW architecture (32 blocks, No Batch Norm)
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=6).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()  # Initialize GradScaler for mixed precision
    criterion = TotalLoss().to(DEVICE)

    # 3. Resume Logic
    start_epoch = 0
    if RESUME_EPOCH > 0:
        checkpoint_path = f"checkpoins_rrdbnet/model_epoch_{RESUME_EPOCH}.pth"
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            # Load weights
            model.load_state_dict(torch.load(checkpoint_path))
            start_epoch = RESUME_EPOCH
            print(f"Resuming successfully from Epoch {start_epoch + 1}")
        else:
            print(f"WARNING: Checkpoint {checkpoint_path} not found! Starting from scratch.")

    # 4. Create Checkpoint Folder
    os.makedirs("checkpoins_rrdbnet", exist_ok=True)

    # --- TRAINING LOOP ---
    # We range from start_epoch to TOTAL_EPOCHS
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        loop = tqdm(train_loader, leave=True)
        epoch_loss = 0
        
        for batch_idx, (_, hr_imgs) in enumerate(loop):
            hr_imgs = hr_imgs.to(DEVICE)
            
            with torch.no_grad():
                lr_imgs = degradation_pipeline(hr_imgs)

            with autocast():  # Enable autocasting for the forward pass
                fake_hr = model(lr_imgs)
                loss = criterion(fake_hr, hr_imgs)

            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Scale the loss
            scaler.step(optimizer)          # Step the optimizer
            scaler.update()                 # Update the scaler

            epoch_loss += loss.item()
            
            # --- TEMPERATURE CONTROL ---
            # Sleep for a tiny bit to let GPU fans catch up
            if COOL_DOWN > 0:
                time.sleep(COOL_DOWN)
            
            # Update Progress Bar
            loop.set_description(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        # Save Checkpoint

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}] Average Loss: {avg_loss:.6f}")

        with open("training_log_rrdbnet.csv", "a") as f:
            if epoch == start_epoch:
                f.write("epoch,average_loss\n")  # Write header if starting fresh
            f.write(f"{epoch+1},{avg_loss:.6f}\n")

        save_path = f"checkpoins_rrdbnet/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    train()