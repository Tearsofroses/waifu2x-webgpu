import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.cuda.amp import GradScaler, autocast
from dotenv import load_dotenv
from src.model import SRResNet
from src.dataset import AnimeDataset
from src.loss import TotalLoss
load_dotenv()

# --- CONFIG ---
# We use a smaller LR because the model is already smart. We just want to tweak it.
LR = 5e-5           
EPOCHS = 15          
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 8))
DEVICE = os.getenv("DEVICE", "cuda")

DATA_PATH = os.getenv("DATASET_PATH", "./data/anime_hq/images")
PRETRAINED_PATH = os.getenv("BASE_MODEL_PATH", "./checkpoints/model_epoch_30.pth")
CHECKPOINT_DIR = "./checkpoints_denoise"

def train_denoise():
    print(f"--- Starting Denoising Transfer Learning on {DEVICE} ---")
    
    # 1. Setup Data with NOISE enabled
    # noise_level=1 triggers the JPEG injection we wrote in dataset.py
    train_dataset = AnimeDataset(high_res_dir=DATA_PATH, scale_factor=2, noise_level=1, limit = 15000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)

    # 2. Setup Model
    model = SRResNet(scale_factor=2, num_res_blocks=32).to(DEVICE)
    
    # 3. Load Pretrained Weights
    if os.path.exists(PRETRAINED_PATH):
        print(f"Loading Base Model: {PRETRAINED_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_PATH))
    else:
        print("ERROR: Base model not found! Training from scratch (NOT RECOMMENDED).")
        return

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = TotalLoss().to(DEVICE)
    
    os.makedirs("checkpoints_denoise", exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, leave=True)
        epoch_loss = 0
        
        for batch_idx, (lr, hr) in enumerate(loop):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            
            with autocast():
                fake_hr = model(lr)
                loss = criterion(fake_hr, hr)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")
        
        torch.save(model.state_dict(), f"checkpoints_denoise/denoise_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_denoise()