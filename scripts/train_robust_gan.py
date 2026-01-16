import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src.model import SRResNet
from src.discriminator import Discriminator
from src.dataset import AnimeDataset
from src.loss import PerceptualLoss
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
BATCH_SIZE = 4            
LR = 1e-5
EPOCHS = 20          
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH = os.getenv("TRAIN_DATA_PATH", "./data/high_res_anime")  # Path to high-res images

# --- RESUME SETTINGS ---
RESUME_EPOCH = 0

# The path to your BEST "Blurry" model (Epoch 29)
PRETRAINED_BASE_MODEL = "checkpoints/model_epoch_29.pth" 

def train_gan():
    print(f"--- Starting GAN Fine-Tuning (2:1 Generator Strategy) on {DEVICE} ---")
    
    # 1. Setup Data
    train_dataset = AnimeDataset(high_res_dir=DATA_PATH, crop_size=96, scale_factor=2, noise_level=1, limit=15000)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True)

    # 2. Setup Models
    generator = SRResNet(scale_factor=2, num_res_blocks=32).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    
    # 3. Resume / Load Logic
    start_epoch = 0
    
    if RESUME_EPOCH > 0:
        # --- SCENARIO A: Resume interrupted GAN training ---
        gen_path = f"checkpoints_robust_gan/gan_epoch_{RESUME_EPOCH}.pth"
        if os.path.exists(gen_path):
            print(f"Resuming GAN training from: {gen_path}")
            generator.load_state_dict(torch.load(gen_path))
            start_epoch = RESUME_EPOCH
        else:
            print(f"ERROR: Could not find checkpoint {gen_path} to resume!")
            return
    else:
        # --- SCENARIO B: Start Fresh from Base Model ---
        if os.path.exists(PRETRAINED_BASE_MODEL):
            print(f"Loading base model to start GAN training: {PRETRAINED_BASE_MODEL}")
            generator.load_state_dict(torch.load(PRETRAINED_BASE_MODEL))
        else:
            print("WARNING: Base model not found! Training from scratch (Results will be bad).")

    # 4. Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=LR)
    opt_disc = optim.Adam(discriminator.parameters(), lr=LR)
    
    # 5. Losses
    criterion_content = PerceptualLoss().to(DEVICE) 
    criterion_adversarial = nn.BCEWithLogitsLoss().to(DEVICE) 

    # New folder for this "Version 2" run
    os.makedirs("checkpoints_robust_gan", exist_ok=True)
    log_file = "gan_training_log_robust.csv"

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, EPOCHS):
        generator.train()
        discriminator.train()
        loop = tqdm(train_loader, leave=True)
        
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        
        for idx, (lr, hr) in enumerate(loop):
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            
            # ---------------------
            #  1. Train Discriminator (ONCE)
            # ---------------------
            fake = generator(lr)
            
            disc_real = discriminator(hr)
            disc_fake = discriminator(fake.detach()) 
            
            loss_d_real = criterion_adversarial(disc_real - torch.mean(disc_fake), torch.ones_like(disc_real))
            loss_d_fake = criterion_adversarial(disc_fake - torch.mean(disc_real), torch.zeros_like(disc_fake))
            loss_disc = (loss_d_real + loss_d_fake) / 2
            
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # -----------------
            #  2. Train Generator (TWICE)
            # -----------------
            # We loop 2 times to let Generator learn faster than Discriminator
            for _ in range(2):
                # IMPORTANT: We MUST re-generate the fake images here.
                # Why? Because G's weights changed in the previous loop iteration.
                fake = generator(lr) 
                
                disc_fake_pred = discriminator(fake)
                disc_real_pred = discriminator(hr).detach()

                loss_gen_real = criterion_adversarial(disc_real_pred - torch.mean(disc_fake_pred), torch.zeros_like(disc_real_pred))
                loss_gen_fake = criterion_adversarial(disc_fake_pred - torch.mean(disc_real_pred), torch.ones_like(disc_fake_pred))
                loss_gen_adversarial = (loss_gen_real + loss_gen_fake) / 2
                
                loss_gen_content = criterion_content(fake, hr)
                
                loss_gen = (5e-3 * loss_gen_adversarial) + loss_gen_content
                
                opt_gen.zero_grad()
                loss_gen.backward()
                opt_gen.step()
            
            # Update metrics (We just log the loss from the last G iteration)
            epoch_g_loss += loss_gen.item()
            epoch_d_loss += loss_disc.item()
            
            loop.set_description(f"Epoch {epoch+1}/{EPOCHS}")
            loop.set_postfix(G_loss=loss_gen.item(), D_loss=loss_disc.item())

        # --- LOGGING & SAVING ---
        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Avg G Loss: {avg_g_loss:.6f} | Avg D Loss: {avg_d_loss:.6f}")

        # Write to CSV
        write_header = not os.path.exists(log_file)
        with open(log_file, "a") as f:
            if write_header:
                f.write("epoch,g_loss,d_loss\n")
            f.write(f"{epoch+1},{avg_g_loss:.6f},{avg_d_loss:.6f}\n")

        torch.save(generator.state_dict(), f"checkpoints_robust_gan/gan_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    train_gan()