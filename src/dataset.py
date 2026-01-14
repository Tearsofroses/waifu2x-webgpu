import os
import random
import io
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class AnimeDataset(Dataset):
    def __init__(self, high_res_dir, crop_size=96, scale_factor=2, noise_level=0, limit = None):
        """
        noise_level: 
          0 = Clean (Standard Upscaling)
          1 = Denoise Mode (Adds random JPEG noise to inputs)
        """
        # --- Recursive Search (Kept from your code) ---
        self.files = []
        for root, _, filenames in os.walk(high_res_dir):
            for filename in filenames:
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.files.append(os.path.join(root, filename))
        
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {high_res_dir}. Did you run extract_data.py?")
            
        print(f"Dataset Loaded: {len(self.files)} images found. (Noise Level: {noise_level})")
        # -----------------------------

        random.shuffle(self.files)
        if limit is not None and len(self.files) > limit:
            print(f"Limiting dataset to {limit} images.")
            self.files = self.files[:limit]

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.noise_level = noise_level
        
        # Pre-calculate low-res size
        self.lr_size = (crop_size // scale_factor, crop_size // scale_factor)
        
        # We apply RandomCrop/Flip to the High Res image FIRST
        self.hr_transform = transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
        ])
    
    def __getitem__(self, index):
        # 1. Load High Res Image
        try:
            img = Image.open(self.files[index]).convert('RGB')
        except Exception as e:
            # Fallback if an image is corrupt
            return self.__getitem__(random.randint(0, len(self.files) - 1))
        
        # 2. Create Ground Truth (High Res)
        # Apply the crop/flip here so HR and LR match
        hr_pil = self.hr_transform(img)
        
        # 3. Create Input (Low Res)
        # Downscale (Bicubic) - This creates the blurry input
        lr_image = hr_pil.resize(self.lr_size, Image.BICUBIC)
        
        # --- NOISE INJECTION (The Waifu2x Trick) ---
        if self.noise_level > 0:
            # 80% chance to add noise (keep 20% clean so it learns both)
            if random.random() < 0.8:
                buffer = io.BytesIO()
                
                # Random quality: 
                # 30 = Terrible (Heavy noise)
                # 85 = Okay (Slight noise)
                quality = random.randint(30, 85) 
                
                lr_image.save(buffer, format='JPEG', quality=quality)
                buffer.seek(0)
                lr_image = Image.open(buffer)
        # -------------------------------------------
        
        # 4. Convert both to Tensor
        hr_tensor = transforms.ToTensor()(hr_pil)
        lr_tensor = transforms.ToTensor()(lr_image)
        
        return lr_tensor, hr_tensor

    def __len__(self):
        return len(self.files)