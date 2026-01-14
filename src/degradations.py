import torch
import torch.nn.functional as F
import random
import numpy as np
from torchvision import transforms

def random_blur(img_tensor):
    """Apply random gaussian blur"""
    if random.random() > 0.5:
        return img_tensor # 50% chance to skip
        
    kernel_size = random.choice([3, 5, 7])
    sigma = random.uniform(0.5, 2.0)
    
    # Create simple gaussian kernel
    x = torch.arange(kernel_size).float() - (kernel_size - 1) / 2
    x = x ** 2
    kernel = torch.exp(-x / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    
    # Make 2D kernel
    kernel_2d = kernel.unsqueeze(1) * kernel.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(3, 1, 1, 1).to(img_tensor.device) # 3 channels
    
    # Apply depthwise convolution (blur)
    pad = (kernel_size - 1) // 2
    return F.conv2d(img_tensor, kernel_2d, padding=pad, groups=3)

def add_noise(img_tensor):
    """Add random Gaussian noise"""
    if random.random() > 0.5:
        return img_tensor
        
    noise_level = random.uniform(0.01, 0.05) # 1% to 5% noise
    noise = torch.randn_like(img_tensor) * noise_level
    return torch.clamp(img_tensor + noise, 0, 1)

def degradation_pipeline(hr_tensor):
    """
    Takes a High-Res (HR) Image tensor (0-1).
    Returns a Low-Res (LR) Image tensor (0-1) that is:
    1. Blurred
    2. Noised
    3. Downscaled (0.5x)
    """
    # 1. Blur
    out = random_blur(hr_tensor)
    
    # 2. Downscale (This creates the pixelation)
    out = F.interpolate(out, scale_factor=0.5, mode='bicubic')
    
    # 3. Noise
    out = add_noise(out)
    
    return out