import torch
import torch.nn as nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # Load VGG19 and freeze it
        vgg = vgg19(weights='DEFAULT')
        # Use features up to the 36th layer (conv5_4)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:36]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Normalization constants for VGG
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr_img, hr_img):
        # Normalize inputs
        sr_norm = (sr_img - self.mean) / self.std
        hr_norm = (hr_img - self.mean) / self.std
        
        # Get features
        sr_features = self.feature_extractor(sr_norm)
        hr_features = self.feature_extractor(hr_norm)
        
        # Calculate loss
        return nn.functional.l1_loss(sr_features, hr_features)

class ColorConsistencyLoss(nn.Module):
    def __init__(self):
        super(ColorConsistencyLoss, self).__init__()
        self.l1 = nn.L1Loss()

    def forward(self, x, y):
        # Calculate average color of the whole image (R, G, B)
        # We pool the height and width down to 1x1
        mean_x = torch.mean(x, dim=[2, 3])
        mean_y = torch.mean(y, dim=[2, 3])
        return self.l1(mean_x, mean_y)

class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        self.l1_loss = nn.L1Loss() # Sharper than MSE
        self.vgg_loss = PerceptualLoss() # Keep your existing VGG class here!
        self.color_loss = ColorConsistencyLoss()

    def forward(self, sr, hr):
        # 1. Pixel Accuracy (Main Driver)
        loss_pixel = self.l1_loss(sr, hr)
        
        # 2. Perceptual Quality (Texture)
        loss_vgg = self.vgg_loss(sr, hr)
        
        # 3. Color Check (Prevents Orange Tint)
        loss_color = self.color_loss(sr, hr)
        
        # Weights: Pixel is king. VGG adds texture. Color ensures safety.
        return loss_pixel + (0.01 * loss_vgg) + (0.5 * loss_color)