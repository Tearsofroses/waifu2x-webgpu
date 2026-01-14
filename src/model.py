import torch
import torch.nn as nn
import math

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Squeeze-and-Excitation block
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # EDSR Style: No Batch Norm!
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.attention = ChannelAttention(channels) 
        # Scale residual to stabilize training (0.1 is standard in EDSR)
        self.res_scale = 0.1

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.attention(residual) # Apply Attention
        return x + (residual * self.res_scale)

class SRResNet(nn.Module):
    def __init__(self, scale_factor=2, num_res_blocks=32): # Increased blocks to 32 for depth
        super(SRResNet, self).__init__()
        
        self.conv_input = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        
        # Deep Residual Chain
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_res_blocks)])
        
        self.conv_mid = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        
        # Upsampling
        upsample_layers = []
        for _ in range(int(math.log(scale_factor, 2))):
            upsample_layers += [
                nn.Conv2d(64, 256, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU()
            ]
        self.upsample = nn.Sequential(*upsample_layers)
        
        self.conv_output = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv_input(x)
        residual = out
        
        out = self.res_blocks(out)
        out = self.conv_mid(out)
        out = out + residual 
        
        out = self.upsample(out)
        out = self.conv_output(out)
        return out