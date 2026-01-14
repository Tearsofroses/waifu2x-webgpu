import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# Add src to path so we can import the model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import SRResNet

class Upscaler:
    def __init__(self, model_path, tile_size=512, overlap=32, device="cuda"):
        self.device = device
        self.scale = 2  # The Robust model is natively 2x
        self.tile_size = tile_size
        self.overlap = overlap
        
        print(f"Loading Universal Model: {model_path}")
        # Initialize standard 2x architecture
        self.model = SRResNet(scale_factor=2, num_res_blocks=32).to(device)
        
        # Load weights (strict=False to be safe, though usually fine)
        self.model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        self.model.eval()
        
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    async def process_batch(self, image, callback=None, start_progress=0, end_progress=100):
        """
        Internal function to run the model once (2x scaling).
        Handles tiling and batching.
        """
        scale = self.scale
        overlap = self.overlap
        tile_size = self.tile_size
        batch_size = 4 # Adjust based on VRAM

        # 1. Pad Image
        import torch.nn.functional as F
        img_tensor = self.to_tensor(image).unsqueeze(0)
        padded_tensor = F.pad(img_tensor, (overlap, overlap, overlap, overlap), mode='reflect')
        padded_pil = self.to_pil(padded_tensor.squeeze(0))
        pad_w, pad_h = padded_pil.size
        
        # 2. Prepare Output Canvas
        out_w = image.width * scale
        out_h = image.height * scale
        final_image = Image.new("RGB", (out_w, out_h))
        
        input_core = tile_size - (overlap * 2)
        output_core = input_core * scale
        crop_margin = overlap * scale

        # 3. Create Tiles
        tiles = []
        coords = []
        for y in range(0, image.height, input_core):
            for x in range(0, image.width, input_core):
                box = (x, y, min(x + tile_size, pad_w), min(y + tile_size, pad_h))
                tile = padded_pil.crop(box)
                
                # Handle edge tiles
                if tile.size != (tile_size, tile_size):
                    new_tile = Image.new("RGB", (tile_size, tile_size))
                    new_tile.paste(tile, (0, 0))
                    tile = new_tile
                
                tiles.append(self.to_tensor(tile))
                coords.append((x, y))

        # 4. Batch Inference
        total_batches = len(range(0, len(tiles), batch_size))
        
        for i, batch_start in enumerate(range(0, len(tiles), batch_size)):
            batch_end = batch_start + batch_size
            batch_tensors = torch.stack(tiles[batch_start:batch_end]).to(self.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    output_batch = self.model(batch_tensors)
            
            # Stitching
            for j, output_tensor in enumerate(output_batch):
                out_tile = self.to_pil(output_tensor.cpu().clamp(0, 1))
                
                # Crop center
                crop_box = (crop_margin, crop_margin, crop_margin + output_core, crop_margin + output_core)
                clean_tile = out_tile.crop(crop_box)
                
                # Paste
                global_idx = batch_start + j
                if global_idx < len(coords):
                    x, y = coords[global_idx]
                    paste_w = min(clean_tile.width, out_w - (x * scale))
                    paste_h = min(clean_tile.height, out_h - (y * scale))
                    if paste_w < clean_tile.width or paste_h < clean_tile.height:
                        clean_tile = clean_tile.crop((0, 0, paste_w, paste_h))
                    final_image.paste(clean_tile, (x * scale, y * scale))
            
            # Progress Update (Mapped to specific range)
            if callback:
                batch_percent = (i + 1) / total_batches
                # Map 0.0-1.0 to start_progress-end_progress
                total_percent = start_progress + (batch_percent * (end_progress - start_progress))
                await callback(int(total_percent))

        return final_image

    async def upscale_smart(self, image, target_scale=2, callback=None):
        """
        Public function that handles the logic:
        - If 2x: Run once.
        - If 4x: Run twice (Double Jump).
        """
        if target_scale == 2:
            print("--- Mode: 2x (Single Pass) ---")
            return await self.process_batch(image, callback, 0, 100)
        
        elif target_scale == 4:
            print("--- Mode: 4x (Double Pass) ---")
            # Pass 1: 0% to 50%
            first_pass = await self.process_batch(image, callback, 0, 50)
            
            # Pass 2: 50% to 100%
            final_pass = await self.process_batch(first_pass, callback, 50, 100)
            return final_pass
        
        else:
            raise ValueError("Only 2x and 4x supported")