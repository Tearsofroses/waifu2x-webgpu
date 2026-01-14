import torch
import torch.onnx
import os
import onnx  # Required for the single-file fix
from src.rrdbnet import RRDBNet
from dotenv import load_dotenv

load_dotenv()

# Input: The .pth file you want to convert
INPUT_MODEL_PATH = os.getenv("EXPORT_MODEL_PATH", "./checkpoints_rrdbnet/model_epoch_30.pth")

# Output: Where the web app lives
OUTPUT_DIR = os.path.join("web_dist", "models")
OUTPUT_FILENAME = "scale2x_denoise.onnx" # Or 'scale2x_clean.onnx'
DEVICE = "cpu"

# Define the models you want to build
# Format: { "checkpoint_path": "PATH_TO_PTH", "output_name": "FILENAME.onnx" }
EXPORT_JOBS = [
    {
        "checkpoint_path": "checkpoints_gan_2/gan_epoch_30.pth",  # Your Standard GAN
        "output_name": "scale2x_clean.onnx"
    },
    {
        "checkpoint_path": "checkpoints_robust_gan/gan_epoch_20.pth",  # Your Robust GAN
        "output_name": "scale2x_denoise.onnx"
    }
]

def export_single_model(job):
    pth_path = job["checkpoint_path"]
    onnx_name = job["output_name"]
    full_output_path = os.path.join(OUTPUT_DIR, onnx_name)

    print(f"\n--- Processing: {onnx_name} ---")
    
    # 0. Check if input exists
    if not os.path.exists(pth_path):
        print(f"!! SKIPPING: Could not find checkpoint at {pth_path}")
        print("   Please check the path in EXPORT_JOBS list.")
        return

    # 1. Initialize Model
    print(f"Loading weights from {pth_path}...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=6).to(DEVICE)
    
    # 2. Load Weights
    state_dict = torch.load(pth_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 3. Create Dummy Input
    # Shape: (Batch=1, Channels=3, Height=256, Width=256)
    dummy_input = torch.randn(1, 3, 256, 256, device=DEVICE)

    # 4. Export to ONNX (Initial Pass)
    print(f"Exporting raw ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        full_output_path,
        export_params=True,
        opset_version=18, 
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    # 5. FORCE SINGLE FILE (The Fix)
    print("Optimizing and merging into a single file...")
    onnx_model = onnx.load(full_output_path)
    onnx.save_model(onnx_model, full_output_path, save_as_external_data=False)
    
    # 6. Verification
    size_mb = os.path.getsize(full_output_path) / (1024 * 1024)
    print(f"Success! Saved {onnx_name} ({size_mb:.2f} MB)")

def main():
    # Setup Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # Run Export Jobs
    for job in EXPORT_JOBS:
        export_single_model(job)

    print("\nAll jobs finished. Check 'web_dist/models' folder.")

if __name__ == "__main__":
    main()