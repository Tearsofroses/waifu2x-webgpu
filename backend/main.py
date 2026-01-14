from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import io
import json
from PIL import Image
import os
from engine import Upscaler

app = FastAPI()

# --- LOAD THE CHAMPION MODEL ---
# Point this to your Robust GAN file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "checkpoints_gan_robust", "gan_epoch_20.pth")

# Initialize Engine
engine = Upscaler(model_path=MODEL_PATH, device="cuda")

@app.get("/")
def health_check():
    return {"status": "Online", "model": "Robust GAN Universal"}

@app.websocket("/ws/upscale")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        # 1. Receive Config
        config_data = await websocket.receive_text()
        config = json.loads(config_data)
        scale = int(config.get("scale", 2)) # User wants 2x or 4x
        
        # 2. Receive Image
        image_data = await websocket.receive_bytes()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        print(f"Job Received: {image.size} -> {scale}x")
        
        # 3. Define Callback
        async def progress_callback(percent):
            await websocket.send_text(json.dumps({
                "status": "processing", 
                "percent": percent
            }))

        # 4. Run the Engine (Smart Mode)
        result_image = await engine.upscale_smart(image, target_scale=scale, callback=progress_callback)
        
        # 5. Send Result
        output_buffer = io.BytesIO()
        result_image.save(output_buffer, format="PNG")
        
        await websocket.send_text(json.dumps({"status": "complete"}))
        await websocket.send_bytes(output_buffer.getvalue())
        print("Job Complete.")

    except Exception as e:
        print(f"Error: {e}")
        await websocket.send_text(json.dumps({"status": "error", "message": str(e)}))