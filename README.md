# Waifu2x-WebGPU: Serverless AI Image Upscaler

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Platform](https://img.shields.io/badge/platform-WebGPU%20|%20WASM-green)
![Model](https://img.shields.io/badge/model-Real--ESRGAN_%7C_SRResNet-red)

**Waifu2x-WebGPU** is a privacy-focused, in-browser image upscaler that runs state-of-the-art super-resolution models directly on the user's device.

Unlike traditional web upscalers that upload images to a cloud server (incurring costs and privacy risks), this project utilizes **ONNX Runtime** and **WebGPU** to execute the AI model entirely within the client's browser.

**[Live Demo](https://tearsofroses.github.io/waifu2x-webgpu/)**

---

## Key Features

* **Zero-Knowledge Privacy:** Images are processed locally in JavaScript memory. No data ever leaves the user's device.
* **WebGPU Acceleration:** Leveraging the latest browser graphics standards for near-native GPU inference speeds.
* **Custom SSResNet and RRDBNet Architecture:** Trained from scratch on anime datasets using a custom degradation pipeline (blur, noise, compression) to handle real-world artifacts.
* **Adaptive Tiling Engine:** Automatically benchmarks client hardware (Low/Mid/High tier) to adjust tile sizes dynamically, preventing crashes on mobile devices or low-VRAM GPUs.

---

## Tech Stack

* **Training:** PyTorch, (SSRNet), Real-ESRGAN (RRDBNet), Perceptual Loss (VGG19), GANs.
* **Inference:** ONNX Runtime Web (WASM + WebGPU).
* **Frontend:** Vanilla JavaScript (ES6+), HTML5, CSS3.
* **DevOps:** GitHub Actions, GitHub Pages (CI/CD).

---

## Quick Start (Local Development)

### 1. Prerequisites
* Python 3.8+
* NVIDIA GPU (Recommended for training)
* Node.js (Optional, for local web server)

### 2. Installation
Clone the repository and install Python dependencies:
```bash
git clone [https://github.com/YourUsername/Waifu2x-WebGPU.git](https://github.com/YourUsername/Waifu2x-WebGPU.git)
cd Waifu2x-WebGPU
pip install -r requirements.txt
```

### 3. Setup Configuration
Create a .env file in the root directory to manage paths:
```bash
DATASET_PATH=./data/anime_hq/images
CHECKPOINT_DIR=./checkpoints
BATCH_SIZE=8
EPOCHS=30
DEVICE=cuda
```

### 4. Running the Web App
To test the interface locally (browsers block WebGPU on file:// URLs, so a server is required):

```bash
cd web_dist
python -m http.server 8080
# Open http://localhost:8080 in Chrome/Edge
```

### Model Training Workflow
If you want to train your own super-resolution model from scratch:

1. Download & Prepare Data

```bash
python scripts/download_data.py
python scripts/extract_data.py
```

2. Train the Model We use a Real-ESRGAN architecture with an "On-the-Fly" degradation pipeline that simulates bad quality images during training.

```bash
python train.py
```

3. Export to Web Convert the PyTorch checkpoint (.pth) to a Web-optimized ONNX file (.onnx).

```bash
python export.py
```
Note: The export script automatically handles Float32 precision for maximum JavaScript compatibility.

### Project Structure
```
Waifu2x-WebGPU/
├── .env                 # Local configuration (Ignored by Git)
├── train.py             # Main training loop (RRDBNet)
├── export.py            # PyTorch -> ONNX converter
├── src/                 # Core AI Logic
│   ├── rrdbnet.py       # Model Architecture
│   ├── dataset.py       # Data Loading & Preprocessing
│   └── degradations.py  # Noise/Blur Simulation
├── web_dist/            # The Frontend Application
│   ├── index.html       # UI
│   ├── main.js          # Inference Logic (Tiling, ONNX Runtime)
│   └── models/          # .onnx files go here
└── scripts/             # Utilities (Downloaders, benchmarks)
```

### Contributing
Contributions are welcome! Please check the issues tab for roadmap items.
1. Fork the project.
2. Create your feature branch (git checkout -b feature/AmazingFeature).
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

### License
Distributed under the MIT License. See LICENSE for more information.