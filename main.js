// --- CONFIGURATION ---
const MODELS = {
    none: './models/scale2x_clean.onnx',
    high: './models/scale2x_denoise.onnx'
};

let TILE_SIZE = 384; 
const PADDING = 32;

// State
let currentSession = null;
let currentModelPath = "";
let inputImage = null;
let hardwareTier = "unknown";

// --- 1. HARDWARE SCOUTING ---
async function detectHardware() {
    if (!navigator.gpu) {
        console.warn("WebGPU not supported. Fallback to WASM.");
        return "cpu";
    }
    try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) return "cpu";
        const limits = adapter.limits;
        
        if (limits.maxTextureDimension2D >= 8192) {
            hardwareTier = "high-end";
            TILE_SIZE = 512; 
        } else if (limits.maxTextureDimension2D >= 4096) {
            hardwareTier = "mid-range";
            TILE_SIZE = 384;
        } else {
            hardwareTier = "low-end";
            TILE_SIZE = 256;
        }
        updateStatus(`Hardware: ${hardwareTier.toUpperCase()} (Tile: ${TILE_SIZE}px)`);
    } catch (e) {
        console.error("Hardware detection failed:", e);
        TILE_SIZE = 256;
    }
}
detectHardware();

// --- 2. File Upload Handler ---
document.getElementById('fileInput').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (evt) => {
        inputImage = new Image();
        inputImage.src = evt.target.result;
        inputImage.onload = () => {
            // 1. Load Logic Image
            document.getElementById('inputPreview').src = inputImage.src;
            
            // 2. Setup Slider Visuals (Show Original in both slots initially)
            const compContainer = document.getElementById('compareContainer');
            const compOriginal = document.getElementById('compOriginal');
            const compResult = document.getElementById('compResult');
            
            compOriginal.src = inputImage.src;
            compResult.src = inputImage.src; 
            
            compContainer.style.display = 'block';
            
            // 3. Fix Layout (Center the box perfectly to the image)
            updateSliderDimensions(inputImage);

            // 4. Enable UI
            document.getElementById('runBtn').disabled = false;
            updateStatus(`Ready. Mode: ${hardwareTier.toUpperCase()}`);
            
            // Initialize slider logic
            initSlider();
        };
    };
    reader.readAsDataURL(file);
});

// --- 3. Main Controller ---
async function startProcessing() {
    if (!inputImage) return;
    
    const noiseLevel = document.getElementById('noiseLevel').value;
    const scaleFactor = parseInt(document.getElementById('scaleFactor').value);
    const btn = document.getElementById('runBtn');
    
    btn.disabled = true;
    
    try {
        const targetPath = MODELS[noiseLevel];
        if (currentModelPath !== targetPath) {
            updateStatus(`Loading Model (${noiseLevel})...`);
            currentSession = await ort.InferenceSession.create(targetPath, {
                executionProviders: ['webgpu', 'wasm']
            });
            currentModelPath = targetPath;
        }

        if (scaleFactor === 2) {
            await processTiled(inputImage, 2);
        } else if (scaleFactor === 4) {
            updateStatus("Pass 1/2 (2x)...");
            const pass1Canvas = document.createElement('canvas');
            await processTiled(inputImage, 2, pass1Canvas);
            
            const pass1Img = new Image();
            pass1Img.src = pass1Canvas.toDataURL();
            await new Promise(r => pass1Img.onload = r);

            updateStatus("Pass 2/2 (4x)...");
            await processTiled(pass1Img, 2, document.getElementById('outputCanvas'));
        }

        updateStatus("Done!");
        
        // --- UPDATE SLIDER WITH RESULT ---
        const resultCanvas = document.getElementById('outputCanvas');
        document.getElementById('compResult').src = resultCanvas.toDataURL();
        
        // Refresh dimensions in case aspect ratio changed slightly
        updateSliderDimensions(inputImage);
        
        document.getElementById('downloadLink').style.display = 'block';

    } catch (e) {
        console.error(e);
        updateStatus("Error: " + e.message);
    } finally {
        btn.disabled = false;
    }
}

// --- 4. Helper: Dynamic Slider Sizing ---
function updateSliderDimensions(img) {
    const container = document.getElementById('compareContainer');
    const maxWidth = 800; // Match CSS max-width
    const maxHeight = window.innerHeight * 0.7; // 70% of screen height
    
    const aspect = img.width / img.height;
    
    // Calculate best fit
    let finalW = maxWidth;
    let finalH = maxWidth / aspect;
    
    if (finalH > maxHeight) {
        finalH = maxHeight;
        finalW = maxHeight * aspect;
    }
    
    // Apply exact pixel sizes to remove black bars
    container.style.width = finalW + 'px';
    container.style.height = finalH + 'px';
}

// --- 5. Slider Logic ---
function initSlider() {
    const container = document.getElementById('compareContainer');
    const modifiedImg = document.getElementById('compOriginal'); // The Top Layer
    const handle = document.getElementById('sliderHandle');
    let active = false;

    // Default to 50%
    modifiedImg.style.clipPath = `inset(0 50% 0 0)`;
    handle.style.left = `50%`;

    const startDrag = () => active = true;
    const endDrag = () => active = false;
    const onMove = (x) => {
        if (!active) return;
        const rect = container.getBoundingClientRect();
        let pos = ((x - rect.left) / rect.width) * 100;
        pos = Math.max(0, Math.min(100, pos));
        
        // Clip Right side of Top Image based on slider position
        // If slider is at 10% (Left), Clip 90% from right -> Shows small strip of Original
        modifiedImg.style.clipPath = `inset(0 ${100 - pos}% 0 0)`;
        handle.style.left = `${pos}%`;
    };

    container.addEventListener('mousedown', startDrag);
    container.addEventListener('touchstart', startDrag);
    
    window.addEventListener('mouseup', endDrag);
    window.addEventListener('touchend', endDrag);

    container.addEventListener('mousemove', (e) => onMove(e.clientX));
    container.addEventListener('touchmove', (e) => onMove(e.touches[0].clientX));
}

// --- 6. Tiling Logic ---
async function processTiled(sourceImage, scale, targetCanvas = null) {
    const outCanvas = targetCanvas || document.getElementById('outputCanvas');
    const ctx = outCanvas.getContext('2d');
    const outWidth = sourceImage.width * scale;
    const outHeight = sourceImage.height * scale;
    outCanvas.width = outWidth;
    outCanvas.height = outHeight;
    
    const step = TILE_SIZE - (2 * PADDING);
    const xSteps = Math.ceil(sourceImage.width / step);
    const ySteps = Math.ceil(sourceImage.height / step);
    const totalTiles = xSteps * ySteps;
    let processedTiles = 0;

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = TILE_SIZE;
    tempCanvas.height = TILE_SIZE;
    const tempCtx = tempCanvas.getContext('2d', { willReadFrequently: true });

    for (let y = 0; y < sourceImage.height; y += step) {
        for (let x = 0; x < sourceImage.width; x += step) {
            updateStatus(`Processing Tile ${processedTiles + 1}/${totalTiles} (Tier: ${hardwareTier})...`);
            
            let srcX = x - PADDING;
            let srcY = y - PADDING;
            
            tempCtx.clearRect(0, 0, TILE_SIZE, TILE_SIZE);
            tempCtx.drawImage(sourceImage, srcX, srcY, TILE_SIZE, TILE_SIZE, 0, 0, TILE_SIZE, TILE_SIZE);

            const tileTensor = await imageToTensor(tempCanvas);
            try {
                const feeds = { input: tileTensor };
                const results = await currentSession.run(feeds);
                const outputTensor = results.output;

                const outStep = step * scale;
                const outPadding = PADDING * scale;
                const destX = (x < PADDING) ? 0 : (x * scale); 
                const destY = (y < PADDING) ? 0 : (y * scale);
                
                drawTensorRegionToCanvas(outputTensor, ctx, destX, destY, outPadding, outPadding, outStep, outStep, outWidth, outHeight);
            } catch (err) {
                console.error("Tile Inference Failed:", err);
                throw new Error("GPU crashed on a tile. Try reloading.");
            }
            processedTiles++;
            await new Promise(r => setTimeout(r, 10));
        }
    }
}

async function imageToTensor(canvas) {
    const ctx = canvas.getContext('2d');
    const { data, width: imgW, height: imgH } = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const float32Data = new Float32Array(3 * imgW * imgH);
    for (let i = 0; i < imgW * imgH; i++) {
        float32Data[i] = data[i * 4] / 255.0; 
        float32Data[i + (imgW * imgH)] = data[i * 4 + 1] / 255.0; 
        float32Data[i + (2 * imgW * imgH)] = data[i * 4 + 2] / 255.0; 
    }
    return new ort.Tensor('float32', float32Data, [1, 3, imgH, imgW]);
}

function drawTensorRegionToCanvas(tensor, ctx, destX, destY, srcX, srcY, srcW, srcH, maxX, maxY) {
    const [batch, channels, height, width] = tensor.dims;
    const data = tensor.data;
    const drawW = Math.min(srcW, maxX - destX);
    const drawH = Math.min(srcH, maxY - destY);
    if (drawW <= 0 || drawH <= 0) return;
    const imgData = ctx.createImageData(drawW, drawH);
    for (let y = 0; y < drawH; y++) {
        for (let x = 0; x < drawW; x++) {
            const globalX = srcX + x;
            const globalY = srcY + y;
            const tensorIdx = (globalY * width) + globalX;
            const imgIdx = (y * drawW + x) * 4;
            const r = data[tensorIdx] * 255.0;
            const g = data[tensorIdx + (width * height)] * 255.0;
            const b = data[tensorIdx + (2 * width * height)] * 255.0;
            imgData.data[imgIdx] = Math.max(0, Math.min(255, r));
            imgData.data[imgIdx + 1] = Math.max(0, Math.min(255, g));
            imgData.data[imgIdx + 2] = Math.max(0, Math.min(255, b));
            imgData.data[imgIdx + 3] = 255;
        }
    }
    ctx.putImageData(imgData, destX, destY);
}

function updateStatus(msg) {
    document.getElementById('statusText').innerText = msg;
}

function downloadImage() {
    const link = document.createElement('a');
    link.download = 'upscaled.png';
    link.href = document.getElementById('outputCanvas').toDataURL();
    link.click();
}