import os
import argparse
import base64
import io
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# Try to safely import PyTorch/Torchvision
try:
    import torch
    import torchvision.transforms as transforms
    from models.generator import ResNetGenerator
    has_torch = True
except (ImportError, OSError) as e:
    has_torch = False
    print(f"Warning: PyTorch or torchvision could not be loaded ({e}). Running in SIMULATED DEMO mode ONLY.")

# Try to safely import OpenCV
try:
    import cv2
    has_cv2 = True
except ImportError:
    cv2 = None
    has_cv2 = False
    print("Warning: OpenCV not found. Falling back to pure Pillow image generators for simulation.")

from download_data import generate_synthetic_xray, generate_synthetic_ct

# App initialization
app = FastAPI(title="SynthoCT Backend Server", version="1.0.0")

# Add CORS Middleware to allow requests from the GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your GitHub Pages domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
generator_A2B = None
generator_B2A = None
is_demo_mode = True
device = "cpu"

class TranslationRequest(BaseModel):
    image: str  # Base64 string of the uploaded image
    direction: str  # 'xray2ct' or 'ct2xray'

class TranslationResponse(BaseModel):
    image: str  # Base64 string of the translated image

def load_models(checkpoint_path, target_device):
    global generator_A2B, generator_B2A, is_demo_mode, device
    device = target_device
    
    if not has_torch:
        print("Live model inference is disabled because PyTorch is not installed.")
        is_demo_mode = True
        return

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint '{checkpoint_path}' not found. Running in SIMULATED DEMO mode.")
        is_demo_mode = True
        return
        
    try:
        print(f"Loading weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        generator_A2B = ResNetGenerator(input_nc=3, output_nc=3).to(device)
        generator_B2A = ResNetGenerator(input_nc=3, output_nc=3).to(device)
        
        generator_A2B.load_state_dict(checkpoint["G_A2B"])
        generator_B2A.load_state_dict(checkpoint["G_B2A"])
        
        generator_A2B.eval()
        generator_B2A.eval()
        is_demo_mode = False
        print("CycleGAN models loaded successfully!")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Falling back to SIMULATED DEMO mode.")
        is_demo_mode = True

def base64_to_pil(base64_str: str) -> Image.Image:
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data)).convert("RGB")

def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"

def simulate_xray_to_ct(img):
    if has_cv2:
        # Original OpenCV-based draw simulation
        img_gray = ImageOps.grayscale(img)
        arr = np.array(img_gray)
        h, w = arr.shape
        ct_arr = np.zeros((h, w), dtype=np.uint8)
        
        cx, cy = w // 2, h // 2
        rx, ry = int(w * 0.38), int(h * 0.35)
        
        cv2.ellipse(ct_arr, (cx, cy), (rx, ry), 0, 0, 360, 65, -1)
        cv2.ellipse(ct_arr, (cx, cy), (rx, ry), 0, 0, 360, 160, 2)
        
        _, thresh = cv2.threshold(arr, 70, 255, cv2.THRESH_BINARY_INV)
        lung_w, lung_h = int(w * 0.25), int(h * 0.35)
        
        cv2.ellipse(ct_arr, (int(w * 0.36), cy), (int(lung_w * 0.5), int(lung_h * 0.5)), 0, 0, 360, 15, -1)
        cv2.ellipse(ct_arr, (int(w * 0.64), cy), (int(lung_w * 0.5), int(lung_h * 0.5)), 0, 0, 360, 15, -1)
        
        cv2.fillConvexPoly(ct_arr, np.array([
            [cx, int(h * 0.73)],
            [int(w * 0.44), int(h * 0.81)],
            [int(w * 0.56), int(h * 0.81)]
        ]), 180)
        cv2.circle(ct_arr, (cx, int(h * 0.77)), int(w * 0.03), 30, -1)
        
        cv2.ellipse(ct_arr, (int(w * 0.45), int(h * 0.52)), (int(w * 0.12), int(h * 0.12)), 0, 0, 360, 75, -1)
        
        rib_angles = [30, 60, 90, 120, 150, 210, 240, 270, 300, 330]
        for angle in rib_angles:
            rad = np.deg2rad(angle)
            x = int(cx + rx * np.cos(rad))
            y = int(cy + ry * np.sin(rad))
            cv2.ellipse(ct_arr, (x, y), (5, 8), angle, 0, 360, 210, -1)
            
        noise = np.random.normal(0, 4, ct_arr.shape)
        ct_arr = np.clip(ct_arr + noise, 0, 255).astype(np.uint8)
        ct_arr = cv2.GaussianBlur(ct_arr, (3, 3), 0)
        
        return Image.fromarray(ct_arr).convert("RGB")
    else:
        # Pure PIL fallback: generate synthetic CT slice to match size
        w, h = img.size
        return generate_synthetic_ct(256).resize((w, h), Image.Resampling.LANCZOS).convert("RGB")

def simulate_ct_to_xray(img):
    if has_cv2:
        # Original OpenCV-based draw simulation
        img_gray = ImageOps.grayscale(img)
        arr = np.array(img_gray)
        h, w = arr.shape
        
        xray_arr = np.ones((h, w), dtype=np.uint8) * 20
        cx, cy = w // 2, h // 2
        
        cv2.ellipse(xray_arr, (cx, int(h * 0.65)), (int(w * 0.4), int(h * 0.55)), 0, 0, 360, 60, -1)
        cv2.ellipse(xray_arr, (int(w * 0.32), int(h * 0.5)), (int(w * 0.13), int(h * 0.28)), 0, 0, 360, 25, -1)
        cv2.ellipse(xray_arr, (int(w * 0.68), int(h * 0.5)), (int(w * 0.13), int(h * 0.28)), 0, 0, 360, 25, -1)
        
        for y in range(int(h * 0.1), int(h * 0.95), 14):
            cv2.rectangle(xray_arr, (int(w * 0.47), y), (int(w * 0.53), y + 9), 110, -1)
            
        cv2.ellipse(xray_arr, (int(w * 0.49), int(h * 0.61)), (int(w * 0.11), int(h * 0.16)), 0, 0, 360, 95, -1)
        
        cv2.line(xray_arr, (int(w * 0.15), int(h * 0.2)), (int(w * 0.45), int(h * 0.25)), 120, 4)
        cv2.line(xray_arr, (int(w * 0.85), int(h * 0.2)), (int(w * 0.55), int(h * 0.25)), 120, 4)
        
        for y in range(int(h * 0.25), int(h * 0.8), 18):
            cv2.arc(xray_arr, (int(w * 0.32), y), (int(w * 0.35), 40), 0, 180, 340, 85, 2)
            cv2.arc(xray_arr, (int(w * 0.68), y), (int(w * 0.35), 40), 0, 200, 360, 85, 2)
            
        noise = np.random.normal(0, 6, xray_arr.shape)
        xray_arr = np.clip(xray_arr + noise, 0, 255).astype(np.uint8)
        xray_arr = cv2.GaussianBlur(xray_arr, (3, 3), 0)
        
        return Image.fromarray(xray_arr).convert("RGB")
    else:
        # Pure PIL fallback: generate synthetic chest X-ray to match size
        w, h = img.size
        return generate_synthetic_xray(256).resize((w, h), Image.Resampling.LANCZOS).convert("RGB")

@app.on_event("startup")
def startup_event():
    # Load model checkpoints from env vars or defaults during production server startup
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "")
    device_type = os.environ.get("DEVICE", "cuda" if (has_torch and torch.cuda.is_available()) else "cpu")
    print(f"Startup: Loading models from checkpoint '{checkpoint_path}' on device '{device_type}'...")
    load_models(checkpoint_path, device_type)

    os.makedirs("static/samples", exist_ok=True)
    
    # Generate static sample images dynamically if they don't exist yet
    xray_s1 = "static/samples/xray_sample1.png"
    xray_s2 = "static/samples/xray_sample2.png"
    ct_s1 = "static/samples/ct_sample1.png"
    ct_s2 = "static/samples/ct_sample2.png"
    
    if not os.path.exists(xray_s1):
        print("Generating xray_sample1.png...")
        generate_synthetic_xray(256).save(xray_s1)
    if not os.path.exists(xray_s2):
        print("Generating xray_sample2.png...")
        generate_synthetic_xray(256).save(xray_s2)
    if not os.path.exists(ct_s1):
        print("Generating ct_sample1.png...")
        generate_synthetic_ct(256).save(ct_s1)
    if not os.path.exists(ct_s2):
        print("Generating ct_sample2.png...")
        generate_synthetic_ct(256).save(ct_s2)
    
    print("Static sample resources verified/created.")

@app.get("/api/status")
def get_status():
    return {
        "status": "simulated" if (is_demo_mode or not has_torch) else "live",
        "device": device
    }

@app.post("/api/translate", response_model=TranslationResponse)
def translate(req: TranslationRequest):
    global generator_A2B, generator_B2A, is_demo_mode, device
    
    try:
        input_img = base64_to_pil(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
    if req.direction not in ["xray2ct", "ct2xray"]:
        raise HTTPException(status_code=400, detail="Invalid translation direction")
        
    try:
        if is_demo_mode or not has_torch:
            # Run Simulated fallbacks
            if req.direction == "xray2ct":
                output_img = simulate_xray_to_ct(input_img)
            else:
                output_img = simulate_ct_to_xray(input_img)
        else:
            # Run CycleGAN inference
            transform = transforms.Compose([
                transforms.Resize((256, 256), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            img_tensor = transform(input_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if req.direction == "xray2ct":
                    out_tensor = generator_A2B(img_tensor)
                else:
                    out_tensor = generator_B2A(img_tensor)
                    
            # Un-normalize
            out_tensor = (out_tensor.squeeze(0).cpu() + 1.0) / 2.0
            out_tensor = torch.clamp(out_tensor, 0, 1)
            
            to_pil = transforms.ToPILImage()
            output_img = to_pil(out_tensor)
            output_img = output_img.resize(input_img.size, Image.Resampling.LANCZOS)
            
        # Convert output back to base64
        output_base64 = pil_to_base64(output_img)
        return TranslationResponse(image=output_base64)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference execution failed: {str(e)}")

# Serve Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

def main():
    parser = argparse.ArgumentParser(description="Launch FastAPI Server for SynthoCT")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to CycleGAN model checkpoint .pt file")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to serve the application on")
    parser.add_argument("--port", type=int, default=8000, help="Port to host the server on")
    parser.add_argument("--device", type=str, default="cuda" if (has_torch and torch.cuda.is_available()) else "cpu", help="Computation device (cuda or cpu)")
    args = parser.parse_args()
    
    # Load model checkpoint (will safely fallback if PyTorch is not available)
    load_models(args.checkpoint, args.device)
    
    # Start web server
    print(f"Launching web server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
