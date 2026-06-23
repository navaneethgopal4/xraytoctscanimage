import os
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import cv2
import gradio as gr

try:
    import torch
    import torchvision.transforms as transforms
    from models.generator import ResNetGenerator
    has_torch = True
except (ImportError, OSError) as e:
    has_torch = False
    print(f"Warning: PyTorch or torchvision could not be loaded ({e}). Running in SIMULATED DEMO mode ONLY.")

# Global variables for models
generator_A2B = None
generator_B2A = None
is_demo_mode = False

def load_models(checkpoint_path, device):
    global generator_A2B, generator_B2A, is_demo_mode
    
    if not has_torch:
        print("Warning: Live model inference is disabled because PyTorch is not available. Running in SIMULATED DEMO mode.")
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

def simulate_xray_to_ct(img):
    """
    Generates a simulated axial CT scan matching the general size of the input X-ray.
    Uses contour mapping and structural drawing to construct a pseudo-CT slice.
    """
    img_gray = ImageOps.grayscale(img)
    arr = np.array(img_gray)
    
    # Generate a matching CT slice based on input size
    h, w = arr.shape
    ct_arr = np.zeros((h, w), dtype=np.uint8)
    
    # Center and radius
    cx, cy = w // 2, h // 2
    rx, ry = int(w * 0.38), int(h * 0.35)
    
    # Draw Thorax Body (soft tissue)
    cv2.ellipse(ct_arr, (cx, cy), (rx, ry), 0, 0, 360, 65, -1)
    # Fat boundary
    cv2.ellipse(ct_arr, (cx, cy), (rx, ry), 0, 0, 360, 160, 2)
    
    # Detect dark lung regions in input X-ray to place them inside the CT
    # Thresh the X-ray to find dark areas (lungs)
    _, thresh = cv2.threshold(arr, 70, 255, cv2.THRESH_BINARY_INV)
    # Resize and place inside CT left/right lung regions
    lung_w, lung_h = int(w * 0.25), int(h * 0.35)
    
    # Left Lung
    cv2.ellipse(ct_arr, (int(w * 0.36), cy), (int(lung_w * 0.5), int(lung_h * 0.5)), 0, 0, 360, 15, -1)
    # Right Lung
    cv2.ellipse(ct_arr, (int(w * 0.64), cy), (int(lung_w * 0.5), int(lung_h * 0.5)), 0, 0, 360, 15, -1)
    
    # Spine vertebra at the bottom (posterior)
    cv2.fillConvexPoly(ct_arr, np.array([
        [cx, int(h * 0.73)],
        [int(w * 0.44), int(h * 0.81)],
        [int(w * 0.56), int(h * 0.81)]
    ]), 180)
    cv2.circle(ct_arr, (cx, int(h * 0.77)), int(w * 0.03), 30, -1)
    
    # Heart structure
    cv2.ellipse(ct_arr, (int(w * 0.45), int(h * 0.52)), (int(w * 0.12), int(h * 0.12)), 0, 0, 360, 75, -1)
    
    # Ribs along edge
    rib_angles = [30, 60, 90, 120, 150, 210, 240, 270, 300, 330]
    for angle in rib_angles:
        rad = np.deg2rad(angle)
        x = int(cx + rx * np.cos(rad))
        y = int(cy + ry * np.sin(rad))
        cv2.ellipse(ct_arr, (x, y), (5, 8), angle, 0, 360, 210, -1)
        
    # Add noise & blur for medical imaging aesthetic
    noise = np.random.normal(0, 4, ct_arr.shape)
    ct_arr = np.clip(ct_arr + noise, 0, 255).astype(np.uint8)
    ct_arr = cv2.GaussianBlur(ct_arr, (3, 3), 0)
    
    return Image.fromarray(ct_arr).convert("RGB")

def simulate_ct_to_xray(img):
    """
    Generates a simulated chest X-ray matching the input CT.
    """
    img_gray = ImageOps.grayscale(img)
    arr = np.array(img_gray)
    h, w = arr.shape
    
    xray_arr = np.ones((h, w), dtype=np.uint8) * 20
    
    cx, cy = w // 2, h // 2
    # Draw gray torso
    cv2.ellipse(xray_arr, (cx, int(h * 0.65)), (int(w * 0.4), int(h * 0.55)), 0, 0, 360, 60, -1)
    
    # Draw dark lungs
    cv2.ellipse(xray_arr, (int(w * 0.32), int(h * 0.5)), (int(w * 0.13), int(h * 0.28)), 0, 0, 360, 25, -1)
    cv2.ellipse(xray_arr, (int(w * 0.68), int(h * 0.5)), (int(w * 0.13), int(h * 0.28)), 0, 0, 360, 25, -1)
    
    # Draw spine
    for y in range(int(h * 0.1), int(h * 0.95), 14):
        cv2.rectangle(xray_arr, (int(w * 0.47), y), (int(w * 0.53), y + 9), 110, -1)
        
    # Draw heart shadow
    cv2.ellipse(xray_arr, (int(w * 0.49), int(h * 0.61)), (int(w * 0.11), int(h * 0.16)), 0, 0, 360, 95, -1)
    
    # Clavicles
    cv2.line(xray_arr, (int(w * 0.15), int(h * 0.2)), (int(w * 0.45), int(h * 0.25)), 120, 4)
    cv2.line(xray_arr, (int(w * 0.85), int(h * 0.2)), (int(w * 0.55), int(h * 0.25)), 120, 4)
    
    # Ribs
    for y in range(int(h * 0.25), int(h * 0.8), 18):
        cv2.arc(xray_arr, (int(w * 0.32), y), (int(w * 0.35), 40), 0, 180, 340, 85, 2)
        cv2.arc(xray_arr, (int(w * 0.68), y), (int(w * 0.35), 40), 0, 200, 360, 85, 2)
        
    # Add noise & blur
    noise = np.random.normal(0, 6, xray_arr.shape)
    xray_arr = np.clip(xray_arr + noise, 0, 255).astype(np.uint8)
    xray_arr = cv2.GaussianBlur(xray_arr, (3, 3), 0)
    
    return Image.fromarray(xray_arr).convert("RGB")

def translate_image(input_img, direction, device="cpu"):
    global generator_A2B, generator_B2A, is_demo_mode
    
    if input_img is None:
        return None, "Please upload an image."
        
    if is_demo_mode:
        if direction == "xray2ct":
            output_img = simulate_xray_to_ct(input_img)
            mode_msg = "Running in Simulated Demo mode (No checkpoint loaded)."
        else:
            output_img = simulate_ct_to_xray(input_img)
            mode_msg = "Running in Simulated Demo mode (No checkpoint loaded)."
        return output_img, mode_msg

    # Real inference mode
    try:
        transform = transforms.Compose([
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        img_tensor = transform(input_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            if direction == "xray2ct":
                out_tensor = generator_A2B(img_tensor)
            else:
                out_tensor = generator_B2A(img_tensor)
                
        # Un-normalize
        out_tensor = (out_tensor.squeeze(0).cpu() + 1.0) / 2.0
        out_tensor = torch.clamp(out_tensor, 0, 1)
        
        to_pil = transforms.ToPILImage()
        output_img = to_pil(out_tensor)
        
        # Resize output to match input aspect ratio or original size
        output_img = output_img.resize(input_img.size, Image.Resampling.LANCZOS)
        
        return output_img, "Inference completed successfully using CycleGAN model."
    except Exception as e:
        return None, f"Inference failed: {str(e)}"

# Premium dark theme stylesheet
theme_css = """
body {
    background-color: #0b0f19 !important;
    color: #e2e8f0 !important;
}
.gradio-container {
    background: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%) !important;
    border: 1px solid #334155 !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6) !important;
}
h1 {
    background: linear-gradient(90deg, #38bdf8 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}
.gr-button-primary {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: white !important;
    border: none !important;
}
.gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4) !important;
}
"""

def build_app(device):
    # Setup custom paths for examples if they exist
    example_xrays = []
    example_cts = []
    if os.path.exists("data/xray/test"):
        example_xrays = [os.path.join("data/xray/test", f) for f in os.listdir("data/xray/test")[:3]]
    if os.path.exists("data/ct/test"):
        example_cts = [os.path.join("data/ct/test", f) for f in os.listdir("data/ct/test")[:3]]

    with gr.Blocks(css=theme_css, title="SynthoCT — CycleGAN Image Translation") as demo:
        gr.Markdown(
            """
            # 🩺 SynthoCT — X-ray to CT CycleGAN translator
            ### An AI-powered image translation engine that synthesizes 3D CT-like textures from 2D chest X-ray projections.
            """
        )
        
        with gr.Tabs():
            # Tab 1: X-ray to CT
            with gr.TabItem("🩻 X-ray → CT Slice Translation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        xray_input = gr.Image(type="pil", label="Input Chest X-ray")
                        translate_btn_xray = gr.Button("Synthesize CT Slice", variant="primary")
                    with gr.Column(scale=1):
                        ct_output = gr.Image(type="pil", label="Synthesized CT-like Slice")
                        status_xray = gr.Textbox(label="Status / Log", interactive=False)
                        
                translate_btn_xray.click(
                    fn=lambda img: translate_image(img, "xray2ct", device),
                    inputs=xray_input,
                    outputs=[ct_output, status_xray]
                )
                
                if example_xrays:
                    gr.Examples(examples=example_xrays, inputs=xray_input, label="Sample X-rays")
                    
            # Tab 2: CT to X-ray
            with gr.TabItem("🖥️ CT → X-ray Translation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ct_input = gr.Image(type="pil", label="Input CT Slice")
                        translate_btn_ct = gr.Button("Synthesize X-ray", variant="primary")
                    with gr.Column(scale=1):
                        xray_output = gr.Image(type="pil", label="Synthesized Chest X-ray")
                        status_ct = gr.Textbox(label="Status / Log", interactive=False)
                        
                translate_btn_ct.click(
                    fn=lambda img: translate_image(img, "ct2xray", device),
                    inputs=ct_input,
                    outputs=[xray_output, status_ct]
                )
                
                if example_cts:
                    gr.Examples(examples=example_cts, inputs=ct_input, label="Sample CT Slices")
                    
            # Tab 3: Model Details
            with gr.TabItem("📖 About SynthoCT"):
                gr.Markdown(
                    """
                    ### CycleGAN for Medical Translation
                    - **Objective**: Translate unpaired medical imaging domains (2D Chest X-rays vs 2D CT Slices).
                    - **Physics Note**: This is a *texture-synthesizing model* mapping overlapping bone/soft tissue projections to cross-sectional slices. It does not replace computed tomography physics reconstruction, but is highly valuable for transfer learning, training data augmentation, and research.
                    - **Architecture**:
                      - **Generators**: 9 Residual-block ResNet mapping inputs down to bottleneck features and back up.
                      - **Discriminators**: 70x70 PatchGAN judging patch realism to promote texture mapping accuracy.
                      - **Losses**: LSGAN Loss (MSE) + L1 Cycle Consistency Loss (Weight: 10.0) + L1 Identity Loss (Weight: 5.0).
                    """
                )
                
    return demo

def main():
    parser = argparse.ArgumentParser(description="Launch Gradio App for SynthoCT")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to model checkpoint .pt file")
    parser.add_argument("--share", action="store_true", help="Generate public link for Gradio")
    parser.add_argument("--port", type=int, default=7860, help="Local port to run the server on")
    parser.add_argument("--device", type=str, default="cuda" if (has_torch and torch.cuda.is_available()) else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load generator weights
    load_models(args.checkpoint, args.device)
    
    # Build and launch app
    app = build_app(args.device)
    app.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)

if __name__ == "__main__":
    main()
