import os
import argparse
import numpy as np
try:
    import cv2
except ImportError:
    cv2 = None
from PIL import Image, ImageDraw, ImageFilter

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_synthetic_xray(size=256):
    """
    Generates a synthetic chest X-ray image using geometric shapes.
    Contains ribs, lung fields, spine, heart, and diaphragm.
    """
    # Create black canvas
    img = Image.new('L', (size, size), 20)
    draw = ImageDraw.Draw(img)
    
    # Torso contour (broad gray area)
    draw.ellipse([size * 0.1, size * 0.1, size * 0.9, size * 1.2], fill=60)
    
    # Lung fields (two large dark ellipses)
    draw.ellipse([size * 0.2, size * 0.25, size * 0.45, size * 0.8], fill=25)
    draw.ellipse([size * 0.55, size * 0.25, size * 0.8, size * 0.8], fill=25)
    
    # Diaphragm (bottom curves)
    draw.ellipse([size * 0.15, size * 0.75, size * 0.5, size * 1.1], fill=50)
    draw.ellipse([size * 0.5, size * 0.75, size * 0.85, size * 1.1], fill=50)
    
    # Spine (vertical columns of segments)
    for y in range(int(size * 0.1), int(size * 0.9), 12):
        draw.rectangle([size * 0.47, y, size * 0.53, y + 8], fill=110)
    
    # Heart shadow (light area overlapping left lung and center)
    draw.ellipse([size * 0.38, size * 0.45, size * 0.6, size * 0.78], fill=95)
    
    # Clavicles
    draw.line([size * 0.15, size * 0.2, size * 0.45, size * 0.25], fill=120, width=4)
    draw.line([size * 0.85, size * 0.2, size * 0.55, size * 0.25], fill=120, width=4)
    
    # Ribs (curved white arcs across lungs)
    for y in range(int(size * 0.25), int(size * 0.8), 16):
        # Left ribs
        draw.arc([size * 0.15, y - 20, size * 0.5, y + 20], start=180, end=340, fill=85, width=3)
        # Right ribs
        draw.arc([size * 0.5, y - 20, size * 0.85, y + 20], start=200, end=360, fill=85, width=3)

    # Convert to numpy to apply vascular markings and noise
    arr = np.array(img, dtype=np.float32)
    
    # Vascular markings (tiny random lines in lungs)
    mask_lungs = ((arr < 40) & (arr > 20))
    noise_vascular = np.random.normal(0, 15, arr.shape)
    arr = np.where(mask_lungs, arr + np.clip(noise_vascular, 0, 30), arr)
    
    # Add general sensor noise and blur slightly for realism
    noise = np.random.normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    if cv2 is not None:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
        return Image.fromarray(arr)
    else:
        # Fallback to PIL GaussianBlur
        return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1))

def generate_synthetic_ct(size=256):
    """
    Generates a synthetic axial chest CT slice using geometric shapes.
    Contains external skin boundary, subcutaneous fat, spine, lungs, ribs, and mediastinum.
    """
    img = Image.new('L', (size, size), 0)
    draw = ImageDraw.Draw(img)
    
    # Thorax body outline (gray tissue)
    draw.ellipse([size * 0.12, size * 0.15, size * 0.88, size * 0.85], fill=65)
    
    # Subcutaneous fat boundary (thin outer line)
    draw.arc([size * 0.12, size * 0.15, size * 0.88, size * 0.85], start=0, end=360, fill=150, width=2)
    
    # Lungs (two large dark cavities)
    # CT slices have lungs at the top-left and top-right (anterior at top, posterior at bottom)
    draw.ellipse([size * 0.2, size * 0.28, size * 0.46, size * 0.68], fill=15)
    draw.ellipse([size * 0.54, size * 0.28, size * 0.8, size * 0.68], fill=15)
    
    # Spine vertebra at the bottom (posterior)
    # Outer bone triangle/shield
    draw.polygon([
        (size * 0.5, size * 0.72),
        (size * 0.44, size * 0.82),
        (size * 0.56, size * 0.82)
    ], fill=180)
    # Spinal canal (inner circle)
    draw.ellipse([size * 0.47, size * 0.75, size * 0.53, size * 0.81], fill=30)
    
    # Sternum at the top (anterior)
    draw.rectangle([size * 0.46, size * 0.18, size * 0.54, size * 0.21], fill=160)
    
    # Heart and mediastinum in the center
    draw.ellipse([size * 0.38, size * 0.4, size * 0.62, size * 0.65], fill=75)
    
    # Ribs along the sides (small bright ellipses)
    rib_angles = [30, 60, 90, 120, 150, 210, 240, 270, 300, 330]
    cx, cy = size * 0.5, size * 0.5
    rx, ry = size * 0.35, size * 0.32
    for angle in rib_angles:
        rad = np.deg2rad(angle)
        x = cx + rx * np.cos(rad)
        y = cy + ry * np.sin(rad)
        draw.ellipse([x - 4, y - 6, x + 4, y + 6], fill=200)

    # Convert to numpy for vascular trees inside lungs and noise
    arr = np.array(img, dtype=np.float32)
    
    # Lung vessels (white spots inside lungs)
    mask_lungs = (arr == 15)
    vessel_noise = np.random.binomial(1, 0.05, arr.shape) * 100
    arr = np.where(mask_lungs, arr + vessel_noise, arr)
    
    # Add noise for CT scan appearance
    noise = np.random.normal(0, 3, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    
    # Slight blur to merge vessel points
    if cv2 is not None:
        arr = cv2.GaussianBlur(arr, (3, 3), 0)
        return Image.fromarray(arr)
    else:
        # Fallback to PIL GaussianBlur
        return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(1))

def main():
    parser = argparse.ArgumentParser(description="Download or generate X-ray and CT datasets.")
    parser.add_argument("--quick", action="store_true", help="Generate a smaller dataset quickly.")
    parser.add_argument("--data_dir", type=str, default="data", help="Target directory for the dataset.")
    args = parser.parse_args()
    
    num_train = 100 if args.quick else 1000
    num_test = 20 if args.quick else 200
    
    print(f"Generating synthetic dataset in '{args.data_dir}'...")
    print(f"Train samples: {num_train}, Test samples: {num_test}")
    
    # Define directories
    xray_train_dir = os.path.join(args.data_dir, "xray", "train")
    xray_test_dir = os.path.join(args.data_dir, "xray", "test")
    ct_train_dir = os.path.join(args.data_dir, "ct", "train")
    ct_test_dir = os.path.join(args.data_dir, "ct", "test")
    
    create_dir(xray_train_dir)
    create_dir(xray_test_dir)
    create_dir(ct_train_dir)
    create_dir(ct_test_dir)
    
    # Generate Train X-rays
    print("Generating train X-rays...")
    for i in range(num_train):
        img = generate_synthetic_xray()
        img.save(os.path.join(xray_train_dir, f"xray_{i:04d}.png"))
        
    # Generate Train CTs
    print("Generating train CT slices...")
    for i in range(num_train):
        img = generate_synthetic_ct()
        img.save(os.path.join(ct_train_dir, f"ct_{i:04d}.png"))
        
    # Generate Test X-rays
    print("Generating test X-rays...")
    for i in range(num_test):
        img = generate_synthetic_xray()
        img.save(os.path.join(xray_test_dir, f"xray_{i:04d}.png"))
        
    # Generate Test CTs
    print("Generating test CT slices...")
    for i in range(num_test):
        img = generate_synthetic_ct()
        img.save(os.path.join(ct_test_dir, f"ct_{i:04d}.png"))
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()
