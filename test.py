import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.generator import ResNetGenerator

def main():
    parser = argparse.ArgumentParser(description="Test / Run Inference with CycleGAN")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_image", type=str, default="output.png", help="Path to save output image")
    parser.add_argument("--direction", type=str, default="xray2ct", choices=["xray2ct", "ct2xray"], help="Translation direction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Initialize Generator
    generator = ResNetGenerator(input_nc=3, output_nc=3).to(args.device)
    
    # Check direction and load correct weights
    if args.direction == "xray2ct":
        generator.load_state_dict(checkpoint["G_A2B"])
        print("Loaded X-ray -> CT Generator (G_A2B)")
    else:
        generator.load_state_dict(checkpoint["G_B2A"])
        print("Loaded CT -> X-ray Generator (G_B2A)")
        
    generator.eval()

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(args.input_image):
        print(f"Error: Input image file '{args.input_image}' not found.")
        return

    # Load and transform image
    img = Image.open(args.input_image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(args.device)

    # Inference
    print("Translating...")
    with torch.no_grad():
        translated_tensor = generator(img_tensor)
        
    # Un-normalize image to [0, 1]
    translated_tensor = (translated_tensor.squeeze(0).cpu() + 1.0) / 2.0
    translated_tensor = torch.clamp(translated_tensor, 0, 1)

    # Convert back to PIL Image and save
    to_pil = transforms.ToPILImage()
    out_img = to_pil(translated_tensor)
    out_img.save(args.output_image)
    print(f"Saved translated image to: {args.output_image}")

if __name__ == "__main__":
    main()
