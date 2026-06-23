import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

from models.generator import ResNetGenerator
from utils.dataset import UnpairedDataset

def calculate_metrics(real_img, gen_img):
    """
    Computes SSIM and PSNR between two numpy images (H, W, C) in range [0, 255] or [0, 1].
    """
    # Convert to grayscale for metrics calculation if 3 channels, or check channel axis
    if len(real_img.shape) == 3 and real_img.shape[2] == 3:
        real_gray = np.array(Image.fromarray(real_img).convert('L'))
        gen_gray = np.array(Image.fromarray(gen_img).convert('L'))
    else:
        real_gray = real_img
        gen_gray = gen_img

    s = ssim(real_gray, gen_gray, data_range=real_gray.max() - real_gray.min())
    p = psnr(real_gray, gen_gray, data_range=real_gray.max() - real_gray.min())
    return s, p

def main():
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN Translation Quality")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save comparison images")
    args = parser.parse_args()

    print(f"Loading model checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # Initialize generators
    G_A2B = ResNetGenerator(input_nc=3, output_nc=3).to(args.device)  # X-ray -> CT
    G_B2A = ResNetGenerator(input_nc=3, output_nc=3).to(args.device)  # CT -> X-ray

    G_A2B.load_state_dict(checkpoint["G_A2B"])
    G_B2A.load_state_dict(checkpoint["G_B2A"])

    G_A2B.eval()
    G_B2A.eval()

    # Load test dataset
    dataset = UnpairedDataset(args.data_dir, phase="test")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output_dir, exist_ok=True)

    # Metrics accumulators
    ssim_translation_scores = []
    psnr_translation_scores = []
    
    ssim_cycle_A_scores = []
    psnr_cycle_A_scores = []
    
    ssim_cycle_B_scores = []
    psnr_cycle_B_scores = []

    print("Evaluating models...")
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(args.device)  # X-ray
            real_B = batch["B"].to(args.device)  # CT (ground truth paired slice for evaluation index)

            # Generate translations
            fake_B = G_A2B(real_A)
            rec_A = G_B2A(fake_B)

            fake_A = G_B2A(real_B)
            rec_B = G_A2B(fake_A)

            # Convert tensors to numpy arrays for metrics [0, 255]
            def to_np(img_tensor):
                img = (img_tensor.squeeze(0).cpu().numpy() + 1.0) / 2.0
                img = np.clip(img * 255, 0, 255).astype(np.uint8)
                return np.transpose(img, (1, 2, 0))

            np_real_A = to_np(real_A)
            np_fake_B = to_np(fake_B)
            np_rec_A = to_np(rec_A)

            np_real_B = to_np(real_B)
            np_fake_A = to_np(fake_A)
            np_rec_B = to_np(rec_B)

            # 1. Translation quality metrics (synthesized CT vs target CT)
            s_trans, p_trans = calculate_metrics(np_real_B, np_fake_B)
            ssim_translation_scores.append(s_trans)
            psnr_translation_scores.append(p_trans)

            # 2. Cycle-consistency metrics (reconstructed X-ray vs real X-ray)
            s_cyc_A, p_cyc_A = calculate_metrics(np_real_A, np_rec_A)
            ssim_cycle_A_scores.append(s_cyc_A)
            psnr_cycle_A_scores.append(p_cyc_A)

            # 3. Cycle-consistency metrics (reconstructed CT vs real CT)
            s_cyc_B, p_cyc_B = calculate_metrics(np_real_B, np_rec_B)
            ssim_cycle_B_scores.append(s_cyc_B)
            psnr_cycle_B_scores.append(p_cyc_B)

            # Save comparison grid for the first 10 images
            if i < 10:
                grid = torch.cat((real_A, fake_B, rec_A, real_B, fake_A, rec_B), 0)
                grid = (grid + 1.0) / 2.0
                save_image(grid, os.path.join(args.output_dir, f"comparison_{i:02d}.png"), nrow=3, normalize=False)

    print("\n================== Evaluation Results ==================")
    print(f"Total test samples evaluated: {len(dataloader)}")
    print("-------------------- Translation --------------------")
    print(f"Average X-ray -> CT Translation SSIM: {np.mean(ssim_translation_scores):.4f}")
    print(f"Average X-ray -> CT Translation PSNR: {np.mean(psnr_translation_scores):.2f} dB")
    print("----------------- Cycle Consistency -----------------")
    print(f"Average X-ray Reconstruction SSIM: {np.mean(ssim_cycle_A_scores):.4f}")
    print(f"Average X-ray Reconstruction PSNR: {np.mean(psnr_cycle_A_scores):.2f} dB")
    print(f"Average CT Reconstruction SSIM: {np.mean(ssim_cycle_B_scores):.4f}")
    print(f"Average CT Reconstruction PSNR: {np.mean(psnr_cycle_B_scores):.2f} dB")
    print("=======================================================")
    print(f"Comparison grids saved to folder: '{args.output_dir}'")

if __name__ == "__main__":
    main()
