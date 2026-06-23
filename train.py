import os
import argparse
import random
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

from models.generator import ResNetGenerator
from models.discriminator import PatchGANDiscriminator
from utils.dataset import UnpairedDataset

class ImageBuffer:
    """
    Buffer to store previously generated fake images to update discriminators.
    Prevents discriminator from forgetting previous states and stabilizes training.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, images):
        to_return = []
        for image in images.data:
            image = torch.unsqueeze(image, 0)
            if len(self.data) < self.max_size:
                self.data.append(image)
                to_return.append(image)
            else:
                if random.uniform(0, 1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[idx].clone())
                    self.data[idx] = image
                else:
                    to_return.append(image)
        return torch.cat(to_return, 0)

def lr_lambda(epoch, total_epochs, decay_epoch):
    """
    Calculates learning rate decay factor.
    Returns 1.0 up to decay_epoch, then decays linearly to 0.0.
    """
    if epoch < decay_epoch:
        return 1.0
    return 1.0 - float(epoch - decay_epoch) / (total_epochs - decay_epoch + 1e-8)

def main():
    parser = argparse.ArgumentParser(description="Train CycleGAN for X-ray to CT Translation")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data folder")
    parser.add_argument("--epochs", type=int, default=100, help="Total number of training epochs")
    parser.add_argument("--decay_epoch", type=int, default=50, help="Epoch to start linear learning rate decay")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (standard is 1)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Initial learning rate")
    parser.add_argument("--lambda_cyc", type=float, default=10.0, help="Weight for cycle consistency loss")
    parser.add_argument("--lambda_id", type=float, default=5.0, help="Weight for identity loss")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Epoch interval to save checkpoints")
    parser.add_argument("--sample_interval", type=int, default=200, help="Step interval to save generated sample grids")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint file to resume training")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # Initialize models
    G_A2B = ResNetGenerator(input_nc=3, output_nc=3).to(args.device)  # X-ray -> CT
    G_B2A = ResNetGenerator(input_nc=3, output_nc=3).to(args.device)  # CT -> X-ray
    D_A = PatchGANDiscriminator(input_nc=3).to(args.device)          # Evaluates X-ray
    D_B = PatchGANDiscriminator(input_nc=3).to(args.device)          # Evaluates CT

    # Weight initialization
    def weights_init_normal(m):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    G_A2B.apply(weights_init_normal)
    G_B2A.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # Define Loss Functions
    criterion_GAN = nn.MSELoss()  # Least Squares GAN (LSGAN)
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers
    optimizer_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=lambda epoch: lr_lambda(epoch, args.epochs, args.decay_epoch)
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=lambda epoch: lr_lambda(epoch, args.epochs, args.decay_epoch)
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=lambda epoch: lr_lambda(epoch, args.epochs, args.decay_epoch)
    )

    # Buffers for fake images
    fake_A_buffer = ImageBuffer()
    fake_B_buffer = ImageBuffer()

    start_epoch = 0

    # Resume training
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            start_epoch = checkpoint["epoch"]
            G_A2B.load_state_dict(checkpoint["G_A2B"])
            G_B2A.load_state_dict(checkpoint["G_B2A"])
            D_A.load_state_dict(checkpoint["D_A"])
            D_B.load_state_dict(checkpoint["D_B"])
            optimizer_G.load_state_dict(checkpoint["optimizer_G"])
            optimizer_D_A.load_state_dict(checkpoint["optimizer_D_A"])
            optimizer_D_B.load_state_dict(checkpoint["optimizer_D_B"])
            lr_scheduler_G.load_state_dict(checkpoint["scheduler_G"])
            lr_scheduler_D_A.load_state_dict(checkpoint["scheduler_D_A"])
            lr_scheduler_D_B.load_state_dict(checkpoint["scheduler_D_B"])
            print(f"Successfully loaded checkpoint at epoch {start_epoch}")
        else:
            print(f"Warning: No checkpoint found at {args.resume}. Starting from scratch.")

    # Data Loader
    dataset = UnpairedDataset(args.data_dir, phase="train")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Training Loop
    for epoch in range(start_epoch, args.epochs):
        for i, batch in enumerate(dataloader):
            # Real Images
            real_A = batch["A"].to(args.device) # X-ray
            real_B = batch["B"].to(args.device) # CT

            # Adversarial ground truths
            # PatchGAN outputs 30x30 output grids for 256x256 images
            out_shape = (real_A.size(0), 1, 30, 30)
            valid = torch.ones(out_shape, requires_grad=False).to(args.device)
            fake = torch.zeros(out_shape, requires_grad=False).to(args.device)

            # ------------------
            #  Train Generators
            # ------------------
            optimizer_G.zero_grad()

            # Identity loss (optional but highly recommended for consistency)
            # G_A2B(B) should equal B (X-ray generator fed CT slice should output CT)
            loss_id_A = criterion_identity(G_B2A(real_A), real_A)
            loss_id_B = criterion_identity(G_A2B(real_B), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            # GAN loss (Adversarial)
            fake_B = G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(D_B(fake_B), valid)

            fake_A = G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(D_A(fake_A), valid)

            loss_GAN = (loss_GAN_A2B + loss_GAN_B2A) / 2

            # Cycle consistency loss
            rec_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(rec_A, real_A)

            rec_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(rec_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total Generator Loss
            loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_identity

            loss_G.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator A
            # ---------------------
            optimizer_D_A.zero_grad()

            # Real loss
            loss_real_A = criterion_GAN(D_A(real_A), valid)
            # Fake loss (buffered)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake_A = criterion_GAN(D_A(fake_A_.detach()), fake)

            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            # ---------------------
            #  Train Discriminator B
            # ---------------------
            optimizer_D_B.zero_grad()

            # Real loss
            loss_real_B = criterion_GAN(D_B(real_B), valid)
            # Fake loss (buffered)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake_B = criterion_GAN(D_B(fake_B_.detach()), fake)

            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            # Progress Logging
            batches_done = epoch * len(dataloader) + i
            if batches_done % 10 == 0:
                print(
                    f"[Epoch {epoch}/{args.epochs}] [Batch {i}/{len(dataloader)}] "
                    f"[D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}, gan: {loss_GAN.item():.4f}, "
                    f"cycle: {loss_cycle.item():.4f}, identity: {loss_identity.item():.4f}]"
                )

            # Save sample images grid
            if batches_done % args.sample_interval == 0:
                # Arrange side by side: [Real X-ray, Fake CT, Reconstructed X-ray] and [Real CT, Fake X-ray, Reconstructed CT]
                img_sample = torch.cat((real_A.data, fake_B.data, rec_A.data, real_B.data, fake_A.data, rec_B.data), 0)
                # Normalize back to [0, 1] for visualization
                img_sample = (img_sample + 1.0) / 2.0
                save_image(img_sample, f"samples/sample_{batches_done}.png", nrow=3, normalize=False)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # Save Checkpoint
        if (epoch + 1) % args.checkpoint_interval == 0 or (epoch + 1) == args.epochs:
            checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "G_A2B": G_A2B.state_dict(),
                "G_B2A": G_B2A.state_dict(),
                "D_A": D_A.state_dict(),
                "D_B": D_B.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D_A": optimizer_D_A.state_dict(),
                "optimizer_D_B": optimizer_D_B.state_dict(),
                "scheduler_G": lr_scheduler_G.state_dict(),
                "scheduler_D_A": lr_scheduler_D_A.state_dict(),
                "scheduler_D_B": lr_scheduler_D_B.state_dict()
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
