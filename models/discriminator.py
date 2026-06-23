import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator (70x70 receptive field).
    Outputs a grid of predictions (each cell represents a patch realism score).
    """
    def __init__(self, input_nc=3, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1: C64 (no InstanceNorm)
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: C128
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: C256
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: C512 (stride 1)
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output Layer: 1 channel output, stride 1
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Test discriminator shape
    x = torch.randn(1, 3, 256, 256)
    disc = PatchGANDiscriminator()
    out = disc(x)
    print("Discriminator output shape:", out.shape)
