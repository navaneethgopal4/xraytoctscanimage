import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    """
    Standard Residual block with Reflection Padding and Instance Normalization.
    """
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0, bias=False),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    """
    Standard ResNet Generator for CycleGAN.
    Contains downsampling blocks, ResNet blocks, and upsampling blocks.
    Default: 9 ResNet blocks for 256x256 images.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(ResNetGenerator, self).__init__()
        
        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling blocks
        in_features = ngf
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            
        # Residual blocks
        for _ in range(n_blocks):
            model += [ResNetBlock(in_features)]
            
        # Upsampling blocks
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Test generator shape
    x = torch.randn(1, 3, 256, 256)
    gen = ResNetGenerator()
    out = gen(x)
    print("Generator output shape:", out.shape)
