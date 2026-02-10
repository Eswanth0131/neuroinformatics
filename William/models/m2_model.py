import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        out = self.relu(out)
        return out


class UpsampleBlock(nn.Module):
    """Upsampling block using pixel shuffle (sub-pixel convolution)"""
    def __init__(self, in_channels, upscale_factor):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2), 
                              kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.relu(x)
        return x


class ImageUpscaler(nn.Module):
    """
    Super-Resolution CNN for upscaling images
    
    Args:
        scale_factor: Upscaling factor (2, 4, or 8)
        num_channels: Number of input/output channels (3 for RGB, 1 for grayscale)
        num_residual_blocks: Number of residual blocks (default: 16)
        base_channels: Number of feature channels (default: 64)
    """
    def __init__(self, scale_factor=2, num_channels=1, num_residual_blocks=16, base_channels=64):
        super(ImageUpscaler, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Initial feature extraction
        self.conv_input = nn.Conv2d(num_channels, base_channels, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )
        
        # Middle convolution
        self.conv_mid = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn_mid = nn.BatchNorm2d(base_channels)
        
        # Upsampling blocks
        upsample_blocks = []
        if scale_factor == 2 or scale_factor == 4 or scale_factor == 8:
            num_upsample = int(scale_factor / 2)
            for _ in range(num_upsample):
                upsample_blocks.append(UpsampleBlock(base_channels, 2))
        else:
            raise ValueError("Scale factor must be 2, 4, or 8")
        
        self.upsample_blocks = nn.Sequential(*upsample_blocks)
        
        # Output convolution
        self.conv_output = nn.Conv2d(base_channels, num_channels, kernel_size=9, padding=4)

        # fix output size
        self.conv_output_sized = nn.Conv2d(base_channels, num_channels, kernel_size=9, padding=4)
        
    def forward(self, x):
        # Store input for final skip connection
        input_upsampled = F.interpolate(x, size=(179, 221), 
                                    mode='bicubic', align_corners=False)
        
        # Initial feature extraction
        out = self.relu(self.conv_input(x))
        residual = out
        
        # Residual blocks
        out = self.residual_blocks(out)
        
        # Middle convolution with skip connection
        out = self.bn_mid(self.conv_mid(out))
        out += residual
        
        # Upsampling
        out = self.upsample_blocks(out)
        
        # Final output
        out = self.conv_output(out)
        # out = self.conv_output_sized(out)
        print(out.shape)
        print(input_upsampled.shape)
        # Add bicubic upsampled input (helps with training)
        out += input_upsampled
        
        return out