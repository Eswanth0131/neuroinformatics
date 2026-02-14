"""
3D Image Upsampling Model
Converts 3D images from shape (112, 138, 40) to (179, 221, 200)

This model uses a U-Net-like architecture with:
- Encoder path with 3D convolutions
- Decoder path with transposed 3D convolutions
- Skip connections for preserving spatial information
- Progressive upsampling to handle different scaling factors per dimension
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D Convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class EncoderBlock(nn.Module):
    """Encoder block with convolution and downsampling"""
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        skip = self.conv_block(x)
        x = self.pool(skip)
        return x, skip


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and convolution"""
    def __init__(self, in_channels, out_channels, scale_factor=(2, 2, 2)):
        super(DecoderBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=scale_factor, 
            stride=scale_factor
        )
        self.conv_block = ConvBlock3D(out_channels * 2, out_channels)
        
    def forward(self, x, skip):
        x = self.conv_transpose(x)
        
        # Handle size mismatches due to non-exact scaling
        if x.shape != skip.shape:
            diff_d = skip.shape[2] - x.shape[2]
            diff_h = skip.shape[3] - x.shape[3]
            diff_w = skip.shape[4] - x.shape[4]
            
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                         diff_h // 2, diff_h - diff_h // 2,
                         diff_d // 2, diff_d - diff_d // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet3DUpsampler(nn.Module):
    """
    3D U-Net architecture for upsampling from (112, 138, 40) to (179, 221, 200)
    
    Input shape: (batch_size, 1, 112, 138, 40)
    Output shape: (batch_size, 1, 179, 221, 200)
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=32):
        super(UNet3DUpsampler, self).__init__()
        
        # Encoder path
        self.encoder1 = EncoderBlock(in_channels, base_features)
        self.encoder2 = EncoderBlock(base_features, base_features * 2)
        self.encoder3 = EncoderBlock(base_features * 2, base_features * 4)
        
        # Bottleneck
        self.bottleneck = ConvBlock3D(base_features * 4, base_features * 8)
        
        # Decoder path
        self.decoder3 = DecoderBlock(base_features * 8, base_features * 4, scale_factor=(2, 2, 2))
        self.decoder2 = DecoderBlock(base_features * 4, base_features * 2, scale_factor=(2, 2, 2))
        self.decoder1 = DecoderBlock(base_features * 2, base_features, scale_factor=(2, 2, 2))
        
        # Additional upsampling layers to reach target size
        # From (56, 69, 20) -> (112, 138, 40) after decoder1
        # Need to go to (179, 221, 200)
        
        # Progressive upsampling
        self.upsample1 = nn.ConvTranspose3d(base_features, base_features // 2, 
                                            kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv_up1 = ConvBlock3D(base_features // 2, base_features // 2)
        
        # Final refinement layer
        self.upsample2 = nn.ConvTranspose3d(base_features // 2, base_features // 4,
                                            kernel_size=(2, 2, 3), stride=(1, 1, 2))
        self.conv_up2 = ConvBlock3D(base_features // 4, base_features // 4)
        
        # Output layer
        self.output_conv = nn.Conv3d(base_features // 4, out_channels, kernel_size=1)

        self.activation_function = nn.ReLU()
        
        # Target output size
        self.target_size = (179, 221, 200)
        
    def forward(self, x):
        # Encoder path
        x, skip1 = self.encoder1(x)  # skip1: (112, 138, 40)
        x, skip2 = self.encoder2(x)  # skip2: (56, 69, 20)
        x, skip3 = self.encoder3(x)  # skip3: (28, 34, 10)
        
        # Bottleneck
        x = self.bottleneck(x)  # (14, 17, 5)
        
        # Decoder path
        x = self.decoder3(x, skip3)  # (28, 34, 10)
        x = self.decoder2(x, skip2)  # (56, 69, 20)
        x = self.decoder1(x, skip1)  # (112, 138, 40)
        
        # Additional upsampling
        x = self.upsample1(x)  # (224, 276, 80)
        x = self.conv_up1(x)
        
        x = self.upsample2(x)  # Approaching target size
        x = self.conv_up2(x)
        
        # Final output
        x = self.output_conv(x)

        x = self.activation_function(x)
        
        # Resize to exact target dimensions if needed
        if x.shape[2:] != self.target_size:
            x = F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=False)
        
        return x


def test_model():
    """Test the model with random input"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = UNet3DUpsampler(in_channels=1, out_channels=1, base_features=32)
    model = model.to(device)
    
    # Create random input
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 112, 138, 40).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    print(f"Output shape: {output_tensor.shape}")
    print(f"Expected output shape: (batch_size, 1, 179, 221, 200)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Model summary
    print("\n" + "="*60)
    print("Model Architecture Summary")
    print("="*60)
    print(model)
    
    return model


if __name__ == "__main__":
    model = test_model()
