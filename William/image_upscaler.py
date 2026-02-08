import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from main import *
PATH = './'
class UpsampleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        
        # Resize to exact target size
        x = F.interpolate(x, size=(179, 221), mode='bicubic', align_corners=False)
        
        x = self.decoder(x)
        return x

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
        # self.conv_output_sized = nn.Conv2d(base_channels, num_channels, kernel_size=9, padding=4)
        
    def forward(self, x):
        # Store input for final skip connection
        input_upsampled = F.interpolate(x, scale_factor=self.scale_factor, 
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
        
        # Add bicubic upsampled input (helps with training)
        out += input_upsampled
        out = F.interpolate(out, size=(179, 221), mode='bicubic', align_corners=False)
        
        return out

def train_loop(dataloader, device, model, loss_fn, optimizer, batch_size=10, ):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        # x_float = X.to(torch.float)
        # print(x_float.shape, y.shape)
        # pred = model(x_float)
        pred = model(X.to(device))
        # print(x_float.shape, pred.shape, y.shape)
        # loss = loss_fn(pred, y.to(torch.float))
        loss = loss_fn(pred, y.to(device))
        

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, device, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.to(device))
            test_loss += loss_fn(pred, y.to(device)).item()
            # correct += (pred.argmax(1) == y.to(device)).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Example usage and training setup
if __name__ == "__main__":
    # Create model
    model = ImageUpscaler(scale_factor=2, num_channels=1, num_residual_blocks=8, base_channels=64)
    # model = UpsampleCNN()
    
    # Example input (batch_size=4, channels=3, height=64, width=64)
    # input_image = torch.randn(4, 3, 64, 64)
    dataset = pd.read_pickle('./data/misc/data.pkl')
    input_image = dataset.loc[0, 'img_l']
    train_dataset = myDataset(dataset)
    test_dataset = myDataset(dataset, 'test')

    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    # Forward pass
    # output_image = model(input_image)
    # print(f"Input shape: {input_image.shape}")
    # print(f"Output shape: {output_image.shape}")
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup example
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss function (common choices for super-resolution)
    criterion = nn.MSELoss()  # or nn.L1Loss()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop example
    # model.train()
    # for epoch in range(1):  # Just one example epoch
    #     for i, data in enumerate(train_loader, 0):
    #         # Assuming you have low_res and high_res images
    #         low_res = data[0].to(device)
    #         high_res = torch.randn(4, 3, 128, 128).to(device)  # 2x upscaled
            
    #         # Forward pass
    #         output = model(low_res)
    #         loss = criterion(output, high_res)
            
    #         # Backward pass
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
            
    #     print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
    
    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_loader, device, model, criterion, optimizer)
        test_loop(test_loader, device, model, criterion)
    print("Done!")
    torch.save(model.state_dict(), os.path.join(PATH, 'model1.pt'))

    # # Inference example
    # model.eval()
    # with torch.no_grad():
    #     test_input = torch.randn(1, 3, 128, 128).to(device)
    #     upscaled = model(test_input)
    #     print(f"\nInference - Input: {test_input.shape}, Output: {upscaled.shape}")
