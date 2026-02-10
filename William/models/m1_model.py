import torch
import torch.nn as nn
import torch.nn.functional as F

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