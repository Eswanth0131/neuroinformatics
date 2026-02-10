"""
Following this paper model, SRCNN: https://arxiv.org/pdf/1501.00092
Assumptions: 
  n is the number of output channels
  c is the number of input channels
  k is ther kernal size aka f_1 or filter size
SRCNN is just that, a simple CNN, 
  NO attention
  NO skip
  NO pooling
  NO none ReLU functions
Initial implementaiton has ouput image size smaller than input image size, they simply compared the center of input image to output image
Difference in setup and loss calculation:
  They take an image, apply blur, then the output image is smaller than the input image. This means that they take a subset of the input image for testing. For our class project set up, we take a subset of the target image instead!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class patch(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super(patch, self).__init__()
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.c1(x)
        out = self.relu(out)
        return out

class mapping(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super(mapping, self).__init__()
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.c1(x)
        out = self.relu(out)
        return out

class reconstruction(nn.Module):
    def __init__(self, in_channels, out_channels, k_size):
        super(reconstruction, self).__init__()
        self.c1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
        )
    def forward(self, x):
        out = self.c1(x)
        return out

class SRCNN(nn.Module):
    def __init__(self, f1=9, f2=1, f3=5, n1=64, n2=32, img_channel=1):
        super(SRCNN, self).__init__()
        self.part1 = patch(img_channel, n1, f1)
        self.part2 = mapping(n1, n2, f2)
        self.part3 = reconstruction(n2, img_channel, f3)

    def forward(self, x):
        out = F.interpolate(x, size=(179, 221), mode='bicubic')
        out = self.part1(out)
        out = self.part2(out)
        out = self.part3(out)
        return out

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
