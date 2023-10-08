import torch
import torch.nn as nn
import torch.nn.functional as func

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, activation="relu", dimension="2d",norm="g"):
        super(ConvBlock, self).__init__()
        if dimension=="2d":
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            if norm=="g":
                self.norm = nn.GroupNorm(max(1, out_channels // 8),out_channels)
            elif norm=="b":
                self.norm = nn.BatchNorm2d(out_channels)
        elif dimension=="3d":
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
            self.norm = nn.BatchNorm3d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.norm(self.conv(x))
        if self.activation == 'relu':
            x = func.relu(x,inplace=True)
        elif self.activation == 'leaky_relu':
            x = func.leaky_relu(x,0.2,inplace=True)
        return x

