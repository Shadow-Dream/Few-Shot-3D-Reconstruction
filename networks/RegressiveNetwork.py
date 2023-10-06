import torch
import torch.nn as nn
import torch.nn.functional as func

class RegressiveNetwork(nn.Module):
    def __init__(self,channels,kernel_size = 3):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, kernel_size, 1, (kernel_size-1)//2)
    
    def forward(self,feature):
        feature = self.conv(feature)
        return feature