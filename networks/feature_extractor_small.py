import torch
from torch import nn
import torch.nn.functional as func
from networks.conv_block import ConvBlock
from networks.trans_conv_block import TransConvBlock

class FeatureExtractorSmall(nn.Module):
    def __init__(self):
        super(FeatureExtractorSmall, self).__init__()
        self.inplanes = 32
        
        self.conv0 = ConvBlock(3, 8, 3, 1, 1,activation="relu",norm='b')
        self.conv1 = ConvBlock(8, 8, 3, 1, 1,activation="relu",norm='b')

        self.conv2 = ConvBlock(8, 16, 5, 2, 2,activation="relu",norm='b')
        self.conv3 = ConvBlock(16, 16, 3, 1, 1,activation="relu",norm='b')
        self.conv4 = ConvBlock(16, 16, 3, 1, 1,activation="relu",norm='b')

        self.conv5 = ConvBlock(16, 32, 5, 2, 2,activation="relu",norm='b')
        self.conv6 = ConvBlock(32, 32, 3, 1, 1,activation="relu",norm='b')
        self.conv7 = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        return x