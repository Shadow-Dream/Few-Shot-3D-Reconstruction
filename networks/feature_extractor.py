import torch
import torch.nn as nn
import torch.nn.functional as func
from networks.ConvBlock import ConvBlock
from networks.TransConvBlock import TransConvBlock

class FeatureExtractor(nn.Module):
    def __init__(self,input_channels = 3,output_channels = 32):
        super(FeatureExtractor, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        channels = output_channels // 4

        self.conv0_0 = ConvBlock(input_channels, channels, stride=1)
        self.conv1_0 = ConvBlock(channels * 1, channels * 2, stride=2)
        self.conv2_0 = ConvBlock(channels * 2, channels * 4, stride=2)
        self.conv3_0 = ConvBlock(channels * 4, channels * 8, stride=2)

        self.conv0_1 = ConvBlock(channels * 1, channels * 1, stride=1)
        self.conv1_1 = ConvBlock(channels * 2, channels * 2, stride=1)
        self.conv2_1 = ConvBlock(channels * 4, channels * 4, stride=1)
        self.conv3_1 = ConvBlock(channels * 8, channels * 8, stride=1)

        self.conv6_0 = TransConvBlock(channels * 8, channels * 4, stride=2)
        self.conv6_1 = ConvBlock(channels * 8, channels * 4, stride=1)
        self.conv6_2 = ConvBlock(channels * 4, channels * 4, stride=1)

        self.conv7_0 = TransConvBlock(channels * 4, channels * 2, stride=2)
        self.conv7_1 = ConvBlock(channels * 4, channels * 2, stride=1)
        self.conv7_2 = ConvBlock(channels * 2, channels * 2, stride=1)

        self.conv8_0 = TransConvBlock(channels * 2, channels, stride=2)
        self.conv8_1 = ConvBlock(channels * 2, channels, stride=1)
        self.conv8_2 = ConvBlock(channels * 1, channels, stride=1)

        self.conv9_0 = ConvBlock(channels, channels * 2, stride = 2, kernel_size=5, padding=2)
        self.conv9_1 = ConvBlock(channels * 2, channels * 2, stride = 1)

        self.conv10_0 = ConvBlock(channels * 2, channels * 4, stride = 2, kernel_size = 5, padding = 2)
        self.conv10_1 = ConvBlock(channels * 4, channels * 4, stride = 1)
        self.conv10_2 = nn.Conv2d(channels * 4, channels * 4, 3, 1, 1, bias=False)

    def forward(self, x):
        f0_0 = self.conv0_0(x)
        f1_0 = self.conv1_0(f0_0)
        f2_0 = self.conv2_0(f1_0)
        f3_0 = self.conv3_0(f2_0)

        f0_1 = self.conv0_1(f0_0)
        f1_1 = self.conv1_1(f1_0)
        f2_1 = self.conv2_1(f2_0)
        f3_1 = self.conv3_1(f3_0)

        f6_0 = self.conv6_0(f3_1)
        f6_0 = func.pad(f6_0, (0, 1, 0, 1))
        cat6_0 = torch.cat([f6_0, f2_1], dim=1)
        f6_1 = self.conv6_1(cat6_0)
        f6_2 = self.conv6_2(f6_1)

        f7_0 = self.conv7_0(f6_2)
        f7_0 = func.pad(f7_0, (0, 1, 0, 1))
        cat7_0 = torch.cat([f7_0, f1_1], dim=1)
        f7_1 = self.conv7_1(cat7_0)
        f7_2 = self.conv7_2(f7_1)

        f8_0 = self.conv8_0(f7_2)
        f8_0 = func.pad(f8_0, (0, 1, 0, 1))
        cat8_0 = torch.cat([f8_0, f0_1], dim=1)
        f8_1 = self.conv8_1(cat8_0)
        f8_2 = self.conv8_2(f8_1)

        f9_0 = self.conv9_0(f8_2)
        f9_1 = self.conv9_1(f9_0)
        f10_0 = self.conv10_0(f9_1)
        f10_1 = self.conv10_1(f10_0)
        f10_2 = self.conv10_2(f10_1)
        return f10_2