# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib

import torch
import torch.nn as nn
from datafolder.folder import Train_Dataset
import torchvision.models as models
import torch
import torchvision
import torch.nn as nn
from torch.nn import init



class block_mid_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(block_mid_1, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('7', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('8', nn.BatchNorm2d(out_channels))
        self.features.add_module('9', nn.ReLU(inplace=True))
        self.features.add_module('10', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('11', nn.BatchNorm2d(out_channels))
        self.features.add_module('12', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)  # 输出 x =[b, 64, 144, 72]

        return x

class block_mid_2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(block_mid_2, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('14', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('15', nn.BatchNorm2d(out_channels))
        self.features.add_module('16', nn.ReLU(inplace=True))
        self.features.add_module('17', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('18', nn.BatchNorm2d(out_channels))
        self.features.add_module('19', nn.ReLU(inplace=True))
        self.features.add_module('20', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('21', nn.BatchNorm2d(out_channels))
        self.features.add_module('22', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)

        return x

