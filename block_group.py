# !/usr/local/bin/python3
import os
import time
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datafolder.folder import Train_Dataset
import torchvision.models as models
import torch
import torchvision
import torch.nn as nn
from torch.nn import init



class block_group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(block_group, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('24', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('25', nn.BatchNorm2d(out_channels))
        self.features.add_module('26', nn.ReLU(inplace=True))
        self.features.add_module('27', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('28', nn.BatchNorm2d(out_channels))
        self.features.add_module('29', nn.ReLU(inplace=True))
        self.features.add_module('30', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('31', nn.BatchNorm2d(out_channels))
        self.features.add_module('32', nn.ReLU(inplace=True))
        self.features.add_module('33', nn.MaxPool2d(2))
        self.features.add_module('34', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('35', nn.BatchNorm2d(out_channels))
        self.features.add_module('36', nn.ReLU(inplace=True))
        # self.features.add_module('37', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        # self.features.add_module('38', nn.BatchNorm2d(out_channels))
        # self.features.add_module('39', nn.ReLU(inplace=True))
        # self.features.add_module('40', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        # self.features.add_module('41', nn.BatchNorm2d(out_channels))
        # self.features.add_module('42', nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.features(x)  # 输出 x =[b, 64, 144, 72]

        return x

