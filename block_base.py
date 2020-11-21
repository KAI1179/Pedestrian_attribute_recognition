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

# from torchviz import make_dot


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class block_base(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(block_base, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('0', nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('1', nn.BatchNorm2d(out_channels))
        self.features.add_module('2', nn.ReLU(inplace=True))
        self.features.add_module('3', nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding))
        self.features.add_module('4', nn.BatchNorm2d(out_channels))
        self.features.add_module('5', nn.ReLU(inplace=True))


    def forward(self, x):
        x = self.features(x)  # 输出 x =[b, 64, 144, 72]

        return x
        