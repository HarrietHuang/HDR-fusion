import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy
import torch.optim as optim
def init_net(net, init_type='orthogonal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

import torch
from torch import nn

# class TrainableEltwiseLayer(nn.Module)
#   def __init__(self, n, h, w):
#     super(TrainableEltwiseLayer, self).__init__()
#     self.weights = nn.Parameter(torch.Tensor(1, n, h, w))  # define the trainable parameter

#   def forward(self, x):
#     # assuming x is of size b-1-h-w
#     return x * self.weights  # element-wise multiplication

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.inc = in_channels
        self.l_conv1 = nn.Conv2d(in_channels,16 , kernel_size=5, stride=1, padding=2)
        # self.l_conv3 = nn.Conv2d(1,16 , kernel_size=3, stride=1, padding=1)

        self.l_conv11 = nn.Conv2d(16, 32, kernel_size=7, stride=1, padding=3)
        # self.l_conv13 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        # self.l_conv40 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=1)
        # self.l_conv4 = nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=1)
        # self.l_conv41 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=1)
        # self.l_conv5 = nn.Conv2d(16, 1, kernel_size=7, stride=1, padding=1)
    def forward(self,x1):
        # print('inc ',self.inc)
        # print(x1.shape)
        x1 = self.l_conv1(x1)
        # print(x1.shape)
        # x2 = self.l_conv3(x1)

        x1 = self.l_conv11(x1)
        # print(x1.shape)
        # x2 = self.l_conv13(x2)

        return x1

class Decoder(nn.Module):
    def __init__(self,in_channels):
        super(Decoder, self).__init__()
        self.l_conv1 = nn.Conv2d(32,32 , kernel_size=7, stride=1, padding=3)
        self.l_conv2 = nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2)
        self.l_conv3 = nn.Conv2d(16, in_channels, kernel_size=5, stride=1, padding=2)


    def forward(self, x):
        x = self.l_conv1(x)
        x = self.l_conv2(x)
        x = self.l_conv3(x)

        return x

class simple_model(nn.Module):
    def __init__(self):
        super(simple_model, self).__init__()
        self.enc1_y =Encoder(in_channels=1)
        self.enc2_y =Encoder(in_channels=1)
        self.dec_y =Decoder(in_channels=1)

        self.enc1_cb =Encoder(in_channels=1)
        self.enc2_cb =Encoder(in_channels=1)
        self.dec_cb =Decoder(in_channels=1)

        self.enc1_cr =Encoder(in_channels=1)
        self.enc2_cr =Encoder(in_channels=1)
        self.dec_cr =Decoder(in_channels=1)


    def forward(self, y1,y2,cb1,cb2,cr1,cr2):
        y1 = self.enc1_y(y1)
        y2 = self.enc2_y(y2)
        y = self.dec_y(y1+y2)

        cb1 = self.enc1_cb(cb1)
        cb2 = self.enc2_cb(cb2)
        cb = self.dec_cb(cb1+cb2)

        cr1 = self.enc1_cr(cr1)
        cr2 = self.enc2_cr(cr2)
        cr = self.dec_cr(cr1+cr2)
        return y, cb, cr
# if __name__ =='__main__':
#     in1 = torch.ones(1,1,512,512).cuda()
#     in2 = torch.ones(1,1,512,512).cuda()
#     bas = simple_model(in_channels=1).cuda()
#     print(bas)
#     print(bas(in1,in2))
#     # skip = net(in1,in2)
# #     print(skip['fir_pool1'])
#     print(bas(in1,in2).shape)
#     summary(bas(in1,skip), (3, 512, 512))