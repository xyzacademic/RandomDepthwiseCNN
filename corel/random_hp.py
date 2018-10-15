#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 15:11:14 2018

@author: xyz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np


def _weights_init(m):
    #    classname = m.__class__.__name__
    #    print(classname)
    if isinstance(m, nn.Conv2d):
        state = 'normal'
        if state == 'normal':
            init.normal_(m.weight, 0, 1)
            init.normal_(m.bias, 0, 1)
        elif state == 'uniform':
            init.uniform_(m.weight, -1, 1)
            init.uniform_(m.bias, -1, 1)
    elif isinstance(m, nn.Linear):
        init.normal_(m.weight, std=0.01)
        init.normal_(m.bias, std=0.01)
        # init.xavier_normal_(m.weight)


class Block(nn.Module):

    def __init__(self, n, k, bias=True):
        super(Block, self).__init__()
        self.pool = nn.AvgPool2d(2,2,0)
        self.conv = nn.Conv2d(n, n, k, groups=n,
                              padding=(k-1)//2,
                              bias=bias
                              )
        self.conv2 = nn.Conv2d(n, n, k, groups=n,
                              padding=(k-1)//2,
                               bias=bias
                              )
        # self.conv3 = nn.Conv2d(n, n, k, groups=n,
        #                        padding=(k - 1) // 2,
        #                       )
        self.apply(_weights_init)

    def forward(self, x):
        out = self.pool(x)
        out = self.conv(out).sign_()
        out = self.conv2(out).sign_()

        return out


class Rh(nn.Module):

    def __init__(self, num_layers=3, kernel_size=3, n=1000, block=Block):
        super(Rh, self).__init__()
        self.kernel_size = kernel_size
        self.n = n
        self.num_layers = num_layers

        self.conv = nn.Conv2d(3, self.n, self.kernel_size,
                              padding=1,
                              )
        self.conv2 = self._make_layer(block, num_layers)

        self.apply(_weights_init)

    def _make_layer(self, block, num_layers):
        layers = []
        for i in range(num_layers):
            layers.append(block(self.n, self.kernel_size))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x).sign_()
        out = self.conv2(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        return out



if __name__ == '__main__':
    x = torch.randn(2, 3, 32, 32)
    net = Rh()
    y = net(Variable(x))
    print(y.size())