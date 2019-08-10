#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:46:25 2019

@author: gaoyi
"""

from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch

def reparameterization(mu, logvar):
    Tensor = torch.cuda.FloatTensor 
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), 2))))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(14469, 512*4),
            nn.SELU(),
            nn.Dropout(0.5),
            nn.Linear(512*4, 512*4),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512*4),
            nn.SELU()
        )

        self.mu = nn.Linear(512*4, 2)
        self.logvar = nn.Linear(512*4,2)

    def forward(self, img):
        x = self.model(img) 
        mu = self.mu(x)
        logvar = self.logvar(x)
        z = reparameterization(mu, logvar)
        return z

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2+6373, 512*2),
            nn.BatchNorm1d(512*2),
            nn.SELU(),
            nn.Linear(512*2, 512*2),
            nn.BatchNorm1d(512*2),
            nn.Linear(512*2, 512*2),
            nn.BatchNorm1d(512*2),
            nn.Linear(512*2, 512*2),
            nn.BatchNorm1d(512*2),
            nn.SELU(),
            nn.Linear(512*2, 512*2),
            nn.BatchNorm1d(512*2),
            nn.SELU(),
            nn.Linear(512*2, 512*2),
            nn.BatchNorm1d(512*2),
            nn.SELU(),
            nn.Linear(512*2,14469),
            
        )

    def forward(self, z):
#        img_flat = self.model(z)
#        img = img_flat.view(img_flat.shape[0], *img_shape)
        img = self.model(z)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2+5, 512*4),
            nn.Dropout(0.5),
            nn.SELU(),
            nn.Linear(512*4, 256*4),
            nn.Dropout(0.5),
            nn.SELU(),
            nn.Linear(256*4, 1),
            nn.Dropout(0.25),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity