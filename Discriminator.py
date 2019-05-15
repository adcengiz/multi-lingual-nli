import io
import os
import re
import time
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable

class Discriminator(nn.Module):

    def __init__(self, n_langs, dis_layers, dis_hidden_dim, dis_dropout, lr_slope):

        super(Discriminator, self).__init__()
        self.n_langs = n_langs
        self.input_dim = config.hidden_dim * self.n_langs
        self.dis_layers = dis_layers
        self.dis_hidden_dim = dis_hidden_dim
        self.dis_dropout = dis_dropout
        self.lr_slope = lr_slope

        layers = []
        for i in range(self.dis_layers + 1):
            if i == 0:
                input_dim = self.input_dim
            else:
                input_dim = self.dis_hidden_dim
            output_dim = self.dis_hidden_dim if i < self.dis_layers else self.n_langs
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(self.lr_slope))
                layers.append(nn.Dropout(self.dis_dropout))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        out = F.log_softmax(out, 1)
        return out
