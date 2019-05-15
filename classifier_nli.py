import random
import os
import re
import time
import functools
import numpy as np
import pandas as pd
from setuptools import setup
from collections import Counter
from collections import defaultdict
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable

class Linear_Layers(nn.Module):
    
    def __init__(self, hidden_size, hidden_size_2, percent_dropout, classes = 3, input_size = 300):
        
        super(Linear_Layers, self).__init__()
        self.num_classes = classes
        self.hidden_size = hidden_size
        self.hidden_size_2 = hidden_size_2
        self.percent_dropout = percent_dropout
        self.num_classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_size, self.hidden_size_2),
            nn.LeakyReLU(0.05),
            nn.Dropout(p=self.percent_dropout),
            nn.Linear(self.hidden_size_2, self.num_classes))
        self.init_weights()
        
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.uniform_(module.bias)

    def forward(self, lstm_out_1, lstm_out_2):
		hidden = torch.cat([lstm_out_1, lstm_out_2, 
			torch.abs(lstm_out_1 - lstm_out_2), 
			torch.mul(lstm_out_1, lstm_out_2)], dim=1)
        hidden = hidden.view(hidden.size(0),-1) 
        out = self.mlp(hidden)
        out = F.log_softmax(out, 1)
        return out
