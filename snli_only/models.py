import pickle
import random
import random
import spacy
import csv
import sys
import errno
import glob
import string
import io
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

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
seed = 1
device = torch.device("cuda" if cuda else "cpu")

class biLSTM(nn.Module):
    
    def __init__(self, hidden_size, embedding_weights, percent_dropout, vocab_size, num_layers=1, input_size=300):

        super(biLSTM, self).__init__()
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.embed_table = torch.from_numpy(embedding_weights).float()
        self.embedding = nn.Embedding.from_pretrained(self.embed_table)
        self.drop_out = nn.Dropout(percent_dropout)
        self.LSTM = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.num_directions = 2 if self.LSTM.bidirectional else 1
        
    def init_hidden(self, batch_size):
        hidden = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        return hidden, c_0
    
    def forward(self, sentence, mask, lengths):
        sort_original = sorted(range(len(lengths)), key=lambda sentence: -lengths[sentence])
        unsort_to_original = sorted(range(len(lengths)), key=lambda sentence: sort_original[sentence])
        sentence = sentence[sort_original]
        _mask = mask[sort_original]
        lengths = lengths[sort_original]

        batch_size, seq_len = sentence.size()
        self.hidden, self.c_0 = self.init_hidden(batch_size)
        embeds = self.embedding(sentence)
        embeds = mask*embeds + (1-_mask)*embeds.clone().detach()
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)

        lstm_out, (self.hidden_1, self.c_1) = self.LSTM(embeds, (self.hidden, self.c_0))
        emb1, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        emb1 = emb1.view(batch_size, -1, 2, self.hidden_size)
        emb1 = torch.max(emb1, dim=1)[0]
        emb1 = torch.cat([emb1[:,i,:] for i in range(self.num_directions)], dim=1)[unsort_to_original]
        return emb1

class Linear_Layers(nn.Module):
    
    def __init__(self, hidden_size, hidden_size_2, percent_dropout, classes=3, input_size=300):
        
        super(Linear_Layers, self).__init__()
        self.num_classes = classes
        self.hidden_size, self.hidden_size_2 = hidden_size, hidden_size_2
        self.percent_dropout = percent_dropout
        self.num_classes = classes
        self.mlp = nn.Sequential(
            nn.Linear(4 * self.hidden_size, self.hidden_size_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.percent_dropout),
            nn.Linear(self.hidden_size_2, self.num_classes))
        self.init_weights()
        
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.uniform_(module.bias)

    def forward(self, out_1, out_2):
        # concat(p, h, |p - h|, p * h)
        hidden = torch.cat([out_1, out_2, torch.abs(out_1 - out_2), torch.mul(out_1, out_2)], dim=1)
        hidden = hidden.view(hidden.size(0),-1) 
        out = self.mlp(hidden)
        out = F.log_softmax(out, 1)
        return out
