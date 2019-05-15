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

class biLSTM(nn.Module):
    
    def __init__(self,
                 hidden_size,
                 embedding_weights,
                 percent_dropout,
                 vocab_size,
                 num_layers = 1,
                 input_size = 300):

        super(biLSTM, self).__init__()
        
        self.num_layers, self.hidden_size = num_layers, hidden_size
        
        self.embed_table = torch.from_numpy(embedding_weights).float()
        self.embedding = nn.Embedding.from_pretrained(self.embed_table)
        self.dropout = percent_dropout
        self.drop_out = nn.Dropout(self.dropout)
        self.LSTM = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        if self.LSTM.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
            
        self.bn = nn.BatchNorm1d(self.hidden_size * self.num_directions)
        
    def init_hidden(self, batch_size):
        hidden = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, c_0
    
    def forward(self, sentence, mask, lengths):
    	# sort - unsort batch
        sort_original = sorted(range(len(lengths)), key=lambda sentence: -lengths[sentence])
        unsort_to_original = sorted(range(len(lengths)), key=lambda sentence: sort_original[sentence])
        sentence = sentence[sort_original]
        _mask = mask[sort_original]
        lengths = lengths[sort_original]
        batch_size, seq_len = sentence.size()
        self.hidden, self.c_0 = self.init_hidden(batch_size)
        
        # embdddings
        embeds = self.embedding(sentence)
        embeds = mask*embeds + (1-_mask)*embeds.clone().detach()
        embeds = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_out, (self.hidden_1, self.c_1) = self.LSTM(embeds, (self.hidden, self.c_0))
        emb1, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        emb1 = emb1.view(batch_size, -1, 2, self.hidden_size)
        emb1 = torch.max(emb1, dim = 1)[0]
        emb1 = torch.cat([emb1[:,i,:] for i in range(self.num_directions)], dim=1)
        emb1 = emb1[unsort_to_original]
        out = self.bn(emb1)

        return out
