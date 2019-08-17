import pickle
import random
import spacy
import errno
import glob
import string
import os
import jieba
import nltk
import functools
import numpy as np
import pandas as pd
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

class XNLIconfig:
    def __init__(self, val_test_lang, max_sent_len, max_vocab_size, epochs, batch_size,
                 embed_dim, hidden_dim, dropout, lr, experiment_lang):
        self.val_test_lang = val_test_lang
        self.max_sent_len = max_sent_len
        self.max_vocab_size = max_vocab_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lr = lr
        self.experiment_lang = experiment_lang
        
config = XNLIconfig(val_test_lang = "de", max_sent_len = 50, max_vocab_size = 100000,
                    epochs = 15, batch_size = 256, embed_dim = 300, hidden_dim = 512, dropout = 0.1, lr = 1e-3,
                    experiment_lang = "de")

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
