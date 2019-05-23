import pickle
import random
import spacy
import csv
import sys
import errno
import glob
import string
import io
import os
import jieba
import re
import nltk
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

# src: always English
def train(LSTM_src, LSTM_trg, discriminator, loader, contrastive_loader, optimizer, dis_optim, epoch):
    LSTM_src.train()
    LSTM_trg.train()
    discriminator.train()
    total_loss = 0
    for batch_idx, ([src_data, src_mask, src_len, trg_data, trg_mask, trg_len],
                    [src_c, src_mc, src_len_c, trg_c, trg_mc, trg_len_c]) in \
        enumerate(zip(loader, contrastive_loader)):
        
        src_data, src_mask = src_data.to(device), src_mask.to(device)
        trg_data, trg_mask = trg_data.to(device), trg_mask.to(device)
        src_c, src_mc = src_c.to(device), src_mc.to(device)
        trg_c, trg_mc = trg_c.to(device), trg_mc.to(device)
        optimizer.zero_grad()
        dis_optim.zero_grad()
        if np.random.random() <= 0.2:
            src_data = src_data + torch.rand(src_data.size()).long().to(device)
            trg_data = trg_data + torch.rand(trg_data.size()).long().to(device)
#             src_c = src_c + torch.rand(src_c.size()).long().to(device)
#             trg_c = trg_c + torch.rand(trg_c.size()).long().to(device)
            
        src_out = LSTM_src(src_data, src_mask, src_len)
        trg_out = LSTM_trg(trg_data, trg_mask, trg_len)
        src_c_out = LSTM_src(src_c, src_mc, src_len_c)
        trg_c_out = LSTM_trg(trg_c, trg_mc, trg_len_c)
        loss = loss_align(src_out, trg_out, src_c_out, trg_c_out, 0.25)
        loss.cuda().backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item() * len(src_data) / len(loader.dataset)
        if (batch_idx+1) % (len(loader.dataset)//(50*config.batch_size)) == 0:
            
            dis_labels_src = torch.zeros(config.batch_size).long()
            dis_labels_trg = torch.ones(config.batch_size).long()
            dis_labels = torch.cat([dis_labels_src, dis_labels_trg], 0)
            idx = torch.randperm(config.batch_size * 2)
            dis_input = torch.cat([src_out, trg_out], 0)
            dis_input = dis_input[idx]
            dis_labels = dis_labels[idx].to(device)
            dis_out = discriminator(dis_input)
            dis_criterion = nn.NLLLoss()
            dis_loss = dis_criterion(dis_out, dis_labels)
            dis_loss.cuda().backward(retain_graph=True)
            dis_optim.step()
            
            loss += (-1) * dis_criterion(dis_out, dis_labels)
            loss.cuda().backward()
            optimizer.step()
            
            torch.save(LSTM_trg.state_dict(), "LSTM_en_{}_{}_epoch_{}".format(config.experiment_lang,
                                                                      config.experiment_lang.upper(),
                                                                      epoch))
            
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(
                epoch, (batch_idx+1) * config.batch_size, len(loader.dataset),
                100. * (batch_idx+1) / len(loader), loss.item()))
            
    optimizer.zero_grad()
    return total_loss

