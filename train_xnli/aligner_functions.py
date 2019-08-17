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
from models import *
from preprocess import *
from Discriminator import *

PAD_IDX = 0
UNK_IDX = 1
label_dict = {"entailment":0, "neutral":1, "contradiction":2}
# opus_path = "data/translation/opus"
europarl_path = "../data/parallel_corpora/europarl"
# un_path = "data/translation/un_parallel_corpora"
snli_path = "../data/nli_corpora/snli/"
align_path = "../data/vecs/wiki.{}.align.vec"
multi_path = "../data/vecs/cc.{}.vec.gz"
multinli_path = "../data/nli_corpora/multinli/multinli_1.0"

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
seed = 1
device = torch.device("cuda" if cuda else "cpu")

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

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def read_xnli(lang):
    fname = "../data/nli_corpora/xnli/xnli.{}.jsonl"
    xnli_dev = pd.read_json(fname.format("dev"), lines=True)
    xnli_test = pd.read_json(fname.format("test"), lines=True)
    if lang == "all":
        dev_data = xnli_dev
        test_data = xnli_test
    else:
        dev_data = xnli_dev[xnli_dev["language"]==lang]
        test_data = xnli_test[xnli_test["language"]==lang]
    return dev_data, test_data

def build_vocab(all_tokens, max_vocab_size):
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2tok = [*vocab]
    tok2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2tok = ["<PAD>", "<UNK>"] + id2tok
    tok2id["<PAD>"], tok2id["<UNK>"] = PAD_IDX, UNK_IDX
    return token2id, id2token

def loss_align(en_rep, target_rep, en_c, target_c, lambda_reg):
    """:param en_rep: output repr of eng encoder (batch_size, hidden_size)
       :param target_rep: output repr of target encoder (batch_size, hidden_size)
       :param en_c: contrastive sentence repr from eng encoder (batch_size, hidden_size)
       :param target_c: contrastive sentence repr form target encoder (batch_size, hidden_size)
       :param lambda_reg: regularization coef [default: 0.25]
        
        Returns: L_align = l2norm (en_rep, target_rep) - lambda_reg( l2norm (en_c, target_rep) + l2norm (en_rep, target_c))
        """
    dist = torch.norm(en_rep - target_rep, 2)
    c_dist = torch.norm(en_c - target_rep, 2) + torch.norm(en_rep - target_c, 2)
    L_align = dist - lambda_reg*(c_dist)
    return L_align

class AlignDataset(Dataset):
    def __init__(self, data, max_sent_len, src_lang, trg_lang,
                 token2id, id2token):
        self.src = [*data["{}_tokenized".format(src_lang)].values]
        self.trg = [*data["{}_tokenized".format(trg_lang)].values]
        self.max_sent_len = int(max_sent_len)
        self.token2id, self.id2token = token2id, id2token
    
    def __len__(self):
        return len(self.src)
    
    def __getitem__(self, row):
        src_ix, trg_ix = [], []
        src_mask, trg_mask = [], []
        for w in self.src[row][:self.max_sent_len]:
            if w in self.token2id.keys():
                src_ix.append(self.token2id[w])
                src_mask.append(0)
            else:
                src_ix.append(UNK_IDX)
                src_mask.append(1)
        for w in self.trg[row][:self.max_sent_len]:
            if w in self.token2id.keys():
                trg_ix.append(self.token2id[w])
                trg_mask.append(0)
            else:
                trg_ix.append(UNK_IDX)
                trg_mask.append(1)
        
        src_list = [src_ix, src_mask, len(src_ix)]
        trg_list = [trg_ix, trg_mask, len(trg_mask)]
        return src_list + trg_list

def align_collate_func(batch, max_sent_len):
    src_data, trg_data = [], []
    src_mask, trg_mask = [], []
    src_len, trg_len = [], []
    
    for datum in batch:
        src_len.append(datum[2])
        trg_len.append(datum[5])
        src_data_padded = np.pad(np.array(datum[0]), pad_width=((0, max_sent_len-datum[2])), mode="constant", constant_values=PAD_IDX)
        src_data.append(src_data_padded)
        src_mask_padded = np.pad(np.array(datum[1]), pad_width=((0, max_sent_len-datum[2])), mode="constant", constant_values=PAD_IDX)
        src_mask.append(src_mask_padded)
        trg_data_padded = np.pad(np.array(datum[3]), pad_width=((0, max_sent_len-datum[5])), mode="constant", constant_values=PAD_IDX)
        trg_data.append(trg_data_padded)
        trg_mask_padded = np.pad(np.array(datum[4]), pad_width=((0, max_sent_len-datum[5])), mode="constant", constant_values=PAD_IDX)
        trg_mask.append(trg_mask_padded)

    ind_dec_order = np.argsort(src_len)[::-1]
    src_data = np.array(src_data)[ind_dec_order]
    trg_data = np.array(trg_data)[ind_dec_order]
    src_mask = np.array(src_mask)[ind_dec_order].reshape(len(batch), -1, 1)
    trg_mask = np.array(trg_mask)[ind_dec_order].reshape(len(batch), -1, 1)
    src_len = np.array(src_len)[ind_dec_order]
    trg_len = np.array(trg_len)[ind_dec_order]

    return [torch.from_numpy(src_data), torch.from_numpy(src_mask).float(), src_len,
            torch.from_numpy(trg_data), torch.from_numpy(trg_mask).float(), trg_len]


# src: always English
def train(LSTM_s, LSTM_t, discriminator, loader, contrastive_loader, optimizer, dis_optim, epoch):
    LSTM_s.train()
    LSTM_t.train()
    discriminator.train()
    total_loss = 0
    for batch_idx, ([s_data, s_mask, s_len, t_data, t_mask, t_len],
                    [s_c, s_mc, s_len_c, t_c, t_mc, t_len_c]) in \
                    enumerate(zip(loader, contrastive_loader)):
        # main samples
        s_data, s_mask = s_data.long().to(device), s_mask.to(device)
        t_data, t_mask = t_data.long().to(device), t_mask.to(device)
        # contrastive samples
        s_c, s_mc = s_c.long().to(device), s_mc.to(device)
        t_c, t_mc = t_c.long().to(device), t_mc.to(device)
        optimizer.zero_grad()
        dis_optim.zero_grad()
        if np.random.random() <= 0.3:
            s_data = s_data + torch.rand(s_data.size()).long().to(device)
            t_data = t_data + torch.rand(t_data.size()).long().to(device)
        s_out = LSTM_s(s_data, s_mask, s_len)
        t_out = LSTM_t(t_data, t_mask, t_len)
        s_c_out = LSTM_s(s_c, s_mc, s_len_c)
        t_c_out = LSTM_t(t_c, t_mc, t_len_c)
        loss = loss_align(s_out, t_out, s_c_out, t_c_out, 0.25)
        loss.cuda().backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item() * len(s_data) / len(loader.dataset)
        if (batch_idx + 1) % (len(loader.dataset)//(50 * config.batch_size)) == 0:
            
            dis_labels_s = torch.zeros(config.batch_size).long()
            dis_labels_t = torch.ones(config.batch_size).long()
            # concat source - target
            dis_labels = torch.cat([dis_labels_s, dis_labels_t], 0)
            idx = torch.randperm(config.batch_size * 2)
            dis_input = torch.cat([s_out, t_out], 0)[idx]
            dis_labels = dis_labels[idx].to(device)
            dis_out = discriminator(dis_input)
            dis_criterion = nn.NLLLoss()
            dis_loss = dis_criterion(dis_out, dis_labels)
            dis_loss.cuda().backward(retain_graph=True)
            dis_optim.step()
            loss += (-1) * dis_criterion(dis_out, dis_labels)
            loss.cuda().backward()
            optimizer.step()
            torch.save(LSTM_t.state_dict(), "LSTM_en_{}_epoch_{}".format(config.experiment_lang.upper(), epoch))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(epoch, (batch_idx+1) * config.batch_size, 
                                                                           len(loader.dataset), 100. * (batch_idx+1) / len(loader), loss.item()))

    optimizer.zero_grad()
    return total_loss


