import pickle
import random
import sys
import functools
import numpy as np
import pandas as pd
from setuptools import setup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable

from models import biLSTM
from preprocess_alignment import config_class

def loss_align(en_rep, target_rep, en_c, target_c, lambda_reg):
    """:param en_rep: output repr of eng encoder (batch_size, hidden_size)
       :param target_rep: output repr of target encoder (batch_size, hidden_size)
       :param en_c: contrastive sentence repr from eng encoder (batch_size, hidden_size)
       :param target_c: contrastive sentence repr form target encoder (batch_size, hidden_size)
       :param lambda_reg: regularization coef [default: 0.5]

    Returns: L_align = l2norm (en_rep, target_rep) - lambda_reg( l2norm (en_c, target_rep) + l2norm (en_rep, target_c))
    """
    dist = torch.norm(en_rep - target_rep, 2)
    c_dist = torch.norm(en_c - target_rep, 2) + torch.norm(en_rep - target_c, 2)
    L_align = dist - lambda_reg*(c_dist)
    return L_align

def train(LSTM_src, LSTM_trg, loader, contrastive_loader, optimizer, epoch, lambda_reg=0.25):
    LSTM_src.train()
    LSTM_trg.train()
    total_loss = 0
    for batch_idx, ([src_data, src_mask, src_len, trg_data, trg_mask, trg_len], 
    				[src_c, src_mc, src_len_c, trg_c, trg_mc, trg_len_c]) in [*enumerate(zip(loader, contrastive_loader))]:
        # correctly matched data
        src_data, src_mask = src_data.to(device), src_mask.to(device)
        trg_data, trg_mask = trg_data.to(device), trg_mask.to(device)
        # contrastive data 
        src_c, src_mc = src_c.to(device), src_mc.to(device)
        trg_c, trg_mc = trg_c.to(device), trg_mc.to(device)
        optimizer.zero_grad()
        src_out = LSTM_src(src_data, src_mask, src_len)
        trg_out = LSTM_trg(trg_data, trg_mask, trg_len)
        src_c_out = LSTM_src(src_c, src_mc, src_len_c)
        trg_c_out = LSTM_trg(trg_c, trg_mc, trg_len_c)
        loss = loss_align(src_out, trg_out, src_c_out, trg_c_out, lambda_reg)
        loss.cuda().backward()
        optimizer.step()
        total_loss += loss.item() * len(src_data) / len(loader.dataset)
        if (batch_idx+1) % (len(loader.dataset)//(20*config.batch_size)) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * config.batch_size, len(loader.dataset),
                100. * (batch_idx+1) / len(loader), loss.item()), end="\r")
            
    optimizer.zero_grad()
    return total_loss


config = config_class(corpus = "multinli",
             val_test_lang = "de",
             max_sent_len = 100,
             max_vocab_size = 200000,
             epochs = 15,
             batch_size = 32,
             embed_dim = 300,
             hidden_dim = 512,
             dropout = 0.1,
             lr = 1e-3)

load_epoch = 6
LSTM_src = biLSTM(hidden_size=config.hidden_dim, embedding_weights=weights_init ,
	num_layers=1, percent_dropout = config.dropout,
	vocab_size=weights_init.shape[0], interaction_type="concat", 
	input_size=config.embed_dim).to(device)

LSTM_src.load_state_dict(torch.load("best_encoder_eng_de_mnli_{}".format(load_epoch)))
# fix the Eng encoder
for param in LSTM_src.parameters():
    param.requires_grad = False

LSTM_trg = biLSTM(hidden_size=config.hidden_dim, embedding_weights=weights_init, 
	num_layers=1, percent_dropout = config.dropout,
	vocab_size=weights_init.shape[0], interaction_type="concat", 
	input_size=config.embed_dim).to(device)

LSTM_trg.load_state_dict(torch.load("best_encoder_eng_de_mnli_{}",format(load_epoch)))

print ("Encoder src:\n", LSTM_src)
print ("Encoder trg:\n", LSTM_trg)

for epoch in range(config.epochs):

    print ("\nepoch = "+str(epoch))
    loss_train = train(LSTM_src=LSTM_src, LSTM_trg=LSTM_trg, loader=align_loader, contrastive_loader=c_align_loader,
                       optimizer = torch.optim.Adam([*LSTM_src.parameters()] + [*LSTM_trg.parameters()], lr=config.lr), 
                       epoch = epoch)

    torch.save(LSTM_trg.state_dict(), "LSTM_en_{}_{}_epoch_{}".format(config.val_test_lang, config.val_test_lang.upper(),
    	epoch))


