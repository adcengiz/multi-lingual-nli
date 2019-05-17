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

from sentence_encoder import biLSTM
from Discriminator import Discriminator
from classifier_nli import Linear_Layers
import preprocess as pre 
import nli_trainer as nlit
import align_trainer as alt

class NLIconfig:
    def __init__(self, corpus, val_test_lang, max_sent_len, max_vocab_size, epochs, batch_size, 
                    embed_dim, hidden_dim, dropout, lr, experiment_lang):
        self.corpus = corpus
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

config = NLIconfig(corpus = "multinli",
             val_test_lang = "en",
             max_sent_len = 40,
             max_vocab_size = 210000,
             epochs = 15,
             batch_size = 64,
             embed_dim = 300,
             hidden_dim = 512,
             dropout = 0.1,
             lr = 1e-3,
             experiment_lang = "zh")

print ("Loading glove vectors for EN.")
aligned_src_vectors = pre.load_glove_vectors("en")
print ("Loading glove vectors for {}.".format(config.experiment_lang.upper()))
aligned_trg_vectors = pre.load_aligned_vectors(config.experiment_lang)

id2token_src = [x+"."+"en" for x in [*aligned_src_vectors.keys()]][:config.max_vocab_size]
id2token_trg = [x +".{}".format(config.experiment_lang) for x in [*aligned_trg_vectors.keys()]][:config.max_vocab_size]
id2token_mutual = ["<PAD>", "<UNK>"] + id2token_src + id2token_trg
vecs_mutual = update_vocab_keys(aligned_src_vectors, aligned_trg_vectors)
token2id_mutual = pre.build_tok2id(id2token_mutual)
weights_init = pre.init_embedding_weights(vecs_mutual, token2id_mutual, id2token_mutual, 300)

# load train and preprocess
print ("Reading {} data.".format(config.corpus))
nli_train, nli_dev, nli_test = pre.read_enli(nli_corpus=config.corpus)
print ("Writing numeric label.")
if config.corpus == "multinli":
    nli_train, nli_dev, _ = pre.write_numeric_label(nli_train, nli_dev, nli_test, nli_corpus=config.corpus)
elif config.corpus == "snli":
    nli_train, nli_dev, nli_test = pre.write_numeric_label(nli_train, nli_dev, nli_test, nli_corpus=config.corpus)

# load val and test and preprocess
print ("Reading XNLI {} data.".format(config.val_test_lang.upper()))
xnli_dev, xnli_test = pre.read_xnli(config.val_test_lang)
_, xnli_dev, xnli_test = pre.write_numeric_label(None, xnli_dev, xnli_test, nli_corpus="xnli")

nli_train, all_train_tokens = pre.tokenize_xnli(nli_train, lang="en")
xnli_dev, _ = pre.tokenize_xnli(xnli_dev, lang="en")
xnli_test, _ = pre.tokenize_xnli(xnli_test, lang="en")

# train
nli_train_dataset = pre.NLIDataset(nli_train, max_sentence_length = config.max_sent_len, token2id = token2id_mutual, id2token = id2token_mutual)
nli_train_loader = torch.utils.data.DataLoader(dataset = nli_train_dataset, batch_size=config.batch_size,
                               collate_fn = lambda x, max_sentence_length=config.max_sent_len: pre.nli_collate_func(x, config.max_sent_len),
                               shuffle = False)

# dev
nli_dev_dataset = pre.NLIDataset(xnli_dev, max_sentence_length = config.max_sent_len, token2id=token2id_mutual, id2token = id2token_mutual)
nli_dev_loader = torch.utils.data.DataLoader(dataset = nli_dev_dataset, batch_size = config.batch_size,
                               collate_fn = lambda x, max_sentence_length=config.max_sent_len: pre.nli_collate_func(x, config.max_sent_len),
                               shuffle = False)

# test
nli_test_dataset = pre.NLIDataset(xnli_test, max_sentence_length = config.max_sent_len, token2id = token2id_mutual, id2token = id2token_mutual)
nli_test_loader = torch.utils.data.DataLoader(dataset = nli_test_dataset, batch_size = config.batch_size,
                               collate_fn = lambda x, max_sentence_length = config.max_sent_len: pre.nli_collate_func(x, config.max_sent_len),
                               shuffle = False)

LSTM_en = biLSTM(config.hidden_dim, weights_init, config.dropout, config.max_vocab_size, 
                 num_layers = 1, input_size = config.embed_dim).to(device)

linear_model = Linear_Layers(hidden_size = 1024, hidden_size_2 = 128, percent_dropout = config.dropout, 
	classes = 3, input_size = config.embed_dim).to(device)

print ("Encoder:\n", LSTM_en)
print ("Classifier:\n", linear_model)

validation_accuracy = [0]
start_epoch = 0

for epoch in range(config.epochs):
    print ("\nepoch = "+str(epoch))
    loss_train = nlit.train(LSTM_en, linear_model, DataLoader = nli_train_loader,
                       criterion = nn.NLLLoss(),
                       optimizer = torch.optim.Adam(list(LSTM_en.parameters()) + list(linear_model.parameters()), 
                                                   lr=config.lr, weight_decay=0),
                       epoch = epoch)
    
    val_acc = nlit.accuracy(LSTM_en, linear_model, nli_dev_loader, nn.NLLLoss(reduction='sum'))
    print ("\n{} Validation Accuracy = {}".format(config.val_test_lang.upper(), val_acc))
    if val_acc <= validation_accuracy[-1]:
        break
    validation_accuracy.append(val_acc)
    torch.save(LSTM_en.state_dict(), "best_encoder_eng_mnli_{}_{}".format(epoch, config.experiment_lang))
    torch.save(linear_model.state_dict(), "best_linear_eng_mnli_{}_{}".format(epoch, config.experiment_lang))
    laoad_epoch = epoch
    

class align_config:
    def __init__(self, corpus, val_test_lang, max_sent_len, max_vocab_size, epochs, batch_size, 
                    embed_dim, hidden_dim, dropout, lr, experiment_lang):
        self.corpus = corpus
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

config = align_config(corpus = "multinli",
             val_test_lang = "zh",
             max_sent_len = 30,
             max_vocab_size = 210000,
             epochs = 15,
             batch_size = 64, 
             embed_dim = 300,
             hidden_dim = 512,
             dropout = 0.1,
             lr = 1e-3,
             experiment_lang = "zh")

data_en_target, all_en_tokens, all_target_tokens = pre.read_and_tokenize_opus_data(lang=config.val_test_lang)
data_en_target["len_en"] = data_en_target["en_tokenized"].apply(lambda x: len(x))
data_en_target["len_{}".format(config.val_test_lang)] = \
data_en_target["{}_tokenized".format(config.val_test_lang)].apply(lambda x: len(x))
data_en_target = data_en_target[(data_en_target["len_en"] > 1)&(data_en_target["len_{}".format(config.val_test_lang)] > 1)]

c_df = pre.create_contrastive_dataset(data_en_target, config.val_test_lang)
c_df = c_df.iloc[torch.randperm(len(c_df))]
# shuffle
c_df["{}_tokenized".format(config.experiment_lang)].iloc[:100000] = c_df["{}_tokenized".format(config.experiment_lang)].iloc[:100000]\
	.apply(lambda x: [np.random.choice(x) for s in range(len(x)-1)])
shuffle_ix = torch.randperm(len(c_df))
c_df["{}_tokenized".format(config.experiment_lang)] = np.array(c_df["{}_tokenized".format(config.experiment_lang)])[shuffle_ix]
c_df["len_en"] = c_df["en_tokenized"].apply(lambda x: len(x))
c_df["len_{}".format(config.val_test_lang)] = \
c_df["{}_tokenized".format(config.val_test_lang)].apply(lambda x: len(x))
c_df = c_df[(c_df["len_en"] > 1)&(c_df["len_{}".format(config.val_test_lang)] > 1)]

align_dataset = pre.AlignDataset(data_en_target, config.max_sent_len, "en", config.experiment_lang,
                             token2id_mutual, id2token_mutual)
align_loader = torch.utils.data.DataLoader(dataset=align_dataset, batch_size=config.batch_size,
                               collate_fn=lambda x, max_sentence_length=config.max_sent_len: pre.align_collate_func(x, config.max_sent_len),
                               shuffle=False)

c_align_dataset = pre.AlignDataset(c_df, config.max_sent_len, "en", config.experiment_lang, 
                               token2id_mutual, id2token_mutual)
c_align_loader = torch.utils.data.DataLoader(dataset=c_align_dataset, batch_size=config.batch_size,
                               collate_fn=lambda x, max_sentence_length=config.max_sent_len: pre.align_collate_func(x, config.max_sent_len),
                               shuffle=False)

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


LSTM_src_model = biLSTM(hidden_size = config.hidden_dim, embedding_weights = weights_init, num_layers=1, percent_dropout = config.dropout, 
             			vocab_size=weights_init.shape[0], input_size=300).to(device)
LSTM_src_model.load_state_dict(torch.load("best_encoder_eng_mnli_{}_{}".format(load_epoch, config.experiment_lang)))

for param in LSTM_src_model.parameters():
    param.requires_grad = False

LSTM_trg_model = biLSTM(hidden_size=config.hidden_dim, embedding_weights=weights_init ,num_layers=1, percent_dropout = config.dropout, 
             			vocab_size=weights_init.shape[0], input_size=300).to(device)
LSTM_trg_model.load_state_dict(torch.load("best_encoder_eng_mnli_{}_{}".format(load_epoch, config.experiment_lang)))

disc = Discriminator(n_langs = 2, dis_layers = 5, dis_hidden_dim = 128, dis_dropout = 0.1).to(device)

print ("Encoder src:\n", LSTM_src_model)
print ("Encoder trg:\n", LSTM_trg_model)
print ("Discriminator:\n", disc)

for epoch in range(config.epochs):
    print ("\nepoch = "+str(epoch))
    
    loss_train = train(LSTM_src=LSTM_src_model, LSTM_trg=LSTM_trg_model, discriminator = disc,
                       loader=align_loader, contrastive_loader=c_align_loader,
                       optimizer = torch.optim.Adam([*LSTM_src_model.parameters()] + [*LSTM_trg_model.parameters()] + [*disc.parameters()],
                                                    lr=config.lr), 
                       dis_optim = torch.optim.Adam([*disc.parameters()],
                                                    lr=config.lr), 
                       epoch = epoch)

    torch.save(LSTM_trg_model.state_dict(), "LSTM_en_{}_{}_epoch_{}".format(config.experiment_lang,
                                                                      config.experiment_lang.upper(),
                                                                      epoch))


