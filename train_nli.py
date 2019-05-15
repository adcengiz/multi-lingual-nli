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

from classifier_nli import Linear_Layers
from Discriminator import Discriminator
from sentence_encoder import biLSTM

PAD_IDX = 0
UNK_IDX = 1
label_dict = {"entailment":0, "neutral":1, "contradiction":2}
opus_path = "/scratch/adc563/nlu_project/data/opus"
europarl_path = "/scratch/adc563/nlu_project/data/europarl"
un_path = "/scratch/adc563/nlu_project/data/un_parallel_corpora"

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
             batch_size = 32,
             embed_dim = 300,
             hidden_dim = 512,
             dropout = 0.1,
             lr = 1e-3,
             experiment_lang = "fr")

def read_xnli(lang):
    fname = "/scratch/adc563/nlu_project/data/XNLI/xnli.{}.jsonl"
    xnli_dev = pd.read_json(fname.format("dev"), lines=True)
    xnli_test = pd.read_json(fname.format("test"), lines=True)
    if lang == "all":
        dev_data = xnli_dev
        test_data = xnli_test
    else:
        dev_data = xnli_dev[xnli_dev["language"]==lang]
        test_data = xnli_test[xnli_test["language"]==lang]
    return dev_data, test_data

def load_aligned_vectors(lang):
    f = "/scratch/adc563/nlu_project/data/aligned_embeddings/wiki.{}.align.vec".format(lang)
    fin = io.open(f, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = [*map(float, tokens[1:])]
    return data

def load_multilingual_vectors(lang):
    fname = "/scratch/adc563/nlu_project/data/multi_lingual_embeddings/cc.{}.300.vec".format(lang)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

def load_glove_vectors(lang):
    f = "/scratch/adc563/nlu_project/HBMP/vector_cache/glove.840B.300d.txt".format(lang)
    fin = io.open(f, "r", encoding="utf-8", newline="\n", errors="ignore")
    n = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = [*map(float, tokens[1:])]
    return data

def read_enli(nli_corpus = "snli"):
    if nli_corpus == "snli":
        path_ = "/scratch/adc563/nlu_project/HBMP/data/snli/snli_1.0/snli_1.0"
        train = pd.read_json("{}_{}.jsonl".format(path_,"train"), lines=True)
        dev = pd.read_json("{}_{}.jsonl".format(path_,"dev"), lines=True)
        test = pd.read_json("{}_{}.jsonl".format(path_,"test"), lines=True)
        # remove - from gold label
        train = train[train["gold_label"] != "-"]
        dev = dev[dev["gold_label"] != "-"]
        test = test[test["gold_label"] != "-"]
    elif nli_corpus == "multinli":
        path_ = "/scratch/adc563/nlu_project/HBMP/data/multinli/multinli_1.0/multinli_1.0"
        train = pd.read_json("{}_{}.jsonl".format(path_,"train"), lines=True)
        dev = pd.read_json("{}_{}_matched.jsonl".format(path_, "dev"), lines=True)
        test = None
        # remove - from gold label
        train = train[train["gold_label"] != "-"]
        dev = dev[dev["gold_label"] != "-"]
    return train, dev, test

def write_numeric_label(train, dev, test, nli_corpus="multinli"):
    if nli_corpus == "multinli":
        for dataset in [train, dev]:
            dataset["gold_label"] = dataset["gold_label"].apply(lambda x: label_dict[x])
    elif nli_corpus == "snli":
        for dataset in [train, dev, test]:
            dataset["gold_label"] = dataset["gold_label"].apply(lambda x: label_dict[x])
    elif nli_corpus == "xnli":
        for dataset in [dev, test]:
            dataset["gold_label"] = dataset["gold_label"].apply(lambda x: label_dict[x])
    else:
        raise ValueError ("NLI corpus name should be in [multinli, snli, xnli]")
    return train, dev, test

def tokenize_xnli(dataset, remove_punc=False, lang="en"):
    all_s1_tokens = []
    all_s2_tokens = []
    for s in ["sentence1", "sentence2"]:
        punc = [*string.punctuation]
        dataset["{}_tokenized".format(s)] = dataset["{}".format(s)].\
        apply(lambda x: "".join(c for c in x if c not in string.punctuation).lower().split(" "))
        dataset["{}_tokenized".format(s)] = dataset["{}_tokenized".format(s)].\
        apply(lambda x: [a+"."+lang for a in x])
    ext = dataset["sentence1_tokenized"].apply(lambda x: all_s1_tokens.extend(x))
    ext1 = dataset["sentence2_tokenized"].apply(lambda x: all_s2_tokens.extend(x))
    all_tokens = all_s1_tokens + all_s2_tokens
    return dataset, all_tokens

def build_vocab(all_tokens, max_vocab_size):
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = [*vocab]
    token2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2token = ['<PAD>', '<UNK>'] + id2token
    token2id["<PAD>"] = 0
    token2id["<UNK>"] = 1
    return token2id, id2token

def build_tok2id(id2token):
    token2id = {}
    for i in range(len(id2token)):
        token2id[id2token[i]] = i
    return token2id

def init_embedding_weights(vectors, token2id, id2token, embedding_size):
    weights = np.zeros((len(id2token), embedding_size))
    for idx in range(2, len(id2token)):
        token = id2token[idx]
        weights[idx] = vectors[token]
    weights[1] = np.random.randn(embedding_size)
    return weights

def update_vocab_keys(src_vocab, trg_vocab):
    for x in [*src_vocab.keys()]:
        src_vocab[x + ".en"] = src_vocab[x]
        src_vocab.pop(x)
    for y in [*trg_vocab.keys()]:
        trg_vocab[y + ".{}".format(config.experiment_lang)] = trg_vocab[y]
        trg_vocab.pop(y)
        
    src_vocab.update(trg_vocab)
    return src_vocab



