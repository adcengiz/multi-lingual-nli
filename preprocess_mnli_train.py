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

PAD_IDX = 0
UNK_IDX = 1
label_dict = {"entailment":0, "neutral":1, "contradiction":2}

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
seed = 1
device = torch.device("cuda" if cuda else "cpu")

opus_path = "/scratch/adc563/nlu_project/data/opus"
europarl_path = "/scratch/adc563/nlu_project/data/europarl"
un_path = "/scratch/adc563/nlu_project/data/un_parallel_corpora"

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
    f = "/scratch/adc563/nlu_project/data/multi_lingual_embeddings/cc.{}.300.vec".format(lang)
    fin = io.open(f, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = [*map(float, tokens[1:])]
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
        apply(lambda x: "".join(c for c in x if c not in string.punctuation).split(" "))
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

def update_vocab_keys(src_vocab, trg_vocab):
    for x in [*src_vocab.keys()]:
        src_vocab[x + ".en"] = src_vocab[x]
        src_vocab.pop(x)
    for y in [*trg_vocab.keys()]:
        trg_vocab[y + ".fr"] = trg_vocab[y]
        trg_vocab.pop(y)
        
    src_vocab.update(trg_vocab)
    return src_vocab

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

class config_class:
    def __init__(self, corpus, val_test_lang, max_sent_len, max_vocab_size, epochs, batch_size, 
                    embed_dim, hidden_dim, dropout, lr):
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


config = config_class(corpus = "multinli",
             val_test_lang = "en",
             max_sent_len = 40,
             max_vocab_size = 150000,
             epochs = 15,
             batch_size = 32, # decreased, because we will have contrastive batch - so x 2
             embed_dim = 300,
             hidden_dim = 512,
             dropout = 0.1,
             lr = 5e-4)

print ("Loading vectors for EN.")
aligned_src_vectors = load_glove_vectors("en")
print ("Loading vectors for FR.")
aligned_trg_vectors = load_aligned_vectors("fr")


id2token_src = [x+".en" for x in [*aligned_src_vectors.keys()]][:config.max_vocab_size]
id2token_trg = [x+".fr" for x in [*aligned_trg_vectors.keys()]][:config.max_vocab_size]
id2token_mutual = ["<PAD>", "<UNK>"] + id2token_src + id2token_trg
vecs_mutual = update_vocab_keys(aligned_src_vectors, aligned_trg_vectors)
token2id_mutual = build_tok2id(id2token_mutual)

weights_init = init_embedding_weights(vecs_mutual, token2id_mutual, id2token_mutual, config.embed_dim)

# load train and preprocess
print ("Reading {} data.".format(config.corpus))
nli_train, nli_dev, nli_test = read_enli(nli_corpus=config.corpus)
print ("Writing numeric label.")
if config.corpus == "multinli":
    nli_train, nli_dev, _ = write_numeric_label(nli_train, nli_dev, nli_test, nli_corpus=config.corpus)
elif config.corpus == "snli":
    nli_train, nli_dev, nli_test = write_numeric_label(nli_train, nli_dev, nli_test, nli_corpus=config.corpus)

# load val and test and preprocess
print ("Reading XNLI {} data.".format(config.val_test_lang.upper()))
xnli_dev, xnli_test = read_xnli(config.val_test_lang)
_, xnli_dev, xnli_test = write_numeric_label(None, xnli_dev, xnli_test, nli_corpus="xnli")

nli_train, all_train_tokens = tokenize_xnli(nli_train, lang="en")
xnli_dev, _ = tokenize_xnli(xnli_dev, lang="en")
xnli_test, _ = tokenize_xnli(xnli_test, lang="en")

class NLIDataset(Dataset):
    def __init__(self, tokenized_dataset, max_sentence_length, token2id, id2token):
        self.sentence1, self.sentence2, self.labels = [*tokenized_dataset["sentence1_tokenized"].values], \
                                                      [*tokenized_dataset["sentence2_tokenized"].values], \
                                                      [*tokenized_dataset["gold_label"].values]
        self.max_sentence_length = int(max_sentence_length)
        self.token2id, self.id2token = token2id, id2token
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, row):
        label = self.labels[row]
        sentence1_word_idx, sentence2_word_idx = [], []
        sentence1_mask, sentence2_mask = [], []
        for word in self.sentence1[row][:self.max_sentence_length]:
            if word in self.token2id.keys():
                sentence1_word_idx.append(self.token2id[word])
                sentence1_mask.append(0)
            else:
                sentence1_word_idx.append(UNK_IDX)
                sentence1_mask.append(1)
        for word in self.sentence2[row][:self.max_sentence_length]:
            if word in self.token2id.keys():
                sentence2_word_idx.append(self.token2id[word])
                sentence2_mask.append(0)
            else:
                sentence2_word_idx.append(UNK_IDX)
                sentence2_mask.append(1)
        sentence1_list = [sentence1_word_idx, sentence1_mask, len(sentence1_word_idx)]
        sentence2_list = [sentence2_word_idx, sentence2_mask, len(sentence2_word_idx)]
        
        return sentence1_list + sentence2_list + [label]

def nli_collate_func(batch, max_sent_length):
    sentence1_data, sentence2_data = [], []
    sentence1_mask, sentence2_mask = [], []
    s1_lengths, s2_lengths = [], []
    labels = []

    for datum in batch:
        s1_lengths.append(datum[2])
        s2_lengths.append(datum[5])
        labels.append(datum[6])
        sentence1_data_padded = np.pad(np.array(datum[0]), pad_width=((0, config.max_sent_len-datum[2])), mode="constant", constant_values=0)
        sentence1_data.append(sentence1_data_padded)
        sentence1_mask_padded = np.pad(np.array(datum[1]), pad_width=((0, config.max_sent_len-datum[2])), mode="constant", constant_values=0)
        sentence1_mask.append(sentence1_mask_padded)
        sentence2_data_padded = np.pad(np.array(datum[3]), pad_width=((0, config.max_sent_len-datum[5])), mode="constant", constant_values=0)
        sentence2_data.append(sentence2_data_padded)
        sentence2_mask_padded = np.pad(np.array(datum[4]), pad_width=((0, config.max_sent_len-datum[5])), mode="constant", constant_values=0)
        sentence2_mask.append(sentence2_mask_padded)
        
    ind_dec_order = np.argsort(s1_lengths)[::-1]
    sentence1_data = np.array(sentence1_data)[ind_dec_order]
    sentence2_data = np.array(sentence2_data)[ind_dec_order]
    sentence1_mask = np.array(sentence1_mask)[ind_dec_order].reshape(len(batch), -1, 1)
    sentence2_mask = np.array(sentence2_mask)[ind_dec_order].reshape(len(batch), -1, 1)
    s1_lengths = np.array(s1_lengths)[ind_dec_order]
    s2_lengths = np.array(s2_lengths)[ind_dec_order]
    labels = np.array(labels)[ind_dec_order]
    
    s1_list = [torch.from_numpy(sentence1_data), torch.from_numpy(sentence1_mask).float(), s1_lengths]
    s2_list = [torch.from_numpy(sentence2_data), torch.from_numpy(sentence2_mask).float(), s2_lengths]
        
    return [torch.from_numpy(sentence1_data), torch.from_numpy(sentence1_mask).float(), s1_lengths,
            torch.from_numpy(sentence2_data), torch.from_numpy(sentence2_mask).float(), s2_lengths,
            labels]

# train
nli_train_dataset = NLIDataset(nli_train, max_sentence_length=config.max_sent_len, token2id=token2id_mutual, id2token=id2token_mutual)
nli_train_loader = torch.utils.data.DataLoader(dataset=nli_train_dataset, batch_size=config.batch_size,
                               collate_fn=lambda x, max_sentence_length=config.max_sent_len: nli_collate_func(x, config.max_sent_len),
                               shuffle=False)

# dev
nli_dev_dataset = NLIDataset(xnli_dev, max_sentence_length=config.max_sent_len, token2id=token2id_mutual, id2token=id2token_mutual)
nli_dev_loader = torch.utils.data.DataLoader(dataset=nli_dev_dataset, batch_size=config.batch_size,
                               collate_fn=lambda x, max_sentence_length=config.max_sent_len: nli_collate_func(x, config.max_sent_len),
                               shuffle=False)

# test
nli_test_dataset = NLIDataset(xnli_test, max_sentence_length=config.max_sent_len, token2id=token2id_mutual, id2token=id2token_mutual)
nli_test_loader = torch.utils.data.DataLoader(dataset=nli_test_dataset, batch_size=config.batch_size,
                               collate_fn=lambda x, max_sentence_length=config.max_sent_len: nli_collate_func(x, config.max_sent_len),
                               shuffle=False)
