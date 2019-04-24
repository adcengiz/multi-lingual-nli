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

opus_path = "/scratch/adc563/nlu_project/data/opus"
europarl_path = "/scratch/adc563/nlu_project/data/europarl"
un_path = "/scratch/adc563/nlu_project/data/un_parallel_corpora"

def read_and_tokenize_europarl_data(lang="de"):
    all_en_tokens = []
    all_target_tokens = []
    path_en = europarl_path + "/{}_en/europarl-v7.{}-en.en".format(lang, lang)
    path_target = europarl_path + "/{}_en/europarl-v7.{}-en.{}".format(lang, lang, lang)
    en_corpus = open(path_en, "r")
    target_corpus = open(path_target, "r")
    en_series = pd.Series(en_corpus.read().split("\n"))
    target_series = pd.Series(target_corpus.read().split("\n"))
    dataset = pd.DataFrame({"en":en_series, lang:target_series})
    for i in ["en", lang]:
        dataset["{}_tokenized".format(i)] = dataset[i].apply(lambda x: "".join(c for c in x if c not in string.punctuation).lower().split(" "))
    dataset["en_tokenized"].apply(lambda x: all_en_tokens.extend(x))
    dataset["{}_tokenized".format(lang)].apply(lambda x: all_target_tokens.extend(x))
    return dataset, all_en_tokens, all_target_tokens

def read_and_tokenize_opus_data(lang="tr"):
    all_en_tokens = []
    all_target_tokens = []
    path_en = opus_path + "/{}_en/OpenSubtitles.en-{}.en".format(lang, lang)
    path_target = opus_path + "/{}_en/OpenSubtitles.en-{}.{}".format(lang, lang, lang)
    en_corpus = open(path_en, "r")
    target_corpus = open(path_target, "r")
    en_series = pd.Series(en_corpus.read().split("\n"))
    target_series = pd.Series(target_corpus.read().split("\n"))
    dataset = pd.DataFrame({"en":en_series, lang:target_series})
    for i in ["en", lang]:
        dataset["{}_tokenized".format(i)] = dataset[i].apply(lambda x: "".join(c for c in x if c not in string.punctuation).lower().split(" "))
    dataset["en_tokenized"].apply(lambda x: all_en_tokens.extend(x))
    dataset["{}_tokenized".format(lang)].apply(lambda x: all_target_tokens.extend(x))
    return dataset, all_en_tokens, all_target_tokens

def build_vocab(all_tokens, max_vocab_size):
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = [*vocab]
    token2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2token = ['<PAD>', '<UNK>'] + id2token
    token2id["<PAD>"] = 0
    token2id["<UNK>"] = 1
    return token2id, id2token

def concat_opus_and_europarl(opus_dataset, all_opus_en_tokens, all_opus_target_tokens,
	europarl_dataset, all_europarl_en_tokens, all_europarl_target_tokens):
	concat_dataset = pd.concat([opus_dataset, europarl_dataset], 1)
	concat_en_tokens = [*set(all_opus_en_tokens + all_europarl_en_tokens)]
	concat_target_tokens = [*set(all_opus_target_tokens + all_europarl_target_tokens)]
	return concat_dataset, concat_en_tokens, concat_target_tokens

# parallel_dataset

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


