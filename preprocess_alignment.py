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

opus_path = "/scratch/adc563/nlu_project/data/opus"
europarl_path = "/scratch/adc563/nlu_project/data/europarl"
un_path = "/scratch/adc563/nlu_project/data/un_parallel_corpora"

def read_and_tokenize_opus_data(lang="de"):
    all_en_tokens = []
    all_target_tokens = []
    path_en = opus_path + "/{}_en/en_data_00".format(lang, lang)
    path_target = opus_path + "/{}_en/{}_data_00".format(lang, lang)
    en_corpus = open(path_en, "r")
    target_corpus = open(path_target, "r")
    en_series = pd.Series(en_corpus.read().split("\n"))
    target_series = pd.Series(target_corpus.read().split("\n"))
    dataset = pd.DataFrame({"en":en_series, lang:target_series})
    for i in ["en", lang]:
        dataset["{}_tokenized".format(i)] = dataset[i].apply(lambda x: "".join(c for c in x if c not in string.punctuation).split(" "))
        dataset["{}_tokenized".format(i)] = dataset["{}_tokenized".format(i)].\
        apply(lambda x:[a+".{}".format(i) for a in x])
    dataset["en_tokenized"].apply(lambda x: all_en_tokens.extend(x))
    dataset["{}_tokenized".format(lang)].apply(lambda x: all_target_tokens.extend(x))
    return dataset, all_en_tokens, all_target_tokens

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
        dataset["{}_tokenized".format(i)] = dataset[i].apply(lambda x: "".join(c for c in x if c not in string.punctuation).split(" "))
        dataset["{}_tokenized".format(i)] = dataset["{}_tokenized".format(i)].apply(lambda x:[a+".{}".format(i) for a in x])
    dataset["en_tokenized"].apply(lambda x: all_en_tokens.extend(x))
    dataset["{}_tokenized".format(lang)].apply(lambda x: all_target_tokens.extend(x))
    return dataset, all_en_tokens, all_target_tokens

def create_contrastive_dataset(dataset, trg_lang):
    shuffle_ix_src = torch.randperm(len(dataset))
    src_c = np.array([*dataset["{}_tokenized".format("en")].values])[shuffle_ix_src]
    trg_c = dataset["{}_tokenized".format(trg_lang)]
    contrastive_df = pd.DataFrame({"en_tokenized": src_c, "{}_tokenized".format(trg_lang): trg_c})
    return contrastive_df

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
             val_test_lang = "fr",
             max_sent_len = 40,
             max_vocab_size = 100000,
             epochs = 15,
             batch_size = 32, # decreased, because we will have contrastive batch - so x 2
             embed_dim = 300,
             hidden_dim = 512,
             dropout = 0.1,
             lr = 1e-3)

print ("Loading vectors for EN.")
aligned_src_vectors = load_glove_vectors("en")
print ("Loading vectors for FR.")
aligned_trg_vectors = load_aligned_vectors(config.val_test_lang)

id2token_src = [x+"."+"en" for x in [*aligned_src_vectors.keys()]][:config.max_vocab_size]
id2token_trg = [x+"."+"fr" for x in [*aligned_trg_vectors.keys()]][:config.max_vocab_size]
id2token_mutual = ["<PAD>", "<UNK>"] + id2token_src + id2token_trg
vecs_mutual = update_vocab_keys(aligned_src_vectors, aligned_trg_vectors)

token2id_mutual = build_tok2id(id2token_mutual)
weights_init = init_embedding_weights(vecs_mutual, token2id_mutual, id2token_mutual, config.embed_dim)

data_en_fr, all_en_tokens, all_fr_tokens = read_and_tokenize_europarl_data(lang=config.val_test_lang)

c_df = create_contrastive_dataset(data_en_fr, config.val_test_lang)
c_df = c_df.iloc[torch.randperm(len(c_df))]

class AlignDataset(Dataset):
    def __init__(self, data, max_sent_len, src_lang, trg_lang, token2id, id2token):
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

align_dataset = AlignDataset(data_en_fr, config.max_sent_len, "en", config.val_test_lang,
                             token2id_mutual, id2token_mutual)
align_loader = torch.utils.data.DataLoader(dataset=align_dataset, batch_size=config.batch_size,
                               collate_fn=lambda x, max_sentence_length=config.max_sent_len: align_collate_func(x, config.max_sent_len),
                               shuffle=False)


c_align_dataset = AlignDataset(c_df, config.max_sent_len, "en", config.val_test_lang, 
                               token2id_mutual, id2token_mutual)
c_align_loader = torch.utils.data.DataLoader(dataset=c_align_dataset, batch_size=config.batch_size,
                               collate_fn=lambda x, max_sentence_length=config.max_sent_len: align_collate_func(x, config.max_sent_len),
                               shuffle=False)



