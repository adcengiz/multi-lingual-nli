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

PAD_IDX = 0
UNK_IDX = 1
label_dict = {"entailment":0, "neutral":1, "contradiction":2}
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
    punc = [*string.punctuation]
    if lang == "ar":
        for s in ["sentence1", "sentence2"]:
            dataset["{}_tokenized".format(s)] = dataset[s].\
            apply(lambda x: [a + ".ar" for a in nltk.tokenize.wordpunct_tokenize(x)])
        ext = dataset["sentence1_tokenized"].apply(lambda x: all_s1_tokens.extend(x))
        ext1 = dataset["sentence2_tokenized"].apply(lambda x: all_s2_tokens.extend(x))
        all_tokens = all_s1_tokens + all_s2_tokens
    elif lang == "zh":
        for s in ["sentence1", "sentence2"]:
            dataset["{}_tokenized".format(s)] = dataset[s].\
            apply(lambda x: [z + ".zh" for z in ' '.join(jieba.cut(x, cut_all=True)).split(" ") if z not in string.punctuation])
        ext = dataset["sentence1_tokenized"].apply(lambda x: all_s1_tokens.extend(x))
        ext1 = dataset["sentence2_tokenized"].apply(lambda x: all_s2_tokens.extend(x))
        all_tokens = all_s1_tokens + all_s2_tokens
    else:
        for s in ["sentence1", "sentence2"]:
            dataset["{}_tokenized".format(s)] = dataset[s].\
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

def read_and_tokenize_opus_data(lang="tr"):
    all_en_tokens = []
    all_target_tokens = []
    path_en = opus_path + "/{}_en/en_data_00".format(lang, lang)
    path_target = opus_path + "/{}_en/{}_data_00".format(lang, lang)
    en_corpus = open(path_en, "r")
    target_corpus = open(path_target, "r")
    en_series = pd.Series(en_corpus.read().split("\n"))
    target_series = pd.Series(target_corpus.read().split("\n"))
    dataset = pd.DataFrame({"en":en_series, lang:target_series})
    if lang == "ar":
        dataset["en_tokenized"] = dataset["en"].apply(lambda x: "".join(c for c in x if c not in string.punctuation).lower().split(" "))
        dataset["en_tokenized"] = dataset["en_tokenized"].apply(lambda x:[a+".en" for a in x])
        dataset["ar_tokenized"] = dataset["ar"].apply(lambda x: [a + ".ar" for a in nltk.tokenize.wordpunct_tokenize(x)])
    elif lang == "zh":
        dataset["en_tokenized"] = dataset["en"].apply(lambda x: "".join(c for c in x if c not in string.punctuation).lower().split(" "))
        dataset["en_tokenized"] = dataset["en_tokenized"].apply(lambda x:[a+".en" for a in x])
        dataset["zh_tokenized"] = dataset["zh"].apply(lambda x: [z + ".zh" for z in ' '.join(jieba.cut(x, cut_all=True)).split(" ") if z not in string.punctuation])
    else:
        for i in ["en", lang]:
            dataset["{}_tokenized".format(i)] = dataset[i].apply(lambda x: "".join(c for c in x if c not in string.punctuation).lower().split(" "))
            dataset["{}_tokenized".format(i)] = dataset["{}_tokenized".format(i)].\
            apply(lambda x:[a+".{}".format(i) for a in x])
    dataset["en_tokenized"].apply(lambda x: all_en_tokens.extend(x))
    dataset["{}_tokenized".format(lang)].apply(lambda x: all_target_tokens.extend(x))
    return dataset, all_en_tokens, all_target_tokens

def read_and_tokenize_europarl_data(lang="de"):
    all_en_tok = []
    all_target_tok = []
    path_en = europarl_path + "/{}_en/europarl-v7.{}-en.en".format(lang, lang)
    path_target = europarl_path + "/{}_en/europarl-v7.{}-en.{}".format(lang, lang, lang)
    en_corpus = open(path_en, "r")
    target_corpus = open(path_target, "r")
    en_series = pd.Series(en_corpus.read().split("\n"))
    target_series = pd.Series(target_corpus.read().split("\n"))
    dataset = pd.DataFrame({"en":en_series, lang:target_series})
    for i in ["en", lang]:
        dataset["{}_tokenized".format(i)] = dataset[i].apply(lambda x: "".join(c for c in x if c not in string.punctuation).lower().split(" "))
        dataset["{}_tokenized".format(i)] = dataset["{}_tokenized".format(i)].apply(lambda x:[a+".{}".format(i) for a in x])
    dataset["en_tokenized"].apply(lambda x: all_en_tok.extend(x))
    dataset["{}_tokenized".format(lang)].apply(lambda x: all_target_tok.extend(x))
    return dataset, all_en_tokens, all_target_tokens

def create_contrastive_dataset(dataset, trg_lang):
    shuffle_ix_src = torch.randperm(len(dataset))
    src_c = np.array([*dataset["{}_tokenized".format("en")].values])[shuffle_ix_src]
    trg_c = dataset["{}_tokenized".format(trg_lang)]
    contrastive_df = pd.DataFrame({"en_tokenized": src_c, "{}_tokenized".format(trg_lang): trg_c})
    return contrastive_df

def update_vocab_keys(src_vocab, trg_vocab):
    for x in [*src_vocab.keys()]:
        src_vocab[x + ".en"] = src_vocab[x]
        src_vocab.pop(x)
    for y in [*trg_vocab.keys()]:
        trg_vocab[y + ".{}".format(config.experiment_lang)] = trg_vocab[y]
        trg_vocab.pop(y)
        
    src_vocab.update(trg_vocab)
    return src_vocab

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
