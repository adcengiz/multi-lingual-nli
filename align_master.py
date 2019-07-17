import pickle
import random
import spacy
import csv
import sys
import errno
import nltk
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
label_dict = {"entailment": 0, "neutral": 1, "contradiction": 2}
opus_path = "/scratch/adc563/nlu_project/data/opus"
europarl_path = "/scratch/adc563/nlu_project/data/europarl"
un_path = "/scratch/adc563/nlu_project/data/un_parallel_corpora"

class NLI_config:
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

config = NLI_config(corpus = "multinli",
                    val_test_lang = "fr",
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
    punc = [*string.punctuation]
    if lang == "ar":
        for s in ["sentence1", "sentence2"]:
            dataset["{}_tokenized".format(s)] = dataset[s].\
            apply(lambda x: [a + ".ar" for a in nltk.tokenize.wordpunct_tokenize(x)])
    elif lang == "zh":
        for s in ["sentence1", "sentence2"]:
            dataset["{}_tokenized".format(s)] = dataset[s].\
            apply(lambda x: [z + ".zh" for z in ' '.join(jieba.cut(x, cut_all=True)).split(" ")])
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
    counter_ = Counter(all_tokens)
    vocab, count = zip(*counter_.most_common(max_vocab_size))
    id2tok = [*vocab]
    tok2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2tok = ['<PAD>', '<UNK>'] + id2tok
    tok2id["<PAD>"] = 0
    tok2id["<UNK>"] = 1
    return tok2id, id2tok

def build_tok2id(id2tok):
    tok2id = {}
    for i in range(len(id2tok)):
        tok2id[id2tok[i]] = i
    return tok2id

def update_vocab_keys(src_vocab, trg_vocab):
    for x in [*src_vocab.keys()]:
        src_vocab[x + ".en"] = src_vocab[x]
        src_vocab.pop(x)
    for y in [*trg_vocab.keys()]:
        trg_vocab[y + ".{}".format(config.experiment_lang)] = trg_vocab[y]
        trg_vocab.pop(y)
    src_vocab.update(trg_vocab)
    return src_vocab

def init_embedding_weights(vecs, tok2id, id2tok, emb_size):
    weights = np.zeros((len(id2tok), emb_size))
    for idx in range(2, len(id2tok)):
        tok = id2tok[idx]
        weights[idx] = vecs[tok]
    weights[1] = np.random.randn(emb_size)
    return weights

def create_contrastive_dataset(data, trg_lang):
    shf_ix = torch.randperm(len(data))
    src_c = np.array([*data["{}_tokenized".format("en")].values])[shf_ix]
    trg_c = data["{}_tokenized".format(trg_lang)]
    c_df = pd.DataFrame({"en_tokenized": src_c, "{}_tokenized".format(trg_lang): trg_c})
    return c_df

def read_and_tokenize_opus_data(lang="tr"):
    all_en_tokens = []
    all_target_tokens = []
    path_en = opus_path + "/{}_en/en_data_00".format(lang)
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
    return dataset, all_en_tok, all_target_tok

def prepare_contrastive_data(config):
    c_df = create_contrastive_dataset(data_en_target, config.val_test_lang)
    c_df = c_df.iloc[torch.randperm(len(c_df))]
    c_df["{}_tokenized".format(config.experiment_lang)].iloc[:100000] = c_df["{}_tokenized".format(config.experiment_lang)].iloc[:100000].apply(lambda x: [np.random.choice(x) for s in range(len(x)-1)])
    shuffle_ix = torch.randperm(len(c_df))
    c_df["{}_tokenized".format(config.experiment_lang)] = np.array(c_df["{}_tokenized".format(config.experiment_lang)])[shuffle_ix]
    c_df["len_en"] = c_df["en_tokenized"].apply(lambda x: len(x))
    c_df["len_{}".format(config.val_test_lang)] = \
    c_df["{}_tokenized".format(config.val_test_lang)].apply(lambda x: len(x))
    c_df = c_df[(c_df["len_en"] > 1)&(c_df["len_{}".format(config.val_test_lang)] > 1)]
    return c_df

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

print ("Loading vectors for EN.")
aligned_src_vectors = load_glove_vectors("en")
print ("Loading vectors for {}.".format(config.experiment_lang.upper()))
aligned_trg_vectors = load_aligned_vectors(config.experiment_lang)

id2token_src = [x+"."+"en" for x in [*aligned_src_vectors.keys()]][:config.max_vocab_size]
id2token_trg = [x+"."+config.experiment_lang for x in [*aligned_trg_vectors.keys()]][:config.max_vocab_size]
id2token_mutual = ["<PAD>", "<UNK>"] + id2token_src + id2token_trg
vecs_mutual = update_vocab_keys(aligned_src_vectors, aligned_trg_vectors)
token2id_mutual = build_tok2id(id2token_mutual)
weights_init = init_embedding_weights(vecs_mutual, token2id_mutual, id2token_mutual, config.embed_dim)

data_en_target, all_en_tokens, all_target_tokens = read_and_tokenize_europarl_data(lang=config.val_test_lang)
c_df = prepare_contrastive_data(config)

align_dataset = AlignDataset(data_en_target, config.max_sent_len, "en", config.experiment_lang,
                             token2id_mutual, id2token_mutual)
align_loader = torch.utils.data.DataLoader(dataset=align_dataset, batch_size=config.batch_size,
                                           collate_fn=lambda x, max_sentence_length=config.max_sent_len: align_collate_func(x, config.max_sent_len),
                                           shuffle=False)
c_align_dataset = AlignDataset(c_df, config.max_sent_len, "en", config.experiment_lang,
                               token2id_mutual, id2token_mutual)
c_align_loader = torch.utils.data.DataLoader(dataset=c_align_dataset, batch_size=config.batch_size,
                                             collate_fn=lambda x, max_sentence_length=config.max_sent_len: align_collate_func(x, config.max_sent_len),
                                             shuffle=False)

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
        if np.random.random() <= 0.3:
            src_data = src_data + torch.rand(src_data.size()).long().to(device)
            trg_data = trg_data + torch.rand(trg_data.size()).long().to(device)
        
        src_out = LSTM_src(src_data, src_mask, src_len)
        trg_out = LSTM_trg(trg_data, trg_mask, trg_len)
        src_c_out = LSTM_src(src_c, src_mc, src_len_c)
        trg_c_out = LSTM_trg(trg_c, trg_mc, trg_len_c)
        loss = loss_align(src_out, trg_out, src_c_out, trg_c_out, 0.25)
        loss.cuda().backward(retain_graph=True)
        optimizer.step()
        total_loss += loss.item() * len(src_data) / len(loader.dataset)
        if (batch_idx + 1) % (len(loader.dataset)//(50 * config.batch_size)) == 0:
            
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(epoch, (batch_idx+1) * config.batch_size, len(loader.dataset), 100. * (batch_idx+1) / len(loader), loss.item()))

optimizer.zero_grad()
return total_loss

load_epoch = 2
LSTM_src_model = biLSTM(hidden_size=config.hidden_dim, embedding_weights=weights_init,
                        num_layers = 1, percent_dropout = config.dropout, vocab_size=weights_init.shape[0],
                        input_size = config.embed_dim).to(device)

LSTM_src_model.load_state_dict(torch.load("best_encoder_eng_mnli_{}_{}".format(load_epoch, config.experiment_lang)))

# fix source encoder parameters
for param in LSTM_src_model.parameters():
    param.requires_grad = False

LSTM_trg_model = biLSTM(hidden_size = config.hidden_dim, embedding_weights = weights_init,
                        num_layers = 1, percent_dropout = config.dropout, vocab_size = weights_init.shape[0],
                        input_size  = config.embed_dim).to(device)

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
        
    torch.save(LSTM_trg_model.state_dict(), "LSTM_en_{}_{}_epoch_{}".format(config.experiment_lang, config.experiment_lang.upper(), epoch))


