import pickle
import random
import random
import spacy
import csv
import string
import io
import os
import re
import torch
import functools
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict
import spacy
from torch.utils.data import Dataset

vector_path = "../../data/aligned_embeddings"
xnli_path = "../../data/XNLI"
mnli_path = "../../data/MultiNLI"
snli_path = "../../data/SNLI"
languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh"]

reg = re.compile("[%s]" % re.escape(string.punctuation))
PAD_IDX = 0
UNK_IDX = 1
MAX_SENTENCE_LENGTH = 100
BATCH_SIZE = 64

def load_vectors(fname):
    fin = io.open(fname, "r", encoding="utf-8", newline="\n", errors="ignore")
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(" ")
        data[tokens[0]] = [*map(float, tokens[1:])]
    return data

def build_lang_dicts(langs = languages):
	language_dict = defaultdict(dict)
	for x in langs:
	    print ("loading vectors for", x)
	    fname = "{}/wiki.{}.align.vec".format(vector_path, x)
	    language_dict[x]["vectors"] = load_vectors(fname)
    return language_dict

def read_xnli():
	xnli_dev = pd.read_csv("{}/xnli.{}.tsv".format(xnli_path, "dev"), sep="\t")
	xnli_test = pd.read_csv("{}/xnli.{}.tsv".format(xnli_path, "test"), sep="\t")
	return xnli_dev, xnli_test

def read_enli(path_mnli = mnli_path, path_snli = snli_path):
	mnli_train = pd.read_json("{}/multinli_1.0_{}.jsonl".format(mnli_path, "train"), lines=True)
	mnli_dev = pd.read_json("{}/multinli_1.0_{}.jsonl".format(mnli_path, "dev"), lines=True)
	mnli_test = pd.read_json("{}/multinli_1.0_{}.jsonl".format(mnli_path, "test"), lines=True)
	snli_train = pd.read_json("{}/snli_1.0_{}.jsonl".format(mnli_path, "train"), lines=True)
	snli_dev = pd.read_json("{}/snli_1.0_{}.jsonl".format(mnli_path, "dev"), lines=True)
	snli_test = pd.read_json("{}/snli_1.0_{}.jsonl".format(mnli_path, "test"), lines=True)
	return mnli_train, mnli_dev, mnli_test, snli_train, snli_dev, snli_test

def read_snli():
	## TODO

def tokenize_dataset(dataset, remove_punc=False):
all_s1_tokens = []
all_s2_tokens = []
for s in ["sentence1", "sentence2"]:
    if remove_punc:
        punc = [*string.punctuation]
        dataset["{}_tokenized".format(s)] = dataset["{}_tokenized".format(s)].\
        apply(lambda x: "".join(c for c in x if c not in punc).lower().split(" "))
    else:
        dataset["{}_tokenized".format(s)] = dataset["{}_tokenized".format(s)].\
        apply(lambda x: x.lower().split(" "))
dataset["sentence1_tokenized"].apply(lambda x: all_s1_tokens.extend(x))
dataset["sentence2_tokenized"].apply(lambda x: all_s2_tokens.extend(x))
all_tokens = all_s1_tokens + all_s2_tokens
return dataset, all_tokens

def tokenize_enli(dataset, remove_punc=True):
    punc = string.punctuation
    all_s1_tokens = []
    all_s2_tokens = []
    for s in [1,2]:
        if remove_punc:
            dataset["sentence{}_tokenized".format(s)] = dataset["sentence{}".format(s)].\
            apply(lambda x: reg.sub("", x).lower().split(" "))
        else:
            dataset["sentence{}_tokenized".format(s)] = dataset["sentence{}".format(s)].\
            apply(lambda x: (reg.sub("", x) + " .").lower().split(" "))
    print ("Tokenizing done.")
    dataset["sentence1_tokenized"].apply(lambda x: all_s1_tokens.extend(x))
    dataset["sentence2_tokenized"].apply(lambda x: all_s2_tokens.extend(x))
    print ("Token collection done.")
    all_tokens = all_s1_tokens + all_s2_tokens
    print ("Concatenation done.")
    return dataset, all_tokens


def build_vocab(all_tokens, max_vocab_size):
    token_counter = Counter(all_tokens)
    vocab, count = zip(*token_counter.most_common(max_vocab_size))
    id2token = [*vocab]
    token2id = dict(zip(vocab, range(2,2+len(vocab))))
    id2token = ['<PAD>', '<UNK>'] + id2token
    token2id["<PAD>"] = PAD_IDX
    token2id["<UNK>"] = UNK_IDX
    return token2id, id2token


class XNLILang:
    def __init__(self, name, max_vocab_size):
        self.name = name
        self.xnli_dev, self.xnli_test = language_dict[self.name]["xnli_dev"], language_dict[self.name]["xnli_test"] 
        self.tokenized_dev, self.all_dev_tokens = tokenize_dataset(self.xnli_dev, remove_punc=False)
        self.tokenized_test, _ = tokenize_dataset(self.xnli_test, remove_punc=False)
        self.token2id, self.id2token = build_vocab(self.all_dev_tokens, max_vocab_size)

class ENLILang:
    def __init__(self, name, max_vocab_size, nli_corpus="mnli"):
        self.name = name
        self.tokenized_train_data, self.all_train_tokens = tokenize_enli(mnli_train)
        self.train_tokens = all_train_tokens
        self.xnli_dev, self.xnli_test = language_dict[self.name]["xnli_dev"], language_dict[self.name]["xnli_test"] 
        self.tokenized_dev, self.all_dev_tokens = tokenize_dataset(self.xnli_dev, remove_punc=False)
        self.tokenized_test, _ = tokenize_dataset(self.xnli_test, remove_punc=False)
        self.token2id, self.id2token = build_vocab(all_train_tokens, max_vocab_size)
    def get_tokens(nli_corpus):
    	assert (nli_corpus in ["mnli","snli"]), "NLI corpus should be either 'mnli' or 'snli'."
    	if nli_corpus == "mnli":
    		self.tokenized_train_data, self.all_train_tokens = tokenize_enli(mnli_train)
    	elif nli_corpus == "snli":
    		self.tokenized_train_data, self.all_train_tokens = tokenize_enli(snli_train)
    	return self

class NLIDataset(Dataset):
    def __init__(self, lang, max_sentence_length=MAX_SENTENCE_LENGTH):
        self.sentence1, self.sentence2, self.labels = lang.tokenized_train_data["sentence1_tokenized"].values, \
                                                      lang.tokenized_train_data["sentence2_tokenized"].values, \
                                                      lang.tokenized_train_data["gold_label"].values
        self.max_sentence_length = max_sentence_length
        self.token2id, self.id2token = lang.token2id, lang.id2token
        
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

def nli_func(batch, max_sent_length):
    s1_data, s2_data = [], []
    s1_mask, s2_mask = [], []
    s1_lengths, s2_lengths = [], []
    labels = []

    for datum in batch:
        s1_lengths.append(datum[2])
        s2_lengths.append(datum[5])
        labels.append(datum[6])
        sentence1_data_padded = np.pad(np.array(datum[0]), pad_width=((0, MAX_SENTENCE_LENGTH-datum[2])), mode="constant", constant_values=0)
        sentence1_data.append(sentence1_data_padded)
        sentence1_mask_padded = np.pad(np.array(datum[1]), pad_width=((0, MAX_SENTENCE_LENGTH-datum[2])), mode="constant", constant_values=0)
        sentence1_mask.append(sentence1_mask_padded)
        sentence2_data_padded = np.pad(np.array(datum[3]), pad_width=((0, MAX_SENTENCE_LENGTH-datum[5])), mode="constant", constant_values=0)
        sentence2_data.append(sentence2_data_padded)
        sentence2_mask_padded = np.pad(np.array(datum[4]), pad_width=((0, MAX_SENTENCE_LENGTH-datum[5])), mode="constant", constant_values=0)
        sentence2_mask.append(sentence2_mask_padded)
        
    ind_dec_order = np.argsort(s1_lengths)[::-1]
    s1_data = np.array(s1_data)[ind_dec_order]
    s2_data = np.array(s2_data)[ind_dec_order]
    s1_mask = np.array(s1_mask)[ind_dec_order].reshape(len(batch), -1, 1)
    s2_mask = np.array(s2_mask)[ind_dec_order].reshape(len(batch), -1, 1)
    s1_lengths = np.array(s1_lengths)[ind_dec_order]
    s2_lengths = np.array(s2_lengths)[ind_dec_order]
    labels = np.array(labels)[ind_dec_order]
    
    s1_list = [torch.from_numpy(s1_data), torch.from_numpy(s1_mask).float(), s1_lengths]
    s2_list = [torch.from_numpy(s2_data), torch.from_numpy(s2_mask).float(), s2_lengths]
        
    return s1_list + s2_list + [torch.from_numpy(labels)]

def preprocess_main():
	# TODO