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

import utils.preprocess_nli

no_cuda = False
cuda = not no_cuda and torch.cuda.is_available()
seed = 1
device = torch.device("cuda" if cuda else "cpu")

label_dict = {"entailment":0,
             "neutral":1,
             "contradiction":2}

# load train vecs
print ("Loading standard training vectors..")
word_vectors = preprocess_nli.load_glove("vector_cache/glove.840B.300d.txt")
token2id_train, id2token_train = preprocess_nli.build_vector_vocab(word_vectors)
word_vector_tensor = torch.from_numpy(np.array(pd.DataFrame(word_vectors).T)).float()

class config_class:
    def __init__(self, max_sentence_length, corpus, val_test_lang, epochs, batch_size, encoder_type, 
                 activation, optimizer, embed_dim, fc_dim, hidden_dim, layers, dropout, learning_rate,
                 lr_patience, lr_decay, lr_reduction_factor, weight_decay,
                 preserve_case, word_embedding, resume_snapshot, early_stopping_patience,
                 save_path, seed):
        
        self.max_sentence_length = max_sentence_length
        self.corpus = corpus
        self.val_test_lang = val_test_lang
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder_type = encoder_type
        self.activation = activation
        self.optimizer = optimizer
        self.embed_dim = embed_dim
        self.fc_dim = fc_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.lr_patience = lr_patience
        self.lr_decay = lr_decay
        self.lr_reduction_factor = lr_reduction_factor
        self.weight_decay = weight_decay
        self.preserve_case = preserve_case
        self.word_embedding = word_embedding
        self.resume_snapshot = resume_snapshot
        self.early_stopping_patience = early_stopping_patience
        self.save_path = save_path
        self.seed = seed
        self.lower = True
        self.embed_size = self.train_vectors.size(0)

config = config_class(max_sentence_length = 30, 
	corpus = "multinli", 
	val_test_lang = "tr", 
	epochs = 20, 
	batch_size = 16, 
	encoder_type = "HBMP", 
	activation = "relu", 
	optimizer = "adam", 
	embed_dim = 300, 
	fc_dim = 300, 
	hidden_dim = 300,
	layers = 1, 
	dropout = 0, 
	learning_rate = 5e-4, 
	lr_patience = 1, 
	lr_decay = 0.99, 
	lr_reduction_factor = 0.2, 
	weight_decay = 0, 
	preserve_case = "store_false",
	word_embedding = "glove.840B.300d", 
	resume_snapshot = "", 
	early_stopping_patience =  3,
	save_path = "results", 
	seed = 1234)

# load val vecs
print ("Loading standard multilingual vectors.")
multilingual_val_vectors = preprocess_nli.load_multilingual_vectors(config.val_test_lang)
token2id_val, id2token_val = preprocess_nli.build_vector_vocab(multilingual_val_vectors)
multilingual_val_vectors = torch.from_numpy(np.array(pd.DataFrame(multilingual_val_vectors).T)).float()
print ("Loading aligned multilingual vectors.")
aligned_val_vectors = preprocess_nli.load_aligned_vectors(config.val_test_lang)
aligned_val_vectors = torch.from_numpy(np.array(pd.DataFrame(aligned_val_vectors).T)).float()

config.train_vectors = word_vector_tensor
config.aligned_val_vectors = aligned_val_vectors
config.multilingual_val_vectors = multilingual_val_vectors

# load train and preprocess
print ("Reading {} data.".format(config.corpus))
nli_train, nli_dev, nli_test = preprocess_nli.read_enli(nli_corpus=config.corpus)
print ("Writing numeric label.")
if config.corpus == "multinli":
	nli_train, nli_dev, _ = preprocess_nli.write_numeric_label(nli_train, nli_dev, nli_test, nli_corpus=config.corpus)
elif config.corpus == "snli":
	nli_train, nli_dev, nli_test = preprocess_nli.write_numeric_label(nli_train, nli_dev, nli_test, nli_corpus=config.corpus)

# load val and test and preprocess
print ("Reading XNLI {} data.".format(config.val_test_lang.upper()))
xnli_dev, xnli_test = preprocess_nli.read_xnli(config.val_test_lang)




if __name__ == '__main__':
    main()