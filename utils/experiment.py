from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import csv
import random
import spacy
import functools
import itertools
import pdb
import io
import os
from underthesea import word_tokenize
from collections import defaultdict, Counter
import jieba
import numpy as np
from docopt import docopt
from setuptools import setup
import pandas as pd
import time
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import preprocessing_utils
import nli_models

mnli_path = "../../data/MultiNLI"
snli_path = "../../data/SNLI"
xnli_path = "../../data/XNLI"
aligned_vector_path = "../../data/aligned_embeddings"
glove_path = "../../data/glove"
languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh"]

# TODO: loading vectors for every language, translation preprocessing

lang2vector = preprocessing_utils.build_aligned_vector_dict(langs = languages)

def train_eval_model_nli(train_dataset_dir, dev_dataset_dir, dev_lang, encoder_model, linear_model, criterion, encoder_optimizer, linear_optimizer, nli_data_type, 
	VALIDATE_EVERY, VOCAB_SIZE, MAX_SENTENCE_LENGTH, N_EPOCHS, BATCH_SIZE, LEARNING_RATE):
	print ("Reading data.")
	mnli_train, mnli_dev, mnli_test, snli_train, snli_dev, snli_test = preprocessing_utils.read_enli(mnli_path, snli_path)
	xnli_dev, _ = preprocessing_utils.read_xnli(xnli_path)
	if dev_lang == None:
		xnli_dev = xnli_dev
	else:
		xnli_dev = xnli_dev[xnli_dev["language"]==dev_lang]

	if nli_data_type == "snli":
		train_data, dev_data = snli_train, xnli_dev
	elif nli_data_type == "mnli":
		train_data, dev_data = mnli_train, xnli_dev
	else:
		raise ValueError("nli_data type should be in: 'mnli', 'snli'")
	print ("Tokenizing data.")
	tokenized_train, all_train_tokens = preprocessing_utils.tokenize_enli(train_data, remove_punc=False)
	tokenized_dev, _ = preprocessing_utils.tokenize_xnli(dev_data, remove_punc=False)
	token2id, id2token = preprocessing_utils.build_vocab(all_train_tokens, VOCAB_SIZE)
 	train_dataset = preprocessing_utils.NLIDataset(tokenized_train, MAX_SENTENCE_LENGTH)
 	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, collate_fn=preprocessing_utils.nli_collate_func, shuffle=False)
 	dev_dataset = preprocessing_utils.NLIDataset(tokenized_dev, MAX_SENTENCE_LENGTH)
 	dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, collate_fn=preprocessing_utils.nli_collate_func, shuffle=False)

 	vectors = # TODO

 	encoder = encoder_model() # TODO, hidden_size, vocab_size etc
 	encoder_opt = encoder_optimizer(encoder.parameters(), lr=LEARNING_RATE)
 	linear_layer = linear_model() # TODO, hidden_size, vocab_size etc
 	linear_opt = linear_optimizer(linear_layer.parameters(), lr=LEARNING_RATE)

 	print ("Starting training.")
	for epoch in range(N_EPOCHS):
		for i, (sentence1, s1_mask, s1_lengths, sentence2, s2_mask, s2_lengths, labels) in enumerate(train_loader):
			sentence1, s1_mask, s1_lengths, sentence2, s2_mask, s2_lengths, labels = sentence1.to(device), s1_mask.to(device), s1_lengths.to(device), sentence2.to(device), s2_mask.to(device), s2_lengths.to(device), labels.to(device)
            encoder.train()
            linear_layer.train()
            encoder_opt.zero_grad()
            linear_opt.zero_grad()
            encoder_outputs = encoder() ## TODO
            outputs = linear_layer(encoder_outputs)
            loss = criterion(outputs, labels)
            loss.backward()
            encoder_opt.step()
            linear_opt.step()
            if i > 0 and i % VALIDATE_EVERY == 0:
                val_acc = test_model(dev_loader, model)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))

def train_eval_model_translation():


def validate_model_nli():

def validate_model_translation():

def test_model_nli():

def test_model_translation():


def hyperparam_search(train_dataset_dir, dev_dataset_dir, model_name, criterion, optimizer, nli_data_type, 
	VALIDATE_EVERY, MAX_SENTENCE_LENGTH, N_EPOCHS, BATCH_SIZE):


def main():
    arguments = docopt(__doc__)

    if arguments["train_eval_model_nli"]:
    	train_eval_model_nli(arguments["<train_dataset_dir>"],
    		        arguments["<dev_dataset_dir>"],
    		        arguments["<dev_lang>"],
    		        arguments["<encoder_model>"],
    		        arguments["<linear_model>"],
    		        arguments["<criterion>"],
    		        arguments["<encoder_optimizer>"],
    		        arguments["<linear_optimizer>"],
                    arguments["<nli_data_type>"],
                    int(arguments["--VALIDATE_EVERY"]),
                    int(arguments["--VOCAB_SIZE"]),
                    int(arguments["--MAX_SENTENCE_LENGTH"]),
                    int(arguments["--N_EPOCHS"]),
                    int(arguments["--BATCH_SIZE"]),
                    float(arguments["--LEARNING_RATE"])
                    )

    elif arguments["train_eval_model_translation"]:
    	train_eval_model_translation(arguments["<train_dataset_dir>"],
    		        arguments["<dev_dataset_dir>"],
    		        arguments["<encoder_model>"],
    		        arguments["<decoder_model>"],
    		        arguments["<criterion>"],
    		        arguments["<encoder_optimizer>"],
    		        arguments["<decoder_optimizer>"],
                    int(arguments["--VALIDATE_EVERY"]),
                    int(arguments["--VOCAB_SIZE"]),
                    int(arguments["--MAX_SENTENCE_LENGTH"]),
                    int(arguments["--N_EPOCHS"]),
                    int(arguments["--BATCH_SIZE"]),
                    arguments["--LANG1"],
                    arguments["--LANG2"],
                    float(arguments["--LEARNING_RATE"])
                    )

    elif arguments["validate_model_nli"]:
        validate_model_nli(arguments["<dev_dataset_dir>"],
        	arguments["<dev_lang>"],
                  	arguments["<model_name>"],
                    arguments["<criterion>"],
                    int(arguments["--MAX_SENTENCE_LENGTH"]),
                    int(arguments["--BATCH_SIZE"]),
                    float(arguments["--LEARNING_RATE"])
                    )

    elif arguments["validate_model_translation"]:
        validate_model_translation(arguments["<dev_dataset_dir>"],
        	arguments["<dev_lang>"],
                  	arguments["<model_name>"],
                    arguments["<criterion>"],
                    int(arguments["--MAX_SENTENCE_LENGTH"]),
                    int(arguments["--BATCH_SIZE"]),
                    arguments["--LANG1"],
                    arguments["--LANG2"],
                    float(arguments["--LEARNING_RATE"])
                    )

    elif arguments["test_model_nli"]:
    	test_model_nli(arguments["<test_dataset_dir>"],
    		       arguments["<model_name>"],
                   arguments["<criterion>"],
    		       int(arguments["--MAX_SENTENCE_LENGTH"]),
                   int(arguments["--BATCH_SIZE"]),
                   float(arguments["--LEARNING_RATE"])
                   )

    elif arguments["test_model_translation"]:
    	test_model_translation(arguments["<test_dataset_dir>"],
    		       arguments["<model_name>"],
                   arguments["<criterion>"],
    		       int(arguments["--MAX_SENTENCE_LENGTH"]),
                   int(arguments["--BATCH_SIZE"]),
                   arguments["--LANG1"],
                   arguments["--LANG2"],
                   float(arguments["--LEARNING_RATE"])
                   )

    elif arguments["hyperparam_search"]:
    	hyperparam_search(arguments["<train_dataset_dir>"],
    		arguments["<dev_dataset_dir>"],
    		arguments["<model_name>"],
    		arguments["<criterion>"],
    		arguments["<optimizer>"],
    		arguments["<nli_data_type>"],
    		int(arguments["--VALIDATE_EVERY"]),
    		int(arguments["--MAX_SENTENCE_LENGTH_LIST"]),
            int(arguments["--N_EPOCHS"]),
            int(arguments["--BATCH_SIZE_LIST"]),
                   arguments["--LANG1"],
                   arguments["--LANG2"],
                    float(arguments["--LEARNING_RATE_LIST"])
                    )

    elif arguments["get_vector_set"]:
    	preprocessing_utils.load_vectors(arguments["<fname>"])


if __name__ == '__main__':
    main()
