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
from sacrebleu import sacrebleu
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

train_dataset_dir = "../../data/"
mnli_path = "../../data/MultiNLI"
snli_path = "../../data/SNLI"
xnli_path = "../../data/XNLI"
aligned_vector_path = "../../data/aligned_embeddings"
glove_path = "../../data/glove"
languages = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "th", "tr", "vi", "zh"]

# TODO: loading vectors for every language, translation preprocessing

# set VOCAB_SIZE = embed_table.size(0)

lang2vector = preprocessing_utils.build_aligned_vector_dict(langs = languages)

## write function to prepare translation data

def prepare_nli_train_data(nli_corpus, VOCAB_SIZE, MAX_SENTENCE_LENGTH, BATCH_SIZE):
	# TODO: Add vectors

	train_data, _, _ = preprocessing_utils.read_enli(nli_corpus=nli_corpus)
	print ("Tokenizing train data.")
	tokenized_train, all_train_tokens = preprocessing_utils.tokenize_enli(train_data, remove_punc=False)
	token2id, id2token = preprocessing_utils.build_vocab(all_train_tokens, VOCAB_SIZE)
	train_dataset = preprocessing_utils.NLIDataset(tokenized_train, MAX_SENTENCE_LENGTH, token2id, id2token)
	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, 
		collate_fn=preprocessing_utils.nli_collate_func, shuffle=False)
	return train_loader, token2id, id2token

def prepare_nli_dev_data(dev_lang, MAX_SENTENCE_LENGTH, BATCH_SIZE, token2id, id2token):
	# TODO: Add vectors

	xnli_dev, _ = preprocessing_utils.read_xnli(xnli_path)
	print ("Tokenizing dev data.")
	if dev_lang == "all":
		dev_data = xnli_dev
	elif dev_lang in languages:
		dev_data = xnli_dev[xnli_dev["language"]==dev_lang]
	else:
		raise ValueError("Please specify a valid language or 'all'.")
	tokenized_dev, _ = preprocessing_utils.tokenize_xnli(dev_data, remove_punc=False)
	dev_dataset = preprocessing_utils.NLIDataset(tokenized_dev, MAX_SENTENCE_LENGTH, token2id, id2token)
	dev_loader = torch.utils.data.DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, 
		collate_fn=preprocessing_utils.nli_collate_func, shuffle=False)
	return dev_loader

def prepare_nli_test_data(test_lang, MAX_SENTENCE_LENGTH, BATCH_SIZE, token2id, id2token):
	# TODO: Add vectors

	_, xnli_test = preprocessing_utils.read_xnli(xnli_path)
	print ("Tokenizing test data.")
	if test_lang == "all":
		test_data = xnli_test
	elif test_lang in languages:
		test_data = xnli_test[xnli_test["language"]==test_lang]
	else:
		raise ValueError("Please specify a valid language or 'all'.")
	tokenized_test, _ = preprocessing_utils.tokenize_xnli(test_data, remove_punc=False)
	test_dataset = preprocessing_utils.NLIDataset(tokenized_test, MAX_SENTENCE_LENGTH, token2id, id2token)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, 
		collate_fn=preprocessing_utils.nli_collate_func, shuffle=False)
	return test_loader

def train_eval_model_nli(train_loader, dev_loader, dev_lang, encoder_model, linear_model, criterion, encoder_optimizer, linear_optimizer,
	enc_hidden_size, interaction_type, num_encoder_layers, enc_input_size, linear_hidden_size, encoder_dropout, linear_dropout, num_classes, linear_input_size,
	VALIDATE_EVERY, VOCAB_SIZE, EMBED_WEIGHTS, N_EPOCHS, LEARNING_RATE):

 	encoder = encoder_model(enc_input_size, hidden_size, EMBED_WEIGHTS, VOCAB_SIZE, encoder_dropout, num_encoder_layers)
 	encoder_opt = encoder_optimizer(encoder.parameters(), lr=LEARNING_RATE)
 	linear_layer = linear_model(linear_hidden_size, linear_dropout, interaction_type, num_classes, linear_input_size)
 	linear_opt = linear_optimizer(linear_layer.parameters(), lr=LEARNING_RATE)
 	print ("Starting training.")
 	val_accs = []
	for epoch in range(N_EPOCHS):
		for i, (sentence1, s1_mask, s1_lengths, sentence2, s2_mask, s2_lengths, labels) in enumerate(train_loader):
			sentence1, s1_mask, s1_lengths, sentence2, s2_mask, s2_lengths, labels = sentence1.to(device), s1_mask.to(device), s1_lengths.to(device), sentence2.to(device), s2_mask.to(device), s2_lengths.to(device), labels.to(device)
            encoder.train()
            linear_layer.train()
            encoder_opt.zero_grad()
            linear_opt.zero_grad()
            outputs_s1 = encoder(sentence1, s1_mask, s1_lengths) 
            outputs_s2 = encoder(sentence2, s2_mask, s2_lengths) 
            out = linear_layer(outputs_s1, outputs_s2)
            loss = criterion(out, labels)
            loss.backward()
            encoder_opt.step()
            linear_opt.step()
            if i > 0 and i % VALIDATE_EVERY == 0:
                val_acc = dev_accuracy(encoder, linear_layer, dev_loader, criterion)
                val_accs.append(val_acc)
                print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                           epoch+1, num_epochs, i+1, len(train_loader), val_acc))

    return val_accs

def train_eval_model_translation():
	# TODO


def validate_model_nli(loader, encoder_model, linear_model, criterion):
	encoder_model.eval()
    linear_model.eval()
    dev_loss = 0
    label_list, output_list = [], []
    with torch.no_grad():
    	for i, (sentence1, s1_mask, s1_lengths, sentence2, s2_mask, s2_lengths, labels) in enumerate(loader):
    		sentence1, s1_mask = sentence1.to(device), s1_mask.to(device),  
            sentence2, s2_mask = sentence2.to(device), s2_mask.to(device),
            labels = labels.to(device)
            output_s1 = encoder_model(sentence1, s1_mask, s1_lengths)
            output_s2 = encoder_model(sentence2, s2_mask, s2_lengths)
            out = linear_model(output_s1, output_s2)
            loss = criterion(out, labels)
            dev_loss += loss.item()/len(dev_loader.dataset)
            output_list.append(out)
            label_list.append(labels)
    return dev_loss, torch.cat(output_list, dim=0), torch.cat(label_list, dim=0)

def accuracy(encoder_model, linear_model, loader, criterion):
	_, predicted, true_labels = validate_model_nli(loader, encoder_model, linear_model, criterion)
	predicted = predicted.max(1)[1]
	return 100 * predicted.eq(true_labels.data.view_as(predicted)).float().mean().item()

def validate_model_translation():
	## get sacrebleu

def test_model_nli(loader, encoder_model, linear_model, criterion):
	encoder_model.eval()
    linear_model.eval()
    test_loss = 0
    label_list, output_list = [], []
    with torch.no_grad():
    	for i, (sentence1, s1_mask, s1_lengths, sentence2, s2_mask, s2_lengths, labels) in enumerate(loader):
    		sentence1, s1_mask = sentence1.to(device), s1_mask.to(device),  
            sentence2, s2_mask = sentence2.to(device), s2_mask.to(device),
            labels = labels.to(device)
            output_s1 = encoder_model(sentence1, s1_mask, s1_lengths)
            output_s2 = encoder_model(sentence2, s2_mask, s2_lengths)
            out = linear_model(output_s1, output_s2)
            loss = criterion(out, labels)
            test_loss += loss.item()/len(dev_loader.dataset)
            output_list.append(out)
            label_list.append(labels)
    return test_loss, torch.cat(output_list, dim=0), torch.cat(label_list, dim=0)

def test_accuracy(loader, encoder_model, linear_model, criterion):
	_, predicted, true_labels = test_model_nli(loader, encoder_model, linear_model, criterion)
	predicted = predicted.max(1)[1]
	return 100 * predicted.eq(true_labels.data.view_as(predicted)).float().mean().item()

def test_model_translation():
	# TODO

def hyperparam_search(train_loader, dev_loader, dev_lang, encoder_model, linear_model, criterion, encoder_optimizer, linear_optimizer, enc_input_size, num_classes,
	linear_input_size, 
	encoder_optimizer_list, linear_optimizer_list, lr_list, enc_hidden_list, linear_hidden_list, epsilon_list, 
	weight_decay_list, enc_dropout_list, linear_dropout_list, num_encoder_layers_list, interaction_types,
	VALIDATE_EVERY, MAX_SENTENCE_LENGTH, N_EPOCHS, VOCAB_SIZE, EMBED_WEIGHTS):
	
	hyper_params = {"LEARNING_RATE":lr_list, "enc_hidden_size":enc_hidden_list, "linear_hidden_size":linear_hidden_list, "epsilon":epsilon_list, 
					"weight_decay":weight_decay_list, "encoder_dropout":enc_dropout_list, "linear_dropout":linear_dropout_list,
					 "num_encoder_layers":num_encoder_layers_list, "encoder_optimizer":encoder_optimizer_list, "linear_optimizer":linear_optimizer_list
                    "interaction_type":interaction_types}

    param_names = [*hyper_params.keys()]
    param_sets = [*itertools.product(*[hyper_params[key] for key in param_names])]
    total_step = len(train_loader)
    n_param_sets = len(param_sets)
    dev_accs = {}

    for p_ in range(n_param_sets):
    	dev_accs[param_sets[p_]] = []
    	if p_ % 5 == 0:
    		print ("{}/{} complete.".format(p_, n_param_sets))
    	params = param_sets[p_]
    	[LEARNING_RATE, enc_hidden_size, linear_hidden_size, epsilon, weight_decay, encoder_dropout, linear_dropout,\
    	num_encoder_layers, encoder_optimizer, linear_optimizer, interaction_type] = [params[param_names.index(x)] for x in param_names]

    	dev_accs[params] = train_eval_model_nli(train_loader, dev_loader, dev_lang, encoder_model, linear_model, criterion, encoder_optimizer, linear_optimizer,
    		enc_hidden_size, interaction_type, num_encoder_layers, enc_input_size, linear_hidden_size, encoder_dropout, linear_dropout,
    		num_classes, linear_input_size, VALIDATE_EVERY, VOCAB_SIZE, EMBED_WEIGHTS, N_EPOCHS, LEARNING_RATE )

    return dev_accs