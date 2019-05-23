import random
import sys
import functools
import numpy as np
import pandas as pd
from setuptools import setup
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.autograd import Variable



def train(RNN, Linear_Classifier, DataLoader, criterion, optimizer, epoch):
    
    RNN.train()
    Linear_Classifier.train()
    total_loss = 0
    for batch_idx, (sentence1, s1_mask, sentence1_lengths, 
                    sentence2, s2_mask, sentence2_lengths, labels) in enumerate(DataLoader):

        sentence1, s1_mask = sentence1.to(device), s1_mask.to(device),  
        sentence2, s2_mask = sentence2.to(device), s2_mask.to(device),
        labels = torch.from_numpy(labels).to(device)
        RNN.train()
        Linear_Classifier.train()
        optimizer.zero_grad()
        output_s1 = RNN(sentence1, s1_mask, sentence1_lengths)
        output_s2 = RNN(sentence2, s2_mask, sentence2_lengths)
        out = Linear_Classifier(output_s1, output_s2)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(sentence1) / len(DataLoader.dataset)
        
        if (batch_idx+1) % (len(DataLoader.dataset)//(20*labels.shape[0])) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx+1) * labels.shape[0], len(DataLoader.dataset),
                100. * (batch_idx+1) / len(DataLoader), loss.item()), end="\r")

    optimizer.zero_grad()
    return total_loss


def test(RNN, Linear_Classifier, DataLoader, criterion):

    RNN.eval()
    Linear_Classifier.eval()
    test_loss = 0
    label_list = []
    output_list = []
    with torch.no_grad():
        for batch_idx, (sentence1, s1_mask, sentence1_lengths, 
                        sentence2, s2_mask, sentence2_lengths, labels) in enumerate(DataLoader):

            sentence1, s1_mask = sentence1.to(device), s1_mask.to(device),  
            sentence2, s2_mask = sentence2.to(device), s2_mask.to(device),
            labels = torch.from_numpy(labels).to(device)
            output_s1 = RNN(sentence1, s1_mask, sentence1_lengths)
            output_s2 = RNN(sentence2, s2_mask, sentence2_lengths)
            out = Linear_Classifier(output_s1, output_s2)
            loss = criterion(out, labels)
            test_loss += loss.item()/len(DataLoader.dataset)
            output_list.append(out)
            label_list.append(labels)
            
    return test_loss, torch.cat(output_list, dim=0), torch.cat(label_list, dim=0)

def accuracy(RNN, Linear_Classifier, DataLoader, criterion):
    _, predicted, true_labels = test(RNN = RNN,  Linear_Classifier = Linear_Classifier, DataLoader = DataLoader, criterion = criterion)
    predicted = predicted.max(1)[1]
    return 100 * predicted.eq(true_labels.data.view_as(predicted)).float().mean().item()