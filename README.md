# Cross-Lingual NLI from Scratch
## What is NLI (Natural Language Inference)?

Given a premise-hypothesis pair, NLI is the task of understanding whether the premise _entails_ the hypothesis, whether it _contradicts_ the hypothesis or neither (the relationship is _neutral_).

<img src="https://github.com/adcengiz/multi-lingual-nli/presentation/spanish_chinese_accuracy.png" width="48">

__Cross-lingual NLI__ involves training a natural language inference model in a language and predicting entailment labels for data in another language. For example, in this project, we train an NLI model on MultiNLI data - which is available only in English - and evaluate it for use in other languages. 

### Train XNLI Folder

Holds the notebook that explains how a target language encoder is aligned to a source language encoder using parallel corpora, and how we use the aligned encoder to perform ```cross-lingual NLI without translation```. Also, below is an explanation of the whole process.

### SNLI-Only

In this folder you can find the notebook and .py files with directions for training an ```SNLI model``` - English-only.

### MultiNLI-Only

Here, you can find the notebook and .py files with directions for training a ```MultiNLI model``` - English-only.

### Translate-Train

Translate-Train method uses machine translation to generate training sets in the XNLI dev and test languages. Here we use the machine-translated training sets provided on [XNLI repo](https://github.com/facebookresearch/XNLI) to reproduce the translate-train results of [Conneau et al. (2018)](https://arxiv.org/pdf/1809.05053.pdf). We use this method as our primary baseline. 

### Translate-Test 

Translate-Test method involves translating the development and test sets to the training/source language (English). Machine-translated dev and test sets are also provided on [XNLI repo](https://github.com/facebookresearch/XNLI). We reproduce the translate-test results to use as our secondary baseline.

## How to perform cross-lingual NLI?

__How does it work?__ Let's say our goal is to perform NLI in German without translating the training set to German (or the test set to English). Each experiment consists of three following steps:
  
  __1) Training on English NLI Data:__ We first train an LSTM encoder and a linear classifier on MultiNLI data. Then we make a copy of the encoder, so that we have two identical encoders; one for the source language (En) and one for the target language (De).
  
  __2) Aligning Encoders:__ Then, by using parallel sentence pairs in English and German (from Europarl or OpenSubtitles 2018 corpora), we align the German encoder to the English encoder so that they produce close sentence representations in the embedding space. We use an adversarial objective in addition to the alignment loss proposed by [Conneau et al. (2018)](https://arxiv.org/pdf/1809.05053.pdf). Specifically, we try to fool a discriminator at the same time with alignment. We incorporate adversarial training to the process, since our goal is to produce close embeddings in the space so that a linear classifier trained for NLI is not able to tell the difference between English embeddings and German embeddings. This way, we can perform cross-lingual NLI without needing translation. 
  
  __3) Inference on XNLI (Non-English NLI Data):__ We build cross-lingual NLI models by training on MultiNLI and aligning encoders for the following languages:
      ```
      en: English,
      ar: Arabic,
      bg: Bulgarian,
      de: German,
      el: Greek,
      es: Spanish,
      fr: French,
      hi: Hindi,
      ru: Russian,
      th: Thai,
      tr: Turkish,
      vi: Vietnamese, &
      zh: Chinese.
      ```
           
### Downloading Aligned fastText Vectors

```curl -o wiki.en.align.vec https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.<lang>.align.vec```

Alternatively, you can use multilingual fastText [vectors](https://fasttext.cc/docs/en/crawl-vectors.html). 

### Packages You Will Need

  ```
  pytorch
  nltk: for standard English tokenizer
  jieba: for Chinese tokenizer
  spacy
  ```
