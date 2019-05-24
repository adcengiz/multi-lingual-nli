## Learning Sentence Representations for Cross-Lingual NLI with an Adversarial Objective

### What is NLI (Natural Language Inference)?

Given a premise-hypothesis pair, NLI is the task of understanding whether the premise _entails_ the hypothesis, whether it _contradicts_ the hypothesis or neither (the relationship is _neutral_).

__A more formal definition:__ Natural Language Inference, also known as Recognizing Textual Entailment, involves determining whether a sentence describing a situation, or premise, shares similar truth conditions, or entails another sentence called the hypothesis. Hypotheses with conflicting truth conditions are said to contradict the premise, and indeterminate relationships between the truth conditions of the two sentences are said to be neutral.

### How to perform cross-lingual NLI?

Cross-lingual NLI involves training a natural language inference model in a language and predicting entailment labels for data in another language. For example, in this project, we train an NLI model on MultiNLI data - which is available only in English - and evaluate it for use in other languages. 

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
      vi: Vietnamese,
      zh: Chinese.
      ```
 
