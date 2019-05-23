## Learning Sentence Representations for Cross-Lingual NLI with an Adversarial Objective

### What is NLI (Natural Language Inference)?

Given a premise-hypothesis pair, NLI is the task of understanding whether the premise _entails_ the hypothesis, whether it _contradicts_ the hypothesis or neither (the relationship is _neutral_).

__A more formal definition:__ Natural Language Inference, also known as Recognizing Textual Entailment, involves determining whether a sentence describing a situation, or premise, shares similar truth conditions, or entails another sentence called the hypothesis. Hypotheses with conflicting truth conditions are said to contradict the premise, and indeterminate relationships between the truth conditions of the two sentences are said to be neutral.

### How to perform cross-lingual NLI?

Cross-lingual NLI involves training a natural language inference model in a language and predicting entailment labels for data in another language. For example, in this project, we train an NLI model on MultiNLI data - which is available only in English - and evaluate it for use in other languages. 

__How does it work?__ Let's say our goal is to perform NLI in German. We first train an LSTM encoder and a linear classifier on MultiNLI data. Then we make a copy of the encoder, so that we have two identical encoders; one for the source language (En) and one for the target language (De). Then, by using parallel sentence pairs in English and German (from Europarl or OpenSubtitles 2018 corpora), we align the German encoder to the English encoder so that they produce close sentence representations in the embedding space. 

We use the following alignment objective:

\begin{equation}
\mathcal{L}(x, y) = dist(x, y) \\ & - \lambda \left[dist(x_c, y) + dist(x, y_c)\right] + \lambda_{adv} \mathcal{L}_{adv}(\theta_{enc}, \mathcal{W} | \theta_D) 
\end{equation}

### Why adversarial training? 

The languages are referred by the following codes throughout the project:
```
en: English
ar: Arabic
bg: Bulgarian
de: German
el: Greek
es: Spanish
fr: French
hi: Hindi
ru: Russian
th: Thai
tr: Turkish
vi: Vietnamese
zh: Chinese
```
