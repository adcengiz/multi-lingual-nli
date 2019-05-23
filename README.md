## Learning Sentence Representations for Cross-Lingual NLI with an Adversarial Objective

### What is NLI (Natural Language Inference)?

Given a premise-hypothesis pair, NLI is the task of understanding whether the premise _entails_ the hypothesis, whether it _contradicts_ the hypothesis or neither (the relationship is _neutral_).

__A more formal definition:__ Natural Language Inference, also known as Recognizing Textual Entailment, involves determining whether a sentence describing a situation, or premise, shares similar truth conditions, or entails another sentence called the hypothesis. Hypotheses with conflicting truth conditions are said to contradict the premise, and indeterminate relationships between the truth conditions of the two sentences are said to be neutral.

### How to perform cross-lingual NLI?

Cross-lingual NLI involves training a natural language inference model in a language and predicting entailment labels for another language. 

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
