# multi-lingual-nli
Investigating the effects of incorporating aligned cross-lingual word embeddings in multilingual NLI tasks 

To run the models, your directory should look like this:
```

multi-lingual-nli
|-- data
|    |-- aligned_embeddings
|    |    |-- wiki.{lang}.align.vec
|    |
|    |-- GloVe
|    |    |-- glove.840B.300d.txt
|    |
|    |-- MultiNLI
|    |    |-- multinli_1.0_dev_matched.jsonl
|    |    |-- multinli_1.0_train.jsonl
|    |
|    |-- SNLI
|    |    |-- snli_1.0_dev.jsonl
|    |    |-- snli_1.0_test.jsonl
|    |    |-- snli_1.0_train.jsonl
|    |
|    |-- translation
|    |    |-- {lang}_en
|    |    |    |-- europarl-v7.{lang}-en.{lang}
|    |    |    |-- europarl-v7.{lang}-en.en
|    |    |
|    |    |-- dev_test
|    |
|    |-- XNLI
|         |-- xnli.dev.jsonl
|         |-- xnli.test.jsonl
|
|-- models 
|    |-- model.py
|
|-- notebooks
|    |-- notebook.ipynb
|    
|-- utils
     |-- preprocessing_utils.py
     |-- experiment.py
```
