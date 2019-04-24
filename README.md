# multi-lingual-nli
Investigating the effects of incorporating aligned cross-lingual word embeddings in multilingual NLI tasks 

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
To run the models, your directory should look like this:
```
├── align_encoders
│   └── your_notebooks.ipynb
├── data
│   ├── aligned_embeddings
│   │   ├── wiki.ar.align.vec
│   │   ├── wiki.ar.en.align.vec
│   │   ├── wiki.bg.en.align.vec
│   │   ├── wiki.de.align.vec
│   │   ├── wiki.de.en.align.vec
│   │   ├── wiki.en.align.vec
│   │   ├── wiki.es.align.vec
│   │   ├── wiki.es.en.align.vec
│   │   ├── wiki.fr.align.vec
│   │   ├── wiki.fr.en.align.vec
│   │   ├── wiki.hi.align.vec
│   │   ├── wiki.ru.align.vec
│   │   ├── wiki.th.align.vec
│   │   ├── wiki.th.en.align.vec
│   │   ├── wiki.tr.align.vec
│   │   ├── wiki.tr.en.align.vec
│   │   ├── wiki.vi.align.vec
│   │   ├── wiki.vi.en.align.vec
│   │   ├── wiki.zh.align.vec
│   │   └── wiki.zh.en.align.vec
│   ├── europarl
│   │   ├── cs_en
│   │   │   ├── europarl-v7.cs-en.cs
│   │   │   └── europarl-v7.cs-en.en
│   │   ├── da_en
│   │   │   ├── europarl-v7.da-en.da
│   │   │   └── europarl-v7.da-en.en
│   │   ├── de_en
│   │   │   ├── europarl-v7.de-en.de
│   │   │   └── europarl-v7.de-en.en
│   │   ├── dev_test
│   │   ├── el_en
│   │   │   ├── europarl-v7.el-en.el
│   │   │   └── europarl-v7.el-en.en
│   │   ├── es_en
│   │   │   ├── europarl-v7.es-en.en
│   │   │   └── europarl-v7.es-en.es
│   │   ├── et_en
│   │   │   ├── europarl-v7.et-en.en
│   │   │   └── europarl-v7.et-en.et
│   │   ├── fi_en
│   │   │   ├── europarl-v7.fi-en.en
│   │   │   └── europarl-v7.fi-en.fi
│   │   ├── fr_en
│   │   │   ├── europarl-v7.fr-en.en
│   │   │   └── europarl-v7.fr-en.fr
│   │   ├── hu_en
│   │   │   ├── europarl-v7.hu-en.en
│   │   │   └── europarl-v7.hu-en.hu
│   │   ├── it_en
│   │   │   ├── europarl-v7.it-en.en
│   │   │   └── europarl-v7.it-en.it
│   │   └── lt_en
│   │       ├── europarl-v7.lt-en.en
│   │       └── europarl-v7.lt-en.lt
│   ├── multi_lingual_embeddings
│   │   ├── cc.ar.300.vec
│   │   ├── cc.de.300.vec
│   │   ├── cc.en.300.vec
│   │   ├── cc.es.300.vec
│   │   ├── cc.fr.300.vec
│   │   ├── cc.hi.300.vec
│   │   ├── cc.ru.300.vec
│   │   ├── cc.th.300.vec
│   │   ├── cc.tr.300.vec
│   │   ├── cc.vi.300.vec
│   │   └── cc.zh.300.vec
│   ├── MultiNLI
│   ├── opus
│   │   ├── ar_en
│   │   │   ├── OpenSubtitles.ar-en.ar
│   │   │   └── OpenSubtitles.ar-en.en
│   │   ├── es_en
│   │   │   ├── OpenSubtitles.en-es.en
│   │   │   └── OpenSubtitles.en-es.es
│   │   ├── ru_en
│   │   │   ├── OpenSubtitles.en-ru.en
│   │   │   └── OpenSubtitles.en-ru.ru
│   │   ├── th_en
│   │   │   ├── OpenSubtitles.en-th.en
│   │   │   └── OpenSubtitles.en-th.th
│   │   ├── tr_en
│   │   │   ├── OpenSubtitles.en-tr.en
│   │   │   └── OpenSubtitles.en-tr.tr
│   │   ├── vi_en
│   │   │   ├── OpenSubtitles.en-vi.en
│   │   │   ├── OpenSubtitles.en-vi.vi
│   │   └── zh_en
│   │       ├── OpenSubtitles.en-zh.en
│   │       └── OpenSubtitles.en-zh.zh
│   ├── SNLI
│   ├── un_parallel_corpora
│   │   └── ru_en
│   │       └── Download?file=UNv1.0-TEI.ru.tar.gz.00
│   └── XNLI
│       ├── README.md
│       ├── xnli.dev.jsonl
│       ├── xnli.dev.tsv
│       ├── xnli.test.jsonl
│       └── xnli.test.tsv
├── HBMP
│   ├── classifier.py
│   ├── corpora.py
│   ├── data
│   │   ├── breaking_nli
│   │   │   └── data
│   │   │       ├── breaking_dev.jsonl
│   │   │       ├── breaking_test.jsonl
│   │   │       ├── breaking_train.jsonl
│   │   │       └── README.txt
│   │   ├── multinli
│   │   │   └── multinli_1.0
│   │   │       ├── multinli_1.0_dev_matched.jsonl
│   │   │       ├── multinli_1.0_dev_matched.txt
│   │   │       ├── multinli_1.0_dev_mismatched.jsonl
│   │   │       ├── multinli_1.0_dev_mismatched.txt
│   │   │       ├── multinli_1.0_train.jsonl
│   │   │       ├── multinli_1.0_train.txt
│   │   │       ├── paper.pdf
│   │   │       └── README.txt
│   │   ├── scitail
│   │   │   └── SciTailV1
│   │   │       ├── all_annotations.tsv
│   │   │       ├── dgem_format
│   │   │       │   ├── README.txt
│   │   │       │   ├── scitail_1.0_structure_dev.tsv
│   │   │       │   ├── scitail_1.0_structure_test.tsv
│   │   │       │   └── scitail_1.0_structure_train.tsv
│   │   │       ├── README.txt
│   │   │       ├── snli_format
│   │   │       │   ├── README.txt
│   │   │       │   ├── scitail_1.0_dev.txt
│   │   │       │   ├── scitail_1.0_test.txt
│   │   │       │   └── scitail_1.0_train.txt
│   │   │       └── tsv_format
│   │   │           ├── scitail_1.0_dev.tsv
│   │   │           ├── scitail_1.0_test.tsv
│   │   │           └── scitail_1.0_train.tsv
│   │   ├── snli
│   │   │   └── snli_1.0
│   │   │       ├── all_nli.jsonl
│   │   │       ├── README.txt
│   │   │       ├── snli_1.0_dev.jsonl
│   │   │       ├── snli_1.0_dev.txt
│   │   │       ├── snli_1.0_test.jsonl
│   │   │       ├── snli_1.0_test.txt
│   │   │       ├── snli_1.0_train.jsonl
│   │   │       └── snli_1.0_train.txt
│   │   └── xnli
│   │       ├── README.md
│   │       ├── xnli.dev.jsonl
│   │       ├── xnli.dev.tsv
│   │       ├── xnli.test.jsonl
│   │       └── xnli.test.tsv
│   ├── download_data.sh
│   ├── embeddings.py
│   ├── evaluate_senteval.py
│   ├── glove.840B.300d
│   ├── HBMP_baseline.ipynb
│   ├── __init__.py
│   └── vector_cache
│       └── glove.840B.300d.txt
├── models
├── notebooks
│   └── reading_data
│       └── loading_data.ipynb
```
