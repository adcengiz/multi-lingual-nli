# multi-lingual-nli
Investigating the effects of incorporating aligned cross-lingual word embeddings in multilingual NLI tasks 

To run the models, your directory should look like this:
```

data
├── aligned_embeddings
│   ├── wiki.ar.align.vec
│   ├── wiki.bg.align.vec
│   ├── wiki.de.align.vec
│   ├── wiki.el.align.vec
│   ├── wiki.en.align.vec
│   ├── wiki.es.align.vec
│   ├── wiki.fr.align.vec
│   ├── wiki.hi.align.vec
│   ├── wiki.ru.align.vec
│   ├── wiki.th.align.vec
│   ├── wiki.tr.align.vec
│   ├── wiki.vi.align.vec
│   └── wiki.zh.align.vec
├── GloVe
│   └── glove.840B.300d.txt
├── MultiNLI
│   ├── multinli_1.0_dev_matched.jsonl
│   ├── multinli_1.0_train.jsonl
│   └── README.txt
├── translation
│   ├── bg_en
│   │   ├── europarl-v7.bg-en.bg
│   │   └── europarl-v7.bg-en.en
│   ├── cs_en
│   │   ├── europarl-v7.cs-en.cs
│   │   └── europarl-v7.cs-en.en
│   ├── da_en
│   │   ├── europarl-v7.da-en.da
│   │   └── europarl-v7.da-en.en
│   ├── de_en
│   │   ├── europarl-v7.de-en.de
│   │   └── europarl-v7.de-en.en
│   ├── dev_test
│   │   ├── ac-dev.{lang}
│   │   ├── ac-devtest.{lang}
│   │   ├── ac-smalldev.{lang}
│   │   ├── ac-smalldev-ref.{lang}.sgm
│   │   ├── ac-smalldev-src.{lang}.sgm
│   │   ├── ac-test.{lang}
│   │   ├── ac-test-ref.{lang}.sgm
│   │   ├── ac-test-src.{lang}.sgm
│   │   ├── common.{lang}
│   │   ├── common.{lang}.info
│   │   ├── wrap.perl
│   │   └── wrap-smalldev.perl
│   ├── el_en
│   │   ├── europarl-v7.el-en.el
│   │   └── europarl-v7.el-en.en
│   ├── es_en
│   │   ├── europarl-v7.es-en.en
│   │   └── europarl-v7.es-en.es
│   ├── et_en
│   │   ├── europarl-v7.et-en.en
│   │   └── europarl-v7.et-en.et
│   ├── fi_en
│   │   ├── europarl-v7.fi-en.en
│   │   └── europarl-v7.fi-en.fi
│   ├── fr_en
│   │   ├── europarl-v7.fr-en.en
│   │   └── europarl-v7.fr-en.fr
│   ├── hu_en
│   │   ├── europarl-v7.hu-en.en
│   │   └── europarl-v7.hu-en.hu
│   ├── it_en
│   │   ├── europarl-v7.it-en.en
│   │   └── europarl-v7.it-en.it
│   └── lt_en
│       ├── europarl-v7.lt-en.en
│       └── europarl-v7.lt-en.lt
└── XNLI
    ├── README.md
    ├── xnli.dev.jsonl
    ├── xnli.dev.tsv
    ├── xnli.test.jsonl
    └── xnli.test.tsv
```
