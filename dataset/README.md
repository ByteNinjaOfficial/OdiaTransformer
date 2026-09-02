---
annotations_creators:
- no-annotation
language_creators:
- found
language:
- en
- as
- bn
- gu
- hi
- kn
- ml
- mr
- or
- pa
- ta
- te
license:
- cc-by-nc-4.0
multilinguality:
- translation
size_categories:
- unknown
source_datasets:
- original
task_categories:
- text-generation
- translation
task_ids: []
pretty_name: Samanantar
tags:
- conditional-text-generation
dataset_info:
- config_name: as
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 35548280
    num_examples: 141227
  download_size: 19404220
  dataset_size: 35548280
- config_name: bn
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 2204943013
    num_examples: 8604580
  download_size: 1197993546
  dataset_size: 2204943013
- config_name: gu
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 678219165
    num_examples: 3067790
  download_size: 370118693
  dataset_size: 678219165
- config_name: hi
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 3598130651
    num_examples: 10125706
  download_size: 1916386168
  dataset_size: 3598130651
- config_name: kn
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 908044345
    num_examples: 4093524
  download_size: 483451767
  dataset_size: 908044345
- config_name: ml
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 1545010918
    num_examples: 5924426
  download_size: 784891308
  dataset_size: 1545010918
- config_name: mr
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 929465791
    num_examples: 3627480
  download_size: 487825355
  dataset_size: 929465791
- config_name: or
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 265698932
    num_examples: 998228
  download_size: 136203687
  dataset_size: 265698932
- config_name: pa
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 876144647
    num_examples: 2980383
  download_size: 470604570
  dataset_size: 876144647
- config_name: ta
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 1518393368
    num_examples: 5264867
  download_size: 753886808
  dataset_size: 1518393368
- config_name: te
  features:
  - name: idx
    dtype: int64
  - name: src
    dtype: string
  - name: tgt
    dtype: string
  splits:
  - name: train
    num_bytes: 1121643183
    num_examples: 4946035
  download_size: 599351284
  dataset_size: 1121643183
configs:
- config_name: as
  data_files:
  - split: train
    path: as/train-*
- config_name: bn
  data_files:
  - split: train
    path: bn/train-*
- config_name: gu
  data_files:
  - split: train
    path: gu/train-*
- config_name: hi
  data_files:
  - split: train
    path: hi/train-*
- config_name: kn
  data_files:
  - split: train
    path: kn/train-*
- config_name: ml
  data_files:
  - split: train
    path: ml/train-*
- config_name: mr
  data_files:
  - split: train
    path: mr/train-*
- config_name: or
  data_files:
  - split: train
    path: or/train-*
- config_name: pa
  data_files:
  - split: train
    path: pa/train-*
- config_name: ta
  data_files:
  - split: train
    path: ta/train-*
- config_name: te
  data_files:
  - split: train
    path: te/train-*
---

# Dataset Card for Samanantar

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Dataset Structure](#dataset-structure)
  - [Data Instances](#data-instances)
  - [Data Fields](#data-fields)
  - [Data Splits](#data-splits)
- [Dataset Creation](#dataset-creation)
  - [Curation Rationale](#curation-rationale)
  - [Source Data](#source-data)
  - [Annotations](#annotations)
  - [Personal and Sensitive Information](#personal-and-sensitive-information)
- [Considerations for Using the Data](#considerations-for-using-the-data)
  - [Social Impact of Dataset](#social-impact-of-dataset)
  - [Discussion of Biases](#discussion-of-biases)
  - [Other Known Limitations](#other-known-limitations)
- [Additional Information](#additional-information)
  - [Dataset Curators](#dataset-curators)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)
  - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://ai4bharat.iitm.ac.in/areas/nmt
- **Repository:**
- **Paper:** [Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages](https://arxiv.org/abs/2104.05596)
- **Leaderboard:**
- **Point of Contact:**

### Dataset Summary

Samanantar is the largest publicly available parallel corpora collection for Indic language: Assamese, Bengali,
Gujarati, Hindi, Kannada, Malayalam, Marathi, Oriya, Punjabi, Tamil, Telugu.

The corpus has 49.6M sentence pairs between English to Indian Languages.

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

Samanantar contains parallel sentences between English (`en`) and 11 Indic language:
- Assamese (`as`),
- Bengali (`bn`),
- Gujarati (`gu`),
- Hindi (`hi`),
- Kannada (`kn`),
- Malayalam (`ml`),
- Marathi (`mr`),
- Odia (`or`),
- Punjabi (`pa`),
- Tamil (`ta`) and
- Telugu (`te`).

## Dataset Structure

### Data Instances

```
{
  'idx': 0,
  'src': 'Prime Minister Narendra Modi met Her Majesty Queen Maxima of the Kingdom of the Netherlands today.',
  'tgt': 'নতুন দিল্লিতে সোমবার প্রধানমন্ত্রী শ্রী নরেন্দ্র মোদীর সঙ্গে নেদারন্যান্ডসের মহারানী ম্যাক্সিমা সাক্ষাৎ করেন।',
  'data_source': 'pmi'
}
```

### Data Fields

- `idx` (int): ID.
- `src` (string): Sentence in source language (English).
- `tgt` (string): Sentence in destination language (one of the 11 Indic languages).
- `data_source` (string): Source of the data.
  For created data sources, depending on the destination language, it might be one of:
  - anuvaad_catchnews
  - anuvaad_DD_National
  - anuvaad_DD_sports
  - anuvaad_drivespark
  - anuvaad_dw
  - anuvaad_financialexpress
  - anuvaad-general_corpus
  - anuvaad_goodreturns
  - anuvaad_indianexpress
  - anuvaad_mykhel
  - anuvaad_nativeplanet
  - anuvaad_newsonair
  - anuvaad_nouns_dictionary
  - anuvaad_ocr
  - anuvaad_oneindia
  - anuvaad_pib
  - anuvaad_pib_archives
  - anuvaad_prothomalo
  - anuvaad_timesofindia
  - asianetnews
  - betterindia
  - bridge
  - business_standard
  - catchnews
  - coursera
  - dd_national
  - dd_sports
  - dwnews
  - drivespark
  - fin_express
  - goodreturns
  - gu_govt
  - jagran-business
  - jagran-education
  - jagran-sports
  - ie_business
  - ie_education
  - ie_entertainment
  - ie_general
  - ie_lifestyle
  - ie_news
  - ie_sports
  - ie_tech
  - indiccorp
  - jagran-entertainment
  - jagran-lifestyle
  - jagran-news
  - jagran-tech
  - khan_academy
  - Kurzgesagt
  - marketfeed
  - mykhel
  - nativeplanet
  - nptel
  - ocr
  - oneindia
  - pa_govt
  - pmi
  - pranabmukherjee
  - sakshi
  - sentinel
  - thewire
  - toi
  - tribune
  - vsauce
  - wikipedia
  - zeebiz

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

### Citation Information

```
@article{10.1162/tacl_a_00452,
    author = {Ramesh, Gowtham and Doddapaneni, Sumanth and Bheemaraj, Aravinth and Jobanputra, Mayank and AK, Raghavan and Sharma, Ajitesh and Sahoo, Sujit and Diddee, Harshita and J, Mahalakshmi and Kakwani, Divyanshu and Kumar, Navneet and Pradeep, Aswin and Nagaraj, Srihari and Deepak, Kumar and Raghavan, Vivek and Kunchukuttan, Anoop and Kumar, Pratyush and Khapra, Mitesh Shantadevi},
    title = "{Samanantar: The Largest Publicly Available Parallel Corpora Collection for 11 Indic Languages}",
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {10},
    pages = {145-162},
    year = {2022},
    month = {02},
    abstract = "{We present Samanantar, the largest publicly available parallel corpora collection for Indic languages. The collection contains a total of 49.7 million sentence pairs between English and 11 Indic languages (from two language families). Specifically, we compile 12.4 million sentence pairs from existing, publicly available parallel corpora, and additionally mine 37.4 million sentence pairs from the Web, resulting in a 4× increase. We mine the parallel sentences from the Web by combining many corpora, tools, and methods: (a) Web-crawled monolingual corpora, (b) document OCR for extracting sentences from scanned documents, (c) multilingual representation models for aligning sentences, and (d) approximate nearest neighbor search for searching in a large collection of sentences. Human evaluation of samples from the newly mined corpora validate the high quality of the parallel sentences across 11 languages. Further, we extract 83.4 million sentence
                    pairs between all 55 Indic language pairs from the English-centric parallel corpus using English as the pivot language. We trained multilingual NMT models spanning all these languages on Samanantar which outperform existing models and baselines on publicly available benchmarks, such as FLORES, establishing the utility of Samanantar. Our data and models are available publicly at Samanantar and we hope they will help advance research in NMT and multilingual NLP for Indic languages.}",
    issn = {2307-387X},
    doi = {10.1162/tacl_a_00452},
    url = {https://doi.org/10.1162/tacl\_a\_00452},
    eprint = {https://direct.mit.edu/tacl/article-pdf/doi/10.1162/tacl\_a\_00452/1987010/tacl\_a\_00452.pdf},
}
```

### Contributions

Thanks to [@albertvillanova](https://github.com/albertvillanova) for adding this dataset.
