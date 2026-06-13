# Intelligent Reporting: A Bilingual Text-to-SQL System for Domain-Specific Reporting

This repository contains the implementation of a bilingual (Arabic and English) Text-to-SQL system developed for [SESAME](https://www.sesame.org.jo/) and as a bachelors graduation project. 
[View the project report/reasearch paper on overleaf](https://www.overleaf.com/read/nwrszttsbxkm#179ac8).

## Overview
The system is designed to translate the user's natural language (NL) reporting requests into executable Oracle g11 SQL queries. It was developed and evaluated on SESAME User Portal (SUP) database using a dedicated bilingual dataset.

The systme consists of 2 stages:
1. Hybrid-seacrh schema retrieval to identifies a subset of schema tables relevant to the user's NL query combining sperse retrieval (using BM25) and dense retrieval (using FAISS) via Reciprocal Rank Fusion (RRF).
2. SQL generation using a pre-trained Text-to-SQL language model fine-tuned on our domain-specific dataset.

## Dataset

The dataset was costructed specifically for this project and is split into 3 sets: training, development (dev) and testing. It containes NL (in both Arabic and English) queries and their corresponding Oracle 11g SQL in addtion to a list of tables used in each SQL entry and their schema in DDL format.

## Model(s)

The LoRA adapter weights are not included in this repository due to file size limitations.

Important Note:

All the models used in this project were downloaded and run locally on SESAME computational resources.
To be able to run the models properly please download the following:
*for the code to run correctly please save them in a directories named `./models` and `./retriever_tests/models` respectively.
- [Llama-3-SQLCoder-8B](https://huggingface.co/defog/llama-3-sqlcoder-8b): Used for inference and fine-tuning.
- [Multilingual-E5-base](https://huggingface.co/intfloat/multilingual-e5-base): Used with the retriever as an embedding model for dense retrieval (using FAISS).


## Installation and Usage
Clone the repository and install the Python libraries listed in the `requirements.txt` file.

**To run the full end-to-end system:**
```bash
cd end-to-end
pyhton end2end.py --ver  --input 
```

**To run inference on the gold schema only:**
```bash
cd gold_inference 
python gold_schema_inference --ver --input
```
Arguments:
- `--ver`: version of the generation model, base or one of the fine-tuned models named vX (X = 1 - 6).
- `--input`: input file name (dev or test).

**To run the evaluation script:**
```bash
cd eval
python eval.py --ver --input --noise
```
Arguments:
- `--ver`: version of the generation model used during inference, base or one of the fine-tuned models named vX (X = 1 - 6).
- `--input`: input file name (dev or test).
- `--noise`: the inference setting. **N** = run using gold schema only, **Y** = run using end-to-end system (retriever schema subset).

>Plese note that the script used to connect to the DB is not uploaded due to privacy limitations. In addition to that, the DB is completely local and cannot be accessed without permission from SESAME, therefore, the executiona accuracy meaasure used in the evaluation script will not run properly.

## Retriever (Final version & Ablation Study versions)

The final version of the hybrid-search retriever and all the ablated versions are found in  `./retriever_tests`

| Version  | Description              |
| --------- | ------------------------ |
| `final_version` | The final version used in the end-to-end system |
| `no_descriptions`   | Omitting descriptions (both stemmed and unstemmed) from the passages (FAISS index and  BM25 corpus) |
| `unstemmed_descriptions`   | Omitting stemmed descriptions only from the passages |
| `no_empty_tables`   | Removing empty tables (tables with no data) prior to creating the  passages instead of filtering them post-retrieval |
| `no_table_name_match`   | Skipping the post-retrieval table-name exact match step |

To run any of them:

```bash
cd retriever_tests
python <filename>.py
```
---

CS492-Graduation Project (2)  
Group# 41:  
Alae Harfouche   159592









