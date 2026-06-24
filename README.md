# AAVE-LLM
Using Machina Mundi's ConvoAAVE dataset, we created a model used in 2026 Summer Camps, bridging the gap we found during last years camps

This repository contains the codebase for fine-tuning Meta's LLaMA 2 (7B) on a curated corpus of African American Vernacular English (AAVE) using a novel **Dialect-Adaptive Representation Alignment (DARA)** technique. The project aims to produce a language model that is not only fluent in AAVE but also linguistically informed, preserving dialect-specific features while maintaining semantic alignment with Mainstream American English (MAE).

## Data
The AAVE corpus is sourced from the [ConvoAAVE-POLLEN](https://github.com/karthiksrikumar/ConvoAAVE-POLLEN) dataset. The raw text file (`data/compiledFINAL.txt`) is automatically downloaded and preprocessed. The corpus contains natural conversational AAVE collected from social media and other informal contexts.

## Project Structure

```text
.
├── data/
│   ├── compiledFINAL.txt          # Raw AAVE text (downloaded automatically)
│   └── processed/                 # Train/val/test splits after preprocessing
├── reports/
│   └── linguistic_report.json     # Linguistic feature analysis
├── config.py                      # Central configuration (paths, hyperparameters)
├── data_preprocessing.py          # Download, clean, segment, and split data
├── utils.py                       # AAVE→MAE rule-based converter, contrastive pair generation
├── linguistic_analysis.py         # POS tagging, dependency parsing, feature extraction
├── model_training.py             # LoRA fine-tuning with DARA contrastive objective
├── evaluation.py                  # Perplexity and text generation
├── README.md
└── requirements.txt              # Python dependencies
```

---

## Methodology

The framework follows a four-stage pipeline designed to preserve dialectal linguistic structure while improving representation quality through contrastive learning.

### 1. Data Collection and Preprocessing

Raw AAVE text is automatically downloaded and stored in `compiledFINAL.txt`. The preprocessing pipeline:

* Cleans and normalizes text
* Segments documents into training examples
* Removes malformed entries
* Creates train, validation, and test splits

### 2. Linguistic Feature Analysis

To characterize the corpus, the system performs:

* Part-of-speech tagging using spaCy
* Dependency parsing using Stanza
* Extraction of AAVE-specific grammatical features
* Frequency analysis of syntactic constructions

Results are saved to `reports/linguistic_report.json`.

### 3. Dialect-Aware Contrastive Learning

For each AAVE sentence, a rule-based converter generates a corresponding Mainstream American English (MAE) variant. These paired examples form positive contrastive pairs.

The training objective combines:

* Standard causal language modeling loss
* Contrastive representation loss between dialectal variants

The total objective is:

[
L = L_{LM} + \lambda L_{contrastive}
]

where:

* (L_{LM}) is the language modeling loss
* (L_{contrastive}) is the contrastive alignment loss
* (\lambda) controls the balance between objectives

### 4. Parameter-Efficient Fine-Tuning

The framework fine-tunes Llama 2 using:

* LoRA (Low-Rank Adaptation)
* 8-bit quantization
* Hugging Face Transformers
* PEFT

This substantially reduces memory requirements while preserving model quality.

### 5. Evaluation

Performance is assessed through:

* Test set perplexity
* Qualitative generation analysis
* Comparison of dialect-sensitive outputs
* Representation alignment metrics
References
Hu, E., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
Gao, T., et al. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings.
Wolfram, W., & Schilling-Estes, N. (2006). American English: Dialects and Variation.
Jones, T. (2015). The AAVE Tense/Aspect System.
License

This project is intended for research purposes only.

Use of LLaMA-based models must comply with Meta's licensing terms and any applicable Hugging Face usage requirements.
