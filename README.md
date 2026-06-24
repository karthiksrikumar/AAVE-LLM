# AAVE-LLM
Using Machina Mundi's ConvoAAVE dataset, we created a model used in 2026 Summer Camps, bridging the gap we found during last years camps

This repository contains the codebase for fine-tuning Meta's LLaMA 2 (7B) on a curated corpus of African American Vernacular English (AAVE) using a novel **Dialect-Adaptive Representation Alignment (DARA)** technique. The project aims to produce a language model that is not only fluent in AAVE but also linguistically informed, preserving dialect-specific features while maintaining semantic alignment with Mainstream American English (MAE).

## Data
The AAVE corpus is sourced from the [ConvoAAVE-POLLEN](https://github.com/karthiksrikumar/ConvoAAVE-POLLEN) dataset. The raw text file (`data/compiledFINAL.txt`) is automatically downloaded and preprocessed. The corpus contains natural conversational AAVE collected from social media and other informal contexts.

## Project Structure
├── data/
│ ├── compiledFINAL.txt # Raw AAVE text (downloaded automatically)
│ └── processed/ # Train/val/test splits after preprocessing
├── reports/
│ └── linguistic_report.json # Linguistic feature analysis
├── config.py # Central configuration (paths, hyperparameters)
├── data_preprocessing.py # Download, clean, segment, and split data
├── utils.py # AAVE→MAE rule-based converter, contrastive pair generation
├── linguistic_analysis.py # POS tagging, dependency parsing, feature extraction
├── model_training.py # LoRA fine-tuning with DARA contrastive objective
├── evaluation.py # Perplexity and text generation
├── README.md
└── requirements.txt # Python dependencies



## Methodology

# AAVE Contrastive Fine-Tuning Framework

The contrastive weight `λ` is a hyperparameter that controls the balance between language modeling loss and contrastive learning objectives.

## Usage

### Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m stanza.download en
```

---

## Step 1: Preprocess Data

```bash
python data_preprocessing.py
```

Downloads the dataset, cleans the text, segments documents, and creates train/validation/test splits.

---

## Step 2: Linguistic Analysis

```bash
python linguistic_analysis.py
```

Generates a JSON report containing:

* Prevalence of AAVE linguistic features
* Part-of-speech frequency statistics
* Dependency relation distributions
* Corpus-level linguistic summaries

---

## Step 3: Fine-Tune the Model

```bash
python model_training.py
```

### Requirements

Access to `meta-llama/Llama-2-7b-hf` from Hugging Face is required.

The training pipeline uses:

* 8-bit quantization
* LoRA (Low-Rank Adaptation)
* Contrastive representation learning

These techniques substantially reduce GPU memory requirements while maintaining performance.

---

## Step 4: Evaluate

```bash
python evaluation.py
```

Computes:

* Test perplexity
* Generation quality metrics
* Sample model outputs

---

## References

1. Hu, E., et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
2. Gao, T., et al. (2021). *SimCSE: Simple Contrastive Learning of Sentence Embeddings.*
3. Wolfram, W., & Schilling-Estes, N. (2006). *American English: Dialects and Variation.*
4. Jones, T. (2015). *The AAVE Tense/Aspect System.*

---

## License

This project is intended for research purposes only.

Use of LLaMA-based models must comply with Meta's licensing terms and any applicable Hugging Face usage requirements.
