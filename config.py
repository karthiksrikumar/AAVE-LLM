# config.py
"""
Central configuration file for the AAVE-LLaMA project.
Contains paths, model hyperparameters, and training settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ProjectConfig:
    # Data
    raw_data_url: str = "https://raw.githubusercontent.com/karthiksrikumar/ConvoAAVE-POLLEN/main/data/compiledFINAL.txt"
    raw_data_path: str = "data/compiledFINAL.txt"
    processed_data_path: str = "data/processed/"
    parallel_data_path: str = "data/parallel/"
    
    # Model
    model_name: str = "meta-llama/Llama-2-7b-hf"  # Requires Hugging Face access for reproducing
    use_8bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Training
    output_dir: str = "./aave-llama-finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    logging_steps: int = 50
    save_steps: int = 500
    eval_steps: int = 500
    max_seq_length: int = 512
    
    # Contrastive loss weight (λ) for DARA
    contrastive_weight: float = 0.15
    temperature: float = 0.07
    
    # Linguistic analysis
    spacy_model: str = "en_core_web_sm"
    stanza_lang: str = "en"
    stanza_processors: str = "tokenize,pos,lemma,depparse"
    
    # Seed
    seed: int = 42

# Singleton instance
config = ProjectConfig()
