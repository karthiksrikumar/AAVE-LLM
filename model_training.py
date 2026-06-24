# model_training.py
"""
Fine-tune LLaMA 2 on the AAVE corpus using LoRA and a novel Dialect-Adaptive
Representation Alignment (DARA) technique. DARA combines standard causal
language modeling with a contrastive objective that pulls the representations
of AAVE sentences and their MAE counterparts closer, encouraging the model
to learn dialect-invariant semantic representations.

References:
- Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
- Gao et al. (2021) "SimCSE: Simple Contrastive Learning of Sentence Embeddings"
- Wolfram & Schilling-Estes (2006) for dialect features
"""

import os
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset as HFDataset
import numpy as np
from typing import Dict, List, Tuple

from config import config
from utils import create_contrastive_pairs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
set_seed(config.seed)

class AAVEContrastiveDataset(Dataset):
    """
    PyTorch Dataset that yields batches containing both AAVE original
    and its MAE version for contrastive loss computation.
    """
    def __init__(self, aave_texts: List[str], mae_texts: List[str], tokenizer, max_len: int):
        self.aave_texts = aave_texts
        self.mae_texts = mae_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.aave_texts)
    
    def __getitem__(self, idx):
        aave = self.aave_texts[idx]
        mae = self.mae_texts[idx]
        
        # Tokenize both versions
        aave_enc = self.tokenizer(
            aave,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        mae_enc = self.tokenizer(
            mae,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "aave_input_ids": aave_enc["input_ids"].squeeze(),
            "aave_attention_mask": aave_enc["attention_mask"].squeeze(),
            "mae_input_ids": mae_enc["input_ids"].squeeze(),
            "mae_attention_mask": mae_enc["attention_mask"].squeeze(),
        }

class DARATrainer(Trainer):
    """
    Custom Trainer that appends a contrastive loss to the standard LM loss.
    The contrastive loss is computed on the hidden states of the last token
    (or mean pooling) of paired AAVE-MAE sentences.
    """
    def __init__(self, *args, contrastive_weight=0.1, temperature=0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Standard LM loss from labels inside inputs (handled by Trainer if labels are present)
        # We'll modify inputs to also include contrastive pairs
        # Extract contrastive tensors
        aave_ids = inputs.pop("aave_input_ids")
        aave_mask = inputs.pop("aave_attention_mask")
        mae_ids = inputs.pop("mae_input_ids")
        mae_mask = inputs.pop("mae_attention_mask")
        
        # Standard LM forward on AAVE (labels are aave_ids shifted inside)
        outputs = model(input_ids=aave_ids, attention_mask=aave_mask, labels=aave_ids)
        lm_loss = outputs.loss
        
        # Compute sentence embeddings for contrastive loss
        # Use mean pooling of last hidden states (excluding padding)
        with torch.no_grad():
            aave_emb = self.get_sentence_embedding(model, aave_ids, aave_mask)
        mae_emb = self.get_sentence_embedding(model, mae_ids, mae_mask)
        
        # Contrastive loss (SimCSE style)
        aave_emb = F.normalize(aave_emb, dim=1)
        mae_emb = F.normalize(mae_emb, dim=1)
        
        # Cosine similarity matrix: (batch_size x batch_size)
        sim_matrix = torch.matmul(aave_emb, mae_emb.T) / self.temperature
        labels = torch.arange(sim_matrix.size(0)).to(sim_matrix.device)
        contrastive_loss = F.cross_entropy(sim_matrix, labels)
        
        total_loss = lm_loss + self.contrastive_weight * contrastive_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    @staticmethod
    def get_sentence_embedding(model, input_ids, attention_mask):
        """Extract sentence embedding from model's last hidden state."""
        outputs = model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        # Mean pooling over non-padded tokens
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_emb = torch.sum(last_hidden * mask_expanded, dim=1)
        mean_emb = sum_emb / mask_expanded.sum(dim=1)
        return mean_emb

def load_and_prepare_data():
    """Load preprocessed AAVE texts and create contrastive pairs."""
    train_file = os.path.join(config.processed_data_path, "train.txt")
    with open(train_file, "r", encoding="utf-8") as f:
        train_sentences = [line.strip() for line in f if line.strip()]
    
    pairs = create_contrastive_pairs(train_sentences)
    aave_list, mae_list = zip(*pairs)
    return list(aave_list), list(mae_list)

def main():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare data
    aave_train, mae_train = load_and_prepare_data()
    train_dataset = AAVEContrastiveDataset(aave_train, mae_train, tokenizer, config.max_seq_length)
    
    # Quantization config for 8-bit training
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=config.use_8bit,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=True,
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "v_proj"]  # Common choice for LLaMA
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        evaluation_strategy="no",  # We skip eval during training for brevity
        fp16=True,
        report_to="none",
        seed=config.seed,
    )
    
    # Instantiate custom trainer
    trainer = DARATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=None,  # We'll handle collation manually inside dataset
        contrastive_weight=config.contrastive_weight,
        temperature=config.temperature,
    )
    
    # For simplicity, use a custom data collator that returns the dict as is.
    # Trainer will pass each batch as a dict (our Dataset returns dict).
    def collate_fn(batch):
        # Batch is a list of dictionaries; stack tensors
        keys = batch[0].keys()
        stacked = {}
        for key in keys:
            stacked[key] = torch.stack([item[key] for item in batch])
        return stacked
    
    trainer.data_collator = collate_fn
    
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
