# data_preprocessing.py
"""
Data preprocessing pipeline for the compiled AAVE corpus.
Downloads raw data, cleans, segments into sentences, and splits
into train/validation/test sets. Also creates a linguistically
enriched version for downstream tasks.
"""

import os
import re
import random
import logging
import requests
from pathlib import Path
from typing import List, Tuple

import spacy
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AAVEPreprocessor:
    """Handles cleaning, sentence segmentation, and dataset splitting
    for the AAVE corpus from ConvoAAVE-POLLEN."""
    
    def __init__(self, raw_path: str = config.raw_data_path):
        self.raw_path = raw_path
        self.nlp = spacy.load(config.spacy_model, disable=["ner", "textcat"])
        # Add custom sentence boundary rules for social media text
        self.nlp.add_pipe("sentencizer", before="parser")
        
    def download_data(self) -> None:
        """Download the compiled AAVE dataset if not present locally."""
        if not os.path.exists(self.raw_path):
            logger.info(f"Downloading dataset from {config.raw_data_url}")
            os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)
            response = requests.get(config.raw_data_url)
            response.raise_for_status()
            with open(self.raw_path, "w", encoding="utf-8") as f:
                f.write(response.text)
        else:
            logger.info(f"Dataset found at {self.raw_path}")
    
    def clean_text(self, text: str) -> str:
        """
        Basic cleaning: remove extra whitespace, fix common encoding issues,
        and normalize some non-standard punctuation.
        """
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)  # collapse whitespace
        text = re.sub(r'“|”', '"', text)  # normalize quotes
        text = re.sub(r'‘|’', "'", text)
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """Segment raw text into sentences using spaCy's sentencizer,
        tuned for informal language."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences
    
    def filter_sentences(self, sentences: List[str], min_len: int = 5, max_len: int = 256) -> List[str]:
        """Remove sentences that are too short or exceed the model's token limit."""
        return [s for s in sentences if min_len <= len(s.split()) <= max_len]
    
    def process_and_save(self) -> Tuple[str, str, str]:
        """
        Full pipeline: download -> clean -> segment -> split -> save.
        Returns paths to train, val, test files.
        """
        self.download_data()
        with open(self.raw_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        
        # Clean and segment
        cleaned = self.clean_text(raw_text)
        sentences = self.segment_sentences(cleaned)
        sentences = self.filter_sentences(sentences)
        logger.info(f"Extracted {len(sentences)} valid sentences.")
        
        # Stratified split to maintain distribution of sentence lengths (heuristic)
        random.seed(config.seed)
        random.shuffle(sentences)
        train, temp = train_test_split(sentences, test_size=0.2, random_state=config.seed)
        val, test = train_test_split(temp, test_size=0.5, random_state=config.seed)
        
        # Save
        os.makedirs(config.processed_data_path, exist_ok=True)
        for subset, name in zip([train, val, test], ["train.txt", "val.txt", "test.txt"]):
            path = os.path.join(config.processed_data_path, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(subset))
            logger.info(f"Saved {len(subset)} sentences to {path}")
        
        return (
            os.path.join(config.processed_data_path, "train.txt"),
            os.path.join(config.processed_data_path, "val.txt"),
            os.path.join(config.processed_data_path, "test.txt")
        )

if __name__ == "__main__":
    preproc = AAVEPreprocessor()
    preproc.process_and_save()
