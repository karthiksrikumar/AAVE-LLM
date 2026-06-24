# linguistic_analysis.py
"""
Linguistic profiling of the AAVE dataset using part-of-speech tagging,
dependency parsing, and feature extraction. Generates a statistical
report on dialect-specific constructions.
Uses Stanza for robust NLP on non-standard text.
"""

import os
import json
from collections import Counter, defaultdict
from typing import List, Dict

import stanza
import numpy as np
from tqdm import tqdm

from config import config
from utils import extract_linguistic_features

stanza.download(config.stanza_lang, processors=config.stanza_processors, logging_level='WARN')
nlp = stanza.Pipeline(lang=config.stanza_lang, processors=config.stanza_processors, use_gpu=False)

class AAVELinguisticAnalyzer:
    """Performs syntactic and lexical analysis over the corpus
    and computes prevalence of AAVE features."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        with open(data_path, "r", encoding="utf-8") as f:
            self.sentences = [line.strip() for line in f if line.strip()]
        self.feature_counts = defaultdict(int)
        self.pos_counts = Counter()
        self.deprel_counts = Counter()
        self.total_sentences = len(self.sentences)
    
    def analyze_sentence(self, sentence: str):
        """Process one sentence to extract POS tags, dependency relations,
        and detect dialect features."""
        doc = nlp(sentence)
        for sent in doc.sentences:
            for word in sent.words:
                self.pos_counts[word.upos] += 1
                self.deprel_counts[word.deprel] += 1
        # Dialect features
        feats = extract_linguistic_features(sentence)
        for feat, present in feats.items():
            if present:
                self.feature_counts[feat] += 1
    
    def run(self) -> Dict:
        """Run full analysis and return results dictionary."""
        print(f"Analyzing {self.total_sentences} sentences...")
        for sent in tqdm(self.sentences, desc="Linguistic Analysis"):
            self.analyze_sentence(sent)
        
        # Normalize feature counts
        feature_ratios = {
            k: round(v / self.total_sentences, 4)
            for k, v in self.feature_counts.items()
        }
        
        results = {
            "total_sentences": self.total_sentences,
            "feature_prevalence": feature_ratios,
            "top_pos_tags": self.pos_counts.most_common(15),
            "top_dep_relations": self.deprel_counts.most_common(15),
        }
        return results
    
    def save_report(self, results: Dict, output_path: str = "reports/linguistic_report.json"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Linguistic report saved to {output_path}")

if __name__ == "__main__":
    # Analyze training set
    train_file = os.path.join(config.processed_data_path, "train.txt")
    analyzer = AAVELinguisticAnalyzer(train_file)
    report = analyzer.run()
    analyzer.save_report(report)
