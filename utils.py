# utils.py
"""
Utility functions for dialect-aware text transformation and
contrastive pair generation. Implements a simple rule-based
AAVE-to-MAE converter to create parallel data for the DARA technique.
References:
- Wolfram & Schilling-Estes (2006) "American English"
- Jones (2015) "AAVE tense/aspect system"
"""

import re
from typing import List, Tuple, Dict

# Mapping of common AAVE lexical items to Mainstream American English equivalents
LEXICAL_MAP: Dict[str, str] = {
    "finna": "going to",
    "tryna": "trying to",
    "gonna": "going to",
    "wanna": "want to",
    "y'all": "you all",
    "ain't": "is not",
    "innit": "isn't it",
    "lemme": "let me",
    "gimme": "give me",
    "outta": "out of",
    "kinda": "kind of",
    "sorta": "sort of",
    "hafta": "have to",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "musta": "must have",
    "betta": "better",
    "bruh": "brother",
    "cuz": "because",
    "wassup": "what's up",
    "ima": "I am going to",
}

# Patterns for grammatical features of AAVE
HABITUAL_BE_PATTERN = re.compile(r'\b(?:she|he|it|that|this|who|what|someone|everybody|they|we|you|people|things)\s+be\s+(\w+)', re.IGNORECASE)
COPULA_ABSENCE_PATTERN = re.compile(r'\b(?:she|he|it|that|this|what|who)\s+(?!is|are|was|were|be|been|being|ain\'t)(\w+ing|\w+ed|a\s+\w+|in\s+|at\s+|on\s+|with\s+|gonna|finna|tryna)', re.IGNORECASE)
NEGATIVE_CONCORD_PATTERN = re.compile(r"\bain't\s+no\b", re.IGNORECASE)

def aave_to_mae_rule_based(sentence: str) -> str:
    """
    Converts an AAVE sentence to a 'standardized' MAE version using
    rule-based lexical substitutions and syntactic adjustments.
    This is a simplified, linguistically-informed converter.
    """
    # 1. Lexical replacement (case-insensitive)
    for aave_word, mae_word in LEXICAL_MAP.items():
        # Use word boundaries to avoid partial replacements
        pattern = re.compile(r'\b' + re.escape(aave_word) + r'\b', re.IGNORECASE)
        sentence = pattern.sub(mae_word, sentence)
    
    # 2. Habitual 'be' -> appropriate form (often 'is/are')
    def habitual_be_replacer(match):
        subject = match.group(0).split()[0].lower()
        # Over-simplified: map 3rd singular to 'is', plural to 'are'
        if subject in ["she", "he", "it", "that", "this", "who", "what"]:
            return f"{subject} is {match.group(1)}"
        else:
            return f"{subject} are {match.group(1)}"
    sentence = HABITUAL_BE_PATTERN.sub(habitual_be_replacer, sentence)
    
    # 3. Copula absence: insert 'is' or 'are' before predicate
    def copula_absence_replacer(match):
        # Extract subject and predicate
        whole = match.group(0)
        parts = whole.split(maxsplit=1)
        if len(parts) == 2:
            subj, pred = parts
            if subj.lower() in ["she", "he", "it", "that", "this", "what", "who"]:
                return f"{subj} is {pred}"
            else:
                return f"{subj} are {pred}"
        return whole
    sentence = COPULA_ABSENCE_PATTERN.sub(copula_absence_replacer, sentence)
    
    # 4. Negative concord: "ain't no" -> "isn't any" (simplified)
    sentence = NEGATIVE_CONCORD_PATTERN.sub("isn't any", sentence)
    
    return sentence

def create_contrastive_pairs(sentences: List[str]) -> List[Tuple[str, str]]:
    """
    Generates (AAVE, MAE) pairs for contrastive training.
    Each pair shares the same semantics but differs in dialectal form.
    """
    pairs = []
    for s in sentences:
        mae_version = aave_to_mae_rule_based(s)
        pairs.append((s, mae_version))
    return pairs

def extract_linguistic_features(sentence: str) -> Dict[str, bool]:
    """
    Detects presence of key AAVE linguistic features in a sentence.
    Returns a dictionary with feature names as keys and boolean values.
    Used for diagnostic analysis.
    """
    features = {
        "habitual_be": bool(HABITUAL_BE_PATTERN.search(sentence)),
        "copula_absence": bool(COPULA_ABSENCE_PATTERN.search(sentence)),
        "negative_concord": bool(NEGATIVE_CONCORD_PATTERN.search(sentence)),
        "lexical_items": any(
            re.search(r'\b' + re.escape(word) + r'\b', sentence, re.IGNORECASE)
            for word in LEXICAL_MAP.keys()
        )
    }
    return features
