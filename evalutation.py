# evaluation.py
"""
Evaluation script for the fine-tuned AAVE-LLaMA model.
Measures perplexity on the held-out test set and generates
sample continuations in both AAVE and MAE styles.
"""

import os
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

from config import config

def load_model_and_tokenizer(model_path: str):
    """Load base model + LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        load_in_8bit=config.use_8bit,
        use_auth_token=True,
    )
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    return model, tokenizer

def compute_perplexity(model, tokenizer, test_file: str, stride: int = 512):
    """Compute perplexity using fixed-length sliding window."""
    test_file = os.path.join(config.processed_data_path, test_file)
    with open(test_file, "r", encoding="utf-8") as f:
        text = f.read()
    
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride), desc="Perplexity"):
        end_loc = min(begin_loc + config.max_seq_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = math.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()

def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 50):
    """Generate continuation from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(config.output_dir)
    
    # Perplexity
    ppl = compute_perplexity(model, tokenizer, "test.txt")
    print(f"Test Perplexity: {ppl:.2f}")
    
    # Sample generations
    prompts = [
        "I ain't gonna lie, yesterday was",
        "She be working at the store but",
        "They finna go to the party if",
    ]
    for prompt in prompts:
        gen = generate_text(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {gen}\n")
