# ============================================================
# BLOCK 1 — Setup, Dataset Loading & Preprocessing
# Edge AI Pipeline | MiniLM-L12 × AG News
# ============================================================
#
# Installation (run once in Colab / terminal):
#   pip install transformers datasets torch evaluate scikit-learn
#   pip install psutil accelerate lion-pytorch torch-optimizer
#
# ============================================================

import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

# ── Global Configuration ──────────────────────────────────────────────────────
MODEL_NAME   = "microsoft/MiniLM-L12-H384-uncased"
DATASET_NAME = "ag_news"
MAX_LENGTH   = 128
SEED         = 42

# Subset sizes — reduce to None to use the full dataset (~120 k / 7.6 k)
TRAIN_SAMPLES = 10_000
EVAL_SAMPLES  =  2_000

# Output directories
OUTPUT_DIR          = "./results"
BEST_MODEL_DIR      = "./best_model"
COMPRESSED_MODEL_DIR = "./compressed_model"

# AG News 4-class label mapping
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS  = 4

# Create output dirs upfront so all blocks can write to them freely
for _dir in (OUTPUT_DIR, BEST_MODEL_DIR, COMPRESSED_MODEL_DIR):
    os.makedirs(_dir, exist_ok=True)

# ── Data Loading & Tokenisation ───────────────────────────────────────────────

def load_and_preprocess_data(
    model_name    = MODEL_NAME,
    max_length    = MAX_LENGTH,
    train_samples = TRAIN_SAMPLES,
    eval_samples  = EVAL_SAMPLES,
    seed          = SEED,
):
    """
    Load AG News, optionally subsample, tokenize, and return HF DatasetDict.

    Returns
    -------
    tokenized : DatasetDict  — keys 'train' and 'test', torch tensors
    tokenizer : PreTrainedTokenizerFast
    """
    print("━" * 60)
    print("  BLOCK 1 — Data Loading & Preprocessing")
    print("━" * 60)

    # ── 1. Load raw dataset ──────────────────────────────────────────────────
    print(f"\n[1/3] Loading '{DATASET_NAME}' from Hugging Face Hub...")
    dataset = load_dataset(DATASET_NAME)

    if train_samples:
        dataset["train"] = (
            dataset["train"].shuffle(seed=seed).select(range(train_samples))
        )
    if eval_samples:
        dataset["test"] = (
            dataset["test"].shuffle(seed=seed).select(range(eval_samples))
        )

    print(f"      Train : {len(dataset['train']):,} examples")
    print(f"      Test  : {len(dataset['test']):,} examples")
    print(f"      Labels: {LABEL_NAMES}")

    # ── 2. Load tokenizer ────────────────────────────────────────────────────
    print(f"\n[2/3] Loading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    # ── 3. Tokenize ──────────────────────────────────────────────────────────
    print("[3/3] Tokenizing (batched)...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],   # drop raw text; keep 'label'
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    print("\n[OK] Preprocessing complete.")
    print(f"     Sample keys : {list(tokenized['train'][0].keys())}")
    print(f"     input_ids shape : {tokenized['train'][0]['input_ids'].shape}")
    print("━" * 60 + "\n")

    return tokenized, tokenizer


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    tok_ds, tok = load_and_preprocess_data()
    print("Label distribution (train):")
    from collections import Counter
    counts = Counter(tok_ds["train"]["labels"].tolist())
    for lbl, cnt in sorted(counts.items()):
        print(f"  {LABEL_NAMES[lbl]:<10} → {cnt:,}")
