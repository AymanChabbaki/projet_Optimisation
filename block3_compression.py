# ============================================================
# BLOCK 3 — Post-Training Compression
#           Dynamic INT8 Quantization (PyTorch)
# ============================================================
#
# Strategy: Dynamic quantization
#   • Weights are converted to INT8 statically (stored as INT8).
#   • Activations are quantized dynamically at inference time.
#   • No calibration dataset required → zero extra data cost.
#   • Targets all nn.Linear layers (the bulk of transformer FLOPs).
#
# Result: ~2–4× size reduction, ~1.5–2× CPU latency speedup,
#         with typically < 0.5% accuracy drop on classification.
# ============================================================

import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from block1_setup_data import MODEL_NAME, NUM_LABELS, BEST_MODEL_DIR, COMPRESSED_MODEL_DIR


# ── Utility: in-memory model size ────────────────────────────────────────────
def get_model_size_mb(model: nn.Module) -> float:
    """
    Estimate model memory footprint (MB) from parameter + buffer tensors.
    Works for both standard FP32 and dynamically-quantized INT8 models.
    """
    total = 0
    for tensor in (*model.parameters(), *model.buffers()):
        total += tensor.nelement() * tensor.element_size()
    return total / 1024**2


# ── Utility: on-disk size ────────────────────────────────────────────────────
def get_disk_size_mb(path: str) -> float:
    """Walk a directory tree and sum all file sizes (MB)."""
    total = 0
    for root, _, files in os.walk(path):
        for fname in files:
            total += os.path.getsize(os.path.join(root, fname))
    return total / 1024**2


# ── Quantization ─────────────────────────────────────────────────────────────
def apply_dynamic_quantization(
    model_dir:  str = BEST_MODEL_DIR,
    output_dir: str = COMPRESSED_MODEL_DIR,
) -> tuple[nn.Module, nn.Module]:
    """
    Load the best fine-tuned model, apply Dynamic INT8 Quantization to all
    nn.Linear layers, and persist the compressed weights.

    Returns
    -------
    quantized_model : nn.Module  — ready for CPU inference
    baseline_model  : nn.Module  — original FP32 model (for comparison)
    """
    print("\n" + "=" * 60)
    print("  BLOCK 3 — Dynamic INT8 Quantization")
    print("=" * 60)

    # ── Load baseline ────────────────────────────────────────────────────────
    print(f"\n[1/4] Loading FP32 baseline from '{model_dir}'...")
    baseline_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=NUM_LABELS
    )
    baseline_model.eval()

    # ── Quantize ─────────────────────────────────────────────────────────────
    print("[2/4] Applying torch.quantization.quantize_dynamic …")
    quantized_model = torch.quantization.quantize_dynamic(
        baseline_model,
        {nn.Linear},       # quantize every Linear projection
        dtype=torch.qint8,
    )
    quantized_model.eval()

    # ── Size report ──────────────────────────────────────────────────────────
    baseline_mb  = get_model_size_mb(baseline_model)
    quantized_mb = get_model_size_mb(quantized_model)
    ratio        = baseline_mb / quantized_mb if quantized_mb else float("inf")

    print(f"\n  In-memory size")
    print(f"    Baseline  (FP32) : {baseline_mb:>8.2f} MB")
    print(f"    Quantized (INT8) : {quantized_mb:>8.2f} MB")
    print(f"    Compression      : {ratio:>8.2f}×")

    # ── Persist ──────────────────────────────────────────────────────────────
    print(f"\n[3/4] Saving quantized weights to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    quant_weights_path = os.path.join(output_dir, "quantized_model.pt")
    torch.save(quantized_model.state_dict(), quant_weights_path)

    # Save config + tokenizer so the model can be reconstructed later
    baseline_model.config.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.save_pretrained(output_dir)

    print(f"    ✔ Weights : {quant_weights_path}")
    print(f"    ✔ Config  : {output_dir}/config.json")
    print(f"    ✔ Tokenizer saved")

    # ── On-disk sizes ─────────────────────────────────────────────────────────
    print("\n[4/4] On-disk size comparison:")
    baseline_disk  = get_disk_size_mb(model_dir)
    quantized_disk = get_disk_size_mb(output_dir)
    print(f"    Baseline  dir  ({model_dir})  : {baseline_disk:>8.2f} MB")
    print(f"    Quantized dir ({output_dir}) : {quantized_disk:>8.2f} MB")
    if quantized_disk > 0:
        print(f"    Disk ratio               : {baseline_disk/quantized_disk:>8.2f}×")

    print("\n[OK] Quantization complete.")
    print("=" * 60 + "\n")

    return quantized_model, baseline_model


# ── Loader (for downstream blocks) ───────────────────────────────────────────
def load_quantized_model(model_dir: str = COMPRESSED_MODEL_DIR) -> nn.Module:
    """
    Reconstruct a previously-saved dynamically-quantized model from disk.

    Steps
    -----
    1. Load the original architecture via config.
    2. Wrap it with quantize_dynamic (mirrors the structure saved in step 1).
    3. Load the INT8 state-dict into that structure.
    """
    print(f"[INFO] Reconstructing quantized model from '{model_dir}'...")
    config = AutoConfig.from_pretrained(model_dir)
    shell  = AutoModelForSequenceClassification.from_config(config)
    shell  = torch.quantization.quantize_dynamic(shell, {nn.Linear}, dtype=torch.qint8)

    weights_path = os.path.join(model_dir, "quantized_model.pt")
    state_dict   = torch.load(weights_path, map_location="cpu")
    shell.load_state_dict(state_dict)
    shell.eval()
    print("[INFO] Quantized model loaded successfully.")
    return shell


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q_model, fp32_model = apply_dynamic_quantization()

    # Quick sanity check: both models produce logits of the right shape
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    dummy = tok("Edge AI is fascinating", return_tensors="pt",
                padding="max_length", truncation=True, max_length=128)
    with torch.no_grad():
        fp32_out  = fp32_model(**dummy).logits
        quant_out = q_model(**dummy).logits
    print(f"FP32  logits : {fp32_out.tolist()}")
    print(f"INT8  logits : {quant_out.tolist()}")
    print("Logit diff (L∞):", (fp32_out - quant_out).abs().max().item())
