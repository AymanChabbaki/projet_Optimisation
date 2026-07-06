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
import warnings
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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


# ── Internal helper ──────────────────────────────────────────────────────────
def _apply_quantization(model: nn.Module) -> nn.Module:
    """
    Apply Dynamic INT8 Quantization to all nn.Linear layers.
    Suppresses the torch.ao deprecation warning introduced in PyTorch 2.10.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        ).eval()


# ── Quantization ─────────────────────────────────────────────────────────────
def apply_dynamic_quantization(
    model_dir:  str = BEST_MODEL_DIR,
    output_dir: str = COMPRESSED_MODEL_DIR,
) -> tuple[nn.Module, nn.Module]:
    """
    Load the best fine-tuned model, apply Dynamic INT8 Quantization to all
    nn.Linear layers, and persist a copy for disk-size measurement.

    Saving strategy
    ---------------
    We use torch.save(whole_model) — NOT state_dict() — to avoid the
    _packed_params key-mismatch bug that affects state_dict loading across
    PyTorch versions (≥ 2.10 changed the quantized Linear internals).

    The FP32 model dir is recorded in a marker file so load_quantized_model()
    can always re-derive from the authoritative FP32 checkpoint instead of
    deserialising quantized weights (safest across versions).

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
    ).eval()

    # ── Quantize ─────────────────────────────────────────────────────────────
    print("[2/4] Applying Dynamic INT8 Quantization to Linear layers...")
    quantized_model = _apply_quantization(baseline_model)

    # ── Size report ──────────────────────────────────────────────────────────
    baseline_mb  = get_model_size_mb(baseline_model)
    quantized_mb = get_model_size_mb(quantized_model)
    ratio        = baseline_mb / quantized_mb if quantized_mb else float("inf")

    print(f"\n  In-memory size")
    print(f"    Baseline  (FP32) : {baseline_mb:>8.2f} MB")
    print(f"    Quantized (INT8) : {quantized_mb:>8.2f} MB")
    print(f"    Compression      : {ratio:>8.2f}x")

    # ── Persist ──────────────────────────────────────────────────────────────
    # Save the full model object (not state_dict) to avoid key-format issues.
    # Also write a marker so load_quantized_model knows the FP32 source dir.
    print(f"\n[3/4] Saving to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)

    torch.save(quantized_model, os.path.join(output_dir, "quantized_model_full.pt"))
    baseline_model.config.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(MODEL_NAME).save_pretrained(output_dir)

    # Marker: stores the path to the FP32 source so load_quantized_model can
    # re-derive the quantized model without touching the saved .pt object.
    with open(os.path.join(output_dir, "fp32_source.txt"), "w") as fh:
        fh.write(os.path.abspath(model_dir))

    print(f"    Saved quantized model, config, and tokenizer.")

    # ── On-disk sizes ─────────────────────────────────────────────────────────
    print("\n[4/4] On-disk size comparison:")
    baseline_disk  = get_disk_size_mb(model_dir)
    quantized_disk = get_disk_size_mb(output_dir)
    print(f"    Baseline  dir : {baseline_disk:>8.2f} MB")
    print(f"    Quantized dir : {quantized_disk:>8.2f} MB")
    if quantized_disk > 0:
        print(f"    Disk ratio    : {baseline_disk/quantized_disk:>8.2f}x")

    print("\n[OK] Quantization complete.")
    print("=" * 60 + "\n")

    return quantized_model, baseline_model


# ── Loader (for downstream blocks) ───────────────────────────────────────────
def load_quantized_model(compressed_dir: str = COMPRESSED_MODEL_DIR) -> nn.Module:
    """
    Return a Dynamic INT8 model for inference.

    Strategy
    --------
    We always load the FP32 checkpoint and re-apply quantization rather than
    deserialising a saved INT8 state_dict.  This is the only approach that is
    robust across PyTorch versions, because the internal key format of
    quantized Linear layers (_packed_params) changed in PyTorch 2.10 and
    causes 'missing weight / bias' errors when loading a state_dict saved
    with a different version.

    Re-applying quantization from FP32 is deterministic, takes < 1 second,
    and requires no saved quantized weights at all.
    """
    # Prefer the recorded FP32 source dir; fall back to BEST_MODEL_DIR
    marker = os.path.join(compressed_dir, "fp32_source.txt")
    if os.path.exists(marker):
        with open(marker) as fh:
            fp32_dir = fh.read().strip()
    else:
        fp32_dir = BEST_MODEL_DIR

    print(f"[INFO] Loading FP32 from '{fp32_dir}' and re-applying INT8 quantization...")
    model = AutoModelForSequenceClassification.from_pretrained(
        fp32_dir, num_labels=NUM_LABELS
    ).eval()
    quantized = _apply_quantization(model)
    print("[INFO] Quantized model ready.")
    return quantized


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
