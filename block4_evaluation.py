# ============================================================
# BLOCK 4 — Edge AI Evaluation & Latency Benchmarking
# Compares: Baseline FP32  vs.  Quantized INT8
# Measures: Disk size · Memory · Latency · Accuracy · F1
# ============================================================

import time
import os
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from block1_setup_data import MODEL_NAME, NUM_LABELS, BEST_MODEL_DIR, COMPRESSED_MODEL_DIR
from block3_compression import get_model_size_mb, get_disk_size_mb, load_quantized_model

# ── Metrics ───────────────────────────────────────────────────────────────────
_acc_metric = evaluate.load("accuracy")
_f1_metric  = evaluate.load("f1")

# ── Latency Measurement ───────────────────────────────────────────────────────

def measure_latency(
    model,
    tokenizer,
    sample_text: str,
    device: torch.device,
    n_warmup: int = 20,
    n_runs:   int = 200,
) -> dict[str, float]:
    """
    Measure single-example inference latency (batch_size=1).

    Warmup runs prime the JIT / cache; timed runs collect wall-clock
    per-example latency in milliseconds using time.perf_counter().

    Returns {'mean_ms': float, 'std_ms': float, 'p95_ms': float}
    """
    model.eval()
    model.to(device)

    enc = tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # Warm-up (not measured)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(**enc)

    # Timed runs
    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(**enc)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - t0) * 1_000)

    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms":  float(np.std(arr)),
        "p95_ms":  float(np.percentile(arr, 95)),
    }


# ── Accuracy / F1 Evaluation ──────────────────────────────────────────────────

def run_inference_accuracy(
    model,
    tokenizer,
    dataset,
    device: torch.device,
    batch_size: int = 64,
) -> dict[str, float]:
    """
    Run full-dataset inference and return accuracy + macro-F1.

    Parameters
    ----------
    dataset : HF Dataset with columns 'text' and ('label' or 'labels')
    """
    model.eval()
    model.to(device)

    label_col = "label" if "label" in dataset.column_names else "labels"
    all_preds: list[int] = []
    all_labels: list[int] = []

    for i in range(0, len(dataset), batch_size):
        batch_texts  = dataset["text"][i : i + batch_size]
        batch_labels = dataset[label_col][i : i + batch_size]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits

        preds = torch.argmax(logits, dim=-1).cpu().tolist()
        all_preds.extend(preds)

        # Handles both list[int] and torch.Tensor
        if hasattr(batch_labels, "tolist"):
            all_labels.extend(batch_labels.tolist())
        else:
            all_labels.extend(batch_labels)

    acc = _acc_metric.compute(predictions=all_preds, references=all_labels)["accuracy"]
    f1  = _f1_metric.compute( predictions=all_preds, references=all_labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}


# ── Full Edge AI Evaluation ───────────────────────────────────────────────────

def full_edge_evaluation(
    eval_samples: int  = 500,
    latency_runs: int  = 200,
) -> dict:
    """
    End-to-end Edge AI report comparing FP32 baseline vs. INT8 quantized model.

    Metrics
    -------
    • Model size on disk (MB)
    • In-memory parameter size (MB)
    • Compression ratio (disk & memory)
    • CPU inference latency — mean, std, P95 (ms)
    • Speedup factor
    • Accuracy & Macro-F1 on eval set
    • Accuracy & F1 drop-off

    Returns
    -------
    report : dict  — all results as a flat dictionary
    """
    # Edge AI inference is always CPU — no GPU acceleration at the edge
    device = torch.device("cpu")

    print("\n" + "=" * 65)
    print("  BLOCK 4 — Edge AI Evaluation Report")
    print("=" * 65)
    print(f"  Eval samples : {eval_samples}")
    print(f"  Latency runs : {latency_runs}  (batch_size=1, CPU)")
    print(f"  Device       : {device}\n")

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("[1/5] Loading evaluation subset...")
    raw_test = (
        load_dataset("ag_news", split="test")
        .shuffle(seed=42)
        .select(range(eval_samples))
    )
    sample_text = raw_test["text"][0]

    # ── Models ───────────────────────────────────────────────────────────────
    print("[2/5] Loading models...")
    tokenizer       = AutoTokenizer.from_pretrained(MODEL_NAME)
    baseline_model  = AutoModelForSequenceClassification.from_pretrained(
        BEST_MODEL_DIR, num_labels=NUM_LABELS
    ).eval()
    quantized_model = load_quantized_model(COMPRESSED_MODEL_DIR)

    # ── 1. Size metrics ───────────────────────────────────────────────────────
    baseline_disk_mb   = get_disk_size_mb(BEST_MODEL_DIR)
    quantized_disk_mb  = get_disk_size_mb(COMPRESSED_MODEL_DIR)
    baseline_mem_mb    = get_model_size_mb(baseline_model)
    quantized_mem_mb   = get_model_size_mb(quantized_model)
    compression_disk   = (baseline_disk_mb  / quantized_disk_mb)  if quantized_disk_mb  else float("inf")
    compression_mem    = (baseline_mem_mb   / quantized_mem_mb)   if quantized_mem_mb   else float("inf")

    # ── 2. Latency ────────────────────────────────────────────────────────────
    print("[3/5] Measuring inference latency...")
    baseline_lat  = measure_latency(baseline_model,  tokenizer, sample_text, device, n_runs=latency_runs)
    quantized_lat = measure_latency(quantized_model, tokenizer, sample_text, device, n_runs=latency_runs)
    speedup        = baseline_lat["mean_ms"] / quantized_lat["mean_ms"] if quantized_lat["mean_ms"] else float("inf")

    # ── 3. Accuracy / F1 ─────────────────────────────────────────────────────
    print("[4/5] Computing accuracy and F1...")
    baseline_metrics  = run_inference_accuracy(baseline_model,  tokenizer, raw_test, device)
    quantized_metrics = run_inference_accuracy(quantized_model, tokenizer, raw_test, device)
    acc_drop = baseline_metrics["accuracy"] - quantized_metrics["accuracy"]
    f1_drop  = baseline_metrics["f1_macro"] - quantized_metrics["f1_macro"]

    # ── 5. Report ─────────────────────────────────────────────────────────────
    print("[5/5] Report\n")
    W = 42  # column width

    def row(label, base, quant, fmt=".2f"):
        print(f"  {label:<{W}} {base:>10{fmt}} {quant:>12{fmt}}")

    def row_str(label, base, quant):
        print(f"  {label:<{W}} {str(base):>10} {str(quant):>12}")

    print(f"  {'Metric':<{W}} {'Baseline':>10} {'INT8 Quant':>12}")
    print("  " + "─" * (W + 24))
    row("Disk size (MB)",                  baseline_disk_mb,             quantized_disk_mb)
    row("In-memory size (MB)",             baseline_mem_mb,              quantized_mem_mb)
    row_str("Compression ratio — disk",   "—",                          f"{compression_disk:.2f}×")
    row_str("Compression ratio — memory", "—",                          f"{compression_mem:.2f}×")
    print("  " + "─" * (W + 24))
    row("Latency mean (ms)",               baseline_lat["mean_ms"],      quantized_lat["mean_ms"])
    row("Latency std  (ms)",               baseline_lat["std_ms"],       quantized_lat["std_ms"])
    row("Latency P95  (ms)",               baseline_lat["p95_ms"],       quantized_lat["p95_ms"])
    row_str("Speedup",                     "—",                          f"{speedup:.2f}×")
    print("  " + "─" * (W + 24))
    row("Accuracy",                        baseline_metrics["accuracy"], quantized_metrics["accuracy"], fmt=".4f")
    print(f"  {'Accuracy drop':<{W}} {'—':>10} {acc_drop:>+12.4f}")
    row("F1 Macro",                        baseline_metrics["f1_macro"], quantized_metrics["f1_macro"], fmt=".4f")
    print(f"  {'F1 Macro drop':<{W}} {'—':>10} {f1_drop:>+12.4f}")
    print("  " + "=" * (W + 24))

    report = {
        "baseline_disk_mb":        baseline_disk_mb,
        "quantized_disk_mb":       quantized_disk_mb,
        "baseline_mem_mb":         baseline_mem_mb,
        "quantized_mem_mb":        quantized_mem_mb,
        "compression_ratio_disk":  compression_disk,
        "compression_ratio_mem":   compression_mem,
        "baseline_latency":        baseline_lat,
        "quantized_latency":       quantized_lat,
        "speedup":                 speedup,
        "baseline_accuracy":       baseline_metrics["accuracy"],
        "quantized_accuracy":      quantized_metrics["accuracy"],
        "accuracy_drop":           acc_drop,
        "baseline_f1":             baseline_metrics["f1_macro"],
        "quantized_f1":            quantized_metrics["f1_macro"],
        "f1_drop":                 f1_drop,
    }

    # Persist report
    import json
    report_path = f"./results/edge_evaluation_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n[INFO] Report saved → '{report_path}'")

    return report


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    report = full_edge_evaluation(eval_samples=300, latency_runs=100)
