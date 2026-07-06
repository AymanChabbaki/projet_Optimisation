# ============================================================
# BLOCK 2 — Optimizer Benchmark Training Loop
# Optimizers: AdamW · Adafactor · Lion · LAMB · SGD
# Tracks: Accuracy · Macro-F1 · Time · Memory · Loss Stability
# ============================================================

import time
import json
import psutil
import numpy as np
import torch
import torch.nn as nn
import evaluate
from dataclasses import dataclass
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    get_linear_schedule_with_warmup,
)
from transformers.optimization import Adafactor, AdafactorSchedule
from torch.optim import SGD
from torch.optim import AdamW  # PyTorch native AdamW (fused when available)

from block1_setup_data import (
    MODEL_NAME, NUM_LABELS, BEST_MODEL_DIR, OUTPUT_DIR, SEED,
    load_and_preprocess_data,
)

# ── Optional optimizers (graceful skip if not installed) ─────────────────────
try:
    from lion_pytorch import Lion
    HAS_LION = True
except ImportError:
    HAS_LION = False
    print("[WARN] lion-pytorch not installed → pip install lion-pytorch")

try:
    import torch_optimizer as topt
    HAS_LAMB = True
except ImportError:
    HAS_LAMB = False
    print("[WARN] torch-optimizer not installed → pip install torch-optimizer")

# ── Evaluation Metrics ────────────────────────────────────────────────────────
_acc_metric = evaluate.load("accuracy")
_f1_metric  = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = _acc_metric.compute(predictions=preds, references=labels)["accuracy"]
    f1  = _f1_metric.compute( predictions=preds, references=labels, average="macro")["f1"]
    return {"accuracy": acc, "f1_macro": f1}


# ── Memory Tracking Callback ──────────────────────────────────────────────────
class MemoryTrackerCallback(TrainerCallback):
    """
    Records RSS CPU memory and (if available) peak GPU memory
    after every optimizer step. Sampled every `sample_every` steps
    to keep overhead low on long runs.
    """

    def __init__(self, sample_every: int = 25):
        self.sample_every = sample_every
        self._step = 0
        self.cpu_mem_mb: list[float] = []
        self.gpu_mem_mb: list[float] = []

    def on_step_end(self, args, state, control, **kwargs):
        self._step += 1
        if self._step % self.sample_every != 0:
            return
        proc = psutil.Process()
        self.cpu_mem_mb.append(proc.memory_info().rss / 1024**2)
        if torch.cuda.is_available():
            self.gpu_mem_mb.append(
                torch.cuda.max_memory_allocated() / 1024**2
            )

    @property
    def peak_cpu(self) -> float:
        return max(self.cpu_mem_mb, default=0.0)

    @property
    def peak_gpu(self) -> float:
        return max(self.gpu_mem_mb, default=0.0)


# ── Optimizer & Scheduler Factory ────────────────────────────────────────────
def _param_groups(model, weight_decay: float = 0.01):
    """Standard grouped params: no weight-decay on bias / LayerNorm."""
    no_decay = {"bias", "LayerNorm.weight"}
    return [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


def build_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    n_train: int,
    num_epochs: int,
    batch_size: int,
):
    """
    Return (optimizer, lr_scheduler) for the requested optimizer name.

    Notes
    -----
    • Lion uses lr × 0.1 as recommended by its authors.
    • Adafactor manages its own learning-rate internally (relative_step=True).
    """
    n_steps    = (n_train // batch_size) * num_epochs
    n_warmup   = max(1, n_steps // 10)
    pg         = _param_groups(model)
    flat_params = [p for p in model.parameters() if p.requires_grad]

    if name == "AdamW":
        opt  = AdamW(pg, lr=lr, eps=1e-8)
        sched = get_linear_schedule_with_warmup(opt, n_warmup, n_steps)

    elif name == "Adafactor":
        # relative_step=True → Adafactor picks its own lr; pass lr=None
        opt   = Adafactor(
            model.parameters(),
            lr=None,
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
        )
        sched = AdafactorSchedule(opt)

    elif name == "Lion":
        if not HAS_LION:
            raise ImportError("Run: pip install lion-pytorch")
        opt   = Lion(pg, lr=lr * 0.1, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, n_warmup, n_steps)

    elif name == "LAMB":
        if not HAS_LAMB:
            raise ImportError("Run: pip install torch-optimizer")
        opt   = topt.Lamb(pg, lr=lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, n_warmup, n_steps)

    elif name == "SGD":
        opt   = SGD(flat_params, lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
        sched = get_linear_schedule_with_warmup(opt, n_warmup, n_steps)

    else:
        raise ValueError(f"Unknown optimizer: '{name}'")

    return opt, sched


# ── Single-Optimizer Training Run ────────────────────────────────────────────
def train_with_optimizer(
    optimizer_name: str,
    tokenized_dataset,
    lr: float      = 2e-5,
    num_epochs: int = 3,
    batch_size: int = 32,
) -> tuple[dict, Trainer]:
    """
    Fine-tune MiniLM-L12 on AG News with one optimizer.

    Returns
    -------
    result : dict  — all tracked metrics
    trainer : Trainer  — trained Trainer object (holds best model checkpoint)
    """
    print(f"\n{'━'*60}")
    print(f"  Optimizer: {optimizer_name}")
    print(f"{'━'*60}")

    # Fresh model per experiment
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    optimizer, scheduler = build_optimizer(
        optimizer_name,
        model,
        lr,
        n_train    = len(tokenized_dataset["train"]),
        num_epochs = num_epochs,
        batch_size = batch_size,
    )

    mem_cb = MemoryTrackerCallback(sample_every=20)

    training_args = TrainingArguments(
        output_dir                  = f"{OUTPUT_DIR}/{optimizer_name}",
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = min(batch_size * 2, 128),
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        logging_dir                 = f"{OUTPUT_DIR}/{optimizer_name}/logs",
        logging_steps               = 50,
        report_to                   = "none",
        seed                        = SEED,
        # Mixed-precision: fp16 on CUDA, plain FP32 on CPU
        fp16                        = torch.cuda.is_available(),
        # pin_memory speeds up CPU→GPU tensor transfers significantly
        dataloader_pin_memory       = torch.cuda.is_available(),
        # 0 workers is required on Windows; increase to 2-4 on Linux/Colab
        dataloader_num_workers      = 0,
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = tokenized_dataset["train"],
        eval_dataset     = tokenized_dataset["test"],
        compute_metrics  = compute_metrics,
        optimizers       = (optimizer, scheduler),
        callbacks        = [mem_cb],
    )

    # ── Train ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0

    # Final evaluation on best checkpoint
    eval_out = trainer.evaluate()

    # ── Parse log history ────────────────────────────────────────────────────
    logs         = trainer.state.log_history
    train_losses = [x["loss"]            for x in logs if "loss"            in x and "eval_loss" not in x]
    eval_losses  = [x["eval_loss"]       for x in logs if "eval_loss"       in x]
    eval_accs    = [x["eval_accuracy"]   for x in logs if "eval_accuracy"   in x]
    eval_f1s     = [x["eval_f1_macro"]   for x in logs if "eval_f1_macro"   in x]

    result = {
        "optimizer":          optimizer_name,
        "final_accuracy":     eval_out["eval_accuracy"],
        "final_f1_macro":     eval_out["eval_f1_macro"],
        "training_time_s":    round(elapsed, 2),
        "peak_cpu_memory_mb": round(mem_cb.peak_cpu, 2),
        "peak_gpu_memory_mb": round(mem_cb.peak_gpu, 2),
        "train_losses":       train_losses,
        "eval_losses":        eval_losses,
        "eval_accuracies":    eval_accs,
        "eval_f1s":           eval_f1s,
        # Loss std-dev over the last 50% of training steps = stability proxy
        "loss_std": float(np.std(train_losses[len(train_losses)//2:])) if train_losses else 0.0,
    }

    print(f"\n  ✔ Accuracy     : {result['final_accuracy']:.4f}")
    print(f"  ✔ F1 (macro)   : {result['final_f1_macro']:.4f}")
    print(f"  ✔ Train time   : {elapsed:.1f} s")
    print(f"  ✔ Peak CPU RAM : {mem_cb.peak_cpu:.0f} MB")
    print(f"  ✔ Peak GPU RAM : {mem_cb.peak_gpu:.0f} MB")
    print(f"  ✔ Loss Std-Dev : {result['loss_std']:.5f}")

    return result, trainer


# ── Master Benchmark Loop ─────────────────────────────────────────────────────
def _auto_batch_size() -> int:
    """Pick batch size based on available VRAM. MiniLM-L12 + fp16 + seq128."""
    if not torch.cuda.is_available():
        return 32
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if vram_gb >= 16:
        return 256
    if vram_gb >= 8:
        return 128
    if vram_gb >= 4:
        return 64   # Quadro T1000 sweet spot
    return 32


def run_optimizer_benchmark(
    tokenized_dataset,
    lr: float       = 2e-5,
    num_epochs: int = 3,
    batch_size: int | None = None,   # None → auto-detect from VRAM
) -> tuple[list[dict], dict]:
    """
    Run all 5 optimizer experiments sequentially.
    Saves the best model and a JSON results file.

    Returns
    -------
    all_results : list[dict]
    best_result : dict   (optimizer with highest final F1)
    """
    OPTIMIZERS = ["AdamW", "Adafactor", "Lion", "LAMB", "SGD"]

    if batch_size is None:
        batch_size = _auto_batch_size()

    all_results  = []
    best_result  = None
    best_trainer = None
    best_f1      = -1.0

    device_info = (
        f"CUDA ({torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory//1024**3} GB)"
        if torch.cuda.is_available() else "CPU"
    )
    print("\n" + "=" * 60)
    print("  BLOCK 2 — Optimizer Benchmark")
    print(f"  Device     : {device_info}")
    print(f"  Optimizers : {OPTIMIZERS}")
    print(f"  Epochs     : {num_epochs}  |  Batch : {batch_size}  |  LR : {lr}")
    print("  NOTE: first 5-10 steps are slow (CUDA kernel warmup) — normal!")
    print("=" * 60)

    for opt_name in OPTIMIZERS:
        try:
            result, trainer = train_with_optimizer(
                opt_name, tokenized_dataset,
                lr=lr, num_epochs=num_epochs, batch_size=batch_size,
            )
            all_results.append(result)

            if result["final_f1_macro"] > best_f1:
                best_f1      = result["final_f1_macro"]
                best_result  = result
                best_trainer = trainer

        except ImportError as e:
            print(f"\n[SKIP] {opt_name} — missing dependency: {e}\n")
        except Exception as e:
            print(f"\n[ERROR] {opt_name} — {e}\n")

    # ── Save best model checkpoint ────────────────────────────────────────────
    if best_trainer is not None:
        best_trainer.save_model(BEST_MODEL_DIR)
        best_trainer.tokenizer.save_pretrained(BEST_MODEL_DIR) if hasattr(best_trainer, "tokenizer") else None
        print(f"\n[INFO] Best model '{best_result['optimizer']}' (F1={best_f1:.4f}) saved → '{BEST_MODEL_DIR}'")

    # ── Persist results ────────────────────────────────────────────────────────
    results_path = f"{OUTPUT_DIR}/benchmark_results.json"
    with open(results_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"[INFO] Benchmark results → '{results_path}'")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {'Optimizer':<12} {'Accuracy':>9} {'F1':>9} {'Time(s)':>9} {'CPU MB':>9}")
    print("  " + "-" * 52)
    for r in sorted(all_results, key=lambda x: x["final_f1_macro"], reverse=True):
        mark = " ★" if r["optimizer"] == best_result["optimizer"] else ""
        print(
            f"  {r['optimizer']:<12} "
            f"{r['final_accuracy']:>9.4f} "
            f"{r['final_f1_macro']:>9.4f} "
            f"{r['training_time_s']:>9.1f} "
            f"{r['peak_cpu_memory_mb']:>9.0f}"
            f"{mark}"
        )
    print("=" * 60)

    return all_results, best_result


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tokenized_dataset, _ = load_and_preprocess_data()
    all_results, best = run_optimizer_benchmark(tokenized_dataset)
    print(f"\nBest → {best['optimizer']}  F1={best['final_f1_macro']:.4f}")
