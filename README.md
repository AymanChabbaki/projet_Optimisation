# Edge AI Pipeline — MiniLM-L12 × AG News

> Fine-tuning · Optimizer Benchmarking · INT8 Quantization · Edge Evaluation  
> Master's project — Data Science & AI

---

## Overview

This pipeline benchmarks five optimizers for fine-tuning a Small Language Model (SLM) on a 4-class text classification task, then compresses the best model for CPU-constrained edge deployment using Dynamic INT8 Quantization.

**Model** — `microsoft/MiniLM-L12-H384-uncased`  
**Dataset** — `ag_news` (World · Sports · Business · Sci/Tech)  
**Task** — Sequence classification (4 classes)

---

## Project Structure

```
projet_Optimisation/
│
├── edge_ai_pipeline.ipynb      ← Main notebook (self-contained, run this)
│
├── block1_setup_data.py        ← Dataset loading & tokenization
├── block2_training_loop.py     ← 5-optimizer benchmark with memory tracking
├── block3_compression.py       ← Dynamic INT8 quantization
├── block4_evaluation.py        ← Edge AI evaluation & latency benchmarking
├── block5_visualization.py     ← Publication-quality plots
├── run_pipeline.py             ← Script orchestrator (non-notebook runner)
│
├── results/                    ← Generated at runtime
│   ├── benchmark_results.json
│   ├── edge_evaluation_report.json
│   ├── plot_train_loss.png
│   ├── plot_eval_loss.png
│   ├── plot_accuracy_f1.png
│   ├── plot_radar.png
│   ├── plot_memory_timing.png
│   └── plot_compression.png
│
├── best_model/                 ← Best FP32 checkpoint (generated at runtime)
└── compressed_model/           ← INT8 weights + config (generated at runtime)
```

---

## Installation

### 1. PyTorch with CUDA (GPU — recommended)

> **Important:** `pip install torch` installs the CPU-only build by default.  
> You must use the PyTorch wheel index to get GPU support.

```bash
# Check your driver's CUDA version first
nvidia-smi

# Install for CUDA 12.6 (compatible with drivers reporting CUDA 12.x–13.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Or CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify:
```python
import torch
print(torch.cuda.is_available())      # True
print(torch.cuda.get_device_name(0))  # e.g. Quadro T1000
```

### 2. All other dependencies

```bash
pip install transformers datasets evaluate scikit-learn
pip install psutil accelerate
pip install lion-pytorch      # Lion optimizer
pip install torch-optimizer   # LAMB optimizer
pip install matplotlib
```

### Google Colab

Paste and run this as the first cell:

```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers datasets evaluate scikit-learn psutil accelerate
!pip install -q lion-pytorch torch-optimizer matplotlib
```

---

## Quickstart

### Option A — Jupyter Notebook (recommended)

Open `edge_ai_pipeline.ipynb` and run cells top-to-bottom.  
Each block is a self-contained section; you can run them independently after the config cell.

### Option B — Python script

```bash
python run_pipeline.py
```

---

## Pipeline Blocks

### Block 1 · Setup, Dataset Loading & Preprocessing

- Downloads `ag_news` from Hugging Face Hub
- Optionally subsamples (default: 10 000 train / 2 000 test for fast iteration)
- Tokenizes with `AutoTokenizer` using `max_length=128`, padding, and truncation
- Returns a `DatasetDict` of PyTorch tensors

| Config variable | Default | Description |
|---|---|---|
| `TRAIN_SAMPLES` | `10_000` | Set `None` for full ~120 k |
| `EVAL_SAMPLES` | `2_000` | Set `None` for full ~7.6 k |
| `MAX_LENGTH` | `128` | Token sequence length |

---

### Block 2 · Optimizer Benchmark Training Loop

Fine-tunes the model five times (one per optimizer) using the HuggingFace `Trainer` API.

| Optimizer | Implementation | Notes |
|---|---|---|
| **AdamW** | `torch.optim.AdamW` | Standard baseline |
| **Adafactor** | `transformers.Adafactor` | Self-adaptive LR, memory-efficient |
| **Lion** | `lion_pytorch.Lion` | Sign-based momentum; LR × 0.1 |
| **LAMB** | `torch_optimizer.Lamb` | Layer-wise adaptive moments |
| **SGD** | `torch.optim.SGD` | Nesterov momentum, weight decay |

**Tracked metrics per optimizer:**

| Metric | How |
|---|---|
| Accuracy | `evaluate` library |
| Macro F1-score | `evaluate` library |
| Training time | `time.perf_counter()` |
| Peak CPU memory | `psutil` RSS |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` |
| Loss stability | Std-dev of loss over the last 50% of training steps |

Best model (highest Macro-F1) is automatically saved to `./best_model/`.

---

### Block 3 · Post-Training Compression

Applies **Dynamic INT8 Quantization** via `torch.quantization.quantize_dynamic`.

```
torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
```

| Property | Value |
|---|---|
| Target layers | All `nn.Linear` (attention projections, FFN) |
| Weight precision | `qint8` (stored as INT8) |
| Activation precision | Quantized dynamically at inference |
| Calibration data needed | None |
| Typical size reduction | 2–4× |
| Typical accuracy drop | < 0.5% on classification |

The compressed weights are saved as `compressed_model/quantized_model.pt` alongside the model config and tokenizer, so the quantized model can be fully reconstructed from disk.

---

### Block 4 · Edge AI Evaluation & Latency Benchmarking

Compares the **FP32 baseline** against the **INT8 quantized** model on metrics relevant to edge deployment.

| Metric | Method |
|---|---|
| Disk size (MB) | Directory file-size walk |
| In-memory size (MB) | Sum of parameter + buffer bytes |
| Compression ratio | Baseline ÷ quantized (disk and memory) |
| Latency mean / std / P95 (ms) | 20 warmup + 200 timed runs, `batch_size=1`, CPU |
| Speedup factor | Baseline latency ÷ quantized latency |
| Accuracy & Macro-F1 drop | Full-pass inference on eval subset |

> Latency is measured on **CPU only**, which reflects realistic edge conditions where no GPU accelerator is available.

---

### Block 5 · Visualization

Six publication-quality plots using a futuristic dark academic theme.

| Plot | File | Description |
|---|---|---|
| 1 | `plot_train_loss.png` | Smoothed training loss — all optimizers overlaid |
| 2 | `plot_eval_loss.png` | Validation loss per epoch |
| 3 | `plot_accuracy_f1.png` | Accuracy & Macro-F1 side by side per epoch |
| 4 | `plot_radar.png` | Radar chart — 5-dim normalized optimizer comparison |
| 5 | `plot_memory_timing.png` | Horizontal bars — training time & peak CPU RAM |
| 6 | `plot_compression.png` | Edge AI dashboard — FP32 vs INT8 on 4 metrics |

**Color palette:**

| Optimizer | Color |
|---|---|
| AdamW | `#00FFFF` Electric cyan |
| Adafactor | `#1E90FF` Dodger blue |
| Lion | `#BF5FFF` Electric violet |
| LAMB | `#FF6B35` Neon orange |
| SGD | `#39FF14` Neon green |

---

## Hardware Notes

| Setup | Expected training time (3 epochs, 10 k samples) |
|---|---|
| NVIDIA GPU (4 GB VRAM) | ~3–6 min per optimizer |
| CPU only | ~15–30 min per optimizer |

The pipeline automatically enables `fp16` mixed-precision training when a CUDA GPU is detected.  
For the Quadro T1000 (4 GB), `batch_size=32` fits comfortably; increase to `64` if VRAM allows.

---

## Expected Results

Approximate benchmarks on the 10 000-sample subset after 3 epochs:

| Optimizer | Accuracy | Macro-F1 | Relative speed |
|---|---|---|---|
| AdamW | ~0.92 | ~0.92 | Baseline |
| Adafactor | ~0.91 | ~0.91 | ~1.1× faster |
| Lion | ~0.91 | ~0.91 | ~1.0× |
| LAMB | ~0.92 | ~0.91 | ~0.9× |
| SGD | ~0.88 | ~0.88 | ~1.2× faster |

Quantization impact (typical):

| Metric | Value |
|---|---|
| Disk compression | ~3–4× |
| CPU latency speedup | ~1.5–2× |
| Accuracy drop | < 0.3% |
| F1 drop | < 0.3% |

---

## References

- [MiniLM: Deep Self-Attention Distillation](https://arxiv.org/abs/2002.10957) — Wang et al., 2020
- [AG News Corpus](https://huggingface.co/datasets/ag_news) — Zhang et al., 2015
- [Symbolic Discovery of Optimization Algorithms (Lion)](https://arxiv.org/abs/2302.06675) — Chen et al., 2023
- [Large Batch Optimization for Deep Learning (LAMB)](https://arxiv.org/abs/1904.00962) — You et al., 2019
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1901.09827) — Shazeer & Stern, 2018
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
