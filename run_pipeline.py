# ============================================================
# MASTER PIPELINE RUNNER
# Orchestrates all 5 blocks end-to-end.
# ============================================================
#
# Run as a script:
#   python run_pipeline.py
#
# Or paste each section into a Jupyter / Colab cell.
# ============================================================

# ── [CELL 0] Install dependencies ────────────────────────────────────────────
# Uncomment and run once in Colab:
#
# !pip install -q transformers datasets torch evaluate scikit-learn
# !pip install -q psutil accelerate
# !pip install -q lion-pytorch        # Lion optimizer
# !pip install -q torch-optimizer     # LAMB optimizer
# !pip install -q matplotlib

# ── [CELL 1] Block 1 — Data ───────────────────────────────────────────────────
from block1_setup_data import load_and_preprocess_data

tokenized_dataset, tokenizer = load_and_preprocess_data()

# ── [CELL 2] Block 2 — Train all optimizers ──────────────────────────────────
from block2_training_loop import run_optimizer_benchmark

all_results, best_result = run_optimizer_benchmark(
    tokenized_dataset,
    lr          = 2e-5,
    num_epochs  = 3,      # ← increase to 5 for production runs
    batch_size  = 32,
)
print(f"\nBest optimizer : {best_result['optimizer']}")
print(f"Best F1        : {best_result['final_f1_macro']:.4f}")

# ── [CELL 3] Block 3 — Quantize best model ───────────────────────────────────
from block3_compression import apply_dynamic_quantization

quantized_model, baseline_model = apply_dynamic_quantization()

# ── [CELL 4] Block 4 — Edge AI evaluation ────────────────────────────────────
from block4_evaluation import full_edge_evaluation

eval_report = full_edge_evaluation(eval_samples=500, latency_runs=200)

# ── [CELL 5] Block 5 — Generate all plots ────────────────────────────────────
from block5_visualization import generate_all_plots

generate_all_plots(results=all_results, eval_report=eval_report)

print("\n" + "=" * 60)
print("  Pipeline complete.")
print("  Outputs in ./results/  |  ./best_model/  |  ./compressed_model/")
print("=" * 60)
