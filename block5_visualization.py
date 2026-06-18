# ============================================================
# BLOCK 5 — Visualization
# Futuristic dark theme — deep blues × electric cyan palette
# Plots: Training Loss · Val Loss · Accuracy/F1 · Radar · Edge AI
# ============================================================

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from pathlib import Path

from block1_setup_data import OUTPUT_DIR

# ─────────────────────────────────────────────────────────────────────────────
# PALETTE & THEME
# ─────────────────────────────────────────────────────────────────────────────

# Per-optimizer color assignment
PALETTE = {
    "AdamW":     "#00FFFF",   # Electric cyan
    "Adafactor": "#1E90FF",   # Dodger blue
    "Lion":      "#BF5FFF",   # Electric violet
    "LAMB":      "#FF6B35",   # Neon orange (contrast accent)
    "SGD":       "#39FF14",   # Neon green
}

# Dark background palette
BG         = "#050A1A"   # Near-black deep blue
BG_PANEL   = "#080E20"   # Slightly lighter panel
GRID_CLR   = "#0D2137"   # Very dark blue for grid lines
TEXT_CLR   = "#D0EEF8"   # Ice-white text
ACCENT     = "#00FFFF"   # Cyan — titles, spines, borders
BASE_CLR   = "#1E90FF"   # Baseline bars
QUANT_CLR  = "#00FFFF"   # Quantized bars


def apply_dark_theme() -> None:
    """Apply the futuristic dark academic theme globally."""
    matplotlib.rcParams.update({
        # Canvas
        "figure.facecolor":      BG,
        "figure.dpi":            120,
        # Axes
        "axes.facecolor":        BG_PANEL,
        "axes.edgecolor":        ACCENT,
        "axes.labelcolor":       TEXT_CLR,
        "axes.titlecolor":       TEXT_CLR,
        "axes.titlepad":         14,
        "axes.labelpad":         8,
        "axes.grid":             True,
        "axes.axisbelow":        True,
        # Grid
        "grid.color":            GRID_CLR,
        "grid.linewidth":        0.75,
        "grid.linestyle":        "--",
        # Ticks
        "xtick.color":           TEXT_CLR,
        "ytick.color":           TEXT_CLR,
        "xtick.labelsize":       9,
        "ytick.labelsize":       9,
        # Legend
        "legend.facecolor":      "#0A1628",
        "legend.edgecolor":      ACCENT,
        "legend.labelcolor":     TEXT_CLR,
        "legend.fontsize":       9,
        "legend.framealpha":     0.85,
        # Lines
        "lines.linewidth":       2.2,
        "lines.antialiased":     True,
        # Font
        "font.family":           "monospace",
        "font.size":             10,
        # Save
        "savefig.facecolor":     BG,
        "savefig.dpi":           180,
        "savefig.bbox":          "tight",
    })


def _spine_glow(ax, color: str = ACCENT, lw: float = 1.5) -> None:
    """Apply colored border to all four spines."""
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(lw)


def _smooth(values: list, window: int = 7) -> np.ndarray:
    """Moving-average smoothing — pads edges to avoid shrinkage."""
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: load saved benchmark results
# ─────────────────────────────────────────────────────────────────────────────

def load_results(path: str = f"{OUTPUT_DIR}/benchmark_results.json") -> list[dict]:
    with open(path) as fh:
        return json.load(fh)


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Training Loss (smoothed step-level)
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_loss(
    results: list[dict],
    save_path: str = f"{OUTPUT_DIR}/plot_train_loss.png",
) -> None:
    """
    Overlay smoothed training-loss curves for all optimizers.
    Raw signal shown at low opacity; smoothed curve on top.
    """
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(13, 6))

    for r in results:
        name   = r["optimizer"]
        losses = r.get("train_losses", [])
        if not losses:
            continue
        color    = PALETTE.get(name, "#FFFFFF")
        x        = np.linspace(0, 1, len(losses))
        raw      = np.array(losses, dtype=float)
        smoothed = _smooth(raw, window=9)

        ax.plot(x, raw,      color=color, alpha=0.12, linewidth=1.0)
        ax.plot(x, smoothed, color=color, label=name, linewidth=2.4)

    ax.set_title("Training Loss — Optimizer Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("Training Progress (normalized steps)")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend(loc="upper right")
    _spine_glow(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[PLOT 1] Training loss  → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Validation Loss per Epoch
# ─────────────────────────────────────────────────────────────────────────────

def plot_eval_loss(
    results: list[dict],
    save_path: str = f"{OUTPUT_DIR}/plot_eval_loss.png",
) -> None:
    apply_dark_theme()
    fig, ax = plt.subplots(figsize=(10, 5))

    for r in results:
        name   = r["optimizer"]
        losses = r.get("eval_losses", [])
        if not losses:
            continue
        color  = PALETTE.get(name, "#FFFFFF")
        epochs = np.arange(1, len(losses) + 1)
        ax.plot(epochs, losses, color=color, label=name,
                marker="o", markersize=7, markerfacecolor=BG,
                markeredgewidth=2)

    ax.set_title("Validation Loss per Epoch", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Loss")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc="upper right")
    _spine_glow(ax)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[PLOT 2] Validation loss → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Accuracy & F1 per Epoch (side by side)
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_f1(
    results: list[dict],
    save_path: str = f"{OUTPUT_DIR}/plot_accuracy_f1.png",
) -> None:
    apply_dark_theme()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for r in results:
        name  = r["optimizer"]
        color = PALETTE.get(name, "#FFFFFF")

        accs = r.get("eval_accuracies", [])
        f1s  = r.get("eval_f1s",        [])

        if accs:
            e = np.arange(1, len(accs) + 1)
            ax1.plot(e, accs, color=color, label=name,
                     marker="^", markersize=8, markerfacecolor=BG,
                     markeredgewidth=2)
        if f1s:
            e = np.arange(1, len(f1s) + 1)
            ax2.plot(e, f1s, color=color, label=name,
                     marker="s", markersize=8, markerfacecolor=BG,
                     markeredgewidth=2)

    for ax, title, ylabel in (
        (ax1, "Validation Accuracy per Epoch",    "Accuracy"),
        (ax2, "Validation Macro-F1 per Epoch",    "Macro F1-Score"),
    ):
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="lower right")
        _spine_glow(ax)

    fig.suptitle(
        "MiniLM-L12 · AG News — Optimizer Benchmark",
        fontsize=15, fontweight="bold", color=ACCENT, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[PLOT 3] Accuracy / F1  → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Multi-Dimensional Radar Chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_benchmark_radar(
    results: list[dict],
    save_path: str = f"{OUTPUT_DIR}/plot_radar.png",
) -> None:
    """
    Radar / spider chart comparing optimizers across 5 normalized dimensions:
    Accuracy · F1 · Speed · CPU Efficiency · Loss Stability.
    """
    apply_dark_theme()

    dims   = ["Accuracy", "F1 Macro", "Speed\n(1/time)", "CPU\nEfficiency", "Loss\nStability"]
    N      = len(dims)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor(BG_PANEL)
    fig.patch.set_facecolor(BG)

    def _norm(vals: list) -> list:
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn + 1e-9) for v in vals]

    accs    = _norm([r["final_accuracy"]              for r in results])
    f1s     = _norm([r["final_f1_macro"]              for r in results])
    speeds  = _norm([1.0 / r["training_time_s"]       for r in results])
    cpu_eff = _norm([1.0 / (r["peak_cpu_memory_mb"] + 1) for r in results])
    stab    = _norm([1.0 / (r["loss_std"] + 1e-6)     for r in results])

    for i, r in enumerate(results):
        name   = r["optimizer"]
        color  = PALETTE.get(name, "#FFFFFF")
        values = [accs[i], f1s[i], speeds[i], cpu_eff[i], stab[i]]
        values += values[:1]

        ax.plot(angles, values, color=color, linewidth=2.2, label=name)
        ax.fill(angles, values, color=color, alpha=0.07)
        # Highlight data points
        ax.scatter(angles[:-1], values[:-1], color=color, s=40, zorder=5)

    # Style
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=11, color=TEXT_CLR)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"],
                       color=TEXT_CLR, fontsize=8)
    ax.spines["polar"].set_color(ACCENT)
    ax.grid(color=GRID_CLR, linewidth=0.9)
    ax.tick_params(colors=TEXT_CLR)

    ax.set_title(
        "Optimizer Benchmark — Multi-Dimensional Radar\n(each axis normalized to [0, 1])",
        fontsize=13, fontweight="bold", color=ACCENT, pad=22,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[PLOT 4] Radar chart     → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Edge AI Compression Dashboard
# ─────────────────────────────────────────────────────────────────────────────

def plot_compression_dashboard(
    eval_report: dict,
    save_path: str = f"{OUTPUT_DIR}/plot_compression.png",
) -> None:
    """
    4-panel bar comparison: Disk Size · Latency · Accuracy · F1 Macro.
    """
    apply_dark_theme()

    panels = {
        "Disk Size (MB)":  (eval_report["baseline_disk_mb"],
                            eval_report["quantized_disk_mb"]),
        "Latency (ms)":    (eval_report["baseline_latency"]["mean_ms"],
                            eval_report["quantized_latency"]["mean_ms"]),
        "Accuracy":        (eval_report["baseline_accuracy"],
                            eval_report["quantized_accuracy"]),
        "F1 Macro":        (eval_report["baseline_f1"],
                            eval_report["quantized_f1"]),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    for ax, (label, (base_val, quant_val)) in zip(axes, panels.items()):
        colors = [BASE_CLR, QUANT_CLR]
        bars   = ax.bar(
            ["Baseline\nFP32", "Quantized\nINT8"],
            [base_val, quant_val],
            color=colors, width=0.5,
            edgecolor=ACCENT, linewidth=1.3,
        )
        # Value annotations
        for bar, val in zip(bars, [base_val, quant_val]):
            h   = bar.get_height()
            fmt = f"{val:.3f}" if val < 10 else f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h * 1.015,
                fmt,
                ha="center", va="bottom",
                color=TEXT_CLR, fontsize=9, fontweight="bold",
            )
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(base_val, quant_val) * 1.20)
        _spine_glow(ax)

    # Shared legend
    legend_handles = [
        Line2D([0], [0], color=BASE_CLR,  linewidth=7, label="Baseline FP32"),
        Line2D([0], [0], color=QUANT_CLR, linewidth=7, label="Quantized INT8"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        ncol=2, fontsize=10, framealpha=0.85,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        f"Edge AI Compression Report — FP32 vs. INT8 Dynamic Quantization\n"
        f"Speedup: {eval_report['speedup']:.2f}×  |  "
        f"Disk compression: {eval_report['compression_ratio_disk']:.2f}×  |  "
        f"Acc drop: {eval_report['accuracy_drop']:+.4f}",
        fontsize=12, fontweight="bold", color=ACCENT, y=1.03,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[PLOT 5] Compression     → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# BONUS PLOT — Memory & Timing Bar Chart (horizontal)
# ─────────────────────────────────────────────────────────────────────────────

def plot_memory_timing(
    results: list[dict],
    save_path: str = f"{OUTPUT_DIR}/plot_memory_timing.png",
) -> None:
    """
    Horizontal bar chart comparing peak CPU memory and training time per optimizer.
    """
    apply_dark_theme()

    names   = [r["optimizer"]          for r in results]
    times   = [r["training_time_s"]    for r in results]
    cpu_mbs = [r["peak_cpu_memory_mb"] for r in results]

    y   = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Training time
    colors = [PALETTE.get(n, "#FFF") for n in names]
    bars1  = ax1.barh(y, times, color=colors, edgecolor=ACCENT, linewidth=1.1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.set_xlabel("Training Time (seconds)")
    ax1.set_title("Training Time per Optimizer", fontweight="bold")
    for bar, val in zip(bars1, times):
        ax1.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}s", va="center", color=TEXT_CLR, fontsize=9)
    _spine_glow(ax1)

    # Peak CPU memory
    bars2 = ax2.barh(y, cpu_mbs, color=colors, edgecolor=ACCENT, linewidth=1.1)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names)
    ax2.set_xlabel("Peak CPU Memory (MB)")
    ax2.set_title("Peak CPU RAM per Optimizer", fontweight="bold")
    for bar, val in zip(bars2, cpu_mbs):
        ax2.text(val * 1.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f} MB", va="center", color=TEXT_CLR, fontsize=9)
    _spine_glow(ax2)

    fig.suptitle(
        "Optimizer Efficiency — Time & Memory",
        fontsize=14, fontweight="bold", color=ACCENT, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[BONUS]  Memory/timing  → {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# MASTER ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_plots(
    results: list[dict] | None   = None,
    eval_report: dict    | None  = None,
) -> None:
    """
    Generate and save all visualization plots.

    Parameters
    ----------
    results     : list of per-optimizer dicts (from block2).
                  If None, loads from OUTPUT_DIR/benchmark_results.json.
    eval_report : dict from block4 full_edge_evaluation().
                  If None, Plot 5 (compression dashboard) is skipped.
    """
    if results is None:
        results = load_results()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    plot_training_loss(results)
    plot_eval_loss(results)
    plot_accuracy_f1(results)
    plot_benchmark_radar(results)
    plot_memory_timing(results)

    if eval_report is not None:
        plot_compression_dashboard(eval_report)
    else:
        # Try loading from disk
        rpt_path = f"{OUTPUT_DIR}/edge_evaluation_report.json"
        if Path(rpt_path).exists():
            with open(rpt_path) as fh:
                plot_compression_dashboard(json.load(fh))
        else:
            print("[INFO] No eval report found — skipping compression dashboard.")

    print(f"\n[DONE] All plots saved in '{OUTPUT_DIR}/'")


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_all_plots()
