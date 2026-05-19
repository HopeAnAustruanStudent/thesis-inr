"""
Analyze sweep results and generate publication-ready tables and plots.

Usage:
    python -m experiments.analyze_results                              # default path
    python -m experiments.analyze_results --results experiments/results/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str | Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def aggregate_by_batch_size(results: list[dict]) -> dict:
    """Compute mean/std of metrics grouped by batch size."""
    groups: dict[str, list[dict]] = {}
    for r in results:
        key = str(r["batch_size"])
        groups.setdefault(key, []).append(r)

    summary = {}
    for bs, runs in groups.items():
        psnrs = [r["psnr"] for r in runs]
        ssims = [r["ssim"] for r in runs]
        times = [r["train_time_s"] for r in runs]
        spec_low = [r["spectral_band_errors"][0] for r in runs]
        spec_mid = [r["spectral_band_errors"][1] for r in runs]
        spec_high = [r["spectral_band_errors"][2] for r in runs]

        summary[bs] = {
            "psnr_mean": np.mean(psnrs),
            "psnr_std": np.std(psnrs),
            "ssim_mean": np.mean(ssims),
            "ssim_std": np.std(ssims),
            "time_mean": np.mean(times),
            "time_std": np.std(times),
            "spec_low_mean": np.mean(spec_low),
            "spec_mid_mean": np.mean(spec_mid),
            "spec_high_mean": np.mean(spec_high),
            "n_images": len(runs),
        }
    return summary


def print_latex_table(summary: dict):
    """Print a LaTeX-formatted results table."""
    order = ["256", "1024", "4096", "16384", "full"]
    order = [k for k in order if k in summary]

    print()
    print(r"\begin{tabular}{lccccc}")
    print(r"\toprule")
    print(r"Batch Size & PSNR (dB) $\uparrow$ & SSIM $\uparrow$ & "
          r"Time (s) & Spec. Low & Spec. High \\")
    print(r"\midrule")
    for bs in order:
        s = summary[bs]
        label = "Full" if bs == "full" else bs
        print(f"{label:>6s} & "
              f"${s['psnr_mean']:.2f} \\pm {s['psnr_std']:.2f}$ & "
              f"${s['ssim_mean']:.3f} \\pm {s['ssim_std']:.3f}$ & "
              f"${s['time_mean']:.1f}$ & "
              f"${s['spec_low_mean']:.4f}$ & "
              f"${s['spec_high_mean']:.3f}$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print()


def plot_aggregated(summary: dict, out_dir: Path):
    """Generate publication-quality aggregated plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    order = ["256", "1024", "4096", "16384", "full"]
    order = [k for k in order if k in summary]
    labels = ["Full" if k == "full" else k for k in order]

    psnr_means = [summary[k]["psnr_mean"] for k in order]
    psnr_stds = [summary[k]["psnr_std"] for k in order]
    ssim_means = [summary[k]["ssim_mean"] for k in order]
    ssim_stds = [summary[k]["ssim_std"] for k in order]
    time_means = [summary[k]["time_mean"] for k in order]

    # --- PSNR + SSIM dual-axis ---
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(order))
    width = 0.35

    bars1 = ax1.bar(x - width / 2, psnr_means, width, yerr=psnr_stds,
                    label="PSNR (dB)", color="#2196F3", edgecolor="black",
                    linewidth=0.5, capsize=3)
    ax1.set_ylabel("PSNR (dB)", color="#2196F3")
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1.set_xlabel("Batch Size")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width / 2, ssim_means, width, yerr=ssim_stds,
                    label="SSIM", color="#4CAF50", edgecolor="black",
                    linewidth=0.5, capsize=3)
    ax2.set_ylabel("SSIM", color="#4CAF50")
    ax2.tick_params(axis="y", labelcolor="#4CAF50")

    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.95))
    fig.suptitle("Average Quality vs Batch Size (Kodak, n=24)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "avg_quality_vs_bs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Time vs batch size ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, time_means, color="#FF9800", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Training Time (s)")
    ax.set_title("Average Training Time vs Batch Size")
    fig.tight_layout()
    fig.savefig(out_dir / "avg_time_vs_bs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Spectral error heatmap ---
    bands = ["Low", "Mid", "High"]
    spec_data = np.array([
        [summary[k][f"spec_{b}_mean"] for k in order]
        for b in ["low", "mid", "high"]
    ])

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(spec_data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(bands)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Frequency Band")
    ax.set_title("Mean Spectral Error by Batch Size and Frequency Band")

    for i in range(3):
        for j in range(len(order)):
            val = spec_data[i, j]
            color = "white" if val > spec_data.max() * 0.5 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=9)

    fig.colorbar(im, ax=ax, label="MSE")
    fig.tight_layout()
    fig.savefig(out_dir / "spectral_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- PSNR efficiency (PSNR / time) ---
    efficiency = [p / t for p, t in zip(psnr_means, time_means)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, efficiency, color="#9C27B0", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("PSNR / Time (dB/s)")
    ax.set_title("Training Efficiency: Quality per Second")
    fig.tight_layout()
    fig.savefig(out_dir / "efficiency_vs_bs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze sweep results")
    parser.add_argument("--results", default="experiments/results/results.json")
    parser.add_argument("--output", default="experiments/results/plots")
    args = parser.parse_args()

    results = load_results(args.results)
    summary = aggregate_by_batch_size(results)

    print(f"Loaded {len(results)} runs across {len(summary)} batch sizes\n")

    # Print summary table
    print("="*60)
    print("  Aggregated Results (mean +/- std across images)")
    print("="*60)
    for bs in ["256", "1024", "4096", "16384", "full"]:
        if bs not in summary:
            continue
        s = summary[bs]
        label = "Full" if bs == "full" else f"bs={bs}"
        print(f"  {label:>10s}: PSNR={s['psnr_mean']:.2f}+-{s['psnr_std']:.2f} dB  "
              f"SSIM={s['ssim_mean']:.3f}+-{s['ssim_std']:.3f}  "
              f"Time={s['time_mean']:.1f}s  "
              f"(n={s['n_images']})")

    print("\n--- LaTeX Table ---")
    print_latex_table(summary)

    # Generate plots
    plot_aggregated(summary, Path(args.output))


if __name__ == "__main__":
    main()
