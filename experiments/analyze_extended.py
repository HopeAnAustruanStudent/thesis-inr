"""
Analysis and plotting for extended experiments.

Usage:
    python -m experiments.analyze_extended
    python -m experiments.analyze_extended --input experiments/extended
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  Warning: {path} not found, skipping.")
        return []
    with open(path) as f:
        return json.load(f)


# ===================================================================
# 1. SIREN vs COIN comparison
# ===================================================================

def plot_arch_comparison(siren_results: list[dict], coin_results: list[dict],
                         out_dir: Path):
    """Compare SIREN vs COIN across batch sizes."""
    if not siren_results or not coin_results:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    def agg(results):
        groups = {}
        for r in results:
            bs = str(r["batch_size"])
            groups.setdefault(bs, []).append(r)
        summary = {}
        for bs, runs in groups.items():
            summary[bs] = {
                "psnr_mean": np.mean([r["psnr"] for r in runs]),
                "psnr_std": np.std([r["psnr"] for r in runs]),
                "ssim_mean": np.mean([r["ssim"] for r in runs]),
                "time_mean": np.mean([r["train_time_s"] for r in runs]),
                "spec_high": np.mean([r["spectral_band_errors"][2] for r in runs]),
            }
        return summary

    siren_agg = agg(siren_results)
    coin_agg = agg(coin_results)

    order = ["256", "1024", "4096", "16384", "full"]
    order = [k for k in order if k in siren_agg and k in coin_agg]
    labels = ["Full" if k == "full" else k for k in order]
    x = np.arange(len(order))
    w = 0.35

    # --- PSNR comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    siren_psnr = [siren_agg[k]["psnr_mean"] for k in order]
    coin_psnr = [coin_agg[k]["psnr_mean"] for k in order]
    siren_std = [siren_agg[k]["psnr_std"] for k in order]
    coin_std = [coin_agg[k]["psnr_std"] for k in order]

    ax.bar(x - w/2, siren_psnr, w, yerr=siren_std, label="SIREN",
           color="#2196F3", edgecolor="black", linewidth=0.5, capsize=3)
    ax.bar(x + w/2, coin_psnr, w, yerr=coin_std, label="COIN",
           color="#FF5722", edgecolor="black", linewidth=0.5, capsize=3)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("SIREN vs COIN: PSNR by Batch Size (Kodak avg)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "siren_vs_coin_psnr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Spectral sensitivity comparison ---
    fig, ax = plt.subplots(figsize=(8, 5))
    siren_spec = [siren_agg[k]["spec_high"] for k in order]
    coin_spec = [coin_agg[k]["spec_high"] for k in order]

    ax.plot(labels, siren_spec, "o-", label="SIREN", linewidth=2, markersize=8)
    ax.plot(labels, coin_spec, "s--", label="COIN", linewidth=2, markersize=8)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("High-Frequency Spectral Error")
    ax.set_title("Architecture Sensitivity: High-Freq Error vs Batch Size")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(plots_dir / "siren_vs_coin_spectral.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Print LaTeX table ---
    print("\n--- SIREN vs COIN LaTeX Table ---")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Arch & Batch Size & PSNR (dB) & SSIM & Spec.\ High \\")
    print(r"\midrule")
    for arch, agg_data in [("SIREN", siren_agg), ("COIN", coin_agg)]:
        for bs in order:
            s = agg_data[bs]
            label = "Full" if bs == "full" else bs
            print(f"{arch} & {label} & "
                  f"${s['psnr_mean']:.2f} \\pm {s['psnr_std']:.2f}$ & "
                  f"${s['ssim_mean']:.3f}$ & "
                  f"${s['spec_high']:.3f}$ \\\\")
        if arch == "SIREN":
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


# ===================================================================
# 2. Iso-time analysis
# ===================================================================

def plot_iso_time(results: list[dict], out_dir: Path):
    if not results:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    budgets = sorted(set(r["time_budget_s"] for r in results))

    for budget in budgets:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for arch in ["siren", "coin"]:
            runs = [r for r in results
                    if r["arch"] == arch and r["time_budget_s"] == budget]
            if not runs:
                continue

            # Aggregate by batch size
            groups = {}
            for r in runs:
                bs = str(r["batch_size"])
                groups.setdefault(bs, []).append(r["psnr"])

            order = ["256", "1024", "4096", "16384", "full"]
            order = [k for k in order if k in groups]
            labels = ["Full" if k == "full" else k for k in order]
            means = [np.mean(groups[k]) for k in order]
            stds = [np.std(groups[k]) for k in order]

            ax_idx = 0 if arch == "siren" else 1
            ax = axes[ax_idx]
            ax.bar(labels, means, yerr=stds, color="#2196F3" if arch == "siren" else "#FF5722",
                   edgecolor="black", linewidth=0.5, capsize=3)
            ax.set_xlabel("Batch Size")
            ax.set_ylabel("PSNR (dB)")
            ax.set_title(f"{arch.upper()} — {budget}s budget")

        fig.suptitle(f"Iso-Time: PSNR with {budget}s Training Budget", fontsize=14)
        fig.tight_layout()
        fig.savefig(plots_dir / f"iso_time_{int(budget)}s.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Print summary
    print(f"\n--- Iso-Time Summary ---")
    for budget in budgets:
        print(f"\n  Budget: {budget}s")
        for arch in ["siren", "coin"]:
            runs = [r for r in results
                    if r["arch"] == arch and r["time_budget_s"] == budget]
            groups = {}
            for r in runs:
                groups.setdefault(str(r["batch_size"]), []).append(r)
            for bs in ["256", "1024", "4096", "16384", "full"]:
                if bs not in groups:
                    continue
                psnr_mean = np.mean([r["psnr"] for r in groups[bs]])
                steps_mean = np.mean([r["total_steps"] for r in groups[bs]])
                print(f"    {arch:5s} bs={bs:>5s}: PSNR={psnr_mean:.2f} dB, "
                      f"steps={steps_mean:.0f}")


# ===================================================================
# 3. Adaptive schedule analysis
# ===================================================================

def plot_adaptive(results: list[dict], out_dir: Path):
    if not results:
        return

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by strategy + arch
    groups = {}
    for r in results:
        key = (r["strategy"], r["arch"])
        groups.setdefault(key, []).append(r)

    strategies = sorted(set(r["strategy"] for r in results))
    archs = sorted(set(r["arch"] for r in results))

    # Bar chart: PSNR by strategy
    for arch in archs:
        strats = [s for s in strategies if (s, arch) in groups]
        means = [np.mean([r["psnr"] for r in groups[(s, arch)]]) for s in strats]
        stds = [np.std([r["psnr"] for r in groups[(s, arch)]]) for s in strats]
        times = [np.mean([r["train_time_s"] for r in groups[(s, arch)]]) for s in strats]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        colors = plt.cm.Set2(np.linspace(0, 1, len(strats)))
        axes[0].bar(strats, means, yerr=stds, color=colors,
                    edgecolor="black", linewidth=0.5, capsize=3)
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_title(f"{arch.upper()}: PSNR by Strategy")
        axes[0].tick_params(axis="x", rotation=30)

        axes[1].bar(strats, times, color=colors,
                    edgecolor="black", linewidth=0.5)
        axes[1].set_ylabel("Training Time (s)")
        axes[1].set_title(f"{arch.upper()}: Time by Strategy")
        axes[1].tick_params(axis="x", rotation=30)

        fig.tight_layout()
        fig.savefig(plots_dir / f"adaptive_{arch}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Print summary
    print(f"\n--- Adaptive Schedule Summary ---")
    for arch in archs:
        print(f"\n  {arch.upper()}:")
        for s in strategies:
            if (s, arch) not in groups:
                continue
            runs = groups[(s, arch)]
            psnr_mean = np.mean([r["psnr"] for r in runs])
            time_mean = np.mean([r["train_time_s"] for r in runs])
            sched = runs[0].get("schedule", "").replace("\u2192", "->")
            print(f"    {s:20s} [{sched:>35s}]: "
                  f"PSNR={psnr_mean:.2f} dB, Time={time_mean:.1f}s")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze extended experiments")
    parser.add_argument("--input", default="experiments/extended")
    parser.add_argument("--siren-results", default="experiments/results/results.json",
                        help="Original SIREN sweep results for comparison")
    args = parser.parse_args()

    inp = Path(args.input)
    siren_orig = load_json(Path(args.siren_results))
    coin_sweep = load_json(inp / "coin_sweep.json")
    iso_time = load_json(inp / "iso_time.json")
    adaptive = load_json(inp / "adaptive.json")

    if siren_orig or coin_sweep:
        plot_arch_comparison(siren_orig, coin_sweep, inp)

    if iso_time:
        plot_iso_time(iso_time, inp)

    if adaptive:
        plot_adaptive(adaptive, inp)

    print("\n  Analysis complete.")


if __name__ == "__main__":
    main()
