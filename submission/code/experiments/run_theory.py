"""
Theoretical experiments: gradient variance, LR scaling, per-image difficulty.

These experiments provide the WHY behind the batch size effects.

Usage:
    python -m experiments.run_theory --experiment grad_variance
    python -m experiments.run_theory --experiment lr_scaling
    python -m experiments.run_theory --experiment difficulty
    python -m experiments.run_theory --experiment all
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from models.siren import SIREN
from models.coin import COIN
from training.sampling import get_sampler, full_image
from training.trainer import Trainer
from evaluation.metrics import evaluate, psnr_from_mse
from evaluation.spectral import spectral_error_by_band


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [256, 1024, 4096, 16384, "full"]
N_STEPS = 2000
LR = 1e-4
LOG_EVERY = 50


def _load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    return img, img_np


def _collect_images(limit: int | None = None) -> list[Path]:
    d = Path("data/images")
    exts = {".png", ".jpg", ".jpeg"}
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if limit:
        imgs = imgs[:limit]
    return imgs


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def _build_siren():
    return SIREN(in_features=2, hidden_features=256, hidden_layers=5,
                 out_features=3, omega_0=30.0)


def _make_sampler(batch_size):
    if batch_size == "full":
        return get_sampler("full_image")
    return get_sampler("random_pixels", n=int(batch_size))


# ===================================================================
# Experiment 1: Gradient Variance Measurement
# ===================================================================

def measure_gradient_variance(
    model: nn.Module,
    image,
    batch_size,
    n_measurements: int = 50,
    device: str = "cpu",
) -> dict:
    """
    At the current model state, measure gradient variance across
    multiple mini-batch samples.

    For each measurement:
      1. Sample a mini-batch
      2. Compute gradient
      3. Record gradient norm

    Then compute mean and variance of gradient norms.
    """
    model.eval()  # don't update BN etc
    criterion = nn.MSELoss()
    sampler = _make_sampler(batch_size)

    grad_norms = []

    for _ in range(n_measurements):
        model.zero_grad()
        coords, colors = sampler(image)
        coords = coords.to(device)
        colors = colors.to(device)

        pred = model(coords)
        loss = criterion(pred, colors)
        loss.backward()

        # Compute total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)
        grad_norms.append(total_norm)

    return {
        "grad_norm_mean": float(np.mean(grad_norms)),
        "grad_norm_std": float(np.std(grad_norms)),
        "grad_norm_cv": float(np.std(grad_norms) / (np.mean(grad_norms) + 1e-8)),
        "n_measurements": n_measurements,
    }


def run_grad_variance(out_dir: Path, image_limit: int | None = None):
    """
    Measure gradient variance at different training stages for each batch size.
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Gradient Variance Analysis")
    print("=" * 70)

    images = _collect_images(limit=image_limit)
    all_results = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Measure at steps: 0 (init), 500, 1000, 1500, 2000
    measure_at_steps = [0, 500, 1000, 1500, 2000]

    for img_path in images:
        img, img_np = _load_image(img_path)
        print(f"\n  Image: {img_path.stem}")

        for arch_name, build_fn in [("siren", _build_siren)]:
            for bs in BATCH_SIZES:
                tag = f"{arch_name}_{img_path.stem}_bs{bs}"
                print(f"    >>> {tag}")

                model = build_fn().to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR)
                criterion = nn.MSELoss()
                sampler = _make_sampler(bs)

                step = 0
                for target_step in measure_at_steps:
                    # Train to target step
                    model.train()
                    while step < target_step:
                        coords, colors = sampler(img)
                        coords = coords.to(DEVICE)
                        colors = colors.to(DEVICE)
                        optimizer.zero_grad()
                        pred = model(coords)
                        loss = criterion(pred, colors)
                        loss.backward()
                        optimizer.step()
                        step += 1

                    # Measure gradient variance
                    gv = measure_gradient_variance(
                        model, img, batch_size=bs,
                        n_measurements=30, device=DEVICE,
                    )

                    all_results.append({
                        "arch": arch_name,
                        "image": img_path.name,
                        "batch_size": bs,
                        "step": target_step,
                        "grad_norm_mean": gv["grad_norm_mean"],
                        "grad_norm_std": gv["grad_norm_std"],
                        "grad_norm_cv": gv["grad_norm_cv"],
                        "loss": loss.item() if step > 0 else None,
                    })

                del model, optimizer
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
                gc.collect()

    _save_json(all_results, out_dir / "grad_variance.json")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Aggregate: CV vs batch size at step 1000
    for step_to_plot in [1000]:
        groups = {}
        for r in all_results:
            if r["step"] != step_to_plot:
                continue
            bs = str(r["batch_size"])
            groups.setdefault(bs, []).append(r)

        order = ["256", "1024", "4096", "16384", "full"]
        order = [k for k in order if k in groups]
        labels = ["Full" if k == "full" else k for k in order]
        cv_means = [np.mean([r["grad_norm_cv"] for r in groups[k]]) for k in order]
        cv_stds = [np.std([r["grad_norm_cv"] for r in groups[k]]) for k in order]
        norm_means = [np.mean([r["grad_norm_std"] for r in groups[k]]) for k in order]

        axes[0].bar(labels, cv_means, yerr=cv_stds,
                    color="#E91E63", edgecolor="black", linewidth=0.5, capsize=3)
        axes[0].set_xlabel("Batch Size")
        axes[0].set_ylabel("Gradient CV (std/mean)")
        axes[0].set_title(f"Gradient Variance (Coefficient of Variation)\nat step {step_to_plot}")

        axes[1].bar(labels, norm_means,
                    color="#9C27B0", edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Batch Size")
        axes[1].set_ylabel("Gradient Norm Std")
        axes[1].set_title(f"Gradient Norm Standard Deviation\nat step {step_to_plot}")

    fig.tight_layout()
    fig.savefig(plots_dir / "grad_variance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Plot: CV over training for each batch size
    fig, ax = plt.subplots(figsize=(8, 5))
    for bs in ["256", "1024", "4096", "16384", "full"]:
        runs = [r for r in all_results if str(r["batch_size"]) == bs]
        steps_agg = {}
        for r in runs:
            steps_agg.setdefault(r["step"], []).append(r["grad_norm_cv"])
        steps = sorted(steps_agg.keys())
        cvs = [np.mean(steps_agg[s]) for s in steps]
        label = "Full" if bs == "full" else f"bs={bs}"
        ax.plot(steps, cvs, "o-", label=label, linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gradient CV")
    ax.set_title("Gradient Variance Over Training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "grad_variance_over_time.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved to {plots_dir}")
    return all_results


# ===================================================================
# Experiment 2: Learning Rate Scaling
# ===================================================================

def run_lr_scaling(out_dir: Path, image_limit: int | None = None):
    """
    Test if linear/sqrt LR scaling compensates for small batch size.

    For each batch size, try:
      - base_lr (1e-4)
      - linear scaling: lr = base_lr * (bs / 16384)  [scale down for small]
      - sqrt scaling: lr = base_lr * sqrt(bs / 16384)
      - inverse: lr = base_lr * (16384 / bs)  [scale up for small]
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Learning Rate Scaling")
    print("=" * 70)

    images = _collect_images(limit=image_limit)
    all_results = []
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    base_lr = 1e-4
    ref_bs = 16384  # reference batch size

    lr_rules = {
        "fixed": lambda bs: base_lr,
        "linear": lambda bs: base_lr * (bs / ref_bs),
        "sqrt": lambda bs: base_lr * math.sqrt(bs / ref_bs),
        "inverse_sqrt": lambda bs: base_lr * math.sqrt(ref_bs / bs),
    }

    test_batch_sizes = [256, 1024, 4096, 16384]

    for img_path in images:
        img, img_np = _load_image(img_path)
        print(f"\n  Image: {img_path.stem}")

        for bs in test_batch_sizes:
            for rule_name, lr_fn in lr_rules.items():
                lr = lr_fn(bs)
                tag = f"lr_{rule_name}_{img_path.stem}_bs{bs}"
                print(f"    >>> {tag} (lr={lr:.2e})")

                model = _build_siren()
                sampler = _make_sampler(bs)
                trainer = Trainer(
                    model=model, sampler=sampler, image=img,
                    n_steps=N_STEPS, lr=lr, device=DEVICE, log_every=LOG_EVERY,
                )
                train_info = trainer.train()
                eval_result = evaluate(model, img_np, device=DEVICE)

                all_results.append({
                    "image": img_path.name,
                    "batch_size": bs,
                    "lr_rule": rule_name,
                    "lr": lr,
                    "psnr": eval_result["psnr"],
                    "ssim": eval_result["ssim"],
                    "train_time_s": train_info["train_time_s"],
                    "final_loss": train_info["loss_history"][-1],
                })

                del model, trainer
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
                gc.collect()

    _save_json(all_results, out_dir / "lr_scaling.json")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Aggregate by batch_size and lr_rule
    for ax_idx, metric in enumerate(["psnr", "ssim"]):
        groups = {}
        for r in all_results:
            key = (str(r["batch_size"]), r["lr_rule"])
            groups.setdefault(key, []).append(r[metric])

        x = np.arange(len(test_batch_sizes))
        width = 0.2
        rules = ["fixed", "linear", "sqrt", "inverse_sqrt"]
        colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

        for i, rule in enumerate(rules):
            means = []
            stds = []
            for bs in test_batch_sizes:
                vals = groups.get((str(bs), rule), [0])
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            axes[ax_idx].bar(x + i * width, means, width, yerr=stds,
                            label=rule, color=colors[i],
                            edgecolor="black", linewidth=0.5, capsize=2)

        axes[ax_idx].set_xlabel("Batch Size")
        axes[ax_idx].set_ylabel(metric.upper())
        axes[ax_idx].set_title(f"{metric.upper()} by Batch Size and LR Rule")
        axes[ax_idx].set_xticks(x + 1.5 * width)
        axes[ax_idx].set_xticklabels([str(bs) for bs in test_batch_sizes])
        axes[ax_idx].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "lr_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print("\n--- LR Scaling Summary ---")
    for bs in test_batch_sizes:
        print(f"\n  bs={bs}:")
        for rule in ["fixed", "linear", "sqrt", "inverse_sqrt"]:
            runs = [r for r in all_results
                    if r["batch_size"] == bs and r["lr_rule"] == rule]
            if not runs:
                continue
            psnr_mean = np.mean([r["psnr"] for r in runs])
            lr = runs[0]["lr"]
            print(f"    {rule:15s} (lr={lr:.2e}): PSNR={psnr_mean:.2f} dB")

    print(f"\n  Plots saved to {plots_dir}")
    return all_results


# ===================================================================
# Experiment 3: Per-Image Difficulty Analysis
# ===================================================================

def compute_image_complexity(img_np: np.ndarray) -> dict:
    """Compute multiple complexity measures for an image."""
    gray = 0.2989 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]

    # 1. Edge density (Sobel-like gradient magnitude)
    gy = np.diff(gray, axis=0)
    gx = np.diff(gray, axis=1)
    # Align dimensions
    gy = gy[:, :-1]
    gx = gx[:-1, :]
    edge_mag = np.sqrt(gx**2 + gy**2)
    edge_density = float(np.mean(edge_mag))

    # 2. High-frequency energy ratio (FFT)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    H, W = gray.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    max_r = min(cy, cx)
    high_freq_mask = r > (max_r * 0.5)
    total_energy = float(np.sum(mag ** 2))
    high_energy = float(np.sum(mag[high_freq_mask] ** 2))
    hf_ratio = high_energy / (total_energy + 1e-8)

    # 3. Spatial variance
    spatial_var = float(np.var(gray))

    return {
        "edge_density": edge_density,
        "hf_energy_ratio": hf_ratio,
        "spatial_variance": spatial_var,
    }


def run_difficulty_analysis(out_dir: Path):
    """Analyze how image complexity interacts with batch size sensitivity."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT: Per-Image Difficulty Analysis")
    print("=" * 70)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load existing SIREN sweep results
    siren_path = Path("experiments/results/results.json")
    if not siren_path.exists():
        print("  ERROR: Need SIREN sweep results first!")
        return []

    with open(siren_path) as f:
        siren_results = json.load(f)

    images = _collect_images()
    all_results = []

    for img_path in images:
        _, img_np = _load_image(img_path)
        complexity = compute_image_complexity(img_np)

        # Get PSNR for each batch size from sweep results
        img_runs = [r for r in siren_results if r["image"] == img_path.name]
        psnr_by_bs = {}
        for r in img_runs:
            psnr_by_bs[str(r["batch_size"])] = r["psnr"]

        # Batch size sensitivity = PSNR(full) - PSNR(bs=256)
        psnr_full = psnr_by_bs.get("full", 0)
        psnr_256 = psnr_by_bs.get("256", 0)
        sensitivity = psnr_full - psnr_256

        all_results.append({
            "image": img_path.name,
            "edge_density": complexity["edge_density"],
            "hf_energy_ratio": complexity["hf_energy_ratio"],
            "spatial_variance": complexity["spatial_variance"],
            "psnr_bs256": psnr_256,
            "psnr_full": psnr_full,
            "bs_sensitivity": sensitivity,
            "psnr_by_bs": psnr_by_bs,
        })

        print(f"  {img_path.stem}: edge={complexity['edge_density']:.4f}  "
              f"hf_ratio={complexity['hf_energy_ratio']:.4f}  "
              f"sensitivity={sensitivity:.2f} dB")

    _save_json(all_results, out_dir / "difficulty.json")

    # --- Plot: sensitivity vs complexity ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    sensitivities = [r["bs_sensitivity"] for r in all_results]
    edge_densities = [r["edge_density"] for r in all_results]
    hf_ratios = [r["hf_energy_ratio"] for r in all_results]
    spatial_vars = [r["spatial_variance"] for r in all_results]

    for ax, metric, label in [
        (axes[0], edge_densities, "Edge Density"),
        (axes[1], hf_ratios, "High-Freq Energy Ratio"),
        (axes[2], spatial_vars, "Spatial Variance"),
    ]:
        ax.scatter(metric, sensitivities, s=60, alpha=0.7,
                   edgecolors="black", linewidth=0.5)

        # Add correlation line
        z = np.polyfit(metric, sensitivities, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(metric), max(metric), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=1.5)

        # Correlation coefficient
        corr = np.corrcoef(metric, sensitivities)[0, 1]
        ax.set_xlabel(label)
        ax.set_ylabel("BS Sensitivity (PSNR_full - PSNR_256)")
        ax.set_title(f"{label}\nr = {corr:.3f}")

        # Label outlier images
        for r in all_results:
            idx = all_results.index(r)
            if r["bs_sensitivity"] > np.mean(sensitivities) + np.std(sensitivities):
                ax.annotate(r["image"][:7], (metric[idx], sensitivities[idx]),
                           fontsize=7, alpha=0.7)

    fig.suptitle("Batch Size Sensitivity vs Image Complexity", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(plots_dir / "difficulty_analysis.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot: PSNR curves by difficulty tier ---
    sorted_by_edge = sorted(all_results, key=lambda r: r["edge_density"])
    easy = sorted_by_edge[:8]   # smoothest 8 images
    medium = sorted_by_edge[8:16]
    hard = sorted_by_edge[16:]  # most textured 8 images

    fig, ax = plt.subplots(figsize=(8, 5))
    bs_order = ["256", "1024", "4096", "16384", "full"]
    labels = ["Full" if k == "full" else k for k in bs_order]

    for tier, tier_name, color in [
        (easy, "Easy (smooth)", "#4CAF50"),
        (medium, "Medium", "#FF9800"),
        (hard, "Hard (textured)", "#F44336"),
    ]:
        means = []
        for bs in bs_order:
            psnrs = [r["psnr_by_bs"].get(bs, 0) for r in tier]
            means.append(np.mean(psnrs))
        ax.plot(labels, means, "o-", label=tier_name, color=color, linewidth=2, markersize=8)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR vs Batch Size by Image Difficulty Tier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "difficulty_tiers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Plots saved to {plots_dir}")
    return all_results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Theory experiments")
    parser.add_argument("--experiment", default="all",
                        choices=["grad_variance", "lr_scaling", "difficulty", "all"])
    parser.add_argument("--output", default="experiments/theory")
    parser.add_argument("--image-limit", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("grad_variance", "all"):
        run_grad_variance(out_dir, image_limit=args.image_limit or 6)

    if args.experiment in ("lr_scaling", "all"):
        run_lr_scaling(out_dir, image_limit=args.image_limit or 6)

    if args.experiment in ("difficulty", "all"):
        run_difficulty_analysis(out_dir)

    print("\n" + "=" * 70)
    print("  ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
