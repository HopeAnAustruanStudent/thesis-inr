"""
Model-size ablation + LPIPS metric.

Tests whether batch size effect persists across different model capacities.

Usage:
    python -m experiments.run_ablation --experiment model_size
    python -m experiments.run_ablation --experiment all
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from models.siren import SIREN
from training.sampling import get_sampler
from training.trainer import Trainer
from evaluation.metrics import evaluate
from evaluation.spectral import spectral_error_by_band


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [256, 1024, 4096, 16384, "full"]
N_STEPS = 2000
LR = 1e-4


def _collect_images(limit=6):
    d = Path("data/images")
    return sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")[:limit]


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


def run_model_size_ablation(out_dir: Path, image_limit: int = 6):
    """Sweep batch sizes at different model widths: 64, 128, 256, 512."""
    print("\n" + "=" * 70)
    print("  Model-Size Ablation")
    print("=" * 70)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    widths = [64, 128, 256, 512]
    images = _collect_images(limit=image_limit)
    all_results = []

    for width in widths:
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img, dtype=np.float32) / 255.0

            for bs in BATCH_SIZES:
                tag = f"w{width}_{img_path.stem}_bs{bs}"
                print(f"  >>> {tag}")

                model = SIREN(
                    in_features=2, hidden_features=width,
                    hidden_layers=5, out_features=3, omega_0=30.0,
                )
                sampler = get_sampler("full_image") if bs == "full" else get_sampler("random_pixels", n=int(bs))

                trainer = Trainer(
                    model=model, sampler=sampler, image=img,
                    n_steps=N_STEPS, lr=LR, device=DEVICE, log_every=100,
                )
                train_info = trainer.train()
                eval_result = evaluate(model, img_np, device=DEVICE)
                spectral = spectral_error_by_band(img_np, eval_result["reconstructed"])

                all_results.append({
                    "width": width,
                    "n_params": model.n_parameters(),
                    "image": img_path.name,
                    "batch_size": bs,
                    "psnr": eval_result["psnr"],
                    "ssim": eval_result["ssim"],
                    "spectral_high": spectral["band_errors"][2],
                    "train_time_s": train_info["train_time_s"],
                })

                del model, trainer
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
                gc.collect()

    _save_json(all_results, out_dir / "model_size_ablation.json")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for width in widths:
        runs = [r for r in all_results if r["width"] == width]
        groups = {}
        for r in runs:
            groups.setdefault(str(r["batch_size"]), []).append(r["psnr"])

        order = [k for k in ["256", "1024", "4096", "16384", "full"] if k in groups]
        labels = ["Full" if k == "full" else k for k in order]
        means = [np.mean(groups[k]) for k in order]
        stds = [np.std(groups[k]) for k in order]
        n_params = runs[0]["n_params"]

        axes[0].errorbar(labels, means, yerr=stds, marker="o", linewidth=2,
                        label=f"width={width} ({n_params//1000}K params)", capsize=3)

    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title("PSNR vs Batch Size by Model Width")
    axes[0].legend(fontsize=9)

    # Sensitivity (full - bs256) vs width
    sensitivities = []
    for width in widths:
        runs = [r for r in all_results if r["width"] == width]
        groups = {}
        for r in runs:
            groups.setdefault(str(r["batch_size"]), []).append(r["psnr"])
        if "full" in groups and "256" in groups:
            sens = np.mean(groups["full"]) - np.mean(groups["256"])
            sensitivities.append(sens)
        else:
            sensitivities.append(0)

    axes[1].bar([str(w) for w in widths], sensitivities,
               color=["#BBDEFB", "#64B5F6", "#2196F3", "#0D47A1"],
               edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Model Width")
    axes[1].set_ylabel("PSNR(full) - PSNR(bs=256) (dB)")
    axes[1].set_title("Batch Size Sensitivity vs Model Capacity")

    fig.tight_layout()
    fig.savefig(plots_dir / "model_size_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Print summary
    print("\n--- Model-Size Ablation Summary ---")
    for width in widths:
        runs = [r for r in all_results if r["width"] == width]
        n_params = runs[0]["n_params"]
        groups = {}
        for r in runs:
            groups.setdefault(str(r["batch_size"]), []).append(r["psnr"])
        print(f"\n  width={width} ({n_params} params):")
        for bs in ["256", "1024", "4096", "16384", "full"]:
            if bs in groups:
                print(f"    bs={bs:>5s}: PSNR={np.mean(groups[bs]):.2f} dB")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="experiments/ablation")
    parser.add_argument("--image-limit", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.output)
    run_model_size_ablation(out_dir, args.image_limit)
    print("\n  ALL DONE")


if __name__ == "__main__":
    main()
