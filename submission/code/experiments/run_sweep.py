"""
Batch-size sweep experiment runner.

Trains a SIREN for each batch size in the sweep configuration,
evaluates PSNR / SSIM, runs spectral analysis, and saves results + plots.

Usage:
    python -m experiments.run_sweep                        # defaults
    python -m experiments.run_sweep --config configs/batch_sweep.yaml
    python -m experiments.run_sweep --images data/images/kodim01.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

import torch

from models.siren import SIREN
from training.sampling import get_sampler
from training.trainer import Trainer
from evaluation.metrics import evaluate
from evaluation.spectral import spectral_error_by_band, plot_spectral_comparison


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_images(image_dir: str | Path, explicit: list[str] | None = None) -> list[Path]:
    """Return a sorted list of image paths."""
    if explicit:
        return [Path(p) for p in explicit]
    d = Path(image_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if not imgs:
        raise FileNotFoundError(f"No images found in {d}")
    return imgs


def _build_model(cfg: dict) -> SIREN:
    return SIREN(
        in_features=2,
        hidden_features=cfg["model"]["hidden_features"],
        hidden_layers=cfg["model"]["hidden_layers"],
        out_features=3,
        omega_0=cfg["model"]["omega_0"],
    )


def _make_sampler(batch_size, image: np.ndarray):
    """Create a sampler for the given batch size."""
    if batch_size == "full":
        return get_sampler("full_image")
    return get_sampler("random_pixels", n=int(batch_size))


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(
    image_path: Path,
    batch_size,
    cfg: dict,
    out_dir: Path,
) -> dict:
    """Train one model, evaluate, return metrics dict."""
    tag = f"{image_path.stem}_bs{batch_size}"
    print(f"\n{'='*60}")
    print(f"  {tag}")
    print(f"{'='*60}")

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0

    model = _build_model(cfg)
    sampler = _make_sampler(batch_size, img_np)

    trainer = Trainer(
        model=model,
        sampler=sampler,
        image=img,
        n_steps=cfg["training"]["n_steps"],
        lr=cfg["training"]["lr"],
        device=cfg["training"]["device"],
        log_every=cfg["training"]["log_every"],
    )
    train_info = trainer.train()

    # Save weights
    weights_dir = out_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    trainer.save(weights_dir / f"{tag}.pt")

    # Evaluate
    eval_result = evaluate(model, img_np, device=cfg["training"]["device"])

    # Spectral analysis
    spectral = spectral_error_by_band(img_np, eval_result["reconstructed"])

    # Save spectral plot
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_spectral_comparison(
        img_np, eval_result["reconstructed"],
        save_path=plots_dir / f"spectral_{tag}.png",
    )
    plt.close(fig)

    return {
        "image": image_path.name,
        "batch_size": batch_size,
        "psnr": eval_result["psnr"],
        "ssim": eval_result["ssim"],
        "spectral_band_errors": spectral["band_errors"],
        "spectral_band_labels": spectral["band_labels"],
        "train_time_s": train_info["train_time_s"],
        "n_parameters": model.n_parameters(),
        "final_loss": train_info["loss_history"][-1],
        "loss_history": train_info["loss_history"],
        "psnr_history": train_info["psnr_history"],
    }


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def plot_sweep_summary(results: list[dict], out_dir: Path):
    """Generate summary plots comparing batch sizes."""
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Group by image
    images = sorted(set(r["image"] for r in results))

    for img_name in images:
        runs = [r for r in results if r["image"] == img_name]
        runs.sort(key=lambda r: r["batch_size"] if isinstance(r["batch_size"], int) else 10**7)

        bs_labels = [str(r["batch_size"]) for r in runs]
        psnrs = [r["psnr"] for r in runs]
        ssims = [r["ssim"] for r in runs]
        times = [r["train_time_s"] for r in runs]

        stem = Path(img_name).stem

        # PSNR vs batch size
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].bar(bs_labels, psnrs, color="#2196F3", edgecolor="black", linewidth=0.5)
        axes[0].set_xlabel("Batch size")
        axes[0].set_ylabel("PSNR (dB)")
        axes[0].set_title(f"{stem}: PSNR vs Batch Size")

        axes[1].bar(bs_labels, ssims, color="#4CAF50", edgecolor="black", linewidth=0.5)
        axes[1].set_xlabel("Batch size")
        axes[1].set_ylabel("SSIM")
        axes[1].set_title(f"{stem}: SSIM vs Batch Size")

        axes[2].bar(bs_labels, times, color="#FF9800", edgecolor="black", linewidth=0.5)
        axes[2].set_xlabel("Batch size")
        axes[2].set_ylabel("Time (s)")
        axes[2].set_title(f"{stem}: Training Time vs Batch Size")

        fig.tight_layout()
        fig.savefig(plots_dir / f"summary_{stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Loss curves overlay
        fig, ax = plt.subplots(figsize=(8, 5))
        for r in runs:
            ax.plot(r["loss_history"], label=f"bs={r['batch_size']}", alpha=0.8)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"{stem}: Training Loss Curves")
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(plots_dir / f"loss_curves_{stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch-size sweep for INR")
    parser.add_argument("--config", default="configs/batch_sweep.yaml")
    parser.add_argument("--images", nargs="*", default=None,
                        help="Explicit image paths (overrides config image_dir)")
    parser.add_argument("--output", default=None,
                        help="Output directory (overrides config)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    out_dir = Path(args.output or cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = _collect_images(
        cfg["data"]["image_dir"],
        explicit=args.images,
    )

    batch_sizes = cfg["sweep"]["batch_sizes"]
    all_results = []       # full results (with loss history) for plotting
    json_results = []       # lightweight results for JSON export

    for img_path in image_paths:
        for bs in batch_sizes:
            result = run_single(img_path, bs, cfg, out_dir)
            all_results.append(result)
            # Strip large arrays before saving to JSON
            json_results.append({
                k: v for k, v in result.items()
                if k not in ("loss_history", "psnr_history")
            })

    # Save results
    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate summary plots (using full results with loss history)
    print("Generating summary plots...")
    plot_sweep_summary(all_results, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
