"""
Extended experiments for the thesis — PhD-level rigor.

Experiments:
  1. COIN batch-size sweep (same as SIREN, for architecture comparison)
  2. Iso-time comparison (fixed wall-clock budget: 30s, 60s, 120s)
  3. Adaptive batch-size schedule
  4. Visual reconstruction grid
  5. VRAM profiling

Usage:
    python -m experiments.run_extended --experiment coin_sweep
    python -m experiments.run_extended --experiment iso_time
    python -m experiments.run_extended --experiment adaptive
    python -m experiments.run_extended --experiment visuals
    python -m experiments.run_extended --experiment all
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch

from models.siren import SIREN
from models.coin import COIN
from training.sampling import get_sampler
from training.trainer import Trainer, AdaptiveTrainer
from evaluation.metrics import evaluate, reconstruct
from evaluation.spectral import spectral_error_by_band


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [256, 1024, 4096, 16384, "full"]
N_STEPS = 2000
LR = 1e-4
LOG_EVERY = 50  # finer logging for psnr-vs-time curves


def _collect_images(image_dir: str = "data/images", limit: int | None = None) -> list[Path]:
    d = Path(image_dir)
    exts = {".png", ".jpg", ".jpeg"}
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts)
    if limit:
        imgs = imgs[:limit]
    return imgs


def _load_image(path: Path) -> tuple[Image.Image, np.ndarray]:
    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    return img, img_np


def _build_siren() -> SIREN:
    return SIREN(in_features=2, hidden_features=256, hidden_layers=5,
                 out_features=3, omega_0=30.0)


def _build_coin() -> COIN:
    return COIN(in_features=2, hidden_features=256, hidden_layers=5,
                out_features=3, n_freqs=10)


def _make_sampler(batch_size):
    if batch_size == "full":
        return get_sampler("full_image")
    return get_sampler("random_pixels", n=int(batch_size))


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ===================================================================
# Experiment 1: COIN batch-size sweep
# ===================================================================

def run_coin_sweep(out_dir: Path, image_limit: int | None = None):
    """Same sweep as SIREN but with COIN architecture."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 1: COIN Batch-Size Sweep")
    print("=" * 70)

    images = _collect_images(limit=image_limit)
    all_results = []

    for img_path in images:
        img, img_np = _load_image(img_path)
        for bs in BATCH_SIZES:
            tag = f"coin_{img_path.stem}_bs{bs}"
            print(f"\n  >>> {tag}")

            model = _build_coin()
            sampler = _make_sampler(bs)
            trainer = Trainer(
                model=model, sampler=sampler, image=img,
                n_steps=N_STEPS, lr=LR, device=DEVICE, log_every=LOG_EVERY,
            )
            train_info = trainer.train()
            eval_result = evaluate(model, img_np, device=DEVICE)
            spectral = spectral_error_by_band(img_np, eval_result["reconstructed"])

            all_results.append({
                "arch": "coin",
                "image": img_path.name,
                "batch_size": bs,
                "psnr": eval_result["psnr"],
                "ssim": eval_result["ssim"],
                "spectral_band_errors": spectral["band_errors"],
                "spectral_band_labels": spectral["band_labels"],
                "train_time_s": train_info["train_time_s"],
                "total_steps": train_info["total_steps"],
                "n_parameters": model.n_parameters(),
                "final_loss": train_info["loss_history"][-1],
                "psnr_vs_time": train_info["psnr_vs_time"],
            })

            del model, trainer
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
            gc.collect()

    _save_json(all_results, out_dir / "coin_sweep.json")
    return all_results


# ===================================================================
# Experiment 2: Iso-time comparison
# ===================================================================

def run_iso_time(out_dir: Path, image_limit: int | None = None):
    """Train SIREN and COIN with fixed time budgets to compare fairly."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 2: Iso-Time Comparison")
    print("=" * 70)

    time_budgets = [30.0, 60.0, 120.0]
    images = _collect_images(limit=image_limit)
    all_results = []

    for img_path in images:
        img, img_np = _load_image(img_path)

        for arch_name, build_fn in [("siren", _build_siren), ("coin", _build_coin)]:
            for bs in BATCH_SIZES:
                for budget in time_budgets:
                    tag = f"isotime_{arch_name}_{img_path.stem}_bs{bs}_{budget}s"
                    print(f"\n  >>> {tag}")

                    model = build_fn()
                    sampler = _make_sampler(bs)
                    trainer = Trainer(
                        model=model, sampler=sampler, image=img,
                        n_steps=999999, lr=LR, device=DEVICE,
                        log_every=LOG_EVERY, time_budget_s=budget,
                    )
                    train_info = trainer.train()
                    eval_result = evaluate(model, img_np, device=DEVICE)
                    spectral = spectral_error_by_band(img_np, eval_result["reconstructed"])

                    all_results.append({
                        "arch": arch_name,
                        "image": img_path.name,
                        "batch_size": bs,
                        "time_budget_s": budget,
                        "psnr": eval_result["psnr"],
                        "ssim": eval_result["ssim"],
                        "spectral_band_errors": spectral["band_errors"],
                        "train_time_s": train_info["train_time_s"],
                        "total_steps": train_info["total_steps"],
                        "n_parameters": model.n_parameters(),
                        "psnr_vs_time": train_info["psnr_vs_time"],
                    })

                    del model, trainer
                    torch.cuda.empty_cache() if DEVICE == "cuda" else None
                    gc.collect()

    _save_json(all_results, out_dir / "iso_time.json")
    return all_results


# ===================================================================
# Experiment 3: Adaptive batch-size schedule
# ===================================================================

def run_adaptive(out_dir: Path, image_limit: int | None = None):
    """
    Test adaptive batch-size schedules against fixed baselines.

    Schedules:
      A) 30% bs=1024 → 70% full           (coarse-to-fine)
      B) 50% bs=4096 → 50% full           (balanced)
      C) 20% bs=256  → 30% bs=4096 → 50% full  (progressive)
    Baselines: fixed full, fixed bs=4096, fixed bs=16384
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENT 3: Adaptive Batch-Size Schedule")
    print("=" * 70)

    schedules = {
        "adaptive_A": [(0.3, 1024), (0.7, "full")],
        "adaptive_B": [(0.5, 4096), (0.5, "full")],
        "adaptive_C": [(0.2, 256), (0.3, 4096), (0.5, "full")],
    }
    baselines = {
        "fixed_full": "full",
        "fixed_16384": 16384,
        "fixed_4096": 4096,
    }

    images = _collect_images(limit=image_limit)
    all_results = []

    for img_path in images:
        img, img_np = _load_image(img_path)

        # Adaptive runs
        for sched_name, schedule in schedules.items():
            for arch_name, build_fn in [("siren", _build_siren), ("coin", _build_coin)]:
                tag = f"{sched_name}_{arch_name}_{img_path.stem}"
                print(f"\n  >>> {tag}")

                model = build_fn()
                trainer = AdaptiveTrainer(
                    model=model, image=img, schedule=schedule,
                    n_steps=N_STEPS, lr=LR, device=DEVICE, log_every=LOG_EVERY,
                )
                train_info = trainer.train()
                eval_result = evaluate(model, img_np, device=DEVICE)
                spectral = spectral_error_by_band(img_np, eval_result["reconstructed"])

                sched_str = " → ".join(f"{int(f*100)}%@bs{b}" for f, b in schedule)
                all_results.append({
                    "strategy": sched_name,
                    "schedule": sched_str,
                    "arch": arch_name,
                    "image": img_path.name,
                    "psnr": eval_result["psnr"],
                    "ssim": eval_result["ssim"],
                    "spectral_band_errors": spectral["band_errors"],
                    "train_time_s": train_info["train_time_s"],
                    "total_steps": train_info["total_steps"],
                    "n_parameters": model.n_parameters(),
                    "psnr_vs_time": train_info["psnr_vs_time"],
                })

                del model, trainer
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
                gc.collect()

        # Fixed baselines (SIREN only — SIREN fixed results already exist,
        # but we re-run with finer logging for psnr_vs_time)
        for base_name, bs in baselines.items():
            tag = f"{base_name}_siren_{img_path.stem}"
            print(f"\n  >>> {tag}")

            model = _build_siren()
            sampler = _make_sampler(bs)
            trainer = Trainer(
                model=model, sampler=sampler, image=img,
                n_steps=N_STEPS, lr=LR, device=DEVICE, log_every=LOG_EVERY,
            )
            train_info = trainer.train()
            eval_result = evaluate(model, img_np, device=DEVICE)
            spectral = spectral_error_by_band(img_np, eval_result["reconstructed"])

            all_results.append({
                "strategy": base_name,
                "schedule": f"100%@bs{bs}",
                "arch": "siren",
                "image": img_path.name,
                "psnr": eval_result["psnr"],
                "ssim": eval_result["ssim"],
                "spectral_band_errors": spectral["band_errors"],
                "train_time_s": train_info["train_time_s"],
                "total_steps": train_info["total_steps"],
                "n_parameters": model.n_parameters(),
                "psnr_vs_time": train_info["psnr_vs_time"],
            })

            del model, trainer
            torch.cuda.empty_cache() if DEVICE == "cuda" else None
            gc.collect()

    _save_json(all_results, out_dir / "adaptive.json")
    return all_results


# ===================================================================
# Experiment 4: Visual reconstruction grid
# ===================================================================

def run_visuals(out_dir: Path):
    """Generate side-by-side reconstruction images for a representative image."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 4: Visual Reconstruction Grid")
    print("=" * 70)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Use kodim01 and kodim23 (parrot — high detail)
    test_images = ["data/images/kodim01.png", "data/images/kodim23.png"]

    for img_path_str in test_images:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"  Skipping {img_path} (not found)")
            continue

        img, img_np = _load_image(img_path)
        H, W, _ = img_np.shape

        n_configs = len(BATCH_SIZES) + 1  # +1 for original
        fig, axes = plt.subplots(2, n_configs, figsize=(4 * n_configs, 8))

        # Row 0: SIREN, Row 1: COIN
        for row, (arch_name, build_fn) in enumerate([
            ("SIREN", _build_siren), ("COIN", _build_coin)
        ]):
            # Original image in first column
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title(f"Original\n{arch_name}", fontsize=10)
            axes[row, 0].axis("off")

            for j, bs in enumerate(BATCH_SIZES, start=1):
                print(f"  >>> {arch_name} {img_path.stem} bs={bs}")
                model = build_fn()
                sampler = _make_sampler(bs)
                trainer = Trainer(
                    model=model, sampler=sampler, image=img,
                    n_steps=N_STEPS, lr=LR, device=DEVICE, log_every=500,
                )
                trainer.train()
                eval_result = evaluate(model, img_np, device=DEVICE)
                recon = eval_result["reconstructed"]

                axes[row, j].imshow(recon)
                axes[row, j].set_title(
                    f"bs={bs}\nPSNR={eval_result['psnr']:.1f} dB", fontsize=10
                )
                axes[row, j].axis("off")

                del model, trainer
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
                gc.collect()

        fig.suptitle(f"Reconstruction Quality: {img_path.stem}", fontsize=14, y=1.02)
        fig.tight_layout()
        save_path = plots_dir / f"visual_grid_{img_path.stem}.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {save_path}")


# ===================================================================
# Experiment 5: VRAM profiling
# ===================================================================

def run_vram_profile(out_dir: Path):
    """Measure peak VRAM for each batch size and architecture."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT 5: VRAM Profiling")
    print("=" * 70)

    if DEVICE != "cuda":
        print("  Skipping — no CUDA device available.")
        return []

    img_path = Path("data/images/kodim01.png")
    img, img_np = _load_image(img_path)
    results = []

    for arch_name, build_fn in [("siren", _build_siren), ("coin", _build_coin)]:
        for bs in BATCH_SIZES:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model = build_fn()
            sampler = _make_sampler(bs)
            trainer = Trainer(
                model=model, sampler=sampler, image=img,
                n_steps=50, lr=LR, device=DEVICE, log_every=50,
            )
            trainer.train()

            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"  {arch_name} bs={bs}: {peak_mb:.1f} MB peak VRAM")

            results.append({
                "arch": arch_name,
                "batch_size": bs,
                "peak_vram_mb": round(peak_mb, 1),
            })

            del model, trainer
            torch.cuda.empty_cache()
            gc.collect()

    _save_json(results, out_dir / "vram_profile.json")

    # Plot
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    for arch in ["siren", "coin"]:
        runs = [r for r in results if r["arch"] == arch]
        labels = [str(r["batch_size"]) for r in runs]
        vrams = [r["peak_vram_mb"] for r in runs]
        ax.plot(labels, vrams, marker="o", label=arch.upper(), linewidth=2)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Peak VRAM (MB)")
    ax.set_title("GPU Memory Usage vs Batch Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "vram_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {plots_dir / 'vram_profile.png'}")

    return results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Extended thesis experiments")
    parser.add_argument("--experiment", default="all",
                        choices=["coin_sweep", "iso_time", "adaptive",
                                 "visuals", "vram", "all"])
    parser.add_argument("--output", default="experiments/extended")
    parser.add_argument("--image-limit", type=int, default=None,
                        help="Limit number of images (for quick testing)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = {
        "vram": lambda: run_vram_profile(out_dir),
        "coin_sweep": lambda: run_coin_sweep(out_dir, args.image_limit),
        "iso_time": lambda: run_iso_time(out_dir, args.image_limit),
        "adaptive": lambda: run_adaptive(out_dir, args.image_limit),
        "visuals": lambda: run_visuals(out_dir),
    }

    if args.experiment == "all":
        for name, fn in experiments.items():
            fn()
    else:
        experiments[args.experiment]()

    print("\n" + "=" * 70)
    print("  ALL DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
