"""
Multi-seed experiment + JPEG baseline comparison.

Usage:
    python -m experiments.run_seeds_and_jpeg --experiment jpeg
    python -m experiments.run_seeds_and_jpeg --experiment seeds
    python -m experiments.run_seeds_and_jpeg --experiment all
"""

from __future__ import annotations

import argparse
import gc
import io
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
from evaluation.metrics import psnr, ssim, evaluate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [256, 1024, 4096, 16384, "full"]


def _collect_images(limit: int | None = None) -> list[Path]:
    d = Path("data/images")
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() == ".png")
    if limit:
        imgs = imgs[:limit]
    return imgs


def _save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ===================================================================
# JPEG Baseline Comparison
# ===================================================================

def jpeg_compress(img_np: np.ndarray, quality: int) -> tuple[np.ndarray, int]:
    """Compress image with JPEG at given quality, return decoded + file size."""
    img_uint8 = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_uint8)

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    jpeg_size = buf.tell()

    buf.seek(0)
    decoded = np.array(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return decoded, jpeg_size


def run_jpeg_comparison(out_dir: Path):
    """Compare INR compression quality against JPEG at similar file sizes."""
    print("\n" + "=" * 70)
    print("  JPEG Baseline Comparison")
    print("=" * 70)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images()
    all_results = []

    # INR model size in bytes
    # SIREN: 264,707 params
    inr_sizes = {
        "float32": 264707 * 4,     # 1,058,828 bytes ~ 1034 KB
        "float16": 264707 * 2,     # 529,414 bytes ~ 517 KB
        "int8":    264707 * 1,     # 264,707 bytes ~ 258 KB
    }

    # JPEG qualities to test — find ones near INR file sizes
    jpeg_qualities = [2, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 95]

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0
        H, W, _ = img_np.shape
        raw_size = H * W * 3  # uncompressed size

        img_result = {
            "image": img_path.name,
            "raw_size": raw_size,
            "inr_sizes": inr_sizes,
            "jpeg": [],
        }

        for q in jpeg_qualities:
            decoded, fsize = jpeg_compress(img_np, q)
            p = psnr(img_np, decoded)
            s = ssim(img_np, decoded)
            img_result["jpeg"].append({
                "quality": q,
                "file_size": fsize,
                "psnr": p,
                "ssim": s,
            })

        all_results.append(img_result)
        print(f"  {img_path.stem}: raw={raw_size} bytes")

    _save_json(all_results, out_dir / "jpeg_comparison.json")

    # Load SIREN sweep results for comparison
    siren_path = Path("experiments/results/results.json")
    with open(siren_path) as f:
        siren_results = json.load(f)

    # Aggregate
    # Average JPEG PSNR at each quality across all images
    jpeg_avg = {}
    for q in jpeg_qualities:
        psnrs = []
        sizes = []
        for r in all_results:
            for j in r["jpeg"]:
                if j["quality"] == q:
                    psnrs.append(j["psnr"])
                    sizes.append(j["file_size"])
        jpeg_avg[q] = {
            "psnr": np.mean(psnrs),
            "size": np.mean(sizes),
            "size_kb": np.mean(sizes) / 1024,
        }

    # Average SIREN PSNR at each batch size
    siren_avg = {}
    for r in siren_results:
        bs = str(r["batch_size"])
        siren_avg.setdefault(bs, []).append(r["psnr"])
    siren_avg = {bs: np.mean(v) for bs, v in siren_avg.items()}

    # --- Plot: PSNR vs file size (rate-distortion curve) ---
    fig, ax = plt.subplots(figsize=(9, 6))

    # JPEG curve
    jpeg_sizes_kb = [jpeg_avg[q]["size_kb"] for q in jpeg_qualities]
    jpeg_psnrs = [jpeg_avg[q]["psnr"] for q in jpeg_qualities]
    ax.plot(jpeg_sizes_kb, jpeg_psnrs, "o-", color="#F44336",
            label="JPEG", linewidth=2, markersize=6)

    # Annotate some JPEG qualities
    for q in [5, 20, 50, 80, 95]:
        ax.annotate(f"q={q}", (jpeg_avg[q]["size_kb"], jpeg_avg[q]["psnr"]),
                   textcoords="offset points", xytext=(8, 4), fontsize=8, color="#F44336")

    # INR points at different quantizations
    inr_markers = {
        "float32": ("s", 12, inr_sizes["float32"] / 1024),
        "float16": ("D", 10, inr_sizes["float16"] / 1024),
        "int8": ("^", 10, inr_sizes["int8"] / 1024),
    }

    colors_bs = {"256": "#90CAF9", "1024": "#42A5F5", "4096": "#1E88E5",
                 "16384": "#1565C0", "full": "#0D47A1"}

    for bs in ["256", "1024", "4096", "16384", "full"]:
        p = siren_avg[bs]
        for quant, (marker, ms, size_kb) in inr_markers.items():
            label = f"SIREN bs={bs} ({quant})" if bs == "full" else None
            if bs != "full" and quant != "float16":
                continue  # Only show float16 for non-full to reduce clutter
            if bs == "full":
                ax.plot(size_kb, p, marker=marker, color=colors_bs[bs],
                       markersize=ms, label=f"SIREN ({quant})" if quant == "float16" or quant == "float32" else None,
                       markeredgecolor="black", linewidth=0.5, zorder=5)

    # Show all batch sizes at float16 as a connected line
    f16_size = inr_sizes["float16"] / 1024
    bs_order = ["256", "1024", "4096", "16384", "full"]
    bs_psnrs = [siren_avg[bs] for bs in bs_order]
    ax.plot([f16_size] * len(bs_order), bs_psnrs, "|", color="#2196F3",
            markersize=15, markeredgewidth=2)
    ax.annotate("SIREN float16\n(bs=256...full)", (f16_size, siren_avg["full"]),
               textcoords="offset points", xytext=(10, -5), fontsize=8, color="#2196F3")

    # Add vertical lines for INR sizes
    for quant, (_, _, size_kb) in inr_markers.items():
        ax.axvline(x=size_kb, color="#BBDEFB", linestyle=":", alpha=0.5)

    ax.set_xlabel("File Size (KB)", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Rate-Distortion: SIREN vs JPEG (Kodak avg, n=24)", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xlim(0, max(jpeg_sizes_kb) * 1.1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "jpeg_vs_inr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Print comparison table ---
    print("\n--- JPEG vs SIREN at similar file sizes ---")
    print(f"  SIREN float16: {inr_sizes['float16']/1024:.0f} KB, "
          f"PSNR range: {siren_avg['256']:.2f} (bs=256) to {siren_avg['full']:.2f} (full)")
    print(f"  SIREN float32: {inr_sizes['float32']/1024:.0f} KB, same PSNR")
    print()

    # Find JPEG quality that matches INR file sizes
    for quant, size in inr_sizes.items():
        size_kb = size / 1024
        # Find closest JPEG
        closest_q = min(jpeg_qualities,
                       key=lambda q: abs(jpeg_avg[q]["size_kb"] - size_kb))
        print(f"  At ~{size_kb:.0f} KB ({quant}):")
        print(f"    JPEG q={closest_q}: {jpeg_avg[closest_q]['psnr']:.2f} dB "
              f"({jpeg_avg[closest_q]['size_kb']:.0f} KB)")
        print(f"    SIREN full:  {siren_avg['full']:.2f} dB")
        print(f"    SIREN bs=16384: {siren_avg['16384']:.2f} dB")
        print()

    return all_results


# ===================================================================
# Multi-Seed Experiment
# ===================================================================

def run_multi_seed(out_dir: Path, n_seeds: int = 3, image_limit: int = 6):
    """Run SIREN sweep with multiple random seeds for statistical rigor."""
    print("\n" + "=" * 70)
    print(f"  Multi-Seed Experiment ({n_seeds} seeds, {image_limit} images)")
    print("=" * 70)

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    images = _collect_images(limit=image_limit)
    all_results = []

    for seed in range(n_seeds):
        print(f"\n  --- Seed {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img, dtype=np.float32) / 255.0

            for bs in BATCH_SIZES:
                tag = f"seed{seed}_{img_path.stem}_bs{bs}"
                print(f"    >>> {tag}")

                torch.manual_seed(seed)  # Reset seed for each run
                model = SIREN(in_features=2, hidden_features=256,
                             hidden_layers=5, out_features=3, omega_0=30.0)

                if bs == "full":
                    sampler = get_sampler("full_image")
                else:
                    sampler = get_sampler("random_pixels", n=int(bs))

                trainer = Trainer(
                    model=model, sampler=sampler, image=img,
                    n_steps=2000, lr=1e-4, device=DEVICE, log_every=100,
                )
                train_info = trainer.train()
                eval_result = evaluate(model, img_np, device=DEVICE)

                all_results.append({
                    "seed": seed,
                    "image": img_path.name,
                    "batch_size": bs,
                    "psnr": eval_result["psnr"],
                    "ssim": eval_result["ssim"],
                    "train_time_s": train_info["train_time_s"],
                })

                del model, trainer
                torch.cuda.empty_cache() if DEVICE == "cuda" else None
                gc.collect()

    _save_json(all_results, out_dir / "multi_seed.json")

    # --- Analysis ---
    print("\n--- Multi-Seed Analysis ---")

    # Compute inter-seed variance vs inter-image variance
    for bs in ["256", "1024", "4096", "16384", "full"]:
        runs = [r for r in all_results if str(r["batch_size"]) == bs]

        # Inter-image variance (fix seed, vary image)
        image_vars = []
        for seed in range(n_seeds):
            seed_runs = [r for r in runs if r["seed"] == seed]
            psnrs = [r["psnr"] for r in seed_runs]
            image_vars.append(np.var(psnrs))
        inter_image = np.mean(image_vars)

        # Inter-seed variance (fix image, vary seed)
        seed_vars = []
        for img_path in images:
            img_runs = [r for r in runs if r["image"] == img_path.name]
            psnrs = [r["psnr"] for r in img_runs]
            seed_vars.append(np.var(psnrs))
        inter_seed = np.mean(seed_vars)

        psnr_mean = np.mean([r["psnr"] for r in runs])
        psnr_std = np.std([r["psnr"] for r in runs])

        print(f"  bs={bs:>5s}: PSNR={psnr_mean:.2f}+-{psnr_std:.2f}  "
              f"image_var={inter_image:.4f}  seed_var={inter_seed:.4f}  "
              f"ratio={inter_image/(inter_seed+1e-8):.1f}x")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot: PSNR distribution per batch size
    bs_order = ["256", "1024", "4096", "16384", "full"]
    box_data = []
    for bs in bs_order:
        runs = [r for r in all_results if str(r["batch_size"]) == bs]
        box_data.append([r["psnr"] for r in runs])

    bp = axes[0].boxplot(box_data, labels=["Full" if b == "full" else b for b in bs_order],
                         patch_artist=True)
    colors = ["#BBDEFB", "#90CAF9", "#42A5F5", "#1E88E5", "#0D47A1"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[0].set_xlabel("Batch Size")
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].set_title(f"PSNR Distribution ({n_seeds} seeds x {image_limit} images)")

    # Variance decomposition
    image_vars_all = []
    seed_vars_all = []
    for bs in bs_order:
        runs = [r for r in all_results if str(r["batch_size"]) == bs]
        img_v = []
        for seed in range(n_seeds):
            seed_runs = [r for r in runs if r["seed"] == seed]
            img_v.append(np.var([r["psnr"] for r in seed_runs]))
        image_vars_all.append(np.mean(img_v))

        seed_v = []
        for img_path in images:
            img_runs = [r for r in runs if r["image"] == img_path.name]
            seed_v.append(np.var([r["psnr"] for r in img_runs]))
        seed_vars_all.append(np.mean(seed_v))

    x = np.arange(len(bs_order))
    w = 0.35
    axes[1].bar(x - w/2, image_vars_all, w, label="Inter-image variance",
               color="#2196F3", edgecolor="black", linewidth=0.5)
    axes[1].bar(x + w/2, seed_vars_all, w, label="Inter-seed variance",
               color="#FF9800", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Batch Size")
    axes[1].set_ylabel("PSNR Variance")
    axes[1].set_title("Variance Decomposition")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Full" if b == "full" else b for b in bs_order])
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plots_dir / "multi_seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Plots saved to {plots_dir}")
    return all_results


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all",
                        choices=["jpeg", "seeds", "all"])
    parser.add_argument("--output", default="experiments/final")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--image-limit", type=int, default=6)
    args = parser.parse_args()

    out_dir = Path(args.output)

    if args.experiment in ("jpeg", "all"):
        run_jpeg_comparison(out_dir)

    if args.experiment in ("seeds", "all"):
        run_multi_seed(out_dir, n_seeds=args.seeds, image_limit=args.image_limit)

    print("\n  ALL DONE")


if __name__ == "__main__":
    main()
