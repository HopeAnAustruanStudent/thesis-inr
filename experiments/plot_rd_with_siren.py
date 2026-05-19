"""
Rate-distortion plot: JPEG vs SIREN at multiple model sizes.

Produces `submission/figures/jpeg_vs_inr.png` (overwrites the existing plot),
extending it with a SIREN curve obtained from the model-size ablation
(widths 64/128/256/512, full-batch, average over 6 images).

File size for SIREN is computed as n_params * 2 bytes (float16 weights).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JPEG_JSON = ROOT / "experiments" / "final" / "jpeg_comparison.json"
ABLATION_JSON = ROOT / "experiments" / "ablation" / "model_size_ablation.json"
OUT = ROOT / "submission" / "figures" / "jpeg_vs_inr.png"


def aggregate_jpeg(data: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Average JPEG file size and PSNR across images, per quality level."""
    per_q: dict[int, list[tuple[int, float]]] = {}
    for entry in data:
        for row in entry.get("jpeg", []):
            per_q.setdefault(row["quality"], []).append((row["file_size"], row["psnr"]))
    qualities = sorted(per_q.keys())
    sizes = np.array([np.mean([s for s, _ in per_q[q]]) for q in qualities]) / 1024.0
    psnrs = np.array([np.mean([p for _, p in per_q[q]]) for q in qualities])
    return sizes, psnrs


def aggregate_siren(data: list[dict]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Average SIREN full-batch PSNR per model width. Size = 2 bytes * n_params (float16)."""
    per_width: dict[int, list[float]] = {}
    params_of_width: dict[int, int] = {}
    for row in data:
        if row["batch_size"] != "full":
            continue
        per_width.setdefault(row["width"], []).append(row["psnr"])
        params_of_width[row["width"]] = row["n_params"]
    widths = sorted(per_width.keys())
    sizes_kb = np.array([params_of_width[w] * 2 / 1024.0 for w in widths])
    psnrs = np.array([np.mean(per_width[w]) for w in widths])
    return sizes_kb, psnrs, widths


def main() -> None:
    jpeg = json.loads(JPEG_JSON.read_text())
    ablation = json.loads(ABLATION_JSON.read_text())

    jpeg_sizes, jpeg_psnrs = aggregate_jpeg(jpeg)
    siren_sizes, siren_psnrs, widths = aggregate_siren(ablation)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(jpeg_sizes, jpeg_psnrs, "o-", color="#1a3a6e", linewidth=2,
            markersize=6, label="JPEG (avg over Kodak)")
    ax.plot(siren_sizes, siren_psnrs, "s-", color="#c0392b", linewidth=2,
            markersize=7, label="SIREN full-batch (float16)")

    for w, sz, p in zip(widths, siren_sizes, siren_psnrs):
        ax.annotate(f"w={w}", (sz, p),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=8, color="#c0392b")

    ax.set_xlabel("File size (KB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Rate–distortion: JPEG vs SIREN (Kodak average)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3, which="both", linestyle=":")
    ax.legend(loc="lower right")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT}")
    print(f"JPEG points: {len(jpeg_sizes)}")
    print(f"SIREN widths: {widths}  sizes={siren_sizes.round(1).tolist()}  "
          f"PSNR={siren_psnrs.round(2).tolist()}")


if __name__ == "__main__":
    main()
