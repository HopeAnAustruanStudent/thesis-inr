"""
Radially-averaged log-magnitude spectra at different batch sizes.

For a representative Kodak image, load the saved SIREN weights at each batch
size, reconstruct, compute the radial profile, and plot all curves against
the reference.  Produces `submission/figures/radial_profiles.png`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from evaluation.metrics import reconstruct
from evaluation.spectral import fft_magnitude, radial_profile
from models.siren import SIREN

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "experiments" / "results" / "weights"
DATA_DIR = ROOT / "data" / "images"
OUT = ROOT / "submission" / "figures" / "radial_profiles.png"

BATCH_SIZES = ["256", "1024", "4096", "16384", "full"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REP_IMAGES = ["kodim01", "kodim23"]   # smooth + detail-heavy


def build_siren() -> SIREN:
    return SIREN(in_features=2, hidden_features=256, hidden_layers=5,
                 out_features=3, omega_0=30.0)


def load_and_reconstruct(stem: str, bs: str) -> tuple[np.ndarray, np.ndarray]:
    img = np.array(Image.open(DATA_DIR / f"{stem}.png").convert("RGB"),
                   dtype=np.float32) / 255.0
    H, W, _ = img.shape
    model = build_siren().to(DEVICE)
    weights = WEIGHTS_DIR / f"{stem}_bs{bs}.pt"
    model.load_state_dict(torch.load(weights, map_location=DEVICE))
    recon = reconstruct(model, H, W, device=DEVICE)
    return img, recon


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd"]

    for ax, stem in zip(axes, REP_IMAGES):
        img, _ = load_and_reconstruct(stem, BATCH_SIZES[-1])
        prof_ref = radial_profile(fft_magnitude(img))
        freqs = np.arange(len(prof_ref))
        ax.plot(freqs, prof_ref, "k-", linewidth=2.2, label="Reference", alpha=0.9)

        for bs, c in zip(BATCH_SIZES, colors):
            _, recon = load_and_reconstruct(stem, bs)
            prof = radial_profile(fft_magnitude(recon))
            label = "full" if bs == "full" else f"bs={bs}"
            ax.plot(freqs, prof, linewidth=1.5, alpha=0.85, color=c, label=label)

        ax.set_xlabel("Radial frequency (pixels)")
        ax.set_title(f"{stem}")
        ax.grid(alpha=0.3, linestyle=":")
        ax.set_xlim(0, len(prof_ref))

    axes[0].set_ylabel("Log-magnitude")
    axes[1].legend(loc="upper right", fontsize=9, ncol=1, framealpha=0.9)
    fig.suptitle("Radially-averaged log-magnitude spectrum: reference vs reconstruction",
                 fontsize=12)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
