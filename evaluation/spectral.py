"""
Spectral (frequency-domain) analysis of INR reconstruction quality.

Computes FFT-based metrics to understand how well the model captures
different frequency bands of the original image.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core spectral helpers
# ---------------------------------------------------------------------------

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) float32 RGB to (H, W) grayscale using luminance."""
    if img.ndim == 2:
        return img
    return 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]


def fft_magnitude(img: np.ndarray) -> np.ndarray:
    """
    Compute the centered 2D FFT log-magnitude spectrum of an image.

    Args:
        img: (H, W, 3) or (H, W) float32 array in [0, 1].

    Returns:
        (H, W) log-magnitude spectrum (centered).
    """
    gray = _to_grayscale(img)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    return magnitude.astype(np.float32)


def radial_profile(spectrum: np.ndarray) -> np.ndarray:
    """
    Compute the radially averaged power spectrum.

    Args:
        spectrum: (H, W) centered magnitude spectrum.

    Returns:
        1D array of mean magnitude per radial frequency bin.
    """
    H, W = spectrum.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)
    profile = np.zeros(max_r, dtype=np.float64)
    counts = np.zeros(max_r, dtype=np.float64)
    mask = r < max_r
    np.add.at(profile, r[mask], spectrum[mask])
    np.add.at(counts, r[mask], 1)
    counts[counts == 0] = 1  # avoid division by zero
    return (profile / counts).astype(np.float32)


# ---------------------------------------------------------------------------
# Band-wise spectral error
# ---------------------------------------------------------------------------

def spectral_error_by_band(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_bands: int = 3,
) -> dict:
    """
    Compute spectral error in frequency bands (low / mid / high).

    Splits the radial frequency axis into *n_bands* equal-width bands and
    reports the mean squared error in each.

    Args:
        original:      (H, W, 3) float32 reference image.
        reconstructed: (H, W, 3) float32 reconstructed image.
        n_bands:       Number of bands (default 3 → low / mid / high).

    Returns:
        dict with keys:
            band_edges:  list of (lo, hi) radial-frequency indices per band.
            band_errors: list of mean squared spectral error per band.
            band_labels: list of human-readable labels.
    """
    spec_orig = fft_magnitude(original)
    spec_recon = fft_magnitude(reconstructed)

    prof_orig = radial_profile(spec_orig)
    prof_recon = radial_profile(spec_recon)

    n_freq = len(prof_orig)
    edges = np.linspace(0, n_freq, n_bands + 1, dtype=int)

    labels = {0: "low", 1: "mid", 2: "high"}
    band_edges = []
    band_errors = []
    band_labels = []

    for i in range(n_bands):
        lo, hi = int(edges[i]), int(edges[i + 1])
        band_edges.append((lo, hi))
        mse = float(np.mean((prof_orig[lo:hi] - prof_recon[lo:hi]) ** 2))
        band_errors.append(mse)
        band_labels.append(labels.get(i, f"band_{i}"))

    return {
        "band_edges": band_edges,
        "band_errors": band_errors,
        "band_labels": band_labels,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectral_comparison(
    original: np.ndarray,
    reconstructed: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot radial frequency profiles of original vs. reconstructed,
    plus per-band error bars.

    Args:
        original:      (H, W, 3) float32 reference image.
        reconstructed: (H, W, 3) float32 reconstructed image.
        save_path:     If given, save figure to this path.

    Returns:
        matplotlib Figure.
    """
    prof_orig = radial_profile(fft_magnitude(original))
    prof_recon = radial_profile(fft_magnitude(reconstructed))
    freqs = np.arange(len(prof_orig))

    band_info = spectral_error_by_band(original, reconstructed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: radial profiles ---
    ax = axes[0]
    ax.plot(freqs, prof_orig, label="Original", linewidth=1.2)
    ax.plot(freqs, prof_recon, label="Reconstructed", linewidth=1.2, alpha=0.8)
    ax.set_xlabel("Radial frequency")
    ax.set_ylabel("Log magnitude")
    ax.set_title("Radial Frequency Profile")
    ax.legend()

    # --- Right: band errors ---
    ax = axes[1]
    colors = ["#4CAF50", "#FFC107", "#F44336"]
    ax.bar(
        band_info["band_labels"],
        band_info["band_errors"],
        color=colors[: len(band_info["band_labels"])],
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Mean Squared Spectral Error")
    ax.set_title("Error by Frequency Band")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
