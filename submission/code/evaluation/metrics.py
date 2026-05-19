"""
Image quality metrics for INR evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def psnr_from_mse(mse: float, max_val: float = 1.0) -> float:
    """Compute PSNR from a scalar MSE value (images in [0, 1])."""
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio between two images.

    Args:
        img1, img2: (H, W, 3) float32 arrays in [0, 1].
        max_val:    Maximum pixel value (1.0 for normalized images).

    Returns:
        PSNR in dB.
    """
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    return psnr_from_mse(mse, max_val)


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.

    Args:
        img1, img2: (H, W, 3) float32 arrays in [0, 1].

    Returns:
        SSIM score in [-1, 1] (higher is better).
    """
    return float(_ssim(img1, img2, channel_axis=2, data_range=1.0))


# ---------------------------------------------------------------------------
# Image reconstruction from trained model
# ---------------------------------------------------------------------------

def reconstruct(model: torch.nn.Module, H: int, W: int, device: str = "cpu") -> np.ndarray:
    """
    Reconstruct a full image by querying the model at every pixel coordinate.

    Args:
        model:  Trained SIREN (or compatible) model.
        H, W:   Image height and width in pixels.
        device: Torch device string.

    Returns:
        (H, W, 3) float32 numpy array in [0, 1].
    """
    from training.sampling import _build_coord_grid

    model.eval()
    coords = torch.from_numpy(_build_coord_grid(H, W)).to(device)  # (H*W, 2)

    with torch.no_grad():
        # Process in chunks to avoid OOM on large images
        chunk = 65536
        parts = []
        for i in range(0, coords.shape[0], chunk):
            parts.append(model(coords[i : i + chunk]).cpu())
        pixels = torch.cat(parts, dim=0)  # (H*W, 3)

    img = pixels.numpy().reshape(H, W, 3)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img


def evaluate(model: torch.nn.Module, ref_image: np.ndarray, device: str = "cpu") -> dict:
    """
    Reconstruct image and compute PSNR + SSIM against reference.

    Args:
        model:     Trained model.
        ref_image: (H, W, 3) float32 array in [0, 1].
        device:    Torch device string.

    Returns:
        dict with keys: psnr, ssim, reconstructed (np.ndarray)
    """
    H, W, _ = ref_image.shape
    recon = reconstruct(model, H, W, device=device)
    return {
        "psnr": psnr(ref_image, recon),
        "ssim": ssim(ref_image, recon),
        "reconstructed": recon,
    }
