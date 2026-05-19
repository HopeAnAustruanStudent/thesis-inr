"""
Pixel-coordinate sampling strategies for INR training.

All samplers return:
    coords:  (N, 2) float32 tensor of (x, y) in [-1, 1], x = col, y = row
    colors:  (N, 3) float32 tensor of RGB values in [0, 1]
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _image_to_tensor(image) -> np.ndarray:
    """Convert PIL Image or numpy array to float32 H×W×3 in [0, 1]."""
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))
    image = np.asarray(image, dtype=np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image  # (H, W, 3)


def _build_coord_grid(H: int, W: int) -> np.ndarray:
    """
    Build a meshgrid of normalized pixel coordinates.

    Returns:
        coords: (H*W, 2) float32 array, columns are (x, y).
                x corresponds to the column axis, y to the row axis,
                both in [-1, 1].
    """
    # linspace gives center of each pixel
    xs = np.linspace(-1, 1, W, dtype=np.float32)  # (W,)
    ys = np.linspace(-1, 1, H, dtype=np.float32)  # (H,)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")  # both (H, W)
    coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)  # (H*W, 2)
    return coords


# ---------------------------------------------------------------------------
# Public sampler functions
# ---------------------------------------------------------------------------

def full_image(image) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return ALL pixel coordinates and colors.

    Args:
        image: PIL Image or (H, W, 3) numpy array.

    Returns:
        coords: (H*W, 2) tensor
        colors: (H*W, 3) tensor
    """
    img = _image_to_tensor(image)
    H, W, _ = img.shape
    coords = _build_coord_grid(H, W)
    colors = img.reshape(-1, 3)
    return torch.from_numpy(coords), torch.from_numpy(colors)


def random_pixels(image, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample n random pixels uniformly without replacement (with replacement
    when n > H*W).

    Args:
        image: PIL Image or (H, W, 3) numpy array.
        n:     Number of pixels to sample per call.

    Returns:
        coords: (n, 2) tensor
        colors: (n, 3) tensor
    """
    img = _image_to_tensor(image)
    H, W, _ = img.shape
    total = H * W
    replace = n > total
    idx = np.random.choice(total, size=n, replace=replace)
    all_coords = _build_coord_grid(H, W)
    coords = all_coords[idx]
    colors = img.reshape(-1, 3)[idx]
    return torch.from_numpy(coords), torch.from_numpy(colors)


def grid_patch(image, patch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random axis-aligned patch of size patch_size × patch_size.

    If the image is smaller than patch_size in either dimension, the entire
    image is returned (equivalent to full_image).

    Args:
        image:      PIL Image or (H, W, 3) numpy array.
        patch_size: Side length of the square patch in pixels.

    Returns:
        coords: (patch_size*patch_size, 2) tensor  (or H*W if image is small)
        colors: (patch_size*patch_size, 3) tensor
    """
    img = _image_to_tensor(image)
    H, W, _ = img.shape

    ph = min(patch_size, H)
    pw = min(patch_size, W)

    row0 = np.random.randint(0, H - ph + 1)
    col0 = np.random.randint(0, W - pw + 1)

    patch = img[row0 : row0 + ph, col0 : col0 + pw, :]  # (ph, pw, 3)

    # Build coords for the patch in global image coordinate space
    xs_global = np.linspace(-1, 1, W, dtype=np.float32)[col0 : col0 + pw]
    ys_global = np.linspace(-1, 1, H, dtype=np.float32)[row0 : row0 + ph]
    grid_y, grid_x = np.meshgrid(ys_global, xs_global, indexing="ij")
    coords = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)
    colors = patch.reshape(-1, 3)

    return torch.from_numpy(coords), torch.from_numpy(colors.copy())


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def get_sampler(mode: str, **kwargs):
    """
    Return a zero-argument callable that samples from the image.

    Args:
        mode:   "full_image", "random_pixels", or "grid_patch".
        kwargs: Extra arguments forwarded to the chosen sampler.
                  - random_pixels: n (int)
                  - grid_patch:    patch_size (int)

    Returns:
        sampler(image) -> (coords, colors)
    """
    if mode == "full_image":
        return full_image
    elif mode == "random_pixels":
        n = kwargs["n"]
        return lambda img: random_pixels(img, n=n)
    elif mode == "grid_patch":
        patch_size = kwargs["patch_size"]
        return lambda img: grid_patch(img, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown sampler mode: {mode!r}. "
                         f"Choose from 'full_image', 'random_pixels', 'grid_patch'.")
