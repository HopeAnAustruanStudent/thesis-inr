"""
Training loop for INR-based image compression.

Supports:
  - Fixed-step training (original)
  - Time-budget training (iso-time experiments)
  - Adaptive batch-size schedule (curriculum training)
  - Wall-clock PSNR logging for convergence analysis
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from evaluation.metrics import psnr_from_mse


class Trainer:
    """
    Trains a coordinate-MLP (SIREN, COIN, etc.) to overfit to a single image.

    Args:
        model:     PyTorch module — coords (N,2) -> colors (N,3).
        sampler:   Callable(image) -> (coords, colors) tensors on CPU.
        image:     PIL Image or numpy array (used as the training target).
        n_steps:   Number of gradient steps (ignored if time_budget_s is set).
        lr:        Adam learning rate.
        device:    "cuda", "cpu", etc.
        log_every: Log PSNR every this many steps.
        time_budget_s: If set, train for this many seconds instead of n_steps.
    """

    def __init__(
        self,
        model: nn.Module,
        sampler: Callable,
        image,
        n_steps: int = 2000,
        lr: float = 1e-4,
        device: str = "cpu",
        log_every: int = 100,
        time_budget_s: float | None = None,
    ):
        self.model = model.to(device)
        self.sampler = sampler
        self.image = image
        self.n_steps = n_steps
        self.device = device
        self.log_every = log_every
        self.time_budget_s = time_budget_s

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Logs
        self.loss_history: list[float] = []
        self.psnr_history: list[tuple[int, float]] = []  # (step, psnr)
        self.psnr_vs_time: list[tuple[float, float]] = []  # (seconds, psnr)
        self.train_time: float = 0.0
        self.total_steps: int = 0

    def train(self) -> dict:
        """Run the training loop.

        Returns:
            dict with keys: loss_history, psnr_history, psnr_vs_time,
                            train_time_s, total_steps
        """
        self.model.train()
        t0 = time.perf_counter()
        step = 0

        use_time_budget = self.time_budget_s is not None

        if use_time_budget:
            pbar = tqdm(desc="Training (timed)", unit="step")
        else:
            pbar = tqdm(range(1, self.n_steps + 1), desc="Training", unit="step")

        while True:
            step += 1

            # Check stopping condition
            if use_time_budget:
                elapsed = time.perf_counter() - t0
                if elapsed >= self.time_budget_s:
                    break
                pbar.update(1)
            else:
                if step > self.n_steps:
                    break

            coords, colors = self.sampler(self.image)
            coords = coords.to(self.device)
            colors = colors.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(coords)
            loss = self.criterion(pred, colors)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            self.loss_history.append(loss_val)

            if step % self.log_every == 0 or step == 1:
                psnr = psnr_from_mse(loss_val)
                self.psnr_history.append((step, psnr))
                elapsed = time.perf_counter() - t0
                self.psnr_vs_time.append((round(elapsed, 3), round(psnr, 4)))
                pbar.set_postfix(loss=f"{loss_val:.6f}", psnr=f"{psnr:.2f} dB")

        pbar.close()
        self.train_time = time.perf_counter() - t0
        self.total_steps = step - 1
        return {
            "loss_history": self.loss_history,
            "psnr_history": self.psnr_history,
            "psnr_vs_time": self.psnr_vs_time,
            "train_time_s": self.train_time,
            "total_steps": self.total_steps,
        }

    def save(self, path: str | Path):
        """Save model weights to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path):
        """Load model weights from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class AdaptiveTrainer:
    """
    Trains with an adaptive batch-size schedule: starts with a small batch
    (fast low-frequency learning) then switches to a large batch (high-frequency
    refinement).

    Args:
        model:        PyTorch module.
        image:        PIL Image or numpy array.
        schedule:     List of (fraction, batch_size) tuples.
                      e.g. [(0.3, 1024), (0.7, "full")] = first 30% at bs=1024,
                      remaining 70% at full batch.
        n_steps:      Total number of gradient steps.
        lr:           Adam learning rate.
        device:       Torch device string.
        log_every:    Logging frequency.
        time_budget_s: If set, use time-based training instead of step-based.
    """

    def __init__(
        self,
        model: nn.Module,
        image,
        schedule: list[tuple[float, int | str]],
        n_steps: int = 2000,
        lr: float = 1e-4,
        device: str = "cpu",
        log_every: int = 100,
        time_budget_s: float | None = None,
    ):
        from training.sampling import get_sampler

        self.model = model.to(device)
        self.image = image
        self.schedule = schedule  # [(fraction, batch_size), ...]
        self.n_steps = n_steps
        self.lr = lr
        self.device = device
        self.log_every = log_every
        self.time_budget_s = time_budget_s

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # Pre-build samplers for each phase
        self.phase_samplers = []
        for _, bs in schedule:
            if bs == "full":
                self.phase_samplers.append(get_sampler("full_image"))
            else:
                self.phase_samplers.append(get_sampler("random_pixels", n=int(bs)))

        # Logs
        self.loss_history: list[float] = []
        self.psnr_history: list[tuple[int, float]] = []
        self.psnr_vs_time: list[tuple[float, float]] = []
        self.phase_history: list[tuple[int, str]] = []  # (step, batch_size)
        self.train_time: float = 0.0
        self.total_steps: int = 0

    def _get_phase(self, progress: float) -> int:
        """Return phase index based on progress fraction [0, 1]."""
        cumulative = 0.0
        for i, (frac, _) in enumerate(self.schedule):
            cumulative += frac
            if progress <= cumulative:
                return i
        return len(self.schedule) - 1

    def train(self) -> dict:
        self.model.train()
        t0 = time.perf_counter()

        use_time_budget = self.time_budget_s is not None
        total = self.time_budget_s if use_time_budget else self.n_steps

        pbar = tqdm(desc="Adaptive Training", unit="step")
        step = 0

        while True:
            step += 1
            elapsed = time.perf_counter() - t0

            if use_time_budget:
                if elapsed >= self.time_budget_s:
                    break
                progress = elapsed / self.time_budget_s
            else:
                if step > self.n_steps:
                    break
                progress = step / self.n_steps

            phase_idx = self._get_phase(progress)
            sampler = self.phase_samplers[phase_idx]

            coords, colors = sampler(self.image)
            coords = coords.to(self.device)
            colors = colors.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(coords)
            loss = self.criterion(pred, colors)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()
            self.loss_history.append(loss_val)

            if step % self.log_every == 0 or step == 1:
                psnr = psnr_from_mse(loss_val)
                self.psnr_history.append((step, psnr))
                self.psnr_vs_time.append((round(elapsed, 3), round(psnr, 4)))
                bs_label = str(self.schedule[phase_idx][1])
                self.phase_history.append((step, bs_label))
                pbar.set_postfix(
                    loss=f"{loss_val:.6f}", psnr=f"{psnr:.2f} dB",
                    phase=f"{phase_idx+1}/{len(self.schedule)}",
                    bs=bs_label,
                )
                pbar.update(self.log_every)

        pbar.close()
        self.train_time = time.perf_counter() - t0
        self.total_steps = step - 1
        return {
            "loss_history": self.loss_history,
            "psnr_history": self.psnr_history,
            "psnr_vs_time": self.psnr_vs_time,
            "phase_history": self.phase_history,
            "train_time_s": self.train_time,
            "total_steps": self.total_steps,
        }

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str | Path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
