"""
SIREN: Implicit Neural Representations with Periodic Activation Functions
Reference: Sitzmann et al., arxiv:2006.09661
"""

import torch
import torch.nn as nn
import numpy as np


class SineLayer(nn.Module):
    """Single linear layer with sinusoidal activation."""

    def __init__(self, in_features, out_features, omega_0=30.0, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        fan_in = self.linear.weight.shape[1]
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform [-1/fan_in, 1/fan_in] * omega_0
                bound = 1.0 / fan_in
            else:
                # Subsequent layers: uniform [-sqrt(6/fan_in)/omega_0, sqrt(6/fan_in)/omega_0]
                bound = np.sqrt(6.0 / fan_in) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN network for image compression.

    Maps 2D pixel coordinates (x, y) in [-1, 1] to RGB values in [0, 1].

    Args:
        in_features:   Input dimensionality (2 for 2D coordinates).
        hidden_features: Width of each hidden layer.
        hidden_layers: Number of hidden layers.
        out_features:  Output dimensionality (3 for RGB).
        omega_0:       Frequency factor for sinusoidal activations.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 3,
        omega_0: float = 30.0,
    ):
        super().__init__()
        self.omega_0 = omega_0

        layers = [SineLayer(in_features, hidden_features, omega_0=omega_0, is_first=True)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=omega_0, is_first=False))
        self.net = nn.Sequential(*layers)

        # Final linear layer — no sine activation, output in raw range then clamped
        final = nn.Linear(hidden_features, out_features)
        fan_in = final.weight.shape[1]
        bound = np.sqrt(6.0 / fan_in) / omega_0
        with torch.no_grad():
            final.weight.uniform_(-bound, bound)
            final.bias.uniform_(-bound, bound)
        self.final_layer = final

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) tensor of (x, y) coordinates in [-1, 1].
        Returns:
            (N, 3) tensor of RGB values in [0, 1].
        """
        x = self.net(coords)
        x = self.final_layer(x)
        return torch.sigmoid(x)  # map to [0, 1]

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
