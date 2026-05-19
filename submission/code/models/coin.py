"""
COIN: Compression with Implicit Neural Representations
Reference: Dupont et al., arxiv:2103.03123

A ReLU-based coordinate MLP baseline for comparison with SIREN.
Uses positional encoding to compensate for the lack of periodic activations.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Fourier feature positional encoding for coordinate inputs.

    Maps each input dimension d to [sin(2^0 π x_d), cos(2^0 π x_d), ...,
    sin(2^{L-1} π x_d), cos(2^{L-1} π x_d)], yielding 2*L features per
    input dimension.

    Args:
        in_features: Input dimensionality (2 for image coordinates).
        n_freqs:     Number of frequency octaves L.
    """

    def __init__(self, in_features: int = 2, n_freqs: int = 10):
        super().__init__()
        self.in_features = in_features
        self.n_freqs = n_freqs
        freqs = torch.tensor(
            [2.0 ** i * math.pi for i in range(n_freqs)], dtype=torch.float32
        )
        self.register_buffer("freqs", freqs)  # (L,)

    @property
    def out_features(self) -> int:
        return self.in_features * self.n_freqs * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_features)
        Returns:
            (N, in_features * 2 * n_freqs)
        """
        # x: (N, D) -> (N, D, 1) * (1, 1, L) -> (N, D, L)
        scaled = x.unsqueeze(-1) * self.freqs.unsqueeze(0).unsqueeze(0)
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1).reshape(
            x.shape[0], -1
        )


class COIN(nn.Module):
    """
    COIN network — ReLU MLP with positional encoding.

    Maps 2D pixel coordinates (x, y) in [-1, 1] to RGB values in [0, 1].

    Args:
        in_features:     Input dimensionality (2 for 2D coordinates).
        hidden_features: Width of each hidden layer.
        hidden_layers:   Number of hidden layers.
        out_features:    Output dimensionality (3 for RGB).
        n_freqs:         Positional encoding frequency octaves.
    """

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 3,
        n_freqs: int = 10,
    ):
        super().__init__()
        self.pe = PositionalEncoding(in_features, n_freqs)

        layers = []
        prev = self.pe.out_features
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev, hidden_features))
            layers.append(nn.ReLU(inplace=True))
            prev = hidden_features
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (N, 2) tensor of (x, y) coordinates in [-1, 1].
        Returns:
            (N, 3) tensor of RGB values in [0, 1].
        """
        x = self.pe(coords)
        x = self.net(x)
        return torch.sigmoid(x)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
