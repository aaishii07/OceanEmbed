from __future__ import annotations

import torch
from torch import nn


class OceanEmbedCNN(nn.Module):
    """Compact CNN encoder with a profile decoder for Colab-friendly training."""

    def __init__(self, in_channels: int, n_depths: int, embedding_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, embedding_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.decoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_depths),
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))
