from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


class OceanProfileDataset(Dataset):
    """In-memory local patches, normalized using training-set statistics only."""

    def __init__(
        self, archive: str | Path, indices: Sequence[int], channels: Sequence[int], stats: dict | None = None
    ) -> None:
        payload = np.load(archive, allow_pickle=False)
        self.x = payload["X"][np.asarray(indices)][:, channels].astype(np.float32)
        self.y = payload["y"][np.asarray(indices)].astype(np.float32)
        if stats is None:
            stats = {
                "x_mean": self.x.mean(axis=(0, 2, 3), keepdims=True),
                "x_std": self.x.std(axis=(0, 2, 3), keepdims=True).clip(1e-6),
                "y_mean": self.y.mean(axis=0, keepdims=True),
                "y_std": self.y.std(axis=0, keepdims=True).clip(1e-6),
            }
        self.stats = stats
        self.x = (self.x - stats["x_mean"]) / stats["x_std"]
        self.y = (self.y - stats["y_mean"]) / stats["y_std"]

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.x[index]), torch.from_numpy(self.y[index])
