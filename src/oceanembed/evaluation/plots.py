from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def save_evaluation_plots(observed: np.ndarray, predicted: np.ndarray, depths_m: np.ndarray, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    mean_obs, mean_pred = observed.mean(0), predicted.mean(0)
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(mean_obs, depths_m, "o-", label="GLORYS target")
    ax.plot(mean_pred, depths_m, "s--", label="OceanEmbed")
    ax.invert_yaxis(); ax.set(xlabel="Temperature (C)", ylabel="Depth (m)", title="Mean predicted profile")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "mean_profile.png", dpi=180); plt.close(fig)
    depth_rmse = np.sqrt(((observed - predicted) ** 2).mean(0))
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(depth_rmse, depths_m, "o-")
    ax.invert_yaxis(); ax.set(xlabel="RMSE (C)", ylabel="Depth (m)", title="Error by depth")
    fig.tight_layout(); fig.savefig(out / "rmse_by_depth.png", dpi=180); plt.close(fig)
