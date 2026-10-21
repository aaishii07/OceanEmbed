from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def profile_metrics(observed: np.ndarray, predicted: np.ndarray, depths_m: np.ndarray) -> dict:
    """Return global and per-depth metrics on values in degrees Celsius."""
    corr = np.corrcoef(observed.ravel(), predicted.ravel())[0, 1]
    return {
        "mae_c": float(mean_absolute_error(observed, predicted)),
        "rmse_c": float(np.sqrt(mean_squared_error(observed, predicted))),
        "r2": float(r2_score(observed, predicted)),
        "correlation": float(corr),
        "by_depth": [
            {
                "depth_m": float(depth),
                "mae_c": float(mean_absolute_error(observed[:, i], predicted[:, i])),
                "rmse_c": float(np.sqrt(mean_squared_error(observed[:, i], predicted[:, i]))),
            }
            for i, depth in enumerate(depths_m)
        ],
    }
