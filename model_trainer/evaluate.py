"""Regression metrics on the original price scale (explicit, no hidden transforms)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(
    y_true: Any,
    y_pred: Any,
    *,
    sample_weight: Any | None = None,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, and R² on the same scale as ``y_true`` / ``y_pred``.

    Callers must pass predictions already inverse-transformed to price units
    (e.g. ``TransformedTargetRegressor.predict`` handles this for log targets).
    """
    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    mae = float(mean_absolute_error(yt, yp, sample_weight=sample_weight))
    mse = float(mean_squared_error(yt, yp, sample_weight=sample_weight))
    rmse = float(math.sqrt(mse))
    r2 = float(r2_score(yt, yp, sample_weight=sample_weight))
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


def metrics_to_mlflow(metrics: dict[str, float]) -> dict[str, float]:
    """Float-only dict safe for ``mlflow.log_metrics``."""
    return {k: float(v) for k, v in metrics.items()}


def format_metrics_line(metrics: dict[str, Any]) -> str:
    """Human-readable one-line summary."""
    return (
        f"MAE={metrics.get('mae', float('nan')):.4f} "
        f"RMSE={metrics.get('rmse', float('nan')):.4f} "
        f"R2={metrics.get('r2', float('nan')):.4f}"
    )
