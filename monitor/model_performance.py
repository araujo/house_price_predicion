"""
Model performance metrics when labels exist; otherwise a structured skip report.

Extend later with business metrics, calibration, or slice analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def compute_model_performance_report(
    y_true: pd.Series | np.ndarray | None,
    y_pred: pd.Series | np.ndarray | None,
) -> dict[str, Any]:
    """
    If both vectors are present and non-empty, return MAE / RMSE / count.

    If either is missing, return ``status: skipped`` with a reason (alerting can key off this).
    """
    if y_true is None or y_pred is None:
        return {
            "kind": "model_performance",
            "status": "skipped",
            "reason": "y_true_or_y_pred_not_provided",
        }
    yt = pd.to_numeric(pd.Series(y_true), errors="coerce")
    yp = pd.to_numeric(pd.Series(y_pred), errors="coerce")
    mask = yt.notna() & yp.notna()
    yt = yt[mask]
    yp = yp[mask]
    if len(yt) == 0:
        return {
            "kind": "model_performance",
            "status": "skipped",
            "reason": "no_aligned_non_null_pairs",
        }

    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    return {
        "kind": "model_performance",
        "status": "ok",
        "n_samples": int(len(yt)),
        "mae": mae,
        "rmse": rmse,
    }


def extract_truth_and_pred_columns(
    df: pd.DataFrame,
    *,
    truth_candidates: tuple[str, ...] = ("actual_price", "price", "y_true", "ground_truth"),
    pred_candidates: tuple[str, ...] = ("predicted_price", "y_pred", "prediction"),
) -> tuple[pd.Series | None, pd.Series | None]:
    """Resolve common column names for labeled evaluation rows."""
    y_true = next((df[c] for c in truth_candidates if c in df.columns), None)
    y_pred = next((df[c] for c in pred_candidates if c in df.columns), None)
    return y_true, y_pred
