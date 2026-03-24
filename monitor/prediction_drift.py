"""
Drift in prediction (or score) distribution vs a baseline reference sample.

Typical use: ``baseline_reference`` = training ``price`` (proxy for expected scale),
``current_predictions`` = batch model outputs. Large KS shift or mean shift ⇒ review model/data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def compute_prediction_drift_report(
    current_predictions: pd.Series | np.ndarray,
    baseline_reference: pd.Series | np.ndarray,
    *,
    ks_pvalue_threshold: float = 0.01,
    mean_relative_shift_threshold: float = 0.25,
) -> dict[str, Any]:
    """
    Statistical comparison of prediction values vs baseline (e.g. historical training target).

    Flags drift if the two-sample KS test is significant **or** relative mean shift exceeds
    ``mean_relative_shift_threshold``.
    """
    cur = pd.to_numeric(pd.Series(current_predictions), errors="coerce")
    cur = cur.dropna().to_numpy(dtype=np.float64)
    base = pd.to_numeric(pd.Series(baseline_reference), errors="coerce")
    base = base.dropna().to_numpy(dtype=np.float64)
    if len(cur) < 2 or len(base) < 2:
        return {
            "kind": "prediction_drift",
            "skipped": True,
            "reason": "insufficient_samples",
            "n_current": int(len(cur)),
            "n_baseline": int(len(base)),
        }

    ks = ks_2samp(base, cur)
    mean_base = float(np.mean(base))
    mean_cur = float(np.mean(cur))
    rel_shift = abs(mean_cur - mean_base) / (abs(mean_base) + 1e-9)
    ks_drift = bool(ks.pvalue < ks_pvalue_threshold)
    mean_drift = bool(rel_shift > mean_relative_shift_threshold)
    drift = ks_drift or mean_drift

    return {
        "kind": "prediction_drift",
        "skipped": False,
        "summary_stats": {
            "baseline_mean": mean_base,
            "baseline_std": float(np.std(base)),
            "current_mean": mean_cur,
            "current_std": float(np.std(cur)),
            "relative_mean_shift": float(rel_shift),
        },
        "ks_statistic": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
        "drift": drift,
        "drift_reason": _pred_drift_reason(
            ks_drift,
            mean_drift,
            ks_pvalue_threshold,
            mean_relative_shift_threshold,
        ),
        "n_baseline": int(len(base)),
        "n_current": int(len(cur)),
    }


def _pred_drift_reason(
    ks_drift: bool,
    mean_drift: bool,
    ks_thr: float,
    mean_thr: float,
) -> str:
    parts: list[str] = []
    if ks_drift:
        parts.append(f"ks_pvalue<{ks_thr}")
    if mean_drift:
        parts.append(f"mean_shift>{mean_thr}")
    return "; ".join(parts) if parts else "none"


def summarize_prediction_series(predictions: pd.Series | np.ndarray) -> dict[str, float]:
    """Lightweight stats for logging (no drift test)."""
    s = pd.to_numeric(pd.Series(predictions), errors="coerce").dropna()
    if s.empty:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
    }
