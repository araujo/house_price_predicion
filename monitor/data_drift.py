"""
Numeric feature distribution drift: Kolmogorov–Smirnov two-sample test and optional PSI.

Designed for ``reference_df`` (e.g. training-period rows) vs ``current_df`` (e.g. inference rows).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _clean_numeric(series: pd.Series) -> np.ndarray:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=np.float64)
    return s


def compute_psi(expected: np.ndarray, actual: np.ndarray, *, n_bins: int = 10) -> float:
    """
    Population Stability Index using quantile bins from the expected (reference) sample.

    Common rule of thumb: PSI < 0.1 stable, 0.1–0.25 moderate, > 0.25 high drift.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return float("nan")
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 2:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=qs)
    a_counts, _ = np.histogram(actual, bins=qs)
    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    eps = 1e-6
    psi = float(np.sum((a_pct - e_pct) * np.log((a_pct + eps) / (e_pct + eps))))
    return psi


def compute_data_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    ks_pvalue_threshold: float = 0.01,
    psi_alert_threshold: float = 0.25,
    min_samples: int = 20,
    include_psi: bool = True,
) -> dict[str, Any]:
    """
    Compare numeric distributions per feature using KS (and optionally PSI).

    Drift flag per feature: ``ks_pvalue < ks_pvalue_threshold`` **or**
    (if PSI computed) ``psi > psi_alert_threshold``.

    Returns a JSON-serializable dict with ``features`` list and ``summary``.
    """
    if feature_columns is None:
        feature_columns = infer_common_numeric_columns(reference_df, current_df)

    rows: list[dict[str, Any]] = []
    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            rows.append(
                {
                    "feature": col,
                    "skipped": True,
                    "reason": "missing_in_one_frame",
                },
            )
            continue
        ref_a = _clean_numeric(reference_df[col])
        cur_a = _clean_numeric(current_df[col])
        if len(ref_a) < min_samples or len(cur_a) < min_samples:
            rows.append(
                {
                    "feature": col,
                    "skipped": True,
                    "reason": "insufficient_samples",
                    "n_reference": int(len(ref_a)),
                    "n_current": int(len(cur_a)),
                },
            )
            continue
        ks = ks_2samp(ref_a, cur_a)
        psi_val: float | None = None
        if include_psi:
            psi_val = compute_psi(ref_a, cur_a)
        ks_drift = bool(ks.pvalue < ks_pvalue_threshold)
        psi_drift = bool(
            include_psi and not math.isnan(psi_val) and psi_val > psi_alert_threshold,
        )
        rows.append(
            {
                "feature": col,
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "psi": None if psi_val is None or math.isnan(psi_val) else float(psi_val),
                "drift": ks_drift or psi_drift,
                "drift_reason": _drift_reason(
                    ks_drift,
                    psi_drift,
                    ks_pvalue_threshold,
                    psi_alert_threshold,
                ),
                "n_reference": int(len(ref_a)),
                "n_current": int(len(cur_a)),
            },
        )

    evaluated = [r for r in rows if not r.get("skipped")]
    drifted = [r for r in evaluated if r.get("drift")]
    return {
        "kind": "data_drift",
        "features": rows,
        "summary": {
            "n_features_evaluated": len(evaluated),
            "n_features_drift_flagged": len(drifted),
            "drift": len(drifted) > 0,
            "ks_pvalue_threshold": ks_pvalue_threshold,
            "psi_alert_threshold": psi_alert_threshold,
        },
    }


def infer_common_numeric_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    *,
    exclude: frozenset[str] | None = None,
) -> list[str]:
    """Columns present in both frames, numeric dtype, excluding metadata targets."""
    ex = exclude or frozenset({"id", "date", "price"})
    common = [c for c in reference_df.columns if c in current_df.columns and c not in ex]
    return [c for c in common if pd.api.types.is_numeric_dtype(reference_df[c])]


def _drift_reason(
    ks_drift: bool,
    psi_drift: bool,
    ks_thr: float,
    psi_thr: float,
) -> str:
    if ks_drift and psi_drift:
        return f"ks_pvalue<{ks_thr} and psi>{psi_thr}"
    if ks_drift:
        return f"ks_pvalue<{ks_thr}"
    if psi_drift:
        return f"psi>{psi_thr}"
    return "none"
