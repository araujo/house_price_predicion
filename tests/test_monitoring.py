"""Tests for monitor package (drift + performance helpers)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from monitor.data_drift import (
    compute_data_drift_report,
    compute_psi,
    infer_common_numeric_columns,
)
from monitor.model_performance import (
    compute_model_performance_report,
    extract_truth_and_pred_columns,
)
from monitor.prediction_drift import compute_prediction_drift_report


def test_compute_psi_stable_distribution() -> None:
    rng = np.random.default_rng(42)
    expected = rng.normal(size=1000)
    actual = rng.normal(size=1000)
    psi = compute_psi(expected, actual)
    assert psi < 0.25


def test_data_drift_same_distribution_low_flags() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(size=400)})
    cur = pd.DataFrame({"x": rng.normal(size=400)})
    r = compute_data_drift_report(
        ref,
        cur,
        feature_columns=["x"],
        ks_pvalue_threshold=0.01,
        min_samples=30,
    )
    assert r["summary"]["n_features_evaluated"] == 1
    assert r["summary"]["n_features_drift_flagged"] == 0
    assert r["summary"]["drift"] is False


def test_data_drift_shifted_feature_flagged() -> None:
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(size=400)})
    cur = pd.DataFrame({"x": rng.normal(size=400, loc=6.0)})
    r = compute_data_drift_report(
        ref,
        cur,
        feature_columns=["x"],
        ks_pvalue_threshold=0.01,
        min_samples=30,
    )
    assert r["summary"]["n_features_drift_flagged"] >= 1
    assert r["summary"]["drift"] is True


def test_prediction_drift_detects_separated_distributions() -> None:
    rng = np.random.default_rng(99)
    baseline = pd.Series(rng.normal(loc=0.0, scale=1.0, size=2000))
    current = pd.Series(rng.normal(loc=12.0, scale=1.0, size=500))
    r = compute_prediction_drift_report(
        current,
        baseline,
        ks_pvalue_threshold=0.01,
        mean_relative_shift_threshold=0.05,
    )
    assert r["skipped"] is False
    assert r["drift"] is True


def test_prediction_drift_report_structure() -> None:
    rng = np.random.default_rng(2)
    baseline = pd.Series(rng.lognormal(mean=12, sigma=0.4, size=500))
    current = pd.Series(rng.lognormal(mean=12, sigma=0.4, size=200))
    r = compute_prediction_drift_report(current, baseline, ks_pvalue_threshold=0.001)
    assert r["kind"] == "prediction_drift"
    assert r["skipped"] is False
    assert "summary_stats" in r
    assert "ks_pvalue" in r
    assert "drift" in r


def test_model_performance_ok() -> None:
    y_true = pd.Series([100.0, 200.0, 300.0])
    y_pred = pd.Series([102.0, 198.0, 305.0])
    r = compute_model_performance_report(y_true, y_pred)
    assert r["status"] == "ok"
    assert r["n_samples"] == 3
    assert r["mae"] > 0
    assert r["rmse"] > 0


def test_model_performance_skipped_without_labels() -> None:
    r = compute_model_performance_report(None, pd.Series([1.0]))
    assert r["status"] == "skipped"
    assert r["reason"] == "y_true_or_y_pred_not_provided"


def test_model_performance_skipped_when_no_aligned_pairs() -> None:
    y_true = pd.Series([1.0, np.nan])
    y_pred = pd.Series([np.nan, 2.0])
    r = compute_model_performance_report(y_true, y_pred)
    assert r["status"] == "skipped"
    assert r["reason"] == "no_aligned_non_null_pairs"


def test_infer_common_numeric_columns_excludes_metadata() -> None:
    ref = pd.DataFrame({"price": [1.0, 2.0], "sqft_living": [1000, 2000], "id": [1, 2]})
    cur = pd.DataFrame({"price": [1.0, 2.0], "sqft_living": [1100, 2100], "id": [1, 2]})
    cols = infer_common_numeric_columns(ref, cur, exclude=frozenset({"id", "price"}))
    assert "sqft_living" in cols
    assert "id" not in cols


def test_write_unified_monitoring_report_structure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from airflow_tasks.monitoring_checks import write_unified_monitoring_report

    rep = tmp_path / "mon"
    md = tmp_path / "md"
    monkeypatch.setenv("HPP_MONITOR_REPORTS_DIR", str(rep))
    monkeypatch.setenv("HPP_MONITORING_OUTPUT_DIR", str(md))

    out = write_unified_monitoring_report(
        {"ok": True, "n_kc_rows": 10},
        {"kind": "data_drift", "summary": {"drift": False, "n_features_evaluated": 3}},
        {"kind": "prediction_drift", "drift": False, "skipped": False, "ks_pvalue": 0.5},
        {"kind": "model_performance", "status": "skipped"},
    )
    assert out["report_version"] == "1.0"
    assert "timestamp_utc" in out
    assert "artifacts" in out
    js = json.loads((rep / "monitoring_latest.json").read_text(encoding="utf-8"))
    assert js["data_drift"]["summary"]["n_features_evaluated"] == 3
    assert (md / "monitoring_summary.md").is_file()


def test_extract_truth_and_pred_columns() -> None:
    df = pd.DataFrame({"actual_price": [1, 2], "predicted_price": [1.1, 1.9]})
    yt, yp = extract_truth_and_pred_columns(df)
    assert yt is not None and yp is not None
    assert len(yt) == 2
