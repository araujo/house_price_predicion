"""Tests for model_trainer split, evaluation, and training smoke."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from data_engineer.feature_engineering import (
    get_final_feature_column_names,
    transform_to_model_features,
)
from data_engineer.preprocessing import load_inference_dataframe, load_training_dataframe
from model_trainer import evaluate
from model_trainer.config import load_training_config
from model_trainer.split import train_val_split
from model_trainer.train import run_training, select_best_row


def test_train_val_split_reproducible() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((100, 3)), columns=list("abc"))
    y = pd.Series(rng.standard_normal(100))
    a1, a2, b1, b2 = train_val_split(X, y, test_size=0.2, random_state=42)
    c1, c2, d1, d2 = train_val_split(X, y, test_size=0.2, random_state=42)
    pd.testing.assert_frame_equal(a1.reset_index(drop=True), c1.reset_index(drop=True))
    pd.testing.assert_series_equal(b1.reset_index(drop=True), d1.reset_index(drop=True))


def test_evaluate_regression_metrics() -> None:
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    m = evaluate.evaluate_regression(y_true, y_pred)
    assert m["mae"] == pytest.approx(10.0)
    assert m["rmse"] == pytest.approx(np.sqrt(np.mean([10.0**2, 10.0**2, 10.0**2])))
    assert "r2" in m


def test_metrics_to_mlflow() -> None:
    m = {"mae": 1.0, "rmse": 2.0, "r2": 0.5, "mse": 4.0}
    out = evaluate.metrics_to_mlflow({k: m[k] for k in ("mae", "rmse", "r2")})
    assert out == {"mae": 1.0, "rmse": 2.0, "r2": 0.5}


def test_load_training_config_prefers_mlflow_env_over_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "training:\n  mlflow_tracking_uri: http://from-yaml:5000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HPP_MLFLOW_TRACKING_URI", "http://from-env:5000")
    cfg = load_training_config(yaml_path)
    assert cfg.mlflow_tracking_uri == "http://from-env:5000"

    monkeypatch.delenv("HPP_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow-only:5000")
    cfg2 = load_training_config(yaml_path)
    assert cfg2.mlflow_tracking_uri == "http://mlflow-only:5000"


def test_run_training_smoke_saves_artifacts(tmp_path: Path) -> None:
    out = tmp_path / "artifacts"
    summary = run_training(
        config_path=None,
        raw_data_dir=None,
        output_dir=out,
        use_mlflow=False,
        models_filter=["baseline_knn"],
        max_rows=800,
    )
    assert (out / "best_model.joblib").exists()
    assert (out / "feature_metadata.json").exists()
    assert (out / "training_report.md").exists()
    meta = json.loads((out / "feature_metadata.json").read_text(encoding="utf-8"))
    assert "final_feature_columns" in meta
    assert summary["best"]["model"] == "baseline_knn"


def test_run_training_multiple_models_selects_best(tmp_path: Path) -> None:
    summary = run_training(
        config_path=None,
        raw_data_dir=None,
        output_dir=tmp_path / "m",
        use_mlflow=False,
        models_filter=["baseline_knn", "hist_gradient_boosting"],
        max_rows=1500,
    )
    assert len(summary["rows"]) == 2
    assert summary["best"] == select_best_row(summary["rows"])


def test_select_best_row_tie_break_rmse_then_mae_then_r2() -> None:
    rows = [
        {"key": "a", "model": "m1", "target_mode": "plain", "rmse": 10.0, "mae": 5.0, "r2": 0.5},
        {"key": "b", "model": "m2", "target_mode": "plain", "rmse": 10.0, "mae": 4.0, "r2": 0.4},
        {"key": "c", "model": "m3", "target_mode": "plain", "rmse": 10.0, "mae": 4.0, "r2": 0.6},
    ]
    best = select_best_row(rows)
    assert best["key"] == "c"


def test_training_inference_feature_columns_match_contract() -> None:
    ref = 2015
    xt = transform_to_model_features(load_training_dataframe(), reference_year=ref)
    xi = transform_to_model_features(load_inference_dataframe(), reference_year=ref)
    expected = list(get_final_feature_column_names())
    assert list(xt.columns) == expected
    assert list(xi.columns) == expected


def test_best_model_joblib_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "art"
    run_training(
        config_path=None,
        raw_data_dir=None,
        output_dir=out,
        use_mlflow=False,
        models_filter=["baseline_knn"],
        max_rows=400,
    )
    loaded = joblib.load(out / "best_model.joblib")
    assert hasattr(loaded, "predict")
