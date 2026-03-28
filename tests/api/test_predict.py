"""Prediction route tests."""

from __future__ import annotations

from pathlib import Path

import app.dependencies.containers as containers
import pandas as pd
import pytest
from app.core.config import get_settings
from app.main import app
from app.schemas.prediction import _EXAMPLE_FULL, _EXAMPLE_MINIMAL
from data_engineer.constants import FUTURE_UNSEEN_FILENAME
from data_engineer.feature_engineering import get_final_feature_column_names
from fastapi.testclient import TestClient

FULL_PATH = "/api/v1/predict/full"
MINIMAL_PATH = "/api/v1/predict/minimal"


def test_predict_full_validation_error() -> None:
    client = TestClient(app)
    r = client.post(FULL_PATH, json={"rows": [{"bedrooms": 1}]})
    assert r.status_code == 422


def test_predict_full_accepts_future_unseen_csv_row(
    api_client_with_local_model: TestClient,
    project_root: Path,
) -> None:
    """Payload aligned with ``data/raw/future_unseen_examples.csv`` (no demographics)."""
    path = project_root / "data" / "raw" / FUTURE_UNSEEN_FILENAME
    row = pd.read_csv(path, nrows=1).iloc[0].to_dict()
    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
    row["zipcode"] = str(int(row["zipcode"]))
    payload = {"rows": [row]}
    r = api_client_with_local_model.post(FULL_PATH, json=payload)
    assert r.status_code == 200, r.text
    assert len(r.json()["predictions"]) == 1


def test_predict_full_batch_shape_and_contract(
    api_client_with_local_model: TestClient,
) -> None:
    payload = {"rows": [_EXAMPLE_FULL, _EXAMPLE_FULL]}
    r = api_client_with_local_model.post(FULL_PATH, json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 2
    assert body["meta"]["n_features"] > 0
    assert len(body["meta"]["feature_columns"]) == body["meta"]["n_features"]
    assert body["model_source"] == "local_artifact"


def test_predict_minimal_matches_full_contract(
    api_client_with_local_model: TestClient,
) -> None:
    r = api_client_with_local_model.post(MINIMAL_PATH, json={"rows": [_EXAMPLE_MINIMAL]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["predictions"]) == 1
    cols = body["meta"]["feature_columns"]
    assert cols == list(get_final_feature_column_names())


def test_predict_minimal_extra_field_rejected(api_client_with_local_model: TestClient) -> None:
    bad = {**_EXAMPLE_MINIMAL, "extra": 1}
    r = api_client_with_local_model.post(MINIMAL_PATH, json={"rows": [bad]})
    assert r.status_code == 422


def test_model_missing_returns_503(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HPP_LOCAL_MODEL_PATH", str(tmp_path / "nonexistent.joblib"))
    monkeypatch.delenv("HPP_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    get_settings.cache_clear()
    containers._prediction_service.cache_clear()
    containers._feature_service.cache_clear()
    containers._model_registry_service.cache_clear()

    client = TestClient(app)
    r = client.post(FULL_PATH, json={"rows": [_EXAMPLE_FULL]})
    assert r.status_code == 503

    get_settings.cache_clear()
    containers._prediction_service.cache_clear()
    containers._feature_service.cache_clear()
    containers._model_registry_service.cache_clear()
