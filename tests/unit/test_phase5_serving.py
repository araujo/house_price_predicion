"""Phase 5: serving path parity with training (no train/serve skew)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from app.core.config import Settings
from app.schemas.prediction import HouseRowFull
from app.services.feature_service import FeatureService
from app.services.model_registry import ModelRegistryService
from data_engineer.constants import FUTURE_UNSEEN_FILENAME
from data_engineer.feature_engineering import transform_to_model_features
from data_engineer.ingestion import load_zipcode_demographics_dataframe
from data_engineer.preprocessing import merge_demographics_by_zipcode
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline


def test_enrich_and_transform_matches_training_merge_and_transform(project_root: Path) -> None:
    """API feature path must match explicit merge + Phase 3 transform used in training."""
    raw = project_root / "data" / "raw"
    s = Settings(raw_data_dir=raw, feature_reference_year=2015)
    fs = FeatureService(s)
    fut = pd.read_csv(raw / FUTURE_UNSEEN_FILENAME, nrows=1)
    row = fut.iloc[0].to_dict()
    row["zipcode"] = str(int(row["zipcode"]))
    hr = HouseRowFull.model_validate(row)
    house = fs.dataframe_from_full_rows([hr])
    demo = load_zipcode_demographics_dataframe(raw)
    merged_ref = merge_demographics_by_zipcode(house, demo)
    x_ref = transform_to_model_features(
        merged_ref,
        reference_year=2015,
        strip_metadata=False,
        fill_demographic_na=True,
    )
    x_api = fs.enrich_and_transform(house)
    pd.testing.assert_frame_equal(x_api, x_ref)


def test_model_registry_prefers_mlflow_when_uri_set_and_load_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HPP_MLFLOW_TRACKING_URI", "file:./mlruns")
    monkeypatch.setenv("HPP_LOCAL_MODEL_PATH", str(tmp_path / "not_used.joblib"))
    dummy = Pipeline([("m", DummyRegressor(strategy="constant", constant=123.0))])

    class _Ver:
        version = "42"

    with (
        patch("mlflow.sklearn.load_model", return_value=dummy) as load_m,
        patch("mlflow.tracking.MlflowClient") as client_cls,
    ):
        client_cls.return_value.get_latest_versions.return_value = [_Ver()]
        svc = ModelRegistryService(Settings())
        lm = svc.get()
        assert lm.source == "mlflow_registry"
        assert lm.model_version == "42"
        assert lm.pipeline is dummy
        load_m.assert_called_once()


def test_model_registry_falls_back_to_local_when_mlflow_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import joblib

    monkeypatch.setenv("HPP_MLFLOW_TRACKING_URI", "file:./mlruns")
    path = tmp_path / "local.joblib"
    dummy = Pipeline([("m", DummyRegressor(strategy="constant", constant=99.0))])
    joblib.dump(dummy, path)
    monkeypatch.setenv("HPP_LOCAL_MODEL_PATH", str(path))

    with patch("mlflow.sklearn.load_model", side_effect=RuntimeError("registry empty")):
        svc = ModelRegistryService(Settings())
        lm = svc.get()
        assert lm.source == "local_artifact"
        assert lm.pipeline is not None


def test_log1p_wrapped_estimator_predict_applies_inverse_in_dollar_space() -> None:
    """Serving calls ``.predict()``; ``TransformedTargetRegressor`` applies ``expm1`` for us."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    y_dollars = rng.uniform(50_000, 800_000, size=20)
    inner = Pipeline([("m", DummyRegressor(strategy="mean"))])
    ttr = TransformedTargetRegressor(regressor=inner, func=np.log1p, inverse_func=np.expm1)
    ttr.fit(X, y_dollars)
    pred = ttr.predict(X[:3])
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0)
