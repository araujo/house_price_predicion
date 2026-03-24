"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _clear_app_caches() -> None:
    import app.dependencies.containers as containers
    from app.core.config import get_settings

    get_settings.cache_clear()
    containers._prediction_service.cache_clear()
    containers._feature_service.cache_clear()
    containers._model_registry_service.cache_clear()


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def trained_model_path(project_root: Path, tmp_path: Path) -> Generator[Path, None, None]:
    """Train a tiny baseline model into tmp_path for API integration tests."""
    _clear_app_caches()
    from model_trainer.train import run_training

    out = tmp_path / "model_out"
    run_training(
        config_path=None,
        raw_data_dir=project_root / "data" / "raw",
        output_dir=out,
        use_mlflow=False,
        models_filter=["baseline_knn"],
        max_rows=600,
    )
    yield out / "best_model.joblib"
    _clear_app_caches()


@pytest.fixture
def api_client_with_local_model(
    project_root: Path,
    trained_model_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[TestClient, None, None]:
    """TestClient with env pointing at tmp model and project raw data."""
    _clear_app_caches()
    monkeypatch.setenv("HPP_LOCAL_MODEL_PATH", str(trained_model_path))
    monkeypatch.setenv("HPP_RAW_DATA_DIR", str(project_root / "data" / "raw"))
    monkeypatch.setenv("HPP_FEATURE_REFERENCE_YEAR", "2015")
    monkeypatch.delenv("HPP_MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    # Import app after env is set so settings + DI caches pick up env.
    from app.main import app

    with TestClient(app) as client:
        yield client
    _clear_app_caches()
