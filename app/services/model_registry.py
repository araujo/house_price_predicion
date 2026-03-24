"""Load production sklearn pipeline: MLflow registry first, local artifact second."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import joblib

if TYPE_CHECKING:
    from app.core.config import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedModel:
    """Resolved estimator + registry metadata for API responses."""

    pipeline: Any
    model_name: str
    model_version: str
    source: str  # mlflow_registry | local_artifact


class ModelRegistryService:
    """Lazy-load sklearn ``Pipeline`` / ``TransformedTargetRegressor`` once per process."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._cached: LoadedModel | None = None

    def get(self) -> LoadedModel:
        if self._cached is None:
            self._cached = self._load()
        return self._cached

    def _load(self) -> LoadedModel:
        s = self._settings
        name = s.mlflow_registered_model_name
        tracking_uri = s.mlflow_tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")

        if tracking_uri:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(tracking_uri)
            try:
                uri = f"models:/{name}/latest"
                pipeline = mlflow.sklearn.load_model(uri)
                client = MlflowClient()
                versions = client.get_latest_versions(name, stages=[])
                ver = versions[0].version if versions else "latest"
                logger.info("Loaded sklearn model from MLflow registry %s version %s", name, ver)
                return LoadedModel(
                    pipeline=pipeline,
                    model_name=name,
                    model_version=str(ver),
                    source="mlflow_registry",
                )
            except Exception as exc:
                logger.warning(
                    "MLflow registry load failed; using local fallback if available: %s",
                    exc,
                )

        path = s.local_model_path
        if not path.is_file():
            raise RuntimeError(
                f"Model artifact not found at {path}. Train with "
                "`python -m model_trainer.train` or set HPP_MLFLOW_TRACKING_URI.",
            )
        pipeline = joblib.load(path)
        logger.info("Loaded sklearn model from local path %s", path)
        return LoadedModel(
            pipeline=pipeline,
            model_name=name,
            model_version="local",
            source="local_artifact",
        )
