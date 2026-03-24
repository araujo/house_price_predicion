"""Training configuration (YAML + defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from model_trainer.pipelines import EstimatorName, TargetMode


@dataclass
class ModelRunConfig:
    name: EstimatorName
    target_mode: TargetMode = "plain"


@dataclass
class TrainingConfig:
    random_state: int = 42
    test_size: float = 0.2
    reference_year: int = 2015
    experiment_name: str = "house_price"
    registered_model_name: str = "house_price_regressor"
    cv_folds: int = 0
    mlflow_tracking_uri: str | None = None
    artifact_subdir: str = "model"
    models: list[ModelRunConfig] = field(default_factory=list)


def default_training_config() -> TrainingConfig:
    """Default candidate grid: KNN baseline + tree models with log1p where useful."""
    return TrainingConfig(
        models=[
            ModelRunConfig(name="baseline_knn", target_mode="plain"),
            ModelRunConfig(name="hist_gradient_boosting", target_mode="log1p"),
            ModelRunConfig(name="random_forest", target_mode="log1p"),
            ModelRunConfig(name="extra_trees", target_mode="log1p"),
        ],
    )


def load_training_config(path: Path | str | None) -> TrainingConfig:
    """Load YAML config; merge onto :func:`default_training_config`."""
    base = default_training_config()
    if path is None:
        return _apply_mlflow_env_override(base)
    p = Path(path)
    if not p.exists():
        return _apply_mlflow_env_override(base)
    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    t = raw.get("training", raw)
    if not t:
        return _apply_mlflow_env_override(base)
    models_raw = t.get("models")
    models: list[ModelRunConfig] = base.models
    if isinstance(models_raw, list):
        models = [
            ModelRunConfig(name=m["name"], target_mode=m.get("target_mode", "plain"))
            for m in models_raw
            if isinstance(m, dict) and "name" in m
        ]
    cfg = TrainingConfig(
        random_state=int(t.get("random_state", base.random_state)),
        test_size=float(t.get("test_size", base.test_size)),
        reference_year=int(t.get("reference_year", base.reference_year)),
        experiment_name=str(t.get("experiment_name", base.experiment_name)),
        registered_model_name=str(t.get("registered_model_name", base.registered_model_name)),
        cv_folds=int(t.get("cv_folds", base.cv_folds)),
        mlflow_tracking_uri=t.get("mlflow_tracking_uri", base.mlflow_tracking_uri),
        artifact_subdir=str(t.get("artifact_subdir", base.artifact_subdir)),
        models=models or base.models,
    )
    return _apply_mlflow_env_override(cfg)


def _apply_mlflow_env_override(cfg: TrainingConfig) -> TrainingConfig:
    """Prefer HPP_MLFLOW_TRACKING_URI / MLFLOW_TRACKING_URI over YAML (e.g. Compose)."""
    env_uri = os.environ.get("HPP_MLFLOW_TRACKING_URI") or os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri and env_uri.strip():
        return replace(cfg, mlflow_tracking_uri=env_uri.strip())
    return cfg
