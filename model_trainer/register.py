"""MLflow model registry helpers."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register_model_from_uri(
    model_uri: str,
    registered_name: str,
    *,
    tracking_uri: str | None = None,
) -> str | None:
    """
    Register a logged model URI (e.g. from ``mlflow.sklearn.log_model`` → ``model_uri``).

    Prefer this over guessing ``runs:/.../path`` so nested runs resolve correctly.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        mv = mlflow.register_model(model_uri=model_uri, name=registered_name)
        logger.info("Registered model %s version %s", registered_name, mv.version)
        return str(mv.version)
    except Exception as exc:
        logger.warning("Model registry unavailable (%s). Skipping registration.", exc)
        return None


def register_model_from_run(
    run_id: str,
    artifact_path: str,
    registered_name: str,
    *,
    tracking_uri: str | None = None,
) -> str | None:
    """Backward-compatible wrapper using ``runs:/id/path``."""
    model_uri = f"runs:/{run_id}/{artifact_path}"
    return register_model_from_uri(model_uri, registered_name, tracking_uri=tracking_uri)


def set_registry_tags(
    run_id: str,
    tags: dict[str, str],
    *,
    tracking_uri: str | None = None,
) -> None:
    """Attach tags to a run (e.g. ``best_model: true``)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    for k, v in tags.items():
        client.set_tag(run_id, k, v)


def ensure_model_dir(path: Path) -> Path:
    """Create local ``model/`` (or custom) directory for artifacts."""
    path.mkdir(parents=True, exist_ok=True)
    return path
