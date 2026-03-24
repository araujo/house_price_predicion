"""Path and flag defaults for Airflow tasks (env-overridable, local-dev friendly)."""

from __future__ import annotations

import os
from pathlib import Path


def raw_data_dir() -> Path:
    return Path(os.environ.get("HPP_RAW_DATA_DIR", "data/raw"))


def model_output_dir() -> Path:
    return Path(os.environ.get("HPP_MODEL_OUTPUT_DIR", "model"))


def training_config_path() -> Path | None:
    p = os.environ.get("HPP_TRAINING_CONFIG")
    if not p:
        default = Path("config/model_config.yaml")
        return default if default.exists() else None
    return Path(p)


def use_mlflow() -> bool:
    return os.environ.get("HPP_USE_MLFLOW", "true").lower() in ("1", "true", "yes")


def batch_output_path() -> Path:
    return Path(os.environ.get("HPP_BATCH_OUTPUT", "data/processed/batch_predictions.csv"))


def monitoring_output_dir() -> Path:
    return Path(os.environ.get("HPP_MONITORING_OUTPUT_DIR", "reports/monitoring"))


def training_baseline_stats_path() -> Path:
    """Optional JSON of column means/stds for lightweight drift checks."""
    default = "reports/training_baseline_stats.json"
    return Path(os.environ.get("HPP_TRAINING_BASELINE_STATS", default))


def feature_reference_year() -> int:
    return int(os.environ.get("HPP_FEATURE_REFERENCE_YEAR", "2015"))


def batch_predictions_for_monitoring() -> Path:
    """Batch predictions CSV (explicit env or default batch output path)."""
    p = os.environ.get("HPP_BATCH_PREDICTIONS_PATH")
    return Path(p) if p else batch_output_path()


def monitor_reports_dir() -> Path:
    return Path(os.environ.get("HPP_MONITOR_REPORTS_DIR", "monitor/reports"))


def monitoring_labels_path() -> Path | None:
    p = os.environ.get("HPP_MONITORING_LABELS_PATH")
    return Path(p) if p else None
