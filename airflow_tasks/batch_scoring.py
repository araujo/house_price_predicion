"""
Batch inference — chained steps with CSV handoffs.

Reuses Phase 2 merge/validation, ``transform_to_model_features`` (Phase 3), and
``app.services.model_registry.ModelRegistryService`` (same load policy as the API).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
from app.core.config import Settings
from app.services.model_registry import ModelRegistryService
from data_engineer.feature_engineering import (
    ZIPCODE_MODEL_CSV_DTYPE,
    prepare_model_input_for_prediction,
    transform_to_model_features,
)
from data_engineer.ingestion import (
    load_future_unseen_examples_dataframe,
    load_zipcode_demographics_dataframe,
)
from data_engineer.preprocessing import merge_demographics_by_zipcode
from data_engineer.validation import (
    run_inference_pipeline_validations,
    validate_inference_feature_presence_after_merge,
)

from airflow_tasks import config

logger = logging.getLogger(__name__)


def _scratch_dir() -> Path:
    p = Path(os.environ.get("HPP_AIRFLOW_SCRATCH", "data/processed/airflow_scratch"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def validate_inference_rows() -> dict[str, Any]:
    """Load ``future_unseen_examples.csv`` and run inference validation (schema, nulls)."""
    raw = config.raw_data_dir()
    logger.info("Validating inference rows from %s", raw.resolve())
    inference = load_future_unseen_examples_dataframe(raw)
    rep = run_inference_pipeline_validations(inference)
    if not rep.ok:
        raise ValueError("inference validation failed: " + "; ".join(rep.errors))
    for w in rep.warnings:
        logger.warning("Inference validation: %s", w)
    return {"n_rows": len(inference), "warnings": list(rep.warnings)}


def merge_inference_with_demographics() -> str:
    """Merge zipcode demographics onto inference rows; validate merged columns."""
    raw = config.raw_data_dir()
    inference = load_future_unseen_examples_dataframe(raw)
    demo = load_zipcode_demographics_dataframe(raw)
    merged = merge_demographics_by_zipcode(inference, demo)
    rep = validate_inference_feature_presence_after_merge(merged)
    if not rep.ok:
        raise ValueError("merged inference validation failed: " + "; ".join(rep.errors))
    path = _scratch_dir() / "batch_merged.csv"
    merged.to_csv(path, index=False)
    logger.info("Wrote merged inference frame: %s rows -> %s", len(merged), path)
    return str(path.resolve())


def engineer_batch_features(merged_csv_path: str) -> str:
    """Phase 3 feature matrix (same transform as training / API)."""
    merged = pd.read_csv(merged_csv_path, dtype=ZIPCODE_MODEL_CSV_DTYPE)
    year = config.feature_reference_year()
    X = transform_to_model_features(
        merged,
        reference_year=year,
        strip_metadata=False,
        fill_demographic_na=True,
    )
    path = _scratch_dir() / "batch_X.csv"
    X.to_csv(path, index=False)
    logger.info("Feature matrix: %s rows %s cols -> %s", len(X), X.shape[1], path)
    return str(path.resolve())


def load_model_score_and_write(merged_csv_path: str, feature_csv_path: str) -> dict[str, Any]:
    """Load registry/local model, predict, write batch output CSV."""
    merged = pd.read_csv(merged_csv_path, dtype=ZIPCODE_MODEL_CSV_DTYPE)
    X = pd.read_csv(feature_csv_path, dtype=ZIPCODE_MODEL_CSV_DTYPE)
    raw = config.raw_data_dir()
    out = config.batch_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    year = config.feature_reference_year()

    local_default = config.model_output_dir() / "best_model.joblib"
    local_path = Path(os.environ.get("HPP_LOCAL_MODEL_PATH", str(local_default)))

    settings_kw: dict[str, Any] = {
        "raw_data_dir": raw,
        "feature_reference_year": year,
        "local_model_path": local_path,
    }
    if os.environ.get("HPP_MLFLOW_TRACKING_URI"):
        settings_kw["mlflow_tracking_uri"] = os.environ["HPP_MLFLOW_TRACKING_URI"]
    if os.environ.get("HPP_MLFLOW_MODEL_NAME"):
        settings_kw["mlflow_registered_model_name"] = os.environ["HPP_MLFLOW_MODEL_NAME"]
    settings = Settings(**settings_kw)
    reg = ModelRegistryService(settings)
    loaded = reg.get()
    X = prepare_model_input_for_prediction(X, logger=logger)
    preds = loaded.pipeline.predict(X)
    df_out = pd.DataFrame({"predicted_price": preds.ravel()}, index=merged.index)
    if "zipcode" in merged.columns:
        df_out.insert(0, "zipcode", merged["zipcode"].astype(str).values)

    if out.suffix.lower() == ".parquet":
        df_out.to_parquet(out, index=False)
    else:
        df_out.to_csv(out, index=False)

    result = {
        "n_rows": int(len(df_out)),
        "output_path": str(out.resolve()),
        "model_name": loaded.model_name,
        "model_version": loaded.model_version,
        "model_source": loaded.source,
    }
    logger.info(
        "Batch scoring complete: rows=%s output=%s model=%s",
        result["n_rows"],
        result["output_path"],
        result["model_source"],
    )
    return result


def summarize_batch_run(score_result: dict[str, Any]) -> dict[str, Any]:
    """Structured log line for operators."""
    logger.info(
        "Batch run summary: predictions=%s path=%s",
        score_result.get("n_rows"),
        score_result.get("output_path"),
    )
    return score_result
