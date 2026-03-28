"""Training DAG steps — call ``data_engineer`` validation and ``model_trainer.train``."""

from __future__ import annotations

import logging
from typing import Any

from data_engineer.ingestion import load_kc_house_dataframe, load_zipcode_demographics_dataframe
from data_engineer.validation import run_training_pipeline_validations
from model_trainer.train import run_training

from airflow_tasks import config

logger = logging.getLogger(__name__)


def validate_training_raw_data() -> dict[str, Any]:
    """
    Validate raw KC house + zipcode demographics before training.

    Raises if validation errors exist so downstream tasks are skipped.
    """
    raw = config.raw_data_dir()
    logger.info("Validating raw data under %s", raw.resolve())
    kc = load_kc_house_dataframe(raw)
    demo = load_zipcode_demographics_dataframe(raw)
    rep = run_training_pipeline_validations(kc, demo)
    if not rep.ok:
        msg = "; ".join(rep.errors)
        logger.error("Raw data validation failed: %s", msg)
        raise ValueError(f"raw data validation failed: {msg}")
    for w in rep.warnings:
        logger.warning("Validation warning: %s", w)
    summary = {
        "ok": True,
        "n_kc_rows": int(len(kc)),
        "n_demo_rows": int(len(demo)),
        "warnings": list(rep.warnings),
    }
    logger.info(
        "Raw data OK: kc_rows=%s demo_rows=%s",
        summary["n_kc_rows"],
        summary["n_demo_rows"],
    )
    return summary


def execute_training() -> dict[str, Any]:
    """
    Load, preprocess, engineer features, train, evaluate, register best in MLflow, save artifacts.

    Delegates to :func:`model_trainer.train.run_training`.
    """
    raw = config.raw_data_dir()
    out = config.model_output_dir()
    cfg = config.training_config_path()
    use_mf = config.use_mlflow()
    logger.info(
        "Starting training raw=%s out=%s config=%s mlflow=%s",
        raw,
        out,
        cfg,
        use_mf,
    )
    result = run_training(
        config_path=cfg,
        raw_data_dir=raw,
        output_dir=out,
        use_mlflow=use_mf,
    )
    best = result.get("best", {})
    logger.info(
        "Training finished: best_model=%s rmse=%s output_dir=%s",
        result.get("best_key"),
        best.get("rmse"),
        result.get("output_dir"),
    )
    return result


def summarize_training_run(train_result: dict[str, Any]) -> dict[str, Any]:
    """Log a short summary for operators (metrics already in MLflow and training_report.md)."""
    best = train_result.get("best") or {}
    summary = {
        "best_key": train_result.get("best_key"),
        "best_rmse": best.get("rmse"),
        "best_mae": best.get("mae"),
        "best_r2": best.get("r2"),
        "output_dir": train_result.get("output_dir"),
    }
    logger.info(
        "Run summary: best_key=%s rmse=%s mae=%s r2=%s artifacts=%s",
        summary["best_key"],
        summary["best_rmse"],
        summary["best_mae"],
        summary["best_r2"],
        summary["output_dir"],
    )
    return summary
