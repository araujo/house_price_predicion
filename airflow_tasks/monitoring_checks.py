"""
Airflow monitoring tasks — delegate to ``monitor.*`` and ``data_engineer`` loaders.

DAG files stay thin; business logic lives in ``monitor/``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from data_engineer.ingestion import load_kc_house_dataframe, load_zipcode_demographics_dataframe
from data_engineer.preprocessing import load_inference_dataframe, load_training_dataframe
from data_engineer.validation import run_training_pipeline_validations
from monitor.data_drift import compute_data_drift_report, infer_common_numeric_columns
from monitor.model_performance import (
    compute_model_performance_report,
    extract_truth_and_pred_columns,
)
from monitor.prediction_drift import compute_prediction_drift_report

from airflow_tasks import config

logger = logging.getLogger(__name__)


def run_schema_and_quality_checks() -> dict[str, Any]:
    """Schema + null + duplicate checks on raw training inputs."""
    raw = config.raw_data_dir()
    kc = load_kc_house_dataframe(raw)
    demo = load_zipcode_demographics_dataframe(raw)
    rep = run_training_pipeline_validations(kc, demo)
    summary = {
        "ok": rep.ok,
        "errors": list(rep.errors),
        "warnings": list(rep.warnings),
        "n_kc_rows": len(kc),
        "n_demo_rows": len(demo),
    }
    if not rep.ok:
        logger.error("Schema/quality checks failed: %s", rep.errors)
    else:
        logger.info("Schema/quality checks passed (warnings=%s)", len(rep.warnings))
    return summary


def run_data_drift_monitoring(quality_summary: dict[str, Any]) -> dict[str, Any]:
    """KS (+ optional PSI) on numeric columns shared by training vs inference merge."""
    if not quality_summary.get("ok"):
        return {"kind": "data_drift", "skipped": True, "reason": "upstream_quality_failed"}

    raw = config.raw_data_dir()
    logger.info("Data drift: loading training + inference frames from %s", raw)
    ref = load_training_dataframe(raw, validate=True)
    cur = load_inference_dataframe(raw, validate=True)
    cols = infer_common_numeric_columns(ref, cur)
    report = compute_data_drift_report(ref, cur, feature_columns=cols)
    logger.info(
        "Data drift summary: evaluated=%s drift_flagged=%s",
        report["summary"]["n_features_evaluated"],
        report["summary"]["n_features_drift_flagged"],
    )
    return report


def run_prediction_drift_monitoring() -> dict[str, Any]:
    """
    Compare batch ``predicted_price`` to training ``price`` distribution (sanity / shift proxy).

    Requires ``HPP_BATCH_PREDICTIONS_PATH`` or default batch output file to exist.
    """
    path = config.batch_predictions_for_monitoring()
    raw = config.raw_data_dir()
    if not path.is_file():
        logger.warning("Prediction drift skipped: no file at %s", path)
        return {
            "kind": "prediction_drift",
            "skipped": True,
            "reason": "predictions_file_missing",
            "path": str(path),
        }

    preds = pd.read_csv(path)
    if "predicted_price" not in preds.columns:
        logger.warning("Prediction drift skipped: no predicted_price in %s", path)
        return {
            "kind": "prediction_drift",
            "skipped": True,
            "reason": "missing_predicted_price_column",
        }

    ref = load_kc_house_dataframe(raw)
    baseline = ref["price"]
    report = compute_prediction_drift_report(preds["predicted_price"], baseline)
    logger.info(
        "Prediction drift: drift=%s ks_pvalue=%s",
        report.get("drift"),
        report.get("ks_pvalue"),
    )
    return report


def run_model_performance_monitoring() -> dict[str, Any]:
    """
    Labeled metrics when ``HPP_MONITORING_LABELS_PATH`` points to a CSV with truth + prediction.

    Columns resolved via :func:`monitor.model_performance.extract_truth_and_pred_columns`.
    """
    labels_path = config.monitoring_labels_path()
    if labels_path is None or not labels_path.is_file():
        logger.info(
            "Model performance skipped: set HPP_MONITORING_LABELS_PATH to a labeled CSV "
            "(e.g. actual_price + predicted_price).",
        )
        return {
            "kind": "model_performance",
            "status": "skipped",
            "reason": "no_labels_file_configured",
        }

    df = pd.read_csv(labels_path)
    y_true, y_pred = extract_truth_and_pred_columns(df)
    report = compute_model_performance_report(y_true, y_pred)
    if report.get("status") == "ok":
        logger.info(
            "Model performance: MAE=%s RMSE=%s n=%s",
            report.get("mae"),
            report.get("rmse"),
            report.get("n_samples"),
        )
    else:
        logger.info("Model performance: %s", report.get("reason"))
    return report


def persist_training_baseline_stats() -> str:
    """Write mean/std for key columns — optional helper for external drift baselines."""
    raw = config.raw_data_dir()
    kc = load_kc_house_dataframe(raw)
    cols = ["price", "sqft_living", "sqft_lot"]
    present = [c for c in cols if c in kc.columns]
    stats = {c: {"mean": float(kc[c].mean()), "std": float(kc[c].std())} for c in present}
    out = config.training_baseline_stats_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote training baseline stats to %s", out)
    return str(out.resolve())


def write_unified_monitoring_report(
    quality: dict[str, Any],
    data_drift: dict[str, Any],
    prediction_drift: dict[str, Any],
    performance: dict[str, Any],
) -> dict[str, Any]:
    """Merge sections; write JSON and Markdown monitoring artifacts."""
    ts = datetime.now(UTC).isoformat()
    payload: dict[str, Any] = {
        "report_version": "1.0",
        "timestamp_utc": ts,
        "quality": quality,
        "data_drift": data_drift,
        "prediction_drift": prediction_drift,
        "model_performance": performance,
    }

    primary = config.monitor_reports_dir()
    primary.mkdir(parents=True, exist_ok=True)
    json_path = primary / "monitoring_latest.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote unified monitoring JSON to %s", json_path)

    md_path = config.monitoring_output_dir()
    md_path.mkdir(parents=True, exist_ok=True)
    md_file = md_path / "monitoring_summary.md"
    dd_sum = data_drift.get("summary") if isinstance(data_drift, dict) else {}
    pr_sum = prediction_drift.get("summary_stats") if isinstance(prediction_drift, dict) else {}
    lines = [
        "# Monitoring summary",
        "",
        "- Schema: `report_version` 1.0",
        f"- Generated (UTC): `{ts}`",
        "",
        "## Quality",
        f"- ok: {quality.get('ok')}",
        f"- kc rows: {quality.get('n_kc_rows')}",
        "",
        "## Data drift",
        f"- evaluated: {dd_sum.get('n_features_evaluated', 'n/a')}",
        f"- drift flagged: {dd_sum.get('n_features_drift_flagged', 'n/a')}",
        f"- overall drift: {dd_sum.get('drift', 'n/a')}",
        "",
        "## Prediction drift",
        f"- skipped: {prediction_drift.get('skipped', False)}",
        f"- drift: {prediction_drift.get('drift', 'n/a')}",
        f"- ks_pvalue: {prediction_drift.get('ks_pvalue', 'n/a')}",
        f"- mean (current): {pr_sum.get('current_mean', 'n/a') if pr_sum else 'n/a'}",
        "",
        "## Model performance",
        f"- status: {performance.get('status', performance)}",
        f"- mae: {performance.get('mae', 'n/a')}",
        f"- rmse: {performance.get('rmse', 'n/a')}",
        "",
    ]
    md_file.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote monitoring Markdown to %s", md_file)

    payload["artifacts"] = {"json": str(json_path.resolve()), "markdown": str(md_file.resolve())}
    return payload
