"""
Batch inference DAG — loads ``future_unseen_examples``, merges demographics, engineers features,
scores with the best model (MLflow or local), writes CSV.

Intermediate steps use CSV handoffs under ``data/processed/airflow_scratch/`` (configurable).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow_tasks.batch_scoring import (
    engineer_batch_features,
    load_model_score_and_write,
    merge_inference_with_demographics,
    summarize_batch_run,
    validate_inference_rows,
)

DEFAULT_ARGS = {
    "owner": "hpp",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="batch_scoring_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["house_price", "batch"],
    doc_md=__doc__,
)
def batch_scoring_pipeline():
    @task(task_id="validate_inference_rows")
    def validate_inference_rows_task() -> dict:
        return validate_inference_rows()

    @task(task_id="merge_demographics")
    def merge_demographics_task() -> str:
        return merge_inference_with_demographics()

    @task(task_id="engineer_features")
    def engineer_features_task(merged_csv_path: str) -> str:
        return engineer_batch_features(merged_csv_path)

    @task(task_id="score_and_write_predictions")
    def score_task(merged_csv_path: str, feature_csv_path: str) -> dict:
        return load_model_score_and_write(merged_csv_path, feature_csv_path)

    @task(task_id="summarize_batch_run")
    def summarize_task(score_result: dict) -> dict:
        return summarize_batch_run(score_result)

    validated = validate_inference_rows_task()
    merged = merge_demographics_task()
    validated >> merged
    features = engineer_features_task(merged)
    scored = score_task(merged, features)
    summarize_task(scored)


dag = batch_scoring_pipeline()
