"""
Model retraining DAG — thin orchestration over ``airflow_tasks.training``.

Steps: validate raw inputs → train / evaluate / register / persist → log summary.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow_tasks.training import (
    execute_training,
    summarize_training_run,
    validate_training_raw_data,
)

DEFAULT_ARGS = {
    "owner": "hpp",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="training_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["house_price", "training"],
    doc_md=__doc__,
)
def training_pipeline():
    @task(task_id="validate_raw_data")
    def validate_raw_data() -> dict:
        return validate_training_raw_data()

    @task(task_id="train_evaluate_register")
    def train_evaluate_register() -> dict:
        return execute_training()

    @task(task_id="summarize_run")
    def summarize_run(train_result: dict) -> dict:
        return summarize_training_run(train_result)

    validated = validate_raw_data()
    trained = train_evaluate_register()
    validated >> trained
    summarize_run(trained)


dag = training_pipeline()
