"""
Monitoring DAG — quality gates, data drift (KS/PSI), prediction drift, optional labeled performance.

Reports: JSON at ``monitor/reports/monitoring_latest.json``;
Markdown at ``reports/monitoring/monitoring_summary.md``.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.decorators import dag, task

DEFAULT_ARGS = {
    "owner": "hpp",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="monitoring_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["house_price", "monitoring"],
    doc_md=__doc__,
)
def monitoring_pipeline():
    @task(task_id="schema_and_quality_checks")
    def schema_and_quality_task() -> dict:
        from airflow_tasks.monitoring_checks import run_schema_and_quality_checks

        return run_schema_and_quality_checks()

    @task(task_id="data_drift_monitoring")
    def data_drift_task(quality_summary: dict) -> dict:
        from airflow_tasks.monitoring_checks import run_data_drift_monitoring

        return run_data_drift_monitoring(quality_summary)

    @task(task_id="prediction_drift_monitoring")
    def prediction_drift_task() -> dict:
        from airflow_tasks.monitoring_checks import run_prediction_drift_monitoring

        return run_prediction_drift_monitoring()

    @task(task_id="model_performance_monitoring")
    def model_performance_task() -> dict:
        from airflow_tasks.monitoring_checks import run_model_performance_monitoring

        return run_model_performance_monitoring()

    @task(task_id="write_unified_monitoring_report")
    def unified_report_task(
        quality: dict,
        data_drift: dict,
        prediction_drift: dict,
        performance: dict,
    ) -> dict:
        from airflow_tasks.monitoring_checks import write_unified_monitoring_report

        return write_unified_monitoring_report(quality, data_drift, prediction_drift, performance)

    q = schema_and_quality_task()
    dd = data_drift_task(q)
    pred_drift = prediction_drift_task()
    mp = model_performance_task()
    unified_report_task(q, dd, pred_drift, mp)


dag = monitoring_pipeline()
