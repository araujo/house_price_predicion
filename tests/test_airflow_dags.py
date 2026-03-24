"""DAG and Airflow task helpers — syntax always; full parse requires ``apache-airflow``."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

DAG_NAMES = ("training_pipeline", "batch_scoring_pipeline", "monitoring_pipeline")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_dag_python_syntax() -> None:
    root = _project_root()
    for name in DAG_NAMES:
        path = root / "dags" / f"{name}.py"
        ast.parse(path.read_text(encoding="utf-8"))


def test_validate_training_raw_data_task(
    project_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(project_root)
    from airflow_tasks.training import validate_training_raw_data

    out = validate_training_raw_data()
    assert out["ok"] is True
    assert out["n_kc_rows"] > 0


def test_airflow_dag_modules_expose_dag() -> None:
    pytest.importorskip(
        "airflow",
        reason=(
            "Apache Airflow 2.x does not support Python 3.13+. "
            "Use Python 3.11–3.12 and: pip install -e '.[airflow]'"
        ),
    )
    import importlib.util

    root = _project_root()
    for name in DAG_NAMES:
        path = root / "dags" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert getattr(mod, "dag", None) is not None
        assert mod.dag.dag_id == name
