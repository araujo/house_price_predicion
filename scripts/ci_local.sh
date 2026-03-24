#!/usr/bin/env bash
# Mirror local checks to GitHub Actions CI (editable install, ruff, pytest, DAG import test).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install --upgrade pip
pip install -e ".[dev,airflow]"
ruff check .
pytest tests/ -q
