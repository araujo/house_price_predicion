.PHONY: install install-dev lint format test run docker-build docker-up docker-down clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest tests/ -q

# Requires Python 3.11–3.12 + pip install -e ".[airflow]"
test-airflow-dags:
	pytest tests/test_airflow_dags.py::test_airflow_dag_modules_expose_dag -q

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
