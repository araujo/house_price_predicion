# House price prediction (MLOps scaffold)

Production-oriented layout for a Seattle house price prediction service: shared **data**, **features**, **training**, **FastAPI** serving, and **Airflow** orchestration.

## Layout

- `app/` — FastAPI (`app/main.py`), routes, services, schemas, config, DI
- `config/` — `model_config.yaml` and other YAML
- `dags/` — Airflow DAG definitions (TaskFlow API)
- `airflow_tasks/` — thin task entrypoints used by DAGs (call `data_engineer`, `model_trainer`, `app`)
- `data/` — `raw/`, `processed/`, etc.
- `data_engineer/` — ingestion, validation, preprocessing, feature engineering
- `model_trainer/` — training, evaluation, MLflow registration
- `monitor/` — drift & performance (KS/PSI, prediction drift, optional labeled MAE/RMSE)
- `scripts/` — helpers (e.g. [`scripts/ci_local.sh`](scripts/ci_local.sh) mirrors CI checks)
- `tests/` — unit, API, Airflow syntax / optional import tests

## Local development

Use Python **3.11** (see [Dependency management](#dependency-management)). Then:

```bash
pip install -e ".[dev]"
make run
# GET http://127.0.0.1:8000/health
```

## Tests

```bash
make test
```

## Docker

```bash
make docker-build
make docker-up
```

Services (default `docker compose up`):

| Service | Port | Role |
|---------|------|------|
| `api` | 8000 | FastAPI |
| `mlflow` | 5000 | MLflow UI (runs + model registry) |
| `postgres` | (internal) | Airflow metadata database |
| `airflow-init` | — | One-shot: `airflow db migrate` + create `admin` user |
| `airflow-webserver` | **8080** | Airflow UI |
| `airflow-scheduler` | — | Executes DAGs |

**URLs:** Airflow [http://127.0.0.1:8080](http://127.0.0.1:8080) (default login `admin` / `admin`), MLflow [http://127.0.0.1:5000](http://127.0.0.1:5000), API [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health).

The [`Dockerfile.mlflow`](Dockerfile.mlflow) image is **minimal** (`python:3.11-slim` + MLflow only); it does not install the house-price application. Training code runs in Airflow/FastAPI images and talks to the server at `http://mlflow:5000`.

The tracking server uses **MLflow 3.5+** Host-header validation (`--allowed-hosts` / `MLFLOW_SERVER_ALLOWED_HOSTS`). The `Host` header includes the **port** (e.g. `mlflow:5000`, `localhost:5000`), so Compose sets patterns `mlflow:*,localhost:*,127.0.0.1:*` for the Compose DNS name, browser access, and port-mapped curls. Adjust if you rename the service (see [MLflow tracking server security](https://mlflow.org/docs/latest/tracking/server-security.html)).

**Artifacts (Compose):** The **`mlflow_data`** named volume is mounted at **`/srv/mlflow_data`** on both the **`mlflow`** service and the **Airflow** services. The server stores SQLite and the default artifact root under that path; run artifact URIs resolve to **`file:///srv/mlflow_data/artifacts/...`**, so the **scheduler** (training) must have the **same directory** available—otherwise `mlflow.log_text` / `log_model` fail with permission or missing path. The **`mlflow`** service uses **`user: ${AIRFLOW_UID:-50000}:0`** so files are owned by the same UID as Airflow tasks. Tracking is still **`http://mlflow:5000`**; HTTP is used for the API and UI, while run files live on the shared volume.

Training DAGs use `HPP_MLFLOW_TRACKING_URI=http://mlflow:5000` and `HPP_USE_MLFLOW=true`, so runs and registered models appear in the MLflow UI after you trigger `training_pipeline`.

On Linux, if bind-mounted dirs are not writable by the Airflow user, set `export AIRFLOW_UID=$(id -u)` before `docker compose up` (see [Airflow docs](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html)).

If you previously ran the `mlflow` container as **root**, the `mlflow_data` volume may contain root-owned files and training can still fail with permission errors. Remove the volume once (`docker compose down -v` — **deletes** MLflow DB and artifacts) or `chown` the volume contents to UID `AIRFLOW_UID`.

**MLflow service must stay up:** the `mlflow` service runs as **root** only for the entry step, then **`gosu ${AIRFLOW_UID}`** starts the server so SQLite and artifacts match the Airflow user. If the container used to run **only** as UID `50000` on a fresh Docker volume, startup could **fail** (cannot `chown`/`mkdir` on a root-owned mount) and the container **exited** — then the hostname `mlflow` did not resolve. The current image fixes that by **chown** at boot, then dropping privileges with **gosu**.

**Verify MLflow + DNS from Airflow:**

```bash
docker compose build mlflow && docker compose up -d mlflow
docker compose logs -f mlflow
# Ctrl+C, then:
docker compose exec airflow-webserver python -c "import socket; print(socket.gethostbyname('mlflow'))"
docker compose exec airflow-webserver python -c "import urllib.request; urllib.request.urlopen('http://mlflow:5000/')"
docker compose exec airflow-webserver airflow tasks test training_pipeline train_evaluate_register 2026-03-24
```

## Monitoring

The `monitor/` package is **framework-agnostic** (usable outside Airflow):

| Module | Role |
|--------|------|
| `data_drift.py` | Numeric feature drift: Kolmogorov–Smirnov + optional PSI per column |
| `prediction_drift.py` | Batch predictions vs training `price` distribution (KS + mean shift) |
| `model_performance.py` | MAE / RMSE when `y_true` and `y_pred` columns exist |

Unified JSON: `monitor/reports/monitoring_latest.json` (override with `HPP_MONITOR_REPORTS_DIR`).  
Markdown summary: `reports/monitoring/monitoring_summary.md`.

| Variable | Role |
|----------|------|
| `HPP_MONITOR_REPORTS_DIR` | Directory for `monitoring_latest.json` |
| `HPP_BATCH_PREDICTIONS_PATH` / `HPP_BATCH_OUTPUT` | Batch predictions CSV for prediction drift |
| `HPP_MONITORING_LABELS_PATH` | Optional CSV with truth + prediction columns for performance metrics |

## Airflow (local)

DAGs orchestrate existing Python modules; business logic stays in `data_engineer`, `model_trainer`, `monitor`, and `airflow_tasks`, not inside DAG files.

### Install (optional)

Apache Airflow **2.x** (used by `Dockerfile.airflow` and DAGs) supports **Python 3.11 and 3.12**, not 3.13+. Use a 3.11/3.12 virtualenv or the Docker image below if `pip install apache-airflow` fails.

For the Airflow CLI and **full DAG import tests** (`pytest` loads each DAG module):

```bash
pip install -e ".[airflow]"
pytest tests/test_airflow_dags.py::test_airflow_dag_modules_expose_dag -q
```

Syntax-only checks (no Airflow install) always run as part of `make test`.

### DAGs

| DAG | Purpose |
|-----|---------|
| `training_pipeline` | Validate raw data → train / evaluate / MLflow register / save `best_model.joblib` |
| `batch_scoring_pipeline` | Validate inference CSV → merge demographics → Phase 3 features → score → CSV output |
| `monitoring_pipeline` | Quality → data drift (KS/PSI) → prediction drift → optional labeled performance → unified JSON + Markdown |

### Environment variables (common)

Paths default to repo-relative values suitable for local runs.

| Variable | Role |
|----------|------|
| `HPP_RAW_DATA_DIR` | Raw CSV directory (default `data/raw`) |
| `HPP_MODEL_OUTPUT_DIR` | Training artifacts (default `model`) |
| `HPP_TRAINING_CONFIG` | Optional path to `model_config.yaml` |
| `HPP_USE_MLFLOW` | `true` / `false` for training DAG |
| `HPP_MLFLOW_TRACKING_URI` | MLflow server URI when using registry |
| `HPP_LOCAL_MODEL_PATH` | Fallback model for batch scoring |
| `HPP_BATCH_OUTPUT` | Batch predictions CSV path |
| `HPP_AIRFLOW_SCRATCH` | Intermediate CSVs for batch DAG |
| `HPP_MONITORING_OUTPUT_DIR` | Monitoring Markdown output |
| `HPP_MONITOR_REPORTS_DIR` | JSON report directory (`monitor/reports`) |
| `HPP_MONITORING_LABELS_PATH` | Labeled CSV for MAE/RMSE (optional) |
| `HPP_TRAINING_BASELINE_STATS` | Optional JSON of column means/std (legacy helper) |
| `HPP_BATCH_PREDICTIONS_PATH` | Overrides default when locating batch predictions for drift |

### Docker Compose (full local stack)

`docker compose up -d` builds (if needed) and starts API, MLflow, Postgres, Airflow init, scheduler, and webserver. DAGs live under `./dags` and are mounted at `/opt/airflow/dags`; project code is installed in the image with `PYTHONPATH=/opt/airflow/project` and the same data/model/config paths as in the table above.

Use the Airflow UI to enable DAGs and trigger **training**, **batch scoring**, or **monitoring** manually. No extra profile is required.

### Production evolution

- Point `HPP_MLFLOW_TRACKING_URI` at a managed MLflow or use artifact store + Postgres for Airflow metadata.
- Schedule `training_pipeline` weekly and `monitoring_pipeline` daily; gate batch scoring on successful training or model version.
- Replace CSV handoffs in batch scoring with object storage + XCom paths if scale requires it.

## CI/CD (GitHub Actions)

### CI — [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

**Triggers:** `pull_request` into `main`, and `push` to `main`.

**Purpose:** fail fast on quality issues before merge; keep `main` deployable.

| Step | What it checks |
|------|----------------|
| Ruff (`ruff check .`) | Same scope as `make lint` — whole repo per Ruff config |
| Pytest (`pytest tests/ -q`) | Same as `make test`; includes `tests/test_airflow_dags.py` (DAG imports when Airflow installs) |
| Docker build | `docker build -f Dockerfile.fastapi` smoke test only (not in `make test`) |

Dependency install uses `pip install -e ".[dev,airflow]"` and **caches pip** via `setup-python`. Concurrent runs on the same branch cancel older jobs (`concurrency`).

Local checks without Docker:

```bash
bash scripts/ci_local.sh
```

To match CI’s Docker step locally: `docker build -f Dockerfile.fastapi -t house-price-api:ci .`

### CD — [`.github/workflows/cd.yml`](.github/workflows/cd.yml)

**Triggers:**

1. **`workflow_run`** — after workflow **CI** completes **successfully** for a **`push` to `main`** (not for PR-only CI runs).
2. **`workflow_dispatch`** — manual run from the Actions tab (builds from `main`).

**Purpose:** produce **versioned container artifacts** tied to the commit SHA; registry push and deploy are **placeholders** until you choose GHCR, ECR, or another registry.

| Image | Dockerfile |
|-------|------------|
| `house-price-api` | [`Dockerfile.fastapi`](Dockerfile.fastapi) |
| `house-price-mlflow` | [`Dockerfile.mlflow`](Dockerfile.mlflow) |
| `house-price-airflow` | [`Dockerfile.airflow`](Dockerfile.airflow) |

Tags per image (deterministic + readable):

- **`<full_git_sha>`** — 40-character commit SHA of the checked-out tree (immutable, traceable to CI).
- **`sha-<short>`** — 7-character prefix for humans and logs.
- **`latest`** — moving label for the most recent CD build from `main` (only use in dev; pin by full SHA in prod).

No registry or cluster is assumed; push/deploy steps are **echo placeholders** only.

**Production evolution:** add `docker/login-action`, push to `ghcr.io/<owner>/<repo>/...`, then deploy with your stack (Helm, Argo CD, Terraform, CodeDeploy). Keep secrets in GitHub **encrypted secrets** or OIDC — never commit credentials.

### Why this layout (interview notes)

- **CI** = quality gate on every change to `main` and every PR targeting `main`.
- **CD** = artifact promotion **after** CI passes on `main`, so broken images are not built from bad commits.
- **Separation** keeps PR feedback fast (CI) while delivery stays tied to merged code (CD).

## Dependency management

- **Source of truth:** [`pyproject.toml`](pyproject.toml) only — install with `pip install -e ".[dev]"` or `pip install -e ".[dev,airflow]"` for local Airflow/DAG tests.
- **Python version:** use **3.11** locally to match Docker images and GitHub Actions. Pinned stacks (e.g. `pandas` 2.1.x) may not ship wheels for newer interpreters, which forces slow or failing source builds.
- Optional extras: `dev`, `airflow`, `mlflow` (the runtime already lists `mlflow` in core dependencies; the extra exists for optional installs).
- [`requirements.txt`](requirements.txt) is **not** authoritative: it documents that flat lockfiles are optional (e.g. `pip-compile`); do not treat it as the primary dependency list.
