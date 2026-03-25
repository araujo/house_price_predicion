# House Price Prediction — End-to-End MLOps Project

Production-style MLOps project for house price prediction, including:

- FastAPI (serving)
- Airflow (orchestration)
- MLflow (experiment tracking + model registry)
- Batch inference
- Monitoring (data drift, prediction drift, quality)

---

## 1. Prerequisites

Make sure you have installed:

- Docker
- Docker Compose
- Git

---

## 2. Setup

Clone the repository:

```bash
git clone git@github.com:araujo/house_price_predicion.git
cd house_price_prediction
```

Start all services:

```bash
docker compose up --build
```

(Optional background mode)

```bash
docker compose up --build -d
```

---

## 3. Services

After startup, access:

- FastAPI / Swagger: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Airflow UI: http://localhost:8080
- MLflow UI: http://localhost:5000

### Airflow credentials
- user: `admin`
- password: `admin`

---

## 4. Validate Services

### API

```bash
curl http://localhost:8000/health
```

Expected: HTTP 200

---

### Airflow

Open UI and verify DAGs:

- training_pipeline
- batch_scoring_pipeline
- monitoring_pipeline

---

### MLflow

Open UI and confirm it loads correctly.

---

## 5. Training Pipeline

### Run via UI

1. Open Airflow  
2. Select training_pipeline  
3. Click Trigger DAG  

### Run via CLI

```bash
docker compose exec airflow-webserver airflow dags trigger training_pipeline
```

### Expected Result

- Data is validated  
- Multiple models are trained  
- Metrics are logged to MLflow  
- Best model is selected  
- Best model is registered  

---

## 6. Check MLflow

Open: http://localhost:5000

Verify:

- Experiments exist  
- Multiple runs logged  
- Metrics (MAE, RMSE, R2)  
- Model registered  

---

## 7. Batch Scoring Pipeline

### Run

```bash
docker compose exec airflow-webserver airflow dags trigger batch_scoring_pipeline
```

### Expected Result

- Features engineered  
- Model loaded  
- Predictions generated  

Output file:

data/processed/batch_predictions.csv

---

## 8. Monitoring Pipeline

### Run

```bash
docker compose exec airflow-webserver airflow dags trigger monitoring_pipeline
```

### Expected Output

monitor/reports/monitoring_latest.json  
reports/monitoring/monitoring_summary.md  

---

## 9. Test Prediction (Swagger)

Open:

http://localhost:8000/docs

Use endpoint:

POST /api/v1/predict/full

---

## 10. End-to-End Flow

1. Start services  
2. Run training  
3. Check MLflow  
4. Run batch scoring  
5. Run monitoring  
6. Test API  

---

## 11. Troubleshooting

Check logs:

```bash
docker compose logs -f airflow-scheduler
docker compose logs -f api
```

---

## 12. Notes

- Training is not exposed via API  
- Airflow orchestrates pipelines  
- MLflow manages models  

---

## 13. Recommended Demo Flow

1. Run training DAG  
2. Show MLflow  
3. Run batch scoring  
4. Run monitoring  
5. Call API  

---

## 14. Future Improvements

- automated retraining  
- alerting  
- feature store  
- CI/CD  

## 15. Architecture Deatil

You can find the architecture detail [here](/docs/architecture.md)