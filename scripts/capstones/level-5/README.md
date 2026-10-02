# Churn model service (Level 5 capstone solution)

A small, production-style ML project: a churn model trained as a scikit-learn pipeline, tuned with
Optuna, tracked with MLflow, served with FastAPI, packaged with Docker, and monitored for drift.
It is the worked solution to the handbook's
[Level 5 capstone](../../../docs/exercises/level-5-capstone.md).

## Layout

| File | What it does |
|---|---|
| `train.py` | Generates (or loads) data, tunes a gradient-boosting pipeline with Optuna, picks a cost-based threshold, logs everything to MLflow, and saves `model/` |
| `app.py` | FastAPI service: `/health`, `/model-info`, and `/predict` with pydantic validation |
| `test_app.py` | API contract tests and model behavior tests (run with pytest or `python test_app.py`) |
| `monitor.py` | Data-quality checks and a drift report (PSI, KS) for a batch of production data |
| `Dockerfile` | Serving image: pinned dependencies, the model, a non-root user, a health check |
| `requirements.txt` | Pinned serving dependencies (used by the image) |
| `requirements-dev.txt` | Training, tracking, monitoring, and test dependencies |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

python train.py                     # ~1 minute: writes model/ and mlflow.db
python test_app.py                  # or: pytest test_app.py
python monitor.py --simulate drift  # writes drift_report.json; exit code 1 on ALERT

uvicorn app:app --port 8000         # then open http://localhost:8000/docs
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"customers": [{"tenure_months": 3, "monthly_charges": 89.5, "support_calls": 4,
       "data_usage_gb": 5.0, "plan": "basic", "contract": "monthly", "payment": "invoice"}]}'
```

## Docker

```bash
python train.py                     # the image copies model/model.joblib and model/metadata.json
docker build -t churn-api:1.0.0 .
docker run --rm -p 8000:8000 churn-api:1.0.0
```

## Experiment tracking

Runs are stored in a local SQLite file (`mlflow.db`) with artifacts in `mlruns/`. Each training run is a
parent run with one nested run per Optuna trial, the test metrics, the model files, and a registered
model version (`churn-capstone`). Browse them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Monitoring and retraining

`monitor.py` compares a batch with `model/reference.csv` (a sample of the training data with model
scores). Data-quality failures (missing columns, out-of-range values, unseen categories) block
scoring and retraining. Otherwise it reports PSI for every feature and for the model's scores
(OK below 0.1, WARN up to 0.25, ALERT above; these are rules of thumb), with KS tests for numeric
features as a secondary signal. A suggested policy: retrain when performance on newly labeled data
drops by more than 0.03 AUC, investigate on drift ALERTs, and retrain at least every 90 days; promote
a retrained model only after it beats the current one on recent holdout data.

## Notes

- The data is synthetic (see `make_churn_data` in `train.py`); `--data your.csv` trains on a CSV
  with the same columns plus `churned`.
- Loading a pickled model runs code: only load `model.joblib` files you created or trust, and serve
  them with the scikit-learn version recorded in `model/metadata.json`.
