"""Level 5 capstone: the churn model as an HTTP service.

Run locally (after `python train.py`):
    uvicorn app:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000/docs for interactive documentation.

Endpoints:
    GET  /health      liveness, plus the model name and version being served
    GET  /model-info  the model's metadata (threshold, metrics, training data hash)
    POST /predict     churn probabilities for 1 to 1,000 customers, with input validation

The model directory comes from the MODEL_DIR environment variable (default: ./model).
"""
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


class Customer(BaseModel):
    """One customer's features. Bounds match the training schema in train.py."""

    model_config = ConfigDict(extra="forbid")          # a misspelled field is an error, not ignored

    tenure_months: int = Field(ge=0, le=600, description="months as a customer")
    monthly_charges: float = Field(ge=1, le=500, description="monthly bill in USD")
    support_calls: int = Field(ge=0, le=100, description="support calls in the last month")
    data_usage_gb: float | None = Field(default=None, ge=0, le=2000,
                                        description="monthly data usage; may be missing")
    plan: Literal["basic", "standard", "premium"]
    contract: Literal["monthly", "annual"]
    payment: Literal["card", "bank", "invoice"]


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    at_risk: bool


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


def create_app(model_dir: str | None = None) -> FastAPI:
    """Build the app. The model is loaded once, at startup, from model_dir or $MODEL_DIR."""
    state = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = Path(model_dir or os.environ.get("MODEL_DIR", "model"))
        state["model"] = joblib.load(path / "model.joblib")
        state["meta"] = json.loads((path / "metadata.json").read_text())
        state["columns"] = state["meta"]["features"]["numeric"] + state["meta"]["features"]["categorical"]
        yield
        state.clear()

    app = FastAPI(title="Churn model", lifespan=lifespan)

    @app.get("/health")
    def health():
        if "model" not in state:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ok", "model": state["meta"]["name"], "model_version": state["meta"]["version"]}

    @app.get("/model-info")
    def model_info():
        meta = state["meta"]
        return {k: meta[k] for k in ["name", "version", "threshold", "metrics", "training_data_sha256", "versions"]}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        started = time.perf_counter()
        frame = pd.DataFrame([c.model_dump() for c in request.customers])[state["columns"]]
        frame["data_usage_gb"] = frame["data_usage_gb"].astype(float)      # None -> NaN, imputed by the pipeline
        proba = state["model"].predict_proba(frame)[:, 1]
        threshold = state["meta"]["threshold"]
        response = PredictResponse(
            model_version=state["meta"]["version"], threshold=threshold,
            predictions=[Prediction(churn_probability=round(float(p), 4), at_risk=bool(p >= threshold))
                         for p in proba])
        state["last_latency_ms"] = (time.perf_counter() - started) * 1000   # a hook for monitoring
        return response

    return app


app = create_app()
