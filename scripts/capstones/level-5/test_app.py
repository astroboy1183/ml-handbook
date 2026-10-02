"""Level 5 capstone: tests for the churn service and model.

Run with pytest if you have it (`pytest test_app.py`), or directly: `python test_app.py`.
Requires a trained model in ./model (run `python train.py` first), or set MODEL_DIR.
"""
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*httpx.*")      # a dependency deprecation notice

import numpy as np
from fastapi.testclient import TestClient

from app import create_app
from train import CATEGORICAL, NUMERIC, make_churn_data

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "model"))
META = json.loads((MODEL_DIR / "metadata.json").read_text())

VALID = {"tenure_months": 3, "monthly_charges": 89.5, "support_calls": 4, "data_usage_gb": 5.0,
         "plan": "basic", "contract": "monthly", "payment": "invoice"}
LOYAL = {"tenure_months": 60, "monthly_charges": 92.0, "support_calls": 0, "data_usage_gb": 45.0,
         "plan": "premium", "contract": "annual", "payment": "card"}


def client():
    return TestClient(create_app(str(MODEL_DIR)))


def predict(c, customers):
    return c.post("/predict", json={"customers": customers})


# ---------- API contract ----------

def test_health_reports_model_version():
    with client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "churn", "model_version": META["version"]}


def test_predict_returns_one_prediction_per_customer():
    with client() as c:
        r = predict(c, [VALID, LOYAL])
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == META["version"] and body["threshold"] == META["threshold"]
    assert len(body["predictions"]) == 2
    for p in body["predictions"]:
        assert 0.0 <= p["churn_probability"] <= 1.0
        assert p["at_risk"] == (p["churn_probability"] >= META["threshold"])


def test_missing_optional_value_is_accepted():
    with client() as c:
        r = predict(c, [{**VALID, "data_usage_gb": None}])
    assert r.status_code == 200


def test_invalid_inputs_are_rejected_with_422():
    bad_cases = [
        {**VALID, "tenure_months": -1},                  # out of range
        {**VALID, "monthly_charges": 8950.0},            # cents instead of dollars
        {**VALID, "plan": "gold"},                       # unknown category
        {**VALID, "monthly_charge": 89.5},               # misspelled field (extra fields forbidden)
        {k: v for k, v in VALID.items() if k != "contract"},   # missing required field
    ]
    with client() as c:
        for case in bad_cases:
            assert predict(c, [case]).status_code == 422, case
        assert c.post("/predict", json={"customers": []}).status_code == 422
        assert c.post("/predict", json={"customers": [VALID] * 1001}).status_code == 422


# ---------- model behavior ----------

def test_high_risk_profile_scores_above_loyal_profile():
    with client() as c:
        p_new, p_loyal = [x["churn_probability"] for x in predict(c, [VALID, LOYAL]).json()["predictions"]]
    assert p_new > p_loyal


def test_invariance_to_batch_order():
    customers = [VALID, LOYAL, {**VALID, "plan": "standard"}]
    with client() as c:
        forward = [x["churn_probability"] for x in predict(c, customers).json()["predictions"]]
        backward = [x["churn_probability"] for x in predict(c, customers[::-1]).json()["predictions"]]
    assert forward == backward[::-1]


def test_directional_more_support_calls_do_not_lower_risk():
    import joblib
    model = joblib.load(MODEL_DIR / "model.joblib")
    X = make_churn_data(n=300, seed=123)[NUMERIC + CATEGORICAL]
    more = X.assign(support_calls=X["support_calls"] + 3)
    # tree ensembles aren't guaranteed monotone; allow tiny decreases from noise
    assert np.mean(model.predict_proba(more)[:, 1] >= model.predict_proba(X)[:, 1] - 0.01) > 0.95


def test_model_beats_baseline_and_floor():
    assert META["metrics"]["test_auc"] >= 0.75
    assert META["best_cv_auc"] > META["baseline_cv_auc"]


# ---------- run without pytest ----------

if __name__ == "__main__":
    tests = [(name, f) for name, f in sorted(globals().items()) if name.startswith("test_") and callable(f)]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"PASS {name}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL {name}: {err}")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
