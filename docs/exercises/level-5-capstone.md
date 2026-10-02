# Level 5 capstone: ship a production-ready model service

> **Level 5 · Capstone** · ⏱️ 12–20 h · Prerequisites: all of [Level 5](../chapters/05-applied-ml/index.md)

This capstone takes a model all the way from data to a monitored service: a leakage-proof pipeline, honest tuning, a cost-based decision threshold, experiment tracking, a validated HTTP API with tests, a Docker image, and a drift monitor with a retraining policy. It's the gate out of the Intermediate tier. Finish it without notes before moving on to [Level 6](../chapters/06-neural-networks/index.md).

## The scenario

You're the first ML engineer at Wavelength, a (fictional) mobile and broadband provider. About a quarter of its monthly customers cancel each year, and the retention team wants to call at-risk customers with an offer before they leave. Until now, a data scientist has emailed them a spreadsheet from a notebook every month. That notebook can't be rerun, nobody knows which model produced last month's list, and in March a billing change quietly broke it for three weeks.

The Head of Retention gives you the brief:

1. "Build a churn model we can trust. Show me it beats a simple baseline, and tell me honestly how good it is."
2. "Each retention offer costs us USD 20. A customer who churns is worth about USD 200 to us, and the offer saves roughly 30% of the churners who get it. Decide who gets an offer to maximize expected profit, not accuracy."
3. "The CRM team wants to call your model from their system, one customer or a batch at a time. It must reject bad input loudly, not guess."
4. "Package it so IT can deploy it without calling you."
5. "Tell me when the model stops being trustworthy, before my team notices."

## The data

Use the generator below (save it as part of your `train.py`). Each row is a customer, and `churned` is whether they cancelled in the following month.

| Column | Meaning |
|---|---|
| `tenure_months` | months as a customer (integer) |
| `monthly_charges` | current monthly bill in USD |
| `support_calls` | support calls in the last month |
| `data_usage_gb` | data used last month; missing for about 5% of customers |
| `plan` | `basic`, `standard`, or `premium` |
| `contract` | `monthly` or `annual` |
| `payment` | `card`, `bank`, or `invoice` |
| `churned` | target: 1 if the customer cancelled |

The generator's parameters let you simulate the future: `charges_shift` and `usage_scale` change the input distribution (data drift), and `premium_effect` changes how premium customers behave (concept drift).

??? example "The data generator: make_churn_data (expand and copy into train.py)"

    ```python
    import numpy as np
    import pandas as pd

    TARGET = "churned"


    def make_churn_data(n=6000, seed=0, charges_shift=0.0, premium_effect=-0.5, usage_scale=1.0):
        """Synthetic churn data for a subscription business.

        charges_shift and usage_scale change P(X) (data drift); premium_effect changes P(y | X) (concept drift).
        About 5% of data_usage_gb values are missing, as in real usage logs.
        """
        rng = np.random.default_rng(seed)
        tenure = rng.integers(1, 72, n)
        plan = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.3, 0.2])
        base_price = pd.Series(plan).map({"basic": 45, "standard": 70, "premium": 95}).to_numpy()
        charges = np.round(np.clip(base_price + charges_shift + rng.normal(0, 12, n), 10, 300), 2)
        contract = np.where(rng.random(n) < 0.35 + 0.004 * tenure, "annual", "monthly")
        payment = rng.choice(["card", "bank", "invoice"], n, p=[0.5, 0.3, 0.2])
        calls = rng.poisson(1.5, n)
        usage = np.round(rng.gamma(2.0, 15.0 * usage_scale, n), 1)
        logit = (-1.0 - 0.02 * tenure + 1.8 * (tenure <= 6)                  # new customers churn most
                 + 0.03 * (charges - base_price) + 0.012 * (charges - 65)    # overpaying for the plan
                 + 0.15 * calls + 1.3 * (calls >= 3) * (contract == "monthly")
                 - 1.0 * (contract == "annual") + 0.6 * (payment == "invoice")
                 + 1.4 * (usage < 8)                                         # barely using the service
                 + np.select([plan == "basic", plan == "premium"], [0.3, premium_effect], 0.0))
        churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
        usage = usage.astype(float)
        usage[rng.random(n) < 0.05] = np.nan
        return pd.DataFrame({"tenure_months": tenure, "monthly_charges": charges, "support_calls": calls,
                             "data_usage_gb": usage, "plan": plan, "contract": contract, "payment": payment,
                             TARGET: churned})
    ```

```python
df = make_churn_data()
print(df.shape, f"churn rate {df['churned'].mean():.3f}")
print(df.head(3).to_string(index=False))
```

```text
(6000, 8) churn rate 0.253
 tenure_months  monthly_charges  support_calls  data_usage_gb     plan contract payment  churned
            61            34.41              1           13.2    basic   annual    bank        0
            46            81.19              1           91.2 standard  monthly    card        0
            37            53.88              1           53.3    basic  monthly    bank        1
```

## What to do

Build a small project (a folder, not a notebook) with the parts below. Use a fixed random seed everywhere.

### Part A: a leakage-proof pipeline

1. Split off a stratified test set (25%) and don't touch it until Part C.
2. Build one scikit-learn `Pipeline` with a `ColumnTransformer` that imputes and scales numeric columns and imputes and one-hot encodes categoricals (tolerating unseen categories), followed by a model.
3. Build a baseline (logistic regression with the same preprocessing) and record its cross-validated ROC AUC.

### Part B: tuning

1. Tune a gradient-boosting model (for example, `HistGradientBoostingClassifier`) with Optuna: at least 20 trials, a TPE sampler with a seed, log-scale search spaces where appropriate, and pruning of weak trials.
2. Report the best CV AUC and compare it with the baseline.

### Part C: decisions and honest evaluation

1. Choose the decision threshold that maximizes expected profit, using *out-of-fold* predictions on the training data, not the test set. Derive the theoretical threshold too and compare.
2. Fit the final pipeline on all training data, and evaluate it **once** on the test set: ROC AUC, average precision, Brier score, the share of customers flagged, and expected profit per customer.
3. Explain any gap between the CV and test scores.

### Part D: tracking and artifacts

1. Log every run to MLflow (a local store is fine): parameters, the CV score of every trial, the final test metrics, the threshold, and the model. Register the final model.
2. Save the fitted pipeline with `joblib`, plus a `metadata.json` with the version, threshold, feature schema (allowed ranges and categories), metrics, a hash of the training data, and library versions. Save a sample of training rows with model scores as a drift reference.

### Part E: the API and its tests

1. Write a FastAPI app with `GET /health` (returning the model version) and `POST /predict` (1 to 1,000 customers per request, returning probabilities and an `at_risk` flag at your threshold). Load the model once, at startup.
2. Validate input with pydantic: types, ranges, allowed categories, a nullable `data_usage_gb`, and no unknown fields.
3. Write tests with FastAPI's `TestClient` (no real server): a health check, a valid batch, at least four kinds of invalid input (including a monthly charge in cents), a missing optional value, an invariance test, a directional test, and a minimum-performance test. Make them runnable with `pytest` and also with plain `python test_app.py`.

### Part F: packaging

1. Write a `requirements.txt` with pinned versions and a `Dockerfile` for the service: a slim Python base image, dependencies installed before the code is copied, a non-root user, a health check, and uvicorn as the command.
2. If you have Docker, build and run it, and call `/predict` with `curl`. (If you don't, the Dockerfile still counts; make sure it's correct by reading it line by line.)

### Part G: monitoring and retraining

1. Write `monitor.py`: given a batch of new customers (a CSV, or one simulated with the generator), run data-quality checks against the saved schema, then compute PSI for every feature and for the model's scores, plus KS tests for numeric features. Write a JSON report and exit with a non-zero code on alerts.
2. Run it on three simulated batches: no drift, data drift (`charges_shift=12, usage_scale=0.6`), and a data bug (charges in cents, a renamed category). Show that it gives a different, sensible verdict for each.
3. Write a retraining policy (a short paragraph): what triggers retraining, what blocks it, and how a retrained model gets promoted.

### Part H: documentation

Write a `README.md` that lets a colleague set up, train, test, serve, and monitor the project from scratch, plus a short model card section (intended use, data, metrics, threshold, limitations).

## Deliverables

- A project folder with `train.py`, `app.py`, `test_app.py`, `monitor.py`, `Dockerfile`, `requirements.txt`, and `README.md`.
- The console output of a full training run, the test run, and the three monitoring runs.
- The MLflow store (or a screenshot of the MLflow UI listing your runs).
- A one-paragraph summary for the Head of Retention: how good the model is, who gets an offer, the expected profit per customer, and how you'll know when it stops working.

## Acceptance checklist

Your project is done when every box is checked:

- [ ] `python train.py` runs end to end from a clean folder, with no manual steps, and is reproducible (same seed, same numbers).
- [ ] All preprocessing lives inside one pipeline, and cross-validation and tuning run on the pipeline, never on preprocessed data.
- [ ] A baseline is reported, and the tuned model's CV AUC beats it.
- [ ] The threshold is chosen from out-of-fold predictions using the stated costs, and the test set is used exactly once.
- [ ] MLflow contains a run per trial and a final run with parameters, metrics, the threshold, and a registered model.
- [ ] The saved artifact includes metadata (version, threshold, schema, metrics, data hash, library versions).
- [ ] The API rejects out-of-range values, unknown categories, unknown fields, and empty or oversized batches with HTTP 422, and accepts a missing `data_usage_gb`.
- [ ] Tests cover the API contract and model behavior, and pass with `python test_app.py`.
- [ ] The Dockerfile pins dependencies, installs them before copying code, runs as a non-root user, and has a health check.
- [ ] The monitor reports OK on the clean batch, a drift alert on the drifted batch, and a blocking data-quality alert on the bugged batch.
- [ ] The README explains setup, training, testing, serving, monitoring, and the retraining policy.

## Hints

??? tip "Hint for Part A"
    Reuse the `ColumnTransformer` pattern from [ML pipelines](../chapters/05-applied-ml/01-ml-pipelines.md): a numeric sub-pipeline (`SimpleImputer(strategy="median")`, `StandardScaler`) and a categorical one (`SimpleImputer(strategy="most_frequent")`, `OneHotEncoder(handle_unknown="ignore")`). Write a `build_pipeline(params)` function so that tuning, final training, and tests all build the model the same way.

??? tip "Hint for Part B"
    Inside the Optuna objective, loop over the CV folds yourself and call `trial.report(mean_so_far, fold)` after each fold, then `trial.should_prune()`. That gives the `MedianPruner` something to prune on. Wrap each trial in `mlflow.start_run(nested=True)` to log it as a child of the training run. See [Hyperparameter tuning](../chapters/05-applied-ml/02-hyperparameter-tuning.md).

??? tip "Hint for Part C, the threshold"
    An offer sent to a customer with churn probability $p$ is worth $0.3 \times 200 \times p - 20$ dollars in expectation, so it pays when $p > 20/60 = 1/3$. With calibrated probabilities, the profit-maximizing threshold should be near 0.33. Get out-of-fold probabilities with `cross_val_predict(..., method="predict_proba")`, compute the profit for a grid of thresholds, and pick the best. See [Imbalanced data](../chapters/05-applied-ml/03-imbalanced-data.md).

??? tip "Hint for Part D"
    MLflow 3 refuses a plain folder as a tracking store by default; use `mlflow.set_tracking_uri("sqlite:///mlflow.db")`. `mlflow.sklearn.log_model` saves scikit-learn models with skops, which may ask you to list trusted types (`skops_trusted_types=[...]`) for gradient-boosting models you created yourself.

??? tip "Hint for Part E"
    Use pydantic `Field(ge=..., le=...)` for ranges, `Literal[...]` for categories, `float | None = None` for the optional field, and `model_config = ConfigDict(extra="forbid")` to reject unknown fields. Load the model in a FastAPI `lifespan` function so importing `app.py` doesn't require a model, and use `with TestClient(app) as client:` in tests so the lifespan runs. See [ML in production](../chapters/05-applied-ml/05-ml-in-production.md).

??? tip "Hint for Part G"
    Save a sample of the training data, with model scores, next to the model. PSI with 10 quantile bins of the reference works well for numeric features; for categoricals, use one bin per category. Run data-quality checks first, and if they fail, don't compute drift at all: the verdict is "fix the pipeline".

## Solution

A complete worked solution, with every file and real outputs, is in the [Level 5 capstone solution](solutions/level-5-capstone.md). Try the whole project yourself first: the point is to discover, by building it, where production ML systems break.
