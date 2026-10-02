"""Level 5 capstone: train, tune, track, and save the churn model.

Usage (from this directory):
    python train.py                     # 25 Optuna trials, writes model/ and mlflow.db
    python train.py --n-trials 10       # faster
    python train.py --data customers.csv  # train on your own CSV with the same columns plus `churned`

Outputs:
    model/model.joblib      the fitted scikit-learn pipeline (preprocessing + model)
    model/metadata.json     version, threshold, schema, metrics, data hash, library versions
    model/reference.csv     a sample of training rows with model scores (the drift reference)
    mlflow.db, mlruns/      the MLflow tracking store (browse with: mlflow ui --backend-store-uri sqlite:///mlflow.db)
"""
import argparse
import hashlib
import json
import logging
import os
import platform
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")      # small data: a few threads are faster than many
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_VERSION = "1.0.0"
NUMERIC = ["tenure_months", "monthly_charges", "support_calls", "data_usage_gb"]
CATEGORICAL = ["plan", "contract", "payment"]
TARGET = "churned"
# Valid input ranges and categories: used by the API schema and by monitoring.
SCHEMA = {
    "tenure_months": [0, 600], "monthly_charges": [1, 500], "support_calls": [0, 100], "data_usage_gb": [0, 2000],
    "plan": ["basic", "standard", "premium"], "contract": ["monthly", "annual"],
    "payment": ["card", "bank", "invoice"],
}
# Business costs (USD): a retention offer costs 20; it saves a churner worth 200 with probability 0.3.
OFFER_COST, CUSTOMER_VALUE, OFFER_SUCCESS = 20.0, 200.0, 0.3

logging.getLogger("mlflow").setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")


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


def build_pipeline(params=None):
    """Preprocessing + gradient boosting, as one estimator."""
    params = params or {}
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])
    model = HistGradientBoostingClassifier(max_iter=200, early_stopping=False, random_state=0, **params)
    return Pipeline([("prep", prep), ("model", model)])


def build_baseline():
    """A logistic regression with the same preprocessing: the model to beat."""
    pipe = build_pipeline()
    pipe.steps[-1] = ("model", LogisticRegression(max_iter=1000))
    return pipe


def data_fingerprint(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:12]


def best_threshold(y, p):
    """Send the offer when the expected saving beats its cost; pick the threshold by expected profit."""
    thresholds = np.round(np.arange(0.05, 0.95, 0.01), 2)
    profit = [np.sum(np.where(p >= t, y * OFFER_SUCCESS * CUSTOMER_VALUE - OFFER_COST, 0.0)) / len(y)
              for t in thresholds]
    k = int(np.argmax(profit))
    return float(thresholds[k]), float(profit[k])


def objective_factory(X, y, cv):
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 4, 48),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 200, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        }
        scores = []
        for fold, (tr, va) in enumerate(cv.split(X, y)):
            pipe = build_pipeline(params).fit(X.iloc[tr], y[tr])
            scores.append(roc_auc_score(y[va], pipe.predict_proba(X.iloc[va])[:, 1]))
            trial.report(float(np.mean(scores)), fold)          # running mean after each fold
            if trial.should_prune():
                raise optuna.TrialPruned()
        cv_auc = float(np.mean(scores))
        with mlflow.start_run(run_name=f"trial {trial.number}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_auc", cv_auc)
        return cv_auc
    return objective


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="CSV with the feature columns and `churned`; default: synthetic data")
    parser.add_argument("--n-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="model")
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    args = parser.parse_args()

    df = pd.read_csv(args.data) if args.data else make_churn_data(seed=args.seed)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.seed)
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=args.seed)
    print(f"data: {len(df):,} rows, churn rate {y.mean():.3f}; train {len(X_train):,}, test {len(X_test):,}")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("churn-capstone")
    with mlflow.start_run(run_name=f"churn {MODEL_VERSION}"):
        mlflow.set_tag("data_sha256", data_fingerprint(X_train))
        mlflow.log_params({"n_trials": args.n_trials, "seed": args.seed, "n_train": len(X_train)})

        # 1. Baseline to beat
        base_p = cross_val_predict(build_baseline(), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        base_auc = roc_auc_score(y_train, base_p)
        mlflow.log_metric("baseline_cv_auc", base_auc)
        print(f"baseline logistic regression: CV AUC {base_auc:.4f}")

        # 2. Tune gradient boosting with Optuna (TPE + median pruning), one nested MLflow run per trial
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=args.seed),
                                    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1))
        study.optimize(objective_factory(X_train, y_train, cv), n_trials=args.n_trials)
        pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        best = study.best_params
        print(f"optuna: {args.n_trials} trials ({pruned} pruned); best CV AUC {study.best_value:.4f}")
        print("best params:", {k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()})
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_auc", study.best_value)

        # 3. Choose the decision threshold from out-of-fold predictions (not the test set)
        oof = cross_val_predict(build_pipeline(best), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        threshold, cv_profit = best_threshold(y_train, oof)
        print(f"threshold {threshold:.2f} (CV expected profit USD {cv_profit:.2f} per customer)")

        # 4. Fit on all training data and evaluate once on the test set
        pipe = build_pipeline(best).fit(X_train, y_train)
        p_test = pipe.predict_proba(X_test)[:, 1]
        flagged = p_test >= threshold
        metrics = {
            "test_auc": roc_auc_score(y_test, p_test),
            "test_avg_precision": average_precision_score(y_test, p_test),
            "test_brier": brier_score_loss(y_test, p_test),
            "test_flag_rate": float(flagged.mean()),
            "test_profit_per_customer": float(np.mean(np.where(
                flagged, y_test * OFFER_SUCCESS * CUSTOMER_VALUE - OFFER_COST, 0.0))),
        }
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        mlflow.log_metrics(metrics)
        mlflow.log_metric("threshold", threshold)
        print("test:", metrics)

        # 5. Save the artifact, its metadata, and a drift reference
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipe, out / "model.joblib")
        reference = X_train.sample(min(2000, len(X_train)), random_state=args.seed)
        reference.assign(score=pipe.predict_proba(reference)[:, 1].round(5)).to_csv(out / "reference.csv", index=False)
        meta = {
            "name": "churn", "version": MODEL_VERSION, "threshold": threshold,
            "features": {"numeric": NUMERIC, "categorical": CATEGORICAL}, "schema": SCHEMA,
            "best_params": best, "metrics": metrics, "baseline_cv_auc": round(base_auc, 4),
            "best_cv_auc": round(study.best_value, 4), "training_rows": len(X_train),
            "training_data_sha256": data_fingerprint(X_train),
            "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                         "pandas": pd.__version__, "numpy": np.__version__},
        }
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_artifacts(str(out), artifact_path="model_files")
        # MLflow saves scikit-learn models with skops, which refuses types it can't vet unless you
        # vouch for them. We created this model ourselves, so we trust its tree predictor type.
        info = mlflow.sklearn.log_model(
            pipe, name="model", input_example=X_train.head(3),
            pip_requirements=[f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}"],
            skops_trusted_types=["numpy.dtype", "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])
        version = mlflow.register_model(info.model_uri, "churn-capstone").version
        print(f"saved {out}/ (model.joblib, metadata.json, reference.csv); "
              f"registered churn-capstone version {version}")


if __name__ == "__main__":
    main()
