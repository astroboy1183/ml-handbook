"""Level 5 capstone: a data-quality and drift report for a batch of production data.

Usage (after `python train.py`):
    python monitor.py --batch new_customers.csv        # check a real batch
    python monitor.py --simulate none                   # a simulated batch with no drift
    python monitor.py --simulate drift                  # prices up, usage down (data drift)
    python monitor.py --simulate bug                    # charges in cents and a renamed category

Compares the batch with model/reference.csv (a sample of training data with model scores):
data-quality checks against the training schema, PSI for every feature and for the model's
scores, and a two-sample KS test for numeric features. Writes drift_report.json, prints a
table, and exits with code 1 if any check is at ALERT level (so a scheduler can stop the
scoring job).
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from train import make_churn_data

PSI_WARN, PSI_ALERT = 0.10, 0.25          # common rules of thumb, not laws
MAX_MISSING, MAX_OUT_OF_RANGE = 0.10, 0.01


def psi_numeric(reference, current, bins=10, eps=1e-4):
    """Population stability index with quantile bins from the reference."""
    reference, current = reference.dropna().to_numpy(), current.dropna().to_numpy()
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(reference, edges)[0] / len(reference)
    a = np.histogram(current, edges)[0] / len(current)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_categorical(reference, current, eps=1e-4):
    cats = sorted(set(reference.dropna()) | set(current.dropna()))
    e = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    a = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def level(psi):
    return "ALERT" if psi > PSI_ALERT else "WARN" if psi > PSI_WARN else "OK"


def quality_checks(batch, meta):
    """Schema checks: required columns, missing rates, ranges, unseen categories."""
    issues = []
    cols = meta["features"]["numeric"] + meta["features"]["categorical"]
    missing_cols = [c for c in cols if c not in batch.columns]
    if missing_cols:
        return [{"check": "columns", "status": "ALERT", "detail": f"missing columns {missing_cols}"}]
    for col in meta["features"]["numeric"]:
        lo, hi = meta["schema"][col]
        miss = batch[col].isna().mean()
        out = (~batch[col].dropna().between(lo, hi)).mean()
        if miss > MAX_MISSING:
            issues.append({"check": f"{col} missing", "status": "ALERT", "detail": f"{miss:.1%} missing"})
        if out > MAX_OUT_OF_RANGE:
            issues.append({"check": f"{col} range", "status": "ALERT",
                           "detail": f"{out:.1%} outside [{lo}, {hi}]"})
    for col in meta["features"]["categorical"]:
        unseen = sorted(set(batch[col].dropna()) - set(meta["schema"][col]))
        if unseen:
            issues.append({"check": f"{col} categories", "status": "ALERT", "detail": f"unseen values {unseen}"})
    return issues


def drift_report(batch, reference, model, meta):
    rows = []
    for col in meta["features"]["numeric"]:
        p = psi_numeric(reference[col], batch[col])
        ks = stats.ks_2samp(reference[col].dropna(), batch[col].dropna())
        rows.append({"feature": col, "psi": round(p, 4), "ks_d": round(float(ks.statistic), 4),
                     "ks_p": float(f"{ks.pvalue:.2g}"), "status": level(p)})
    for col in meta["features"]["categorical"]:
        p = psi_categorical(reference[col], batch[col])
        rows.append({"feature": col, "psi": round(p, 4), "ks_d": None, "ks_p": None, "status": level(p)})
    cols = meta["features"]["numeric"] + meta["features"]["categorical"]
    scores = pd.Series(model.predict_proba(batch[cols])[:, 1])
    p = psi_numeric(reference["score"], scores)
    rows.append({"feature": "model score", "psi": round(p, 4), "ks_d": None, "ks_p": None, "status": level(p)})
    summary = {"mean_score_reference": round(float(reference["score"].mean()), 4),
               "mean_score_batch": round(float(scores.mean()), 4),
               "flag_rate_batch": round(float((scores >= meta["threshold"]).mean()), 4)}
    return rows, summary


def simulate(kind, n=2000, seed=99):
    if kind == "none":
        return make_churn_data(n=n, seed=seed)
    if kind == "drift":
        return make_churn_data(n=n, seed=seed, charges_shift=12.0, usage_scale=0.6)
    if kind == "bug":
        df = make_churn_data(n=n, seed=seed)
        df["monthly_charges"] = df["monthly_charges"] * 100           # dollars -> cents upstream
        df["plan"] = df["plan"].replace({"premium": "Premium"})        # a renamed category
        return df
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--batch", help="CSV of production rows with the model's feature columns")
    src.add_argument("--simulate", choices=["none", "drift", "bug"])
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--out", default="drift_report.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "metadata.json").read_text())
    model = joblib.load(model_dir / "model.joblib")
    reference = pd.read_csv(model_dir / "reference.csv")
    batch = pd.read_csv(args.batch) if args.batch else simulate(args.simulate)
    source = args.batch or f"simulated ({args.simulate})"

    issues = quality_checks(batch, meta)
    report = {"model_version": meta["version"], "batch": source, "rows": len(batch), "quality": issues}
    if issues:
        report["drift"], report["summary"] = [], {}
        report["status"] = "ALERT"
        report["action"] = "Block scoring and retraining; fix the data pipeline first."
    else:
        rows, summary = drift_report(batch, reference, model, meta)
        report["drift"], report["summary"] = rows, summary
        worst = max((r["status"] for r in rows), key=["OK", "WARN", "ALERT"].index)
        report["status"] = worst
        report["action"] = {"OK": "No action.",
                            "WARN": "Investigate the drifting features; watch performance as labels arrive.",
                            "ALERT": "Major drift: check impact on performance; consider retraining."}[worst]
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f"drift report for model {meta['version']} | batch: {source} | {len(batch):,} rows")
    if issues:
        for i in issues:
            print(f"  [{i['status']}] {i['check']}: {i['detail']}")
    else:
        table = pd.DataFrame(report["drift"]).set_index("feature")
        table["ks_p"] = table["ks_p"].map(lambda v: "-" if pd.isna(v) else f"{v:.2g}")
        print(table.to_string(na_rep="-"))
        print("summary:", report["summary"])
    print(f"status: {report['status']} -> {report['action']}")
    sys.exit(1 if report["status"] == "ALERT" else 0)


if __name__ == "__main__":
    main()
