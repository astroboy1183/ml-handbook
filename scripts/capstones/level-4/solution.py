"""Level 4 capstone solution: a model bake-off on synthetic loan-default data.

Usage:
    python solution.py      # prints every result and writes figures/fig1.png, figures/fig2.png

Uses make_loan_data.py from the same directory. Mirrors docs/exercises/solutions/level-4-capstone.md.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_loan_data import make_loans  # noqa: E402

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (train_test_split, RepeatedStratifiedKFold, StratifiedKFold,
                                     GridSearchCV, cross_validate)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

FIG_DIR = Path(__file__).resolve().parent / "figures"
_fig_count = [0]


def save_figure():
    """Save the current figure to figures/fig<N>.png instead of showing it."""
    _fig_count[0] += 1
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / f"fig{_fig_count[0]}.png", dpi=110, bbox_inches="tight")
    plt.close("all")


df = make_loans()
print(df.shape, "| default rate:", round(df["default"].mean(), 3))
print("missing shares:", df.isna().mean()[lambda s: s > 0].round(3).to_dict())
print("default rate by purpose:", df.groupby("purpose")["default"].mean().round(3).to_dict())
print("default rate by term:", df.groupby("term_months")["default"].mean().round(3).to_dict())

TARGET = "default"
cat_cols = ["home_ownership", "purpose"]
num_cols = [c for c in df.columns if c not in cat_cols + [TARGET]]
X = df[cat_cols + num_cols].astype({c: float for c in num_cols})   # floats everywhere (partial dependence needs it)
y = df[TARGET].to_numpy()

X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)
outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
print(f"development rows {len(X_dev)}, test rows {len(X_test)}, "
      f"default rate dev {y_dev.mean():.3f} / test {y_test.mean():.3f}")

def scaled_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num_cols)])

def spline_prep():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                              SplineTransformer(n_knots=5, degree=3), StandardScaler()), num_cols)])

def tree_prep():
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols)])

def hgb_prep():                                    # categories first, NaN passed through
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", "passthrough", num_cols)])

models = {
    "baseline (prior)": (Pipeline([("prep", scaled_prep()), ("clf", DummyClassifier(strategy="prior"))]), {}),
    "logistic L2": (Pipeline([("prep", scaled_prep()), ("clf", LogisticRegression(max_iter=2000))]),
                    {"clf__C": [0.01, 0.1, 1, 10]}),
    "logistic L1": (Pipeline([("prep", scaled_prep()),
                              ("clf", LogisticRegression(l1_ratio=1.0, solver="liblinear", max_iter=2000))]),
                    {"clf__C": [0.01, 0.1, 1]}),
    "logistic L2 + splines": (Pipeline([("prep", spline_prep()), ("clf", LogisticRegression(max_iter=5000))]),
                              {"clf__C": [0.1, 1]}),
    "kNN": (Pipeline([("prep", scaled_prep()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [25, 75, 150]}),
    "SVM (RBF)": (Pipeline([("prep", scaled_prep()), ("clf", SVC(kernel="rbf"))]),
                  {"clf__C": [0.3, 1, 3], "clf__gamma": ["scale", 0.01]}),
    "random forest": (Pipeline([("prep", tree_prep()),
                                ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=0))]),
                      {"clf__min_samples_leaf": [1, 5, 20]}),
    "gradient boosting (HGB)": (Pipeline([("prep", hgb_prep()),
                                          ("clf", HistGradientBoostingClassifier(
                                              categorical_features=[0, 1], max_iter=500, early_stopping=True,
                                              validation_fraction=0.15, n_iter_no_change=30, random_state=0))]),
                                {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [7, 15]}),
}
print(len(models), "models;", "grid sizes:", {k: int(np.prod([len(v) for v in g.values()])) for k, (_, g) in models.items()})

cv_results, fold_auc, chosen = {}, {}, {}
for name, (pipe, grid) in models.items():
    est = GridSearchCV(pipe, grid, cv=inner, scoring="roc_auc") if grid else pipe
    scoring = {"auc": "roc_auc", "ap": "average_precision"}
    if name != "SVM (RBF)":
        scoring["brier"] = "neg_brier_score"
    t0 = time.perf_counter()
    res = cross_validate(est, X_dev, y_dev, cv=outer, scoring=scoring, n_jobs=4, return_estimator=bool(grid))
    wall = time.perf_counter() - t0
    fold_auc[name] = res["test_auc"]
    cv_results[name] = {
        "AUC": f"{res['test_auc'].mean():.3f} ± {res['test_auc'].std():.3f}",
        "avg precision": f"{res['test_ap'].mean():.3f} ± {res['test_ap'].std():.3f}",
        "Brier": f"{-res['test_brier'].mean():.4f}" if "test_brier" in res else "n/a",
        "fit+tune s/fold": round(res["fit_time"].mean(), 2),
    }
    if grid:
        chosen[name] = pd.Series([str(e.best_params_) for e in res["estimator"]]).value_counts().to_dict()
table = pd.DataFrame(cv_results).T.sort_values("AUC", ascending=False)
print(table.to_string())

for name, counts in chosen.items():
    print(f"{name:<25} {counts}")

order = table.index[::-1]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.boxplot([fold_auc[n] for n in order], orientation="horizontal", tick_labels=list(order))
for k, n in enumerate(order, start=1):
    ax.scatter(fold_auc[n], np.full(10, k) + np.random.default_rng(k).uniform(-0.12, 0.12, 10), s=12, color="k", alpha=0.6)
ax.set_xlabel("ROC AUC on each of the 10 outer folds")
ax.set_title("Nested CV: four models are tied at the top")
plt.tight_layout()
save_figure()

def corrected_ttest(a, b, n_train, n_test):
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * d.var(ddof=1))
    return d.mean(), t, 2 * stats.t.sf(abs(t), df=J - 1)

best = table.index[0]
n_test_fold = len(X_dev) // 5
n_train_fold = len(X_dev) - n_test_fold
print(f"reference model: {best}")
print(f"{'compared with':<25}{'mean diff':>10}{'corrected p':>13}{'naive p':>10}")
for name in table.index[1:]:
    diff, t, p = corrected_ttest(fold_auc[best], fold_auc[name], n_train_fold, n_test_fold)
    naive_p = stats.ttest_rel(fold_auc[best], fold_auc[name]).pvalue
    print(f"{name:<25}{diff:>10.4f}{p:>13.4f}{naive_p:>10.4f}")

finalists = ["logistic L2", "logistic L2 + splines", "random forest", "gradient boosting (HGB)"]
final = {}
for name in finalists:
    pipe, grid = models[name]
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").fit(X_dev, y_dev)
    t0 = time.perf_counter(); model = clone(gs.best_estimator_).fit(X_dev, y_dev); fit_s = time.perf_counter() - t0
    t0 = time.perf_counter(); p = model.predict_proba(X_test)[:, 1]; pred_ms = 1000 * (time.perf_counter() - t0)
    final[name] = {"model": model, "p": p, "params": gs.best_params_, "fit_s": fit_s, "pred_ms": pred_ms}

rng = np.random.default_rng(0)
boot = [rng.integers(0, len(y_test), len(y_test)) for _ in range(1000)]
def capture_at(p, share=0.2):
    top = np.argsort(-p)[: int(share * len(p))]
    return y_test[top].sum() / y_test.sum()

print(f"{'model':<25}{'test AUC':>9}{'95% bootstrap CI':>19}{'Brier':>8}{'capture@20%':>13}{'fit s':>7}{'predict ms':>11}")
for name, f in final.items():
    aucs = [roc_auc_score(y_test[b], f["p"][b]) for b in boot]
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    print(f"{name:<25}{roc_auc_score(y_test, f['p']):>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>19}"
          f"{brier_score_loss(y_test, f['p']):>8.4f}{capture_at(f['p']):>13.3f}{f['fit_s']:>7.2f}{f['pred_ms']:>11.1f}")
for name, f in final.items():
    print(f"  {name}: {f['params']}")

a, b = final["gradient boosting (HGB)"]["p"], final["logistic L2 + splines"]["p"]
diffs = np.array([roc_auc_score(y_test[i], a[i]) - roc_auc_score(y_test[i], b[i]) for i in boot])
print(f"HGB minus spline logistic, test AUC: {roc_auc_score(y_test, a) - roc_auc_score(y_test, b):+.4f}, "
      f"95% paired bootstrap CI [{np.percentile(diffs, 2.5):+.4f}, {np.percentile(diffs, 97.5):+.4f}]")

lr = final["logistic L2"]["model"]
names = lr.named_steps["prep"].get_feature_names_out()
coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=names)
print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10).round(3).to_string())

hgb = final["gradient boosting (HGB)"]["model"]
pi = permutation_importance(hgb, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=0, n_jobs=2)
imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False)
print("permutation importance (drop in test AUC):")
print(imp.round(4).to_string())

fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
PartialDependenceDisplay.from_estimator(hgb, X_dev, ["credit_score", "utilization", "debt_to_income", "late_payments_2y"],
                                        ax=ax, response_method="predict_proba", grid_resolution=40)
fig.suptitle("Gradient boosting: partial dependence of predicted default probability", y=1.03)
plt.tight_layout()
save_figure()
