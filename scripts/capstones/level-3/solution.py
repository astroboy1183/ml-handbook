"""Level 3 capstone solution: a churn classifier from scratch in NumPy, matched with scikit-learn.

Usage:
    cd scripts/capstones/level-3
    python solution.py        # prints every result and writes figures/fig1.png, figures/fig2.png

This is the code from docs/exercises/solutions/level-3-capstone.md, as one script.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from make_churn_data import make_churn_data  # noqa: E402

FIG_DIR = Path(__file__).resolve().with_name("figures")
_fig_count = [0]


def _save_figure(*args, **kwargs):
    """Replace plt.show(): save each figure to figures/figN.png instead of opening a window."""
    _fig_count[0] += 1
    FIG_DIR.mkdir(exist_ok=True)
    plt.gcf().savefig(FIG_DIR / f"fig{_fig_count[0]}.png", dpi=110, bbox_inches="tight", facecolor="white")
    plt.close("all")


plt.show = _save_figure

# ---- Part A: splits and exploration ----
print('\nPart A: splits and exploration')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

df = make_churn_data()
NUM = ["tenure_months", "monthly_charges", "num_products", "support_tickets_90d",
       "satisfaction_score", "days_since_last_login", "age"]
CAT = ["contract", "payment_method", "internet_service", "region", "autopay"]


def prepare(frame):
    """Model columns only; autopay stored as text so it's treated as a category."""
    out = frame[NUM + CAT].copy()
    out["autopay"] = np.where(frame["autopay"], "yes", "no")
    return out, frame["churned"].to_numpy()


idx = np.arange(len(df))
idx_rest, idx_te = train_test_split(idx, test_size=0.2, stratify=df["churned"], random_state=0)
idx_tr, idx_va = train_test_split(idx_rest, test_size=0.25, stratify=df["churned"].to_numpy()[idx_rest],
                                  random_state=0)
X_tr_df, y_tr = prepare(df.iloc[idx_tr])
X_va_df, y_va = prepare(df.iloc[idx_va])
X_te_df, y_te = prepare(df.iloc[idx_te])
for name, yy in [("train", y_tr), ("validation", y_va), ("test", y_te)]:
    print(f"{name:>10}: {len(yy):>5} customers, churn rate {yy.mean():.3f}")

train = df.iloc[idx_tr]                                   # exploration uses the training set only
print("\nmissing share (train):", train[NUM].isna().mean()[lambda s: s > 0].round(3).to_dict())
print("churn rate by contract:", train.groupby("contract")["churned"].mean().round(3).to_dict())
sat_missing = train["satisfaction_score"].isna()
print(f"churn rate, satisfaction answered: {train.loc[~sat_missing, 'churned'].mean():.3f}, "
      f"missing: {train.loc[sat_missing, 'churned'].mean():.3f}")


# ---- Part B: preprocessing from scratch ----
print('\nPart B: preprocessing from scratch')
class Preprocessor:
    """Median imputation, missing indicators, and standardization for numeric columns;
    one-hot encoding with the first (alphabetical) level dropped for categorical columns.
    Every statistic is learned in fit() from the training data only."""

    def fit(self, frame):
        self.medians_ = {c: np.nanmedian(self._col(frame, c)) for c in NUM}
        self.missing_ = [c for c in NUM if np.isnan(self._col(frame, c)).any()]
        numeric = self._numeric(frame)
        self.mean_, self.std_ = numeric.mean(axis=0), numeric.std(axis=0)
        self.levels_ = {c: sorted(frame[c].unique()) for c in CAT}
        self.names_ = (NUM + [f"{c}_missing" for c in self.missing_]
                       + [f"{c}={lv}" for c in CAT for lv in self.levels_[c][1:]])
        return self

    @staticmethod
    def _col(frame, c):
        return frame[c].to_numpy(dtype=float, na_value=np.nan)

    def _numeric(self, frame):
        cols = []
        for c in NUM:
            v = self._col(frame, c)
            cols.append(np.where(np.isnan(v), self.medians_[c], v))
        indicators = [np.isnan(self._col(frame, c)).astype(float) for c in self.missing_]
        return np.column_stack(cols + indicators)

    def transform(self, frame):
        numeric = (self._numeric(frame) - self.mean_) / self.std_
        dummies = [(frame[c].to_numpy() == lv).astype(float) for c in CAT for lv in self.levels_[c][1:]]
        return np.column_stack([numeric] + dummies)


prep = Preprocessor().fit(X_tr_df)
X_tr, X_va, X_te = prep.transform(X_tr_df), prep.transform(X_va_df), prep.transform(X_te_df)
print("feature matrix:", X_tr.shape)
print("features:", prep.names_)
n_num = len(NUM) + len(prep.missing_)
print("train numeric means ~0:", np.allclose(X_tr[:, :n_num].mean(axis=0), 0),
      "| validation numeric means (first 3):", np.round(X_va[:, :3].mean(axis=0), 3))


# ---- Part C: logistic regression with L2, from scratch ----
print('\nPart C: logistic regression with L2, from scratch')
def add_intercept(X):
    return np.column_stack([np.ones(len(X)), X])


def objective(w, Xb, y, lam):
    z = Xb @ w
    return np.mean(np.logaddexp(0, z) - y * z) + 0.5 * lam * w[1:] @ w[1:]


def gradient(w, Xb, y, lam):
    g = Xb.T @ (expit(Xb @ w) - y) / len(y)
    g[1:] += lam * w[1:]
    return g


def numerical_gradient(f, w, h=1e-6):
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w); e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g


def fit_logreg(X, y, lam, tol=1e-8, max_iter=100_000):
    """L2-regularized logistic regression by gradient descent with step 1/L."""
    Xb = add_intercept(X)
    L = 0.25 * np.linalg.eigvalsh(Xb.T @ Xb / len(y))[-1] + lam
    w = np.zeros(Xb.shape[1])
    for it in range(max_iter):
        g = gradient(w, Xb, y, lam)
        if np.linalg.norm(g) < tol:
            break
        w -= g / L
    return w, it


def predict_proba(w, X):
    return expit(add_intercept(X) @ w)


# Gradient check at three random points
rng = np.random.default_rng(0)
Xb_tr = add_intercept(X_tr)
for k in range(3):
    w_rand = rng.normal(0, 0.5, Xb_tr.shape[1])
    ga = gradient(w_rand, Xb_tr, y_tr, 1e-2)
    gn = numerical_gradient(lambda w: objective(w, Xb_tr, y_tr, 1e-2), w_rand)
    print(f"gradient check {k + 1}: relative error = {np.linalg.norm(ga - gn) / np.linalg.norm(ga + gn):.1e}")

w_demo, iters = fit_logreg(X_tr, y_tr, lam=1e-3)
print(f"fit with lambda = 1e-3: {iters} iterations, objective {objective(w_demo, Xb_tr, y_tr, 1e-3):.5f}, "
      f"|gradient| = {np.linalg.norm(gradient(w_demo, Xb_tr, y_tr, 1e-3)):.1e}")


# ---- Part D: choosing the regularization strength ----
print('\nPart D: choosing the regularization strength')
def stratified_folds(y, k=5, seed=0):
    """Shuffle each class separately and deal its indices round-robin into k folds."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        members = rng.permutation(np.flatnonzero(y == cls))
        fold[members] = np.arange(len(members)) % k
    return [(np.flatnonzero(fold != f), np.flatnonzero(fold == f)) for f in range(k)]


def log_loss_np(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def roc_auc_np(y, s):
    """Probability that a random positive outranks a random negative (Mann-Whitney)."""
    r = rankdata(s)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


folds = stratified_folds(y_tr, k=5, seed=0)
print("positives per validation fold:", [int(y_tr[va].sum()) for _, va in folds])
lambdas = np.logspace(-5, 0, 11)
cv = {}
for lam in lambdas:
    losses, aucs = [], []
    for tr, va in folds:
        p_fold = Preprocessor().fit(X_tr_df.iloc[tr])
        w, _ = fit_logreg(p_fold.transform(X_tr_df.iloc[tr]), y_tr[tr], lam)
        p = predict_proba(w, p_fold.transform(X_tr_df.iloc[va]))
        losses.append(log_loss_np(y_tr[va], p)); aucs.append(roc_auc_np(y_tr[va], p))
    cv[lam] = (np.mean(losses), np.std(losses) / np.sqrt(len(folds)), np.mean(aucs))
    print(f"lambda = {lam:.1e}: CV log-loss {cv[lam][0]:.5f} (se {cv[lam][1]:.5f}), CV ROC-AUC {cv[lam][2]:.4f}")

best_lam = min(cv, key=lambda l: cv[l][0])
limit = cv[best_lam][0] + cv[best_lam][1]
one_se_lam = max(l for l in lambdas if cv[l][0] <= limit)
print(f"best lambda = {best_lam:.1e}; one-standard-error rule picks {one_se_lam:.1e}")

lam = best_lam
w_hat, iters = fit_logreg(X_tr, y_tr, lam)
print(f"final fit on the training set: {iters} iterations")


# ---- Part E: the threshold ----
print('\nPart E: the threshold')
VALUE_SAVED, OFFER_COST, P_STAY = 250.0, 30.0, 0.4


def profit(y, p, t):
    """Expected profit (USD) of calling everyone with p >= t, relative to calling nobody."""
    call = p >= t
    return P_STAY * VALUE_SAVED * np.sum(call & (y == 1)) - OFFER_COST * np.sum(call)


t_theory = OFFER_COST / (P_STAY * VALUE_SAVED)
p_va = predict_proba(w_hat, X_va)
grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
val_profit = np.array([profit(y_va, p_va, t) for t in grid])
t_best = grid[np.argmax(val_profit)]
print(f"theoretical threshold {t_theory:.2f}; best on validation {t_best:.2f} "
      f"(validation profit USD {val_profit.max():,.0f} vs USD {profit(y_va, p_va, t_theory):,.0f} at {t_theory:.2f})")


# ---- Part E: test-set evaluation ----
print('\nPart E: test-set evaluation')
def pr_points(y, s):
    """Precision and recall at every distinct score threshold, highest first."""
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(s) - 1]
    tp = np.cumsum(ys)[last]
    fp = (last + 1) - tp
    return tp / (last + 1), tp / y.sum(), fp / (len(y) - y.sum())


def average_precision_np(y, s):
    precision, recall, _ = pr_points(y, s)
    return np.sum(np.diff(np.r_[0, recall]) * precision)


def calibration_bins(y, p, n_bins=10):
    """Mean predicted probability and observed churn rate in quantile bins."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, n_bins - 1)
    return (np.array([p[which == b].mean() for b in range(n_bins)]),
            np.array([y[which == b].mean() for b in range(n_bins)]))


p_te = predict_proba(w_hat, X_te)
call = p_te >= t_best
tp, fp = np.sum(call & (y_te == 1)), np.sum(call & (y_te == 0))
fn, tn = np.sum(~call & (y_te == 1)), np.sum(~call & (y_te == 0))
mean_pred, frac_pos = calibration_bins(y_te, p_te)
test_metrics = {
    "ROC-AUC": roc_auc_np(y_te, p_te),
    "average precision (PR-AUC)": average_precision_np(y_te, p_te),
    "Brier score": np.mean((p_te - y_te) ** 2),
    "log-loss": log_loss_np(y_te, p_te),
}
for k_, v in test_metrics.items():
    print(f"{k_:>27}: {v:.4f}")
print(f"{'churn rate / mean predicted':>27}: {y_te.mean():.4f} / {p_te.mean():.4f}")
print(f"\nat threshold {t_best:.2f}: TP {tp}, FP {fp}, FN {fn}, TN {tn}; "
      f"precision {tp / (tp + fp):.3f}, recall {tp / (tp + fn):.3f}, customers called {call.sum()} of {len(y_te)}")
print(f"max calibration gap across 10 bins: {np.max(np.abs(mean_pred - frac_pos)):.3f}")

print("\ntest profit relative to calling nobody:")
for label, t in [("chosen threshold", t_best), ("theoretical 0.30", t_theory), ("default 0.5", 0.5),
                 ("call everybody", 0.0)]:
    print(f"{label:>18}: USD {profit(y_te, p_te, t):>8,.0f}")


# ---- Part E: figures ----
print('\nPart E: figures')
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
precision, recall, fpr = pr_points(y_te, p_te)
axes[0].plot(np.r_[0, fpr], np.r_[0, recall], lw=2, label=f"model: AUC {test_metrics['ROC-AUC']:.3f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="random")
axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate (recall)")
axes[0].set_title("ROC curve (test)", fontsize=10); axes[0].legend(fontsize=8)
axes[1].plot(recall, precision, lw=2, label=f"model: AP {test_metrics['average precision (PR-AUC)']:.3f}")
axes[1].axhline(y_te.mean(), color="k", ls="--", lw=1, label=f"random: {y_te.mean():.3f}")
axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
axes[1].set_title("precision-recall curve (test)", fontsize=10); axes[1].legend(fontsize=8)
axes[2].plot(mean_pred, frac_pos, "o-", label="model (10 quantile bins)")
axes[2].plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
axes[2].set_xlabel("mean predicted churn probability"); axes[2].set_ylabel("observed churn rate")
axes[2].set_title("reliability diagram (test)", fontsize=10); axes[2].legend(fontsize=8)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(grid, val_profit / 1000, lw=2)
ax.axvline(t_theory, color="tab:green", ls="--", label=f"theory: {t_theory:.2f}")
ax.axvline(t_best, color="tab:red", ls=":", label=f"best on validation: {t_best:.2f}")
ax.axvline(0.5, color="gray", ls="-.", label="default 0.5")
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("threshold (call if churn probability >= threshold)")
ax.set_ylabel("validation profit vs calling nobody (USD thousands)")
ax.set_title("profit curve", fontsize=10); ax.legend(fontsize=8)
plt.tight_layout()
plt.show()


# ---- Part F: matching scikit-learn ----
print('\nPart F: matching scikit-learn')
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.calibration import calibration_curve


def sk_pipeline(C):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), NUM),
        ("cat", OneHotEncoder(drop="first"), CAT),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, tol=1e-12, max_iter=10_000))])


C = 1 / (lam * len(y_tr))
sk = sk_pipeline(C).fit(X_tr_df, y_tr)
print(f"lambda = {lam:.1e}  ->  C = {C:.4f}")
print("features match:", np.allclose(sk["pre"].transform(X_te_df), X_te))
w_sk = np.r_[sk["clf"].intercept_, sk["clf"].coef_.ravel()]
p_sk = sk.predict_proba(X_te_df)[:, 1]
print(f"max |coefficient difference|: {np.abs(w_sk - w_hat).max():.1e}")
print(f"max |probability difference|: {np.abs(p_sk - p_te).max():.1e}")

sk_metrics = {"ROC-AUC": roc_auc_score(y_te, p_te), "average precision (PR-AUC)": average_precision_score(y_te, p_te),
              "Brier score": brier_score_loss(y_te, p_te), "log-loss": log_loss(y_te, p_te)}
for k_ in test_metrics:
    print(f"{k_:>27}: ours {test_metrics[k_]:.6f}   sklearn {sk_metrics[k_]:.6f}")
frac_sk, mean_sk = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
print("calibration bins match:", np.allclose(frac_sk, frac_pos) and np.allclose(mean_sk, mean_pred))

print("\ncross-validated log-loss with our folds, ours vs sklearn:")
for l in [1e-4, 1e-2, 1e-1]:
    s = -cross_val_score(sk_pipeline(1 / (l * len(folds[0][0]))), X_tr_df, y_tr, cv=folds, scoring="neg_log_loss")
    print(f"lambda = {l:.0e}: ours {cv[l][0]:.5f}   sklearn {s.mean():.5f}")


# ---- Which features matter? ----
print('\nWhich features matter?')
coefs = pd.Series(w_hat[1:], index=prep.names_).sort_values(key=np.abs, ascending=False)
print(coefs.round(3).head(8).to_string())


# ---- Summary for Priya ----
print('\nSummary for Priya')
called = int((p_te >= t_best).sum())
saved = profit(y_te, p_te, t_best)
print(f"Call customers whose predicted churn probability is at least {t_best:.0%}: "
      f"{called} of {len(y_te)} test customers ({called / len(y_te):.0%}).")
print(f"Expected profit on these {len(y_te)} customers: USD {saved:,.0f}, versus USD 0 for calling nobody and "
      f"USD {profit(y_te, p_te, 0.0):,.0f} for calling everybody.")
