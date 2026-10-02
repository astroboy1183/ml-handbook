"""Level 1 capstone solution: linear regression three ways, from scratch.

1. Normal equations (linear algebra)
2. Gradient descent (calculus and optimization), with a gradient check
3. Statistical inference (standard errors, t-statistics, p-values, and
   confidence intervals), checked against statsmodels, plus a simulation
   that checks the coverage of the confidence intervals.

Run:
    python scripts/capstones/level-1/linreg_three_ways.py
"""

import numpy as np
import statsmodels.api as sm
from scipy import stats

NAMES = ["intercept", "area_m2", "occupants", "outdoor_temp_c", "dist_coast_km"]
TRUE_BETA = np.array([20.0, 0.5, 12.0, -1.5, 0.0])   # dist_coast_km has no real effect
NOISE_SD = 15.0


def make_data(n=200, seed=42):
    """Synthetic monthly electricity bills (USD) for n homes."""
    rng = np.random.default_rng(seed)
    area = rng.normal(120, 30, n).clip(40, None)       # floor area, m^2
    occupants = rng.poisson(1.5, n) + 1                # people living there
    temp = rng.normal(15, 8, n)                        # average outdoor temperature, deg C
    dist = rng.uniform(0, 50, n)                       # distance to the coast, km
    X = np.column_stack([np.ones(n), area, occupants, temp, dist])
    y = X @ TRUE_BETA + rng.normal(0, NOISE_SD, n)
    return X, y


# ---------------------------------------------------------------- Part A
def fit_normal_equations(X, y):
    """Least squares via the normal equations X^T X beta = X^T y."""
    return np.linalg.solve(X.T @ X, X.T @ y)


def r_squared(X, y, beta):
    resid = y - X @ beta
    return 1 - resid @ resid / np.sum((y - y.mean()) ** 2)


# ---------------------------------------------------------------- Part B
def mse(w, X, y):
    r = X @ w - y
    return r @ r / len(y)


def mse_grad(w, X, y):
    return 2 / len(y) * X.T @ (X @ w - y)


def numerical_grad(f, w, h=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w)
        e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)
    return g


def grad_check(X, y, seed=0):
    w = np.random.default_rng(seed).normal(size=X.shape[1])
    g_a = mse_grad(w, X, y)
    g_n = numerical_grad(lambda v: mse(v, X, y), w)
    return np.linalg.norm(g_a - g_n) / (np.linalg.norm(g_a) + np.linalg.norm(g_n))


def standardize(X):
    """Standardize every column except the intercept; return the scaled matrix and the scaling."""
    mu, sd = X[:, 1:].mean(axis=0), X[:, 1:].std(axis=0)
    return np.column_stack([np.ones(len(X)), (X[:, 1:] - mu) / sd]), mu, sd


def unstandardize(w, mu, sd):
    """Map coefficients fitted on standardized features back to the original units."""
    slopes = w[1:] / sd
    intercept = w[0] - np.sum(w[1:] * mu / sd)
    return np.r_[intercept, slopes]


def gradient_descent(X, y, eta, tol=1e-10, max_iter=100_000):
    w = np.zeros(X.shape[1])
    losses = [mse(w, X, y)]
    for it in range(1, max_iter + 1):
        g = mse_grad(w, X, y)
        w = w - eta * g
        losses.append(mse(w, X, y))
        if np.linalg.norm(g) < tol:
            break
    return w, np.array(losses), it


# ---------------------------------------------------------------- Part C
def ols_inference(X, y, level=0.95):
    n, p = X.shape
    beta = fit_normal_equations(X, y)
    resid = y - X @ beta
    df = n - p
    sigma2 = resid @ resid / df                         # unbiased noise variance
    cov = sigma2 * np.linalg.inv(X.T @ X)               # Cov(beta_hat) = sigma^2 (X^T X)^-1
    se = np.sqrt(np.diag(cov))
    t = beta / se
    pvals = 2 * stats.t.sf(np.abs(t), df)
    crit = stats.t.ppf(0.5 + level / 2, df)
    ci = np.column_stack([beta - crit * se, beta + crit * se])
    return {"beta": beta, "se": se, "t": t, "p": pvals, "ci": ci, "sigma2": sigma2, "df": df}


def coverage_simulation(reps=2000, n=200, level=0.95):
    covered = np.zeros(len(TRUE_BETA))
    betas, ses = [], []
    for r in range(reps):
        X, y = make_data(n, seed=1000 + r)
        res = ols_inference(X, y, level)
        covered += (res["ci"][:, 0] <= TRUE_BETA) & (TRUE_BETA <= res["ci"][:, 1])
        betas.append(res["beta"])
        ses.append(res["se"])
    return covered / reps, np.std(betas, axis=0, ddof=1), np.mean(ses, axis=0)


def main():
    np.set_printoptions(suppress=True)
    X, y = make_data()
    print(f"data: X {X.shape}, y {y.shape}, mean bill {y.mean():.2f} USD")
    print(f"rank of X: {np.linalg.matrix_rank(X)}")

    # Part A
    beta_ne = fit_normal_equations(X, y)
    beta_ls = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_ne
    print("\n== Part A: normal equations ==")
    for name, b in zip(NAMES, beta_ne):
        print(f"  {name:15s} {b:10.4f}")
    print(f"  agrees with lstsq: {np.allclose(beta_ne, beta_ls)}")
    print(f"  max |X^T r|: {np.abs(X.T @ resid).max():.2e}  (residuals orthogonal to columns)")
    print(f"  R^2: {r_squared(X, y, beta_ne):.4f}")

    # Part B
    print("\n== Part B: gradient descent ==")
    print(f"  gradient check relative error: {grad_check(X, y):.2e}")
    H_raw = 2 / len(y) * X.T @ X
    Xs, mu, sd = standardize(X)
    H_std = 2 / len(y) * Xs.T @ Xs
    print(f"  condition number of the Hessian: raw {np.linalg.cond(H_raw):.2e}, standardized {np.linalg.cond(H_std):.2f}")
    eta = 1 / np.linalg.eigvalsh(H_std).max()
    w, losses, iters = gradient_descent(Xs, y, eta)
    beta_gd = unstandardize(w, mu, sd)
    print(f"  eta = 1/lambda_max = {eta:.4f}; converged in {iters} iterations")
    print(f"  loss never increased: {bool(np.all(np.diff(losses) <= 1e-12))}")
    print(f"  max |beta_gd - beta_ne|: {np.abs(beta_gd - beta_ne).max():.2e}")
    eta_raw = 1 / np.linalg.eigvalsh(H_raw).max()
    w_raw, _, _ = gradient_descent(X, y, eta_raw, max_iter=10_000)
    print(f"  raw features, 10,000 iterations: max |beta - beta_ne| = {np.abs(w_raw - beta_ne).max():.2f}")

    # Part C
    print("\n== Part C: statistical inference ==")
    res = ols_inference(X, y)
    print(f"  sigma_hat = {np.sqrt(res['sigma2']):.3f} (true {NOISE_SD}), df = {res['df']}")
    print(f"  {'term':15s} {'coef':>9s} {'se':>8s} {'t':>8s} {'p':>9s}   95% CI")
    for i, name in enumerate(NAMES):
        print(f"  {name:15s} {res['beta'][i]:9.4f} {res['se'][i]:8.4f} {res['t'][i]:8.2f} "
              f"{res['p'][i]:9.2e}   ({res['ci'][i, 0]:.3f}, {res['ci'][i, 1]:.3f})")
    sm_fit = sm.OLS(y, X).fit()
    checks = {
        "coefficients": np.allclose(res["beta"], sm_fit.params),
        "standard errors": np.allclose(res["se"], sm_fit.bse),
        "t-statistics": np.allclose(res["t"], sm_fit.tvalues),
        "p-values": np.allclose(res["p"], sm_fit.pvalues),
        "confidence intervals": np.allclose(res["ci"], sm_fit.conf_int(0.05)),
    }
    for k, v in checks.items():
        print(f"  matches statsmodels {k}: {v}")

    # Part D
    print("\n== Part D: does the 95% CI cover the truth 95% of the time? ==")
    cover, emp_sd, mean_se = coverage_simulation()
    for i, name in enumerate(NAMES):
        print(f"  {name:15s} coverage {cover[i]:.3f}   sd of estimates {emp_sd[i]:.4f}   mean SE {mean_se[i]:.4f}")


if __name__ == "__main__":
    main()
