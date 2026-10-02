# Level 1 Capstone: Linear Regression Three Ways, from Scratch

> **Level 1 · Capstone** · ⏱️ 8–12 h · Prerequisites: all of [Level 1: Math foundations](../chapters/01-math-foundations/index.md)

Fit the same linear regression three ways, using only NumPy for the core work: with the **normal equations** (linear algebra), with **gradient descent** (calculus and optimization), and with **statistical inference** (standard errors, t-statistics, p-values, and confidence intervals). Then check your inference against statsmodels and by simulation. The three approaches must agree, and you must be able to explain why.

## Scenario

You've joined a small home-energy startup as its first data scientist. The company wants to tell customers which factors drive their monthly electricity bill, and by how much, with honest error bars. The product manager, Priya, has three questions:

1. How much does each extra square meter of floor area, each extra occupant, and each degree of outdoor temperature add to (or subtract from) the monthly bill?
2. How sure are we? Priya wants a range, not just a number, for each effect.
3. The sales team claims that homes closer to the coast have higher bills. Is there any evidence for that?

The production data isn't ready yet, so the engineering lead, Wei, has written a simulator with known true coefficients. That's an advantage: because you know the truth, you can check whether your methods recover it, and whether your confidence intervals are as trustworthy as they claim.

Use exactly this data generator (copy it into your code):

```python
import numpy as np

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


X, y = make_data()
print(X.shape, y.shape)
print(np.round(X[:3], 2))
print(np.round(y[:3], 2))
```

```text
(200, 5) (200,)
[[  1.   129.14   1.    25.01  49.12]
 [  1.    88.8    2.    10.83  47.65]
 [  1.   142.51   2.    11.52   3.59]]
[80.58 75.81 95.71]
```

The first column of `X` is all ones, so the first coefficient is the intercept. The model is

$$
y_i = \beta_0 + \beta_1\,\text{area}_i + \beta_2\,\text{occupants}_i + \beta_3\,\text{temp}_i + \beta_4\,\text{dist}_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma^2),
$$

or $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ in matrix form, with $n = 200$ homes and $p = 5$ coefficients. Pretend you don't know `TRUE_BETA` or `NOISE_SD` until you're asked to compare against them.

## Tasks

Use NumPy and SciPy for everything except the explicit checks against statsmodels and `np.linalg.lstsq`. Don't use scikit-learn.

### Part A: The normal equations (linear algebra)

1. **Derive** the normal equations $\mathbf{X}^\top\mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\mathbf{y}$ twice: once from projection geometry (the residual is orthogonal to the column space) and once from calculus (set the gradient of the squared error to zero). Write both derivations in your report.
2. Check that $\mathbf{X}$ has full column rank. Explain what would go wrong if it didn't.
3. Implement `fit_normal_equations(X, y)` with `np.linalg.solve` (not `inv`). Compare the result with `np.linalg.lstsq`.
4. Verify numerically that the residuals are orthogonal to every column of $\mathbf{X}$, and compute $R^2 = 1 - \frac{\sum_i r_i^2}{\sum_i (y_i - \bar{y})^2}$.

### Part B: Gradient descent (calculus and optimization)

1. **Derive** the gradient and the Hessian of the mean squared error $\mathcal{L}(\boldsymbol{\beta}) = \frac{1}{n}\lVert \mathbf{X}\boldsymbol{\beta} - \mathbf{y} \rVert^2$.
2. Implement the loss and its gradient, and **gradient-check** the gradient against central finite differences at a random point. Report the relative error.
3. Compute the condition number of the Hessian on the raw features and on standardized features (every column except the intercept scaled to mean 0 and standard deviation 1). Explain what the difference means for gradient descent.
4. Implement gradient descent on the standardized features, choosing the learning rate from the Hessian's largest eigenvalue. Stop when the gradient's norm falls below $10^{-10}$. Plot the loss against the iteration number on a log scale, and report the number of iterations.
5. Convert the standardized coefficients back to the original units, and confirm they match the normal-equation solution.
6. Run gradient descent on the raw, unstandardized features for 10,000 iterations with the largest safe learning rate. How close does it get? Explain the result with the condition number.

### Part C: Statistical inference

1. **Derive** the covariance of the estimator, $\operatorname{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$, under the model's assumptions (fixed $\mathbf{X}$, independent errors with mean 0 and variance $\sigma^2$).
2. Estimate the noise variance with $\hat{\sigma}^2 = \frac{\text{RSS}}{n - p}$, where RSS is the residual sum of squares. Explain why the denominator is $n - p$.
3. For every coefficient, compute the standard error, the t-statistic for $H_0: \beta_j = 0$, the two-sided p-value from the t distribution with $n - p$ degrees of freedom, and a 95% confidence interval. Present a table.
4. Fit the same model with `statsmodels.api.OLS(y, X).fit()` and show that your coefficients, standard errors, t-statistics, p-values, and confidence intervals all match to numerical precision.
5. Answer Priya's three questions in plain language, in USD per month, with ranges.

### Part D: Check the inference by simulation

1. Generate 2,000 independent datasets with `make_data(200, seed=1000 + r)` for `r = 0, ..., 1999`. For each, compute the 95% confidence intervals.
2. For each coefficient, report the fraction of intervals that contain the true value (the **coverage**), the standard deviation of the 2,000 estimates, and the average of the 2,000 standard errors.
3. Do the intervals deliver what they promise? Does the standard error formula describe the actual spread of the estimates?

### Stretch goals (optional)

- Compute **bootstrap** standard errors (resample homes, refit) for the original dataset and compare them with the formula.
- Change `make_data` so the noise standard deviation grows with floor area (**heteroscedasticity**). Rerun Part D. What happens to coverage? Compare with statsmodels' robust standard errors, `fit(cov_type="HC3")`.
- Add a column `area_ft2 = area_m2 * 10.7639` to $\mathbf{X}$ and see what each of the three methods does.

## Deliverables

1. A script or notebook, `linreg_three_ways.py` (or `.ipynb`), that runs top to bottom with fixed seeds and prints every number in your report.
2. A comparison table of the coefficients from the normal equations, gradient descent, `lstsq`, and statsmodels.
3. A plot of the gradient descent loss curve on a log scale.
4. The inference table from Part C, and the simulation table from Part D.
5. A short written report (about one page) containing your three derivations (Parts A1, B1, and C1), your answers to Priya's questions, and your conclusions from Parts B6 and D.

## Acceptance checklist

- [ ] `X` has rank 5, and the normal-equation solution matches `np.linalg.lstsq` with `np.allclose`.
- [ ] The residuals are orthogonal to all columns: $\max_j \lvert (\mathbf{X}^\top\mathbf{r})_j \rvert$ is tiny relative to the size of $\mathbf{X}^\top\mathbf{y}$.
- [ ] The gradient check's relative error is below $10^{-7}$.
- [ ] The learning rate is justified from the Hessian's eigenvalues, and the loss never increases.
- [ ] Gradient descent on standardized features converges, and its coefficients (in original units) match the normal equations to within $10^{-6}$.
- [ ] You explain, using the condition number, why gradient descent on raw features is still far from the solution after 10,000 iterations.
- [ ] Your standard errors, t-statistics, p-values, and confidence intervals match statsmodels (`np.allclose` returns `True` for each).
- [ ] The derivation of $\operatorname{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2(\mathbf{X}^\top\mathbf{X})^{-1}$ is correct and states its assumptions.
- [ ] Simulated coverage is between 93.5% and 96.5% for every coefficient, and the mean standard error is close to the empirical standard deviation of the estimates.
- [ ] Priya's three questions are answered in plain language, with units (USD per month) and confidence intervals, and the coastal-distance claim is addressed correctly: "no evidence of an effect" is not the same as "proof of no effect".
- [ ] Everything runs from a clean start with fixed seeds.

## Hints

??? tip "Hint for Part A: two derivations"

    Geometry: predictions $\mathbf{X}\boldsymbol{\beta}$ live in the column space of $\mathbf{X}$. The closest point to $\mathbf{y}$ is the one where the residual $\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}$ is perpendicular to every column, so $\mathbf{X}^\top(\mathbf{y} - \mathbf{X}\hat{\boldsymbol{\beta}}) = \mathbf{0}$. Calculus: expand $\lVert \mathbf{X}\boldsymbol{\beta} - \mathbf{y} \rVert^2$ and use $\nabla(\mathbf{a}^\top\boldsymbol{\beta}) = \mathbf{a}$ and $\nabla(\boldsymbol{\beta}^\top\mathbf{A}\boldsymbol{\beta}) = 2\mathbf{A}\boldsymbol{\beta}$ for symmetric $\mathbf{A}$. See [Linear algebra](../chapters/01-math-foundations/01-linear-algebra.md) and [Calculus and gradients](../chapters/01-math-foundations/03-calculus-and-gradients.md).

??? tip "Hint for Part B: choosing the learning rate"

    The MSE is a quadratic with constant Hessian $\mathbf{H} = \frac{2}{n}\mathbf{X}^\top\mathbf{X}$. Gradient descent multiplies the error along each eigenvector by $1 - \eta\lambda_i$, so it's stable for $\eta < 2/\lambda_{\max}$, and $\eta = 1/\lambda_{\max}$ is a safe, fast choice. Use `np.linalg.eigvalsh` because $\mathbf{H}$ is symmetric. See [Information theory and optimization](../chapters/01-math-foundations/06-information-theory-and-optimization.md).

??? tip "Hint for Part B: converting standardized coefficients back"

    If $z_j = (x_j - \mu_j)/s_j$ and the fitted model is $w_0 + \sum_j w_j z_j$, substitute and collect terms: the slope on $x_j$ is $w_j/s_j$, and the intercept is $w_0 - \sum_j w_j\mu_j/s_j$.

??? tip "Hint for Part C: the covariance of the estimator"

    Substitute $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ into $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ to get $\hat{\boldsymbol{\beta}} = \boldsymbol{\beta} + \mathbf{A}\boldsymbol{\varepsilon}$ with $\mathbf{A} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$. Then use $\operatorname{Cov}(\mathbf{A}\boldsymbol{\varepsilon}) = \mathbf{A}\operatorname{Cov}(\boldsymbol{\varepsilon})\mathbf{A}^\top$ and $\operatorname{Cov}(\boldsymbol{\varepsilon}) = \sigma^2\mathbf{I}$.

??? tip "Hint for Part C: p-values and intervals"

    `scipy.stats.t.sf(abs(t), df)` gives one tail; double it for a two-sided p-value. The 95% interval uses `scipy.stats.t.ppf(0.975, df)` as the critical value, with `df = n - p`. statsmodels' `conf_int()` returns the same intervals. See [Statistics](../chapters/01-math-foundations/05-statistics.md).

??? tip "Hint for Part D: keeping it fast"

    Each simulated fit is one $5 \times 5$ solve and one $5 \times 5$ inverse, so 2,000 repetitions take a few seconds. Reuse your Part C function inside the loop, and accumulate a boolean array of "covered" per coefficient.

## Solution

When you're done, or truly stuck, compare with the [worked solution](solutions/level-1-capstone.md). The complete script is in the repository at `scripts/capstones/level-1/linreg_three_ways.py`.

## Next

You've finished Level 1. Move on to [Level 2: The data science workflow](../chapters/02-data-science-workflow/index.md).
