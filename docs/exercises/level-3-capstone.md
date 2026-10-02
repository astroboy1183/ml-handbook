# Level 3 capstone: a churn classifier from scratch, matched with scikit-learn

> **Level 3 · Capstone** · ⏱️ 10–14 h · Prerequisites: all of [Level 3](../chapters/03-ml-fundamentals/index.md)

This capstone puts the whole level to work on one realistic problem. You'll build an L2-regularized logistic regression for customer churn entirely in NumPy, including the preprocessing, the optimizer, the gradient check, cross-validation, and the metrics. Then you'll choose a decision threshold that maximizes profit, evaluate the model honestly, and show that scikit-learn, given the same problem, produces the same answer. It's the gate to [Level 4](../chapters/04-ml-algorithms/index.md): finish it without notes before moving on.

## The scenario

You're the first data scientist at NetCo, a regional internet and TV provider with a churn problem: about one customer in five cancels in a given month. The retention team can call customers and offer them a one-month bill credit, and they want to know whom to call.

The Head of Retention, Priya, gives you the economics:

| Item | Value |
|---|---|
| Cost of a retention offer (credit plus agent time) | USD 30 per customer contacted |
| Chance that a would-be churner who gets the offer stays | 40% |
| Value of keeping a customer who would have churned | USD 250 in future margin |
| Customers who weren't going to churn | the offer is wasted (USD 30), nothing else changes |
| Customers not contacted | no cost, no gain (the baseline) |

Priya's requests:

1. "Tell me which customers to call next month, and how much money that's worth compared with calling nobody or calling everybody."
2. "The analytics vendor quoted us a black-box model. I want something we understand and can check. Build it from first principles and show me it agrees with a standard library."
3. "When the model says 30%, I want that to mean 30%. Finance will use these numbers to forecast."

## The data

Expand the generator, save it as `make_churn_data.py`, and run it (or run the cells below). It returns one row per customer with these columns:

| Column | Type | Meaning |
|---|---|---|
| `customer_id` | text | identifier (not a feature) |
| `tenure_months` | integer | months as a customer |
| `monthly_charges` | float | current monthly bill, USD |
| `contract` | category | month-to-month, one-year, or two-year |
| `payment_method` | category | credit card, bank transfer, electronic check, mailed check |
| `internet_service` | category | fiber, dsl, or none |
| `autopay` | boolean | whether the bill is paid automatically |
| `num_products` | integer | number of products subscribed |
| `support_tickets_90d` | integer | support tickets filed in the last 90 days |
| `satisfaction_score` | integer 1–5, nullable | last survey answer; missing if the customer didn't answer |
| `days_since_last_login` | float, nullable | days since the customer last used the account portal |
| `age` | integer, nullable | customer age |
| `region` | category | north, south, east, or west |
| `churned` | 0/1 | **the label**: cancelled within the next month |

Some values are missing, and not all of them are missing at random. Treat that as part of the problem.

??? example "The capstone data generator: make_churn_data.py (expand and run this first)"

    ```python
    """Generate the Level 3 capstone data: a synthetic churn data set for a subscription telecom company.

    Usage:
        python make_churn_data.py            # writes churn.csv in the current directory
        from make_churn_data import make_churn_data
        df = make_churn_data()               # a pandas DataFrame, one row per customer

    The data are synthetic, seeded, and realistically imperfect: mixed column types, missing
    values (some missing at random, some not), and about 20% churners.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd


    def make_churn_data(n=10_000, seed=42):
        """Return a DataFrame of n customers with a binary `churned` label (1 = cancelled next month)."""
        rng = np.random.default_rng(seed)

        contract = rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.25, 0.20])
        extra = np.select([contract == "one-year", contract == "two-year"], [12, 24], 0)
        tenure = np.clip(np.round(rng.gamma(1.5, 14, n) + extra), 1, 72).astype(int)
        internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.35, 0.20])
        n_products = 1 + rng.binomial(4, 0.3, n)
        monthly = (20 + 50 * (internet == "fiber") + 30 * (internet == "dsl") + 5 * n_products
                   + rng.normal(0, 8, n)).clip(15, None).round(2)
        payment = rng.choice(["credit card", "bank transfer", "electronic check", "mailed check"], n,
                             p=[0.30, 0.25, 0.30, 0.15])
        autopay = np.where(np.isin(payment, ["credit card", "bank transfer"]),
                           rng.random(n) < 0.7, rng.random(n) < 0.1)
        tickets = rng.poisson(0.4 + 0.6 * (internet == "fiber"), n)
        satisfaction = np.clip(np.round(3.6 - 0.4 * tickets + rng.normal(0, 1, n)), 1, 5)
        last_login = rng.exponential(10, n).round(1)
        age = np.clip(np.round(rng.normal(45, 14, n)), 18, 90)
        region = rng.choice(["north", "south", "east", "west"], n)

        logit = (-1.75 - 0.045 * tenure + 0.02 * (monthly - 65)
                 + 1.1 * (contract == "month-to-month") - 0.9 * (contract == "two-year")
                 + 0.45 * (payment == "electronic check") + 0.35 * (internet == "fiber")
                 + 0.30 * tickets - 0.4 * autopay + 0.035 * last_login
                 - 0.45 * (satisfaction - 3) - 0.10 * (n_products - 2) - 0.005 * (age - 45))
        churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

        # Missing values. Survey answers are missing more often for unhappy customers (not at random);
        # login tracking and age have gaps unrelated to anything (completely at random).
        sat_missing = rng.random(n) < np.where(satisfaction <= 2, 0.35, 0.10)
        login_missing = rng.random(n) < 0.06
        age_missing = rng.random(n) < 0.04

        df = pd.DataFrame({
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "autopay": autopay,
            "num_products": n_products,
            "support_tickets_90d": tickets,
            "satisfaction_score": pd.array(np.where(sat_missing, np.nan, satisfaction), dtype="Int64"),
            "days_since_last_login": np.where(login_missing, np.nan, last_login),
            "age": pd.array(np.where(age_missing, np.nan, age), dtype="Int64"),
            "region": region,
            "churned": churned,
        })
        return df


    if __name__ == "__main__":
        out = Path("churn.csv")
        make_churn_data().to_csv(out, index=False)
        print(f"wrote {out.resolve()}")
    ```

```python
df = make_churn_data()
print(df.shape)
print(f"churn rate: {df['churned'].mean():.3f}")
print(df.head(3).T)
```

```text
(10000, 14)
churn rate: 0.196
                                      0               1             2
customer_id                     C100000         C100001       C100002
tenure_months                        13              42            50
monthly_charges                    78.8           51.75         72.82
contract                       one-year  month-to-month      two-year
payment_method         electronic check     credit card  mailed check
internet_service                  fiber             dsl           dsl
autopay                            True            True         False
num_products                          4               2             2
support_tickets_90d                   1               0             0
satisfaction_score                    3            <NA>             5
days_since_last_login               7.5            23.2           5.5
age                                  42              63            54
region                            north            east          west
churned                               0               0             0
```

## What to do

Use only NumPy (and pandas for loading) for Parts B through E. You may use scikit-learn's `train_test_split` for the initial split, and scikit-learn is required in Part F.

### Part A: Splits and the plan

1. Split the data into **training (60%)**, **validation (20%)**, and **test (20%)** sets, stratified by the label, with a fixed seed. Write down, before you start, what each set will be used for. The test set is used exactly once, at the end.
2. Explore the training set only: the churn rate, missing-value rates, and churn rate by `contract` and by whether `satisfaction_score` is missing. What do you conclude about the missingness?

### Part B: Preprocessing from scratch

3. Write a preprocessor class with `fit(df)` and `transform(df)` methods that: imputes numeric columns with their training medians, adds a 0/1 **missing indicator** for each numeric column that has missing values in training, standardizes the numeric and indicator columns with training statistics, and one-hot encodes the categorical columns, dropping the first (alphabetical) level of each. `autopay` should be treated as a category.
4. Show that the preprocessor uses only training statistics (for example, by transforming the validation set and checking that its columns aren't exactly mean 0 and standard deviation 1).

### Part C: Logistic regression with L2, from scratch

5. Implement the objective $J(\mathbf{w}) = \frac{1}{n}\sum_i\left[\log(1 + e^{z_i}) - y_i z_i\right] + \frac{\lambda}{2}\sum_{j \geq 1} w_j^2$, with $z_i = \mathbf{w}^\top\mathbf{x}_i$ and an unpenalized intercept $w_0$, and its gradient. Derive the gradient in your write-up.
6. **Gradient check:** compare your analytical gradient with central differences at three random points. The relative error must be below $10^{-6}$.
7. Fit the model with gradient descent. Choose the step size from a bound on the curvature (justify it), and stop on a gradient-norm tolerance. Report the number of iterations.

### Part D: Choosing the regularization strength

8. Implement **stratified 5-fold cross-validation** on the training set from scratch. The preprocessor must be refit inside each fold.
9. Evaluate a logarithmic grid of $\lambda$ values (for example $10^{-5}$ to $1$) by mean validation log-loss, and report the fold-to-fold spread. Choose $\lambda$, and say whether the one-standard-error rule would change your choice.
10. Refit on the full training set with the chosen $\lambda$.

### Part E: Threshold, evaluation, and calibration

11. From Priya's economics, derive the probability threshold above which calling a customer has positive expected value.
12. On the **validation** set, compute the total profit (relative to calling nobody) for a grid of thresholds and pick the best. Compare it with your derived threshold.
13. On the **test** set, once: compute ROC-AUC (from scratch, using the rank formula), PR-AUC as average precision (from scratch), the Brier score, log-loss, the confusion matrix and precision and recall at your threshold, and the profit at your threshold versus calling nobody, calling everybody, and using a 0.5 threshold.
14. Draw the ROC curve, the precision-recall curve, and a reliability diagram (10 quantile bins, computed from scratch). Is the model calibrated well enough for Priya's third request?

### Part F: Match it with scikit-learn

15. Build the same model as a scikit-learn `Pipeline` (a `ColumnTransformer` with `SimpleImputer(strategy="median", add_indicator=True)` and `StandardScaler` for numeric columns and `OneHotEncoder(drop="first")` for categorical ones, followed by `LogisticRegression`). Work out the value of `C` that corresponds to your $\lambda$, and explain why.
16. Show that the transformed feature matrices, the coefficients, and the predicted probabilities match yours (to about $10^{-4}$ or better), and that scikit-learn's metric functions reproduce your test metrics.
17. Using your own folds, show that `cross_val_score` with the pipeline reproduces your cross-validated log-loss for each $\lambda$.

## Deliverables

- A script or notebook that runs top to bottom with one command and reproduces every number.
- A one-page summary for Priya: whom to call (the threshold, in plain words), the expected profit per month on the test customers compared with the two baselines, the model's discrimination and calibration in one sentence each, and the three features with the largest effects, with a caution about interpreting them.
- Three figures: ROC and PR curves, the reliability diagram, and profit against threshold on the validation set.

## Acceptance checklist

- [ ] The data are split into training, validation, and test sets with stratification and a fixed seed, and the test set is used once, at the end.
- [ ] All preprocessing statistics (medians, means, standard deviations, category levels) come from the training data only, and are refit inside each CV fold.
- [ ] Missing values are imputed, and missing indicators are included; the write-up says why the satisfaction indicator matters.
- [ ] The log-loss gradient is derived, implemented with a numerically stable loss, and passes a gradient check with relative error below $10^{-6}$.
- [ ] The intercept is not penalized, and the optimizer's step size is justified from a curvature bound.
- [ ] Stratified 5-fold CV is implemented from scratch, and $\lambda$ is chosen from a logarithmic grid with the fold spread reported.
- [ ] The cost-optimal threshold is derived (0.3 for Priya's numbers) and compared with the threshold chosen on validation data.
- [ ] Test-set ROC-AUC, average precision, Brier score, log-loss, confusion matrix, and profit are reported, computed from scratch.
- [ ] Profit is compared with calling nobody, calling everybody, and the default 0.5 threshold.
- [ ] A reliability diagram is drawn from scratch, and the calibration question is answered with evidence.
- [ ] The scikit-learn pipeline uses $C = 1/(\lambda n)$, and its features, coefficients, probabilities, metrics, and CV scores match the from-scratch versions.
- [ ] The summary fits on one page, leads with the recommendation, and states money in USD.

## Hints

??? tip "Hint for Part A"
    `train_test_split` twice: first hold out 20% as the test set, then split the remaining 80% into 75% training and 25% validation (which is 60% and 20% of the total). Pass `stratify=` both times. Compare the churn rate of customers with and without a satisfaction score: if they differ a lot, the missingness carries information, and a missing indicator lets the model use it.

??? tip "Hint for Part B"
    `series.to_numpy(dtype=float, na_value=np.nan)` turns a nullable integer column into floats with `NaN`. Store the training medians, means, standard deviations, the list of columns with missing values, and the sorted category levels as attributes in `fit`; `transform` must use only those.

??? tip "Hint for Part C"
    The per-example loss simplifies to $\log(1 + e^{z}) - yz$, which `np.logaddexp(0, z)` computes stably, and its derivative with respect to $z$ is $\sigma(z) - y$. The Hessian of the mean log-loss is $\frac{1}{n}\mathbf{X}^\top\mathbf{S}\mathbf{X}$ with $S_{ii} = p_i(1 - p_i) \leq \frac{1}{4}$, so $L = \frac{1}{4}\lambda_{\max}\left(\frac{1}{n}\mathbf{X}^\top\mathbf{X}\right) + \lambda$ bounds the curvature, and $\eta = 1/L$ is a safe step size.

??? tip "Hint for Part D"
    To build stratified folds, shuffle the indices of each class separately and deal them out round-robin to the $k$ folds. Fit the preprocessor on the fold's training part, transform both parts, fit, and score. Keep the folds fixed across all $\lambda$ values so the comparison is paired.

??? tip "Hint for Part E"
    Calling a customer with churn probability $p$ is worth $p \times 0.4 \times 250 - 30$ in expectation. For the rank formula, `scipy.stats.rankdata` handles ties. Average precision is $\sum_k (R_k - R_{k-1})P_k$ over the thresholds where recall increases.

??? tip "Hint for Part F"
    scikit-learn minimizes $C\sum_i \ell_i + \frac{1}{2}\lVert\mathbf{w}\rVert^2$. Divide by $Cn$ and compare with your objective. Tighten `tol` and raise `max_iter` so scikit-learn's solver converges as precisely as yours. `cross_val_score` accepts a list of `(train_idx, val_idx)` pairs as `cv`, and `scoring="neg_log_loss"`.

## Solution

A complete worked solution, with all code and outputs, is in [the Level 3 capstone solution](solutions/level-3-capstone.md). Try the whole capstone before you look, and compare against the [acceptance checklist](#acceptance-checklist) rather than line by line.
