# Level 4 capstone: a model bake-off

> **Level 4 · Capstone** · ⏱️ 8–12 h · Prerequisites: all of [Level 4](../chapters/04-ml-algorithms/index.md), plus [cross-validation](../chapters/03-ml-fundamentals/04-generalization-bias-variance.md) and [evaluation metrics](../chapters/03-ml-fundamentals/06-evaluation-metrics.md) from Level 3

You've learned a dozen algorithms. This capstone asks the question every practitioner faces on a new tabular problem: **which one should we actually use?** You'll run a fair, rigorous comparison of a baseline, regularized linear models, k-nearest neighbors, an SVM, a random forest, and gradient boosting on the same data, under the same validation, with simple tuning. Then you'll weigh accuracy against uncertainty, training cost, and interpretability, and write a recommendation a decision-maker can act on. Finish it without notes before moving on to [Level 5](../chapters/05-applied-ml/index.md).

## The scenario

You're the data scientist at LoanCo, an online lender of personal loans between USD 1,000 and USD 60,000. Today, applications are scored with a hand-built points system that the credit team suspects is leaving money on the table. The Head of Credit Risk asks you:

> "We want a model that ranks applicants by their risk of default. I've heard everything from 'logistic regression is all you need' to 'just use XGBoost'. Compare the reasonable options properly, and tell me which one we should build, how sure you are that it's better, and what we give up with it. Our regulator expects us to explain the main drivers of every decision, so keep that in mind."

The decision isn't only about the highest score. The model will be retrained monthly, must score applications in real time, and must come with explanations of its main risk drivers.

## The data

The dataset is synthetic but realistic: 5,000 past loan applications with a binary outcome `default` (1 if the borrower defaulted within the loan's first two years). Expand the generator, save it as `make_loan_data.py`, and run it (or call `make_loans()` directly).

| Column | Meaning |
|---|---|
| `income_kusd` | Annual income, thousands of USD |
| `credit_score` | Bureau credit score (450 to 850); about 3% missing |
| `debt_to_income` | Monthly debt payments divided by monthly income |
| `utilization` | Share of available revolving credit in use |
| `late_payments_2y` | Number of late payments in the last two years |
| `employment_years` | Years with the current employer; about 8% missing |
| `loan_amount_kusd` | Requested amount, thousands of USD |
| `term_months` | 36 or 60 |
| `open_credit_lines` | Number of open credit accounts |
| `home_ownership` | `rent`, `mortgage`, or `own` |
| `purpose` | `debt_consolidation`, `home_improvement`, `car`, `small_business`, or `other` |
| `noise_1` to `noise_3` | Columns with no relationship to default (you're not told this in real life) |
| `default` | Target: 1 = defaulted |

??? example "The capstone data generator: make_loan_data.py (expand and run this first)"

    ```python
    """Generate the Level 4 capstone dataset: synthetic personal-loan applications and defaults.

    Usage:
        python make_loan_data.py                 # writes loans.csv in the current directory
        from make_loan_data import make_loans
        df = make_loans()                        # pandas DataFrame

    The data is synthetic and seeded. Default risk has nonlinear effects (a credit-score
    cliff, a U-shaped utilization effect, saturating late payments), interactions (debt
    ratio matters more on 60-month loans, small-business loans are riskier for low
    incomes), two categorical features, missing values, and a few pure-noise columns.
    """
    from pathlib import Path

    import numpy as np
    import pandas as pd


    def make_loans(n=5000, seed=42):
        rng = np.random.default_rng(seed)
        income = rng.lognormal(np.log(58), 0.45, n).round(1)                       # thousands of USD
        credit = np.clip(rng.normal(680, 60, n), 450, 850).round()
        dti = np.clip(rng.beta(2.2, 5, n) + 0.08 * (income < 35), 0, 0.95).round(3)
        utilization = rng.beta(2, 3, n).round(3)
        late = rng.poisson(0.25 + 1.2 * (credit < 620), n)
        emp_years = rng.exponential(6, n).round(1)
        loan_amount = np.clip(rng.lognormal(np.log(12), 0.6, n), 1, 60).round(1)   # thousands of USD
        term = rng.choice([36, 60], n, p=[0.65, 0.35])
        lines = rng.poisson(8, n)
        home = rng.choice(["rent", "mortgage", "own"], n, p=[0.45, 0.4, 0.15])
        purpose = rng.choice(["debt_consolidation", "home_improvement", "car", "small_business", "other"],
                             n, p=[0.45, 0.15, 0.15, 0.1, 0.15])
        noise = rng.normal(size=(n, 3))

        logit = (
            -4.0
            + 1.6 / (1 + np.exp((credit - 615) / 12))                 # cliff below about 615
            - 0.006 * (credit - 680)
            + 16.0 * (utilization - 0.45) ** 2                        # U-shape around 45%
            + 1.1 * np.log1p(late)                                    # saturating late payments
            + 2.5 * dti * (1 + 1.2 * (term == 60))                    # interaction: DTI x long term
            - 0.6 * np.log(income / 58)
            + 0.9 * ((purpose == "small_business") & (income < 50))  # interaction: purpose x income
            + 0.25 * (home == "rent")
            - 0.03 * np.minimum(emp_years, 10)
            + 0.15 * np.log(loan_amount / income)
        )
        default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

        df = pd.DataFrame({
            "income_kusd": income, "credit_score": credit, "debt_to_income": dti,
            "utilization": utilization, "late_payments_2y": late, "employment_years": emp_years,
            "loan_amount_kusd": loan_amount, "term_months": term, "open_credit_lines": lines,
            "home_ownership": home, "purpose": purpose,
            "noise_1": noise[:, 0].round(3), "noise_2": noise[:, 1].round(3), "noise_3": noise[:, 2].round(3),
            "default": default,
        })
        # Missing values: employment years more often missing for renters; some credit scores missing
        miss_emp = rng.random(n) < np.where(home == "rent", 0.12, 0.05)
        df.loc[miss_emp, "employment_years"] = np.nan
        df.loc[rng.random(n) < 0.03, "credit_score"] = np.nan
        return df


    if __name__ == "__main__":
        out = Path("loans.csv")
        make_loans().to_csv(out, index=False)
        print(f"wrote {out.resolve()}")
    ```

```python
df = make_loans()
print(df.shape, "| default rate:", round(df["default"].mean(), 3))
```

```text
(5000, 15) | default rate: 0.207
```

The generator's comments describe the kinds of structure it contains. Try to discover them from the data and the models rather than reading the formula: in a real project, nobody hands you the data-generating process.

## What to do

### Part A: Validation design (decide before you model)

1. Hold out a stratified **test set** of 20% that you touch only once, at the very end.
2. On the remaining **development set**, set up **nested cross-validation**: an outer loop of repeated stratified k-fold (for example, 5 folds × 2 repeats) for comparing models, and an inner loop (for example, 3 folds) for tuning each model's hyperparameters. Every model must use the **same outer folds**.
3. Choose a **primary metric** and justify it in two sentences. Add at least one secondary metric, including one that checks the quality of predicted probabilities.

### Part B: Candidate models

Build each candidate as a scikit-learn `Pipeline` so that imputation, encoding, and scaling are learned inside each training fold:

1. A **baseline** that ignores the features (`DummyClassifier`).
2. At least two **regularized linear models**: L2 logistic regression and an L1 (or elastic net) version.
3. **k-nearest neighbors**.
4. An **SVM** with an RBF kernel.
5. A **random forest**.
6. **Gradient boosting**: `HistGradientBoostingClassifier`, or XGBoost or LightGBM.

Give each model a small, sensible tuning grid (2 to 6 settings) chosen from what you learned in the chapters: what are the one or two hyperparameters that matter most for each?

### Part C: Comparison

1. Run the nested cross-validation for every model. Report the **mean and standard deviation** of each metric across outer folds, and the training-plus-tuning time per fold.
2. Record which hyperparameters the inner searches chose on each outer fold. Are the choices stable?
3. Plot the per-fold scores of all models in one figure.

### Part D: Are the differences real?

1. For the best model by mean score, compare it with every other model using the per-fold differences. Use the **corrected resampled t-test** (Nadeau and Bengio, 2003), and also show what a naive paired t-test would have said. Explain the difference in a few sentences.
2. Decide which models are statistically indistinguishable from the best, and say so plainly.

### Part E: Final check, cost, and interpretability

1. Refit the top three or four candidates on the whole development set (with tuning), and evaluate each **once** on the test set, with a bootstrap 95% confidence interval for the primary metric.
2. Report a business-facing metric too, such as the share of all defaults that fall among the 20% of applicants each model ranks riskiest.
3. Measure the training time and the prediction time of each final model.
4. Explain each finalist: coefficients for linear models; permutation importance and partial dependence for tree ensembles. Check that the noise columns aren't driving predictions.

### Part F: The recommendation

Write a recommendation of at most one page (about 500 words) for the Head of Credit Risk: the answer first, then the evidence (with uncertainty), the trade-offs (accuracy, interpretability, cost, maintenance), the risks, and next steps.

### Stretch goals (optional)

- Add XGBoost and LightGBM alongside HistGradientBoosting and compare them under the same folds.
- Calibrate the SVM with `CalibratedClassifierCV` so it can be scored on probability quality too.
- Use a cost-based metric: suppose a default costs LoanCo USD 8,000 on average and a rejected good applicant costs USD 1,200 in lost margin, and choose a threshold for each finalist that minimizes expected cost.
- Engineer features that let a linear model capture the structure you found (for example, spline features or a hinge on credit score), and see how close it gets to the best model.

## Deliverables

1. **Code**: a script or a notebook that runs cleanly top to bottom and reproduces every number from the seed.
2. **Results table**: per-model means and standard deviations, with timings.
3. **Figures**: at least two (per-fold scores; an interpretability plot), saved as PNG files.
4. **Recommendation**: `recommendation.md`, at most one page.

## Acceptance checklist

- [ ] The test set is created first and used exactly once, after all model choices are made.
- [ ] All preprocessing lives inside pipelines; no statistic is computed on data outside the training fold.
- [ ] Every model is tuned in an inner loop and evaluated on the same outer folds.
- [ ] Results are reported as mean ± standard deviation across outer folds, not as a single split.
- [ ] Model comparisons use paired fold differences with a corrected test, and the report explains why the naive test is overconfident.
- [ ] Statistically tied models are described as tied, not ranked by tiny differences.
- [ ] The test-set result has a confidence interval and agrees (within uncertainty) with the cross-validation estimate.
- [ ] Training time, prediction time, and interpretability are compared, not just accuracy.
- [ ] At least one interpretability method is applied to each finalist, and the noise columns are checked.
- [ ] The recommendation states the decision in its first two sentences, quantifies the uncertainty, and names the trade-offs.

## Hints

??? tip "Hint for Part A"
    `cross_validate(GridSearchCV(pipe, grid, cv=inner), X_dev, y_dev, cv=outer, ...)` *is* nested cross-validation: each outer training fold gets its own grid search. Create the `outer` splitter once, with a fixed `random_state`, and pass the same object to every model so they all see the same folds.

??? tip "Hint for Part B"
    Use a `ColumnTransformer`: one-hot encode the categories, and impute (with `add_indicator=True`) and standardize the numbers for linear models, kNN, and the SVM. Trees don't need scaling. `HistGradientBoostingClassifier` handles missing values natively and accepts ordinal-encoded categories through `categorical_features`. Good first grids: `C` for logistic regression and the SVM (plus `gamma`), `n_neighbors` for kNN, `min_samples_leaf` for the forest, `learning_rate` and `max_leaf_nodes` (with early stopping) for boosting.

??? tip "Hint for Part D"
    For per-fold differences $d_j$ over $J$ folds, the corrected statistic is $t = \bar{d} / \sqrt{(1/J + n_{\text{test}}/n_{\text{train}})\,\hat{\sigma}^2_d}$ with $J - 1$ degrees of freedom, where $n_{\text{test}}$ and $n_{\text{train}}$ are the sizes of one outer test and training fold. `scipy.stats.t.sf` gives the tail probability.

??? tip "Hint for Part E"
    For the bootstrap, resample test rows with replacement 1,000 times and recompute the AUC each time; take the 2.5th and 97.5th percentiles. Use the *same* resamples for two models to get a paired interval for their difference. For partial dependence, `PartialDependenceDisplay.from_estimator` works directly on a fitted pipeline with a DataFrame input; convert integer columns to float first.

??? tip "Hint for Part F"
    Draft the first two sentences before anything else. If two models are statistically tied, the tie-breakers are the things the Head of Credit Risk cares about: explainability to the regulator, operational simplicity, and how the model will be monitored and retrained.

## Solution

When you're done, or truly stuck, compare with the [worked solution](solutions/level-4-capstone.md). It includes the full code, every output, and a sample recommendation.

## Next

Congratulations on finishing Level 4. [Level 5: Applied ML and MLOps](../chapters/05-applied-ml/index.md) turns a model like the one you just chose into a reliable service: leakage-proof pipelines, serious hyperparameter tuning, imbalanced data, interpretability, deployment, and monitoring.
