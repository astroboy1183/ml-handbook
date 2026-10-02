# Level 5: Applied ML and MLOps

> **Level 5 · Overview** · ⏱️ ~4–5 weeks at about an hour a day · Prerequisites: [Level 3](../03-ml-fundamentals/index.md) and [Level 4](../04-ml-algorithms/index.md), plus [Feature engineering](../02-data-science-workflow/04-feature-engineering.md)

Level 5 turns models into systems you can trust. You learn to build leakage-proof pipelines, tune them honestly, handle rare classes, explain predictions, ship a model as a monitored service, and check it for fairness, privacy, and security problems.

## What this level covers

Levels 3 and 4 taught you how learning algorithms work. This level is about everything around them that decides whether a model helps or harms once it leaves your notebook.

You start with **pipelines**: chaining preprocessing and a model into one object, so that cross-validation, tuning, and deployment all use exactly the same steps, and leakage has nowhere to hide. **Hyperparameter tuning** covers grid search, random search, Bayesian optimization with Optuna, pruning, and successive halving, and then the uncomfortable part: why the best score from a search is optimistic, and how nested cross-validation fixes it. **Imbalanced data** replaces accuracy with metrics and decision thresholds based on real costs, and shows what class weights and SMOTE really do to a model (and to its probabilities).

The second half moves from building models to answering for them. **Interpretability** explains models globally and one prediction at a time, with permutation importance, partial dependence, SHAP, LIME, and counterfactuals, and shows where each misleads. **ML in production** covers the lifecycle: saving and versioning models, batch and online serving, a FastAPI service with validation and tests, Docker, MLflow experiment tracking, drift detection, monitoring, and retraining. **Responsible ML** measures fairness with several incompatible definitions, protects privacy with k-anonymity and differential privacy, documents models with model cards, and introduces adversarial and poisoning attacks.

## What you'll be able to do

- Build any tabular model as a single scikit-learn pipeline, with custom transformers, and prove it doesn't leak.
- Tune efficiently with random search and Optuna, and report performance honestly with nested cross-validation.
- Choose metrics and thresholds for imbalanced problems from business costs, and keep probabilities calibrated.
- Explain a model's behavior and individual predictions, and recognize when an explanation is misleading.
- Ship a model as a tested, validated HTTP service in a container, track experiments, and monitor for data and concept drift.
- Audit a model for fairness across groups, reason about the trade-offs between fairness criteria, and apply basic privacy and security defenses.

## Chapters

| # | Chapter | What it covers | Time |
|---|---|---|---|
| 1 | [ML pipelines](01-ml-pipelines.md) | Leakage demo, `Pipeline` and `ColumnTransformer`, custom transformers, `set_output`, reproducibility, joblib, project structure | ~60 min read + 3–4 h practice |
| 2 | [Hyperparameter tuning](02-hyperparameter-tuning.md) | Grid vs random search, TPE and Optuna, pruning, successive halving, the winner's curse, nested CV | ~60 min read + 3–4 h practice |
| 3 | [Imbalanced data](03-imbalanced-data.md) | Metrics under imbalance, cost-optimal thresholds, class weights, SMOTE in a pipeline, cost curves, recalibration | ~55 min read + 3–4 h practice |
| 4 | [Interpretability](04-interpretability.md) | Coefficients, impurity vs permutation importance, PDP and ICE, Shapley values and SHAP, LIME, counterfactuals, pitfalls | ~65 min read + 4–5 h practice |
| 5 | [ML in production](05-ml-in-production.md) | Lifecycle, versioning, batch vs online, FastAPI and pydantic, Docker, MLflow, PSI and KS drift, monitoring, retraining, testing | ~70 min read + 5–6 h practice |
| 6 | [Responsible ML](06-responsible-ml.md) | Sources of bias, fairness metrics and impossibility, mitigation, k-anonymity, differential privacy, model cards, adversarial attacks | ~70 min read + 4–5 h practice |

## How to study this level

- **Rerun every leak.** Each chapter shows an evaluation going wrong (leaky encoding, tuning on the test folds, resampling before CV). Run the wrong version and the right one yourself, and watch the gap. Seeing it once makes you suspicious for life.
- **Put numbers on decisions.** Whenever a chapter picks a threshold or a metric, write down the cost of each kind of error for a problem you care about, and redo the calculation with your numbers.
- **Build the project skeleton early.** From chapter 1 on, keep a small project folder (`train.py`, a `build_pipeline` function, a `tests/` folder) and grow it with each chapter. By chapter 5, most of the capstone will already exist.
- **Explain one prediction a day.** Pick a row, compute its SHAP values, and write one sentence a customer would understand. Then check whether the explanation survives correlated features.
- **Read the outputs, not just the code.** Several chapters contain surprises (a default model matching the tuned one, a logistic regression beating gradient boosting, a counterfactual exploiting a model artifact, a fairness fix creating a new unfairness). The surprises are the lessons.
- **Keep the references open:** the [scikit-learn cheat sheet](../../cheatsheets/scikit-learn.md) covers the estimator API, pipelines, model selection, and metrics on one page.

!!! tip "This is what employers test"
    Interviews and real projects rarely ask you to derive an algorithm. They ask how you'd evaluate a model without fooling yourself, what you'd do about a rare class, how you'd explain a prediction to a regulator, and how you'd know a deployed model had stopped working. This level is the answer to those questions.

## Capstone

The level, and the Intermediate tier, ends with **[shipping a production-ready model service](../../exercises/level-5-capstone.md)**: a churn model built as a pipeline, tuned with Optuna, tracked with MLflow, served with FastAPI behind input validation and tests, packaged with Docker, and watched by a drift monitor with a retraining policy. Finish it without notes, and you're ready for the Expert tier, starting with [Level 6: Neural networks from scratch](../06-neural-networks/index.md).
