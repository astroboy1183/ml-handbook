# 🔧 Intermediate Tier: Machine Learning

**Levels 3–5 · Goal: understand, build, and ship classical ML** · ~12–16 weeks at about an hour a day

You can analyze data. Now you'll learn how machines learn from it. You'll derive and implement the core algorithms from scratch so you know exactly what they do, master the model families that win on real-world tabular data, and then learn the engineering it takes to make a model reliable in production.

!!! info "This tier is being written"
    The Beginner tier is complete. This tier's chapters are in progress: pages marked "Coming soon" list exactly what each chapter will cover. Start with the [Beginner tier](beginner.md) if you haven't already.

## What you'll be able to do

- Explain generalization, overfitting, the bias-variance trade-off, and regularization precisely.
- Implement linear and logistic regression, decision trees, k-means, PCA, and more from scratch, and match scikit-learn's results.
- Choose the right model for a problem, validate it without fooling yourself, and pick the metric that matches the business goal.
- Tune hyperparameters efficiently, handle imbalanced data, and explain predictions with SHAP.
- Package a model behind an API, track experiments, and monitor for drift.
- Recognize and measure unfairness in a model.

## The levels

<div class="grid cards" markdown>

-   :material-numeric-3-circle:{ .lg .middle } **Level 3: ML fundamentals**

    ---

    What ML is, linear and logistic regression, generalization, regularization, metrics, and gradient descent.

    [:octicons-arrow-right-24: Start Level 3](../chapters/03-ml-fundamentals/index.md)

-   :material-numeric-4-circle:{ .lg .middle } **Level 4: ML algorithms in depth**

    ---

    kNN, naive Bayes, trees, ensembles and boosting, SVMs, clustering, dimensionality reduction, time series, anomaly detection, and recommenders.

    [:octicons-arrow-right-24: Start Level 4](../chapters/04-ml-algorithms/index.md)

-   :material-numeric-5-circle:{ .lg .middle } **Level 5: Applied ML and MLOps**

    ---

    Pipelines, hyperparameter tuning, imbalanced data, interpretability, ML in production, and responsible ML.

    [:octicons-arrow-right-24: Start Level 5](../chapters/05-applied-ml/index.md)

</div>

## Tier checkpoint

You're ready for the [Expert tier](expert.md) when you can do all of these without notes:

- [ ] Derive the gradient of the logistic regression loss and implement training from scratch.
- [ ] Explain the bias-variance trade-off with a learning curve, and say what you'd do about each failure mode.
- [ ] Explain why gradient-boosted trees usually beat neural networks on tabular data.
- [ ] Choose between accuracy, F1, ROC-AUC, and PR-AUC for a given problem, and justify a decision threshold.
- [ ] Build a leakage-proof scikit-learn pipeline with preprocessing, tuning, and nested cross-validation.
- [ ] Serve a model behind an HTTP API and describe how you'd detect when it starts going wrong.

See the [full roadmap](../roadmap.md) for every concept in this tier.
