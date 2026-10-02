# Level 3: Machine Learning Fundamentals

> **Level 3 · Overview** · ⏱️ ~4–5 weeks at about an hour a day · Prerequisites: the Beginner tier, [Level 0](../00-python-for-data/index.md) to [Level 2](../02-data-science-workflow/index.md), especially [Calculus and gradients](../01-math-foundations/03-calculus-and-gradients.md), [Statistics](../01-math-foundations/05-statistics.md), and [Information theory and optimization](../01-math-foundations/06-information-theory-and-optimization.md)

Level 3 opens the Intermediate tier. It's where you stop analyzing data and start building models that learn from it. You'll derive, implement from scratch, and then use in scikit-learn the two models that underlie most of machine learning, linear and logistic regression, and you'll learn the ideas that decide whether *any* model works on new data: generalization, regularization, evaluation, and optimization.

## What this level covers

Every model in the rest of this handbook, from decision trees to transformers, is built from the same few ideas. This level teaches them on models simple enough to understand completely.

You start with **what learning means**: a hypothesis space, a loss function, and empirical risk minimization, plus why doing well on training data proves nothing, and the scikit-learn API you'll use from here on. Then come the two foundational models. **Linear regression** gets the full treatment: least squares derived with calculus and with geometry, the normal equation, gradient descent, assumptions and diagnostics, polynomial features, and how to read coefficients without fooling yourself. **Logistic regression** turns scores into probabilities: you derive the log-loss from maximum likelihood and its gradient from the chain rule, and extend it to many classes with the softmax.

The middle of the level is about generalization. **The bias-variance trade-off** explains why test error is U-shaped as models grow, and you'll derive the decomposition and then watch it happen in a simulation. You'll learn to split data properly, cross-validate (including stratified, grouped, and time-series splits), read learning curves, and meet double descent. **Regularization** is the main tool for controlling overfitting: ridge's closed form, the lasso's sparsity (algebraically and geometrically), the elastic net, the Bayesian view, and early stopping.

The level ends with two chapters you'll use constantly. **Evaluation metrics** covers every common regression and classification metric, ROC and precision-recall curves, calibration, and how to choose a decision threshold from business costs. **Gradient descent in depth** explains batch, stochastic, and mini-batch training, why the learning rate has a hard limit, how conditioning controls convergence speed, and how to debug a training loop.

## What you'll be able to do

- Frame a problem as supervised, unsupervised, self-supervised, or reinforcement learning, and name its features, label, loss, and metric.
- Derive the least squares solution, the MSE gradient, the log-loss from the Bernoulli likelihood, its gradient $\frac{1}{n}\mathbf{X}^\top(\sigma(\mathbf{X}\mathbf{w}) - \mathbf{y})$, and the softmax cross-entropy gradient.
- Implement linear regression, logistic regression, softmax regression, ridge, the lasso (coordinate descent), and batch, stochastic, and mini-batch gradient descent in NumPy, and verify them against scikit-learn.
- Use the scikit-learn estimator API: `fit`, `predict`, `transform`, pipelines, `train_test_split`, and `cross_val_score`.
- Derive the bias-variance decomposition, diagnose underfitting and overfitting from validation and learning curves, and choose the right cross-validation scheme.
- Regularize linear and logistic models, choose the strength by cross-validation, and explain why the lasso produces zeros.
- Pick metrics that match the problem, read ROC and PR curves, check and fix calibration, and choose a threshold that minimizes expected cost.
- Diagnose a slow or diverging training run using the learning rate, conditioning, and a gradient check.

## Chapters

| # | Chapter | What it covers | Time |
|---|---|---|---|
| 1 | [What is machine learning?](01-what-is-machine-learning.md) | Learning from data, the kinds of learning, features, parameters and hyperparameters, losses, ERM, hypothesis spaces, generalization, no free lunch, the scikit-learn API | ~55 min read + 3 h practice |
| 2 | [Linear regression](02-linear-regression.md) | The model, least squares, the normal equation, gradient descent, assumptions and diagnostics, polynomial features, interpreting coefficients | ~65 min read + 4–5 h practice |
| 3 | [Logistic regression and classification](03-logistic-regression.md) | Why not linear regression, the sigmoid, log-loss from MLE and its gradient, decision boundaries, softmax, the probabilistic view | ~65 min read + 4–5 h practice |
| 4 | [Generalization and the bias-variance trade-off](04-generalization-bias-variance.md) | Under- and overfitting, train/validation/test, the decomposition (derived and simulated), capacity, learning curves, k-fold, stratified and time-series CV, double descent | ~65 min read + 4 h practice |
| 5 | [Regularization](05-regularization.md) | Ridge's closed form and shrinkage, the lasso and sparsity, elastic net, the geometric and Bayesian views, early stopping, choosing the strength | ~65 min read + 4 h practice |
| 6 | [Evaluation metrics](06-evaluation-metrics.md) | MAE, MSE, RMSE, R², the confusion matrix, precision, recall, F1, ROC and PR curves, calibration, thresholds, business costs | ~70 min read + 4 h practice |
| 7 | [Gradient descent in depth](07-gradient-descent-in-depth.md) | Batch, SGD, mini-batch, learning rates, convergence, conditioning and feature scaling, loss landscapes, momentum, debugging | ~65 min read + 4 h practice |

Keep the [evaluation metrics cheat sheet](../../cheatsheets/metrics.md) open while you work through chapters 4 to 6 and the capstone.

## How to study this level

- **Derive before you code.** Each chapter derives its key results: the normal equation, the log-loss gradient, the bias-variance decomposition, ridge's closed form, soft-thresholding, the stable learning rate. Close the page and redo each derivation on paper. If you can't, you don't own it yet.
- **Then implement from scratch, then check against the library.** Every model in this level is implemented in NumPy and compared with scikit-learn. Do the same yourself, and don't stop at "close enough": find out why any difference exists (a different penalty scaling, an unpenalized intercept, a convergence tolerance). Those discrepancies are where most of the learning is.
- **Run the experiments, then change them.** Rerun the bias-variance simulation with more data, the double-descent demo with a ridge penalty, the cost-threshold analysis with different costs. Predict the result before you run it.
- **Keep a held-out test set sacred.** From this level on, every project gets a test set that you look at once. Practicing that discipline on small problems makes it automatic on big ones.
- **Check gradients numerically.** Every time you write a gradient, check it with central differences. It takes two minutes and prevents the most expensive bugs in ML.

!!! tip "Why spend a whole level on two linear models?"
    Linear and logistic regression are still among the most widely deployed models in industry, because they're fast, stable, and interpretable. More importantly, every concept in this level (losses, likelihoods, gradients, regularization, validation, calibration, conditioning) carries over directly to trees, ensembles, and neural networks. A neural network classifier is, at its last layer, a softmax regression trained by mini-batch gradient descent.

## Capstone

The level ends with **[building a churn classifier from scratch in NumPy, then matching it with scikit-learn](../../exercises/level-3-capstone.md)**. You'll prepare a messy customer data set, implement L2-regularized logistic regression with a gradient check, tune it with cross-validation, choose a decision threshold from the business cost of retention offers, evaluate it with ROC-AUC, PR-AUC, and calibration, and show that scikit-learn agrees with you. Finish it without notes, and you're ready for [Level 4: ML algorithms in depth](../04-ml-algorithms/index.md).
