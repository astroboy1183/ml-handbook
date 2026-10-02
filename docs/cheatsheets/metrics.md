# Evaluation Metrics Cheat Sheet

A dense reference for every common regression and classification metric: its formula, when to use it, its pitfalls, and the scikit-learn function. It ends with threshold selection and calibration.

Related chapters: [Evaluation metrics](../chapters/03-ml-fundamentals/06-evaluation-metrics.md) · [Generalization and the bias-variance trade-off](../chapters/03-ml-fundamentals/04-generalization-bias-variance.md) · [Logistic regression and classification](../chapters/03-ml-fundamentals/03-logistic-regression.md) · [Imbalanced data](../chapters/05-applied-ml/03-imbalanced-data.md)

Notation: $y_i$ is the true value, $\hat{y}_i$ the prediction, $\hat{p}_i$ a predicted probability of the positive class, $n$ the number of evaluation examples, $\bar{y}$ the mean of $y$, and $\pi$ the share of positives (prevalence). All functions below live in `sklearn.metrics` unless noted.

```python
import numpy as np
from sklearn import metrics
```

## Regression metrics

| Metric | Formula | Use when | Pitfalls | scikit-learn |
|---|---|---|---|---|
| MAE | $\frac{1}{n}\sum\lvert y_i - \hat{y}_i\rvert$ | errors cost in proportion to size; outliers shouldn't dominate | optimal prediction is the median, not the mean | `mean_absolute_error` |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | large errors are disproportionately bad; it's the training loss | squared units; dominated by a few big errors | `mean_squared_error` |
| RMSE | $\sqrt{\text{MSE}}$ | as MSE, but reported in the target's units | always $\geq$ MAE; a big gap means heavy-tailed errors | `root_mean_squared_error` |
| $R^2$ | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | comparing with the predict-the-mean baseline | depends on the spread of $y$ in the evaluation set; negative means worse than the mean; not comparable across data sets | `r2_score` |
| Adjusted $R^2$ | $1 - (1 - R^2)\frac{n - 1}{n - d - 1}$ | in-sample comparison of linear models with different numbers of features $d$ | a weak substitute for a held-out score | (compute by hand) |
| MAPE | $\frac{1}{n}\sum\left\lvert\frac{y_i - \hat{y}_i}{y_i}\right\rvert$ | stakeholders think in percentages | explodes near $y = 0$; penalizes over-prediction more than under-prediction | `mean_absolute_percentage_error` (returns a fraction, not %) |
| Median AE | $\operatorname{median}\lvert y_i - \hat{y}_i\rvert$ | a typical error, robust to outliers | ignores the tail entirely | `median_absolute_error` |
| RMSLE | $\sqrt{\frac{1}{n}\sum(\log(1 + y_i) - \log(1 + \hat{y}_i))^2}$ | positive, skewed targets where relative error matters | targets must be $> -1$; under-prediction costs more than over-prediction | `root_mean_squared_log_error` |
| Pinball (quantile $\tau$) | $\frac{1}{n}\sum\max(\tau e_i, (\tau - 1)e_i)$, $e_i = y_i - \hat{y}_i$ | evaluating quantile forecasts (for example the 90th percentile of demand) | only meaningful for the quantile the model targets | `mean_pinball_loss(alpha=τ)` |
| Max error | $\max_i\lvert y_i - \hat{y}_i\rvert$ | worst-case guarantees | one outlier decides it | `max_error` |

**Rules of thumb.** Match the metric to the cost of errors, and train with a consistent loss (MSE for means, MAE for medians, pinball for quantiles). Report MAE or RMSE in real units, and $R^2$ as a check against the baseline. Compare models on the same evaluation set only.

## The confusion matrix

| | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP | FN (type II error, a miss) |
| **Actually negative** | FP (type I error, a false alarm) | TN |

`confusion_matrix(y_true, y_pred)` returns `[[TN, FP], [FN, TP]]` for labels 0 and 1 (true labels on rows, sorted). Unpack it with `tn, fp, fn, tp = confusion_matrix(y, y_hat).ravel()`. A confusion matrix describes **one threshold**; every metric in the next table changes when the threshold does.

## Classification metrics at a threshold

| Metric | Formula | Use when | Pitfalls | scikit-learn |
|---|---|---|---|---|
| Accuracy | $\frac{\text{TP} + \text{TN}}{n}$ | balanced classes, equal error costs | useless under imbalance: always compare with the majority baseline $\max(\pi, 1 - \pi)$ | `accuracy_score` |
| Balanced accuracy | $\frac{1}{2}(\text{TPR} + \text{TNR})$ | imbalanced classes, both classes matter | still ignores costs | `balanced_accuracy_score` |
| Precision (PPV) | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | false alarms are costly (blocking payments, manual review) | depends on prevalence; undefined if nothing is flagged | `precision_score` |
| Recall, sensitivity, TPR | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | misses are costly (disease screening, fraud) | trivially 1 if you flag everything | `recall_score` |
| Specificity, TNR | $\frac{\text{TN}}{\text{TN} + \text{FP}}$ | the cost of bothering negatives matters | looks great under heavy imbalance even with many FPs | `recall_score(pos_label=0)` |
| False positive rate | $\frac{\text{FP}}{\text{FP} + \text{TN}} = 1 - \text{TNR}$ | the x-axis of ROC; alarm rate among negatives | a "small" FPR times millions of negatives is many alarms | (from the confusion matrix) |
| F1 | $\frac{2PR}{P + R} = \frac{2\text{TP}}{2\text{TP} + \text{FP} + \text{FN}}$ | one number balancing precision and recall; TN uninteresting | no business meaning; single threshold; ignores TN | `f1_score` |
| F-beta | $\frac{(1 + \beta^2)PR}{\beta^2 P + R}$ | recall is $\beta$ times as important as precision ($F_2$ for screening, $F_{0.5}$ for precision) | choosing $\beta$ is a hidden cost assumption | `fbeta_score(beta=β)` |
| Matthews correlation (MCC) | $\frac{\text{TP}\cdot\text{TN} - \text{FP}\cdot\text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}}$ | a single balanced summary using all four cells; ranges from −1 to 1 | hard to explain to stakeholders | `matthews_corrcoef` |
| Precision at $k$ | share of positives among the $k$ highest scores | fixed capacity ("we can review 200 a day") | depends on $k$ and on how many positives exist | (sort scores and compute) |

`classification_report(y, y_hat, digits=3)` prints precision, recall, F1, and support per class plus the averages.

**Precision depends on prevalence; recall doesn't.** By Bayes' theorem:

$$
\text{precision} = \frac{\text{TPR}\cdot\pi}{\text{TPR}\cdot\pi + \text{FPR}\cdot(1 - \pi)}.
$$

A test with TPR 0.9 and FPR 0.05 has precision 0.65 at $\pi = 10\%$ but 0.15 at $\pi = 1\%$. Evaluate on data with the production class balance.

### Multiclass averaging

| `average=` | How | Use when |
|---|---|---|
| `"macro"` | unweighted mean of per-class scores | every class matters equally, including rare ones |
| `"weighted"` | mean weighted by class support | you want the typical example's performance; hides rare-class failures |
| `"micro"` | pool TP, FP, FN over classes, then compute | overall performance; for single-label problems micro P = micro R = micro F1 = accuracy |
| `None` | one score per class | always look at this table for rare classes |

## Ranking and probability metrics (no threshold)

| Metric | Formula or definition | Use when | Pitfalls | scikit-learn |
|---|---|---|---|---|
| ROC AUC | area under TPR vs FPR; equals $P(s(\mathbf{x}^+) > s(\mathbf{x}^-))$ | comparing how well models rank; classes not extremely imbalanced | ignores calibration and prevalence; dominated by thresholds you'd never use; 0.5 is random | `roc_auc_score`, `roc_curve` |
| PR AUC / average precision | $\text{AP} = \sum_k (R_k - R_{k-1})P_k$ | rare positives; false alarms matter | baseline is the prevalence $\pi$, not 0.5; don't compare across data sets with different $\pi$ | `average_precision_score`, `precision_recall_curve` |
| Log-loss (cross-entropy) | $-\frac{1}{n}\sum[y_i\log\hat{p}_i + (1 - y_i)\log(1 - \hat{p}_i)]$ | judging probabilities; it's the training loss of logistic regression | unbounded: one confident mistake dominates; clip probabilities | `log_loss` |
| Brier score | $\frac{1}{n}\sum(\hat{p}_i - y_i)^2$ | judging probabilities on an interpretable 0–1 scale | mixes calibration and discrimination; small values look good under imbalance (compare with $\pi(1 - \pi)$, the score of predicting $\pi$) | `brier_score_loss` |
| Multiclass ROC AUC | one-vs-rest or one-vs-one averages | multiclass ranking quality | specify `multi_class="ovr"` or `"ovo"` and pass probabilities for all classes | `roc_auc_score(multi_class=...)` |

**ROC vs PR.** ROC answers "how well does the model separate the classes?" PR answers "when the model flags something, how often is it right, and how much does it catch?" With 1% positives, prefer PR; when in doubt, report both.

Computing AUC from ranks (the Mann–Whitney form):

```python
from scipy.stats import rankdata

def auc_from_ranks(y, s):
    r = rankdata(s)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

y = np.array([1, 1, 1, 0, 0, 0, 0]); s = np.array([0.9, 0.6, 0.35, 0.7, 0.4, 0.3, 0.1])
print(auc_from_ranks(y, s), metrics.roc_auc_score(y, s))
```

```text
0.75 0.75
```

## Calibration

A model is **calibrated** if, among cases given probability $p$, a fraction $p$ are positive: $P(y = 1 \mid \hat{p} = p) = p$.

| Tool | What it does | scikit-learn |
|---|---|---|
| Reliability diagram | bin predictions; plot mean predicted vs observed rate; the diagonal is perfect | `sklearn.calibration.calibration_curve(y, p, n_bins=10, strategy="quantile")`, `CalibrationDisplay` |
| Expected calibration error | $\sum_b \frac{n_b}{n}\lvert\bar{p}_b - \bar{y}_b\rvert$ over bins $b$ | (compute from the bins) |
| Brier score, log-loss | proper scoring rules: minimized by the true probabilities | `brier_score_loss`, `log_loss` |
| Platt scaling | fit $\sigma(a s + b)$ on held-out scores $s$; 2 parameters, needs little data | `CalibratedClassifierCV(method="sigmoid")` |
| Isotonic regression | fit a non-decreasing step function; flexible, needs more data (roughly a thousand or more examples) | `CalibratedClassifierCV(method="isotonic")` |

**Calibration pitfalls.**

- Calibrate on data the model wasn't trained on. `CalibratedClassifierCV(cv=5)` handles this with internal cross-validation.
- Recalibration is monotone, so it doesn't change ROC AUC; AUC can't detect miscalibration.
- Class weights, resampling (SMOTE, undersampling), and strong regularization all distort probabilities. If you use them and need probabilities, recalibrate on data with the true class balance.
- A shift in prevalence between training and production breaks calibration even for a good model. Monitor the mean predicted probability against the observed rate.

## Threshold selection

A model outputs scores; a decision needs a threshold $t$: flag when $\hat{p} \geq t$. The default `predict` uses $t = 0.5$, which is optimal only for calibrated probabilities with equal error costs.

**From costs (calibrated probabilities).** With false positive cost $c_{\text{FP}}$, false negative cost $c_{\text{FN}}$, and correct decisions costing nothing, flagging is cheaper in expectation when $(1 - p)c_{\text{FP}} < p\,c_{\text{FN}}$:

$$
t^* = \frac{c_{\text{FP}}}{c_{\text{FP}} + c_{\text{FN}}}.
$$

With values instead of costs (an action costs $c$ and, on a true positive, gains $v$): act when $p\,v > c$, so $t^* = c/v$. Example: a retention offer costs USD 30 and saves a churner worth USD 250 with probability 0.4, so $t^* = 30/(0.4 \times 250) = 0.3$.

| Approach | How | When |
|---|---|---|
| Cost formula | $t^* = c_{\text{FP}}/(c_{\text{FP}} + c_{\text{FN}})$ | probabilities are calibrated; costs are known |
| Empirical cost minimization | compute total cost on validation data for a grid of thresholds; pick the minimum | always valid; doesn't need calibration |
| Constraint | the threshold that achieves a required recall (or precision) on validation data | regulatory or safety requirements |
| Capacity (top $k$) | flag the $k$ highest scores | a fixed review budget; evaluate with precision at $k$ |
| Automated | `sklearn.model_selection.TunedThresholdClassifierCV(estimator, scoring=...)` tunes the threshold by cross-validation; `FixedThresholdClassifier(estimator, threshold=t)` applies a chosen one | inside a scikit-learn workflow |

**Threshold pitfalls.**

- Choose the threshold on validation data (or cross-validated predictions), never on the test set.
- Report the expected cost or profit at the chosen threshold next to two baselines: act on nobody and act on everybody.
- Cost curves are often flat near the optimum; a small difference between the formula and the empirical choice is usually noise.
- Re-check the threshold when prevalence or costs change.

## Which metric when?

| Situation | Primary metric | Also report |
|---|---|---|
| Regression, errors cost linearly | MAE | RMSE, $R^2$ |
| Regression, big misses are very costly | RMSE | MAE, worst-case errors |
| Forecast ranges or service levels | pinball loss at each quantile | coverage of the intervals |
| Balanced classification, equal costs | accuracy or log-loss | confusion matrix |
| Imbalanced, rare positives | average precision, recall at a fixed precision | ROC AUC, confusion matrix at the chosen threshold |
| Ranking quality for model comparison | ROC AUC | PR curve |
| Probabilities used as numbers | log-loss, Brier score | reliability diagram |
| A real business decision | expected cost or profit at the chosen threshold | the act-on-nobody and act-on-everybody baselines |
| Multiclass with rare classes | macro F1 | per-class table |

## Universal pitfalls

- **Evaluate on held-out data** that played no part in training, tuning, or threshold choice.
- **Compare with a baseline**: predict the mean, the majority class, or the prevalence.
- **Use the same evaluation set and folds** when comparing models, and look at the spread across folds before declaring a winner.
- **Respect structure in the split**: groups (`GroupKFold`) and time (`TimeSeriesSplit`).
- **Mind scikit-learn's sign convention**: scorers are "higher is better," so losses appear as `neg_mean_squared_error`, `neg_log_loss`, and so on.
- **Mind the input**: `roc_auc_score`, `average_precision_score`, `log_loss`, and `brier_score_loss` take scores or probabilities (`predict_proba(X)[:, 1]`), not hard labels.
