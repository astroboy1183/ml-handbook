# scikit-learn Cheat Sheet

A dense reference for scikit-learn 1.5+ (tested on 1.9): the estimator API, preprocessing, the model zoo, model selection, pipelines, metrics, persistence, and the pitfalls that cost the most time.

Related chapters: [What is machine learning?](../chapters/03-ml-fundamentals/01-what-is-machine-learning.md) · [ML pipelines](../chapters/05-applied-ml/01-ml-pipelines.md) · [Hyperparameter tuning](../chapters/05-applied-ml/02-hyperparameter-tuning.md) · [Evaluation metrics](../chapters/03-ml-fundamentals/06-evaluation-metrics.md)

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
```

## The estimator API

Every object follows the same small contract:

| Method / attribute | Who has it | What it does |
|---|---|---|
| `est = Model(**hyperparams)` | all | stores hyperparameters only; no data, no computation |
| `est.fit(X, y)` | all (`y` optional for unsupervised) | learns from data; returns `self` |
| `est.predict(X)` | predictors | labels (classifiers) or values (regressors) |
| `est.predict_proba(X)` | most classifiers | class probabilities, columns ordered as `est.classes_` |
| `est.decision_function(X)` | linear models, SVMs, boosting | raw scores (margin or log-odds) |
| `est.transform(X)` / `fit_transform(X, y)` | transformers | new features |
| `est.score(X, y)` | predictors | accuracy (classifiers) or R² (regressors) |
| `est.get_params()` / `set_params(**p)` | all | read or change hyperparameters (nested: `step__param`) |
| `attr_` (trailing underscore) | fitted estimators | learned values: `coef_`, `classes_`, `feature_importances_`, `n_features_in_`, `feature_names_in_` |
| `sklearn.base.clone(est)` | all | an unfitted copy with the same hyperparameters |

`X` is 2-D (`n_samples × n_features`: a DataFrame or array); `y` is 1-D. Pass `X[["col"]]`, not `X["col"]`, for a single feature.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_train, y_train)
print(clf.classes_, clf.predict_proba(X_test[:2]).round(3), f"accuracy {clf.score(X_test, y_test):.3f}")
```

```text
[0 1] [[0.004 0.996]
 [1.    0.   ]] accuracy 0.958
```

## Preprocessing

| Task | Class (`sklearn.preprocessing` unless noted) | Notes |
|---|---|---|
| Standardize | `StandardScaler()` | mean 0, SD 1; for linear models, SVMs, kNN, PCA |
| Scale to a range | `MinMaxScaler()` | sensitive to outliers |
| Robust scaling | `RobustScaler()` | median and IQR |
| Log, power | `FunctionTransformer(np.log1p)`, `PowerTransformer()` | Yeo-Johnson by default |
| Quantiles | `QuantileTransformer(output_distribution="normal")` | rank-based |
| Bin | `KBinsDiscretizer(n_bins=5, encode="ordinal")` | |
| Polynomials, interactions | `PolynomialFeatures(degree=2, include_bias=False)` | features grow fast |
| Splines | `SplineTransformer()` | smooth nonlinear effects for linear models |
| One-hot | `OneHotEncoder(handle_unknown="ignore", min_frequency=20)` | `sparse_output=False` for dense |
| Ordinal | `OrdinalEncoder(categories=[["S", "M", "L"]])` | give the order explicitly |
| Target encoding | `TargetEncoder()` | cross-fits in `fit_transform`; for high cardinality |
| Impute | `sklearn.impute.SimpleImputer(strategy="median")` | `add_indicator=True` flags missingness |
| Impute (model-based) | `KNNImputer()`, `IterativeImputer()` (needs `sklearn.experimental.enable_iterative_imputer`) | slower |
| Select features | `sklearn.feature_selection.SelectKBest(f_classif, k=20)`, `SelectFromModel(...)`, `RFECV(...)` | always inside the pipeline |
| Text | `sklearn.feature_extraction.text.TfidfVectorizer()` | one text column, as a 1-D input |
| Reduce dimension | `sklearn.decomposition.PCA(n_components=0.95)` | keep 95% of variance |

## Model zoo imports

| Family | Classification | Regression |
|---|---|---|
| Linear | `linear_model.LogisticRegression` | `linear_model.LinearRegression`, `Ridge`, `Lasso`, `ElasticNet` |
| Linear (large data) | `linear_model.SGDClassifier` | `linear_model.SGDRegressor` |
| Neighbors | `neighbors.KNeighborsClassifier` | `neighbors.KNeighborsRegressor` |
| Naive Bayes | `naive_bayes.GaussianNB`, `MultinomialNB` | |
| SVM | `svm.SVC(probability=False)`, `LinearSVC` | `svm.SVR`, `LinearSVR` |
| Tree | `tree.DecisionTreeClassifier` | `tree.DecisionTreeRegressor` |
| Bagging | `ensemble.RandomForestClassifier`, `ExtraTreesClassifier` | `ensemble.RandomForestRegressor`, `ExtraTreesRegressor` |
| Boosting | `ensemble.HistGradientBoostingClassifier` (fast; handles NaN and categoricals) | `ensemble.HistGradientBoostingRegressor` |
| Stacking, voting | `ensemble.StackingClassifier`, `VotingClassifier` | `StackingRegressor`, `VotingRegressor` |
| Neural net (small) | `neural_network.MLPClassifier` | `neural_network.MLPRegressor` |
| Baseline | `dummy.DummyClassifier(strategy="most_frequent")` | `dummy.DummyRegressor(strategy="median")` |

Unsupervised: `cluster.KMeans`, `DBSCAN`, `AgglomerativeClustering`, `mixture.GaussianMixture`, `decomposition.PCA`, `manifold.TSNE`, `ensemble.IsolationForest`. XGBoost (`xgboost.XGBClassifier`) and LightGBM (`lightgbm.LGBMClassifier`) follow the same API.

## Splitting and cross-validation

| Situation | Splitter (`sklearn.model_selection`) |
|---|---|
| Classification | `StratifiedKFold(5, shuffle=True, random_state=0)` (default for classifiers when `cv=5`, without shuffling) |
| Regression | `KFold(5, shuffle=True, random_state=0)` |
| Small data, less noise | `RepeatedStratifiedKFold(n_splits=5, n_repeats=10)` |
| Groups (patients, users) | `GroupKFold(5)` or `StratifiedGroupKFold`; pass `groups=` |
| Time order | `TimeSeriesSplit(n_splits=5, gap=...)` |
| One holdout | `train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)` |

```python
from sklearn.model_selection import StratifiedKFold, cross_validate

cv = StratifiedKFold(5, shuffle=True, random_state=0)
res = cross_validate(clf, X_train, y_train, cv=cv, scoring=["roc_auc", "neg_log_loss"], return_train_score=True)
print({k: round(float(v.mean()), 3) for k, v in res.items() if "test" in k or "train" in k})
```

```text
{'test_roc_auc': 0.995, 'train_roc_auc': 0.998, 'test_neg_log_loss': -0.076, 'train_neg_log_loss': -0.048}
```

Also: `cross_val_score` (one metric), `cross_val_predict` (out-of-fold predictions, for thresholds and stacking), `learning_curve`, `validation_curve`. Scores are "higher is better", so losses appear negated (`neg_log_loss`, `neg_root_mean_squared_error`).

## Hyperparameter search

| Tool | Use when |
|---|---|
| `GridSearchCV(est, param_grid, cv=cv, scoring=...)` | 1 to 2 hyperparameters, few values |
| `RandomizedSearchCV(est, param_distributions, n_iter=60, random_state=0)` | the default; use `scipy.stats.loguniform(1e-3, 1e2)` for scale-like parameters |
| `HalvingRandomSearchCV(...)` (needs `from sklearn.experimental import enable_halving_search_cv`) | many candidates, large data |
| Optuna (`optuna.create_study(...).optimize(objective, n_trials=50)`) | expensive trials, conditional spaces, pruning |
| `TunedThresholdClassifierCV(est, scoring=...)` | tune the decision threshold by CV |

After `fit`: `best_params_`, `best_score_` (optimistic: the max over many noisy scores), `best_estimator_` (refit on all data), `cv_results_` (as a DataFrame: `pd.DataFrame(search.cv_results_)`). For an honest estimate, use nested CV: `cross_val_score(GridSearchCV(...), X, y, cv=outer)`.

## Pipelines and ColumnTransformer

```python
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

df = X_train.assign(size=np.where(X_train["mean radius"] > 14, "large", "small"))   # add a categorical
numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
prep = ColumnTransformer([
    ("num", numeric, make_column_selector(dtype_include="number")),
    ("cat", categorical, ["size"]),
])                                                       # remainder="drop" by default
pipe = Pipeline([("prep", prep), ("model", HistGradientBoostingClassifier(random_state=0))])
search = GridSearchCV(pipe, {"model__learning_rate": [0.05, 0.1], "prep__num__impute__strategy": ["mean", "median"]},
                      cv=cv, scoring="roc_auc").fit(df, y_train)
print(search.best_params_)
print(search.best_estimator_.named_steps["prep"].get_feature_names_out()[[0, -1]])
```

```text
{'model__learning_rate': 0.05, 'prep__num__impute__strategy': 'mean'}
['num__mean radius' 'cat__size_small']
```

| Need | How |
|---|---|
| Name steps automatically | `make_pipeline(...)`, `make_column_transformer(...)` |
| Access a step | `pipe.named_steps["model"]`, `pipe[-1]`, `pipe[:-1]` (all but the model) |
| Parameter path | `step__substep__param` (double underscores) |
| DataFrame output | `pipe.set_output(transform="pandas")` or `sklearn.set_config(transform_output="pandas")` |
| Feature names | `pipe[:-1].get_feature_names_out()` |
| Keep unlisted columns | `ColumnTransformer(..., remainder="passthrough")` |
| Stateless custom step | `FunctionTransformer(func, feature_names_out="one-to-one")` |
| Stateful custom step | subclass `BaseEstimator, TransformerMixin`; `__init__` stores args; `fit` sets `attr_` and returns `self` |
| Resampling (SMOTE) | `imblearn.pipeline.Pipeline` with `imblearn.over_sampling.SMOTE()`; resamples only during `fit` |
| Cache slow steps | `Pipeline(..., memory="cachedir")` |

## Metrics

| Task | Functions (`sklearn.metrics`) | `scoring=` string |
|---|---|---|
| Classification, labels | `accuracy_score`, `balanced_accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `fbeta_score`, `matthews_corrcoef` | `"accuracy"`, `"balanced_accuracy"`, `"f1"`, `"precision"`, `"recall"` |
| Classification, scores | `roc_auc_score`, `average_precision_score`, `log_loss`, `brier_score_loss` | `"roc_auc"`, `"average_precision"`, `"neg_log_loss"`, `"neg_brier_score"` |
| Reports | `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`, `RocCurveDisplay`, `PrecisionRecallDisplay` | |
| Curves | `roc_curve`, `precision_recall_curve`, `sklearn.calibration.calibration_curve` | |
| Regression | `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `r2_score`, `mean_absolute_percentage_error` | `"neg_mean_absolute_error"`, `"neg_root_mean_squared_error"`, `"r2"` |
| Clustering | `silhouette_score`, `adjusted_rand_score` | |
| Custom | `make_scorer(func, greater_is_better=False)` | pass the scorer object |

Multiclass: `f1_score(..., average="macro" | "weighted" | "micro")`, `roc_auc_score(..., multi_class="ovr")`. Calibrate probabilities with `sklearn.calibration.CalibratedClassifierCV(est, method="sigmoid" | "isotonic", cv=5)`. Explain models with `sklearn.inspection.permutation_importance` and `PartialDependenceDisplay.from_estimator(..., kind="both")`.

## Persistence

```python
import json
from pathlib import Path

import joblib
import sklearn

Path("models").mkdir(exist_ok=True)
joblib.dump(search.best_estimator_, "models/model.joblib")                 # the whole fitted pipeline
Path("models/meta.json").write_text(json.dumps({"sklearn": sklearn.__version__, "cv_auc": search.best_score_}))
loaded = joblib.load("models/model.joblib")
print(np.allclose(loaded.predict_proba(df[:5]), search.predict_proba(df[:5])))
```

```text
True
```

- Save the **whole pipeline**, not just the model, so preprocessing travels with it.
- Record library versions; load with the same scikit-learn version (otherwise `InconsistentVersionWarning`, and possibly wrong results).
- Loading a pickle or joblib file can execute code: only load files you trust. `skops.io` is a safer format for sharing; ONNX (`skl2onnx`) serves models outside Python.
- Custom classes must be importable (from a module, not a notebook's `__main__`) wherever the model is loaded.

## Common pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| Preprocessing before splitting (`fit_transform` on all data) | CV score too good; production worse | put every learned step in the pipeline; CV the pipeline |
| Resampling or feature selection before CV | near-perfect CV recall or accuracy | `imblearn` pipeline; `SelectKBest` inside the pipeline |
| `fit_transform` on the test set | train/test features on different scales | `transform` only on test data (a pipeline does this for you) |
| Reporting `best_score_` | tuned model disappoints | nested CV or a once-used test set |
| Accuracy on imbalanced data | 98% accuracy, zero recall | average precision, recall at a precision, expected cost |
| Default 0.5 threshold | few positives flagged | threshold from costs, or `TunedThresholdClassifierCV` |
| Trusting `feature_importances_` | noise features look important | `permutation_importance` on held-out data |
| Random K-fold on time series or grouped data | optimistic scores | `TimeSeriesSplit`, `GroupKFold` |
| Unscaled features for SVM, kNN, regularized linear models | poor or unstable fits; convergence warnings | `StandardScaler` in the pipeline |
| Missing `random_state` | results change every run | seed every estimator and splitter |
| `X["col"]` instead of `X[["col"]]` | "Expected 2D array" error | keep `X` 2-D |
| One-hot without `handle_unknown="ignore"` | crash on a new category in production | set it, or `min_frequency` to group rare levels |
| `predict_proba` column order assumed | probabilities for the wrong class | read `est.classes_` |
| Probabilities from rebalanced models used as-is | inflated risk estimates | prior correction or `CalibratedClassifierCV` |
| Pickled model loaded with another version | warnings, wrong predictions | pin versions; store them with the model |
