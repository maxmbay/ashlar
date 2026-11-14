# Using Ashlar like scikit-learn

[Ashlar](../README.md)'s estimators are scikit-learn estimators. This page says what they take, what they return and in what shape, which scikit-learn tools work, how a model is saved, and how the dressed estimator differs.

## What `fit` takes

`X` is a table: a two-dimensional NumPy array, a dict of column name to one-dimensional array, a pandas `DataFrame`, or a Polars `DataFrame`. No preprocessing is needed.

- A numeric column may hold `NaN`. It is filled with the training median and scaled ([docs/tabular.md](tabular.md#defaults) has the map).
- An integer column stays numeric unless you name it in `categorical`.
- A string or object column is categorical. It gets a learned embedding of width 8; `categorical_embeddings=[]` gives one-hot instead. A category not seen in training, or a missing one, reads the unknown row, so prediction on new categories works.
- `set_features=["tags"]` names columns whose cells are lists; `cyclical={"hour": 24}` turns a column into a sine and a cosine.

`y` is one column for one target, `(rows, columns)` for a joint head, strings or integers for `ASRClassifier`, and real numbers for `ASROrdinalClassifier`. For a label that was not seen exactly, pass one built with `ashlar.observed` ([docs/observed.md](observed.md)); it is a NumPy structured array, so `train_test_split` and `KFold` slice it like any `y`.

`fit` also takes `sample_weight=`, `validation=(X_val, y_val)` for your own early-stopping rows, `groups=` for a grouped split, `search=True` for the small default grid ([docs/tabular.md](tabular.md#search)), and `resume=True` to continue training.

## What the methods return

| Method | Shape, one target | Joint head | Notes |
|---|---|---|---|
| `predict(X)` | `(rows,)` | `(rows, columns)` | The mean for most heads, the median for `flow` and `quantile`, the location for `student_t`; class labels for a classifier. |
| `predict_proba(X)` | `(rows, classes)` | | Classifiers only, in `classes_` order; a binary target has two columns. |
| `predict_dist(X)` | `Distribution`, `batch_shape (rows,)` | `event_shape (columns,)` | Every head with a likelihood; not `task="regression"`, which predicts a point. [docs/distributions.md](distributions.md). |
| `predict_quantiles(X, levels)` | `(rows, len(levels))` | `(rows, columns, len(levels))` | Exact for density heads. |
| `predict_interval(X, coverage)` | `(rows, 2)` | `(rows, columns, 2)`; `target="sum"` gives `(rows, 2)` | Split-conformal after `calibrate` ([docs/heads.md](heads.md#calibrated-intervals)). |
| `predict_exceedance(X, t)` | `(rows,)` | per column, or of a `target` | `P(y > t)`; recalibrated after `calibrate`. |
| `sample(X, n)` | `(n, rows)` | `(n, rows, columns)` | |
| `log_prob(X, y)` | `(rows,)` | `(rows,)` | A log density for a continuous head, a log mass for a count or a class, and for `hurdle` a log mass at zero and a log density above it. Every density integrates to 1; the `histogram` head's does so within its edges at ±4 standardized SD, and reads a value past them in the end bin. |
| `score(X, y)` | a float | | R² for a regressor, accuracy for a classifier, the mean log-likelihood for an `observed` label. |

## scikit-learn tools

`ASRModel` and its forms inherit `BaseEstimator`, so `get_params`, `set_params`, `clone`, and the repr work, and the constructor arguments are the whole configuration. `__sklearn_tags__` tags a classification, ordinal, or tree model as a classifier and anything else as a regressor, so `cross_val_score` stratifies folds for a classifier and the `roc_auc` scorer reads `predict_proba`. `score` is what the default scorer reads, so `cross_val_score(model, X, y)` and `GridSearchCV(model, grid)` need no `scoring=`. `Pipeline` works with Ashlar as the last step; put a column-dropping or scaling step before it only if you need it, since the estimator already imputes and scales.

```python
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

cross_val_score(ASRClassifier(), X, labels, cv=5, scoring="roc_auc")
GridSearchCV(ASRRegressor(task="mixture"), {"weight_decay": [0.0, 1e-4]}, cv=3).fit(X, y)
```

Every fit on a CPU compiles once per epoch and then trains, so a grid of many small fits is slower per fit than a tree model; [docs/tabular.md](tabular.md#search) has the built-in search, which trains the winner once.

## Saving and loading

`model.save("models/delay")` writes a directory of JSON and Equinox weight files, with no pickle, and `ASRModel.load("models/delay")` returns the class that was saved, with the same predictions, the calibration, and the optimizer state for `fit(..., resume=True)`. A custom head or backbone is recorded by its module path and imported on load, so load only models from sources you trust.

## The dressed estimator

`ASRDressed`, which `ashlar.dressed()` returns, is a scikit-learn regressor too: `get_params`, `set_params`, `clone`, and `is_regressor` work, so `cross_val_score` and `GridSearchCV` treat it as one, and a fitted model pickles. It has `fit(X, y)`, `predict`, `predict_dist`, `predict_quantiles`, `log_prob`, `sample`, and `score` (R² of the mean). Its `log_prob` is an exact log density computed in closed form from the fitted parts, with no sampling; for a target with at most 20 distinct values it is the log probability of the value, `-inf` off the values, and `predict_dist(X).probabilities()` gives each row's probability of each value. It has no `calibrate`, `predict_interval`, `predict_exceedance`, `save`, or `load` yet: read an interval from `predict_quantiles` and a tail probability from `1 - dist.cdf(t)`. It takes exact numeric labels only, at least 20 rows per member (100 with the default five), and no sample weights. [docs/dressed.md](dressed.md) is the guide.
