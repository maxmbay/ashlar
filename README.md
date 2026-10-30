<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img alt="ashlar" src="docs/images/logo-light.svg" width="720">
  </picture>
</p>

Ashlar fits tables and forecasts series with JAX neural networks. You call `fit` and `predict`. Categorical columns become learned vectors. A regression or a forecast can return a whole distribution, not just a point: a Gaussian, a mixture, a spline flow, a histogram, or a set of quantiles.

It slots in wherever a gradient-boosted model does. It follows the scikit-learn estimator interface, takes a pandas or Polars frame as it is, string columns included, and works with `clone`, `Pipeline`, `cross_val_score`, and `GridSearchCV`. Swap `LGBMRegressor()` or `XGBRegressor()` for `ASRRegressor()`, and `LGBMClassifier()` for `ASRClassifier()`. On eight public tables of half a million to two million rows, the defaults win or tie on five and are close behind on two. For a predictive distribution they beat the trees on most tables: [the comparison below](#ashlar-beside-gradient-boosted-trees) has every number, including where the trees still win.

- [Install](#install)
- [Quick start](#quick-start)
- [Ashlar beside gradient-boosted trees](#ashlar-beside-gradient-boosted-trees)
- [Pick a head for the shape of the target](#pick-a-head-for-the-shape-of-the-target)
- [Learned embeddings](#learned-embeddings)
- [Documentation](#documentation)

## Install

From a checkout of [maxmbay/ashlar](https://github.com/maxmbay/ashlar):

```bash
pip install -e ".[dataframes]"
```

The import name is `ashlar`. The distribution name is `ashlar-tabular`. The PyPI project named `ashlar` is an unrelated microscopy stitcher, so install from this checkout. CPU is enough. Set `JAX_PLATFORMS=cpu` to run on CPU.

## Quick start

A table. `X` is a NumPy array, a dict of columns, or a pandas or Polars frame. Numeric columns are imputed and scaled. String columns get a learned embedding.

```python
from ashlar import ASRRegressor, ASRClassifier, ASROrdinalClassifier, ASROutcomeTree

model = ASRRegressor().fit(X, y)              # where you had LGBMRegressor()
model.predict(X_new)

clf = ASRClassifier().fit(X, labels)          # where you had LGBMClassifier(); predict, predict_proba
rank = ASROrdinalClassifier().fit(X, ratings) # predict, predict_score (expected label)
tree = ASROutcomeTree({"wicket": None, "runs": {"dot": None, "scoring": ["four", "six", "ran"]}}).fit(X, outcomes)
                                              # one branch softmax per node; predict_conditional(X, "scoring")
```

`score` is accuracy for a classifier and R² for a regressor, and scikit-learn tags `ASRClassifier()` as a classifier, so `cross_val_score` stratifies its folds and the `roc_auc` scorer reads `predict_proba`.

A regression can return its whole predictive distribution:

```python
model = ASRRegressor(task="mixture").fit(X, y)   # or gaussian, flow, histogram, quantile
model.predict_quantiles(X_new, [0.1, 0.5, 0.9])
model.sample(X_new, n=1000)
model.log_prob(X_new, y_new)
```

A series, with a distributional head.

```python
from ashlar import ASRForecaster

model = ASRForecaster(lookback=48, horizon=1, head="mixture").fit(series)
model.predict(history)                     # one point per horizon step
model.log_prob(history, value)             # density of a value
model.sample(history, key, n=1000)         # draws
```

The heads are `mse` (the default), `quantile`, `gaussian`, `mixture`, `flow`, and `histogram`. `ASRHistoryForecaster` takes the same heads and reads the whole prefix instead of a window.

## Ashlar beside gradient-boosted trees

Eight public tables from OpenML, from 515 thousand to 2 million rows, against LightGBM, XGBoost, and CatBoost. Ashlar is `ASRRegressor()` or `ASRClassifier()` with no arguments. Every model holds out the same share of the training rows for early stopping, and the trees are early-stopped at learning rate 0.05 with up to 4,000 trees. One 80/20 split per table for now; more seeds are running.

| Table | Rows | Target, metric | Ashlar | LightGBM | XGBoost | CatBoost |
|---|---:|---|---:|---:|---:|---:|
| Higgs | 2,000,000 | signal, accuracy ↑ | **0.762** | 0.749 | 0.752 | 0.750 |
| Covertype | 581,012 | 7 forest types, accuracy ↑ | **0.967** | 0.963 | 0.948 | 0.910 |
| Year (MSD) | 515,345 | release year, RMSE ↓ | **8.698** | 8.754 | 8.710 | 8.776 |
| Zurich transport | 2,000,000 | delay, RMSE ↓ | 3.052 | **3.047** | **3.047** | **3.047** |
| Airline delay | 2,000,000 | departure delay in minutes, RMSE ↓ | **29.04** | 29.06 | 29.11 | 29.05 |
| Click prediction | 1,997,410 | click, accuracy ↑ | 0.842 | **0.844** | **0.844** | **0.844** |
| Airlines | 539,383 | delayed, accuracy ↑ | 0.660 | 0.666 | 0.664 | **0.668** |
| NYC green taxi | 581,835 | tip in $, RMSE ↓ | 1.977 | 1.752 | 1.869 | **1.758** |

Ashlar is clearly ahead on Higgs and covertype, slightly ahead on year and airline delay, level on Zurich, and a few tenths of a point behind on click and airlines. The trees win the taxi tips clearly. On ROC AUC the gap on click is wider (0.738 against 0.756). Ashlar used every core of a 12-core laptop and took 3 to 13 minutes per table; each tree ran on one thread and took from 12 seconds (LightGBM on airline delay) to 26 minutes (CatBoost on the taxi tips). The two airline tables carry only the schedule, with no weather. Zurich's columns include the weather that occurred, so that table models delay given the weather, not a forecast. Per-table detail, AUC, log loss, and timings are in [docs/benchmarks.md](docs/benchmarks.md#large-tables).

On five smaller tables, three random splits each, the defaults are level with or ahead of the trees on two and close behind on the others:

| Table | Rows | Target, metric | Ashlar | XGBoost | LightGBM |
|---|---:|---|---:|---:|---:|
| California housing | 20,640 | value in $100k, RMSE ↓ | 0.472 | 0.464 | **0.457** |
| Diamonds | 53,940 | log price, RMSE ↓ | 0.089 | **0.085** | **0.085** |
| Adult income | 48,842 | income over $50k, accuracy ↑ | 0.864 | 0.876 | **0.877** |
| Ames housing | 2,930 | log sale price, RMSE ↓ | **0.117** | 0.122 | 0.120 |
| Abalone | 4,177 | rings, RMSE ↓ | **2.166** | 2.225 | 2.231 |

Before this release the defaults scored 0.563 on California and 0.108 on diamonds. Three changes did most of that: a wider residual MLP (three blocks of 256), a piecewise-linear embedding of every numeric column, and a cosine learning-rate schedule with a batch and epoch budget that grow with the table. [docs/benchmarks.md](docs/benchmarks.md#what-each-change-buys) has the ablation. When accuracy matters more than time, `tabm_piecewise()` puts a 32-member TabM ensemble on the same embeddings.

### Beside other neural packages

PyTabKit's tuned defaults for RealMLP, TabM, FT-Transformer, and MLP-PLR, on one split (seed 0) of each table and the same CPU. Seconds are for one fit.

| Model | California RMSE ↓ | Diamonds RMSE ↓ | Adult accuracy ↑ | NYC green taxi tips RMSE ↓ |
|---|---:|---:|---:|---:|
| Ashlar, default | 0.475 (28 s) | 0.0876 (2 min) | 0.864 (27 s) | 1.977 (3 min) |
| Ashlar, `tabm_piecewise()` | **0.440** (4 min) | 0.0842 (8 min) | **0.869** (4 min) | 1.979 (15 min) |
| TabM (PyTabKit) | 0.453 (11 min) | **0.0797** (49 min) | 0.863 (8 min) | **1.798** (4.2 h) |
| RealMLP (PyTabKit) | 0.472 (13 min) | 0.0882 (11 min) | 0.868 (4 min) | 1.934 (67 min) |
| FT-Transformer (PyTabKit) | 0.466 (11 min) | 0.0853 (38 min) | 0.864 (14 min) | — |
| MLP-PLR (PyTabKit) | 0.489 (2 min) | 0.0896 (5 min) | 0.862 (2 min) | — |
| LightGBM, early-stopped | 0.446 (2 s) | 0.0841 (6 s) | 0.875 (1 s) | 1.752 (1 min) |

`tabm_piecewise()` is the best neural model on California and adult, and PyTabKit's TabM on diamonds and the taxi tips, where it took more than four hours on this CPU. The default is level with RealMLP and FT-Transformer at a fraction of their time. On these small tables LightGBM is still ahead of every neural model except on California, where `tabm_piecewise()` edges it. Across three splits, `tabm_piecewise()` scores 0.447 on California, 0.0845 on diamonds, and 0.865 on adult.

So use Ashlar where you would use a gradient-boosted model, and expect it to match the trees on most large tables. Expect a predictive distribution, vectors for your categories, and a JAX model you can differentiate and compose. Use a tree when a point estimate on a table like the taxi tips is the whole job, or when you need a fit in seconds.

## Pick a head for the shape of the target

A tree regressor predicts a mean. For an interval you fit one more model per quantile, or put a fixed-width normal around the mean. Neither gives a density you can evaluate or sample from. An Ashlar head gives the whole predictive distribution from one fit, and the head is a choice you make for the shape of the target.

| The target looks like | `task` | Why |
|---|---|---|
| A bell whose width changes from row to row | `gaussian` | Two numbers per row, a mean and a scale. Cheap and easy to read. |
| A few distinct regimes, or a sharp peak with a long tail | `mixture` | Five normals with learned weights. Each mode gets its own location and spread. |
| Skewed, heavy-tailed, censored, or with a spike | `flow` | A monotone spline bends a normal into almost any one-dimensional shape and still gives an exact density. |
| Any shape, when coarse resolution is enough | `histogram` | A softmax over 16 fixed bins across ±4 standard deviations. No shape assumption. Each bin is half a standard deviation wide, and the two end bins are open-ended. |
| Only the interval matters | `quantile` | Pinball loss at 0.1, 0.5, and 0.9. No density and no draws. |

```python
model = ASRRegressor(task="mixture").fit(X, y)
model.predict(X_new)                       # the mean
model.predict_quantiles(X_new, [0.1, 0.9])
model.sample(X_new, n=1000)                # draws, on the scale of y
model.log_prob(X_new, y_new)               # log density of what happened
```

The forecasters take the same heads by name.

### Large tables

CRPS scores the whole predictive distribution in the target's units, lower is better. NLL is the negative log density at the value that happened. The trees have two ways to give a distribution: the early-stopped mean with a normal whose width is the validation residual, and CatBoost's `RMSEWithUncertainty`, which fits a mean and a variance per row.

| Table | Score | Ashlar `mixture` | CatBoost mean + variance | Best tree mean + normal |
|---|---|---:|---:|---:|
| Airline delay, minutes | CRPS | **9.17** | 11.46 | 12.17 |
| | NLL | **2.66** | 4.72 | 4.81 |
| | 80% coverage | 80% | 94% | 94% |
| Year | CRPS | **4.05** | 4.41 | 4.61 |
| | NLL | **3.08** | 3.42 | 3.58 |
| | 80% coverage | 77% | 83% | 85% |
| Zurich transport | CRPS | **1.62** | 1.74 | 1.75 |
| | NLL | **2.00** | 2.53 | 2.53 |
| | 80% coverage | 79% | 80% | 79% |
| NYC green taxi tips, $ | CRPS | 0.59 | **0.46** | 0.66 |
| | NLL | **−1.97** | 0.70 | 1.98 |
| | 80% coverage | 82% | 83% | 95% |

The mixture has the best likelihood on every table and the best CRPS on three of four. Its intervals are honest: an interval that promises 80% holds about 80% of held-out values, where the trees' normal holds 94% of airline delays because one width has to serve every flight. On airline delay the mixture's 80% interval is less than half as wide as the trees' (33 minutes against 75) and still covers 80%. The taxi tips are the exception on CRPS: the trees' mean is so much closer there (RMSE 1.75 against 1.98) that CatBoost's mean and variance win on CRPS, while the mixture still puts far more density on the tips that happened, many of them exactly zero.

### Smaller tables

![Average predictive density of four heads on held-out taxi tips.](docs/images/tip-heads.png)

NYC taxi tips with the payment type hidden. A third of tips are exactly zero, the cash fares, and the rest are a skewed bump around $2. The Gaussian head has to cover both with one bell, so it puts only 8% of its mass within 25¢ of zero and spills below it. The flow and the mixture grow a spike there and put 42% and 37% within 25¢ of zero. The histogram sees the spike but cannot make it narrower than one bin.

| Target | Shape | `gaussian` | `mixture` | `flow` | `histogram` | Tree mean + normal |
|---|---|---:|---:|---:|---:|---:|
| Taxi tips, $ | spike at zero, skewed | 0.950 | **0.872** | 0.958 | 0.917 | 1.001 |
| California value, $100k | capped at 5.0 | 0.225 | **0.205** | 0.208 | 0.229 | 0.236 |
| Diamond price, $1,000 | skewed, spread grows with size | 0.216 | **0.202** | 0.232 | 0.343 | 0.274 |
| Abalone rings | near normal, counts | 1.105 | **1.076** | 1.088 | 1.089 | 1.157 |
| Melbourne min. temperature, °C, one day ahead | seasonal, near normal | 1.264 | **1.243** | 1.248 | 1.262 | — |

CRPS, three splits each. The mixture is the best head on every table here, and every Ashlar density head beats a tree mean with a constant-width normal on California, diamonds, and abalone. The right head still depends on the shape: the histogram is the worst head on diamond prices, where most of the mass sits in two or three of its fixed bins.

The intervals are better calibrated too. With an interval that promises 80%, the three LightGBM quantile models cover 70% of held-out California values, 69% of diamond prices, and 72% of abalone rings. Ashlar's mixture covers 76%, 80%, and 78%. On the temperature forecast the mixture beats LightGBM on the same 28-day window on every interval score: pinball 0.554 against 0.573, and coverage 83% against 77%. Every column, with pinball loss, coverage, width, and log-likelihood, is in [docs/benchmarks.md](docs/benchmarks.md).

## Learned embeddings

Every string column becomes a learned vector, 8 numbers by default. Row 0 of each table is for a missing value or a category that never appeared in training, so new categories at prediction time are not an error. The vectors are ordinary arrays you can read back:

```python
model = ASRRegressor().fit(X, y)
names = [column.name for column in model.encoder_.embedding_columns]
zones = model.core_.categorical_embeddings.tables[names.index("pickup_zone")]   # (1 + levels, 8)
```

![Learned taxi pickup-zone vectors, projected to two dimensions.](docs/images/taxi-zones.png)

That table is a model of NYC taxi trip duration from the pickup and drop-off zones, 191 and 203 levels, plus the hour, the weekday, and the borough. There is no distance column. The model was never told where a zone is, yet its first principal component tracks how long trips from that zone run (correlation 0.73), and both airports land far from everything else.

On accuracy alone the embeddings are no magic, and on this small, wide table they lose. Log-duration RMSE, three splits:

| Model | RMSE ↓ |
|---|---:|
| Ashlar, defaults, 8-d embeddings | 0.681 |
| Ashlar, defaults, one-hot | 0.651 |
| Ashlar, longer schedule, 8-d embeddings | 0.682 |
| Ashlar, longer schedule, 16-d embeddings | 0.694 |
| Ashlar, longer schedule, one-hot | 0.580 |
| XGBoost, early-stopped | 0.580 |
| LightGBM, early-stopped, native categorical splits | **0.520** |

With 5,000 training rows spread over about 200 zones on each side, a learned vector per zone has little data behind it, and one-hot with a longer schedule (learning rate `3e-3`, up to 100 epochs, patience 20, chosen by hand on the first split) does better, level with XGBoost. LightGBM is best. What the embeddings add is a representation: 8 numbers per zone instead of a 200-column one-hot block, a geometry you can inspect, cluster, or hand to another model, and a defined vector for a zone the model has never seen.

## Documentation

- [docs/tabular.md](docs/tabular.md): fitting a table. Defaults, recipes, tasks, distributions, embeddings, set columns, and search.
- [docs/forecast.md](docs/forecast.md): the forecasters and each head, call by call.
- [docs/benchmarks.md](docs/benchmarks.md): every number on this page, per table and per metric, and how it was run.
- [`examples/`](examples): runnable scripts. [`examples/benchmark_large.py`](examples/benchmark_large.py) and [`examples/benchmark_readme.py`](examples/benchmark_readme.py) produce the numbers here, and [`examples/ablation_gbm_parity.py`](examples/ablation_gbm_parity.py) the ablations.
