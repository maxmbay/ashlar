<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/logo-dark.svg">
    <img alt="ashlar" src="https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/logo-light.svg" width="720">
  </picture>
</p>

Ashlar fits tables and forecasts series with JAX neural networks. You call `fit` and `predict`. Categorical columns become learned vectors. A regression or a forecast can return a whole distribution, not just a point: a Gaussian, a mixture, a spline flow, a histogram, or a set of quantiles.

It slots in wherever a gradient-boosted model does. Swap `LGBMRegressor()` for `ASRRegressor()` and `LGBMClassifier()` for `ASRClassifier()`: the scikit-learn interface, a pandas or Polars frame as it is, string columns included, and `clone`, `Pipeline`, `cross_val_score`, and `GridSearchCV` all work. On eight public tables of half a million to two million rows the defaults win or tie on five, and for a predictive distribution they beat the trees on most tables. [The benchmarks](https://github.com/maxmbay/ashlar/blob/main/docs/benchmarks.md) have every number, including where the trees still win.

## Install

```bash
pip install -e ".[dataframes]"
```

From a checkout of [maxmbay/ashlar](https://github.com/maxmbay/ashlar). The import name is `ashlar` and the distribution name is `ashlar-tabular`; the PyPI project named `ashlar` is an unrelated microscopy stitcher. CPU is enough.

## Quick start

If you have an XGBoost or LightGBM call, you already have an Ashlar one. Regression:

```python
model = XGBRegressor().fit(X, y)                  # before
model = ASRRegressor().fit(X, y)                  # after
model.predict(X_new)
```

Classification:

```python
model = XGBClassifier().fit(X, labels)            # before
model = ASRClassifier().fit(X, labels)            # after
model.predict(X_new), model.predict_proba(X_new), model.score(X_new, labels_new)
```

`X` is a NumPy array, a dict of columns, or a pandas or Polars frame. Numeric columns are imputed and scaled, string columns get a learned embedding, and a category that never appeared in training is not an error. `clone`, `Pipeline`, `cross_val_score`, and `GridSearchCV` work as they did.

One argument turns the same model into a conditional predictive distribution, which a gradient-boosted tree cannot return from a single fit:

```python
model = ASRRegressor(task="mixture").fit(X, y)

model.predict(X_new)                        # still the mean
model.predict_quantiles(X_new, [0.1, 0.9])  # an interval, on the scale of y
model.sample(X_new, n=1000)                 # draws
model.log_prob(X_new, y_new)                # log density of what happened
```

No second model per quantile, and no fixed-width normal around the mean.

## Architecture is set by keyword arguments

A model is one set of keyword arguments with four independent slots: how the columns are encoded, the backbone that reads them, the head that answers, and the training schedule. Changing the head makes the same network return a density instead of a number. Changing the backbone leaves the head unchanged.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/architecture-dark.svg">
    <img alt="Four courses of model choices: encode, backbone, head, and train, with one path lit." src="https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/architecture-light.svg" width="900">
  </picture>
</p>

```python
ASRRegressor()                                      # the defaults above
ASRRegressor(task="mixture")                        # same net, now a predictive density
ASRRegressor(task="flow", backbone="tabm")          # skewed target, ensemble trunk
ASRClassifier(backbone="transformer")               # attention over the columns
```

None of these are wrapper classes: every line is `ASRModel` with different arguments, so a configuration is a dict you can save, version, or sweep. `backbone=` also takes your own Equinox module, and `task="custom"` takes your own head and objective. [docs/models.md](https://github.com/maxmbay/ashlar/blob/main/docs/models.md) is the tour of what goes in each slot.

## Conditional predictive distributions, one head per target shape

The distribution is conditional on the row, so one fitted model returns a different shape for every row, and the head sets which shapes it can represent.

![Four heads on taxi fare, under three conditions.](https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/fare-heads.png)

One fit per head on taxi fares, given the zones, the hour, the weekday, the passengers, and the cab color, with distance and the total held out. A different head wins each condition, by held-out log density: the `mixture` on Manhattan hops, the `flow` on the heavier Brooklyn tail, and the `histogram` on the Queens corridor, where 32% of trips run at the $52 airport flat fare. Only the histogram grows a real spike there, 23% of its mass against the mixture's 5%, because a spike at one value is what a smooth density cannot make. Its resolution is yours: `head_kwargs={"bins": 48}`, 16 by default.

`gaussian`, `mixture`, `flow`, `histogram`, and `quantile` for a number; `ASRClassifier` for classes, `ASROrdinalClassifier` for an ordered scale, and `ASROutcomeTree` for nested outcomes. [docs/heads.md](https://github.com/maxmbay/ashlar/blob/main/docs/heads.md) says which to reach for and why.

## Forecasting

A series reaches Ashlar two ways. `ASRForecaster` reads a window of the series itself and emits the next steps:

```python
from ashlar import ASRForecaster

model = ASRForecaster(lookback=48, horizon=24, head="mixture").fit(series)
model.predict(history)                     # one point per horizon step
model.sample(history, key, n=1000)         # draws
```

Or the series is already a table of lags and calendar columns, and every tabular head applies to it. `cyclical=` turns a calendar column into a sine and a cosine, so hour 23 sits beside hour 0 and days near each other in the year share what the model learns about them:

```python
from ashlar import tabm_piecewise

model = tabm_piecewise(task="mixture", cyclical={"hour": 24.0, "weekday": 7.0, "day_of_year": 365.25})
model.fit(lags, pickups)
```

![Hourly taxi demand over a held-out day, with the 80% band around the median forecast.](https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/forecast-fan.png)

Hourly NYC taxi pickups, July 2014 to January 2015. Each row is one hour, and reads the 24 hours of the previous day, the same hour a week earlier, and the calendar. A sixth of the days were held out whole, so each is forecast 1 to 24 hours ahead from midnight, by one fit rather than one model per step. Over the 34 held-out days the median forecast is off by 2,678 pickups on average, against 3,806 for the same hour a week earlier, and the 80% interval holds 73% of held-out hours.

`ASRHistoryForecaster` reads the whole prefix instead of a window. [docs/forecast.md](https://github.com/maxmbay/ashlar/blob/main/docs/forecast.md) covers both, call by call.

## Nested outcomes as a tree of labels

Some targets are a tree, not a flat set of classes: a ball is a wicket, a dot, or runs, and runs are a four or a six. `ASROutcomeTree` fits one backbone with a softmax per split, so a label that stops early (`"runs"`, with no detail recorded) still trains the splits above it.

![One backbone with a softmax per split, over a tree of ball outcomes.](https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/outcome-tree.png)

```python
from ashlar import ASROutcomeTree

model = ASROutcomeTree({"wicket": None, "dot": None, "runs": ["four", "six"]}).fit(X, outcomes)
model.predict_proba(X_new)                  # the leaves
model.predict_conditional(X_new, "runs")    # P(four or six | runs, x)
```

Every categorical column in that model is a learned vector, and the branch probabilities are a function of those vectors together with the rest of the row, not of either alone:

![Learned pickup-zone vectors beside P(tip) as a function of fare, for four zones.](https://raw.githubusercontent.com/maxmbay/ashlar/main/docs/images/outcome-embeddings.png)

An outcome tree over taxi tips, with the leaves `no_tip`, `under_two`, and `two_or_more`. Each pickup zone becomes an 8-dimensional vector, and the vectors arrange themselves by how often trips from that zone are tipped. The right panel sweeps the fare with every other column fixed: P(tip) climbs 0.30 across the sweep from Central Harlem North and 0.29 from Manhattan Valley, but only 0.09 from Battery Park City, which starts high and stays there. The zone does not shift one curve up or down, it changes the response to the fare.

## Results

Eight public tables of half a million to two million rows, against LightGBM, XGBoost, and CatBoost. Ashlar is `ASRRegressor()` or `ASRClassifier()` with no arguments.

| Table | Rows | Target, metric | Ashlar | Best tree |
|---|---:|---|---:|---:|
| Higgs | 2,000,000 | signal, accuracy ↑ | **0.762** | 0.752 |
| Covertype | 581,012 | 7 forest types, accuracy ↑ | **0.967** | 0.963 |
| Year (MSD) | 515,345 | release year, RMSE ↓ | **8.698** | 8.710 |
| Airline delay | 2,000,000 | delay in minutes, RMSE ↓ | **29.04** | 29.05 |
| Zurich transport | 2,000,000 | delay, RMSE ↓ | 3.052 | **3.047** |
| Click prediction | 1,997,410 | click, accuracy ↑ | 0.842 | **0.844** |
| Airlines | 539,383 | delayed, accuracy ↑ | 0.660 | **0.668** |
| NYC green taxi | 581,835 | tip in $, RMSE ↓ | 1.977 | **1.752** |

Four wins, one tie within a thousandth, two a few tenths of a point behind, and one clear loss. Higgs and covertype are not close: on Higgs the defaults beat every tree by more than a point of accuracy (AUC 0.846 against 0.835), and on covertype by half a point against the best tree and five points against CatBoost.

Ask for a distribution and the gap widens. CRPS scores a whole predictive distribution in the target's units, NLL is the negative log density of what happened, and both are lower-is-better:

| Table | Score | Ashlar `mixture` | Best tree distribution |
|---|---|---:|---:|
| Airline delay, minutes | CRPS | **9.17** | 11.46 |
| | NLL | **2.66** | 4.72 |
| Year (MSD) | CRPS | **4.05** | 4.41 |
| | NLL | **3.08** | 3.42 |
| Zurich transport | CRPS | **1.62** | 1.74 |
| | NLL | **2.00** | 2.53 |
| NYC green taxi tips, $ | CRPS | 0.59 | **0.46** |
| | NLL | **−1.97** | 0.70 |

The best likelihood on all four, and the best CRPS on three. The intervals are honest too: an Ashlar interval that promises 80% holds about 80% of held-out values, where a tree mean with a fitted normal holds 94% of airline delays because one width has to serve every flight. On airline delay the mixture's 80% interval is less than half as wide, 33 minutes against 75, and still covers 80%.

On five smaller tables the defaults win two (Ames, abalone) and trail LightGBM on the rest, where `tabm_piecewise()` closes most of the gap. [docs/benchmarks.md](https://github.com/maxmbay/ashlar/blob/main/docs/benchmarks.md) has every number: all eight large tables per metric, the five small ones, RealMLP, TabM, and FT-Transformer, the ablation behind the defaults, and the forecast heads.

## Documentation

- [docs/models.md](https://github.com/maxmbay/ashlar/blob/main/docs/models.md): the four slots, and the research models beyond the defaults.
- [docs/heads.md](https://github.com/maxmbay/ashlar/blob/main/docs/heads.md): every head and the shape of target it suits.
- [docs/tabular.md](https://github.com/maxmbay/ashlar/blob/main/docs/tabular.md): fitting a table. Defaults, recipes, tasks, embeddings, set columns, and search.
- [docs/forecast.md](https://github.com/maxmbay/ashlar/blob/main/docs/forecast.md): the two forecasters, call by call.
- [docs/benchmarks.md](https://github.com/maxmbay/ashlar/blob/main/docs/benchmarks.md): every number, per table and per metric, and how it was run.
- [`examples/`](https://github.com/maxmbay/ashlar/tree/main/examples): runnable scripts, including the benchmarks and the figures on this page.
