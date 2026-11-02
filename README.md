<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/logo-dark.svg">
    <img alt="ashlar" src="docs/images/logo-light.svg" width="720">
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/ashlar-tabular/"><img alt="PyPI" src="https://img.shields.io/pypi/v/ashlar-tabular?color=147a72"></a>
  <a href="https://pypi.org/project/ashlar-tabular/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/ashlar-tabular?color=147a72"></a>
  <a href="https://github.com/maxmbay/ashlar/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/maxmbay/ashlar/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-147a72"></a>
</p>

Ashlar fits tables with JAX neural networks and returns a whole distribution for every row, not just a number: its spread, its skew, the fixed amounts a claim tends to land on, a ceiling it cannot pass. It drops in where XGBoost or LightGBM does, with the same scikit-learn calls on a pandas or Polars frame. On eight public tables of up to two million rows, its accuracy is within half a percent of the best gradient-boosted tree on three, ahead on two, and behind on three, by 13% on taxi tips. Its distributions tie or beat the best tree distributions, LightGBMLSS's among them, on three of the four regression tables. A label can say what was actually observed, such as a sale capped by an empty shelf or a patient still alive at the last visit, and the model learns what lies behind it.

![Sixteen held-out flights from the airline delay table, one ridge each: the departure delay that one ASRRegressor(task="mixture") fit predicts for that flight, skewed to the right, some with two humps, each with a stem at zero for the chance of leaving on the minute. Under each ridge, the 80% interval of LightGBMLSS's mixture of three normals for the same flight, and a dot for the delay that happened. At the right, both models' chance of a delay over an hour.](docs/images/readme/hero-flights.png)

One fit on 1.6 million US flights from 1987 to 2008, schedule only, no weather. On CRPS, the error of the whole predicted distribution in minutes (lower is better), it ties LightGBMLSS's mixture of three normals, the best tree distribution here: 9.17 against 9.14. Calibrated to hold 80%, its intervals are 7% wider than LightGBMLSS's, but they hold 79% to 81% in every fifth of flights, from those it rates calmest to the riskiest; LightGBMLSS's hold 77% to 86%. Its chance of an hour's delay is within 0.6 points of what happened in every bin of 1,000 flights or more, where LightGBMLSS's is up to 8.3 points off. It trains in 4 minutes on all 12 cores of a laptop; LightGBMLSS takes 20 minutes on six threads.

## Install

```bash
pip install "ashlar-tabular[dataframes]"
```

The import name is `ashlar`. This installs JAX's CPU build, which is all Ashlar needs: every number on this page was trained on a CPU.

<details>
<summary>Training on an NVIDIA GPU</summary>

Add JAX's CUDA build (Linux). It brings its own CUDA libraries, so only the NVIDIA driver has to be on the machine:

```bash
pip install "ashlar-tabular[dataframes]" "jax[cuda12]"
python -c "import jax; print(jax.devices())"   # should list a CudaDevice
```

Ashlar code does not change: JAX puts the arrays on the GPU when it finds one. For other CUDA versions and platforms, see [JAX's install guide](https://docs.jax.dev/en/latest/installation.html).

</details>

## Quick start

If you have an XGBoost or LightGBM call, you already have an Ashlar one:

```python
from ashlar import ASRClassifier, ASRRegressor
from xgboost import XGBClassifier, XGBRegressor

model = XGBRegressor().fit(X, y)          # before
model = ASRRegressor().fit(X, y)          # after
model.predict(X_new)

model = XGBClassifier().fit(X, labels)    # before
model = ASRClassifier().fit(X, labels)    # after
model.predict_proba(X_new)
```

`X` is a NumPy array, a dict of columns, or a pandas or Polars frame, with no preprocessing. Numeric columns are imputed and scaled inside the model, string columns get a learned embedding, and a category never seen in training is not an error. `clone`, `Pipeline`, `cross_val_score`, and `GridSearchCV` work as they did.

One argument turns the same model into a predictive distribution for every row, with exact quantiles, tail probabilities, draws, and likelihoods from one fit:

```python
model = ASRRegressor(task="mixture").fit(X, y)

model.predict(X_new)                          # still the mean
model.predict_quantiles(X_new, [0.1, 0.9])    # any quantile, from the same fit
model.predict_exceedance(X_new, 60)           # P(y > 60), row by row
model.sample(X_new, n=1000)                   # draws
model.log_prob(X_new, y_new)                  # how likely what happened was

model.calibrate(X_holdout, y_holdout)         # rows the model did not see
model.predict_interval(X_new, coverage=0.8)   # split-conformal: holds 80% of new rows
```

No second model per quantile, and no fixed-width normal around the mean.

![One held-out flight, Delta from Washington Dulles to Atlanta at 17:25 on a Friday in August: its predicted delay distribution with the calibrated 80% interval solid, the median and mean marked, LightGBMLSS's raw 80% interval under it, and the 49 minutes it was late. Under the distribution, each call of the quick start with its answer placed on the same delay axis: predict 16.4 beside LightGBM's 21.4, predict_quantiles −3.4 to 53.5, predict_exceedance 0.086, five draws from sample, log_prob −6.17, predict_interval after calibrate −3.4 to 52.5, and the top three columns of explain for the 0.9 quantile.](docs/images/readme/one-flight.png)

Those calls on one held-out flight, from the same fit as the ridges above, each answer placed on the delay axis of the distribution it came from.

## Against the best trees on eight large tables

![Two panels over eight OpenML tables, each row a dot for Ashlar's change against the best tree on that table, left of zero when Ashlar is better. Left, test error against the best of LightGBM, XGBoost, and CatBoost: three regression tables within half a percent, Higgs and covertype ahead, click prediction and airlines a percent or two behind, taxi tips 13% behind. Right, the predicted distribution against the best tree distribution: CRPS level with LightGBMLSS's mixture on airline delay, 3% and 2% lower than it on song release years and Zurich transport, and 36% higher than it on taxi tips; log loss lower on Higgs and covertype, higher on click prediction and airlines.](docs/images/readme/scoreboard.png)

Eight OpenML tables of half a million to two million rows. Each dot is Ashlar's change against the best gradient-boosted tree on that table, left of zero when Ashlar is better: test error on the left, the predicted distribution's CRPS or log loss on the right. LightGBM trains far faster, in seconds to minutes a table, where Ashlar takes minutes on every core of a laptop. [Both tables, the training times, and five smaller tables](docs/showcase.md#against-the-best-trees-on-eight-large-tables).

## What it can do

Each capability below is a keyword argument, a label, or a method on the same model. Each tile opens its showcase in [docs/showcase.md](docs/showcase.md): a figure against the tool you would otherwise use (a gradient-boosted tree, a mixed model, R's `gamlss`, survival AFT, or DeepAR), the numbers, where that tool wins, and the code.

<table>
  <tr>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#a-shape-for-every-target"><img src="docs/images/readme/thumbs/shapes.png" alt="A shape for every target"></a><br><a href="docs/showcase.md#a-shape-for-every-target"><b>A shape for every target</b></a><br><sub>42% of claims sit at four amounts: 33% predicted, CatBoost 3%</sub></td>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#lifetimes-including-the-ones-still-running"><img src="docs/images/readme/thumbs/survival.png" alt="Lifetimes, including the ones still running"></a><br><a href="docs/showcase.md#lifetimes-including-the-ones-still-running"><b>Lifetimes, still running</b></a><br><sub>4 of 5 splits calibrated, against 0 of 5 for tuned AFT</sub></td>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#labels-that-say-what-was-observed"><img src="docs/images/readme/thumbs/observed.png" alt="Labels that say what was observed"></a><br><a href="docs/showcase.md#labels-that-say-what-was-observed"><b>Labels that say what was observed</b></a><br><sub>All four shelves empty: 11.4% predicted, 11.1% seen, 0.7% from separate fits (simulated)</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#several-outcomes-one-model"><img src="docs/images/readme/thumbs/outcomes.png" alt="Several outcomes, one model"></a><br><a href="docs/showcase.md#several-outcomes-one-model"><b>Several outcomes, one model</b></a><br><sub>Bikes per hour, calibrated to 80%: 80 riders wide, against LightGBM's 85</sub></td>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#small-groups-borrow-from-large-ones"><img src="docs/images/readme/thumbs/pooling.png" alt="Small groups borrow from large ones"></a><br><a href="docs/showcase.md#small-groups-borrow-from-large-ones"><b>Small groups borrow from large ones</b></a><br><sub>CRPS 0.096 on stores with 1 to 5 days, against GPBoost's 0.125</sub></td>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#a-model-you-can-read"><img src="docs/images/readme/thumbs/additive.png" alt="A model you can read"></a><br><a href="docs/showcase.md#a-model-you-can-read"><b>A model you can read</b></a><br><sub>gamlss's shapes in 10 seconds a fit, against 4 to 6 minutes</sub></td>
  </tr>
  <tr>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#intervals-that-hold"><img src="docs/images/readme/thumbs/intervals.png" alt="Intervals that hold"></a><br><a href="docs/showcase.md#intervals-that-hold"><b>Intervals that hold</b></a><br><sub>One calibration keeps 21 of 24 hours on target; LightGBM's quantiles, 10</sub></td>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#act-on-the-distribution"><img src="docs/images/readme/thumbs/act.png" alt="Act on the distribution"></a><br><a href="docs/showcase.md#act-on-the-distribution"><b>Act on the distribution</b></a><br><sub>One fit, any cost ratio; LightGBM refits cost up to 23% more</sub></td>
    <td align="center" valign="top" width="33%"><a href="docs/showcase.md#forecasting-work-in-progress"><img src="docs/images/readme/thumbs/forecast.png" alt="Forecasting, work in progress"></a><br><a href="docs/showcase.md#forecasting-work-in-progress"><b>Forecasting, work in progress</b></a><br><sub>12-day total CRPS 2,623 against DeepAR's 3,881</sub></td>
  </tr>
</table>

## Under the hood

A model is five choices, drawn below as five courses of stone: how the columns are encoded, the backbone that reads them, the head that answers, the label, and the training schedule. Four are keyword arguments; the label is what you pass to `fit`, an exact `y` or one that says what was observed. Every capability on this page is a value in one of them, so they combine. The path lit below is one model: an additive backbone under a negative binomial head, fitted to sales that a sold-out shelf capped.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/architecture-dark.svg">
    <img alt="Five courses of model choices: encode, backbone, head, label, and train, with one path lit." src="docs/images/architecture-light.svg" width="100%">
  </picture>
</p>

Every configuration is one class, `ASRModel`, so a model is a dict of arguments you can save, version, or sweep. [The distribution object, `jax.grad` through the whole model, and saving without pickle](docs/showcase.md#under-the-hood) are in the showcase, and [docs/models.md](docs/models.md) walks through each choice.

## Documentation

- [docs/showcase.md](docs/showcase.md): each capability in the gallery in full, with its figure against the tool you would otherwise use, its numbers, where that tool wins, and the code.
- [docs/models.md](docs/models.md): the four slots, and the research models beyond the defaults.
- [docs/heads.md](docs/heads.md): every head and the shape of target it suits, calibrated intervals, and member uncertainty.
- [docs/distributions.md](docs/distributions.md): the one distribution object every head returns, joint heads, constraints, missing inputs, decisions, explanations, and JAX transforms.
- [docs/observed.md](docs/observed.md): censored, interval, survival, rounded, truncated, noisy, and total labels, and which heads take them.
- [docs/additive.md](docs/additive.md): additive models of the whole distribution, a neural GAMLSS, read back with `shape_functions`.
- [docs/effects.md](docs/effects.md): entity effects. Stores, patients, or products pooled across groups and updated from a few rows of a new group.
- [docs/tabular.md](docs/tabular.md): fitting a table. Defaults, recipes, tasks, embeddings, set columns, and search.
- [docs/forecast.md](docs/forecast.md): the two series forecasters, and many series with covariates as a table of forecast origins with a joint head over the whole path.
- [docs/benchmarks.md](docs/benchmarks.md): every number on this page, per table and per metric, with the baselines and how each was run, including where the trees win.
- [`examples/`](examples/): runnable scripts, including the benchmarks, and [`examples/readme/`](examples/readme/), which redraws every figure on this page from committed results.
- [CHANGELOG.md](CHANGELOG.md): what changed in each release.

## Acknowledgements

The `tabm` backbone and the `tabm()` and `tabm_piecewise()` recipes follow TabM, by Yury Gorishniy, Akim Kotelnikov, and Artem Babenko ([paper](https://arxiv.org/abs/2410.24210), [code](https://github.com/yandex-research/tabm)). The piecewise-linear and periodic numeric embeddings follow [Gorishniy, Rubachev, and Babenko (2022)](https://arxiv.org/abs/2203.05556). TabM is also the strongest neural baseline in our benchmarks.
