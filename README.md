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

Ashlar fits tables with JAX neural networks and returns a whole distribution for every row, not just a number: its spread, its skew, the fixed amounts a claim tends to land on, a ceiling it cannot pass. It drops in where XGBoost or LightGBM does, with the same scikit-learn calls on a pandas or Polars frame as it is. On eight public tables of up to two million rows it is level with the best gradient-boosted trees on accuracy, and its distributions beat theirs on three of the four regression tables. A label can say what was actually observed, such as a sale capped by an empty shelf or a patient still alive at the last visit, and the model learns what lies behind it.

![Sixteen held-out flights from the airline delay table, one ridge each: the departure delay that one ASRRegressor(task="mixture") fit predicts for that flight, skewed to the right, some with two humps, each with a stem at zero for the chance of leaving on the minute. Under each ridge, CatBoost RMSEWithUncertainty's 80% interval for the same flight, wider, symmetric, and reaching 20 to 40 minutes early where almost no flight leaves, and a dot for the delay that happened. At the right, both models' chance of a delay over an hour.](docs/images/readme/hero-flights.png)

One fit on 1.6 million US flights, schedule only, no weather. Its 80% intervals hold 80% of held-out delays at 33 minutes wide, where CatBoost's per-flight normal holds 94% at 67; CRPS 9.17 against 11.46.

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

`X` is a NumPy array, a dict of columns, or a pandas or Polars frame as it is. Numeric columns are imputed and scaled inside the model, string columns get a learned embedding, and a category never seen in training is not an error. `clone`, `Pipeline`, `cross_val_score`, and `GridSearchCV` work as they did.

One argument turns the same model into a predictive distribution for every row, which a gradient-boosted tree cannot return from one fit:

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

![One held-out flight, Delta from Washington Dulles to Atlanta at 17:25 on a Friday in August: its predicted delay distribution with the calibrated 80% interval solid, the median and mean marked, CatBoost RMSEWithUncertainty's wider interval under it, and the 49 minutes it was late. Under the distribution, each call of the quick start with its answer placed on the same delay axis: predict 16.4 beside LightGBM's 21.4, predict_quantiles −3.4 to 53.5, predict_exceedance 0.086, five draws from sample, log_prob −6.17, predict_interval after calibrate −3.4 to 52.5, and the top three columns of explain for the 0.9 quantile.](docs/images/readme/one-flight.png)

Those calls on one held-out flight, from the same fit as the ridges above, each answer placed on the delay axis of the distribution it came from.

## Level with the trees on accuracy. Ahead on distributions.

![Two panels over eight OpenML tables, each row a dot for Ashlar's change against the best tree on that table, left of zero when Ashlar is better. Left, test error against the best of LightGBM, XGBoost, and CatBoost: three regression tables within a fraction of a percent, Higgs and covertype ahead, click prediction and airlines a percent or two behind, taxi tips 13% behind. Right, the predicted distribution against the best tree's: CRPS 20%, 8%, and 7% lower on airline delay, song release years, and Zurich transport, 28% higher on taxi tips; log loss lower on Higgs and covertype, higher on click prediction and airlines.](docs/images/readme/scoreboard.png)

Eight OpenML tables of half a million to two million rows, no tuning on any side; one random split each so far, two on Higgs and covertype. On point accuracy Ashlar's default is within 0.2% of the best of LightGBM, XGBoost, and CatBoost on airline delay, song release years, and Zurich transport, ahead on Higgs (accuracy 0.763 against 0.753) and covertype (0.967 against 0.963), and behind on click prediction, airlines, and taxi tips, where its RMSE is 13% higher. Asked for a distribution, `task="mixture"` scores a CRPS 20% below the best tree distribution's on airline delay, 8% below on release years, and 7% below on Zurich transport; on taxi tips, a spike at $0 and a long tail, CatBoost's per-row normal is 28% ahead. Ashlar trains in minutes on a laptop CPU, where LightGBM takes seconds to minutes.

<details>
<summary>Both tables: point accuracy and distributions, ± one standard deviation where a table has more than one split</summary>

| Table | Rows | Target, metric | Ashlar | Best tree |
|---|---:|---|---:|---:|
| Airline delay | 2,000,000 | delay in minutes, RMSE ↓ | **29.04** | 29.05 (CatBoost) |
| Year (MSD) | 515,345 | release year, RMSE ↓ | **8.698** | 8.710 (XGBoost) |
| Zurich transport | 2,000,000 | delay, RMSE ↓ | 3.052 | **3.047** (LightGBM) |
| NYC green taxi tips | 581,835 | tip in $, RMSE ↓ | 1.977 | **1.752** (LightGBM) |
| Higgs | 2,000,000 | signal, accuracy ↑ | **0.763** ± 0.0005 | 0.753 ± 0.0009 (XGBoost) |
| Covertype | 581,012 | 7 forest types, accuracy ↑ | **0.967** ± 0.0003 | 0.963 ± 0.0005 (LightGBM) |
| Click prediction | 1,997,410 | click, accuracy ↑ | 0.842 | **0.844** (LightGBM) |
| Airlines | 539,383 | delayed, accuracy ↑ | 0.660 | **0.667** (CatBoost) |

| Table | Score | Ashlar | Best tree distribution |
|---|---|---:|---:|
| Airline delay, delay in minutes | CRPS | **9.17** | 11.46 (CatBoost RMSEWithUncertainty) |
|  | NLL | **2.66** | 4.72 (CatBoost RMSEWithUncertainty) |
| Year (MSD), release year | CRPS | **4.05** | 4.41 (CatBoost RMSEWithUncertainty) |
|  | NLL | **3.08** | 3.42 (CatBoost RMSEWithUncertainty) |
| Zurich transport, delay | CRPS | **1.62** | 1.74 (CatBoost RMSEWithUncertainty) |
|  | NLL | **2.00** | 2.53 (CatBoost RMSEWithUncertainty) |
| NYC green taxi tips, tip in $ | CRPS | 0.59 | **0.46** (CatBoost RMSEWithUncertainty) |
|  | NLL | **−1.97** | 0.70 (CatBoost RMSEWithUncertainty) |
| Higgs | Log loss | **0.482** ± 0.0006 | 0.496 ± 0.0007 (XGBoost) |
| Covertype | Log loss | **0.094** ± 0.0003 | 0.114 ± 0.002 (LightGBM) |
| Click prediction | Log loss | 0.394 | **0.385** (XGBoost) |
| Airlines | Log loss | 0.614 | **0.606** (CatBoost) |

Ashlar is `ASRRegressor()` or `ASRClassifier()` with no arguments in the first table; in the second it is `ASRRegressor(task="mixture")` on the regression tables and the default classifier's probabilities on the others. The best tree distribution is the lowest of LightGBM, XGBoost, and CatBoost with a normal of one width, and CatBoost `RMSEWithUncertainty`, which fits a mean and a variance per row. CRPS is in the target's units, NLL and log loss are per row, and all three are lower-is-better.

</details>

On five smaller tables the defaults win two (Ames, abalone) and trail LightGBM on the rest, where `tabm_piecewise()` closes most of the gap. [docs/benchmarks.md](docs/benchmarks.md#large-tables) has every number.

## What it can do

Each capability below is a keyword argument, a label, or a method on the same model, and each is shown against the tool you would otherwise use: a gradient-boosted tree, a mixed model, R's `gamlss`, survival AFT, or DeepAR.

<table>
  <tr>
    <td align="center" width="33%"><a href="#a-shape-for-every-target"><img src="docs/images/readme/thumbs/shapes.png" alt="A shape for every target"></a><br><a href="#a-shape-for-every-target"><b>A shape for every target</b></a><br><sub>42% of claims sit at four amounts: 33% predicted, CatBoost 3%</sub></td>
    <td align="center" width="33%"><a href="#lifetimes-including-the-ones-still-running"><img src="docs/images/readme/thumbs/survival.png" alt="Lifetimes, including the ones still running"></a><br><a href="#lifetimes-including-the-ones-still-running"><b>Lifetimes, still running</b></a><br><sub>4 of 5 splits calibrated, against 0 of 5 for tuned AFT</sub></td>
    <td align="center" width="33%"><a href="#labels-that-say-what-was-observed"><img src="docs/images/readme/thumbs/observed.png" alt="Labels that say what was observed"></a><br><a href="#labels-that-say-what-was-observed"><b>Labels that say what was observed</b></a><br><sub>All four shelves empty: 11.4% predicted, 11.1% seen, 0.7% from separate fits (simulated)</sub></td>
  </tr>
  <tr>
    <td align="center" width="33%"><a href="#several-outcomes-one-model"><img src="docs/images/readme/thumbs/outcomes.png" alt="Several outcomes, one model"></a><br><a href="#several-outcomes-one-model"><b>Several outcomes, one model</b></a><br><sub>Bikes out per hour: 81% inside an 80% band, against LightGBM's 68%</sub></td>
    <td align="center" width="33%"><a href="#small-groups-borrow-from-large-ones"><img src="docs/images/readme/thumbs/pooling.png" alt="Small groups borrow from large ones"></a><br><a href="#small-groups-borrow-from-large-ones"><b>Small groups borrow from large ones</b></a><br><sub>CRPS 0.096 at 1 to 5 rows per store, against GPBoost's 0.125</sub></td>
    <td align="center" width="33%"><a href="#a-model-you-can-read"><img src="docs/images/readme/thumbs/additive.png" alt="A model you can read"></a><br><a href="#a-model-you-can-read"><b>A model you can read</b></a><br><sub>gamlss's shapes in 10 seconds a fit, against 4 to 45 minutes</sub></td>
  </tr>
  <tr>
    <td align="center" width="33%"><a href="#intervals-that-hold"><img src="docs/images/readme/thumbs/intervals.png" alt="Intervals that hold"></a><br><a href="#intervals-that-hold"><b>Intervals that hold</b></a><br><sub>One calibration keeps 21 of 24 hours on target; CQR, 10</sub></td>
    <td align="center" width="33%"><a href="#act-on-the-distribution"><img src="docs/images/readme/thumbs/act.png" alt="Act on the distribution"></a><br><a href="#act-on-the-distribution"><b>Act on the distribution</b></a><br><sub>One fit, any cost ratio; LightGBM refits cost up to 23% more</sub></td>
    <td align="center" width="33%"><a href="#forecasting-work-in-progress"><img src="docs/images/readme/thumbs/forecast.png" alt="Forecasting, work in progress"></a><br><a href="#forecasting-work-in-progress"><b>Forecasting, work in progress</b></a><br><sub>12-day total CRPS 2,623, against DeepAR's 3,881</sub></td>
  </tr>
</table>

## A shape for every target

How spread out is this count, which amounts do claims settle at, and can a value go past its cap?

![Three tiles of held-out data as grey bars, with Ashlar's head in teal and the best tree distribution in dashed clay. A count: bike rentals in the 256 hours LightGBM puts at 350 to 450, where the negative binomial head follows the bars, LightGBM's Poisson is far too narrow, and a negative binomial with one dispersion around it is too wide. A spike: freMTPL2 claim costs on a log axis, 42% of claims in one narrow bar at four fixed amounts; the mixture rises to it and CatBoost's normal stays flat. A ceiling: California home values in block groups earning $70k or more, 41% of them at the $500k cap; the mixture stacks its mass at the cap while LightGBM's normal spills past it.](docs/images/readme/shapes.png)

A tree gives you a mean, a normal around it, or a few quantiles; in Ashlar the head is the shape, picked with one keyword. On hourly bike rentals the `negative_binomial` head's 80% interval holds 80% of hours at 71 rentals wide, against 86% at 104 for LightGBM's Poisson with one fitted dispersion, and 54% for the Poisson alone. On freMTPL2 motor claims, 41.5% of which sit at four fixed amounts, the `mixture` puts 32.9% of its mass there against 3.2% for CatBoost's `RMSEWithUncertainty`; on California home values, in block groups earning $70,000 or more, it puts 0.6% past the $500,001 cap against 18.6% for LightGBM's normal. The trees hold their own elsewhere: a LightGBM classifier and a gamma regressor composed by hand tie the `hurdle` head on taxi tips, and on the $52 airport flat fare CatBoost's per-row normal has the best CRPS.

```python
import numpy as np
from ashlar import ASRRegressor

counts = ASRRegressor(task="negative_binomial").fit(X, rentals)  # a dispersion per row
costs = ASRRegressor(task="mixture").fit(X, np.log(cost))        # mass at fixed amounts
tips = ASRRegressor(task="hurdle", head_kwargs={"severity": "gamma"}).fit(X, tip)
p_no_tip = tips.predict_dist(X_new).cdf(0.0)                     # P(exactly $0) per row
```

[Heads and the shapes they fit](docs/heads.md) · [Counts against LightGBM](docs/benchmarks.md#against-a-negative-binomial-around-lightgbms-mean) · [Spikes, caps, and skew](docs/benchmarks.md#spikes-caps-and-skew) · [The trees' hurdle](docs/benchmarks.md#against-lightgbms-two-model-hurdle)

## Lifetimes, including the ones still running

How long until it happens, for this patient, and will it happen at all?

![Survival curves of five SUPPORT test patients from Ashlar's mixture head, each its own shape and some crossing, beside tuned XGBoost AFT's curves, all one shape shifted along time; on the right, the D-calibration bars of one split, the mixture's near 10% in every bin and AFT's not](docs/images/readme/survival.png)

A patient alive at the last visit is not a missing value. `observed.survival` scores them by the chance of living past it, and the `mixture` head gives each patient a curve of its own shape. On SUPPORT, 8,873 seriously ill patients in five random splits, it passes D-calibration on 4 of 5 splits, against 0 of 5 for XGBoost's AFT model tuned on a 42-fit grid. Its NLL of the log time is 1.733 against 1.874, and 1.061 against 1.142 on METABRIC; on GBSG the two tie. Ranking ties too, at a concordance of 0.623 each. With `head_kwargs={"cure": True}` the head also learns who never has the event: on simulated data where 39.7% never do, it predicts 39.8%.

```python
import numpy as np
from ashlar import ASRRegressor, observed

y = observed.survival(np.log(days), died)                 # False: alive when last seen
model = ASRRegressor(task="mixture").fit(X, y)
p_year = model.predict_exceedance(X_new, np.log(365.25))  # P(alive at one year) per row
```

[Survival on three clinical tables](docs/benchmarks.md#survival-on-three-clinical-tables) · [Survival times](docs/observed.md#survival-times) · [Times that may never come](docs/observed.md#times-that-may-never-come)

## Labels that say what was observed

What would we have sold if the shelf had not emptied?

![One FreshRetailNet-50K product over its 20 test days: sales bars that stop at the stock on sold-out days, with the demand they hid above them; 10% to 90% bands of demand from Ashlar and from XGBoost AFT, both fitted to the censored sales, and LightGBM's 90% quantile fitted to sales. Beside it, each fit's order cost above the oracle's over three seeds: AFT 4.7% below, Ashlar 4.0% above, the fits that read sold-out days as demand 12% to 17% above](docs/images/readme/sold-out.png)

A sold-out day recorded the stock, not the demand. On FreshRetailNet-50K, with a simulated stock, reading those days as demand or dropping them makes orders cost 12% to 17% more than a fit to the true demand, whichever model does it. Read as censored, Ashlar's orders cost 4.0% more, and XGBoost's AFT model, the one tree that reads censoring, is ahead at 4.7% less.

![Nine label builders of ashlar.observed drawn on one density: what each label says of y, and the part of the density the head trains on](docs/images/readme/label-grammar.png)

AFT reads censoring and intervals, with three error families. No tree has a likelihood for a noisy label, a total over several rows, or a joint label with a cell per column. From meter readings 2 to 12 days apart, Ashlar learns the daily load nearly as well as from the days, a CRPS of 19.0 against 18.0 MWh, though weekly totals cannot place it within the week. For four products that sell out together (simulated), it expects all four empty on 11.4% of days, against 11.1% seen and 0.7% from separate fits. On drives with delayed entry, AFT fitted the usual way won.

```python
from ashlar import ASRRegressor, observed

y = observed.censored(sales, sold_out)          # a sold-out day says demand >= sales
model = ASRRegressor(task="mixture").fit(X, y)
order = model.predict_quantiles(X_new, [0.75])  # a quantile of demand, not of sales

kept = observed.truncated(y, lower=1)           # the table keeps only days with a sale
loads = observed.total(reading, period)         # daily loads from multi-day readings
```

[Sold-out shelves on FreshRetailNet-50K](docs/benchmarks.md#sold-out-shelves-on-freshretailnet-50k) · [Daily loads from meter readings](docs/benchmarks.md#daily-loads-from-meter-readings) · [Several products that sell out together](docs/benchmarks.md#several-products-that-sell-out-together) · [Labels that say what was observed](docs/observed.md)

## Several outcomes, one model

How many claims will this policy make, and what will each one cost, given how many there are?

![Left: one French motor policy as task="chain" draws it, its chance of 0 to 4 claims as bars and, beside each count of one or more, the cost per claim given that count, its mean rising from €1,690 at one claim to €1,996 at four. Right: the share of 135,599 held-out policies whose total cost for the year is above t, from €1 to €1M on log axes: the chain and LightGBM Poisson × gamma both follow what happened, and two heads fitted apart sit far below it](docs/images/readme/claims-chain.png)

`task="chain"` fits one head after another, each reading the outcomes before it, as one model with one exact likelihood. On French motor insurance, 677,991 policies, a negative binomial claim count and then a hurdle cost per claim that reads the count tie LightGBM Poisson × gamma, the two models a gradient-boosting actuary would compose by hand, on the year's total cost: a CRPS of 92.5 against 92.7 over three splits, with LightGBM's total a little better calibrated. The chain's joint likelihood is better by 0.011 nats per policy, and it comes from one model. Fitted apart, the same two heads cannot know that a cost is zero exactly when the count is, and they give a policy a claim 0.24% of the time against 3.66% observed. Neither is right everywhere: both put too much weight above €5,000, LightGBM more, and the chain too little just above €1,000, where a third of claims sit at three fixed amounts.

```python
from ashlar import ASRRegressor

model = ASRRegressor(task="chain",
                     head_kwargs={"columns": ["negative_binomial", "hurdle"]})
model.fit(X, Y)                      # Y: claim count, then cost per claim (0 if none)
draws = model.sample(X_new, n=1000)  # (1000, rows, 2): a count, then a cost given it

def total(y):                        # any function of the outcomes
    return y[..., 0] * y[..., 1]

model.calibrate(X_cal, Y_cal, target=total)
model.predict_exceedance(X_new, 5000, target=total)  # P(the year's total > €5,000)
```

![Two held-out hours of Capital Bikeshare rentals, a Tuesday 08:00 commute in clay and a Sunday 14:00 in teal, stacked as isometric blocks on one floor of casual against registered riders, as the logo is, with an ink dot where each hour landed. Beside them, the 80% interval of the hour's total holds 81% of held-out hours for the joint model, 81% with one correlation fitted after the fact, 78% for two heads fitted apart, 68% for LightGBM quantiles of the total, and 79% for LightGBM's per-group quantiles summed, at width 90 against 79](docs/images/readme/joint-blocks.png)

Outcomes that are all numbers get a joint head: `task="multivariate_gaussian"`, `"multivariate_mixture"`, or `"multivariate_flow"`. Casual and registered riders share the shocks the weather and the calendar miss, so their errors do not cancel in the total: one joint fit's 80% interval for the bikes out in an hour holds 81% of held-out hours over three splits, against 78% for two heads fitted apart and 68% for LightGBM quantile regression of the total. One correlation fitted after the fact also holds 81%; the gain comes from modelling the dependence at all.

[Outcomes of different kinds, chained](docs/distributions.md#outcomes-of-different-kinds-chained) · [Several targets at once](docs/distributions.md#several-targets-at-once) · [Chained claim counts and costs, the benchmark](docs/benchmarks.md#chained-claim-counts-and-costs) · [Casual and registered riders, the benchmark](docs/benchmarks.md#casual-and-registered-riders)

## Small groups borrow from large ones

What does a store we opened yesterday look like, and how fast does it become itself?

![Left: every seen Rossmann store's effect on log sales against its training rows, and one new store's offset narrowing from the population's interval onto its own level as the model reads its first 1 to 35 days. Right: CRPS by rows per store for Ashlar, GPBoost, a linear mixed model and an ordinary embedding, Ashlar's the lowest in every bucket](docs/images/readme/store-pooling.png)

`random_effects` gives every store an offset drawn from a spread the model learns, so a store with three days of sales sits between its own mean and the average store, as far towards its own as three days warrant. On Rossmann's 1,115 stores, over three splits, the CRPS on stores with 1 to 5 training rows is 0.096 against GPBoost's 0.125. A store held out of training gets the population's interval, which holds 75% of its days against 36% for an ordinary embedding, and `with_context` reads its first days without refitting; after five, its interval holds 70%, short of 80%. GPBoost's and the mixed model's intervals hold nearer 80%, at a CRPS 29% to 101% higher on the stores they trained on. Nesting stores within their type added nothing.

```python
import numpy as np
from ashlar import ASRRegressor

model = ASRRegressor(task="gaussian",
                     random_effects={"store": ["intercept", "embedding"]})
model.fit(X, np.log(sales))                # X has a "store" column
model.random_effects_["store"]["tau"]      # the spread of stores around the average one
opened = model.with_context(X_first_days, np.log(first_days_sales))  # no refit
opened.predict_interval(X_next_days, 0.8)  # the new store's own 80% interval
```

[Entity effects, the benchmark](docs/benchmarks.md#entity-effects) · [Entity effects, the guide](docs/effects.md) · [New groups from a few rows](docs/effects.md#new-groups-from-a-few-rows)

## A model you can read

Which hours make rentals unpredictable, not just fewer?

![Ashlar's additive negative binomial on hourly bike rentals beside R's gamlss, term by term: the hour on working days and on other days, temperature, and humidity, on the log mean (top) and the log dispersion (bottom), the two tools' curves on top of each other; on the right, seconds per fit on a log axis and the test CRPS of each](docs/images/readme/additive-gamlss.png)

`backbone="additive"` makes each parameter of the distribution a sum of one shape per column, plus the pairs you name, so the fitted model is a set of curves you can read. On hourly bike rentals, Ashlar's additive negative binomial draws the shapes R's `gamlss` draws with penalized splines: over the training hours its log-mean curves differ from `gamlss`'s by 0.01 to 0.02 on average, and its dispersion's, harder to learn, by 0.06 to 0.18. Humid hours and cold ones raise the dispersion: rain makes rentals less predictable as well as fewer. Over three splits the CRPS is 23.46 against `gamlss`'s 23.40, and a fit takes 10 seconds where `gamlss` took 4 to 45 minutes. Readable shapes cost accuracy: the residual network, which learns every interaction, scores 16.48. On abalone, `gamlss`'s normal beats the additive `gaussian`, CRPS 1.115 against 1.156.

```python
from ashlar import ASRRegressor

model = ASRRegressor(task="negative_binomial", backbone="additive",
                     backbone_kwargs={"pairs": [["hr", "workingday"]]}).fit(X, rentals)
shapes = model.shape_functions(X)
shapes.term("hum", "dispersion")    # humidity's part of each row's dispersion
shapes.importance("dispersion")     # which columns move the spread, largest first
```

[Additive models against R's gamlss](docs/benchmarks.md#additive-models-against-rs-gamlss) · [Additive models, the guide](docs/additive.md)

## Intervals that hold

Will an 80% interval hold 80% at 3 a.m. as well as at 5 p.m., and when a cell is blank?

![Coverage of the 80% interval in each hour of the day on hourly bike rentals, raw, after one calibration, and after a table per hour: after one calibration Ashlar's dots stay inside 75% to 85% in 21 of 24 hours while LightGBM CQR's run from 71% to 98%; on the right, Ashlar's calibrated interval is narrower than CQR's on three of four tables](docs/images/readme/intervals.png)

Each interval comes from the row's own distribution, so one `calibrate` on held-out rows also holds hour by hour. On hourly bike rentals it leaves 3 of 24 hours outside 75% to 85%, against 14 for LightGBM's quantile models with the same conformal step (CQR), too wide at night and too narrow by day. A margin per hour brings CQR to 2, but 26% wider (92.3 against 73.5 rentals). At equal coverage Ashlar's interval is narrower than CQR's on 3 of 4 tables, and 2% wider on abalone.

![Coverage of the 80% interval as 0% to 50% of the numeric cells go blank at test time, on California housing and bike rentals: with_uncertain_inputs(32) holds 81% to 83%, the median fill falls to 41% and 48%, LightGBM given NaN to 67% and 36%, and LightGBM trained with blanks holds 80%; on the right, one California row's distribution widening as its income, latitude, and longitude go blank](docs/images/readme/intervals-missing.png)

A blank cell is drawn, not filled. With 30% of California's cells blank, `with_uncertain_inputs(32)` still covers 83%, against 50.5% for the median fill and 68% for LightGBM given NaN. LightGBM trained and calibrated with the same share blank also holds 80% and is narrower on California ($141k against $178k), but it needs that share in advance and a fit for each.

```python
from ashlar import ASRRegressor

model = ASRRegressor(task="mixture").fit(X_train, y_train)
model.calibrate(X_cal, y_cal)                            # groups="hr": a table per hour
interval = model.predict_interval(X_test, coverage=0.8)  # low and high per row
blanks = model.with_uncertain_inputs(32)                 # blank cells drawn, not filled
interval = blanks.predict_interval(X_with_blanks, coverage=0.8)
```

[Per hour, beside LightGBM CQR](docs/benchmarks.md#per-hour-beside-lightgbm-cqr) · [Calibrated intervals](docs/benchmarks.md#calibrated-intervals) · [Masked cells beside LightGBM](docs/benchmarks.md#masked-cells-beside-lightgbm) · [Calibration](docs/heads.md#calibrated-intervals) · [Missing and noisy inputs](docs/distributions.md#missing-and-noisy-inputs)

## Act on the distribution

How many should we order at each cost ratio, and which features make this hour uncertain?

![Left: LightGBM's newsvendor cost above Ashlar's at eight cost ratios on hourly bike rentals, refitted at each ratio, level at 1:1 and 23% to 28% higher at 19:1, while Ashlar reads every ratio off one fit; right: circles of mean absolute attribution per feature for the mean, the 80% width, and P(rentals > 500), Ashlar's explain beside SHAP on LightGBM, where working day takes 17% of the width against 11% of the mean](docs/images/readme/act.png)

`predict_order(X, under, over)` reads the cost-optimal order off the fitted distribution, so one fit serves every cost ratio. On hourly bike rentals, LightGBM's quantile regression refitted at each of eight ratios is level at 1:1 and costs up to 23% more on held-out hours the further into the tail. `explain` attributes any function of the distribution to the features: working day takes 17% of the attribution of the 80% interval's width, against 11% of the mean's. SHAP on LightGBM needs two quantile models for the width, which crossed on 0.35% of hours, and a classifier for each threshold.

```python
from ashlar import ASRRegressor

model = ASRRegressor(task="mixture").fit(X_train, y_train)
order = model.predict_order(X_test, under=9, over=1)  # 9:1 costs: the 0.9 quantile
why = model.explain(X_test, "interval_width", coverage=0.8, steps=256)
why.importance()                                      # mean |attribution| per feature
```

[Eight cost ratios from one fit](docs/benchmarks.md#hourly-bike-rentals-eight-cost-ratios-from-one-fit) · [Explanations beside SHAP](docs/benchmarks.md#explanations-beside-shap) · [Deciding from the distribution](docs/distributions.md#deciding-from-the-distribution) · [Explaining the distribution](docs/distributions.md#explaining-the-distribution)

## Forecasting, work in progress

This part is a work in progress: forecasts of the total over the coming days hold up against DeepAR, and the busiest day's probability is not calibrated yet.

![Forty of 400 sales paths for one Rossmann store over the next 12 days, from Ashlar's joint normal (each path keeps its level) and from the same days drawn independently (the paths cross); on the right, the 12-day total from each set of paths, the joint's 80% interval holding the realized total and the independent one's missing it; under the paths, the closed, promotion, and holiday days the model is told about](docs/images/readme/forecast-paths.png)

`horizon_frame` turns a long table of many series into one row per series and forecast origin, with the lags, the days ahead that are known, and the static columns, and a joint head forecasts the whole 12-day path at once. On Rossmann's daily sales, 300 stores and three seeds, the covariates cut the daily loss from 1,326 to 478. The joint normal's 80% interval for the 12-day total holds 76% of totals, where the same days drawn independently hold 51%, and its CRPS for the total is 2,623 against 3,881 for DeepAR. Chronos-2, zero-shot, is level with it day by day, 467.6 against 477.6, and draws no paths. The peak day is unsolved: DeepAR calibrates it best, Brier 0.192 against 0.206 for Ashlar's best head, and no model beats always saying 25%.

```python
import numpy as np
from ashlar import ASRRegressor
from ashlar.horizon import horizon_frame

frame = horizon_frame(sales, target="log_sales", lookback=28, horizon=12,
                      series="store", time="day",
                      known=["open", "promo", "holiday"], static=["store_type"])
model = ASRRegressor(task="multivariate_gaussian",
                     head_kwargs={"rank": 4}).fit(frame.X, frame.Y)
totals = np.expm1(model.sample(frame.X, n=400)).sum(axis=-1)  # 400 totals per row
```

[Forecast paths with covariates](docs/benchmarks.md#forecast-paths-with-covariates) · [Many series, covariates, and paths](docs/forecast.md#many-series-covariates-and-paths) · [The single-series forecasters](docs/forecast.md)

## Under the hood

A model is one set of keyword arguments in four slots: how the columns are encoded, the backbone that reads them, the head that answers, and the training schedule. The label is a fifth choice, made in `fit`: an exact `y`, or one that says what was observed. Every capability on this page is a value in one of them, so they combine. The path lit below is one model: an additive backbone under a negative binomial head, fitted to sales that a sold-out shelf capped.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/images/architecture-dark.svg">
    <img alt="Five courses of model choices: encode, backbone, head, label, and train, with one path lit." src="docs/images/architecture-light.svg" width="100%">
  </picture>
</p>

None of these are wrapper classes: every configuration is `ASRModel` with different arguments, so it is a dict you can save, version, or sweep. `backbone=` also takes your own Equinox module, and `task="custom"` your own head and objective. [docs/models.md](docs/models.md) is the tour of each slot.

**One distribution object for every head.** `model.predict_dist(X)` returns an `ashlar.distributions.Distribution` whatever the head: `log_prob`, `sample`, `mean`, and, for one target, exact `cdf` and `quantile`. It is a JAX pytree, so it goes through `jax.jit`, `jax.vmap`, and `jax.grad`. Encode the rows once and the gradient runs from the raw inputs, through imputation, scaling, and the network, to any function of the distribution:

```python
import jax

batch = model.encode(X_new)
def p_late(numeric):                  # P(delay > 60 min), summed over the rows
    dist = model.forward_dist(numeric, batch.categorical_onehot, batch.category_index,
                              batch.set_tokens, batch.set_mask)
    return (1.0 - dist.cdf(60.0)).sum()
jax.grad(p_late)(batch.numeric_raw)   # per row: how each raw input moves P(late)
```

**Save and load without pickle.** `model.save("models/delay")` writes a directory of JSON and Equinox weight files, and `ASRModel.load("models/delay")` brings it back as the class it was saved as, with identical predictions. A custom head or backbone is recorded by its module path and imported on load, so load only models you trust.

Also in the box, documented but not shown here: TabM, transformer, and set backbones, hyperparameter search, ordinal and nested-outcome classifiers, member (epistemic) uncertainty, and the series forecasters `ASRForecaster` and `ASRHistoryForecaster`.

## Documentation

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
