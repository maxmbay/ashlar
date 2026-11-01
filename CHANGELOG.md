# Changelog

Every release of `ashlar-tabular` that changes something you can see. Versions follow [semantic versioning](https://semver.org); before 1.0, a minor version may change the API.

## Unreleased

### Added

- **Calibrated joint outputs.** `calibrate` works on `multivariate_gaussian`, `multivariate_mixture`, and `multivariate_flow`. It keeps a table of probability integral transforms for each column and for each function of the columns it is given with `target`: always their sum, and also a weight per column (that weighted sum) or a function defined with `def`. `predict_interval`, `predict_quantiles`, and `predict_exceedance` take the same `target`, so `predict_interval(X, 0.8, target="sum")` is a calibrated interval for the total. A function's table is saved under its name; pass the function again after `load`. Joint calibration needs rows whose cells are all exact. [docs/distributions.md](docs/distributions.md#calibrating-the-columns-and-the-total) has the detail. ([#31](https://github.com/maxmbay/ashlar/issues/31))
- `predict_exceedance(X, threshold)`: P(y > threshold) per row for every head with a CDF, one probability per column on a joint head, or for a function of the columns with `target`. After `calibrate` the model's own `P(y ≤ threshold)` is replaced by the share of calibration rows whose transform was at most that much.
- `Distribution.project(weights)`: the exact distribution of a weighted sum of a joint outcome's columns, a normal under `LowRankMultivariateNormal` and `Independent` normals, and a mixture of normals under `Mixture` and `AffineTransformed` of them. The flow has no closed form and raises `NotImplementedError`.
- `examples/joint_demand.py calibrated`: the joint demand benchmark with the last 15% of each origin's training weeks held out to calibrate the total and each site.
- **Joint heads train on the energy and variogram scores.** `objective="energy"` trains a joint head on the energy score, and a dict blends proper scores with the log density, such as `{"nll": 1.0, "energy": 1.0}` or `{"nll": 1.0, "variogram": 1.0}`. The variogram score is refused on its own, since it cannot place the columns. Both are estimated without bias from 16 fresh reparameterized draws per row and step, and a mixture's from every component, weighted by the component probabilities. New `ashlar.objectives.energy_score`, `variogram_score`, `score_draws`, `score_weights`, and `make_joint_score_objective`. A callable objective with `stochastic = True` is passed a fresh `key=` on every training step. On the joint demand benchmark neither score did better than the log density, and the energy score alone made the total's interval narrower and less often right; `calibrate` is what fixes the total's coverage. [docs/distributions.md](docs/distributions.md#training-on-other-proper-scores) has the detail. ([#33](https://github.com/maxmbay/ashlar/issues/33))
- `examples/joint_demand.py scored`: the rank-2 normal and the flow trained on the energy score, alone and added to the log density, and the flow on the log density plus the variogram score, on the splits of the `calibrated` stage. Results in [docs/benchmarks.md](docs/benchmarks.md#training-on-other-proper-scores).

## 0.0.4 — 2026-10-31

### Added

- **Labels that say what was observed.** `ashlar.observed` builds a `y` that records what was seen about each value rather than the value itself: `exact`, `interval`, `censored` (right or left), `survival` (with delayed entry), `rounded`, `truncated`, `noisy`, and group totals (`total`, `grouped`). They compose, and `concatenate` mixes kinds in one table. Pass one as `y` to `fit`, `log_prob`, `score`, or `calibrate`, and the head trains on the probability it gives to the observation; `predict_dist` still describes the value itself. Every univariate head with a CDF takes them: `gaussian`, `mixture`, `flow`, `histogram`, `student_t`, the count heads, `hurdle`, and `normal`. Noisy labels need `gaussian`, `mixture`, or `normal`, and totals `gaussian`, `normal`, or `poisson`. A label is a NumPy structured array, so scikit-learn splits it. [docs/observed.md](docs/observed.md) has the detail. ([#29](https://github.com/maxmbay/ashlar/pull/29))
- `calibrate` takes interval and censored rows, drawing each row's PIT uniformly over what its label allows.
- Group totals train each group whole inside one minibatch and keep it on one side of an internal validation split. A split that cuts a group raises.
- `Distribution.log_cdf`, `log_survival`, `log_cdf_left`, `log_survival_left`, `log_interval`, and `cdf_interval` on every univariate distribution, computed in log space so they stay finite far into the tails, and `convolve_normal` on the normal, mixture, and affine distributions.
- `examples/benchmark_observed.py`: demand fitted from sales when a third of days sell out, and daily values fitted from weekly totals, each against the usual fits and an oracle. Its `figure` stage draws the README figure `docs/images/sold-out-demand.png`. Its results are in [docs/benchmarks.md](docs/benchmarks.md#labels-that-say-what-was-observed).
- **Joint labels.** The joint heads take a label with a cell per target column, shape `(rows, columns)`, each cell exact, an interval, censored, or rounded: `exact`, `interval`, `censored`, `survival` (without delayed entry), and `rounded` accept tables, and `observed.stack` puts one-column labels side by side. A row scores the joint density of its exact cells times the probability of its coarse cells given them. `fit`, `log_prob`, and `score` take them; truncation, noise, and totals stay one-column. ([#30](https://github.com/maxmbay/ashlar/pull/30))
- `Distribution.log_box(lower, upper, points)` on `LowRankMultivariateNormal`, `AutoregressiveFlow` (one block), `Mixture`, `AffineTransformed`, and `Independent`. The joint normal runs the GHK simulator on its low-rank factors, one Kalman update per column, exact columns first and then the coarse ones from the least likely, so a row with at most one coarse column is closed form; the flow runs it in its own column order. Several coarse columns are integrated at quasi-random `box_points`, 32 per row in training and 256 when scoring. Checked against SciPy's multivariate normal CDF and against quadrature of the flow's density.
- **Joint mixture head.** `task="multivariate_mixture"` predicts a mixture of joint normals, five by default (`head_kwargs={"components": 5, "rank": 2}`), each with its own weight, location, and low-rank covariance per row, for dependence that comes in regimes. New `joint_mixture_distribution`. ([#30](https://github.com/maxmbay/ashlar/pull/30))
- `examples/benchmark_observed.py joint`: four products whose shelves empty together, the three joint heads fitted to the true demand, to sales, and to the censored joint label, against a censored mixture per product. The censored joint normal comes within 0.013 nats of the fit to the true demand. The per-product fits are 1.36 nats behind it and expect every shelf to empty on 0.7% of days, where it happened on 11.1%; the censored joint fits expect 9.1% to 12.1%. Its `joint_figure` stage draws `docs/images/joint-sold-out.png`, and the results are in [docs/benchmarks.md](docs/benchmarks.md#several-products-that-sell-out-together).

### Fixed

- **The `flow` head gives each row its own spread and tails.** It was a spline on a fixed range with identity tails, so the spread past that range, and so every far quantile and tail probability, was the same for every row: on a lognormal target whose spread more than doubles across rows, it put 0.13% of every row's mass above a high threshold, where the truth is 1.8% on average and 4.4% in the riskiest tenth of rows. The head now ends in a shift and a scale per row, `y = shift + scale * spline(z)`, and gives 1.8% and 4.3%. Its output is two values wider per step. Models and forecasters saved with the old head load and predict as they did. ([#29](https://github.com/maxmbay/ashlar/pull/29))
- `jax.grad` through a Student-t's `cdf` or `quantile` in its degrees of freedom, or a negative binomial's `cdf` in its dispersion, raised, because JAX's `betainc` has no gradient in its shape arguments. Both now differentiate, by central differences in those arguments.

## 0.0.3 — 2026-10-31

### Added

- **One distribution object for every head.** `model.predict_dist(X)` returns an `ashlar.distributions.Distribution` in the units of `y` for every task that has a likelihood: the distribution heads, `normal`, the count heads, and, as a `Categorical`, classification, ordinal, and tree models. It has explicit `batch_shape` and `event_shape`, `log_prob`, `sample`, `mean`, `variance`, and for one target column `cdf` and `quantile`. It is an Equinox module, so it passes through `jax.jit`, `jax.vmap`, and `jax.grad`, and the draws of the normal, Student-t, and flow heads are reparameterized. `dist[rows]` slices it like an array. [docs/distributions.md](docs/distributions.md) has the detail. ([#23](https://github.com/maxmbay/ashlar/pull/23))
- `ASRForecaster.predict_dist` and `ASRHistoryForecaster.predict_dist` return the same objects, one marginal per horizon step.
- **Joint outputs.** `task="multivariate_gaussian"` takes `y` with one column per outcome and predicts their joint normal, with a covariance `W Wᵀ + diag(D)` of learned rank (`head_kwargs={"rank": 2}` by default) that changes from row to row. It trains on the joint log density. `predict_dist` gives the joint distribution in the units of `y`; `marginals()` gives each column on its own, and `Independent(dist.marginals())` the same columns with the dependence dropped. `predict` returns every column's mean and `predict_quantiles` each column's quantiles, `(rows, columns, levels)`. ([#24](https://github.com/maxmbay/ashlar/pull/24))
- **Joint flow.** `task="multivariate_flow"` takes the same `y` as `multivariate_gaussian` and predicts a conditional masked autoregressive flow over the columns: each column's conditional is a spline whose knots, shift, and scale come from a masked network reading the columns before it and the row. The joint can be skewed and its dependence nonlinear. Exact `log_prob`, reparameterized draws, and Monte Carlo moments; `head_kwargs={"hidden": 64, "depth": 1}`. New `AutoregressiveFlow` and `Empirical` distributions. ([#26](https://github.com/maxmbay/ashlar/pull/26))
- **Hurdle head.** `task="hurdle"` predicts the probability that `y` is exactly zero and, otherwise, a `lognormal` (default) or `gamma` amount (`head_kwargs={"severity": "gamma"}`), on the raw target. New `LogNormal`, `Gamma`, and `Hurdle` distributions. Calibration and quantiles handle the atom at zero. ([#24](https://github.com/maxmbay/ashlar/pull/24))
- `Distribution.cdf_left(y)`, `P(Y < y)`, so the randomized PIT works for any distribution with an atom.
- `examples/joint_demand.py`: eight electricity clients forecast jointly a day ahead and scored at 25 rolling origins against the same marginals made independent, independent per-client heads, and independent heads with a fixed residual correlation. Its results are in the README and in [docs/benchmarks.md](docs/benchmarks.md#joint-demand-across-locations). ([#25](https://github.com/maxmbay/ashlar/pull/25))

### Changed

Upgrading from 0.0.2: quantiles, intervals, calibration, and member uncertainty will differ slightly because they are now computed exactly rather than from draws, and code that checks the type of `predict_dist` on a `normal` model, or indexes its `(rows, 1)` moments, needs updating.

- `sample`, `log_prob`, `predict_quantiles`, `predict_interval`, and `calibrate` now read `predict_dist`, so they agree by construction.
- `predict_quantiles` and `predict_interval` are exact instead of read from 1,000 draws: closed form for the gaussian, flow, and histogram heads, and bisection on the CDF of the member mixture otherwise. They no longer depend on `n_samples` or `seed`, which are kept for custom distributions without a quantile.
- `calibrate` computes each held-out row's PIT from the CDF instead of ranking it among draws, randomized over the jump for a count head.
- `predict_uncertainty` reads each member's variance in closed form instead of from draws; `n_samples` and `seed` are unused.
- `predict_dist` on a `normal` model returns the new `Mixture` (or a `Normal` for `policy="mean_parameters"`) instead of `NormalMixture` and `MeanParameterNormal`. Those two classes still import. Its `mean()` and `variance()` are now `(rows,)` for one target column, not `(rows, 1)`.
- `log_prob` and `sample` work on classifiers too: a log probability of the observed label, and drawn labels.
- The calibrated-interval benchmark was rerun with exact quantiles; the `mixture` column moved by at most 0.9 points.

## 0.0.2 — 2026-10-30

### Added

- **Calibrated intervals.** `model.calibrate(X_holdout, y_holdout)` fits split-conformal calibration on rows the model did not train on. After that, `model.predict_interval(X, coverage=0.8)` gives intervals that cover the requested share of new, exchangeable rows. The method depends on the head:
  - Density heads use distributional conformal prediction on a randomized PIT.
  - The `quantile` head uses conformalized quantile regression.
  - Calibration is saved with the model and cleared by `fit`.
  - `calibrated=False` reads the raw head. ([#13](https://github.com/maxmbay/ashlar/pull/13))
- **Student-t head.** `task="student_t"` adds a heavy-tailed head with learned degrees of freedom, above 2. ([#14](https://github.com/maxmbay/ashlar/pull/14))
- **Count heads.**
  - `task="poisson"` and `task="negative_binomial"` model non-negative integer targets on the raw count. They give whole-number draws and quantiles, plus a log mass you can compare across models.
  - `head_kwargs={"zero_inflated": True}` adds a zero-inflation component.
  - Targets that are not counts are rejected. The forecasters do not take these heads. ([#14](https://github.com/maxmbay/ashlar/pull/14))
- **Member uncertainty.** `model.predict_uncertainty(X)` splits predictive uncertainty into `epistemic` (disagreement between ensemble members), `aleatoric` (the noise the head predicts) and `total`.
  - Classifiers and ordinal models report entropies instead, with mutual information as the epistemic part.
  - It needs an ensemble backbone such as `backbone="tabm"`. A one-member backbone reports zero epistemic uncertainty. ([#15](https://github.com/maxmbay/ashlar/pull/15))
- **Benchmarks** for all three features. They are in `examples/benchmark_uncertainty.py`, with results in `docs/benchmarks.md` under Calibrated intervals, Count heads and Member uncertainty. ([#17](https://github.com/maxmbay/ashlar/pull/17))

### Changed

- After `calibrate`, `predict_quantiles` returns recalibrated levels by default. On a model that was never calibrated it behaves as before.

### Notes

- The benchmarks turned up two calibration bugs, and both were fixed before release:
  - Calibration could not move a head whose PIT fell on the grid of its draws.
  - Calibrated intervals undercovered by about 0.7 points, because they read interpolated quantiles while the PIT ranked against order statistics.
- The member-uncertainty benchmark has two splits for California and one each for diamonds and bike rentals. Read it as indicative.

## 0.0.1 — 2026-10-30

First release on PyPI.
