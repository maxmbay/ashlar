# Changelog

Every release of `ashlar-tabular` that changes something you can see. Versions follow [semantic versioning](https://semver.org); before 1.0, a minor version may change the API.

## Unreleased

### Fixed

- **The `flow` head gives each row its own spread and tails.** It was a spline on a fixed range with identity tails, so the spread past that range, and so every far quantile and tail probability, was the same for every row: on a lognormal target whose spread more than doubles across rows, it put 0.13% of every row's mass above a high threshold, where the truth is 1.8% on average and 4.4% in the riskiest tenth of rows. The head now ends in a shift and a scale per row, `y = shift + scale * spline(z)`, and gives 1.8% and 4.3%. Its output is two values wider per step. Models and forecasters saved with the old head load and predict as they did.

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
