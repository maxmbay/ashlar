# Changelog

Every release of `ashlar-tabular` that changes something you can see. Versions follow [semantic versioning](https://semver.org); before 1.0, a minor version may change the API.

## Unreleased

### Added

- **One distribution object for every head.** `model.predict_dist(X)` returns an `ashlar.distributions.Distribution` in the units of `y` for every task that has a likelihood: the distribution heads, `normal`, the count heads, and, as a `Categorical`, classification, ordinal, and tree models. It has explicit `batch_shape` and `event_shape`, `log_prob`, `sample`, `mean`, `variance`, and for one target column `cdf` and `quantile`. It is an Equinox module, so it passes through `jax.jit`, `jax.vmap`, and `jax.grad`, and the draws of the normal, Student-t, and flow heads are reparameterized. `dist[rows]` slices it like an array. [docs/distributions.md](docs/distributions.md) has the detail.
- `ASRForecaster.predict_dist` and `ASRHistoryForecaster.predict_dist` return the same objects, one marginal per horizon step.

### Changed

- `sample`, `log_prob`, `predict_quantiles`, `predict_interval`, and `calibrate` now read `predict_dist`, so they agree by construction.
- `predict_quantiles` and `predict_interval` are exact instead of read from 1,000 draws: closed form for the gaussian, flow, and histogram heads, and bisection on the CDF of the member mixture otherwise. They no longer depend on `n_samples` or `seed`, which are kept for custom distributions without a quantile.
- `calibrate` computes each held-out row's PIT from the CDF instead of ranking it among draws, randomized over the jump for a count head.
- `predict_uncertainty` reads each member's variance in closed form instead of from draws; `n_samples` and `seed` are unused.
- `predict_dist` on a `normal` model returns the new `Mixture` (or a `Normal` for `policy="mean_parameters"`) instead of `NormalMixture` and `MeanParameterNormal`. Those two classes still import. Its `mean()` and `variance()` are now `(rows,)` for one target column, not `(rows, 1)`.
- `log_prob` and `sample` work on classifiers too: a log probability of the observed label, and drawn labels.

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
