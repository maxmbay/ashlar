# Changelog

Every release of `ashlar-tabular` that changes something you can see. Versions follow [semantic versioning](https://semver.org); before 1.0, a minor version may change the API.

## Unreleased

Planned as 0.0.2.

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
