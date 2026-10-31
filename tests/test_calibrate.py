"""Split-conformal calibration of the distribution heads."""

import numpy as np
import pytest

from ashlar import ASRModel
from ashlar.calibrate import conformal_quantile, interval_levels, pit_calibration, pit_values, recalibrated_levels
from ashlar.errors import AshlarError


def _heteroscedastic(n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = X[:, 0] + (0.3 + np.abs(X[:, 1])) * rng.normal(size=n)
    return X, y


def _small(task):
    return ASRModel(
        task=task,
        backbone="tabm",
        backbone_kwargs={"n_blocks": 1, "d_block": 16, "k": 4},
        max_epochs=3,
    )


def test_conformal_quantile_is_the_finite_sample_rank():
    scores = np.arange(1, 10, dtype=float)
    assert conformal_quantile(scores, 0.8) == 8.0
    assert conformal_quantile(scores, 0.95) == np.inf


def test_pit_is_uniform_when_the_draws_are_right_and_randomized_on_ties():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(2000, 3000))
    pit = pit_values(draws, rng.normal(size=3000))
    assert abs(np.mean(pit < 0.2) - 0.2) < 0.03
    counts = rng.poisson(1.0, size=(2000, 3000)).astype(float)
    pit = pit_values(counts, rng.poisson(1.0, size=3000))
    assert abs(np.mean(pit < 0.5) - 0.5) < 0.03


def test_overconfident_pit_widens_the_interval():
    rng = np.random.default_rng(0)
    # y twice as spread as the model believes: PIT piles up at both ends.
    from scipy.stats import norm

    calibration = pit_calibration(norm.cdf(2.0 * rng.normal(size=500)))
    low, high = interval_levels(calibration, 0.8)
    assert low < 0.03 and high > 0.97
    assert recalibrated_levels(calibration, [0.5])[0] == pytest.approx(0.5, abs=0.05)


@pytest.mark.slow
@pytest.mark.parametrize("task", ["gaussian", "mixture", "histogram"])
def test_calibrated_interval_covers_on_new_rows(task, tmp_path):
    X, y = _heteroscedastic(900, 0)
    X_cal, y_cal = _heteroscedastic(600, 1)
    X_new, y_new = _heteroscedastic(2000, 2)
    model = _small(task).fit(X, y).calibrate(X_cal, y_cal, n_samples=400)
    interval = model.predict_interval(X_new, 0.8, n_samples=400)
    covered = np.mean((y_new >= interval[:, 0]) & (y_new <= interval[:, 1]))
    assert 0.75 < covered < 0.86
    raw = model.predict_interval(X_new, 0.8, n_samples=400, calibrated=False)
    assert raw.shape == interval.shape
    model.save(tmp_path / "model")
    restored = ASRModel.load(tmp_path / "model")
    assert restored.calibration_ == model.calibration_
    assert np.allclose(restored.predict_interval(X_new, 0.8, n_samples=400), interval, atol=1e-4)


def test_quantile_head_takes_the_cqr_margin():
    X, y = _heteroscedastic(900, 0)
    X_cal, y_cal = _heteroscedastic(600, 1)
    X_new, y_new = _heteroscedastic(2000, 2)
    model = _small("quantile").fit(X, y)
    raw = model.predict_quantiles(X_new)
    model.calibrate(X_cal, y_cal)
    calibrated = model.predict_quantiles(X_new)
    assert np.allclose(calibrated[:, 1], raw[:, 1])
    assert np.allclose(calibrated[:, 2] - raw[:, 2], raw[:, 0] - calibrated[:, 0])
    interval = model.predict_interval(X_new)
    covered = np.mean((y_new >= interval[:, 0]) & (y_new <= interval[:, 1]))
    assert 0.75 < covered < 0.86
    with pytest.raises(AshlarError):
        model.predict_interval(X_new, 0.9)


@pytest.mark.slow
def test_fit_clears_the_calibration_and_misuse_raises():
    X, y = _heteroscedastic(400, 0)
    model = _small("gaussian").fit(X, y)
    with pytest.raises(AshlarError):
        model.predict_interval(X, calibrated=True)
    with pytest.raises(AshlarError):
        model.calibrate(X[:5], y[:5])
    model.calibrate(X[:100], y[:100], n_samples=200)
    assert model.calibration_ is not None
    model.fit(X, y)
    assert model.calibration_ is None
    with pytest.raises(AshlarError):
        ASRModel(max_epochs=2).fit(X, y).calibrate(X, y)
