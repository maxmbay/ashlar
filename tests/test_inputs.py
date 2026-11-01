"""Missing and noisy inputs widen the predictive distribution; the input copula scores rows."""

import numpy as np
import pytest
from scipy import stats

from ashlar import ASRRegressor
from ashlar.errors import AshlarError
from ashlar.inputs import InputModel


def _correlated(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 0.9 * a + np.sqrt(1 - 0.81) * rng.normal(size=n)
    c = rng.exponential(size=n)
    return np.stack([a, b, c], axis=1)


def test_copula_recovers_the_correlation_and_conditionals():
    values = _correlated()
    model = InputModel.fit(values)
    assert model.correlation[0, 1] == pytest.approx(0.9 * (1 - 0.01), abs=0.03)
    assert abs(model.correlation[0, 2]) < 0.06
    rows = np.array([[np.nan, 1.5, 1.0], [np.nan, np.nan, np.nan], [0.1, 0.2, 0.3]])
    draws = model.draw(rows, 4000, np.random.default_rng(1))
    assert draws.shape == (4000, 3, 3)
    # Given b = 1.5, a is near 0.9 * 1.5 with sd about sqrt(1 - 0.81).
    assert np.mean(draws[:, 0, 0]) == pytest.approx(1.35, abs=0.15)
    assert np.std(draws[:, 0, 0]) < 0.6
    np.testing.assert_array_equal(draws[:, 0, 1:], np.broadcast_to(rows[0, 1:], (4000, 2)))
    np.testing.assert_array_equal(draws[:, 2], np.broadcast_to(rows[2], (4000, 3)))
    assert np.std(draws[:, 1, 2]) == pytest.approx(1.0, abs=0.15)
    assert np.all(draws[:, 1, 2] >= -0.1)
    noisy = model.draw(rows[2:], 4000, np.random.default_rng(2), noise=np.array([[0.0, 0.5, 0.0]]))
    assert np.std(noisy[:, 0, 1]) == pytest.approx(0.5, abs=0.05)
    assert np.all(noisy[:, 0, 0] == 0.1)


def test_log_density_matches_a_normal_and_flags_outliers():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(20000, 1))
    model = InputModel.fit(values)
    grid = np.linspace(-2, 2, 41)[:, None]
    error = model.log_density(grid) - stats.norm.logpdf(grid[:, 0])
    # Piecewise-constant between 129 empirical quantiles: unbiased, a little noisy.
    assert abs(error.mean()) < 0.05
    assert np.abs(error).max() < 0.3
    joint = InputModel.fit(_correlated())
    usual, odd = joint.log_density(np.array([[1.0, 0.9, 1.0], [1.0, -0.9, 1.0]]))
    assert usual > odd + 5
    assert np.isfinite(joint.log_density(np.array([[np.nan, 0.0, 1.0], [np.nan, np.nan, np.nan]]))).all()
    restored = InputModel.from_json(joint.to_json())
    np.testing.assert_allclose(restored.log_density(np.array([[1.0, 0.9, 1.0]])), joint.log_density(np.array([[1.0, 0.9, 1.0]])))


def test_uncertain_inputs_widen_intervals_and_save(tmp_path):
    values = _correlated(1500)
    rng = np.random.default_rng(0)
    y = 2 * values[:, 0] + values[:, 2] + 0.2 * rng.normal(size=1500)
    X = {"a": values[:, 0], "b": values[:, 1], "c": values[:, 2], "city": rng.choice(["x", "y"], 1500).astype(object)}
    model = ASRRegressor(task="gaussian", max_epochs=4, random_state=0).fit(X, y)
    rows = {key: value[:20].copy() for key, value in X.items()}
    rows["a"][:10] = np.nan
    plain = model.predict_interval(rows, 0.8)
    wide = model.with_uncertain_inputs(16).predict_interval(rows, 0.8)
    widths = (plain[:, 1] - plain[:, 0], wide[:, 1] - wide[:, 0])
    assert np.all(widths[1][:10] > widths[0][:10])
    np.testing.assert_allclose(widths[1][10:], widths[0][10:], rtol=1e-4)
    dist = model.predict_dist(rows, input_draws=8)
    assert dist.batch_shape == (20,)
    noisy = model.with_uncertain_inputs(16, noise={"c": 1.0}).predict_interval(rows, 0.8)
    assert np.all(noisy[10:, 1] - noisy[10:, 0] > widths[0][10:])
    with pytest.raises(AshlarError):
        model.with_uncertain_inputs(4, noise={"city": 1.0}).predict_interval(rows, 0.8)
    with pytest.raises(AshlarError):
        model.with_uncertain_inputs(0)
    assert model.input_log_density(rows).shape == (20,)
    model.save(tmp_path / "m")
    loaded = ASRRegressor.load(tmp_path / "m")
    np.testing.assert_allclose(loaded.input_log_density(rows), model.input_log_density(rows), rtol=1e-6)
    np.testing.assert_allclose(loaded.with_uncertain_inputs(16).predict_interval(rows, 0.8), wide, rtol=1e-4)
    loaded.input_model_ = None
    with pytest.raises(AshlarError, match="fit_inputs"):
        loaded.with_uncertain_inputs(4)
    loaded.fit_inputs(X)
    assert loaded.input_model_ is not None


@pytest.mark.slow
def test_drawing_missing_cells_holds_coverage():
    values = _correlated(6000, seed=1)
    rng = np.random.default_rng(1)
    y = 2 * values[:, 0] + values[:, 2] + 0.3 * rng.normal(size=6000)
    X = values[:4000].astype(np.float32)
    model = ASRRegressor(task="mixture", max_epochs=30, random_state=0).fit(X, y[:4000])
    test = values[4000:].copy()
    missing = rng.random(len(test)) < 0.5
    test[missing, 0] = np.nan
    truth = y[4000:]
    covered = {}
    for name, view in (("median", model), ("draws", model.with_uncertain_inputs(32))):
        lower, upper = view.predict_interval(test.astype(np.float32), 0.8).T
        covered[name] = np.mean(((truth >= lower) & (truth <= upper))[missing])
    # The rows with a missing cell: the median's interval is far too narrow, the draws' about right.
    assert covered["median"] < 0.5
    assert 0.72 < covered["draws"] < 0.88
