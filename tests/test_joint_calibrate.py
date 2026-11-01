"""Calibrating a joint head: each column, the sum, and any named function of the columns."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import ASRModel, ASRRegressor
from ashlar.calibrate import SUM, joint_targets, pit_calibration, recalibrated_cdf, target_spec, weights_name
from ashlar.distributions import (
    AffineTransformed,
    AutoregressiveFlow,
    Independent,
    LowRankMultivariateNormal,
    Mixture,
    Normal,
    joint_mixture_distribution,
)
from ashlar.errors import AshlarError, ShapeError


def peak(y):
    """The largest column: a function of the joint outcome that no marginal gives."""
    return y.max(axis=-1)


def _shared_shock(n, seed):
    """Three columns that share one shock whose size depends on a feature, in different units."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    shock = rng.normal(size=n) * (0.5 + np.abs(x[:, 2]))
    y = np.stack([x[:, 0] + shock, -x[:, 1] + shock, x[:, 0] * x[:, 1] + shock], axis=1) + 0.3 * rng.normal(size=(n, 3))
    return x, y * np.array([1.0, 10.0, 3.0])


def _covered(interval, values):
    return np.mean((values >= interval[..., 0]) & (values <= interval[..., 1]), axis=0)


def _weighted_sum_moments(dist, weights):
    """Mean and variance of ``weights · Y`` from the joint mean and covariance."""
    w = jnp.asarray(weights, dtype=jnp.float32)
    mean = jnp.sum(w * dist.mean(), axis=-1)
    variance = jnp.einsum("...i,...ij,...j->...", w, dist.covariance(), w)
    return np.asarray(mean), np.asarray(variance)


def test_a_weighted_sum_of_joint_normals_is_exact():
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    rows, members, d, rank = 5, 3, 4, 2
    joint = LowRankMultivariateNormal(
        jax.random.normal(keys[0], (rows, members, d)),
        jax.random.normal(keys[1], (rows, members, d, rank)),
        jnp.exp(jax.random.normal(keys[2], (rows, members, d))),
    )
    weights = np.array([1.0, 1.0, -1.0, 2.0])
    flat = jax.random.normal(keys[3], (rows, members, 3 * (1 + d * (2 + rank))))
    cases = [
        joint,
        Mixture(joint),
        AffineTransformed(Mixture(joint), jnp.array([1.0, 2.0, 3.0, 4.0]), jnp.array([1.0, 10.0, 100.0, 0.5])),
        AffineTransformed(Mixture(joint_mixture_distribution(flat, d, 3)), 5.0, 2.0),
        Independent(Normal(jnp.arange(4.0), jnp.array([1.0, 2.0, 0.5, 3.0]))),
    ]
    for dist in cases:
        projected = dist.project(weights)
        assert projected.batch_shape == dist.batch_shape and projected.event_shape == ()
        mean, variance = _weighted_sum_moments(dist, weights)
        assert np.allclose(projected.mean(), mean, rtol=1e-4, atol=1e-4)
        assert np.allclose(projected.variance(), variance, rtol=1e-4, atol=1e-3)
    # The sum of a normal's columns is the normal it says: its own CDF at its own quantile.
    total = cases[2].project(np.ones(d))
    assert np.allclose(total.cdf(total.quantile(0.9)), 0.9, atol=1e-4)


def test_the_flow_and_other_columns_have_no_closed_form_sum():
    d, hidden = 2, 8
    flow = AutoregressiveFlow(
        jnp.zeros((3, 1, hidden + d * 25)), jnp.zeros((1, 1, hidden, d)), jnp.zeros((1, 1, d * 25, hidden)), d
    )
    with pytest.raises(NotImplementedError):
        flow.project(np.ones(d))
    with pytest.raises(NotImplementedError):
        Mixture(flow).project(np.ones(d))
    with pytest.raises(NotImplementedError):
        Normal(0.0, 1.0).project(np.ones(1))


def test_a_recalibrated_probability_keeps_a_calibrated_model_and_tempers_an_overconfident_one():
    from scipy.stats import norm

    rng = np.random.default_rng(0)
    calibrated = pit_calibration(rng.uniform(size=4000))
    p = np.linspace(0.0, 1.0, 11)
    assert np.allclose(recalibrated_cdf(calibrated, p), p, atol=0.02)
    # y twice as spread as the model believes: a small P(Y <= t) was too small, a large one too large.
    overconfident = pit_calibration(norm.cdf(2.0 * rng.normal(size=4000)))
    fixed = recalibrated_cdf(overconfident, np.array([0.02, 0.5, 0.98]))
    assert fixed[0] > 0.1 and abs(fixed[1] - 0.5) < 0.03 and fixed[2] < 0.9
    values = recalibrated_cdf(overconfident, np.linspace(0, 1, 101))
    assert np.all(np.diff(values) >= 0) and values[0] == 0.0 and values[-1] == 1.0
    assert np.all((values[1:-1] > 0) & (values[1:-1] < 1))


def test_targets_are_named_and_checked():
    assert target_spec("sum", 3)[0] == SUM
    name, spec = target_spec(peak, 3)
    assert name == "peak" and spec["function"] is peak
    name, spec = target_spec([1, 0, -1], 3)
    assert name == weights_name([1, 0, -1]) and np.allclose(spec["weights"], [1, 0, -1])
    assert list(joint_targets(None, 3)) == [SUM]
    assert list(joint_targets([peak, [1.0, -1.0, 0.0]], 3)) == [SUM, "peak", weights_name([1, -1, 0])]
    assert list(joint_targets([2.0, 0.0, 1.0], 3)) == [SUM, weights_name([2, 0, 1])]
    for bad in (lambda y: y[..., 0], [1.0, 2.0], [0.0, 0.0, 0.0], [1.0, np.nan, 0.0]):
        with pytest.raises(AshlarError):
            target_spec(bad, 3)
    with pytest.raises(AshlarError):
        joint_targets("peak", 3)


def test_a_joint_model_calibrates_its_sum_and_saves_it(tmp_path):
    x, y = _shared_shock(400, 0)
    model = ASRRegressor(task="multivariate_gaussian", max_epochs=2, random_state=0).fit(x[:300], y[:300])
    with pytest.raises(AshlarError):
        model.predict_interval(x[300:], target="sum", calibrated=True)
    model.calibrate(x[300:], y[300:])
    assert list(model.calibration_["targets"]) == [SUM] and len(model.calibration_["columns"]) == 3
    interval = model.predict_interval(x[300:], 0.8, target="sum")
    assert interval.shape == (100, 2) and np.all(interval[:, 0] < interval[:, 1])
    probability = model.predict_exceedance(x[300:], 0.0, target="sum", calibrated=False)
    exact = 1.0 - np.asarray(model.predict_dist(x[300:]).project(np.ones(3)).cdf(0.0))
    assert np.allclose(probability, exact, atol=1e-6)
    restored = ASRModel.load(model.save(tmp_path / "joint"))
    assert restored.calibration_ == model.calibration_
    assert np.allclose(restored.predict_interval(x[300:], 0.8, target="sum"), interval, atol=1e-4)


@pytest.mark.slow
def test_a_joint_model_reads_every_target_and_refuses_what_it_was_not_calibrated_on(tmp_path):
    x, y = _shared_shock(400, 0)
    model = ASRRegressor(task="multivariate_gaussian", max_epochs=2, random_state=0).fit(x[:300], y[:300])
    model.calibrate(x[300:], y[300:], target=[peak, [1.0, -1.0, 0.0]])
    assert set(model.calibration_["targets"]) == {SUM, "peak", weights_name([1, -1, 0])}
    assert len(model.calibration_["columns"]) == 3
    assert model.predict_interval(x[:5], 0.8).shape == (5, 3, 2)
    assert model.predict_interval(x[:5], 0.8, target="sum").shape == (5, 2)
    assert model.predict_interval(x[:5], 0.8, target=[1.0, -1.0, 0.0]).shape == (5, 2)
    assert model.predict_quantiles(x[:5], [0.1, 0.5, 0.9], target=peak).shape == (5, 3)
    assert model.predict_quantiles(x[:5], [0.1, 0.9]).shape == (5, 3, 2)
    assert model.predict_exceedance(x[:5], 0.0, target="sum").shape == (5,)
    assert model.predict_exceedance(x[:5], [0.0, 1.0, 2.0]).shape == (5, 3)
    probability = model.predict_exceedance(x[:5], 0.0, target="sum", calibrated=False)
    exact = 1.0 - np.asarray(model.predict_dist(x[:5]).project(np.ones(3)).cdf(0.0))
    assert np.allclose(probability, exact, atol=1e-6)

    # A target the calibration does not cover is refused rather than read raw.
    with pytest.raises(AshlarError):
        model.predict_interval(x[:5], target=[0.0, 1.0, 1.0])
    assert model.predict_interval(x[:5], target=[0.0, 1.0, 1.0], calibrated=False).shape == (5, 2)
    with pytest.raises(AshlarError):
        model.calibrate(x[300:], y[300:], target=lambda values: values.sum(axis=-1))
    with pytest.raises(AshlarError):
        ASRRegressor(task="gaussian", max_epochs=1).fit(x, y[:, 0]).calibrate(x, y[:, 0], target="sum")
    with pytest.raises(ShapeError):
        model.calibrate(x[300:], y[300:, :2])

    restored = ASRModel.load(model.save(tmp_path / "joint"))
    assert restored.calibration_ == model.calibration_
    for target in (None, "sum", [1.0, -1.0, 0.0]):
        assert np.allclose(restored.predict_interval(x[:20], 0.8, target=target), model.predict_interval(x[:20], 0.8, target=target), atol=1e-4)
    # A function's PIT table is saved; the function is not, so it comes back by name.
    with pytest.raises(AshlarError):
        restored.predict_interval(x[:5], target="peak")
    assert np.allclose(restored.predict_interval(x[:20], target=peak), model.predict_interval(x[:20], target=peak))
    model.fit(x[:300], y[:300])
    assert model.calibration_ is None


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, kwargs",
    [
        # Without its covariance the normal's sum is too narrow, so calibration has something to fix.
        ("multivariate_gaussian", {"head_kwargs": {"rank": 0}}),
        ("multivariate_mixture", {"head_kwargs": {"components": 3}}),
        ("multivariate_flow", {}),
    ],
)
def test_calibrated_coverage_of_the_sum_a_function_and_each_column(task, kwargs):
    x, y = _shared_shock(2000, 0)
    x_cal, y_cal = _shared_shock(2000, 1)
    x_new, y_new = _shared_shock(3000, 2)
    model = ASRRegressor(task=task, max_epochs=15, random_state=0, **kwargs).fit(x, y)
    model.calibrate(x_cal, y_cal, target=peak, n_samples=500)
    for target, values in ((None, y_new), ("sum", y_new.sum(axis=1)), (peak, peak(y_new))):
        interval = model.predict_interval(x_new, 0.8, target=target, n_samples=500)
        assert np.all(np.abs(_covered(interval, values) - 0.8) < 0.03), (target, _covered(interval, values))
    if task == "multivariate_gaussian":
        raw = model.predict_interval(x_new, 0.8, target="sum", calibrated=False)
        assert _covered(raw, y_new.sum(axis=1)) < 0.74
    # The calibrated chance the total exceeds its 80th percentile matches how often it does.
    capacity = np.quantile(y_new.sum(axis=1), 0.8)
    probability = model.predict_exceedance(x_new, capacity, target="sum", n_samples=500)
    assert abs(probability.mean() - 0.2) < 0.025
