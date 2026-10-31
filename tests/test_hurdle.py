"""The hurdle task: an exact zero with one probability, a positive amount otherwise."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import ASRModel, ASRRegressor
from ashlar.distributions import Gamma, Hurdle, LogNormal, Mixture, randomized_pit
from ashlar.errors import AshlarError


@pytest.mark.parametrize("severity", [LogNormal(jnp.array([0.0, 1.0]), jnp.array([0.5, 1.0])), Gamma(jnp.array([2.0, 5.0]), jnp.array([0.7, 3.0]))])
def test_hurdle_moments_quantiles_and_pit(severity):
    dist = Hurdle(jnp.array([0.3, 0.6]), severity)
    draws = np.asarray(dist.sample(jax.random.PRNGKey(0), (200000,)))
    assert np.allclose((draws == 0).mean(axis=0), [0.3, 0.6], atol=0.005)
    assert np.allclose(draws.mean(axis=0), dist.mean(), rtol=0.02)
    assert np.allclose(draws.var(axis=0), dist.variance(), rtol=0.06)
    quantiles = dist.quantile(jnp.array([0.2, 0.5, 0.9])[:, None])
    assert np.allclose(quantiles[0], 0.0) and float(quantiles[1, 1]) == 0.0
    assert np.allclose(quantiles, np.quantile(draws, [0.2, 0.5, 0.9], axis=0), rtol=0.03, atol=1e-6)
    assert np.allclose(dist.log_prob(jnp.zeros(2)), np.log([0.3, 0.6]))
    pit = np.asarray(randomized_pit(dist, jnp.asarray(draws[:20000]), jax.random.PRNGKey(1)))
    assert np.all(np.abs(np.histogram(pit, bins=5, range=(0, 1))[0] / pit.size - 0.2) < 0.01)
    mixture = Mixture(Hurdle(jnp.array([[0.3, 0.6]]), LogNormal(jnp.zeros((1, 2)), jnp.ones((1, 2)))))
    assert float(mixture.quantile(0.3)[0]) == 0.0 and float(mixture.quantile(0.6)[0]) > 0.0


def test_hurdle_fit_recovers_the_zero_probability(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3000, 2)).astype(np.float32)
    zero = 1.0 / (1.0 + np.exp(-1.5 * x[:, 0]))
    y = np.where(rng.random(3000) < zero, 0.0, np.exp(1.0 + 0.5 * x[:, 1] + 0.4 * rng.normal(size=3000))).astype(np.float32)
    model = ASRRegressor(task="hurdle", max_epochs=30).fit(x[:2500], y[:2500])
    dist = model.predict_dist(x[2500:])
    assert np.corrcoef(np.asarray(dist.cdf(0.0)), zero[2500:])[0, 1] > 0.95
    assert np.isfinite(model.log_prob(x[2500:], y[2500:])).all()
    assert np.allclose(model.predict(x[2500:]), dist.mean(), rtol=1e-4)
    model.calibrate(x[2500:2750], y[2500:2750])
    interval = model.predict_interval(x[2750:], coverage=0.8)
    assert np.mean((y[2750:] >= interval[:, 0]) & (y[2750:] <= interval[:, 1])) >= 0.75
    restored = ASRModel.load(model.save(tmp_path / "hurdle"))
    assert np.allclose(restored.predict(x[:20]), model.predict(x[:20]), atol=1e-5)
    gamma = ASRRegressor(task="hurdle", head_kwargs={"severity": "gamma"}, max_epochs=2).fit(x[:500], y[:500])
    assert isinstance(gamma.predict_dist(x[:3]).components.severity, Gamma)
    with pytest.raises(AshlarError):
        ASRRegressor(task="hurdle", max_epochs=1).fit(x[:50], y[:50] - 1.0)
