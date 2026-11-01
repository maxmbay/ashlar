"""Decisions read off a distribution, and training through the newsvendor cost."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from ashlar import ASRRegressor
from ashlar.decide import (
    best_action,
    critical_ratio,
    cvar,
    exceedance,
    expected_utility,
    newsvendor_cost,
    newsvendor_order,
)
from ashlar.distributions import LowRankMultivariateNormal, Normal
from ashlar.errors import AshlarError


def test_closed_form_decisions_on_a_normal():
    dist = Normal(jnp.array([0.0, 10.0]), jnp.array([1.0, 2.0]))
    assert critical_ratio(9, 1) == pytest.approx(0.9)
    np.testing.assert_allclose(newsvendor_order(dist, 9, 1), [stats.norm.ppf(0.9), 10 + 2 * stats.norm.ppf(0.9)], rtol=1e-5)
    np.testing.assert_allclose(exceedance(dist, 1.0), [stats.norm.sf(1.0), stats.norm.sf(-4.5)], rtol=1e-5)
    # CVaR of a standard normal at 0.95 is pdf(z) / 0.05.
    expected = stats.norm.pdf(stats.norm.ppf(0.95)) / 0.05
    np.testing.assert_allclose(cvar(dist, 0.95)[0], expected, rtol=2e-3)
    np.testing.assert_allclose(newsvendor_cost(jnp.array([3.0, 1.0]), jnp.array([1.0, 3.0]), 9, 1), [18.0, 2.0])
    with pytest.raises(AshlarError):
        critical_ratio(0, 1)


def test_decisions_by_draws_and_on_a_joint_total():
    dist = Normal(jnp.array([0.0, 5.0]), jnp.array([1.0, 1.0]))
    key = jax.random.key(0)
    mean = expected_utility(dist, lambda y: y, key, n=4000)
    np.testing.assert_allclose(mean, [0.0, 5.0], atol=0.1)

    def cost(order, y):
        return newsvendor_cost(y, order, 9, 1)

    best, costs = best_action(dist, jnp.linspace(-2, 8, 101), cost, key, n=4000)
    assert costs.shape == (101, 2)
    np.testing.assert_allclose(best, [stats.norm.ppf(0.9), 5 + stats.norm.ppf(0.9)], atol=0.15)
    joint = LowRankMultivariateNormal(jnp.zeros((1, 2)), jnp.zeros((1, 2, 1)), jnp.ones((1, 2)))
    np.testing.assert_allclose(exceedance(joint, 0.0, weights=[1.0, 1.0]), [0.5], atol=1e-5)
    with pytest.raises(AshlarError):
        exceedance(joint, 0.0)


def test_gradient_of_the_order_reaches_the_parameters():
    def order(loc, scale):
        return newsvendor_order(Normal(loc, scale), 9, 1).sum()

    d_loc, d_scale = jax.grad(order, argnums=(0, 1))(jnp.array([0.0]), jnp.array([2.0]))
    np.testing.assert_allclose(d_loc, [1.0], rtol=1e-4)
    np.testing.assert_allclose(d_scale, [stats.norm.ppf(0.9)], rtol=1e-3)


def test_predict_order_is_the_critical_ratio_quantile():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3)).astype(np.float32)
    y = np.exp(0.5 * X[:, 0] + 0.5 * rng.standard_gamma(2.0, size=600))
    model = ASRRegressor(task="gaussian", max_epochs=3, random_state=0).fit(X, y)
    order = model.predict_order(X[:50], 9, 1)
    assert order.shape == (50,)
    np.testing.assert_allclose(order, model.predict_quantiles(X[:50], [0.9])[:, 0])
    model.calibrate(X[400:], y[400:])
    np.testing.assert_allclose(model.predict_order(X[:50], 9, 1), model.predict_quantiles(X[:50], [0.9])[:, 0])
