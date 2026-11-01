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
    newsvendor_objective,
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


def test_newsvendor_objective_trains_saves_and_orders(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 3)).astype(np.float32)
    y = np.exp(0.5 * X[:, 0] + 0.5 * rng.standard_gamma(2.0, size=600))
    objective = newsvendor_objective("gaussian", 9, 1)
    model = ASRRegressor(task="gaussian", objective=objective, max_epochs=3, random_state=0).fit(X, y)
    order = model.predict_order(X[:50], 9, 1)
    np.testing.assert_allclose(order, model.predict_quantiles(X[:50], [0.9])[:, 0])
    model.save(tmp_path / "m")
    loaded = ASRRegressor.load(tmp_path / "m")
    assert repr(loaded.objective) == repr(objective)
    np.testing.assert_allclose(loaded.predict_order(X[:50], 9, 1), order, rtol=1e-5)
    with pytest.raises(AshlarError):
        newsvendor_objective("quantile", 9, 1)
    with pytest.raises(AshlarError):
        newsvendor_objective("mixture", 9, 1, nll=0.0, cost=0.0)


@pytest.mark.slow
def test_training_on_the_cost_fixes_a_misspecified_head():
    """A symmetric head on a skewed target orders badly from its likelihood; the cost moves its 0.9 quantile."""
    rng = np.random.default_rng(0)

    def data(n):
        X = rng.normal(size=(n, 4)).astype(np.float32)
        scale = 0.4 + 0.3 * np.tanh(X[:, 1])
        return X, np.exp(0.5 * X[:, 0] + scale * rng.standard_gamma(2.0, size=n))

    X, y = data(4000)
    X_test, y_test = data(10000)
    costs = {}
    for name, objective in (("nll", None), ("cost", newsvendor_objective("gaussian", 9, 1, nll=0.0))):
        model = ASRRegressor(task="gaussian", objective=objective, max_epochs=30, random_state=0).fit(X, y)
        costs[name] = float(np.mean(newsvendor_cost(y_test, model.predict_order(X_test, 9, 1), 9, 1)))
    assert costs["cost"] < 0.9 * costs["nll"]
