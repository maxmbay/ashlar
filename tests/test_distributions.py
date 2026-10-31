"""One distribution interface for every head: shapes, agreement with the head math, and JAX transforms."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import ASRForecaster, ASRModel, ASRRegressor
from ashlar.distributional import decode, flow_output_count, flow_parameter_count, log_density
from ashlar.distributions import (
    AffineTransformed,
    Categorical,
    Distribution,
    Flow,
    Histogram,
    Independent,
    LowRankMultivariateNormal,
    Mixture,
    NegativeBinomial,
    Normal,
    Poisson,
    QuantileLevels,
    StudentT,
    head_distribution,
    randomized_pit,
)

KEY = jax.random.PRNGKey(0)


def _continuous():
    keys = jax.random.split(KEY, 6)
    return {
        "normal": Normal(jnp.array([0.0, 1.0, -2.0]), jnp.array([1.0, 0.5, 2.0])),
        "student_t": StudentT(jnp.array([0.0, 1.0, -2.0]), jnp.array([1.0, 0.5, 2.0]), jnp.array([3.0, 5.0, 2.5])),
        "mixture": Mixture(Normal(jax.random.normal(keys[0], (3, 4)) * 2, jnp.full((3, 4), 0.7)), jax.random.normal(keys[1], (3, 4))),
        "flow": Flow(jax.random.normal(keys[2], (3, flow_output_count()))),
        "flow_bare_spline": Flow(jax.random.normal(keys[5], (3, flow_parameter_count()))),
        "histogram": Histogram(jax.random.normal(keys[3], (3, 16))),
        "affine": AffineTransformed(Mixture(Flow(jax.random.normal(keys[4], (3, 5, flow_parameter_count())))), 10.0, 3.0),
    }


@pytest.mark.slow
@pytest.mark.parametrize("name", list(_continuous()))
def test_continuous_shapes_moments_and_quantiles(name):
    dist = _continuous()[name]
    assert dist.batch_shape == (3,) and dist.event_shape == ()
    draws = np.asarray(dist.sample(KEY, (40000,)))
    assert draws.shape == (40000, 3)
    mean, variance = np.asarray(dist.mean()), np.asarray(dist.variance())
    tolerance = 4 * np.sqrt(variance / draws.shape[0])
    if name != "student_t":  # heavy tails make the sample variance itself noisy
        assert np.all(np.abs(draws.mean(axis=0) - mean) < tolerance + 1e-3)
        assert np.allclose(draws.var(axis=0), variance, rtol=0.08)
    levels = jnp.array([0.05, 0.5, 0.95])[:, None]
    quantiles = dist.quantile(levels)
    assert quantiles.shape == (3, 3)
    assert np.allclose(dist.cdf(quantiles), np.broadcast_to(levels, (3, 3)), atol=2e-4)
    assert np.allclose(np.quantile(draws, [0.05, 0.5, 0.95], axis=0), quantiles, atol=0.08 * np.sqrt(variance).max())
    grid = jnp.linspace(-60.0, 80.0, 70001)[:, None]
    mass = np.trapezoid(np.exp(np.asarray(dist.log_prob(grid))), dx=float(grid[1, 0] - grid[0, 0]), axis=0)
    if name != "histogram":  # the histogram reads values past its edges in the end bins
        assert np.allclose(mass, 1.0, atol=5e-3)


@pytest.mark.parametrize(
    "dist",
    [
        Poisson(jnp.array([0.5, 4.0, 30.0])),
        Poisson(jnp.array([0.5, 4.0, 30.0]), jnp.array([0.3, 0.1, 0.0])),
        NegativeBinomial(jnp.array([0.5, 4.0, 30.0]), jnp.array([0.2, 1.0, 0.5]), jnp.array([0.0, 0.2, 0.05])),
    ],
)
@pytest.mark.slow
def test_count_cdf_quantile_and_moments(dist):
    support = jnp.arange(0.0, 400.0)[:, None]
    mass = np.exp(np.asarray(dist.log_prob(support)))
    assert np.allclose(mass.sum(axis=0), 1.0, atol=1e-4)
    assert np.allclose(np.cumsum(mass, axis=0)[[0, 3, 40]], dist.cdf(jnp.array([0.0, 3.0, 40.0])[:, None]), atol=1e-4)
    values = np.arange(400.0)[:, None]
    assert np.allclose((mass * values).sum(axis=0), dist.mean(), rtol=1e-4)
    assert np.allclose((mass * values**2).sum(axis=0) - np.square(dist.mean()), dist.variance(), rtol=1e-3)
    levels = jnp.array([0.1, 0.5, 0.9, 0.99])[:, None]
    quantiles = np.asarray(dist.quantile(levels))
    cumulative = np.cumsum(mass, axis=0)
    expected = np.argmax(cumulative >= np.asarray(levels)[:, :, None].transpose(0, 2, 1) - 1e-6, axis=1)
    assert np.array_equal(quantiles, expected)
    draws = dist.sample(KEY, (5,))
    assert draws.shape == (5, 3) and np.all(np.asarray(draws) == np.round(np.asarray(draws)))


@pytest.mark.slow
def test_randomized_pit_is_uniform_for_counts():
    dist = Poisson(jnp.full((20000,), 2.0))
    draws = dist.sample(KEY)
    pit = np.asarray(randomized_pit(dist, draws, jax.random.PRNGKey(1)))
    assert abs(np.mean(pit < 0.2) - 0.2) < 0.015 and abs(np.mean(pit < 0.7) - 0.7) < 0.015


@pytest.mark.slow
def test_low_rank_density_matches_dense_and_draws_match_covariance():
    from scipy.stats import multivariate_normal

    factor = jax.random.normal(KEY, (4, 5, 2))
    diag = jnp.linspace(0.2, 1.0, 5)
    dist = LowRankMultivariateNormal(jnp.arange(5.0), factor, diag)
    value = jax.random.normal(jax.random.PRNGKey(3), (4, 5))
    dense = np.asarray(dist.covariance())
    expected = [multivariate_normal(np.arange(5.0), dense[i]).logpdf(np.asarray(value[i])) for i in range(4)]
    assert np.allclose(dist.log_prob(value), expected, atol=1e-4)
    draws = np.asarray(dist.sample(KEY, (60000,)))
    assert np.abs(np.cov(draws[:, 1].T) - dense[1]).max() < 0.1
    independent = LowRankMultivariateNormal(jnp.zeros(5), jnp.zeros((5, 0)), diag)
    assert np.allclose(independent.log_prob(value), Independent(Normal(jnp.zeros(5), jnp.sqrt(diag))).log_prob(value))


@pytest.mark.slow
def test_joint_marginals_and_independent_marginals():
    components = LowRankMultivariateNormal(jnp.zeros((4, 3, 2)), jnp.ones((4, 3, 2, 1)), jnp.full((4, 3, 2), 0.25))
    joint = AffineTransformed(Mixture(components), jnp.array([10.0, -1.0]), jnp.array([2.0, 0.5]))
    assert joint.batch_shape == (4,) and joint.event_shape == (2,)
    marginals = joint.marginals()
    assert marginals.batch_shape == (4, 2) and marginals.event_shape == ()
    assert np.allclose(marginals.mean(), joint.mean()) and np.allclose(marginals.variance(), joint.variance())
    assert np.allclose(marginals.quantile(0.5), jnp.array([10.0, -1.0]), atol=1e-4)
    independent = Independent(marginals)
    draws = np.asarray(joint.sample(KEY, (20000,)))
    loose = np.asarray(independent.sample(KEY, (20000,)))
    assert np.corrcoef(draws[:, 0].T)[0, 1] > 0.7 and abs(np.corrcoef(loose[:, 0].T)[0, 1]) < 0.05
    assert np.allclose(joint.covariance()[0], np.cov(draws[:, 0].T), rtol=0.05)


@pytest.mark.slow
def test_distributions_are_pytrees_for_jit_vmap_grad_and_indexing():
    dist = _continuous()["affine"]
    value = jnp.array([9.0, 10.0, 11.0])
    assert np.allclose(jax.jit(lambda d, v: d.log_prob(v))(dist, value), dist.log_prob(value), atol=1e-5)
    rows = jax.vmap(lambda d, v: d.log_prob(v))(dist, value)
    assert np.allclose(rows, dist.log_prob(value), atol=1e-5)
    assert dist[1:].batch_shape == (2,) and np.allclose(dist[1:].log_prob(value[1:]), dist.log_prob(value)[1:])
    with pytest.raises(Exception):
        dist[..., 0]

    def loss(loc):
        return jnp.sum(Normal(loc, 1.0).sample(KEY, (8,)))

    assert np.allclose(jax.grad(loss)(jnp.zeros(3)), 8.0)

    def tail(loc):
        return Mixture(Normal(loc, jnp.ones(2))).quantile(0.9)

    loc = jnp.array([0.0, 2.0])
    step = 1e-3
    numeric = (tail(loc + jnp.array([step, 0.0])) - tail(loc - jnp.array([step, 0.0]))) / (2 * step)
    assert np.isclose(jax.grad(tail)(loc)[0], numeric, rtol=1e-2)


@pytest.mark.slow
def test_head_distribution_matches_the_training_density():
    for kind, width in [("gaussian", 2), ("mixture", 15), ("flow", flow_output_count()), ("flow", flow_parameter_count()), ("histogram", 16), ("student_t", 3)]:
        flat = jax.random.normal(KEY, (6, width))
        value = jax.random.normal(jax.random.PRNGKey(2), (6,))
        expected = log_density(kind, decode(kind, flat, 1), value[:, None])[:, 0]
        assert np.allclose(head_distribution(kind, flat).log_prob(value), expected, atol=1e-5), kind


def test_quantile_levels_and_categorical():
    levels = QuantileLevels(values=jnp.array([[1.0, 2.0, 3.0]]), levels=(0.1, 0.5, 0.9))
    assert float(levels.quantile(0.9)[0]) == 3.0 and float(levels.median()[0]) == 2.0
    with pytest.raises(NotImplementedError):
        levels.quantile(0.7)
    with pytest.raises(NotImplementedError):
        levels.log_prob(jnp.zeros(1))
    classes = Categorical(probs=jnp.array([[0.2, 0.8]]))
    assert np.isclose(classes.log_prob(jnp.array([1])), np.log(0.8))
    assert isinstance(classes, Distribution)


def _table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = (x[:, 0] + rng.normal(scale=0.3 + 0.3 * np.abs(x[:, 1]), size=n)).astype(np.float32) * 5 + 20
    return x, y


@pytest.mark.slow
@pytest.mark.parametrize("task", ["gaussian", "mixture", "flow", "histogram", "student_t"])
def test_estimator_methods_read_predict_dist(task):
    x, y = _table()
    model = ASRRegressor(task=task, max_epochs=3, backbone="tabm", backbone_kwargs={"k": 4, "n_blocks": 1, "d_block": 32}).fit(x, y)
    dist = model.predict_dist(x[:50])
    assert isinstance(dist, AffineTransformed) and dist.batch_shape == (50,)
    assert np.allclose(model.log_prob(x[:50], y[:50]), dist.log_prob(jnp.asarray(y[:50])), atol=1e-5)
    quantiles = model.predict_quantiles(x[:50], [0.1, 0.5, 0.9])
    assert np.allclose(quantiles, np.asarray(dist.quantile(jnp.array([0.1, 0.5, 0.9])[:, None])).T, atol=1e-4)
    assert model.sample(x[:50], n=7).shape == (7, 50)
    # The draws follow the member mixture that log_prob and the quantiles describe.
    draws = model.sample(x[:5], n=20000, seed=3)
    spread = np.asarray(dist[:5].stddev())
    assert np.allclose(np.median(draws, axis=0), quantiles[:5, 1], atol=0.05 * spread.max())


@pytest.mark.slow
def test_classifier_predict_dist_is_categorical():
    x, _y = _table()
    labels = np.where(x[:, 0] > 0, "up", "down")
    model = ASRModel(task="classification", max_epochs=3).fit(x, labels)
    dist = model.predict_dist(x[:10])
    assert isinstance(dist, Categorical)
    assert np.allclose(dist.probs, model.predict_proba(x[:10]), atol=1e-6)
    assert np.allclose(model.log_prob(x[:10], labels[:10]), np.log(model.predict_proba(x[:10])[np.arange(10), (labels[:10] == "up").astype(int)]), atol=1e-5)
    assert set(np.unique(model.sample(x[:10], n=5))) <= {"up", "down"}


@pytest.mark.slow
def test_forecaster_predict_dist_matches_log_prob():
    series = np.sin(np.arange(400) / 6.0) * 10 + np.random.default_rng(0).normal(size=400)
    model = ASRForecaster(lookback=24, horizon=3, head="mixture", max_epochs=2).fit(series)
    histories = np.stack([series[i : i + 24] for i in range(5)])
    future = np.stack([series[i + 24 : i + 27] for i in range(5)])
    dist = model.predict_dist(histories)
    assert dist.batch_shape == (5, 3)
    assert np.allclose(dist.log_prob(jnp.asarray(future, dtype=jnp.float32)), model.log_prob(histories, future), atol=1e-4)
    assert model.predict_dist(series[:24]).batch_shape == (3,)
