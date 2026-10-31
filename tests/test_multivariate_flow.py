"""The multivariate_flow task: a conditional masked autoregressive flow over several target columns."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import ASRModel, ASRRegressor
from ashlar.distributions import (
    AffineTransformed,
    AutoregressiveFlow,
    Empirical,
    Mixture,
    joint_flow_distribution,
)
from ashlar.errors import AshlarError, ShapeError
from ashlar.heads import MultivariateFlowHead
from ashlar.objectives import make_joint_objective


def _raw(rows=5, members=1, d=3, depth=1, scale=3.0, seed=0):
    head = MultivariateFlowHead(8, members, n_events=d, n_classes=2, key=jax.random.PRNGKey(seed), depth=depth, hidden=16)
    context, w_in, w_out = head(jax.random.normal(jax.random.PRNGKey(seed + 1), (rows, members, 8)), key=None)
    return context * scale, w_in, w_out * scale


def test_head_output_and_errors():
    context, w_in, w_out = MultivariateFlowHead(8, 2, n_events=3, n_classes=2, key=jax.random.PRNGKey(0), depth=2)(
        jnp.zeros((5, 2, 8)), key=None
    )
    assert context.shape[:2] == (5, 2) and w_in.shape == (2, 2, 64, 3) and w_out.shape[:2] == (2, 2)
    with pytest.raises(ShapeError):
        MultivariateFlowHead(8, 1, n_events=3, n_classes=2, key=jax.random.PRNGKey(0), depth=0)
    with pytest.raises(ShapeError):
        AutoregressiveFlow(context[..., :-1], w_in, w_out, 3)


def test_zero_parameters_are_the_standard_normal():
    context, w_in, w_out = _raw(scale=0.0)
    dist = AutoregressiveFlow(context, w_in, w_out, 3)
    y = jax.random.normal(jax.random.PRNGKey(4), (5, 1, 3))
    assert np.allclose(dist.log_prob(y), jnp.sum(jax.scipy.stats.norm.logpdf(y), axis=-1), atol=1e-5)


@pytest.mark.parametrize("depth", [1, 2])
def test_inverse_undoes_forward_and_density_is_the_change_of_variables(depth):
    context, w_in, w_out = _raw(depth=depth)
    dist = AutoregressiveFlow(context, w_in, w_out, 3)
    z = jax.random.normal(jax.random.PRNGKey(3), (5, 1, 3))
    base, logdet = dist.inverse(dist.forward(z))
    assert np.allclose(base, z, atol=1e-4)

    def forward(noise, row):
        return AutoregressiveFlow(row[None], w_in, w_out, 3).forward(noise[None])[0]

    jacobian = np.asarray(jax.vmap(jax.jacfwd(forward))(z[:, 0], context[:, 0]))
    assert np.allclose(logdet[:, 0], -np.linalg.slogdet(jacobian)[1], atol=1e-4)
    if depth == 1:
        # One block is autoregressive: column j never depends on a later column.
        assert np.all(np.triu(jacobian, 1) == 0) and np.all(np.abs(np.tril(jacobian, -1)[:, 1:, :1]) > 0)


def test_member_mixture_draws_moments_marginals_and_indexing():
    dist = Mixture(joint_flow_distribution(_raw(rows=2, members=2), 3))
    draws = np.asarray(dist.sample(jax.random.PRNGKey(5), (20000,)))
    assert draws.shape == (20000, 2, 3)
    assert np.allclose(dist.mean(), draws.mean(axis=0), atol=0.1)
    marginal = dist.marginals()
    assert np.allclose(marginal.quantile(0.5), np.median(draws, axis=0), atol=0.1)
    assert np.allclose(marginal.cdf(marginal.quantile(0.8)), 0.8, atol=0.01)
    assert dist[1:].batch_shape == (1,) and np.allclose(dist[1:].log_prob(draws[:3, 1:]), dist.log_prob(draws[:3])[:, 1:])
    single = dist.components.marginals()
    assert isinstance(single, Empirical) and single.batch_shape == (2, 2, 3)


def test_objective_is_the_member_mean_negative_log_density():
    raw = _raw(rows=6, members=2)
    y = jax.random.normal(jax.random.PRNGKey(2), (6, 3))
    expected = -jnp.mean(joint_flow_distribution(raw, 3).log_prob(y[:, None, :]))
    assert np.isclose(make_joint_objective(3, "flow")(raw, y, jnp.ones(6)), expected, rtol=1e-5)


def _curved(n=3000, seed=0):
    """Two columns with zero correlation and strong dependence: the second is the square of the first's noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    first = rng.normal(size=n)
    second = (1.0 + 0.5 * np.abs(x[:, 1])) * (first**2 - 1.0) + 0.2 * rng.normal(size=n)
    y = np.stack([x[:, 0] + first, 3.0 * x[:, 1] + second], axis=1)
    return x, (y * np.array([1.0, 20.0]) + np.array([5.0, -40.0])).astype(np.float32)


@pytest.mark.slow
def test_fit_beats_the_joint_normal_on_curved_dependence(tmp_path):
    x, y = _curved()
    train, test = slice(0, 2500), slice(2500, None)
    flow = ASRRegressor(task="multivariate_flow", max_epochs=40, random_state=0).fit(x[train], y[train])
    normal = ASRRegressor(task="multivariate_gaussian", max_epochs=40, random_state=0).fit(x[train], y[train])
    # The normal sees no correlation to model; the flow sees the curve.
    assert flow.log_prob(x[test], y[test]).mean() > normal.log_prob(x[test], y[test]).mean() + 0.3


def test_the_flow_task_predicts_samples_and_saves(tmp_path):
    x, y = _curved()
    train, test = slice(0, 600), slice(2500, None)
    flow = ASRRegressor(task="multivariate_flow", max_epochs=2, random_state=0).fit(x[train], y[train])
    dist = flow.predict_dist(x[test])
    assert isinstance(dist, AffineTransformed) and dist.batch_shape == (500,) and dist.event_shape == (2,)
    assert flow.predict(x[test]).shape == (500, 2)
    assert np.allclose(flow.predict(x[test][:5]), dist[:5].mean(), atol=1e-3)
    quantiles = flow.predict_quantiles(x[test][:6], [0.1, 0.5, 0.9])
    assert quantiles.shape == (6, 2, 3) and np.all(np.diff(quantiles, axis=-1) > 0)
    assert flow.sample(x[test][:4], n=7).shape == (7, 4, 2)
    assert flow.predict_uncertainty(x[test][:4])["total"].shape == (4, 2)

    restored = ASRModel.load(flow.save(tmp_path / "flow"))
    assert np.allclose(restored.log_prob(x[test], y[test]), flow.log_prob(x[test], y[test]), atol=1e-4)
    with pytest.raises(AshlarError):
        flow.calibrate(x[test], y[test])
