"""The multivariate_gaussian task: one joint distribution over several target columns."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import ASRModel, ASRRegressor
from ashlar.distributions import AffineTransformed, Independent, joint_normal_distribution
from ashlar.errors import AshlarError, ShapeError
from ashlar.heads import MultivariateNormalHead
from ashlar.objectives import make_joint_objective


def _shared_shock(n=2400, seed=0):
    """Three columns that share one shock whose size depends on a feature."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    shock = rng.normal(size=n) * (0.3 + np.abs(x[:, 2]))
    noise = 0.3 * rng.normal(size=(n, 3))
    y = np.stack([x[:, 0] + shock, -x[:, 1] + shock, x[:, 0] * x[:, 1] + 0.5 * shock], axis=1) + noise
    return x, (y * np.array([1.0, 10.0, 100.0]) + np.array([0.0, 50.0, -20.0])).astype(np.float32)


def test_head_width_rank_and_errors():
    head = MultivariateNormalHead(8, 2, n_events=4, n_classes=2, key=jax.random.PRNGKey(0))
    assert head.rank == 2 and head(jnp.zeros((5, 2, 8)), key=None).shape == (5, 2, 4 * 4)
    assert MultivariateNormalHead(8, 1, n_events=3, n_classes=2, key=jax.random.PRNGKey(0), rank=0).rank == 0
    with pytest.raises(ShapeError):
        MultivariateNormalHead(8, 1, n_events=3, n_classes=2, key=jax.random.PRNGKey(0), rank=4)


def test_objective_is_the_member_mean_joint_negative_log_density():
    raw = jax.random.normal(jax.random.PRNGKey(1), (6, 2, 3 * 4))
    y = jax.random.normal(jax.random.PRNGKey(2), (6, 3))
    expected = -jnp.mean(joint_normal_distribution(raw, 3).log_prob(y[:, None, :]))
    assert np.isclose(make_joint_objective(3)(raw, y, jnp.ones(6)), expected, rtol=1e-6)


@pytest.mark.slow
def test_fit_learns_dependence_and_reports_in_target_units(tmp_path):
    x, y = _shared_shock()
    train, test = slice(0, 2000), slice(2000, None)
    joint = ASRRegressor(task="multivariate_gaussian", max_epochs=25).fit(x[train], y[train])
    loose = ASRRegressor(task="multivariate_gaussian", head_kwargs={"rank": 0}, max_epochs=25).fit(x[train], y[train])
    assert joint.predict(x[test]).shape == (400, 3)
    # Modelling the shared shock is worth a lot of log density on held-out rows.
    assert joint.log_prob(x[test], y[test]).mean() > loose.log_prob(x[test], y[test]).mean() + 0.5

    dist = joint.predict_dist(x[test])
    assert isinstance(dist, AffineTransformed) and dist.batch_shape == (400,) and dist.event_shape == (3,)
    draws = np.asarray(dist[:3].sample(jax.random.PRNGKey(0), (8000,)))
    assert np.corrcoef(draws[:, 0].T)[0, 1] > 0.3
    assert np.allclose(dist.mean(), joint.predict(x[test]), rtol=1e-4, atol=1e-3)
    # The columns come back in their own units.
    assert 3 < np.median(np.asarray(dist.stddev())[:, 1]) / np.median(np.asarray(dist.stddev())[:, 0]) < 30

    quantiles = joint.predict_quantiles(x[test][:5], [0.1, 0.5, 0.9])
    assert quantiles.shape == (5, 3, 3)
    assert np.allclose(quantiles[:, :, 1], np.asarray(dist.marginals()[:5].quantile(0.5)), atol=1e-3)
    assert joint.sample(x[test][:4], n=6).shape == (6, 4, 3)
    assert joint.predict_uncertainty(x[test][:4])["total"].shape == (4, 3)

    independent = Independent(dist.marginals())
    assert np.allclose(independent.mean(), dist.mean()) and np.allclose(independent.variance(), dist.variance())

    restored = ASRModel.load(joint.save(tmp_path / "joint"))
    assert np.allclose(restored.log_prob(x[test], y[test]), joint.log_prob(x[test], y[test]), atol=1e-4)
    with pytest.raises(AshlarError):
        joint.calibrate(x[test], y[test])
