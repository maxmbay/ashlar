"""Training a joint head on the energy and variogram scores, alone or blended with its log density."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import norm

from ashlar import ASRModel, ASRRegressor, observed
from ashlar.distributions import Independent, LowRankMultivariateNormal, Mixture, Normal
from ashlar.errors import AshlarError
from ashlar.objectives import (
    energy_score,
    make_joint_objective,
    make_joint_score_objective,
    score_draws,
    score_weights,
    variogram_score,
)


def _shared_shock(n, seed):
    """Three columns that share one shock whose size depends on a feature, in different units."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    shock = rng.normal(size=n) * (0.5 + np.abs(x[:, 2]))
    y = np.stack([x[:, 0] + shock, -x[:, 1] + shock, x[:, 0] * x[:, 1] + shock], axis=1) + 0.3 * rng.normal(size=(n, 3))
    return x, y * np.array([1.0, 10.0, 3.0])


def _absolute_normal(mean, variance):
    """``E|Z|`` for ``Z ~ N(mean, variance)``."""
    scale = np.sqrt(variance)
    return 2.0 * scale * norm.pdf(mean / scale) + mean * (2.0 * norm.cdf(mean / scale) - 1.0)


def _variogram_noise(points, probabilities, order=0.5):
    """The variance of the variogram's mean over draws, averaged over pairs: what the estimate subtracts."""
    from ashlar.objectives import _VARIOGRAM_FLOOR

    left, right = np.triu_indices(points.shape[-1], k=1)
    drawn = jnp.power(jnp.square(points[..., left] - points[..., right]) + _VARIOGRAM_FLOOR, 0.5 * order)
    weight = jnp.moveaxis(probabilities, -1, 0)[..., None]
    noise = jnp.sum(jnp.square(weight) * jnp.var(drawn, axis=1, ddof=1), axis=0) / points.shape[1]
    return jnp.mean(noise, axis=-1)


def _averaged(fn, dist, y, draws, count=4000):
    keys = jax.random.split(jax.random.PRNGKey(0), count)
    values = np.asarray(jax.jit(jax.vmap(lambda key: fn(*score_draws(dist, key, draws), y)))(keys), dtype=np.float64)
    return values.mean(axis=0), values.std(axis=0) / np.sqrt(count)


def test_the_energy_score_is_unbiased_with_two_draws_and_across_mixture_components():
    # One column: the energy score is the CRPS, which has a closed form for a mixture of normals.
    weights, means, scales, y = np.array([0.2, 0.5, 0.3]), np.array([-1.0, 0.5, 2.0]), np.array([0.5, 1.0, 0.3]), 0.7
    exact = sum(weights[k] * _absolute_normal(means[k] - y, scales[k] ** 2) for k in range(3))
    exact -= 0.5 * sum(
        weights[k] * weights[j] * _absolute_normal(means[k] - means[j], scales[k] ** 2 + scales[j] ** 2)
        for k in range(3)
        for j in range(3)
    )
    mixture = Mixture(Independent(Normal(jnp.asarray(means)[None, :, None], jnp.asarray(scales)[None, :, None])), jnp.log(weights)[None])
    estimate, error = _averaged(energy_score, mixture, jnp.array([[y]]), draws=6)
    assert abs(estimate - exact) < 4 * error + 1e-3
    single = Independent(Normal(jnp.array([[0.3]]), jnp.array([[1.7]])))
    estimate, error = _averaged(energy_score, single, jnp.array([[y]]), draws=2)
    assert abs(estimate - (_absolute_normal(0.3 - y, 1.7**2) - 0.5 * _absolute_normal(0.0, 2 * 1.7**2))) < 4 * error


def test_the_variogram_estimate_removes_the_bias_of_a_few_draws():
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    joint = LowRankMultivariateNormal(
        jax.random.normal(keys[0], (2, 3)), jax.random.normal(keys[1], (2, 3, 1)), jnp.exp(jax.random.normal(keys[2], (2, 3)))
    )
    y = jnp.array([[0.2, -0.4, 1.0], [1.0, 1.0, -1.0]])
    reference = np.asarray(variogram_score(*score_draws(joint, jax.random.PRNGKey(9), 200_000), y))
    estimate, error = _averaged(variogram_score, joint, y, draws=3)
    assert np.all(np.abs(estimate - reference) < 4 * error + 2e-3)
    points, probabilities = score_draws(joint, jax.random.PRNGKey(2), 3)
    # The square of three draws' mean alone is far too large on average: it adds that mean's variance.
    plug_in = np.mean(
        [
            np.asarray(variogram_score(*score_draws(joint, key, 3), y))
            + np.asarray(_variogram_noise(*score_draws(joint, key, 3)))
            for key in jax.random.split(jax.random.PRNGKey(3), 2000)
        ],
        axis=0,
    )
    assert np.all(plug_in > reference + 0.05)
    assert variogram_score(points, probabilities, y).shape == (2,)
    assert np.allclose(variogram_score(points[..., :1], probabilities, y[..., :1]), 0.0)


def test_score_names_and_blends_are_checked():
    assert score_weights(None) == {"nll": 1.0}
    assert score_weights("energy") == {"energy": 1.0}
    assert score_weights({"nll": 1, "energy": 0.5, "variogram": 0}) == {"nll": 1.0, "energy": 0.5}
    assert score_weights({"energy": 1.0, "variogram": 2.0}) == {"energy": 1.0, "variogram": 2.0}
    assert score_weights(lambda raw, y, w: 0.0) is None
    for bad in ("variogram", {"variogram": 1.0, "nll": 0.0}, "crps", {"energy": -1.0}, {"energy": 0.0}, {"energy": "a"}, 3):
        with pytest.raises(AshlarError):
            score_weights(bad)


@pytest.mark.parametrize("family, width", [("normal", 3 * 4), ("mixture", 2 * (1 + 3 * 4))])
def test_a_blend_adds_drawn_scores_to_the_log_density_and_differentiates(family, width):
    raw = jax.random.normal(jax.random.PRNGKey(0), (16, 2, width))
    y = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
    weight = jnp.ones(16)
    components = 2 if family == "mixture" else None
    nll = make_joint_objective(3, family, components=components)
    alone = make_joint_score_objective(3, family, {"nll": 1.0}, components=components)
    assert not getattr(alone, "stochastic") and np.isclose(alone(raw, y, weight), nll(raw, y, weight))
    blend = make_joint_score_objective(3, family, {"nll": 1.0, "energy": 1.0, "variogram": 1.0}, components=components)
    assert blend.stochastic
    with pytest.raises(AshlarError):
        blend(raw, y, weight)
    key = jax.random.PRNGKey(2)
    value, grad = jax.value_and_grad(lambda r: blend(r, y, weight, key=key))(raw)
    assert np.isfinite(value) and np.all(np.isfinite(grad)) and float(jnp.abs(grad).max()) > 0
    energy = make_joint_score_objective(3, family, {"energy": 1.0}, components=components)
    assert float(blend(raw, y, weight, key=key)) > float(energy(raw, y, weight, key=key))


def test_only_a_joint_head_takes_a_score_by_name():
    x, y = _shared_shock(60, 0)
    with pytest.raises(AshlarError):
        ASRRegressor(task="gaussian", objective="energy", max_epochs=1).fit(x, y[:, 0])
    with pytest.raises(AshlarError):
        ASRRegressor(task="multivariate_gaussian", objective="variogram", max_epochs=1).fit(x, y)
    with pytest.raises(AshlarError):
        ASRModel(task="custom", head="multivariate_normal", objective="energy", point_fn=np.mean).fit(x, y)
    # A censored cell has no single value to set the draws against.
    label = observed.stack([observed.censored(y[:, 0], y[:, 0] > 1.0), observed.exact(y[:, 1]), observed.exact(y[:, 2])])
    with pytest.raises(AshlarError):
        ASRRegressor(task="multivariate_gaussian", objective="energy", max_epochs=1).fit(x, label)


@pytest.mark.slow
@pytest.mark.parametrize(
    "task, kwargs",
    [
        ("multivariate_gaussian", {}),
        ("multivariate_mixture", {"head_kwargs": {"components": 3}}),
        ("multivariate_flow", {}),
    ],
)
def test_each_joint_head_trains_on_a_blend_and_saves_it(task, kwargs, tmp_path):
    x, y = _shared_shock(800, 0)
    x_new, y_new = _shared_shock(1000, 1)
    objective = {"nll": 1.0, "energy": 1.0, "variogram": 0.5}
    model = ASRRegressor(task=task, objective=objective, max_epochs=6, random_state=0, **kwargs).fit(x, y)
    assert len(model.history_["val_loss"]) == 6 and np.all(np.isfinite(model.history_["val_loss"]))
    # Validation reads the same draws every epoch, so a refit gives the same numbers.
    again = ASRRegressor(task=task, objective=objective, max_epochs=6, random_state=0, **kwargs).fit(x, y)
    assert np.allclose(again.history_["val_loss"], model.history_["val_loss"], rtol=1e-4)
    log_density = np.asarray(model.log_prob(x_new, y_new))
    assert np.all(np.isfinite(log_density))
    restored = ASRModel.load(model.save(tmp_path / "scored"))
    assert restored.objective == objective
    assert np.allclose(np.asarray(restored.log_prob(x_new[:50], y_new[:50])), log_density[:50], atol=1e-4)
    # Exact cells in a joint label train the same way as plain values.
    label = observed.stack([observed.exact(y[:, column]) for column in range(3)])
    exact = ASRRegressor(task=task, objective=objective, max_epochs=1, random_state=0, **kwargs).fit(x, label)
    plain = ASRRegressor(task=task, objective=objective, max_epochs=1, random_state=0, **kwargs).fit(x, y)
    assert np.allclose(exact.history_["val_loss"], plain.history_["val_loss"], rtol=1e-4)


@pytest.mark.slow
def test_the_energy_score_trains_the_total_of_an_overconfident_flow():
    # On the shared shock, the flow fitted on its log density alone gives the sum an 80% interval that holds
    # 84%; adding the energy score brings it to 82% at the same log density (seed 0, 15 epochs).
    x, y = _shared_shock(2000, 0)
    x_new, y_new = _shared_shock(3000, 2)
    total = y_new.sum(axis=1)
    covered, density = {}, {}
    for name, objective in (("nll", None), ("blend", {"nll": 1.0, "energy": 1.0})):
        model = ASRRegressor(task="multivariate_flow", objective=objective, max_epochs=15, random_state=0).fit(x, y)
        interval = model.predict_interval(x_new, 0.8, target="sum", n_samples=500, calibrated=False)
        covered[name] = np.mean((total >= interval[:, 0]) & (total <= interval[:, 1]))
        density[name] = float(np.mean(np.asarray(model.log_prob(x_new, y_new))))
    assert abs(covered["blend"] - 0.8) < abs(covered["nll"] - 0.8)
    assert density["blend"] > density["nll"] - 0.1
