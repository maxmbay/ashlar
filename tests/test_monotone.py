"""Monotone constraints and the quantile_spline head: every quantile moves one way with a constrained feature."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ashlar import ASRRegressor
from ashlar.distributions import QuantileSpline
from ashlar.errors import AshlarError
from ashlar.monotone import MonotoneTerm, check_monotone, output_signs


def _spline(seed=0, rows=2):
    raw = jax.random.normal(jax.random.PRNGKey(seed), (rows, 4))
    return QuantileSpline(jnp.linspace(0.0, 1.0, rows), jax.nn.softplus(raw), jax.nn.softplus(-raw))


def test_equal_slopes_are_a_logistic():
    slope = np.sqrt(3.0) / np.pi
    spline = QuantileSpline(jnp.asarray([0.3, 1.0]), jnp.full((2, 4), slope), jnp.full((2, 4), slope))
    value = jnp.asarray([[-3.0], [0.2], [2.5], [5.0]])
    z = (np.asarray(value) - np.asarray([0.3, 1.0])) / slope
    expected = -z - 2.0 * np.log1p(np.exp(-z)) - np.log(slope)
    assert np.allclose(spline.log_prob(value), expected, atol=1e-5)
    assert np.allclose(spline.cdf(value), 1.0 / (1.0 + np.exp(-z)), atol=1e-6)
    assert np.allclose(spline.mean(), [0.3, 1.0], atol=1e-5)
    assert np.allclose(spline.variance(), 1.0, atol=5e-3)


def test_the_spline_is_a_distribution():
    spline = _spline()
    grid = jnp.linspace(-40.0, 40.0, 200_001)[:, None]
    mass = np.trapezoid(np.exp(np.asarray(spline.log_prob(grid))), np.asarray(grid[:, 0]), axis=0)
    assert np.allclose(mass, 1.0, atol=2e-3)
    levels = jnp.asarray([[0.001], [0.2], [0.5], [0.8], [0.999]])
    assert np.allclose(spline.cdf(spline.quantile(levels)), levels, atol=1e-5)
    draws = spline.sample(jax.random.PRNGKey(1), (200_000,))
    assert np.allclose(draws.mean(axis=0), spline.mean(), atol=0.02)
    assert np.allclose(draws.var(axis=0), spline.variance(), rtol=0.03)
    # The log density is smooth enough to train: finite gradients in every parameter.
    grads = jax.grad(lambda d: d.log_prob(jnp.asarray([0.4, -2.0])).sum())(spline)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(grads))


def test_a_quantile_rises_with_the_median_and_upper_slopes_and_falls_with_lower():
    spline = _spline(rows=1)
    levels = jnp.asarray([[0.02], [0.3], [0.5], [0.7], [0.98]])
    base = np.asarray(spline.quantile(levels))
    up = QuantileSpline(spline.median + 0.1, spline.upper * 1.2, spline.lower * 0.8)
    assert np.all(np.asarray(up.quantile(levels)) >= base)


def test_the_monotone_term_never_falls():
    term = MonotoneTerm(np.tile(np.linspace(-2, 2, 9), (2, 1)), 3, [0, 2], [1, -1], output_signs("quantile_spline", 9))
    leaves, tree = jax.tree.flatten(term)
    keys = jax.random.split(jax.random.PRNGKey(0), len(leaves))
    term = jax.tree.unflatten(tree, [leaf if leaf.ndim == 2 else leaf + jax.random.normal(k, leaf.shape) for leaf, k in zip(leaves, keys)])
    z = jnp.zeros((200, 3)).at[:, 0].set(jnp.linspace(-4, 4, 200)).at[:, 2].set(jnp.linspace(-4, 4, 200))
    out = np.asarray(term(z))
    signs = np.asarray(output_signs("quantile_spline", 9))
    # Feature 0 rises and feature 2 falls together here, so check each alone.
    only_first = np.asarray(term(jnp.zeros((200, 3)).at[:, 0].set(jnp.linspace(-4, 4, 200))))
    assert np.all(np.diff(only_first, axis=0) * signs >= -1e-6)
    only_last = np.asarray(term(jnp.zeros((200, 3)).at[:, 2].set(jnp.linspace(-4, 4, 200))))
    assert np.all(np.diff(only_last, axis=0) * signs <= 1e-6)
    assert out.shape == (200, 3, 9)
    assert np.allclose(term.without(z)[:, [0, 2]], 0.0)


def _table(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    frame = pd.DataFrame(x, columns=["price", "quality", "other"])
    frame["kind"] = np.where(rng.random(n) < 0.5, "a", "b")
    y = -0.8 * x[:, 0] + np.sin(2 * x[:, 1]) + np.exp(-0.3 * x[:, 0]) * rng.normal(size=n)
    return frame, y


@pytest.mark.parametrize("task", ["gaussian", "student_t", "quantile_spline", "negative_binomial", "regression"])
def test_every_quantile_moves_one_way(task, tmp_path):
    frame, y = _table()
    if task == "negative_binomial":
        y = np.round(np.exp(0.5 - 0.3 * frame["price"].to_numpy()) * 2.0)
    model = ASRRegressor(task=task, monotone={"price": -1}, max_epochs=5, random_state=0).fit(frame, y)
    rows = frame.iloc[:50]
    series = []
    for value in np.linspace(-3.5, 3.5, 15):
        grid = rows.copy()
        grid["price"] = value
        series.append(model.predict(grid)[:, None] if task == "regression" else model.predict_quantiles(grid, [0.05, 0.5, 0.95]))
    assert np.all(np.diff(np.stack(series), axis=0) <= 1e-5)
    loaded = ASRRegressor.load(model.save(tmp_path / task))
    assert np.allclose(loaded.predict(rows), model.predict(rows), atol=1e-5)
    assert loaded.monotone == {"price": -1}


def test_monotone_refuses_what_it_cannot_constrain():
    frame, y = _table(200)
    for bad in ({"price": 2}, {"price": True}, {}, ["price"]):
        with pytest.raises(AshlarError):
            check_monotone(bad)
    with pytest.raises(AshlarError, match="one way"):
        ASRRegressor(task="mixture", monotone={"price": 1}).fit(frame, y)
    with pytest.raises(AshlarError, match="numeric"):
        ASRRegressor(task="gaussian", monotone={"kind": 1}, max_epochs=1).fit(frame, y)
    with pytest.raises(AshlarError, match="numeric"):
        ASRRegressor(task="gaussian", monotone={"missing": 1}, max_epochs=1).fit(frame, y)
    with pytest.raises(AshlarError, match="cure"):
        ASRRegressor(task="gaussian", monotone={"price": 1}, head_kwargs={"cure": True}).fit(frame, y)


def test_the_quantile_spline_head_trains_and_scores():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(800, 2)).astype(np.float32)
    y = np.exp(0.5 * x[:, 0] + 0.4 * rng.normal(size=800))
    model = ASRRegressor(task="quantile_spline", max_epochs=10, random_state=0).fit(x, y)
    assert np.isfinite(model.log_prob(x, y)).all()
    quantiles = model.predict_quantiles(x[:5], [0.1, 0.5, 0.9])
    assert np.all(np.diff(quantiles, axis=1) > 0)
