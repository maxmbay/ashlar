"""A chain with pooled groups and monotone features: offsets per column, directions per column, rows that end early."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from micro import MICRO

from ashlar import ASRRegressor, observed
from ashlar.chain import (
    Chain,
    chain_columns,
    chain_monotone_signs,
    chain_targets,
    resolve_columns,
)
from ashlar.effects import RandomEffects, target_nodes
from ashlar.errors import AshlarError

INF = np.inf
COLUMNS = [{"kind": "categorical", "classes": 2}, {"kind": "gaussian", "monotone": {"rate": -1}}, "hurdle"]


def _loans(n: int = 400, seed: int = 0):
    """A cause, a log time, and a loss, with lenders that shift the cause's odds and the time."""
    rng = np.random.default_rng(seed)
    lender = rng.integers(0, 12, n)
    shift = rng.normal(0.0, 1.0, 12)
    rate = rng.normal(size=n)
    cause = (rng.random(n) < 1.0 / (1.0 + np.exp(-(rate + shift[lender])))).astype(float)
    time = 1.0 - 0.4 * rate + 0.5 * cause + 0.3 * shift[lender] + 0.3 * rng.normal(size=n)
    loss = np.where(cause == 1, np.exp(rng.normal(0.3 * rate, 0.5)), 0.0)
    X = {"rate": rate, "lender": np.array([f"L{i}" for i in lender], dtype=object)}
    return X, np.stack([cause, time, loss], axis=1)


def _fit(X, y, **kwargs):
    return ASRRegressor(task="chain", head_kwargs={"columns": COLUMNS}, categorical=["lender"], max_epochs=3, **MICRO, **kwargs).fit(X, y)


def test_targets_and_signs_follow_each_column():
    columns = resolve_columns(chain_columns(["categorical", "gaussian", "hurdle", "histogram"]), np.array([[2.0, 0.1, 1.0, 0.2]]))
    width = max(column.width for column in columns)
    targets, labels = chain_targets(columns)
    # A class column has one target per class after the first; the hurdle moves its amount; the histogram has none.
    assert labels == [(0, 1), (0, 2), (1, None), (2, None)]
    assert targets == [(1,), (2,), (width,), (2 * width + 1,)]
    two = resolve_columns(chain_columns([{"kind": "categorical", "classes": 2}, {"kind": "gaussian", "monotone": {"rate": -1}}]), np.zeros((1, 2)))
    signs = np.asarray(chain_monotone_signs(two, 4, {"rate": 1, "size": -1}))
    width = max(column.width for column in two)
    assert signs.shape == (2, 2 * (width + 4))
    # "rate" raises the second class's odds and, overridden, lowers the time; "size" keeps its own direction.
    assert signs[0, 1] == 1 and signs[0, width] == -1 and signs[1, 1] == 1 and signs[1, width] == 1
    assert np.count_nonzero(signs) == 4
    with pytest.raises(AshlarError, match="classes"):
        chain_monotone_signs(columns[:2], 4, {"rate": 1})
    with pytest.raises(AshlarError, match="does not name"):
        chain_monotone_signs(two, 4, {"size": 1})


def test_offsets_move_each_target_on_its_own_nodes():
    module = RandomEffects([3], [0], [[-1]], [0, 2], None, 1.0, targets=[(0,), (2,)])
    # Group 1 has offsets +1 on the first target and -2 on the second; group 2 sits at 0.
    module = eqx.tree_at(lambda tree: tree.means[0], module, module.means[0].at[1].set(jnp.array([1.0, -2.0])))
    groups = jnp.array([[1], [2]])
    mean, _ = module.offsets(groups, jnp.zeros((2, 0)))
    assert mean.shape == (2, 2) and np.allclose(mean[0], [1.0, -2.0]) and np.allclose(mean[1], 0.0)
    raw = jnp.zeros((2, 1, 4))
    moved = module(raw, groups, jnp.zeros((2, 0)), training=False)
    assert moved.shape == (2, 16, 4)
    # Output 1 and 3 are untouched; output 0 and 2 average to the group's offsets over the nodes.
    assert np.allclose(moved[..., 1], 0.0) and np.allclose(moved[..., 3], 0.0)
    assert np.allclose(moved[0, :, 0].mean(), 1.0, atol=1e-5) and np.allclose(moved[0, :, 2].mean(), -2.0, atol=1e-5)
    nodes = target_nodes(16, 2)
    assert not np.allclose(nodes[0], nodes[1]) and np.allclose(np.sort(nodes[0]), np.sort(nodes[1]))
    # A chain's shared weights repeat along the member axis with the nodes.
    out = module((raw, jnp.ones((1, 3))), groups, jnp.zeros((2, 0)), training=True)
    assert out[0].shape == (2, 8, 4) and out[1].shape == (8, 3)


def test_pooled_monotone_chain_fits_reports_and_saves(tmp_path):
    X, y = _loans()
    model = _fit(X, y, random_effects={"lender": ["intercept", "embedding"]}, monotone={"rate": 1})
    report = model.random_effects_["lender"]
    # Three targets: the second class's logit, the time's location, the loss's amount.
    assert report["tau"].shape == (1, 3) and report["mean"].shape == (12, 1, 3)
    values = model.log_prob(X, y)
    assert np.isfinite(values).all()
    loaded = ASRRegressor.load(model.save(tmp_path / "pooled"))
    assert np.allclose(loaded.log_prob(X, y), values, atol=1e-5)
    with pytest.raises(AshlarError, match="chain"):
        model.with_context({k: v[:5] for k, v in X.items()}, y[:5])


def test_monotone_chain_moves_each_column_its_own_way():
    X, y = _loans()
    model = _fit(X, y, monotone={"rate": 1})
    term = model.core_.monotone
    z = jnp.linspace(-3.0, 3.0, 7)[:, None]
    shapes = np.asarray(term(z))  # (rows, members, outputs) added to the chain's parameters
    width = max(column.width for column in model.chain_columns_)
    # The second class's odds rise with the rate, and the time's location falls, in every member.
    assert np.all(np.diff(shapes[:, :, 1], axis=0) >= 0) and np.all(np.diff(shapes[:, :, 1], axis=0).sum(axis=0) > 0)
    assert np.all(np.diff(shapes[:, :, width], axis=0) <= 0)
    # The hurdle's amount takes the model's direction; nothing else moves.
    assert np.all(np.diff(shapes[:, :, 2 * width + 1], axis=0) >= 0)
    moved = np.flatnonzero(np.abs(shapes).sum(axis=(0, 1)) > 0)
    assert set(moved) == {1, width, 2 * width + 1}
    # The trunk does not read the rate.
    assert np.allclose(term.without(z), 0.0)


def test_a_row_can_end_with_a_censored_time_and_an_unknown_loss():
    X, y = _loans()
    running = np.arange(len(y)) % 4 == 0
    label = observed.stack(
        [
            observed.interval(np.where(running, -INF, y[:, 0]), np.where(running, INF, y[:, 0])),
            observed.censored(np.where(running, y[:, 1] - 0.5, y[:, 1]), running),
            observed.interval(np.where(running, -INF, y[:, 2]), np.where(running, INF, y[:, 2])),
        ]
    )
    model = _fit(X, label)
    scores = model.log_prob(X, label)
    assert np.isfinite(scores).all()
    # A running row scores some cause and a time past its censoring point, so more than an exact row's density.
    assert scores[running].mean() > scores[~running].mean()
    # A known loss after a censored time is still refused.
    bad = observed.stack([observed.exact(y[:, 0]), observed.censored(y[:, 1], running), observed.exact(y[:, 2])])
    with pytest.raises(AshlarError, match="before the last column"):
        _fit(X, bad)


def test_box_likelihood_of_an_early_end_integrates_the_rest():
    columns = resolve_columns(chain_columns(["categorical", "gaussian", "hurdle"]), np.array([[1.0, 0.2, 1.0], [0.0, -0.1, 0.0]]))
    from ashlar.chain import ChainHead

    head = ChainHead(5, 2, n_events=3, n_classes=2, key=jax.random.PRNGKey(0), columns=columns)
    leaves, tree = jax.tree.flatten(head)
    keys = jax.random.split(jax.random.PRNGKey(1), len(leaves))
    head = jax.tree.unflatten(tree, [leaf + 0.3 * jax.random.normal(key, leaf.shape) for leaf, key in zip(leaves, keys)])
    chain = Chain(*head(jax.random.normal(jax.random.PRNGKey(2), (3, 2, 5)), key=None), columns)
    t = 0.4
    lower = jnp.array([[[-INF, t, -INF]]] * 3)
    upper = jnp.array([[[INF, INF, INF]]] * 3)
    got = chain.log_box(lower, upper)
    # Σ_c p(c) P(T > t | c): the loss integrates to one.
    expected = 0.0
    for code in (0.0, 1.0):
        values = jnp.full((3, 2, 3), 0.0).at[..., 0].set(code)
        logit = chain.conditional(0, values).log_prob(jnp.full((3, 2), code))
        tail = chain.conditional(1, values).log_interval(jnp.full((3, 2), t), jnp.full((3, 2), INF))
        expected = expected + jnp.exp(logit + tail)
    assert np.allclose(got, np.log(expected), atol=1e-4)
