"""Entity effects: pooled offsets on the location, pooled embeddings, and updates from new rows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ashlar import ASRModel, ASRRegressor
from ashlar.effects import (
    RandomEffects,
    check_random_effects,
    embedding_leaves,
    nesting,
    laplace_update,
    location_outputs,
    positive,
    unit_nodes,
)
from ashlar.errors import AshlarError


def _stores(groups=60, tau=0.7, sigma=0.5, low=4, high=30, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(low, high, size=groups)
    group = np.repeat(np.arange(groups), sizes)
    effect = rng.normal(0.0, tau, size=groups)
    x = rng.normal(size=(len(group), 2))
    y = np.sin(x[:, 0]) + 0.5 * x[:, 1] + effect[group] + rng.normal(0.0, sigma, size=len(group))
    frame = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "store": [f"s{index}" for index in group]})
    return frame, y, group, effect


def _small(task="gaussian", **kwargs):
    params = dict(
        task=task,
        backbone_kwargs={"n_blocks": 1, "d_block": 16, "dropout": 0.0},
        max_epochs=4,
        patience=4,
        batch_size=64,
        random_state=0,
    )
    params.update(kwargs)
    return ASRRegressor(**params)


def test_terms_are_checked_and_ordered():
    assert check_random_effects(None) == {}
    assert check_random_effects({"store": "intercept"}) == {"store": ("intercept",)}
    assert check_random_effects({"store": ["embedding", "price", "intercept"]}) == {
        "store": ("intercept", "price", "embedding")
    }
    for bad in ({}, {"store": []}, {"store": 3}, {"store": ["price", "price"]}, {"store": ["store"]}, ["store"]):
        with pytest.raises(AshlarError):
            check_random_effects(bad)
    nested = {"region": ["intercept", "embedding"], "store": {"terms": ["embedding", "intercept"], "within": "region"}}
    assert check_random_effects(nested)["store"] == ("intercept", "embedding")
    assert nesting(nested) == {"store": "region"}
    assert embedding_leaves(check_random_effects(nested), nesting(nested)) == ["store"]
    for bad in (
        {"store": {"terms": "intercept", "within": "region"}},
        {"region": "intercept", "store": {"terms": "embedding", "within": "region"}},
        {"a": {"terms": "intercept", "within": "b"}, "b": {"terms": "intercept", "within": "a"}},
        {"store": {"terms": "intercept", "inside": "region"}},
    ):
        with pytest.raises(AshlarError):
            check_random_effects(bad)


def test_nodes_and_outputs():
    nodes = unit_nodes(8)
    assert abs(float(nodes.mean())) < 1e-6 and abs(float(np.mean(nodes**2)) - 1.0) < 1e-5
    assert location_outputs("gaussian", 2) == ((0,), (1,))
    assert location_outputs("mixture", 15) == ((0, 3, 6, 9, 12), (1, 4, 7, 10, 13))
    assert location_outputs("regression", 1) == ((0,), ())
    assert location_outputs("hurdle", 3) == ((1,), None)
    assert location_outputs("poisson", 1) == ((0,), None)
    with pytest.raises(AshlarError):
        location_outputs("histogram", 16)


def test_offsets_read_the_posterior_for_seen_groups_and_the_prior_otherwise():
    module = RandomEffects([3], [0], [[-1, 0]], (0,), (1,), 0.01)
    module = jax.tree_util.tree_map(lambda leaf: leaf, module)
    means = module.means[0].at[1].set(jnp.asarray([0.4, -0.2])).at[2].set(jnp.asarray([-0.3, 0.1]))
    module = eqx_set(module, means)
    index = jnp.asarray([[1], [2], [0]], dtype=jnp.int32)
    z = jnp.asarray([[2.0], [1.0], [3.0]])
    mean, variance = module.offsets(index, z)
    sd = float(positive(module.raw_scales[0][1, 0]))
    tau = np.asarray(module.taus()[0])
    assert np.allclose(mean, [0.4 - 0.4, -0.3 + 0.1, 0.0], atol=1e-6)
    assert np.allclose(variance, [sd**2 * (1 + 4), sd**2 * 2, tau[0] ** 2 + 9 * tau[1] ** 2], atol=1e-5)
    raw = jnp.zeros((3, 2, 2))
    closed = module.shift(raw, mean, variance, None)
    assert closed.shape == (3, 2, 2)
    assert np.allclose(closed[..., 0], mean[:, None], atol=1e-6)
    assert np.allclose(positive(closed[..., 1]) ** 2, positive(jnp.zeros(())) ** 2 + variance[:, None], atol=1e-5)
    nodes = module.shift(raw, mean, variance, 8)
    assert nodes.shape == (3, 16, 2)
    assert np.allclose(nodes[..., 0].mean(axis=1), mean, atol=1e-5)
    assert np.allclose(nodes[..., 0].var(axis=1), variance, atol=1e-5)
    # At the start every posterior is its prior, so only tau's own prior is in the penalty.
    fresh = RandomEffects([3], [0], [[-1]], (0,), (1,), 0.5)
    tau0 = float(fresh.taus()[0][0])
    assert np.allclose(fresh.penalty(), 0.5 * tau0**2 / 2.0, atol=1e-6)


def eqx_set(module, means):
    import equinox as eqx

    return eqx.tree_at(lambda tree: tree.means, module, (means,))


def test_laplace_is_the_conjugate_update_for_a_normal():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(3), [1, 4, 20])
    sigma, tau = 0.5, 0.8
    residual = rng.normal(size=len(groups))

    def normal(offset):
        return (residual - offset) / sigma**2, np.full(len(offset), -1.0 / sigma**2)

    mean, variance = laplace_update(normal, np.zeros((3, 1)), np.full((3, 1), tau**2), np.ones((len(groups), 1)), groups, 3)
    counts = np.bincount(groups)
    precision = counts / sigma**2 + 1.0 / tau**2
    expected = np.bincount(groups, weights=residual) / sigma**2 / precision
    assert np.allclose(mean[:, 0], expected, atol=1e-8)
    assert np.allclose(variance[:, 0], 1.0 / precision, atol=1e-10)


def test_the_estimator_refuses_what_it_cannot_pool():
    frame, y, _, _ = _stores(groups=8)
    with pytest.raises(AshlarError, match="location"):
        ASRModel(task="classification", random_effects={"store": "intercept"}).fit(frame, y > 0)
    with pytest.raises(AshlarError, match="monotone"):
        _small(monotone={"x0": 1}, random_effects={"store": ["intercept", "x0"]}).fit(frame, y)
    with pytest.raises(AshlarError, match="numeric"):
        _small(random_effects={"store": ["intercept", "nope"]}).fit(frame, y)
    with pytest.raises(AshlarError, match="shared minibatches"):
        _small(share_training_batches=False, random_effects={"store": "intercept"}).fit(frame, y)


def test_a_small_fit_predicts_saves_and_takes_context(tmp_path):
    frame, y, _, _ = _stores(groups=12)
    model = _small(random_effects={"store": ["intercept", "x1", "embedding"]}).fit(frame, y)
    report = model.random_effects_["store"]
    assert report["terms"] == ["intercept", "x1"]
    assert report["mean"].shape == (12, 2) and report["sd"].shape == (12, 2) and report["tau"].shape == (2,)
    assert report["embedding"].shape == (12, 8)
    unseen = frame.iloc[:4].assign(store="new")
    assert np.isfinite(np.asarray(model.predict_dist(unseen).stddev())).all()
    missing = frame.iloc[:4].assign(store=None)
    assert np.allclose(model.predict(missing), model.predict(unseen), atol=1e-5)
    model.save(tmp_path / "pooled")
    loaded = ASRModel.load(tmp_path / "pooled")
    assert np.allclose(loaded.predict(frame.iloc[:20]), model.predict(frame.iloc[:20]), atol=1e-5)
    regions = frame.assign(region=[f"r{int(name[1:]) % 3}" for name in frame["store"]])
    nested = _small(
        random_effects={"region": ["intercept", "embedding"], "store": {"terms": ["intercept", "embedding"], "within": "region"}}
    ).fit(regions, y)
    assert nested.random_effects_["store"]["within"] == "region"
    assert nested.core_.random_effects.embedding_width == 8
    # An unseen store of a known region reads the region's embedding and offset, so regions still differ.
    fresh = regions.iloc[:2].assign(store="new", region=["r0", "r1"])
    assert not np.allclose(*np.asarray(nested.predict(fresh.iloc[[0, 0]].assign(region=["r0", "r1"]))))
    nested.save(tmp_path / "nested")
    assert np.allclose(ASRModel.load(tmp_path / "nested").predict(regions.iloc[:9]), nested.predict(regions.iloc[:9]), atol=1e-5)
    context = frame.iloc[:3].assign(store="new")
    adapted = model.with_context(context, y[:3])
    assert "new" in adapted.random_effects_["store"]["categories"]
    assert "new" not in model.random_effects_["store"]["categories"]
    assert not np.allclose(adapted.predict(unseen), model.predict(unseen))


@pytest.mark.slow
def test_the_fit_recovers_tau_and_shrinks_small_groups():
    frame, y, group, effect = _stores(groups=300, low=2, high=40)
    model = ASRRegressor(task="gaussian", random_effects={"store": "intercept"}, random_state=0, max_epochs=200).fit(frame, y)
    report = model.random_effects_["store"]
    assert abs(float(report["tau"][0]) - 0.7) < 0.1
    order = [int(name[1:]) for name in report["categories"]]
    estimate = report["mean"][:, 0]
    assert np.corrcoef(effect[order], estimate)[0, 1] > 0.9
    # Small groups are pulled in further than their raw residual means.
    counts = np.bincount(group)[order]
    assert np.mean(np.abs(estimate[counts <= 4])) < np.mean(np.abs(effect[order][counts <= 4]))
    unseen = frame.iloc[:50].assign(store="new")
    sd = np.asarray(model.predict_dist(unseen).stddev())
    assert abs(float(np.mean(sd)) - np.sqrt(0.5**2 + 0.7**2)) < 0.08


@pytest.mark.slow
def test_context_rows_give_a_new_group_the_conjugate_posterior():
    frame, y, group, effect = _stores(groups=300, low=8, high=40)
    train = group < 200
    model = ASRRegressor(task="gaussian", random_effects={"store": "intercept"}, random_state=0, max_epochs=200)
    model.fit(frame[train], y[train])
    first = np.zeros(len(group), dtype=bool)
    for store in range(200, 300):
        first[np.flatnonzero(group == store)[:3]] = True
    adapted = model.with_context(frame[first], y[first])
    report = adapted.random_effects_["store"]
    names = [f"s{store}" for store in range(200, 300)]
    rows = [report["categories"].index(name) for name in names]
    tau = float(report["tau"][0])
    conjugate = np.sqrt(1.0 / (3.0 / 0.5**2 + 1.0 / tau**2))
    assert abs(float(np.mean(report["sd"][rows, 0])) - conjugate) < 0.05
    assert np.corrcoef(effect[200:], report["mean"][rows, 0])[0, 1] > 0.85
    test = (~train) & (~first)
    before = np.sqrt(np.mean((model.predict(frame[test]) - y[test]) ** 2))
    after = np.sqrt(np.mean((adapted.predict(frame[test]) - y[test]) ** 2))
    assert after < 0.75 * before


@pytest.mark.slow
def test_a_pooled_embedding_learns_a_group_interaction():
    rng = np.random.default_rng(1)
    groups = 80
    group = np.repeat(np.arange(groups), 40)
    slope = rng.normal(0.0, 1.0, size=groups)
    x = rng.normal(size=(len(group), 2))
    y = slope[group] * np.tanh(2.0 * x[:, 0]) + 0.3 * x[:, 1] + rng.normal(0.0, 0.3, size=len(group))
    frame = pd.DataFrame({"x0": x[:, 0], "x1": x[:, 1], "store": [f"s{index}" for index in group]})
    train = rng.uniform(size=len(group)) < 0.75
    scores = {}
    for name, terms in (("offset", ["intercept"]), ("embedding", ["intercept", "embedding"])):
        model = ASRRegressor(task="gaussian", random_effects={"store": terms}, random_state=0, max_epochs=150)
        model.fit(frame[train], y[train])
        scores[name] = np.sqrt(np.mean((model.predict(frame[~train]) - y[~train]) ** 2))
    assert scores["embedding"] < 0.7 * scores["offset"]


@pytest.mark.slow
def test_a_poisson_intercept_recovers_tau_on_the_log_mean():
    rng = np.random.default_rng(2)
    groups = 200
    group = np.repeat(np.arange(groups), rng.integers(5, 40, size=groups))
    effect = rng.normal(0.0, 0.5, size=groups)
    x = rng.normal(size=(len(group), 1))
    counts = rng.poisson(np.exp(0.5 + 0.3 * x[:, 0] + effect[group]))
    frame = pd.DataFrame({"x0": x[:, 0], "store": [f"s{index}" for index in group]})
    model = ASRRegressor(task="poisson", random_effects={"store": "intercept"}, random_state=0, max_epochs=150)
    model.fit(frame, counts)
    assert abs(float(model.random_effects_["store"]["tau"][0]) - 0.5) < 0.1
