"""Entity effects: pooled offsets on the location, pooled embeddings, and updates from new rows."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ashlar import ASRModel, ASRRegressor
from ashlar.effects import (
    PREDICT_NODES,
    TAU_PRIOR,
    RandomEffects,
    check_random_effects,
    embedding_leaves,
    laplace_update,
    location_outputs,
    nesting,
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


def test_a_group_seen_only_in_the_validation_rows_is_known_after_fit():
    frame, y, _, _ = _stores(groups=12)
    # A temporal split holds out the last rows, so the last store's one row is only ever a validation row.
    frame = pd.concat([frame, frame.iloc[[0]].assign(store="late")], ignore_index=True)
    y = np.append(y, y.max() + 3.0)
    model = _small(random_effects={"store": "intercept"}, split="temporal").fit(frame, y)
    report = model.random_effects_["store"]
    late = report["categories"].index("late")
    tau = float(report["tau"][0])
    assert report["mean"][late, 0] > 0.0 and report["sd"][late, 0] < tau
    assert model.predict(frame.iloc[[-1]])[0] > model.predict(frame.iloc[[-1]].assign(store="new"))[0]


def test_fit_finishes_each_group_at_its_posterior_given_the_network():
    # The identity holds for any network, so a tiny one trained for four epochs checks it.
    frame, y, group, _ = _stores(groups=40, low=1, high=6, seed=3)
    model = _small(random_effects={"store": "intercept"}).fit(frame, y)
    report = model.random_effects_["store"]
    assert len(report["categories"]) == 40
    order = [int(name[1:]) for name in report["categories"]]
    # The conjugate posterior of each store's offset, from the network's offset-free prediction of its rows.
    population = frame.assign(store=None)
    location = np.asarray(model.predict(population))
    tau = float(report["tau"][0])
    noise = np.asarray(model.predict_dist(population).stddev()) ** 2 - tau**2
    precision = np.bincount(group, weights=1.0 / noise) + 1.0 / tau**2
    mean = np.bincount(group, weights=(y - location) / noise) / precision
    assert np.max(np.abs(report["mean"][:, 0] - mean[order])) < 1e-4
    assert np.max(np.abs(report["sd"][:, 0] - np.sqrt(1.0 / precision[order]))) < 1e-4


def test_with_context_is_the_conjugate_update_given_the_network():
    # As for fit's finishing step, the identity holds for any network, so a tiny one checks it.
    frame, y, _, _ = _stores(groups=12, seed=4)
    model = _small(random_effects={"store": "intercept"}).fit(frame, y)
    before = model.random_effects_["store"]
    rng = np.random.default_rng(5)
    context = pd.DataFrame({"x0": rng.normal(size=6), "x1": rng.normal(size=6), "store": ["s0"] * 3 + ["new"] * 3})
    target = rng.normal(size=6)
    after = model.with_context(context, target).random_effects_["store"]
    population = context.assign(store=None)
    location = np.asarray(model.predict(population))
    tau = float(before["tau"][0])
    noise = np.asarray(model.predict_dist(population).stddev()) ** 2 - tau**2
    seen = before["categories"].index("s0")
    # A seen store updates from its fitted posterior, a new one from the prior.
    for name, prior_mean, prior_var, rows in (
        ("s0", before["mean"][seen, 0], before["sd"][seen, 0] ** 2, slice(0, 3)),
        ("new", 0.0, tau**2, slice(3, 6)),
    ):
        precision = 1.0 / prior_var + np.sum(1.0 / noise[rows])
        mean = (prior_mean / prior_var + np.sum((target[rows] - location[rows]) / noise[rows])) / precision
        index = after["categories"].index(name)
        assert abs(after["mean"][index, 0] - mean) < 1e-6
        assert abs(after["sd"][index, 0] - np.sqrt(1.0 / precision)) < 1e-6


def test_an_offset_moves_every_row_alike_and_an_embedding_interacts():
    frame, y, _, _ = _stores(groups=12, seed=6)
    rows = frame.iloc[:30]
    for terms in (["intercept"], ["intercept", "embedding"]):
        model = _small(random_effects={"store": terms}).fit(frame, y)
        report = model.random_effects_["store"]
        offset = report["mean"][report["categories"].index("s3"), 0]
        gap = np.asarray(model.predict(rows.assign(store="s3"))) - np.asarray(model.predict(rows.assign(store="new")))
        if terms == ["intercept"]:
            # The store is out of the network, so its offset moves every row's location by the same amount.
            assert np.allclose(gap, offset, atol=1e-5)
        else:
            # Its embedding goes through the network, so its effect differs from row to row.
            assert np.std(gap) > 1e-4


def test_the_penalty_is_the_kl_of_each_seen_group_per_row():
    # tau's gradient comes from this closed form, which is what makes tau the spread of the groups.
    module = RandomEffects([3], [0], [[-1]], (0,), (1,), 0.25)
    module = eqx_set(module, module.means[0].at[1:, 0].set(jnp.asarray([0.5, -1.0, 2.0])))
    tau = float(module.taus()[0][0])
    sd = float(positive(module.raw_scales[0][1, 0]))
    mean = np.array([0.5, -1.0, 2.0])
    kl = np.log(tau / sd) + (sd**2 + mean**2) / (2.0 * tau**2) - 0.5
    assert np.isclose(float(module.penalty()), 0.25 * (kl.sum() + tau**2 / (2.0 * TAU_PRIOR**2)), rtol=1e-5)


def test_laplace_finds_the_mode_of_a_poisson_posterior():
    # A Poisson likelihood is not quadratic in the offset on the log mean, so this takes several Newton steps.
    rng = np.random.default_rng(1)
    groups = np.repeat(np.arange(3), [1, 5, 30])
    eta = rng.normal(0.5, 0.3, size=len(groups))
    counts = rng.poisson(np.exp(eta + np.array([0.8, -0.5, 0.3])[groups]))
    tau = 0.6

    def poisson(offset):
        rate = np.exp(eta + offset)
        return counts - rate, -rate

    mean, variance = laplace_update(poisson, np.zeros((3, 1)), np.full((3, 1), tau**2), np.ones((len(groups), 1)), groups, 3)
    rate = np.exp(eta + mean[groups, 0])
    assert np.max(np.abs(np.bincount(groups, weights=counts - rate) - mean[:, 0] / tau**2)) < 1e-8
    assert np.allclose(variance[:, 0], 1.0 / (np.bincount(groups, weights=rate) + 1.0 / tau**2), rtol=1e-8)


def test_fit_finishes_a_poisson_group_at_its_posterior_mode():
    rng = np.random.default_rng(2)
    group = np.repeat(np.arange(10), rng.integers(2, 12, size=10))
    x = rng.normal(size=len(group))
    counts = rng.poisson(np.exp(0.5 + 0.3 * x + rng.normal(0.0, 0.5, size=10)[group]))
    frame = pd.DataFrame({"x0": x, "store": [f"s{index}" for index in group]})
    model = _small(task="poisson", random_effects={"store": "intercept"}).fit(frame, counts)
    report = model.random_effects_["store"]
    tau = float(report["tau"][0])
    # An unknown store reads N(0, tau²) on nodes, so the network's own log mean is the population's less their lift.
    lift = np.log(np.mean(np.exp(tau * unit_nodes(PREDICT_NODES).astype(np.float64))))
    eta = np.log(np.asarray(model.predict_dist(frame.assign(store=None)).mean(), dtype=np.float64)) - lift
    order = [int(name[1:]) for name in report["categories"]]
    mean = np.zeros(10)
    mean[order] = report["mean"][:, 0]
    rate = np.exp(eta + mean[group])
    gradient = np.bincount(group, weights=counts - rate) - mean / tau**2
    assert np.max(np.abs(gradient) / (1.0 + np.bincount(group, weights=counts))) < 1e-5
