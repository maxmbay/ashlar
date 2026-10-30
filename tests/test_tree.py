import numpy as np
import pandas as pd
import pytest

import jax
import jax.numpy as jnp

from ashlar import ASRModel, ASROutcomeTree
from ashlar.errors import AshlarError
from ashlar.tree import OutcomeTree, empirical_edge_prior, tree_objective

SPEC = {"swing": {"contact": ["in_play", "foul"], "miss": None}, "take": ["called_strike", "ball"]}


def _small(**kwargs):
    params = dict(
        backbone_kwargs={"n_blocks": 1, "d_block": 16, "dropout": 0.0},
        max_epochs=8,
        patience=3,
        batch_size=64,
        learning_rate=1e-2,
    )
    params.update(kwargs)
    return ASROutcomeTree(SPEC, **params)


def _data(n=800, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    swing = rng.random(n) < 1 / (1 + np.exp(-2 * X.a))
    contact = rng.random(n) < 1 / (1 + np.exp(-2 * X.b))
    in_play = rng.random(n) < 0.4
    strike = rng.random(n) < 0.3
    y = np.where(
        swing,
        np.where(contact, np.where(in_play, "in_play", "foul"), "miss"),
        np.where(strike, "called_strike", "ball"),
    ).astype(object)
    return X, y


def test_tree_parses_paths_and_leaves():
    tree = OutcomeTree(SPEC)
    assert tree.names[0] is None
    assert tree.leaf_names == ["in_play", "foul", "miss", "called_strike", "ball"]
    assert tree.path("foul") == ["swing", "contact", "foul"]
    assert tree.n_edges == 8
    assert [tree.names[c] for c in tree.children[tree.node("swing")]] == ["contact", "miss"]


def test_tree_rejects_bad_specs():
    with pytest.raises(AshlarError):
        OutcomeTree({"only": None})
    with pytest.raises(AshlarError):
        OutcomeTree({"a": ["x", "y"], "b": ["x", "z"]})
    with pytest.raises(AshlarError):
        OutcomeTree({"a": None, "b": ["single"]})


def test_node_log_probabilities_are_products_of_branches():
    tree = OutcomeTree(SPEC)
    logits = jax.random.normal(jax.random.PRNGKey(0), (4, 3, tree.n_edges))
    log_p = tree.node_log_probabilities(logits)
    p = np.asarray(jnp.exp(log_p))
    np.testing.assert_allclose(p[..., 0], 1.0, rtol=1e-6)
    np.testing.assert_allclose(p[..., list(tree.leaves)].sum(-1), 1.0, rtol=1e-5)
    for node in tree.internal:
        kids = list(tree.children[node])
        np.testing.assert_allclose(p[..., kids].sum(-1), p[..., node], rtol=1e-5)


def test_partial_label_loss_is_the_internal_node_log_probability():
    tree = OutcomeTree(SPEC)
    logits = jax.random.normal(jax.random.PRNGKey(1), (2, 1, tree.n_edges))
    log_p = tree.node_log_probabilities(logits)
    swing = tree.node("swing")
    loss = tree_objective(log_p, jnp.asarray([swing, swing]), jnp.ones(2))
    np.testing.assert_allclose(float(loss), -float(jnp.mean(log_p[:, 0, swing])), rtol=1e-6)
    grads = jax.grad(lambda z: tree_objective(tree.node_log_probabilities(z), jnp.asarray([swing, 0]), jnp.ones(2)))(logits)
    assert bool(jnp.isfinite(grads).all())
    # A swing label says nothing about contact versus miss or the take split.
    below = [e for e, parent in enumerate(tree.edge_parent) if parent != 0]
    np.testing.assert_allclose(np.asarray(grads)[..., below], 0.0, atol=1e-7)


def test_empirical_edge_prior_counts_partial_labels_on_the_path():
    tree = OutcomeTree(SPEC)
    nodes = tree.encode(["in_play", "swing", "ball", "ball"])
    prior = empirical_edge_prior(tree, nodes, np.ones(4), alpha=0.0)
    # Root split: two swings, two takes.
    assert prior[:2] == [0.5, 0.5]


def test_fit_predict_conditionals_and_partial_labels():
    X, y = _data()
    y[:100] = "swing"
    y[100:120] = None
    model = _small().fit(X, y)
    proba = model.predict_proba(X)
    assert list(model.classes_) == OutcomeTree(SPEC).leaf_names
    np.testing.assert_allclose(proba.sum(1), 1.0, rtol=1e-5)
    nodes = model.predict_node_proba(X)
    tree = model.tree_
    swing = model.predict_conditional(X)[:, 0]
    contact = model.predict_conditional(X, "swing")[:, 0]
    in_play = model.predict_conditional(X, "contact")[:, 0]
    np.testing.assert_allclose(swing * contact * in_play, proba[:, 0], rtol=1e-4)
    np.testing.assert_allclose(nodes[:, tree.node("swing")], swing, rtol=1e-5)
    assert model.conditional_names("swing") == ["contact", "miss"]
    assert np.corrcoef(swing, X.a)[0, 1] > 0.8
    assert np.corrcoef(contact, X.b)[0, 1] > 0.8
    assert set(model.predict(X)) <= set(model.classes_)
    assert 0.0 <= model.score(X, y) <= 1.0


def test_tree_save_load_round_trip(tmp_path):
    X, y = _data(300)
    model = _small(max_epochs=2).fit(X, y)
    model.save(tmp_path)
    loaded = ASRModel.load(tmp_path)
    np.testing.assert_allclose(loaded.predict_proba(X), model.predict_proba(X), rtol=1e-6)
    assert loaded.tree == OutcomeTree(SPEC).spec


def test_tree_config_errors():
    with pytest.raises(AshlarError):
        ASRModel(task="tree").fit(pd.DataFrame({"a": [0.0, 1.0]}), ["x", "y"])
    with pytest.raises(AshlarError):
        ASRModel(task="classification", tree=SPEC).fit(pd.DataFrame({"a": [0.0, 1.0]}), ["x", "y"])
    X, y = _data(50)
    y[0] = "not_a_node"
    with pytest.raises(AshlarError):
        _small(max_epochs=1).fit(X, y)


def test_tree_is_a_sklearn_classifier():
    from sklearn.base import clone, is_classifier

    model = _small()
    assert is_classifier(model)
    assert clone(model).get_params()["tree"] == SPEC
