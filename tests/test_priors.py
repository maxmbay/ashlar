import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ashlar import DEFAULT_RECIPE, ASRRegressor
from ashlar.errors import AshlarError
from ashlar.priors import (
    BART_DEPTH_COEF,
    BART_FEATURE_COEF,
    BART_GATE_COEF,
    BART_LEARNING_RATE,
    BART_TREES,
    GRADIENT_EMBED_DIM,
    GRADIENT_EMBED_SCALE,
    GRADIENT_LAMBDA,
    GRADIENT_LEARNING_RATE,
    VARIATIONAL_BINS,
    VARIATIONAL_LEARNING_RATE,
    ASRBartPriorEnsemble,
    ASRGradientSparseNet,
    ASRVariationalPairwise,
    _category_indices,
    _category_vocabs,
    _embedded_features,
    _embedded_loss,
    _embedded_multi_loss,
    _gradient_columns,
    _init_embedded,
    _init_sparse_multi,
    _multi_logits,
    _multi_loss,
    _sparse_logits,
)


def test_residual_default_stays_a_regressor():
    model = ASRRegressor()
    assert model.task == "regression"
    assert model.standardize_target is True
    assert model.backbone == "residual_mlp"
    assert DEFAULT_RECIPE["task"] == "regression"
    assert DEFAULT_RECIPE["standardize_target"] is True
    assert BART_TREES == 40
    assert BART_LEARNING_RATE == 0.05
    assert BART_FEATURE_COEF == 1e-3
    assert BART_GATE_COEF == 1e-3
    assert BART_DEPTH_COEF == 0.1
    assert VARIATIONAL_BINS == 4
    assert VARIATIONAL_LEARNING_RATE == 1e-2
    assert GRADIENT_LAMBDA == 1e-2
    assert GRADIENT_LEARNING_RATE == 1e-3
    assert GRADIENT_EMBED_DIM == 8
    assert GRADIENT_EMBED_SCALE == 0.02
    assert DEFAULT_RECIPE["categorical_embedding_dim"] == 8


def test_priors_learn_a_conjunction():
    train_x, train_y = _conjunction(900, 0)
    fresh_x, fresh_y = _conjunction(300, 1)
    bart = ASRBartPriorEnsemble(random_state=0, max_epochs=35, patience=12).fit(train_x, train_y, "classification")
    variational = ASRVariationalPairwise(random_state=0, max_epochs=25, patience=12).fit(train_x, train_y, "classification")
    gradient = ASRGradientSparseNet(random_state=0, max_epochs=25, patience=12).fit(train_x, train_y, "classification")
    assert float(np.mean(bart.predict(fresh_x) == fresh_y)) > 0.9
    assert float(np.mean(variational.predict(fresh_x) == fresh_y)) > 0.9
    assert float(np.mean(gradient.predict(fresh_x) == fresh_y)) > 0.9
    assert bart.feature_peak_ > 0.5
    assert bart.gate_margin_ < 0.35

    labels = np.where(train_y == 1, "UP", "DOWN")
    named = ASRBartPriorEnsemble(random_state=0, max_epochs=8, patience=4).fit(train_x, labels, "classification")
    assert set(np.unique(named.predict(fresh_x[:20]))).issubset({"DOWN", "UP"})


def test_gradient_penalty_handles_more_than_two_classes():
    rows, labels = _three_class(180, 0)
    model = ASRGradientSparseNet(random_state=0, max_epochs=20, patience=8).fit(rows, labels, "classification")
    fresh_rows, fresh_labels = _three_class(90, 1)
    assert model.n_classes_ == 3
    assert model.model_.w3.shape == (128, 3)
    assert float(np.mean(model.predict(fresh_rows) == fresh_labels)) > 0.9

    names = np.array(["red", "green", "blue"])[labels]
    named = ASRGradientSparseNet(random_state=0, max_epochs=12, patience=6).fit(rows, names, "classification")
    assert set(np.unique(named.predict(fresh_rows[:20]))).issubset({"red", "green", "blue"})

    binary_rows, binary_labels = _conjunction(80, 0)
    binary = ASRGradientSparseNet(random_state=0, max_epochs=1, patience=1).fit(binary_rows, binary_labels, "classification")
    assert binary.n_classes_ == 2
    assert binary.model_.w3.shape == (128, 1)
    assert binary.model_.embeddings == ()
    assert binary.categorical_ is False

    try:
        ASRGradientSparseNet(random_state=0, max_epochs=1, patience=1).fit(rows[:10], np.zeros(10), "classification")
    except AshlarError:
        pass
    else:
        raise AssertionError("one class was accepted")

    net = _init_sparse_multi(jax.random.PRNGKey(0), 4, 3)
    batch = jnp.asarray(np.random.default_rng(0).normal(size=(8, 4)), dtype=jnp.float32)
    target = jnp.asarray([0, 1, 2, 0, 1, 2, 1, 0], dtype=jnp.int32)
    loss, grads = eqx.filter_value_and_grad(_multi_loss)(net, batch, target)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    logits = _multi_logits(net, batch)
    likelihood = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target))

    def one_logits(row):
        return _multi_logits(net, row[None, :])[0]

    jacobian = jax.vmap(jax.jacobian(one_logits))(batch)
    penalty = GRADIENT_LAMBDA * jnp.mean(jnp.abs(jacobian))
    assert np.isfinite(float(loss))
    assert float(jnp.mean(jnp.abs(jacobian))) > 0.0
    assert abs(float(loss) - float(likelihood + penalty)) < 1e-5


def test_gradient_penalty_learns_category_embeddings():
    color = np.array(["red", "blue", None, "red", "blue"], dtype=object)
    roles = ("categorical",)
    vocabs = _category_vocabs([color], ["color"], roles, np.arange(3))
    assert vocabs == (("red", "blue"),)
    indices = _category_indices([color], roles, vocabs)
    assert indices[:, 0].tolist() == [1, 2, 0, 1, 2]
    unseen = _category_indices([np.array(["green", None], dtype=object)], roles, vocabs)
    assert unseen[:, 0].tolist() == [0, 0]

    mixed = {
        "code": np.array([0, 1, 0, 1], dtype=np.int64),
        "flag": np.array([True, False, True, False]),
        "color": np.array(["red", "blue", "red", "blue"], dtype=object),
    }
    _names, _columns, mixed_roles = _gradient_columns(mixed)
    assert mixed_roles == ("numeric", "numeric", "categorical")

    vocabs = (("red", "blue"),)
    net = _init_embedded(jax.random.PRNGKey(0), 1 + GRADIENT_EMBED_DIM, vocabs, 2, False)
    assert net.w3.shape == (128, 1)
    assert net.embeddings[0].shape == (3, 8)
    numeric = jnp.asarray(np.random.default_rng(0).normal(size=(4, 1)), dtype=jnp.float32)
    cats = jnp.asarray([[1], [2], [0], [1]], dtype=jnp.int32)
    target = jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    loss, grads = eqx.filter_value_and_grad(_embedded_loss)(net, numeric, cats, target)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert float(jnp.sum(jnp.abs(grads.embeddings[0]))) > 0.0
    features = _embedded_features(net, numeric, cats)
    logit = _sparse_logits(net, features)
    likelihood = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, target))

    def one_logit(row):
        return _sparse_logits(net, row[None, :])[0]

    gradient = jax.vmap(jax.grad(one_logit))(features)
    penalty = GRADIENT_LAMBDA * jnp.mean(jnp.abs(gradient))
    assert np.isfinite(float(loss))
    assert float(jnp.mean(jnp.abs(gradient))) > 0.0
    assert abs(float(loss) - float(likelihood + penalty)) < 1e-5

    multi = _init_embedded(jax.random.PRNGKey(1), GRADIENT_EMBED_DIM, (("a", "b", "c"),), 3, True)
    assert multi.w3.shape == (128, 3)
    assert multi.embeddings[0].shape == (4, 8)
    assert multi.w1.shape == (8, 128)
    multi_index = jnp.asarray([[1], [2], [3], [0]], dtype=jnp.int32)
    multi_target = jnp.asarray([0, 1, 2, 1], dtype=jnp.int32)
    empty = jnp.zeros((4, 0), dtype=jnp.float32)
    multi_loss, multi_grads = eqx.filter_value_and_grad(_embedded_multi_loss)(multi, empty, multi_index, multi_target)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(multi_grads))
    assert float(jnp.sum(jnp.abs(multi_grads.embeddings[0]))) > 0.0
    multi_features = _embedded_features(multi, empty, multi_index)
    multi_logits = _multi_logits(multi, multi_features)
    multi_likelihood = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(multi_logits, multi_target))

    def one_logits(row):
        return _multi_logits(multi, row[None, :])[0]

    jacobian = jax.vmap(jax.jacobian(one_logits))(multi_features)
    multi_penalty = GRADIENT_LAMBDA * jnp.mean(jnp.abs(jacobian))
    assert np.isfinite(np.asarray(jacobian)).all()
    assert abs(float(multi_loss) - float(multi_likelihood + multi_penalty)) < 1e-5

    train, train_y = _color_frame(360, 0)
    fresh, fresh_y = _color_frame(120, 1)
    model = ASRGradientSparseNet(random_state=0, max_epochs=30, patience=12).fit(train, train_y, "classification")
    assert model.categorical_ is True
    assert model.model_.embeddings[0].shape == (3, 8)
    assert model.model_.w1.shape == (1 + 8, 128)
    assert model.model_.w3.shape == (128, 1)
    assert float(np.mean(model.predict(fresh) == fresh_y)) > 0.8
    missing = {"x": np.array([0.2]), "color": np.array([None], dtype=object)}
    unknown = {"x": np.array([0.2]), "color": np.array(["green"], dtype=object)}
    assert np.allclose(model._categorical_logits(missing), model._categorical_logits(unknown))
    assert model.predict(unknown).shape == (1,)

    only_color, only_y = _color_frame(80, 2)
    bare = {key: value for key, value in only_color.items() if key == "color"}, only_y
    smoke = ASRGradientSparseNet(random_state=0, max_epochs=1, patience=1).fit(bare[0], bare[1], "classification")
    assert smoke.model_.w1.shape == (8, 128)
    assert smoke.mean_.shape == (0,)
    assert smoke.predict({"color": np.array(["green"], dtype=object)}).shape == (1,)

    groups, group_y = _three_colors(90, 3)
    grouped = ASRGradientSparseNet(random_state=0, max_epochs=20, patience=8).fit(groups, group_y, "classification")
    fresh_groups, fresh_group_y = _three_colors(40, 4)
    assert grouped.n_classes_ == 3
    assert grouped.model_.w3.shape == (128, 3)
    assert grouped.model_.embeddings[0].shape == (4, 8)
    assert float(np.mean(grouped.predict(fresh_groups) == fresh_group_y)) > 0.8


def _color_frame(n_rows: int, seed: int):
    generator = np.random.default_rng(seed)
    color = np.array(["red", "blue"], dtype=object)[generator.integers(0, 2, size=n_rows)]
    labels = np.where(color == "red", "UP", "DOWN")
    return {"x": generator.normal(size=n_rows), "color": color}, labels


def _three_colors(n_per_class: int, seed: int):
    generator = np.random.default_rng(seed)
    names = np.array(["red", "green", "blue"], dtype=object)
    labels = np.repeat(names, n_per_class)
    order = generator.permutation(len(labels))
    labels = labels[order]
    color = labels.copy()
    frame = {"x": generator.normal(size=len(labels)), "color": color}
    return frame, labels


def _three_class(n_per_class: int, seed: int):
    generator = np.random.default_rng(seed)
    centers = np.array([[-2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    labels = np.repeat(np.arange(3, dtype=np.int32), n_per_class)
    rows = centers[labels] + 0.35 * generator.normal(size=(len(labels), 2))
    noise = generator.normal(size=(len(labels), 3))
    return np.concatenate([rows, noise], axis=1), labels


def _conjunction(n_rows: int, seed: int):
    generator = np.random.default_rng(seed)
    rows = generator.normal(size=(n_rows, 6))
    labels = ((rows[:, 0] > 0) & (rows[:, 1] > 0)).astype(np.int32)
    return rows, labels
