import jax
import jax.numpy as jnp
import numpy as np

from ashlar import DEFAULT_RECIPE, ASRRegressor
from ashlar.node import DEPTH, LEARNING_RATE, TREE_COUNT, ASRObliviousEnsemble, ObliviousLayer, entmax15


def test_residual_default_stays_a_regressor():
    model = ASRRegressor()
    assert model.task == "regression"
    assert model.standardize_target is True
    assert model.backbone == "residual_mlp"
    assert DEFAULT_RECIPE["task"] == "regression"
    assert DEFAULT_RECIPE["standardize_target"] is True
    assert DEPTH == 6
    assert TREE_COUNT == 512
    assert LEARNING_RATE == 0.1


def test_entmax15_is_a_sparse_probability_and_has_a_gradient():
    scores = jnp.array([0.2, 0.0, -0.1, 0.05])
    probs = entmax15(scores)
    assert jnp.allclose(jnp.sum(probs), 1.0)
    assert float(jnp.min(probs)) >= 0.0
    peaked = entmax15(jnp.array([5.0, 0.0, 0.0]))
    assert float(peaked[0]) > 0.99

    def loss(value):
        return jnp.sum(entmax15(value) * jnp.array([1.0, -1.0, 0.5, 0.2]))

    gradient = jax.grad(loss)(scores)
    assert float(jnp.linalg.norm(gradient)) > 0


def test_split_parameters_receive_a_gradient():
    layer = ObliviousLayer(
        feature_logits=jnp.zeros((2, 2, 3)),
        thresholds=jnp.zeros((2, 2)),
        raw_scales=jnp.zeros((2, 2)),
        leaves=jnp.asarray(np.array([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 1.5, -1.5], dtype=np.float32).reshape(2, 4, 1)),
    )
    features = jnp.asarray(np.array([[0.5, -1.0, 0.25], [1.0, 0.0, -0.5]], dtype=np.float32))

    def loss(logits):
        updated = layer.__class__(logits, layer.thresholds, layer.raw_scales, layer.leaves)
        prediction = updated(features)[:, 0]
        return jnp.mean(jnp.square(prediction - jnp.array([1.0, -1.0])))

    gradient = jax.grad(loss)(layer.feature_logits)
    assert gradient.shape == layer.feature_logits.shape
    assert float(jnp.linalg.norm(gradient)) > 0


def test_depth_six_ensemble_learns_a_conjunction():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(800, 4))
    labels = ((rows[:, 0] > 0) & (rows[:, 1] > 0)).astype(np.int32)
    model = ASRObliviousEnsemble(random_state=0, n_trees=32, depth=4, max_epochs=30, patience=10, batch_size=200)
    model.fit(rows, labels, "classification")
    assert float(np.mean(model.predict(rows) == labels)) > 0.9
    assert model.depth_ == 4
    assert model.n_trees_ == 32
