import jax
import jax.numpy as jnp
import numpy as np

from ashlar.errors import ShapeError, WeightError
from ashlar.layers import LinearBatchEnsemble, TabMBackbone, _scaling
from ashlar.objectives import regression_objective
from ashlar.predict import MeanParameterNormal, NormalMixture, mean_logits, mean_probabilities
from ashlar.weights import weighted_mean


def test_batch_weight_sum_matches_the_documented_reduction():
    per_example = jnp.array([1.0, 2.0, 3.0])
    weights = jnp.array([1.0, 0.0, 3.0])
    assert np.isclose(float(weighted_mean(per_example, weights)), 2.5)


def test_zero_weight_matches_dropping_the_row():
    raw = jnp.array([[[0.0], [2.0]], [[5.0], [5.0]]])
    y = jnp.array([0.0, 100.0])
    kept = regression_objective(raw[:1], y[:1], jnp.ones(1))
    with_zero = regression_objective(raw, y, jnp.array([1.0, 0.0]))
    assert np.isclose(float(kept), float(with_zero))


def test_negative_weight_is_rejected():
    from ashlar.weights import validate_sample_weight

    try:
        validate_sample_weight(np.array([1.0, -0.1]), 2)
    except WeightError:
        return
    raise AssertionError("negative weights were accepted")


def test_mixture_variance_is_not_parameter_averaging():
    loc = jnp.array([[[-5.0], [5.0]]], dtype=jnp.float32)
    scale = jnp.ones_like(loc)
    mixture = NormalMixture(loc=loc, scale=scale)
    averaged = MeanParameterNormal(loc=jnp.mean(loc, axis=1), scale=jnp.mean(scale, axis=1))
    assert np.isclose(float(mixture.variance()[0, 0]), 26.0)
    assert np.isclose(float(averaged.variance()[0, 0]), 1.0)
    assert np.isclose(float(mixture.mean()[0, 0]), 0.0)
    assert np.isclose(float(averaged.mean()[0, 0]), 0.0)


def test_event_axis_is_not_reduced_with_members():
    loc = jnp.array([[[-5.0, 0.0], [5.0, 0.0]]], dtype=jnp.float32)
    scale = jnp.ones_like(loc)
    variance = NormalMixture(loc=loc, scale=scale).variance()
    assert variance.shape == (1, 2)
    assert np.isclose(float(variance[0, 0]), 26.0)
    assert np.isclose(float(variance[0, 1]), 1.0)


def test_probability_average_differs_from_logit_average():
    logits = jnp.array([[[4.0, 0.0], [0.0, 1.0]]], dtype=jnp.float32)
    probabilities = mean_probabilities(logits)
    from_logits = mean_logits(logits)
    assert probabilities.shape == (1, 2)
    assert not np.allclose(probabilities, from_logits, atol=1e-3)


def test_scaling_chunks_share_one_draw():
    key = jax.random.PRNGKey(0)
    scaling = _scaling(key, (2, 5), "random-signs", [2, 3], jnp.float32)
    assert jnp.all(scaling[:, 0] == scaling[:, 1])
    assert jnp.all(scaling[:, 2] == scaling[:, 4])


def test_tabm_forward_matches_the_batch_ensemble_equation():
    key = jax.random.PRNGKey(7)
    backbone = TabMBackbone(
        3,
        key=key,
        n_blocks=1,
        d_block=2,
        dropout=0.0,
        k=2,
        start_scaling_init="random-signs",
        scaling_chunks=[1, 2],
    )
    layer = backbone.layers[0]
    x = jnp.array([[0.2, -0.4, 0.6], [1.0, 0.0, -1.0]], dtype=jnp.float32)
    actual = backbone(x, key=None)
    viewed = jnp.broadcast_to(x[:, None, :], (2, 2, 3))
    expected = ((viewed * layer.r) @ layer.weight.T) * layer.s + layer.bias
    expected = jax.nn.relu(expected)
    assert actual.shape == (2, 2, 2)
    assert np.allclose(actual, expected, atol=1e-6)
    assert jnp.all(layer.r[:, 0] == layer.r[:, 0])
    assert jnp.all(layer.r[:, 1] == layer.r[:, 2])


def test_tabm_mini_scales_the_input_once_and_shares_the_trunk():
    key = jax.random.PRNGKey(7)
    backbone = TabMBackbone(
        3,
        key=key,
        n_blocks=2,
        d_block=4,
        dropout=0.0,
        k=2,
        start_scaling_init="normal",
        scaling_chunks=[1, 2],
        arch_type="tabm-mini",
    )
    first, second = backbone.layers
    assert backbone.adapter.shape == (2, 3)
    assert jnp.all(backbone.adapter[:, 1] == backbone.adapter[:, 2])
    assert first.weight.shape == (4, 3) and first.bias.shape == (4,)
    x = jnp.array([[0.2, -0.4, 0.6], [1.0, 0.0, -1.0]], dtype=jnp.float32)
    viewed = jnp.broadcast_to(x[:, None, :], (2, 2, 3)) * backbone.adapter
    hidden = jax.nn.relu(viewed @ first.weight.T + first.bias)
    expected = jax.nn.relu(hidden @ second.weight.T + second.bias)
    assert np.allclose(backbone(x, key=None), expected, atol=1e-6)
    try:
        TabMBackbone(3, key=key, arch_type="packed")
    except ShapeError:
        return
    raise AssertionError("an unknown arch_type was accepted")


def test_batch_ensemble_rejects_a_missing_member_axis():
    layer = LinearBatchEnsemble(
        weight=jnp.ones((2, 3)),
        r=jnp.ones((2, 3)),
        s=jnp.ones((2, 2)),
        bias=jnp.zeros((2, 2)),
    )
    try:
        layer(jnp.ones((4, 3)))
    except ShapeError:
        return
    raise AssertionError("a 2D input was accepted by LinearBatchEnsemble")


def test_feature_dropout_drops_fields_only_while_training():
    from ashlar.core import drop_features

    encoded = jnp.ones((2, 4))
    chunks = (1, 3)
    key = jax.random.PRNGKey(0)
    untouched = drop_features(encoded, chunks, 0.0, key)
    inference = drop_features(encoded, chunks, 0.5, None)
    assert np.allclose(untouched, encoded)
    assert np.allclose(inference, encoded)
    dropped = np.asarray(drop_features(encoded, chunks, 0.5, key))
    assert dropped.shape == (2, 4)
    # The width-3 field is one column group: its three entries share a mask.
    assert np.allclose(dropped[:, 1], dropped[:, 2])
    assert np.allclose(dropped[:, 2], dropped[:, 3])
    assert not np.allclose(dropped, encoded)


def test_tree_branch_starts_as_a_no_op_and_moves_only_the_marked_outputs():
    import equinox as eqx

    from ashlar.layers import TREE_GAIN, TreeBranch

    rng = np.random.default_rng(0)
    numeric = rng.normal(size=(64, 3)).astype(np.float32)
    onehot = np.zeros((64, 0), dtype=np.float32)
    index = rng.integers(0, 4, size=(64, 2)).astype(np.int32)
    # Three numeric columns and a one-hot copy of the second embedded column (3 levels and the unknown row).
    branch = TreeBranch(3, 0, [(1, 4)], (1.0, 0.0), False, 8, 3, key=jax.random.PRNGKey(0), sample=(numeric, onehot, index))
    inputs = np.asarray(branch.inputs(numeric, onehot, index))
    assert inputs.shape == (64, 7)
    assert np.allclose(inputs.mean(axis=0), 0.0, atol=1e-5)
    outputs = jnp.ones((64, 2, 2))
    assert np.allclose(branch(outputs, numeric, onehot, index), outputs)
    # Every tree's leaf weights sum to one, so leaves of one add TREE_GAIN to the marked output only.
    leafy = eqx.tree_at(lambda b: b.layer.leaves, branch, jnp.ones_like(branch.layer.leaves))
    moved = np.asarray(leafy(outputs, numeric, onehot, index))
    assert np.allclose(moved[..., 0], 1.0 + TREE_GAIN, atol=1e-4) and np.allclose(moved[..., 1], 1.0)
    # A training step with a batch per member passes batch * members rows, member fastest.
    varied = eqx.tree_at(lambda b: b.layer.leaves, branch, jax.random.normal(jax.random.PRNGKey(2), branch.layer.leaves.shape))
    flat = np.asarray(varied(jnp.zeros((32, 2, 2)), numeric, onehot, index))
    assert np.allclose(flat, np.asarray(varied.term(numeric, onehot, index)).reshape(32, 2, 2), atol=1e-6)
    # Classification normalizes the log-probabilities again.
    logits = jax.nn.log_softmax(jnp.zeros((64, 2, 2)), axis=-1)
    classed = TreeBranch(3, 0, [], (1.0, 1.0), True, 8, 3, key=jax.random.PRNGKey(1), sample=(numeric, onehot, index))
    classed = eqx.tree_at(lambda b: b.layer.leaves, classed, jnp.ones_like(classed.layer.leaves) * jnp.array([1.0, -1.0]))
    probs = np.exp(np.asarray(classed(logits, numeric, onehot, index)))
    assert np.allclose(probs.sum(axis=-1), 1.0, atol=1e-5) and np.all(probs[..., 0] > 0.99)
    try:
        TreeBranch(0, 0, [], (1.0,), False, 4, 2, key=jax.random.PRNGKey(0))
    except ShapeError:
        return
    raise AssertionError("a tree branch with no inputs was accepted")


def test_feature_gates_shut_fields_at_or_below_zero_and_draw_noise_only_while_training():
    import equinox as eqx

    from ashlar.layers import GATE_SIGMA, GATE_SPEED, GATE_START, FeatureGates

    gates = FeatureGates((1, 3), strength=0.1)
    # Field 0 at mu = -0.1 (shut), field 1 at mu = 0.8 (open, scaled by 0.8 at prediction).
    offset = (jnp.array([-0.1, 0.8]) - GATE_START) / GATE_SPEED
    gates = eqx.tree_at(lambda g: g.offset, gates, offset)
    encoded = jnp.ones((5, 4))
    predicted = np.asarray(gates(encoded))
    assert np.allclose(predicted[:, 0], 0.0)
    assert np.allclose(predicted[:, 1:], 0.8)
    assert gates.open_fields().tolist() == [False, True]
    trained = np.asarray(gates(encoded, key=jax.random.PRNGKey(0)))
    # A training step draws a gate per row and field: a width-3 field shares it, and values stay in [0, 1].
    assert np.allclose(trained[:, 1], trained[:, 2]) and np.allclose(trained[:, 2], trained[:, 3])
    assert trained.min() >= 0.0 and trained.max() <= 1.0
    assert not np.allclose(trained, predicted)
    expected = 0.1 * (jax.scipy.stats.norm.cdf(-0.1 / GATE_SIGMA) + jax.scipy.stats.norm.cdf(0.8 / GATE_SIGMA))
    assert np.isclose(float(gates.penalty()), float(expected), rtol=1e-5)
