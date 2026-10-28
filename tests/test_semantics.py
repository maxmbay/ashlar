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
