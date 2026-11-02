"""The additive backbone: one network per column or pair, so every head output is a sum of per-term shapes."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ashlar import ASRRegressor
from ashlar.errors import AshlarError, ShapeError
from ashlar.layers import AdditiveBackbone
from ashlar.preprocess import FeatureEncoder
from ashlar.shapes import additive_terms, check_pairs, output_names


def _table(n=600, seed=0):
    rng = np.random.default_rng(seed)
    hour = rng.integers(0, 24, n)
    work = rng.integers(0, 2, n)
    frame = pd.DataFrame(
        {
            "color": rng.choice(["red", "green", "blue"], n),
            "x1": rng.uniform(-2.0, 2.0, n),
            "hour": hour,
            "x2": rng.normal(size=n),
            "work": work,
        }
    )
    loc = np.sin(1.5 * frame["x1"]) + 0.5 * (frame["color"] == "red") + np.where(work == 1, np.cos(2 * np.pi * hour / 24), 0.3)
    y = loc + 0.3 * np.exp(0.3 * frame["x2"]) * rng.normal(size=n)
    return frame, np.asarray(y)


def _small(task="gaussian", **kwargs):
    params = dict(
        task=task,
        backbone="additive",
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "d_term": 4},
        max_epochs=3,
        patience=3,
        batch_size=128,
        random_state=0,
    )
    params.update(kwargs)
    return ASRRegressor(**params)


def test_pairs_and_terms_are_checked():
    assert check_pairs(None) == ()
    assert check_pairs([["a", "b"], ("c", "a")]) == (("a", "b"), ("c", "a"))
    for bad in ("ab", [["a"]], [["a", "a"]], [["a", "b"], ["b", "a"]], [[1, 2]], [("a", "b", "c")]):
        with pytest.raises(AshlarError):
            check_pairs(bad)
    # A cyclical column's two fields are one term; an excluded (monotone) column has none; a pair reads both columns.
    terms, columns = additive_terms(["x", "hour", "hour", "color"], (("hour", "color"),), {"x"})
    assert terms == ((1, 2), (3,), (1, 2, 3))
    assert columns == (("hour",), ("color",), ("hour", "color"))
    with pytest.raises(AshlarError):
        additive_terms(["x", "y"], (("x", "z"),))
    with pytest.raises(AshlarError):
        additive_terms(["x"], (), {"x"})


def test_each_term_reads_only_its_own_fields():
    backbone = AdditiveBackbone(7, key=jax.random.PRNGKey(0), feature_chunks=[2, 1, 4], terms=[(0,), (1,), (2,), (0, 2)], k=2)
    assert backbone.d_out == 4 * backbone.d_term and backbone.k == 2
    x = jax.random.normal(jax.random.PRNGKey(1), (5, 7))
    base = backbone.term_outputs(x)
    assert base.shape == (5, 2, 4, backbone.d_term)
    moved = backbone.term_outputs(x.at[:, 2].add(1.0))  # field 1 is column 2
    changed = np.abs(np.asarray(moved - base)).max(axis=(0, 1, 3))
    assert changed[1] > 0 and np.all(changed[[0, 2, 3]] == 0)
    moved = backbone.term_outputs(x.at[:, 4].add(1.0))  # field 2 is columns 3 to 6
    changed = np.abs(np.asarray(moved - base)).max(axis=(0, 1, 3))
    assert changed[2] > 0 and changed[3] > 0 and changed[0] == 0 and changed[1] == 0
    assert backbone(x, key=None).shape == (5, 2, backbone.d_out)
    for bad in ([(0, 0)], [(3,)], [()], [(0,), (0,)]):
        with pytest.raises(ShapeError):
            AdditiveBackbone(7, key=jax.random.PRNGKey(0), feature_chunks=[2, 1, 4], terms=bad)


def test_fields_follow_the_core_order():
    frame = pd.DataFrame({"c": ["a", "b", "a"], "x": [1.0, 2.0, 3.0], "o": ["u", "v", "u"], "y": [0.5, 0.1, 0.2]})
    encoder = FeatureEncoder.fit(frame, embeddings=["c"])
    assert encoder.fields(16, 8) == [("x", 16), ("y", 16), ("o", 3), ("c", 8)]
    assert encoder.fields(16, 8, legacy=True) == [("c", 8), ("x", 16), ("o", 3), ("y", 16)]
    assert encoder.feature_chunks(None, 8) == [1, 1, 3, 8]


def test_additive_model_is_the_sum_of_its_terms(tmp_path):
    frame, y = _table()
    model = _small(backbone_kwargs={"n_blocks": 1, "d_block": 8, "d_term": 4, "pairs": [["hour", "work"]]})
    model.fit(frame, y)
    shapes = model.shape_functions(frame)
    assert shapes.names == ("x1", "hour", "x2", "work", "color", "hour:work")
    assert shapes.columns[-1] == ("hour", "work") and shapes.outputs == ("loc", "scale")
    raw = np.asarray(model.predict_raw(frame)).mean(axis=1)
    location = raw[:, 0] * model.target_std_ + model.target_mean_
    np.testing.assert_allclose(shapes.total()[:, 0], location, atol=1e-4)
    np.testing.assert_allclose(shapes.total()[:, 1], raw[:, 1], atol=1e-4)
    np.testing.assert_allclose(shapes.values.mean(axis=0), 0.0, atol=1e-6)
    # Moving one column moves only its terms.
    moved = frame.assign(x1=frame["x1"] + 0.5)
    before, after = model.shape_functions(frame, center=False), model.shape_functions(moved, center=False)
    change = np.abs(after.values - before.values).max(axis=(0, 2))
    assert change[0] > 0 and np.all(change[1:] == 0)
    assert list(shapes.importance("loc")) and shapes.to_frame("scale").shape == (len(frame), 6)
    assert shapes.term("x1", "loc").shape == (len(frame),)
    model.save(tmp_path / "model")
    loaded = ASRRegressor.load(tmp_path / "model")
    np.testing.assert_allclose(np.asarray(loaded.predict_raw(frame)), np.asarray(model.predict_raw(frame)))
    np.testing.assert_allclose(loaded.shape_functions(frame).values, shapes.values)


def test_monotone_features_and_counts_are_terms():
    frame, y = _table()
    counts = np.random.default_rng(1).poisson(np.exp(0.5 + 0.4 * frame["x2"].to_numpy()))
    model = _small("negative_binomial", monotone={"x2": 1}, cyclical={"hour": 24})
    model.fit(frame, counts)
    shapes = model.shape_functions(frame)
    assert shapes.outputs == ("log_mean", "dispersion")
    # The cyclical hour is one term; the monotone feature is its own rising shape, on the log mean only.
    assert shapes.names == ("x1", "hour", "work", "color", "x2")
    order = np.argsort(frame["x2"].to_numpy())
    rising = shapes.term("x2", "log_mean")[order]
    assert np.all(np.diff(rising) >= -1e-6)
    np.testing.assert_allclose(shapes.term("x2", "dispersion"), 0.0, atol=1e-6)
    raw = np.asarray(model.predict_raw(frame)).mean(axis=1)
    np.testing.assert_allclose(shapes.total(), raw, atol=1e-4)


def test_random_effects_offsets_are_a_term():
    frame, y = _table()
    frame = frame.assign(store=[f"s{index % 12}" for index in range(len(frame))])
    model = _small(random_effects={"store": "intercept"})
    model.fit(frame, y)
    shapes = model.shape_functions(frame)
    assert shapes.names[-1] == "store (random effect)"
    np.testing.assert_allclose(shapes.term("store (random effect)", "scale"), 0.0, atol=1e-7)
    raw = np.asarray(model.core_(*_inputs(model, frame), key=None, offsets=False)).mean(axis=1)
    offset = np.asarray(model.core_.random_effects.offsets(_inputs(model, frame)[2], _inputs(model, frame)[0])[0])
    location = (raw[:, 0] + offset) * model.target_std_ + model.target_mean_
    np.testing.assert_allclose(shapes.total()[:, 0], location, atol=1e-4)


def _inputs(model, frame):
    encoded = model.encoder_.transform(model._with_cyclical(frame))
    numeric_z = model.numeric_transform_(encoded.numeric_raw)
    return numeric_z, encoded.categorical_onehot, encoded.category_index, encoded.set_tokens, encoded.set_mask


def test_additive_refusals():
    frame, y = _table(n=80)
    for kwargs in (
        {"task": "multivariate_flow"},
        {"bottleneck": 8},
        {"share_training_batches": False},
        {"monotone": {"x1": 1}, "backbone_kwargs": {"pairs": [["x1", "hour"]]}},
    ):
        with pytest.raises(AshlarError):
            _small(**kwargs)._check_config()
    with pytest.raises(AshlarError):
        ASRRegressor(backbone_kwargs={"pairs": [["x1", "hour"]]})._check_config()
    with pytest.raises(AshlarError):
        _small(backbone_kwargs={"terms": [[0]]})._check_config()
    with pytest.raises(AshlarError):
        _small(backbone_kwargs={"pairs": [["x1", "nope"]]}).fit(frame, y)
    with pytest.raises(AshlarError):
        ASRRegressor(task="gaussian", max_epochs=1, backbone_kwargs={"n_blocks": 1, "d_block": 8}).fit(frame, y).shape_functions(frame)


def test_output_names():
    assert output_names("gaussian", 3, cure=True) == ("loc", "scale", "cure_logit")
    assert output_names("mixture", 15)[:3] == ("loc_1", "scale_1", "weight_1")
    assert output_names("negative_binomial", 3) == ("log_mean", "dispersion", "zero_logit")
    assert output_names("hurdle", 3) == ("zero_logit", "severity_loc", "severity_scale")
    assert output_names("classification", 2, classes=np.asarray([0, 1])) == ("0", "1")
    assert output_names("histogram", 2) == ("bin_1", "bin_2")
    assert output_names("gaussian", 5) == tuple(f"output_{index}" for index in range(1, 6))


def test_a_transformer_saved_in_schema_order_still_loads(tmp_path):
    """Models saved before the fields followed the core's order build their core the old way."""
    frame, y = _table(n=120)
    model = ASRRegressor(
        backbone="transformer",
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "n_heads": 2},
        categorical_embeddings=[],
        max_epochs=1,
        random_state=0,
    ).fit(frame, y)
    widths = [int(weight.shape[1]) for weight in model.core_.backbone.projection.weights]
    assert widths == [16, 16, 16, 16, 4]
    # Rebuild the core the old way, as such a save holds it, and save it without the field order.
    model._legacy_field_order = True
    model.core_ = model._build_core(model.encoder_, model.n_events_, model.n_classes_, jax.random.PRNGKey(3))
    legacy = [int(weight.shape[1]) for weight in model.core_.backbone.projection.weights]
    assert legacy == [4, 16, 16, 16, 16]
    model.save(tmp_path / "legacy", save_optimizer=False)
    path = tmp_path / "legacy" / "metadata.json"
    payload = json.loads(path.read_text())
    del payload["field_order"]
    path.write_text(json.dumps(payload))
    loaded = ASRRegressor.load(tmp_path / "legacy")
    np.testing.assert_allclose(np.asarray(loaded.predict(frame)), np.asarray(model.predict(frame)), rtol=1e-6)


@pytest.mark.slow
def test_additive_gaussian_recovers_the_shapes_of_location_and_scale():
    rng = np.random.default_rng(0)
    n = 6000
    frame = pd.DataFrame({"a": rng.uniform(-2, 2, n), "b": rng.uniform(-2, 2, n), "c": rng.normal(size=n)})
    loc = np.sin(1.5 * frame["a"]) + 0.5 * frame["b"] ** 2
    scale = np.exp(0.5 * frame["c"].clip(-2, 2))
    y = np.asarray(loc + 0.2 * scale * rng.normal(size=n))
    model = ASRRegressor(task="gaussian", backbone="additive", max_epochs=60, random_state=0).fit(frame, y)
    shapes = model.shape_functions(frame)
    truth_a = np.sin(1.5 * frame["a"]) - np.sin(1.5 * frame["a"]).mean()
    assert np.corrcoef(shapes.term("a", "loc"), truth_a)[0, 1] > 0.98
    assert np.corrcoef(shapes.term("b", "loc"), frame["b"] ** 2)[0, 1] > 0.98
    # The spread grows with c and with nothing else.
    assert np.corrcoef(shapes.term("c", "scale"), frame["c"].clip(-2, 2))[0, 1] > 0.9
    importance = shapes.importance("scale")
    assert list(importance)[0] == "c"
