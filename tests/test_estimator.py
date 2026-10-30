import numpy as np

import jax.numpy as jnp

from ashlar.errors import AshlarError, ShapeError, WeightError
from ashlar.estimator import TabularEstimator, _apply_class_weight
from ashlar.heads import OrdinalHead, cut_raw_from_class_prior
from ashlar.layers import DEFAULT_BOTTLENECK_WIDTH, SOFT_STAGE_SHRINKAGE, SoftStageHistogram
from ashlar.objectives import (
    classification_objective,
    gaussian_bin_probabilities,
    make_ordinal_expectation_objective,
    ordinal_expectation_squared_error,
    ordinal_log_probabilities,
    ordinal_objective,
    ordered_thresholds,
)
from ashlar.recipes import CLASSIFIER_RECIPE, DEFAULT_RECIPE, ORDINAL_RECIPE, TABM_PAPER, classifier, ordinal, tabm
from ashlar.train import surface_learning_rate
from ashlar.search import DEFAULT_SEARCH


def _regression(task="regression", **kwargs):
    params = dict(
        task=task,
        backbone="residual_mlp",
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0},
        max_epochs=6,
        patience=3,
        batch_size=8,
        validation_size=0.25,
        learning_rate=1e-2,
        random_state=0,
    )
    params.update(kwargs)
    return TabularEstimator(**params)


def test_paper_recipe_records_tabm_defaults_without_training():
    assert TABM_PAPER["backbone_kwargs"]["k"] == 32
    assert TABM_PAPER["backbone_kwargs"]["d_block"] == 512
    assert TABM_PAPER["backbone_kwargs"]["n_blocks"] == 3
    assert TABM_PAPER["learning_rate"] == 2e-3


def test_negative_weights_do_not_fit():
    model = _regression()
    x = np.ones((8, 2))
    y = np.ones(8)
    try:
        model.fit(x, y, sample_weight=np.array([1, 1, 1, 1, -1, 1, 1, 1], dtype=float))
    except WeightError:
        assert not getattr(model, "is_fitted_", False)
        return
    raise AssertionError("negative sample_weight was accepted")


def test_incompatible_head_is_rejected():
    try:
        _regression(task="normal", head="logits").fit(np.ones((8, 1)), np.ones(8))
    except AshlarError:
        return
    raise AssertionError("logits head was accepted for a normal task")


def test_weighted_outlier_changes_the_fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 1)).astype(np.float32)
    y = (2 * x[:, 0]).astype(np.float32)
    y = y.copy()
    y[0] = 50.0
    uniform = _regression(validation_size=0, max_epochs=20)
    weighted = _regression(validation_size=0, max_epochs=20)
    weights = np.ones(len(y))
    weights[0] = 0.0
    uniform.fit(x, y)
    weighted.fit(x, y, sample_weight=weights)
    query = np.array([[0.0]])
    uniform_pred = float(np.asarray(uniform.predict(query)).reshape(-1)[0])
    weighted_pred = float(np.asarray(weighted.predict(query)).reshape(-1)[0])
    assert abs(uniform_pred - weighted_pred) > 1e-4


def test_temporal_split_does_not_fit_preprocessing_on_the_tail():
    x = np.array([[0.0], [0.0], [0.0], [100.0]], dtype=np.float32)
    y = np.zeros(4, dtype=np.float32)
    model = _regression(validation_size=0.25, split="temporal", max_epochs=1, patience=0)
    model.fit(x, y)
    mean = float(model.numeric_transform_.mean[0])
    assert abs(mean) < 1e-5


def test_group_split_holds_out_whole_groups():
    x = np.arange(8, dtype=np.float32)[:, None]
    y = x[:, 0].copy()
    groups = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    model = _regression(split="group", validation_size=0.5, max_epochs=1, patience=0)
    model.fit(x, y, groups=groups)
    # The training mean is either 1.5 or 5.5, never the grand mean 3.5.
    mean = float(model.numeric_transform_.mean[0])
    assert abs(mean - 3.5) > 1.0


def test_unseen_category_and_missing_number_predict():
    x = {
        "value": np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32),
        "city": np.array(["ny", "sf", "ny", "sf", "ny", "sf", "ny", "sf"], dtype=object),
    }
    y = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    model = _regression(validation_size=0, max_epochs=2, categorical=["city"])
    model.fit(x, y)
    prediction = model.predict({"value": np.array([np.nan]), "city": np.array(["boston"], dtype=object)})
    assert np.isfinite(prediction).all()


def test_save_reload_and_resume(tmp_path):
    x = np.linspace(-1, 1, 16, dtype=np.float32)[:, None]
    y = x[:, 0] ** 2
    model = _regression(validation_size=0, max_epochs=3, task="normal", prediction_policy="mixture")
    model.fit(x, y)
    first = model.predict(x)
    mixture = model.predict_dist(x[:2])
    averaged = model.predict_dist(x[:2], policy="mean_parameters")
    assert mixture.loc.shape[1] == 1  # residual MLP has one member
    assert averaged.loc.ndim == 2
    restored = type(model).load(model.save(tmp_path / "model"))
    assert np.allclose(first, restored.predict(x), atol=1e-6)
    restored.fit(x, y, resume=True)
    assert restored.history_["train_loss"]


def test_classification_policies_and_explicit_none_weight():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    left = _regression(task="classification", validation_size=0.2, max_epochs=4, prediction_policy="mean_probabilities")
    right = _regression(task="classification", validation_size=0.2, max_epochs=4, prediction_policy="mean_probabilities")
    left.fit(x, y, sample_weight=None)
    right.fit(x, y)
    assert np.allclose(left.predict_proba(x), right.predict_proba(x), atol=1e-6)
    labels = left.predict(x)
    assert set(np.unique(labels)).issubset({0, 1})


def test_gradient_flows_through_standardization():
    import jax

    x = np.linspace(0, 1, 12, dtype=np.float32)[:, None]
    y = 3 * x[:, 0] + 1
    model = _regression(validation_size=0, max_epochs=5)
    model.fit(x, y)
    cat = jax.numpy.zeros((1, 0), dtype=jax.numpy.float32)
    value = jax.numpy.array([0.4], dtype=jax.numpy.float32)

    def loss(raw):
        return model.forward_transformed(raw[None, :], cat)[0]

    gradient = np.array(jax.grad(loss)(value).tolist(), dtype=float)
    assert gradient.tolist() != []
    assert np.isfinite(gradient).all()
    assert abs(gradient[0]) > 0


def test_default_recipe_is_the_no_argument_estimator():
    params = TabularEstimator().get_params()
    for key, value in DEFAULT_RECIPE.items():
        assert params[key] == value
    assert params["weight_decay"] == 1e-4
    assert params["feature_dropout"] == 0.0
    assert params["patience"] == 16
    assert params["validation_size"] == 0.2
    assert DEFAULT_SEARCH["n_trials"] == 2
    assert DEFAULT_SEARCH["space"]["weight_decay"] == [1e-4, 1e-3]


def test_search_is_off_unless_requested_and_true_picks_a_listed_rate():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 1)).astype(np.float32)
    y = x[:, 0].astype(np.float32)
    plain = _regression(max_epochs=2, patience=0)
    plain.fit(x, y)
    assert plain.search_ is None
    assert plain.resolved_["learning_rate"] == plain.learning_rate
    searched = _regression(max_epochs=2, patience=0, learning_rate=1e-3)
    searched.fit(x, y, search=True)
    chosen = searched.search_["chosen"]["weight_decay"]
    assert chosen in DEFAULT_SEARCH["space"]["weight_decay"]
    assert len(searched.search_["trials"]) == 2
    assert searched.weight_decay == 1e-4
    assert searched.learning_rate == 1e-3
    assert searched.search_ is not None
    plain_again = _regression(max_epochs=1, patience=0, weight_decay=0.0, feature_dropout=0.0)
    plain_again.fit(x, y)
    assert plain_again.search_ is None
    assert plain_again.core_.feature_dropout == 0.0


def test_learned_embedding_and_set_encoder_accept_unseen_and_empty():
    rng = np.random.default_rng(2)
    n = 20
    rooms = rng.normal(size=n).astype(np.float32)
    city = rng.choice(["ny", "sf"], size=n).astype(object)
    tags = np.empty(n, dtype=object)
    for index, value in enumerate([["red"], ["red", "blue"], ["blue"], []] * 5):
        tags[index] = value
    y = rooms + np.where(city == "sf", 0.2, 0.0)
    model = _regression(
        backbone="set",
        backbone_kwargs={"d_block": 8, "dropout": 0.0},
        categorical_embeddings=["city"],
        set_features=["tags"],
        categorical_embedding_dim=4,
        max_epochs=2,
        patience=0,
        validation_size=0.25,
    )
    model.fit({"rooms": rooms, "city": city, "tags": tags}, y.astype(np.float32))
    assert model.core_.categorical_embeddings is not None
    assert model.core_.set_embeddings is not None
    query_tags = np.empty(1, dtype=object)
    query_tags[0] = []
    prediction = model.predict(
        {"rooms": np.array([0.1], dtype=np.float32), "city": np.array(["boston"], dtype=object), "tags": query_tags}
    )
    assert np.isfinite(np.asarray(prediction)).all()
    widths = model.encoder_.categorical_widths
    assert widths == []


def test_transformer_backbone_fits_a_feature_set():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 2)).astype(np.float32)
    y = x[:, 0] - x[:, 1]
    model = _regression(
        backbone="transformer",
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "n_heads": 2, "dropout": 0.0},
        max_epochs=1,
        patience=0,
        validation_size=0.25,
    )
    model.fit(x, y.astype(np.float32))
    assert model.core_.backbone.k == 1
    assert np.isfinite(np.asarray(model.predict(x[:2]))).all()


def test_classifier_recipe_predicts_binary_and_multiclass_labels():
    params = classifier().get_params()
    for key, value in CLASSIFIER_RECIPE.items():
        assert params[key] == value
    assert params["task"] == "classification"
    rng = np.random.default_rng(4)
    binary_x = rng.normal(size=(28, 2)).astype(np.float32)
    binary_x[0, 0] = np.nan
    binary_y = np.where(binary_x[:, 1] > 0, "yes", "no")
    binary = classifier(max_epochs=4, patience=2, batch_size=8, learning_rate=1e-2, random_state=0)
    binary.fit(binary_x, binary_y)
    labels = np.asarray(binary.predict(binary_x))
    probabilities = np.asarray(binary.predict_proba(binary_x))
    assert set(np.unique(labels)).issubset({"no", "yes"})
    assert probabilities.shape == (len(binary_x), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert list(binary.classes_) == ["no", "yes"]
    cities = np.array(["ny", "sf", "la"] * 12, dtype=object)
    multi_x = {"rooms": rng.normal(size=36).astype(np.float32), "city": cities}
    multi_y = np.array(["low", "mid", "high"] * 12)
    multi = classifier(max_epochs=3, patience=2, batch_size=12, learning_rate=1e-2, random_state=1)
    multi.fit(multi_x, multi_y)
    multi_labels = np.asarray(multi.predict({"rooms": np.array([0.0, np.nan]), "city": np.array(["boston", "sf"], dtype=object)}))
    multi_probabilities = np.asarray(multi.predict_proba({"rooms": np.array([0.0, np.nan]), "city": np.array(["boston", "sf"], dtype=object)}))
    assert multi_probabilities.shape == (2, 3)
    assert set(multi_labels).issubset(set(multi.classes_))
    assert np.allclose(multi_probabilities.sum(axis=1), 1.0, atol=1e-5)


def test_classification_search_scores_weighted_accuracy():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(32, 2)).astype(np.float32)
    y = (x[:, 0] > 0).astype(int)
    plain = classifier(max_epochs=2, patience=0, batch_size=8, learning_rate=1e-2, random_state=0)
    plain.fit(x, y)
    assert plain.search_ is None
    searched = classifier(max_epochs=2, patience=0, batch_size=8, learning_rate=1e-3, random_state=0)
    searched.fit(x, y, search=True)
    scores = [trial["val_score"] for trial in searched.search_["trials"]]
    assert [trial["metric"] for trial in searched.search_["trials"]] == ["weighted_accuracy", "weighted_accuracy"]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert searched.search_["chosen"]["weight_decay"] == DEFAULT_SEARCH["space"]["weight_decay"][int(np.argmax(scores))]
    assert searched.search_["chosen"]["learning_rate"] == 1e-3
    assert searched.learning_rate == 1e-3
    assert "val_score" not in plain.history_


def test_classification_embeddings_transformer_and_set():
    rng = np.random.default_rng(6)
    n = 24
    rooms = rng.normal(size=n).astype(np.float32)
    city = np.array(["ny", "sf"] * (n // 2), dtype=object)
    y = np.where(rooms + np.where(city == "sf", 0.5, 0.0) > 0, "a", "b")
    shared = dict(max_epochs=2, patience=0, batch_size=8, learning_rate=1e-2, categorical_embeddings=["city"], categorical_embedding_dim=4, random_state=0)
    transformer = classifier(backbone="transformer", backbone_kwargs={"n_blocks": 1, "d_block": 8, "n_heads": 2, "dropout": 0.0}, **shared)
    pooled = classifier(backbone="set", backbone_kwargs={"d_block": 8, "dropout": 0.0}, **shared)
    transformer.fit({"rooms": rooms, "city": city}, y)
    pooled.fit({"rooms": rooms, "city": city}, y)
    query = {"rooms": np.array([0.2]), "city": np.array(["boston"], dtype=object)}
    for model in (transformer, pooled):
        probabilities = np.asarray(model.predict_proba(query))
        assert probabilities.shape == (1, 2)
        assert np.isfinite(probabilities).all()
        assert model.encoder_.categorical_widths == []
        assert model.core_.categorical_embeddings is not None


def test_class_weight_and_sample_weight_multiply_once():
    y = np.array([0, 0, 1])
    sample_weight = np.array([1.0, 2.0, 3.0])
    combined = _apply_class_weight(y, y, 2, {0: 1.0, 1: 4.0}, sample_weight)
    assert np.allclose(combined, [1.0, 2.0, 12.0])
    raw = jnp.array([[[3.0, 0.0]], [[0.2, 0.0]], [[0.0, 2.0]]], dtype=jnp.float32)
    labels = jnp.array([0, 0, 1], dtype=jnp.int32)
    once = float(classification_objective(raw, labels, jnp.asarray(combined, dtype=jnp.float32)))
    squared = sample_weight * np.array([1.0, 1.0, 16.0])
    twice = float(classification_objective(raw, labels, jnp.asarray(squared, dtype=jnp.float32)))
    assert abs(once - twice) > 1e-6


def test_quantile_scaling_matches_sklearn_and_standard_stays_zscore():
    from sklearn.preprocessing import QuantileTransformer

    from ashlar.preprocess import FeatureEncoder

    rng = np.random.default_rng(0)
    train = rng.normal(size=(48, 3))
    train[0, 0] = np.nan
    train[:, 2] = np.resize(np.array([1.0, 1.0, 2.0, 2.0, 2.0]), train.shape[0])
    standard = FeatureEncoder.fit(train, numeric_scaling="standard")
    column = np.asarray(train[:, 0], dtype=np.float64)
    finite = np.isfinite(column)
    filled = np.where(finite, column, float(np.median(column[finite])))
    expected = (0.5 - filled.mean()) / filled.std()
    got = float(standard.numeric_transform(jnp.asarray([[0.5, 0.0, 1.0]], dtype=jnp.float32))[0, 0])
    assert standard.numeric_transform.scaling == "standard"
    assert abs(got - expected) < 1e-5

    encoder = FeatureEncoder.fit(train, numeric_scaling="quantile")
    medians = np.array([column.median for column in encoder.columns if column.role == "numeric"], dtype=np.float64)
    numeric = np.asarray(train, dtype=np.float64)
    finite = np.isfinite(numeric)
    imputed = np.where(finite, numeric, medians)
    transformer = QuantileTransformer(
        n_quantiles=min(1000, len(train)),
        output_distribution="normal",
        subsample=len(train),
        random_state=0,
    )
    transformer.fit(imputed)
    query = np.array([[0.2, -0.4, 1.0], [8.0, -3.0, 2.0], [medians[0], 0.1, 1.5]], dtype=np.float64)
    expected_q = transformer.transform(query)
    got_q = np.asarray(encoder.numeric_transform(jnp.asarray(query, dtype=jnp.float32)), dtype=np.float64)
    missing = np.array([[np.nan, 0.1, 1.5]], dtype=np.float32)
    filled = np.array([[np.float32(medians[0]), 0.1, 1.5]], dtype=np.float32)
    assert np.allclose(
        encoder.numeric_transform(jnp.asarray(missing)),
        encoder.numeric_transform(jnp.asarray(filled)),
    )
    assert encoder.numeric_transform.scaling == "quantile"
    assert np.isfinite(got_q).all()
    assert np.max(np.abs(got_q - expected_q)) < 1e-4


def test_quantile_tabr_knot_count_skips_the_zscore():
    from ashlar.preprocess import FeatureEncoder

    rng = np.random.default_rng(0)
    train = rng.normal(size=(900, 2))
    train[:, 0] = np.exp(train[:, 0])
    encoder = FeatureEncoder.fit(train, numeric_scaling="quantile_tabr")
    assert encoder.numeric_transform.scaling == "quantile_tabr"
    assert encoder.numeric_transform.quantiles.shape == (30, 2)
    mapped = np.asarray(encoder.numeric_transform(jnp.asarray(train[:4], dtype=jnp.float32)))
    zscore = (train[:4] - train.mean(axis=0)) / train.std(axis=0)
    assert np.isfinite(mapped).all()
    assert np.max(np.abs(mapped - zscore)) > 0.2


def test_regression_labels_are_standardized_unless_another_task_or_an_old_file(tmp_path):
    import json

    import pytest

    assert DEFAULT_RECIPE["standardize_target"] is True
    assert CLASSIFIER_RECIPE["standardize_target"] is False
    assert ORDINAL_RECIPE["standardize_target"] is False
    assert TabularEstimator().standardize_target is True
    assert classifier().standardize_target is False
    assert ordinal().standardize_target is False
    assert tabm().standardize_target is True
    assert tabm(task="ordinal", prediction_policy="mean_probabilities").standardize_target is False
    assert TabularEstimator(task="classification").standardize_target is False
    with pytest.raises(AshlarError, match="standardize_target"):
        TabularEstimator(task="classification", standardize_target=True)._check_config()

    y = np.full(8, 5.0, dtype=np.float32)
    x = np.linspace(0, 1, len(y), dtype=np.float32)[:, None]
    constant = _regression(max_epochs=1, patience=0, validation_size=0)
    constant.fit(x, y)
    assert constant.standardize_target is True
    assert abs(float(constant.target_mean_) - 5.0) < 1e-5
    assert abs(float(constant.target_std_) - 1.0) < 1e-5
    assert np.allclose(np.asarray(constant.predict(x[:2]), dtype=np.float64), 5.0, atol=0.5)

    y_tiny = np.array([1e-4, -1e-4, 2e-4, -2e-4, 0.0, 1e-4, -1e-4, 0.0], dtype=np.float32)
    tiny = _regression(max_epochs=1, patience=0, validation_size=0)
    tiny.fit(np.arange(len(y_tiny), dtype=np.float32)[:, None], y_tiny)
    assert abs(float(tiny.target_std_) - float(np.std(y_tiny))) < 1e-8
    restored_scale = np.asarray(tiny.predict(np.arange(len(y_tiny), dtype=np.float32)[:, None]), dtype=np.float64)
    assert np.isfinite(restored_scale).all()
    assert np.max(np.abs(restored_scale)) < 1.0

    off = _regression(max_epochs=1, patience=0, validation_size=0, standardize_target=False)
    off.fit(x, np.linspace(0, 1, len(y), dtype=np.float32))
    directory = tmp_path / "old-scale"
    off.save(directory)
    meta_path = directory / "metadata.json"
    payload = json.loads(meta_path.read_text())
    del payload["standardize_target"]
    meta_path.write_text(json.dumps(payload))
    restored = TabularEstimator.load(directory)
    assert restored.standardize_target is False
    assert np.allclose(off.predict(x), restored.predict(x), atol=1e-6)


def test_label_zscore_is_inverted_on_predict():
    y = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0], dtype=np.float32)
    x = np.linspace(0, 1, len(y), dtype=np.float32)[:, None]
    model = _regression(max_epochs=1, patience=0, validation_size=0, standardize_target=True)
    model.fit(x, y)
    assert abs(model.target_mean_ - float(y.mean())) < 1e-5
    assert abs(model.target_std_ - float(y.std())) < 1e-5
    pred = np.asarray(model.predict(x[:2]), dtype=np.float64)
    assert np.isfinite(pred).all()
    assert abs(float(pred.mean()) - float(y.mean())) < 8.0


def test_classification_tabm_shares_batches_and_skips_the_label_scale():
    import pytest

    model = tabm(task="classification")
    assert model.task == "classification"
    assert model.prediction_policy == "mean_probabilities"
    assert model.standardize_target is False
    assert model.share_training_batches is True
    assert model.numeric_scaling == "robust"
    assert "numeric_scaling" not in TABM_PAPER
    model._check_config()
    assert tabm().prediction_policy == "mean_members"
    assert tabm().standardize_target is True
    assert tabm(task="classification", numeric_scaling="quantile_tabr").numeric_scaling == "quantile_tabr"
    assert tabm(task="classification", prediction_policy="mean_logits").prediction_policy == "mean_logits"
    with pytest.raises(AshlarError, match="share_training_batches"):
        tabm(task="classification", share_training_batches=False)._check_config()
    with pytest.raises(AshlarError, match="standardize_target"):
        tabm(task="classification", standardize_target=True)._check_config()

    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 3)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int32)
    small = tabm(
        task="classification",
        backbone_kwargs={
            "n_blocks": 1,
            "d_block": 8,
            "dropout": 0.0,
            "k": 4,
            "start_scaling_init": "random-signs",
        },
        max_epochs=1,
        patience=0,
        batch_size=8,
        validation_size=0.25,
    )
    small.fit(x, y)
    assert small.target_mean_ is None
    assert small.target_std_ is None
    assert small.core_.backbone.share_training_batches is True
    pred = np.asarray(small.predict(x[:4]))
    proba = np.asarray(small.predict_proba(x[:4]), dtype=np.float64)
    assert pred.shape == (4,)
    assert proba.shape == (4, 2)
    assert np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_unshared_member_targets_and_a_small_tabm_fit():
    from ashlar.objectives import regression_objective
    from ashlar.recipes import tabm

    raw = jnp.zeros((2, 2, 1))
    separate = jnp.array([[0.0, 1.0], [0.0, 0.0]])
    loss = float(regression_objective(raw, separate, jnp.ones((2, 2))))
    assert abs(loss - 0.25) < 1e-6
    rng = np.random.default_rng(0)
    x = rng.normal(size=(24, 3)).astype(np.float32)
    y = x[:, 0].astype(np.float32)
    model = tabm(
        backbone_kwargs={
            "n_blocks": 1,
            "d_block": 8,
            "dropout": 0.0,
            "k": 4,
            "start_scaling_init": "random-signs",
        },
        max_epochs=1,
        patience=0,
        batch_size=8,
        validation_size=0.25,
        share_training_batches=False,
    )
    model.fit(x, y)
    assert model.core_.backbone.share_training_batches is False
    pred = np.asarray(model.predict(x[:2]))
    assert pred.shape == (2,)
    assert np.isfinite(pred).all()


def test_linear_schedule_stays_on_adamw():
    model = _regression(max_epochs=2, patience=0, validation_size=0.25, learning_rate_schedule="linear")
    x = np.linspace(0, 1, 16, dtype=np.float32)[:, None]
    model.fit(x, x[:, 0])
    assert model.learning_rate_schedule == "linear"
    assert np.isfinite(np.asarray(model.predict(x[:2]))).all()


def test_surface_learning_rate_shrinks_when_data_is_small_or_steep():
    wide = surface_learning_rate(10_000, curvature=20.0, grad_rms=0.5)
    small = surface_learning_rate(2_000, curvature=20.0, grad_rms=0.5)
    steep = surface_learning_rate(10_000, curvature=80.0, grad_rms=0.5)
    assert wide == 0.05
    assert small == 0.01
    assert steep == 0.0125
    assert small < wide
    assert steep < wide
    fallback = surface_learning_rate(5_000, curvature=-1.0, grad_rms=10.0)
    assert fallback == 0.05
    assert surface_learning_rate(5_000, curvature=-1.0, grad_rms=2.0) == 0.1
    assert surface_learning_rate(10_000, curvature=1e-12, grad_rms=0.0) == 0.1
    assert surface_learning_rate(10, curvature=1e6, grad_rms=1.0) == 1e-5
    assert DEFAULT_RECIPE["learning_rate_schedule"] == "cosine"
    assert DEFAULT_RECIPE["learning_rate"] == 2e-3


def test_surface_schedule_sets_an_adamw_rate_from_the_batch(tmp_path):
    model = _regression(
        max_epochs=2,
        patience=0,
        validation_size=0.25,
        learning_rate=2e-3,
        learning_rate_schedule="surface",
        weight_decay=0.0,
    )
    x = np.linspace(0, 1, 24, dtype=np.float32)[:, None]
    model.fit(x, x[:, 0])
    chosen = model.history_["chosen_learning_rate"]
    assert model.learning_rate == 2e-3
    assert model.learning_rate_ == chosen
    assert 1e-5 <= chosen <= 1e-1
    assert model.history_["surface_warmup_steps"] >= 1
    assert model.history_["surface_n_train"] < len(x)
    assert np.isfinite(model.history_["surface_curvature"])
    assert np.isfinite(model.history_["surface_grad_rms"])
    pred = np.asarray(model.predict(x[:2]))
    assert np.isfinite(pred).all()
    restored = type(model).load(model.save(tmp_path / "surface"))
    assert np.allclose(pred, np.asarray(restored.predict(x[:2])))
    assert DEFAULT_RECIPE["learning_rate_schedule"] == "cosine"


def test_residual_activation_default_is_silu_and_relu_differs():
    import jax

    from ashlar.layers import ResidualMLP

    key = jax.random.PRNGKey(0)
    default = ResidualMLP(2, key=key, n_blocks=1, d_block=4)
    relu = ResidualMLP(2, key=key, n_blocks=1, d_block=4, activation="relu")
    assert default.activation == "silu"
    x = jnp.array([[-1.0, 0.5]], dtype=jnp.float32)
    assert not np.allclose(np.asarray(default(x, key=None)), np.asarray(relu(x, key=None)))


def test_softmax_ordinal_head_can_peak_on_one_class():
    import equinox as eqx
    import jax

    from ashlar.heads import SoftmaxHead

    head = SoftmaxHead(4, 1, n_events=1, n_classes=4, key=jax.random.PRNGKey(0))
    head = eqx.tree_at(lambda module: module.output.weight, head, jnp.zeros_like(head.output.weight))
    bias = jnp.zeros_like(head.output.bias).at[0, 2].set(8.0)
    head = eqx.tree_at(lambda module: module.output.bias, head, bias)
    hidden = jnp.ones((3, 1, 4), dtype=jnp.float32)
    probabilities = np.asarray(jnp.exp(head(hidden, key=None)))
    assert probabilities.shape == (3, 1, 4)
    assert np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)
    assert np.all(probabilities.argmax(axis=-1) == 2)
    try:
        ordinal(head="logits", max_epochs=1, patience=0, validation_size=0).fit(np.ones((8, 1)), np.arange(8) % 3)
    except AshlarError:
        return
    raise AssertionError("a logits head was accepted for an ordinal task")


def test_empirical_bayes_cuts_match_the_prior_and_do_not_move():
    import equinox as eqx
    import jax

    prior = np.array([0.02, 0.08, 0.25, 0.35, 0.2, 0.08, 0.02], dtype=np.float64)
    raw = cut_raw_from_class_prior(prior, k=3)
    latent = jnp.zeros((5, 3), dtype=jnp.float32)
    probabilities = np.asarray(jnp.exp(ordinal_log_probabilities(latent, raw)))
    assert np.allclose(probabilities[:, 0, :], prior, atol=1e-4)
    assert np.allclose(probabilities[:, 2, :], prior, atol=1e-4)
    head = OrdinalHead(4, 2, n_events=1, n_classes=7, key=jax.random.PRNGKey(0), class_prior=prior.tolist())
    assert head.freeze_class_prior is True
    assert np.allclose(np.asarray(head.output.bias), 0.0)
    hidden = jnp.ones((6, 2, 4), dtype=jnp.float32)

    def loss(module):
        return -jnp.mean(module(hidden, key=None)[:, :, 3])

    grads = eqx.filter_grad(loss)(head)
    assert np.allclose(np.asarray(grads.cut_raw), 0.0)
    assert float(jnp.sum(jnp.abs(grads.output.weight))) > 0.0
    plain = OrdinalHead(4, 2, n_events=1, n_classes=7, key=jax.random.PRNGKey(0))
    plain_grads = eqx.filter_grad(loss)(plain)
    assert float(jnp.sum(jnp.abs(plain_grads.cut_raw))) > 0.0


def test_ordinal_empirical_prior_embeds_a_category(tmp_path):
    rng = np.random.default_rng(0)
    measure = rng.normal(size=80).astype(np.float32)
    group = np.where(measure > 0, "high", "low").astype(object)
    y = np.where(measure < -0.5, 3, np.where(measure < 0.5, 5, 8)).astype(np.int32)
    model = ordinal(
        max_epochs=3,
        patience=0,
        batch_size=16,
        head_kwargs={"class_prior": "empirical"},
        categorical_embeddings=["group"],
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0},
    )
    model.fit({"measure": measure, "group": group}, y)
    assert model.class_prior_.shape == (3,)
    assert np.isclose(model.class_prior_.sum(), 1.0)
    assert model.core_.head.freeze_class_prior is True
    assert model.core_.categorical_embeddings is not None
    score = np.asarray(model.predict_score({"measure": measure, "group": group}), dtype=np.float64)
    assert np.isfinite(score).all()
    restored = type(model).load(model.save(tmp_path / "prior"))
    assert restored.core_.head.freeze_class_prior is True
    assert np.allclose(score, np.asarray(restored.predict_score({"measure": measure, "group": group}), dtype=np.float64))
    assert ordinal().head_kwargs is None
    assert DEFAULT_RECIPE["learning_rate_schedule"] == "cosine"
    assert DEFAULT_RECIPE["task"] == "regression"
    try:
        TabularEstimator(
            max_epochs=1,
            patience=0,
            validation_size=0.25,
            head_kwargs={"class_prior": "empirical"},
        ).fit(measure[:, None], measure)
    except AshlarError:
        return
    raise AssertionError("a regression fit accepted the ordinal class prior")


def test_ordinal_log_probabilities_respect_order():
    cuts = jnp.array([[-1.0, 0.2, 0.2]], dtype=jnp.float32)
    thresholds = np.asarray(ordered_thresholds(cuts))
    assert np.all(np.diff(thresholds, axis=1) > 0)
    latent = jnp.linspace(-4.0, 4.0, 9, dtype=jnp.float32)[:, None]
    log_probs = ordinal_log_probabilities(latent, cuts)
    probabilities = np.asarray(jnp.exp(log_probs[:, 0, :]))
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    expectation = probabilities @ np.arange(probabilities.shape[1])
    assert np.all(np.diff(expectation) > 0)


def test_ordinal_recipe_returns_labels_and_expected_score(tmp_path):
    params = ordinal().get_params()
    for key, value in ORDINAL_RECIPE.items():
        assert params[key] == value
    assert DEFAULT_RECIPE["task"] == "regression"
    assert DEFAULT_RECIPE["feature_dropout"] == 0.0
    assert params["task"] == "ordinal"
    rng = np.random.default_rng(0)
    x = rng.normal(size=(96, 1)).astype(np.float32)
    y = np.where(x[:, 0] < -0.4, 2, np.where(x[:, 0] < 0.5, 5, 9)).astype(np.int32)
    model = ordinal(
        max_epochs=25,
        patience=8,
        batch_size=16,
        learning_rate=2e-2,
        backbone_kwargs={"n_blocks": 1, "d_block": 16, "dropout": 0.0},
        random_state=0,
    )
    model.fit(x, y)
    labels = np.asarray(model.predict(x))
    probabilities = np.asarray(model.predict_proba(x))
    score = np.asarray(model.predict_score(x), dtype=np.float64)
    assert list(model.classes_) == [2, 5, 9]
    assert set(np.unique(labels)).issubset({2, 5, 9})
    assert probabilities.shape == (len(x), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(score, probabilities @ np.array([2.0, 5.0, 9.0]), atol=1e-5)
    low = float(score[x[:, 0] < -0.8].mean())
    high = float(score[x[:, 0] > 0.8].mean())
    assert high > low + 1.0
    restored = type(model).load(model.save(tmp_path / "ordinal"))
    assert np.allclose(score, np.asarray(restored.predict_score(x), dtype=np.float64), atol=1e-5)
    try:
        ordinal(max_epochs=1, patience=0, validation_size=0).fit(x[:8], np.array(["low"] * 8, dtype=object))
    except AshlarError:
        return
    raise AssertionError("string labels were accepted for an ordinal task")


def test_late_bottleneck_narrows_the_head_and_stays_off_the_residual_default(tmp_path):
    assert DEFAULT_BOTTLENECK_WIDTH == 32
    assert TabularEstimator().bottleneck is None
    assert DEFAULT_RECIPE["bottleneck"] is None
    assert ordinal().bottleneck is None
    assert tabm().bottleneck is None
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    y = np.where(x[:, 0] < 0, 3, 7).astype(np.int32)
    common = dict(
        max_epochs=2,
        patience=0,
        validation_size=0,
        batch_size=16,
        backbone_kwargs={"n_blocks": 1, "d_block": 64, "dropout": 0.0},
        random_state=0,
    )
    plain = ordinal(**common)
    plain.fit(x, y)
    assert plain.core_.bottleneck is None
    assert int(plain.core_.backbone.d_out) == 64
    assert plain.core_.head.output.weight.shape == (1, 64, 1)

    narrow = ordinal(**common, bottleneck=True)
    narrow.fit(x, y)
    assert int(narrow.core_.bottleneck.width) == 32
    assert int(narrow.core_.backbone.d_out) == 64
    assert narrow.core_.head.output.weight.shape == (1, 32, 1)
    hidden = jnp.ones((4, 1, 64), dtype=jnp.float32)
    projected = narrow.core_.bottleneck(hidden)
    assert projected.shape == (4, 1, 32)
    assert bool(jnp.all(projected >= 0))
    restored = type(narrow).load(narrow.save(tmp_path / "bottleneck"))
    assert restored.bottleneck == 32
    assert restored.core_.head.output.weight.shape == (1, 32, 1)
    assert np.allclose(narrow.predict_score(x), restored.predict_score(x), atol=1e-5)

    ensemble = tabm(
        task="ordinal",
        prediction_policy="mean_probabilities",
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0, "k": 2},
        bottleneck=4,
        max_epochs=1,
        patience=0,
        validation_size=0,
        batch_size=8,
        random_state=0,
    )
    ensemble.fit(x, y)
    assert int(ensemble.core_.backbone.d_out) == 8
    assert int(ensemble.core_.bottleneck.width) == 4
    assert ensemble.core_.head.output.weight.shape == (2, 4, 1)

    wide = ordinal(**common, bottleneck=64)
    try:
        wide.fit(x, y)
    except ShapeError:
        pass
    else:
        raise AssertionError("a bottleneck as wide as the backbone was accepted")
    bad = ordinal(**common, bottleneck=0)
    try:
        bad.fit(x, y)
    except AshlarError:
        pass
    else:
        raise AssertionError("bottleneck width 0 was accepted")


def test_histogram_target_is_unimodal_and_ordinal_only(tmp_path):
    assert DEFAULT_RECIPE["histogram_variance"] is None
    assert ordinal().histogram_variance is None
    probs = np.asarray(gaussian_bin_probabilities(jnp.array([3]), 7, 0.5))
    assert probs.shape == (1, 7)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    center = probs[0]
    assert center[3] > center[2] > center[1] > center[0]
    assert center[3] > center[4] > center[5] > center[6]
    assert center[3] < 0.8
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    y = np.where(x[:, 0] < 0, 3, 7).astype(np.int32)
    common = dict(
        max_epochs=2,
        patience=0,
        validation_size=0,
        batch_size=16,
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0},
        random_state=0,
    )
    try:
        ordinal(**common, histogram_variance=0.5).fit(x, y)
    except AshlarError:
        pass
    else:
        raise AssertionError("histogram variance was accepted on the cumulative head")
    try:
        TabularEstimator(task="regression", histogram_variance=0.5, max_epochs=1, patience=0, validation_size=0).fit(x, y)
    except AshlarError:
        pass
    else:
        raise AssertionError("histogram variance was accepted for regression")
    model = ordinal(**common, head="softmax", histogram_variance=0.5)
    model.fit(x, y)
    score = np.asarray(model.predict_score(x), dtype=np.float64)
    assert score.shape == (len(x),)
    assert np.isfinite(score).all()
    restored = type(model).load(model.save(tmp_path / "histogram"))
    assert restored.histogram_variance == 0.5
    assert restored.head == "softmax"
    assert np.allclose(score, np.asarray(restored.predict_score(x), dtype=np.float64), atol=1e-5)


def test_soft_stage_histogram_passes_class_gradients_and_stays_off_by_default(tmp_path):
    import equinox as eqx
    import jax

    assert SOFT_STAGE_SHRINKAGE == 0.3
    assert DEFAULT_RECIPE["soft_stages"] is None
    assert ordinal().soft_stages is None
    assert tabm().soft_stages is None
    module = SoftStageHistogram(3, 4, 8, shrinkage=SOFT_STAGE_SHRINKAGE, key=jax.random.PRNGKey(0))
    raw = jax.random.normal(jax.random.PRNGKey(1), (6, 3))
    assert np.allclose(np.asarray(module(raw)), 0.0)
    module = eqx.tree_at(lambda item: item.left_logits, module, jnp.zeros((8, 4)).at[:, 0].set(1.0))
    module = eqx.tree_at(lambda item: item.right_logits, module, jnp.zeros((8, 4)).at[:, 1].set(1.0))

    def module_loss(item, features):
        log_probs = jax.nn.log_softmax(item(features), axis=-1)
        labels = jnp.array([0, 1, 2, 3, 0, 1])
        return -jnp.mean(log_probs[jnp.arange(features.shape[0]), labels])

    module_grads = eqx.filter_grad(module_loss)(module, raw)
    for name in ("feature_logits", "threshold", "log_temperature", "left_logits", "right_logits"):
        assert np.any(np.abs(np.asarray(getattr(module_grads, name))) > 0.0)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2)).astype(np.float32)
    y = np.where(x[:, 0] < 0, 3, 7).astype(np.int32)
    common = dict(
        max_epochs=1,
        patience=0,
        validation_size=0,
        batch_size=16,
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0},
        random_state=0,
    )
    try:
        TabularEstimator(task="regression", soft_stages=2, max_epochs=1, patience=0, validation_size=0).fit(x, y)
    except AshlarError:
        pass
    else:
        raise AssertionError("soft stages were accepted for regression")
    model = ordinal(**common, soft_stages=2)
    model.fit(x, y)
    assert model.core_.soft_histogram is not None
    assert int(model.core_.soft_histogram.n_stages) == 2
    assert float(model.core_.soft_histogram.shrinkage) == 0.3
    numeric, categorical = model.prepare(x)
    numeric = jnp.asarray(numeric)
    categorical = jnp.asarray(categorical)
    y_enc = jnp.asarray(np.searchsorted(model.classes_, y), dtype=jnp.int32)
    core = eqx.tree_at(
        lambda item: item.soft_histogram.left_logits,
        model.core_,
        jnp.zeros_like(model.core_.soft_histogram.left_logits).at[:, 0].set(0.5),
    )
    core = eqx.tree_at(
        lambda item: item.soft_histogram.right_logits,
        core,
        jnp.zeros_like(core.soft_histogram.right_logits).at[:, 1].set(0.5),
    )

    def core_loss(item):
        log_probs = item(
            numeric,
            categorical,
            jnp.zeros((len(y), item.n_embedding_columns), dtype=jnp.int32),
            jnp.zeros((len(y), item.n_set_columns, item.set_length), dtype=jnp.int32),
            jnp.zeros((len(y), item.n_set_columns, item.set_length), dtype=jnp.float32),
            key=None,
        )
        return ordinal_objective(log_probs, y_enc, jnp.ones((len(y),), dtype=jnp.float32))

    core_grads = eqx.filter_grad(core_loss)(core)
    for name in ("feature_logits", "threshold", "log_temperature", "left_logits", "right_logits"):
        assert np.any(np.abs(np.asarray(getattr(core_grads.soft_histogram, name))) > 0.0)
    assert np.any(np.abs(np.asarray(core_grads.backbone.input_layer.weight)) > 0.0)
    restored = type(model).load(model.save(tmp_path / "soft-stages"))
    assert restored.soft_stages == 2
    assert np.allclose(model.predict_score(x), restored.predict_score(x), atol=1e-5)


def test_rare_class_stays_in_training_and_the_rest_is_stratified():
    import pytest
    from sklearn.model_selection import train_test_split

    from ashlar.estimator import _split_indices

    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    train, val = _split_indices(len(y), y, mode="iid", validation_size=0.25, groups=None, random_state=0, task="ordinal")
    expected_train, expected_val = train_test_split(
        np.arange(len(y)), test_size=0.25, random_state=0, stratify=y
    )
    assert np.array_equal(train, np.asarray(expected_train))
    assert np.array_equal(val, np.asarray(expected_val))

    rare = np.array([0, 0, 0, 0, 1, 1, 1, 1, 9])
    train, val = _split_indices(len(rare), rare, mode="iid", validation_size=0.25, groups=None, random_state=0, task="classification")
    assert 8 in set(train.tolist())
    assert 8 not in set(val.tolist())
    assert set(rare[train].tolist()) >= {0, 1, 9}
    assert set(rare[val].tolist()) == {0, 1}
    common = np.arange(8)
    common_train, common_val = train_test_split(
        common, test_size=0.25, random_state=0, stratify=rare[:8]
    )
    assert np.array_equal(train[:-1], np.asarray(common_train))
    assert np.array_equal(val, np.asarray(common_val))

    with pytest.raises(AshlarError, match="nothing left to stratify"):
        _split_indices(3, np.array([4, 5, 6]), mode="iid", validation_size=0.25, groups=None, random_state=0, task="ordinal")

    x = np.arange(len(rare), dtype=np.float32)[:, None]
    model = ordinal(max_epochs=1, patience=0, validation_size=0.25, batch_size=4, random_state=0)
    model.fit(x, rare)
    assert 9 in set(np.asarray(model.classes_).tolist())
    assert TabularEstimator().task == "regression"


def test_expected_label_squared_error_sits_beside_the_ordinal_likelihood(tmp_path):
    import jax
    import pytest

    log_probs = jnp.log(jnp.array([[[0.25, 0.75]]], dtype=jnp.float32))
    y = jnp.array([0], dtype=jnp.int32)
    weights = jnp.ones((1,), dtype=jnp.float32)
    values = np.array([3.0, 7.0], dtype=np.float64)
    nll = float(ordinal_objective(log_probs, y, weights))
    penalty = float(ordinal_expectation_squared_error(log_probs, y, weights, values))
    assert abs(penalty - 9.0) < 1e-5
    mixed = make_ordinal_expectation_objective(values, 1.0)
    assert abs(float(mixed(log_probs, y, weights)) - (nll + 9.0)) < 1e-4
    half = make_ordinal_expectation_objective(values, 0.5)
    assert abs(float(half(log_probs, y, weights)) - (nll + 4.5)) < 1e-4
    grads = jax.grad(lambda logits: mixed(logits, y, weights))(log_probs)
    assert np.any(np.abs(np.asarray(grads)) > 0.0)

    assert ordinal().expectation_weight is None
    assert TabularEstimator().expectation_weight is None
    with pytest.raises(AshlarError, match="expectation_weight"):
        TabularEstimator(task="regression", expectation_weight=1.0)._check_config()
    with pytest.raises(AshlarError, match="histogram_variance"):
        ordinal(expectation_weight=1.0, head="softmax", histogram_variance=0.5)._check_config()

    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 2)).astype(np.float32)
    y = np.where(x[:, 0] < 0, 3, 7).astype(np.int32)
    model = ordinal(
        expectation_weight=1.0,
        max_epochs=1,
        patience=0,
        validation_size=0,
        batch_size=8,
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0},
        random_state=0,
    )
    model.fit(x, y)
    assert model.expectation_weight == 1.0
    assert model.standardize_target is False
    score = np.asarray(model.predict_score(x[:4]), dtype=np.float64)
    assert score.shape == (4,)
    assert np.isfinite(score).all()
    restored = type(model).load(model.save(tmp_path / "expectation"))
    assert restored.expectation_weight == 1.0
    assert np.allclose(score, np.asarray(restored.predict_score(x[:4]), dtype=np.float64), atol=1e-5)


def test_robust_scale_and_default_categorical_embeddings(tmp_path):
    import jax
    import pytest

    from ashlar.preprocess import FeatureEncoder

    assert DEFAULT_RECIPE["numeric_scaling"] == "robust"
    assert DEFAULT_RECIPE["categorical_embeddings"] is None
    assert DEFAULT_RECIPE["num_embeddings"] == "piecewise"
    assert tabm().numeric_scaling == "robust"
    assert tabm(numeric_scaling="quantile_tabr").numeric_scaling == "quantile_tabr"

    column = np.arange(1, 10, dtype=np.float64).reshape(-1, 1)
    robust = FeatureEncoder.fit(column)
    assert robust.numeric_transform.scaling == "robust"
    assert abs(float(robust.columns[0].median) - 5.0) < 1e-6
    assert abs(float(robust.columns[0].mean) - 5.0) < 1e-6
    assert abs(float(robust.columns[0].std) - 4.0) < 1e-6

    def clip(z):
        return z / np.sqrt(1.0 + (z * z) / 9.0)

    query = jnp.asarray([[5.0], [9.0], [1000.0], [np.nan]], dtype=jnp.float32)
    got = np.asarray(robust.numeric_transform(query), dtype=np.float64).reshape(-1)
    assert abs(got[0] - 0.0) < 1e-5
    assert abs(got[1] - clip(1.0)) < 1e-5
    assert abs(got[2] - clip((1000.0 - 5.0) / 4.0)) < 1e-4
    assert abs(got[2]) < 3.0
    assert abs(got[3] - 0.0) < 1e-5
    again = FeatureEncoder.from_json(robust.to_json())
    assert again.numeric_scaling == "robust"
    assert np.allclose(again.numeric_transform(query), robust.numeric_transform(query))

    standard = FeatureEncoder.fit(column, numeric_scaling="standard")
    filled = column.reshape(-1)
    expected = (5.0 - filled.mean()) / filled.std()
    standard_got = float(standard.numeric_transform(jnp.asarray([[5.0]], dtype=jnp.float32))[0, 0])
    assert standard.numeric_transform.scaling == "standard"
    assert abs(standard_got - expected) < 1e-5
    assert [leaf.shape for leaf in jax.tree.leaves(robust.numeric_transform)] == [
        leaf.shape for leaf in jax.tree.leaves(standard.numeric_transform)
    ]
    old_schema = standard.to_json()
    del old_schema["numeric_scaling"]
    restored = FeatureEncoder.from_json(old_schema)
    assert restored.numeric_scaling == "standard"
    assert np.allclose(
        restored.numeric_transform(jnp.asarray([[5.0]], dtype=jnp.float32)),
        standard.numeric_transform(jnp.asarray([[5.0]], dtype=jnp.float32)),
    )

    constant = FeatureEncoder.fit(np.ones((6, 1)))
    assert abs(float(constant.columns[0].std) - 1.0) < 1e-6
    constant_out = np.asarray(constant.numeric_transform(jnp.ones((1, 1), dtype=jnp.float32))).reshape(-1)
    assert abs(float(constant_out[0])) < 1e-6
    tied = FeatureEncoder.fit(np.array([[0.0], [0.0], [0.0], [0.0], [10.0]]))
    assert abs(float(tied.columns[0].median) - 0.0) < 1e-6
    assert abs(float(tied.columns[0].std) - 5.0) < 1e-6

    tail = np.arange(1, 11, dtype=np.float32)[:, None]
    tail[-1, 0] = 1000.0
    held = _regression(split="temporal", validation_size=0.1, max_epochs=1, patience=0)
    held.fit(tail, np.zeros(len(tail), dtype=np.float32))
    assert abs(float(held.numeric_transform_.std[0]) - 4.0) < 1e-4
    assert abs(float(held.numeric_transform_.mean[0]) - 5.0) < 1e-4

    value = np.arange(8, dtype=np.float32)
    city = np.array(["ny", "sf", "ny", "sf", "ny", "sf", "la", "la"], dtype=object)
    color = np.array(["red", "blue"] * 4, dtype=object)
    y = value.copy()
    embedded = _regression(max_epochs=1, patience=0, validation_size=0)
    embedded.fit({"value": value, "city": city}, y)
    assert embedded.categorical_embeddings is None
    assert embedded.embedded_columns_ == ["city"]
    assert embedded.encoder_.categorical_widths == []
    assert embedded.core_.categorical_embeddings is not None
    unseen = embedded.predict({"value": np.array([np.nan], dtype=np.float32), "city": np.array(["boston"], dtype=object)})
    assert np.isfinite(np.asarray(unseen)).all()
    reloaded = type(embedded).load(embedded.save(tmp_path / "embed"))
    assert reloaded.embedded_columns_ == ["city"]
    assert reloaded.numeric_scaling == "robust"
    assert np.allclose(embedded.predict({"value": value[:2], "city": city[:2]}), reloaded.predict({"value": value[:2], "city": city[:2]}))

    one_hot = _regression(max_epochs=1, patience=0, validation_size=0, categorical_embeddings=[])
    one_hot.fit({"value": value, "city": city, "color": color}, y)
    assert one_hot.embedded_columns_ == []
    assert one_hot.encoder_.categorical_widths
    assert one_hot.core_.categorical_embeddings is None

    partial = _regression(max_epochs=1, patience=0, validation_size=0, categorical_embeddings=["city"])
    partial.fit({"value": value, "city": city, "color": color}, y)
    assert partial.embedded_columns_ == ["city"]
    assert partial.encoder_.categorical_widths

    codes = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    integer = _regression(max_epochs=1, patience=0, validation_size=0)
    integer.fit({"value": value, "code": codes}, y)
    assert integer.embedded_columns_ == []
    assert integer.encoder_.n_numeric == 2
    named = _regression(max_epochs=1, patience=0, validation_size=0, categorical=["code"])
    named.fit({"value": value, "code": codes}, y)
    assert named.embedded_columns_ == ["code"]

    tags = np.empty(8, dtype=object)
    for index, item in enumerate([["red"], ["blue"]] * 4):
        tags[index] = item
    pooled = _regression(max_epochs=1, patience=0, validation_size=0, set_features=["tags"])
    pooled.fit({"value": value, "tags": tags}, y)
    assert pooled.embedded_columns_ == []
    assert pooled.core_.set_embeddings is not None

    refused = _regression(numeric_scaling="nope", max_epochs=1, patience=0, validation_size=0)
    with pytest.raises(AshlarError, match="robust"):
        refused.fit(column.astype(np.float32), np.zeros(len(column), dtype=np.float32))
