"""Scikit-learn tags and score, numeric embeddings, distribution tasks, and the scanned loop."""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score

from ashlar import ASRModel, ASRClassifier, ASROrdinalClassifier
from ashlar.errors import AshlarError
from ashlar.layers import PiecewiseLinearEmbeddings, quantile_edges
from ashlar.train import chunked_weighted

import jax.numpy as jnp


def _regression(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    y = np.sin(2 * X[:, 0]) + X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_tags_follow_the_task():
    assert is_classifier(ASRClassifier())
    assert is_classifier(ASROrdinalClassifier())
    assert is_regressor(ASRModel())
    assert is_regressor(ASRModel(task="mixture"))
    assert is_classifier(clone(ASRClassifier()))


@pytest.mark.slow
def test_score_is_accuracy_or_r2_and_scorers_read_probabilities():
    X, y = _regression()
    labels = (y > np.median(y)).astype(int)
    model = ASRClassifier(max_epochs=3)
    auc = cross_val_score(model, X, labels, cv=3, scoring="roc_auc")
    assert np.all(auc > 0.6)
    fitted = model.fit(X, labels)
    assert fitted.score(X, labels) == pytest.approx(np.mean(fitted.predict(X) == labels))
    regressor = ASRModel(max_epochs=5).fit(X, y)
    assert regressor.score(X, y) > 0.3


@pytest.mark.parametrize("kind", ["periodic", "piecewise"])
def test_numeric_embeddings_fit_save_and_load(kind, tmp_path):
    X, y = _regression()
    model = ASRModel(num_embeddings=kind, num_embedding_dim=8, max_epochs=3).fit(X, y)
    model.save(tmp_path / "model")
    restored = ASRModel.load(tmp_path / "model")
    assert np.allclose(model.predict(X), restored.predict(X), atol=1e-5)


@pytest.mark.slow
def test_piecewise_edges_are_frozen_and_start_linear():
    X, y = _regression()
    model = ASRModel(num_embeddings="piecewise", num_embedding_dim=4, max_epochs=3, weight_decay=0.1)
    model.fit(X, y)
    encoded = model.encoder_.transform(X[: int(0.8 * len(X))])
    numeric_z = np.asarray(model.numeric_transform_(encoded.numeric_raw))
    edges = np.asarray(model.core_.embeddings.edges)
    assert edges.shape[1] == 49
    assert np.all(np.diff(edges, axis=1) >= 0)
    fresh = PiecewiseLinearEmbeddings(quantile_edges(numeric_z, 4), 3, key=__import__("jax").random.PRNGKey(0))
    z = jnp.asarray(numeric_z[:5])
    linear = fresh.linear_bias + fresh.linear_weight * z[..., None]
    assert np.allclose(np.asarray(fresh(z)), np.asarray(linear).reshape(5, -1))


def test_piecewise_encoding_is_fill_fraction():
    layer = PiecewiseLinearEmbeddings(jnp.asarray([[0.0, 1.0, 2.0, 4.0]]), 2, key=__import__("jax").random.PRNGKey(0))
    encoded = np.asarray(layer.encode(jnp.asarray([[-1.0], [0.5], [3.0], [6.0]])))[:, 0, :]
    assert np.allclose(encoded[0], [-1.0, 0.0, 0.0])
    assert np.allclose(encoded[1], [0.5, 0.0, 0.0])
    assert np.allclose(encoded[2], [1.0, 1.0, 0.5])
    assert np.allclose(encoded[3], [1.0, 1.0, 2.0])


@pytest.mark.parametrize("task", ["gaussian", "mixture", "flow", "histogram", "quantile"])
def test_distribution_tasks_predict_on_the_target_scale(task, tmp_path):
    X, y = _regression(600)
    y = 50.0 + 10.0 * y
    model = ASRModel(
        task=task,
        backbone="tabm",
        backbone_kwargs={"n_blocks": 1, "d_block": 16, "k": 4},
        max_epochs=4,
    ).fit(X, y)
    point = model.predict(X)
    assert point.shape == (len(y),)
    assert abs(np.mean(point) - np.mean(y)) < 5.0
    quantiles = model.predict_quantiles(X)
    assert quantiles.shape == (len(y), 3)
    assert np.all(quantiles[:, 0] <= quantiles[:, 2])
    if task == "quantile":
        with pytest.raises(AshlarError):
            model.log_prob(X, y)
    else:
        assert model.sample(X[:7], n=11).shape == (11, 7)
        assert np.isfinite(model.log_prob(X, y)).all()
    model.save(tmp_path / "model")
    restored = ASRModel.load(tmp_path / "model")
    assert np.allclose(point, restored.predict(X), atol=1e-4)


def test_chunked_weighted_matches_one_pass():
    rng = np.random.default_rng(0)
    values = jnp.asarray(rng.normal(size=1000))
    weights = jnp.asarray(rng.uniform(size=1000))

    def weighted(v, w):
        return jnp.sum(v * w) / jnp.sum(w)

    whole = float(weighted(values, weights))
    assert chunked_weighted(weighted, (values, weights), weights, chunk_rows=128) == pytest.approx(whole, rel=1e-5)


def test_cosine_schedule_trains():
    X, y = _regression()
    model = ASRModel(learning_rate_schedule="cosine", max_epochs=4).fit(X, y)
    assert len(model.history_["train_loss"]) == 4


@pytest.mark.slow
def test_cyclical_columns_become_points_on_a_circle(tmp_path):
    rng = np.random.default_rng(0)
    hour = rng.uniform(0, 24, size=600)
    y = np.cos(2 * np.pi * hour / 24) + 0.05 * rng.normal(size=600)
    X = {"hour": hour, "noise": rng.normal(size=600)}
    model = ASRModel(cyclical={"hour": 24}, max_epochs=5).fit(X, y)
    assert [column for column in model.encoder_.feature_names if column.startswith("hour")] == ["hour_sin", "hour_cos"]
    near_midnight = model.predict({"hour": np.array([23.9, 0.1]), "noise": np.zeros(2)})
    assert abs(near_midnight[0] - near_midnight[1]) < 0.2
    model.save(tmp_path / "model")
    restored = ASRModel.load(tmp_path / "model")
    assert np.allclose(model.predict(X), restored.predict(X), atol=1e-5)
    with pytest.raises(AshlarError):
        ASRModel(cyclical={"minute": 60}).fit(X, y)
