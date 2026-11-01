"""The traceable predict_dist path and attributions of distributional targets."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import ASRRegressor
from ashlar.errors import AshlarError
from ashlar.explain import explain, target_function


def _location_scale(n=3000, seed=0):
    """``loc`` moves the location, ``scale`` the spread, ``noise`` nothing, ``city`` shifts by 1.5."""
    rng = np.random.default_rng(seed)
    X = {
        "loc": rng.normal(size=n).astype(np.float32),
        "scale": rng.uniform(-1, 1, size=n).astype(np.float32),
        "noise": rng.normal(size=n).astype(np.float32),
        "city": rng.choice(["a", "b", "c"], n).astype(object),
    }
    y = 2 * X["loc"] + np.exp(1.2 * X["scale"]) * rng.normal(size=n) + 1.5 * (X["city"] == "a")
    return X, y


def _rows(X, n):
    return {key: value[:n] for key, value in X.items()}


def test_forward_dist_matches_predict_dist_and_has_input_gradients():
    X, y = _location_scale(n=800)
    model = ASRRegressor(task="mixture", max_epochs=3, random_state=0).fit(X, y)
    rows = _rows(X, 6)
    batch = model.encode(rows)
    dist = model.forward_dist(batch.numeric_raw, batch.categorical_onehot, batch.category_index, batch.set_tokens, batch.set_mask)
    np.testing.assert_allclose(np.asarray(dist.mean()), np.asarray(model.predict_dist(rows).mean()), rtol=1e-5, atol=1e-5)

    def q90(numeric):
        out = model.forward_dist(numeric, batch.categorical_onehot, batch.category_index, batch.set_tokens, batch.set_mask)
        return out.quantile(0.9).sum()

    grad = np.asarray(jax.jit(jax.grad(q90))(batch.numeric_raw))
    assert grad.shape == (6, 3)
    assert np.all(np.isfinite(grad))
    assert np.abs(grad).sum() > 0
    numeric = jnp.asarray(batch.numeric_raw).at[0, 1].set(jnp.nan)
    assert float(jax.grad(q90)(numeric)[0, 1]) == 0.0
    raw = model.forward_raw(batch.numeric_raw, batch.categorical_onehot, batch.category_index, batch.set_tokens, batch.set_mask)
    np.testing.assert_allclose(np.asarray(model.dist_from_raw(raw).mean()), np.asarray(dist.mean()), rtol=1e-6)


def test_explain_shapes_and_completeness():
    X, y = _location_scale(n=800)
    model = ASRRegressor(task="gaussian", max_epochs=3, random_state=0).fit(X, y)
    result = model.explain(_rows(X, 20), "quantile", level=0.9, steps=16)
    assert result.values.shape == (20, 4)
    assert result.feature_names == ["loc", "scale", "noise", "city"]
    assert result.kinds == ["numeric", "numeric", "numeric", "categorical"]
    np.testing.assert_allclose(result.prediction, model.predict_quantiles(_rows(X, 20), [0.9]).ravel(), rtol=1e-4, atol=1e-4)
    assert np.max(np.abs(result.numeric_gap())) < 0.05 * np.max(np.abs(result.prediction - result.baseline)) + 1e-3
    assert set(result.importance()) == {"loc", "scale", "noise", "city"}
    linear = model.explain(_rows(X, 20), "mean", method="gradient_x_input")
    assert linear.values.shape == (20, 4)


@pytest.mark.slow
def test_missing_cells_get_zero_attribution():
    X, y = _location_scale(n=800)
    model = ASRRegressor(task="gaussian", max_epochs=3, random_state=0).fit(X, y)
    rows = _rows(X, 10)
    rows["loc"] = rows["loc"].copy()
    rows["loc"][:5] = np.nan
    result = model.explain(rows, "median", steps=4)
    assert np.all(result.values[:5, 0] == 0.0)


@pytest.mark.slow
def test_targets_and_errors():
    with pytest.raises(AshlarError):
        target_function("exceedance")
    with pytest.raises(AshlarError):
        target_function("nope")
    with pytest.raises(AshlarError):
        target_function("quantile", level=1.5)
    X, y = _location_scale(n=600)
    point = ASRRegressor(task="regression", max_epochs=2).fit(X, y)
    with pytest.raises(AshlarError):
        point.explain(_rows(X, 5))
    joint = ASRRegressor(task="multivariate_gaussian", max_epochs=2).fit(X, np.stack([y, -y], axis=1))
    with pytest.raises(AshlarError, match="one number per row"):
        explain(joint, _rows(X, 5), "mean")
    projected = explain(joint, _rows(X, 5), lambda d: d.project(jnp.array([1.0, 1.0])).mean(), steps=2)
    assert projected.values.shape == (5, 4)


@pytest.mark.slow
def test_cyclical_columns_fold_back():
    rng = np.random.default_rng(0)
    hour = rng.integers(0, 24, 800).astype(np.float32)
    X = {"hour": hour, "x": rng.normal(size=800).astype(np.float32)}
    y = np.sin(2 * np.pi * hour / 24) + 0.1 * rng.normal(size=800)
    model = ASRRegressor(task="gaussian", max_epochs=3, cyclical={"hour": 24}).fit(X, y)
    result = model.explain(_rows(X, 8), "median", steps=4)
    assert result.feature_names == ["hour", "x"]
    assert result.values.shape == (8, 2)


@pytest.mark.slow
def test_explain_separates_location_from_spread():
    X, y = _location_scale()
    model = ASRRegressor(task="mixture", max_epochs=30, random_state=0).fit(X, y)
    rows = _rows(X, 300)
    median = model.explain(rows, "median").importance()
    width = model.explain(rows, "interval_width", coverage=0.8).importance()
    spread = model.explain(rows, "std").importance()
    assert next(iter(median)) == "loc"
    assert next(iter(width)) == "scale"
    assert next(iter(spread)) == "scale"
    assert median["city"] > 3 * median["noise"]
    assert width["scale"] > 5 * width["loc"]
