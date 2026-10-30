"""Student-t, Poisson, and negative binomial heads, with and without a structural zero."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from ashlar import ASRForecaster, ASRModel, ASRRegressor
from ashlar.distributional import decode, initial_bias, log_density, output_width, point_values, sample_values
from ashlar.errors import AshlarError


def _raw(head, zero_inflated=False, log_mean=0.0, rows=3):
    width = output_width(head, 1, zero_inflated=zero_inflated)
    bias = initial_bias(head, 1, zero_inflated=zero_inflated, log_mean=log_mean)
    noise = 0.3 * jax.random.normal(jax.random.PRNGKey(0), (rows, width))
    return decode(head, jnp.broadcast_to(bias, (rows, width)) + noise, 1)


def test_log_masses_match_scipy():
    mean, zero = _raw("poisson", log_mean=1.0)
    counts = jnp.asarray([[0.0], [2.0], [7.0]])
    expected = stats.poisson.logpmf(np.asarray(counts), np.asarray(mean))
    assert np.allclose(log_density("poisson", (mean, zero), counts), expected, atol=1e-4)
    mean, dispersion, zero = _raw("negative_binomial", log_mean=1.0)
    size = 1.0 / np.asarray(dispersion)
    expected = stats.nbinom.logpmf(np.asarray(counts), size, size / (size + np.asarray(mean)))
    assert np.allclose(log_density("negative_binomial", (mean, dispersion, zero), counts), expected, atol=1e-4)
    loc, scale, df = _raw("student_t")
    value = jnp.asarray([[0.5], [-1.0], [3.0]])
    expected = stats.t.logpdf(np.asarray(value), np.asarray(df), np.asarray(loc), np.asarray(scale))
    assert np.allclose(log_density("student_t", (loc, scale, df), value), expected, atol=1e-4)


@pytest.mark.parametrize("head", ["poisson", "negative_binomial"])
def test_zero_inflation_sums_to_one_and_samples_match_the_mean(head):
    decoded = _raw(head, zero_inflated=True, log_mean=1.5)
    assert float(decoded[-1][0, 0]) > 0.0
    counts = jnp.arange(0.0, 400.0)[:, None, None] * jnp.ones((1, 3, 1))
    mass = jnp.sum(jnp.exp(jax.vmap(lambda value: log_density(head, decoded, value))(counts)), axis=0)
    assert np.allclose(mass, 1.0, atol=1e-3)
    draws = sample_values(head, decoded, jax.random.PRNGKey(1), 20000)
    assert np.all(np.asarray(draws) == np.round(np.asarray(draws)))
    assert np.allclose(np.mean(draws, axis=0), point_values(head, decoded), rtol=0.05)
    plain = _raw(head, log_mean=1.5)
    assert np.all(np.asarray(plain[-1]) == 0.0)


def _counts(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3)).astype(np.float32)
    mean = np.exp(1.0 + 0.7 * X[:, 0])
    y = rng.negative_binomial(2, 2 / (2 + mean)).astype(float)
    y[(X[:, 1] > 0) & (rng.uniform(size=n) < 0.4)] = 0
    return X, y


@pytest.mark.parametrize(
    "task, kwargs",
    [("poisson", None), ("negative_binomial", None), ("negative_binomial", {"zero_inflated": True}), ("student_t", None)],
)
def test_estimator_fits_predicts_and_round_trips(task, kwargs, tmp_path):
    X, y = _counts()
    model = ASRRegressor(task=task, head_kwargs=kwargs, max_epochs=6).fit(X, y)
    assert model.standardize_target == (task == "student_t")
    point = model.predict(X)
    if task != "student_t":  # a Student-t point is its location, below the mean of skewed counts
        assert abs(np.mean(point) - np.mean(y)) < 0.25 * np.mean(y)
    assert np.isfinite(model.log_prob(X, y)).all()
    quantiles = model.predict_quantiles(X[:50], [0.1, 0.5, 0.9])
    assert np.all(np.diff(quantiles, axis=1) >= 0)
    if task != "student_t":
        assert np.all(quantiles == np.round(quantiles))
        assert np.all(model.sample(X[:20], n=30) >= 0)
    model.save(tmp_path / "model")
    restored = ASRModel.load(tmp_path / "model")
    assert np.allclose(point, restored.predict(X), atol=1e-4)
    assert np.allclose(model.log_prob(X, y), restored.log_prob(X, y), atol=1e-4)


def test_negative_binomial_beats_poisson_on_overdispersed_counts():
    X, y = _counts(3000)
    scores = {}
    for task in ("poisson", "negative_binomial"):
        model = ASRRegressor(task=task, max_epochs=10).fit(X[:2000], y[:2000])
        scores[task] = float(np.mean(model.log_prob(X[2000:], y[2000:])))
    assert scores["negative_binomial"] > scores["poisson"]


def test_count_heads_reject_what_they_cannot_model():
    X, y = _counts(200)
    with pytest.raises(AshlarError):
        ASRRegressor(task="poisson", max_epochs=1).fit(X, y + 0.5)
    with pytest.raises(AshlarError):
        ASRRegressor(task="poisson", max_epochs=1).fit(X, -y - 1)
    with pytest.raises(AshlarError):
        ASRRegressor(task="poisson", standardize_target=True, max_epochs=1).fit(X, y)
    with pytest.raises(AshlarError):
        ASRRegressor(task="gaussian", head_kwargs={"zero_inflated": True}, max_epochs=1).fit(X, y)
    with pytest.raises(AshlarError):
        ASRForecaster(head="poisson")


def test_forecaster_takes_the_student_t_head(tmp_path):
    rng = np.random.default_rng(0)
    series = np.sin(np.arange(300) / 5.0) + 0.2 * rng.standard_t(3, size=300)
    model = ASRForecaster(lookback=16, horizon=4, head="student_t", max_epochs=2, patience=1).fit(series)
    assert model.predict(series[-40:]).shape[-1] == 4
    restored = ASRForecaster.load(model.save(tmp_path / "t"))
    future = series[-4:]
    assert np.allclose(model.log_prob(series[-44:-4], future), restored.log_prob(series[-44:-4], future))
