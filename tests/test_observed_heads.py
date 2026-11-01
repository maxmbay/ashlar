"""Observed labels on more heads: noise by quadrature, cure fractions, and group totals on a grid."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ashlar import ASRClassifier, ASROrdinalClassifier, ASRRegressor, observed
from ashlar.distributions import Cured, Hurdle, LogNormal, NegativeBinomial, Normal, NormalNoise, StudentT
from ashlar.errors import AshlarError
from ashlar.objectives import grid_total_log_likelihood

INF = np.inf


def test_quadrature_noise_matches_the_closed_form_for_a_normal():
    base = Normal(jnp.zeros(3), jnp.ones(3))
    value = jnp.asarray([0.0, 1.5, -3.0])
    exact = base.convolve_normal(0.7)
    quadrature = NormalNoise(base, 0.7)
    assert np.allclose(quadrature.log_prob(value), exact.log_prob(value), atol=1e-4)
    assert np.allclose(quadrature.log_interval(value - 0.5, value + 1.0), exact.log_interval(value - 0.5, value + 1.0), atol=1e-4)
    assert np.allclose(quadrature.cdf(value), exact.cdf(value), atol=1e-5)
    # No noise leaves the distribution as it was.
    assert np.allclose(NormalNoise(base, 0.0).log_prob(value), base.log_prob(value), atol=1e-6)
    assert isinstance(StudentT(jnp.zeros(2), jnp.ones(2), jnp.full(2, 4.0)).convolve_normal(0.3), NormalNoise)


def test_a_hurdle_with_noise_keeps_the_zero_mass_on_rows_without_noise():
    hurdle = Hurdle(jnp.full(3, 0.3), LogNormal(jnp.zeros(3), jnp.full(3, 0.5)))
    value = jnp.asarray([0.0, 1.0, 2.0])
    assert np.allclose(hurdle.convolve_normal(0.0).log_prob(value), hurdle.log_prob(value), atol=1e-5)
    noisy = hurdle.convolve_normal(0.2)
    # A recorded value below zero is possible, and the density integrates to one.
    assert np.isfinite(noisy.log_prob(jnp.asarray([-0.3, 0.0, 1.0]))).all()
    grid = jnp.linspace(-2.0, 30.0, 40_001)
    mass = np.trapezoid(np.exp(np.asarray(Hurdle(jnp.full((1,), 0.3), LogNormal(jnp.zeros(1), jnp.full(1, 0.5))).convolve_normal(0.2).log_prob(grid[:, None]))), np.asarray(grid), axis=0)
    assert np.allclose(mass, 1.0, atol=2e-3)
    assert np.allclose(noisy.mean(), hurdle.mean(), atol=1e-5)


@pytest.mark.parametrize("task", ["student_t", "flow", "histogram", "hurdle"])
def test_noisy_labels_fit_on_the_quadrature_heads(task):
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = np.exp(0.3 * x[:, 0]) if task == "hurdle" else 1.0 + x[:, 0] + rng.normal(size=n)
    if task == "hurdle":
        y = np.where(rng.random(n) < 0.3, 0.0, y)
    noise = np.where(np.arange(n) % 2 == 0, 0.3, 0.0)
    label = observed.noisy(y + noise * rng.normal(size=n), noise)
    model = ASRRegressor(task=task, max_epochs=2, random_state=0).fit(x, label)
    assert np.isfinite(model.log_prob(x[:20], label[:20])).all()


def test_heads_without_quadrature_still_refuse_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(100, 2)).astype(np.float32)
    label = observed.noisy(rng.poisson(2.0, size=100).astype(float), 0.5)
    with pytest.raises(AshlarError, match="noisy"):
        ASRRegressor(task="poisson", max_epochs=1).fit(x, label)


def test_a_cured_time_is_defective():
    cured = Cured(Normal(jnp.zeros(2), jnp.ones(2)), jnp.asarray([0.2, 0.6]))
    assert np.allclose(cured.cdf(jnp.asarray([50.0, 50.0])), [0.8, 0.4], atol=1e-6)
    # A row censored at t: never, or later than t.
    expected = np.log(np.asarray([0.2, 0.6]) + np.asarray([0.8, 0.4]) * 0.5)
    assert np.allclose(cured.log_survival(jnp.zeros(2)), expected, atol=1e-5)
    assert np.isinf(np.asarray(cured.quantile(jnp.asarray([0.9, 0.5])))).all()
    assert np.isfinite(np.asarray(cured.quantile(jnp.asarray([0.5, 0.2])))).all()


def test_the_cure_head_fits_censored_times_and_refuses_the_other_heads():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=(n, 2)).astype(np.float32)
    never = rng.random(n) < 0.4
    time = np.where(never, INF, 1.0 + 0.5 * x[:, 0] + 0.5 * rng.normal(size=n))
    limit = rng.uniform(0.0, 4.0, size=n)
    label = observed.censored(np.minimum(time, limit), time > limit)
    model = ASRRegressor(task="gaussian", head_kwargs={"cure": True}, max_epochs=3, random_state=0).fit(x, label)
    assert np.isfinite(model.log_prob(x[:20], label[:20])).all()
    with pytest.raises(AshlarError, match="cure"):
        ASRRegressor(task="poisson", head_kwargs={"cure": True}, max_epochs=1).fit(x, np.round(limit))


def test_grid_totals_match_the_exact_sum_of_normals():
    rows = 6
    loc = jnp.linspace(-1.0, 1.0, rows)[:, None]
    scale = jnp.linspace(0.5, 1.5, rows)[:, None]
    dist = Normal(loc, scale)
    segment = jnp.asarray([0, 0, 0, 1, 1, 1])
    totals = jnp.asarray([0.3, 2.0])
    observation = jnp.stack([totals, totals, jnp.full(2, -INF), jnp.full(2, INF)], axis=1)
    value = grid_total_log_likelihood(dist, segment, 2, 3, observation, discrete=False)
    sums = Normal(jax_sum(loc, segment), jnp.sqrt(jax_sum(jnp.square(scale), segment)))
    assert np.allclose(value, sums.log_prob(totals[:, None]), atol=1e-3)
    # An interval of the total is the probability of the sum's interval.
    observation = jnp.stack([totals - 1.0, totals + 0.5, jnp.full(2, -INF), jnp.full(2, INF)], axis=1)
    value = grid_total_log_likelihood(dist, segment, 2, 3, observation, discrete=False)
    assert np.allclose(value, sums.log_interval(totals[:, None] - 1.0, totals[:, None] + 0.5), atol=1e-3)


def jax_sum(values, segment):
    import jax

    return jax.ops.segment_sum(values, segment, num_segments=2)


def test_grid_totals_of_one_count_row_are_its_mass():
    dist = NegativeBinomial(jnp.asarray([[3.0]]), jnp.asarray([[2.0]]))
    for count in (0.0, 2.0, 9.0):
        observation = jnp.asarray([[count, count, -INF, INF]])
        value = grid_total_log_likelihood(dist, jnp.zeros(1, jnp.int32), 1, 1, observation, discrete=True)
        assert np.allclose(value, dist.log_prob(jnp.asarray([[count]])), atol=1e-4)


@pytest.mark.parametrize("task", ["negative_binomial", "student_t"])
def test_group_totals_fit_on_grid_heads(task):
    rng = np.random.default_rng(0)
    groups, size = 60, 5
    n = groups * size
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = rng.poisson(np.exp(0.5 + 0.5 * x[:, 0])).astype(float)
    group = np.repeat(np.arange(groups), size)
    label = observed.total(np.bincount(group, weights=y)[group], group)
    model = ASRRegressor(task=task, max_epochs=2, split="group", random_state=0).fit(x, label, groups=group)
    assert np.isfinite(model.score(x, label))
    with pytest.raises(AshlarError):
        ASRRegressor(task=task, max_epochs=1, split="group").fit(x, observed.noisy(label, 0.5), groups=group)


@pytest.mark.slow
def test_group_totals_on_a_grid_recover_the_row_mean():
    rng = np.random.default_rng(0)
    weeks, days = 500, 7
    n = weeks * days
    x = rng.normal(size=(n, 2)).astype(np.float32)
    mu = np.exp(1.0 + 0.6 * x[:, 0])
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))
    week = np.repeat(np.arange(weeks), days)
    label = observed.total(np.bincount(week, weights=y)[week], week)
    model = ASRRegressor(task="negative_binomial", max_epochs=40, split="group", random_state=0).fit(x, label, groups=week)
    assert np.corrcoef(model.predict(x), mu)[0, 1] > 0.95


@pytest.mark.slow
@pytest.mark.parametrize("task", ["student_t", "flow"])
def test_noise_is_subtracted_by_the_quadrature_heads(task):
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    y = 1.0 + x[:, 0] + 0.5 * rng.standard_t(4, size=n)
    noise = rng.uniform(0.2, 1.5, n)
    recorded = y + noise * rng.normal(size=n)
    told = ASRRegressor(task=task, max_epochs=30, random_state=0).fit(x[:2500], observed.noisy(recorded[:2500], noise[:2500]))
    naive = ASRRegressor(task=task, max_epochs=30, random_state=0).fit(x[:2500], recorded[:2500])
    assert told.log_prob(x[2500:], y[2500:]).mean() > naive.log_prob(x[2500:], y[2500:]).mean() + 0.05


def test_one_of_reads_sets_masks_and_frames():
    label = observed.one_of([{"cat", "dog"}, "bird", None])
    assert label.classes.tolist() == ["bird", "cat", "dog"]
    assert label.mask.tolist() == [[False, True, True], [True, False, False], [True, True, True]]
    frame = pd.DataFrame({"dog": [True, False], "cat": [True, True]})
    from_frame = observed.one_of(frame)
    assert from_frame.classes.tolist() == ["cat", "dog"] and from_frame.mask.tolist() == [[True, True], [True, False]]
    assert observed.one_of(["a"], classes=["a", "b"]).mask.tolist() == [[True, False]]
    with pytest.raises(AshlarError):
        observed.one_of(np.zeros((2, 3), bool))
    with pytest.raises(AshlarError):
        observed.one_of(["a", "z"], classes=["a", "b"])


def _classes(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2)).astype(np.float32)
    logits = np.stack([2 * x[:, 0], np.zeros(n), -2 * x[:, 0]], axis=1)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=row) for row in p])
    return x, y


def test_a_classifier_trains_on_sets_of_classes():
    x, y = _classes()
    names = np.array(["low", "mid", "high"])[y]
    # Half the rows only say whether the class is low or not.
    sets = [name if i % 2 == 0 else ({"low"} if name == "low" else {"mid", "high"}) for i, name in enumerate(names)]
    model = ASRClassifier(max_epochs=20, random_state=0).fit(x, observed.one_of(sets))
    assert model.classes_.tolist() == ["high", "low", "mid"]
    probabilities = model.predict_proba(x)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    low = probabilities[:, 1]
    assert np.corrcoef(low, x[:, 0])[0, 1] > 0.5
    with pytest.raises(AshlarError, match="class_weight"):
        ASRClassifier(class_weight="balanced", max_epochs=1).fit(x, observed.one_of(sets))


def test_an_ordinal_classifier_trains_on_ranges_of_classes():
    x, y = _classes()
    lower = np.where(np.arange(len(y)) % 2 == 0, y, np.maximum(y - 1, 0)).astype(float)
    upper = np.where(np.arange(len(y)) % 2 == 0, y, y).astype(float)
    for head in ("cumulative_logit", "softmax"):
        kwargs = {} if head == "cumulative_logit" else {"head": "softmax"}
        model = ASROrdinalClassifier(max_epochs=20, random_state=0, **kwargs).fit(x, observed.interval(lower, upper))
        assert model.classes_.tolist() == [0.0, 1.0, 2.0]
        probabilities = model.predict_proba(x)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert np.corrcoef(model.predict_score(x), y)[0, 1] > 0.5
    with pytest.raises(AshlarError, match="truncated"):
        ASROrdinalClassifier(max_epochs=1).fit(x, observed.truncated(observed.interval(lower, upper), lower=0))
