"""Coarse labels as likelihoods: the label builders, the interval math on every head, and the estimator paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from ashlar import ASRRegressor, observed
from ashlar.distributional import flow_output_count
from ashlar.distributions import (
    AffineTransformed,
    Flow,
    Gamma,
    Histogram,
    Hurdle,
    LogNormal,
    Mixture,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)
from ashlar.errors import AshlarError

INF = np.inf


# ---------------------------------------------------------------- labels


def test_label_builders_fill_the_fields():
    values = np.array([1.0, 2.0, 3.0])
    label = observed.censored(values, [False, True, False])
    assert label.dtype == observed.DTYPE and label.shape == (3,)
    assert np.array_equal(label["lower"], values)
    assert np.array_equal(label["upper"], [1.0, INF, 3.0])
    left = observed.censored(values, np.array([1, 0, 0]), side="left")
    assert np.array_equal(left["lower"], [-INF, 2.0, 3.0])
    assert np.array_equal(observed.rounded(values, 0.5)["lower"], values - 0.25)
    times = observed.survival([5.0, 7.0], event=[True, False], entry=[1.0, 2.0])
    assert np.array_equal(times["upper"], [5.0, INF]) and np.array_equal(times["truncate_lower"], [1.0, 2.0])
    both = observed.truncated(observed.truncated(values, lower=0.0), upper=[5.0, 6.0, 7.0])
    assert np.array_equal(both["truncate_lower"], [0.0] * 3) and np.array_equal(both["truncate_upper"], [5.0, 6.0, 7.0])
    noisy = observed.noisy(observed.noisy(values, 3.0), 4.0)
    assert np.allclose(noisy["noise"], 5.0)
    totals = observed.total([10.0, 10.0, 4.0], group=[0, 0, -1])
    assert np.array_equal(observed.group_sizes(totals), [2, 2, 1])
    assert np.allclose(observed.representative(totals), [5.0, 5.0, 4.0])
    stacked = observed.concatenate([values, label])
    assert len(stacked) == 6 and observed.is_observed(stacked) and not observed.is_observed(values)
    assert observed.fields(label).shape == (3, 6)
    assert np.allclose(observed.representative(observed.interval([0.0, -INF, 1.0], [2.0, 4.0, INF])), [1.0, 4.0, 1.0])


@pytest.mark.parametrize(
    "build",
    [
        lambda: observed.interval([2.0], [1.0]),
        lambda: observed.interval([np.nan], [1.0]),
        lambda: observed.interval([INF], [INF]),
        lambda: observed.truncated([3.0], lower=4.0),
        lambda: observed.truncated([3.0], lower=3.0, upper=3.0),
        lambda: observed.noisy([1.0], -1.0),
        lambda: observed.rounded([1.0], 0.0),
        lambda: observed.censored([1.0], [2]),
        lambda: observed.censored([1.0], [True], side="up"),
        lambda: observed.total([3.0, 4.0], group=[0, 0]),
        lambda: observed.grouped([3.0], group=[-2]),
    ],
)
def test_impossible_labels_are_refused(build):
    with pytest.raises(AshlarError):
        build()


def test_scikit_learn_splits_a_label_like_any_target():
    from sklearn.model_selection import KFold, train_test_split

    label = observed.censored(np.arange(10.0), np.arange(10) % 3 == 0)
    x = np.arange(20.0).reshape(10, 2)
    _, _, y_train, y_test = train_test_split(x, label, test_size=0.3, random_state=0)
    assert y_train.dtype == observed.DTYPE and len(y_train) + len(y_test) == 10
    for train, _ in KFold(3).split(x):
        assert observed.is_observed(label[train])


# ---------------------------------------------------------------- interval probabilities


def _reference(dist_cdf, dist_sf, lower, upper):
    """log P(lower <= Y <= upper) in float64, subtracting on the side away from 1."""
    out = []
    for a, b in zip(lower, upper):
        below_b = 1.0 if b == INF else dist_cdf(b)
        below_a = 0.0 if a == -INF else dist_cdf(a, left=True)
        if below_b < 0.5:
            out.append(np.log(below_b - below_a))
        else:
            above_a = 1.0 if a == -INF else dist_sf(a, left=True)
            above_b = 0.0 if b == INF else dist_sf(b)
            out.append(np.log(above_a - above_b))
    return np.asarray(out)


LOWER = np.array([-INF, -9.0, -3.0, 0.5, 4.0, 7.5, -1.0, -2.0])
UPPER = np.array([-6.0, -8.0, INF, 0.7, 4.5, INF, 1.0, INF])


def test_normal_and_student_t_intervals_hold_in_the_far_tails():
    normal = Normal(jnp.zeros(8), jnp.ones(8))
    expected = _reference(lambda v, left=False: stats.norm.cdf(v), lambda v, left=False: stats.norm.sf(v), LOWER, UPPER)
    assert np.allclose(normal.log_interval(LOWER, UPPER), expected, rtol=2e-4, atol=1e-4)
    t = StudentT(jnp.full(8, 0.5), jnp.full(8, 2.0), jnp.full(8, 3.0))
    ref = stats.t(3.0, loc=0.5, scale=2.0)
    expected = _reference(lambda v, left=False: ref.cdf(v), lambda v, left=False: ref.sf(v), LOWER, UPPER)
    assert np.allclose(t.log_interval(LOWER, UPPER), expected, rtol=2e-4, atol=1e-4)
    # Beyond float32's reach of the CDF itself: 40 standard deviations up still has a finite log probability.
    far = float(normal[:1].log_interval(jnp.array([40.0]), jnp.array([INF]))[0])
    assert np.isclose(far, stats.norm.logsf(40.0), rtol=1e-4)


def test_count_intervals_with_a_structural_zero_and_a_far_tail():
    lower = np.array([0.0, 3.0, 40.0, 0.0, 7.0, 120.0])
    upper = np.array([0.0, INF, INF, 2.0, 7.0, 130.0])
    mean, zero = 5.0, 0.2

    def cdf(v, left=False):
        k = np.ceil(v) - 1 if left else np.floor(v)
        return 0.0 if k < 0 else zero + (1 - zero) * stats.poisson.cdf(k, mean)

    def sf(v, left=False):
        k = np.ceil(v) - 1 if left else np.floor(v)
        return 1.0 if k < 0 else (1 - zero) * stats.poisson.sf(k, mean)

    got = np.asarray(Poisson(jnp.full(6, mean), jnp.full(6, zero)).log_interval(lower, upper))
    expected = _reference(cdf, sf, lower[:5], upper[:5])
    assert np.allclose(got[:5], expected, rtol=1e-4, atol=1e-4)
    # The CDF underflows float32 here; the log mass of the nearest count keeps it finite and close.
    exact = np.log(np.sum((1 - zero) * stats.poisson.pmf(np.arange(120, 131), mean)))
    assert np.isfinite(got[5]) and abs(got[5] - exact) < 0.1
    size, dispersion = 2.0, 0.5
    nb = stats.nbinom(size, size / (size + 4.0))
    got = NegativeBinomial(jnp.full(5, 4.0), jnp.full(5, dispersion)).log_interval(lower[:5], upper[:5])
    expected = _reference(
        lambda v, left=False: 0.0 if (np.ceil(v) - 1 if left else np.floor(v)) < 0 else nb.cdf(np.ceil(v) - 1 if left else np.floor(v)),
        lambda v, left=False: 1.0 if (np.ceil(v) - 1 if left else np.floor(v)) < 0 else nb.sf(np.ceil(v) - 1 if left else np.floor(v)),
        lower[:5],
        upper[:5],
    )
    assert np.allclose(got, expected, rtol=1e-4, atol=1e-4)


def test_shape_parameters_now_have_gradients():
    """JAX's betainc has no gradient in its shape arguments; the Student-t's df and the negative binomial's dispersion need one."""
    lower, upper = jnp.array([1.5, -INF, 0.2]), jnp.array([INF, -2.0, 0.9])

    def t_interval(df):
        return jnp.sum(StudentT(jnp.zeros(3), jnp.ones(3), df).log_interval(lower, upper))

    def reference(df):
        out = 0.0
        for a, b in zip(np.asarray(lower), np.asarray(upper)):
            out += np.log(stats.t.cdf(b, df) - stats.t.cdf(a, df))
        return out

    grad = jax.grad(t_interval)(jnp.full(3, 4.0))
    numeric = (reference(4.0 + 1e-4) - reference(4.0 - 1e-4)) / 2e-4
    assert np.isclose(float(jnp.sum(grad)), numeric, rtol=2e-2)
    # The bisection quantile's implicit gradient goes through the CDF, so it reaches df too.
    slope = jax.grad(lambda df: StudentT(0.0, 1.0, df).quantile(0.95))(4.0)
    assert np.isclose(float(slope), (stats.t.ppf(0.95, 4.0 + 1e-3) - stats.t.ppf(0.95, 4.0 - 1e-3)) / 2e-3, rtol=2e-2)

    def nb_tail(dispersion):
        return NegativeBinomial(4.0, dispersion).log_interval(6.0, INF)

    def nb_reference(dispersion):
        size = 1.0 / dispersion
        return stats.nbinom.logsf(5, size, size / (size + 4.0))

    numeric = (nb_reference(0.5 + 1e-5) - nb_reference(0.5 - 1e-5)) / 2e-5
    assert np.isclose(float(jax.grad(nb_tail)(0.5)), numeric, rtol=2e-2)


def test_positive_and_mixed_families():
    lower, upper = np.array([0.0, 1.0, 30.0]), np.array([1.0, 2.0, INF])
    gamma = stats.gamma(3.0, scale=2.0 / 3.0)
    got = Gamma(jnp.full(3, 2.0), jnp.full(3, 3.0)).log_interval(lower, upper)
    assert np.allclose(got, [np.log(gamma.cdf(1.0)), np.log(gamma.cdf(2.0) - gamma.cdf(1.0)), gamma.logsf(30.0)], rtol=1e-4)
    log_normal = stats.lognorm(1.0)
    got = LogNormal(jnp.zeros(3), jnp.ones(3)).log_interval(lower, upper)
    assert np.allclose(got, [np.log(log_normal.cdf(1.0)), np.log(log_normal.cdf(2.0) - log_normal.cdf(1.0)), log_normal.logsf(30.0)], rtol=1e-4)
    # The hurdle's zero is an atom: [0, 0] is its probability, and [0, b] includes it.
    hurdle = Hurdle(jnp.full(4, 0.3), LogNormal(jnp.zeros(4), jnp.ones(4)))
    got = hurdle.log_interval(np.array([0.0, 0.0, 1.0, -INF]), np.array([0.0, 2.0, INF, 0.5]))
    expected = [np.log(0.3), np.log(0.3 + 0.7 * log_normal.cdf(2.0)), np.log(0.7 * log_normal.sf(1.0)), np.log(0.3 + 0.7 * log_normal.cdf(0.5))]
    assert np.allclose(got, expected, rtol=1e-5)
    # A mixture in the units of y: each component's interval, mixed, through the affine map.
    mix = AffineTransformed(Mixture(Normal(jnp.array([[-2.0, 3.0]] * 3), jnp.array([[0.5, 1.0]] * 3))), 10.0, 2.0)

    def cdf(v):
        z = (v - 10.0) / 2.0
        return 0.5 * stats.norm.cdf(z, -2, 0.5) + 0.5 * stats.norm.cdf(z, 3, 1.0)

    got = mix.log_interval(np.array([-INF, 10.0, 20.0]), np.array([0.0, 12.0, INF]))
    assert np.allclose(got, [np.log(cdf(0.0)), np.log(cdf(12.0) - cdf(10.0)), np.log(1 - cdf(20.0))], rtol=1e-5)
    loc_grad = jax.grad(lambda loc: jnp.sum(AffineTransformed(mix.base, loc, 2.0).log_interval(jnp.array([-INF, 20.0, 10.0]), jnp.array([0.0, INF, 12.0]))))(10.0)
    assert np.isfinite(float(loc_grad))


def test_flow_and_histogram_intervals():
    raw = jax.random.normal(jax.random.PRNGKey(0), (4, flow_output_count())) * 0.5
    flow = Flow(raw)
    lower, upper = jnp.array([-INF, -1.0, 0.5, -0.3]), jnp.array([0.0, 1.0, INF, 0.4])
    cdf = lambda v: np.asarray(flow.cdf(jnp.asarray(v, dtype=jnp.float32)))
    expected = np.log(np.array([cdf([0.0] * 4)[0], cdf([1.0] * 4)[1] - cdf([-1.0] * 4)[1], 1 - cdf([0.5] * 4)[2], cdf([0.4] * 4)[3] - cdf([-0.3] * 4)[3]]))
    assert np.allclose(flow.log_interval(lower, upper), expected, rtol=1e-4, atol=1e-5)
    grads = jax.grad(lambda r: jnp.sum(Flow(r).log_interval(jnp.array([20.0] * 4), jnp.array([INF] * 4))))(raw)
    assert np.isfinite(np.asarray(grads)).all()
    histogram = Histogram(jax.random.normal(jax.random.PRNGKey(1), (3, 16)))
    probabilities = np.asarray(jax.nn.softmax(histogram.logits))
    got = np.asarray(histogram.log_interval(jnp.array([-INF, 3.9, 10.0]), jnp.array([0.0, INF, INF])))
    assert np.isclose(got[0], np.log(probabilities[0, :8].sum()), rtol=1e-5)
    # Past an end bin's center a bound reads the outer half of that bin, as log_prob reads a value out there.
    assert np.allclose(got[1:], np.log(probabilities[1:, -1] / 2), rtol=1e-5)


def test_noise_widens_a_normal_and_each_mixture_component():
    convolved = Mixture(Normal(jnp.array([[0.0, 2.0]]), jnp.array([[1.0, 0.5]]))).convolve_normal(jnp.array([0.5]))
    assert np.allclose(convolved.components.scale, [[np.sqrt(1.25), np.sqrt(0.5)]])
    with pytest.raises(NotImplementedError):
        StudentT(0.0, 1.0, 3.0).convolve_normal(1.0)


# ---------------------------------------------------------------- likelihood and PIT


def test_log_likelihood_reads_each_kind_of_label():
    dist = Normal(jnp.full(4, 1.0), jnp.full(4, 2.0))
    label = observed.concatenate([observed.exact([0.5]), observed.interval([0.0], [3.0]), observed.censored([4.0], [True]), observed.exact([2.0])])
    label = observed.truncated(label, lower=[-INF, -INF, -INF, 0.0])
    label = observed.noisy(label, [0.0, 0.0, 0.0, 1.5])
    part = observed.fields(label)
    got = observed.log_likelihood(dist, part[:, 0], part[:, 1], part[:, 2], part[:, 3], part[:, 4])
    wide = stats.norm(1.0, 2.5)
    expected = [
        stats.norm.logpdf(0.5, 1.0, 2.0),
        np.log(stats.norm.cdf(3.0, 1.0, 2.0) - stats.norm.cdf(0.0, 1.0, 2.0)),
        stats.norm.logsf(4.0, 1.0, 2.0),
        wide.logpdf(2.0) - wide.logsf(0.0),
    ]
    assert np.allclose(got, expected, rtol=1e-5)


def test_randomized_pit_of_censored_draws_is_uniform():
    key_draw, key_cap, key_pit = jax.random.split(jax.random.PRNGKey(3), 3)
    n = 20000
    dist = Normal(jnp.zeros(n), jnp.ones(n))
    value = np.asarray(dist.sample(key_draw))
    cap = np.asarray(jax.random.normal(key_cap, (n,))) + 0.5
    label = observed.censored(np.minimum(value, cap), value >= cap)
    pit = np.asarray(observed.randomized_pit(dist, label["lower"], label["upper"], key_pit))
    assert np.mean(value >= cap) > 0.3
    assert stats.kstest(pit, "uniform").pvalue > 1e-3
    # Treating the caps as values instead is far from uniform.
    naive = np.asarray(dist.cdf(jnp.asarray(np.minimum(value, cap))))
    assert stats.kstest(naive, "uniform").pvalue < 1e-10


# ---------------------------------------------------------------- estimator


def _sold_out(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    mean = np.exp(1.6 + 0.5 * x[:, 0])
    demand = rng.negative_binomial(4, 4 / (4 + mean))
    stock = rng.poisson(mean * 1.05)
    sold_out = demand >= stock
    return x, demand, np.minimum(demand, stock), sold_out


def test_a_censored_fit_recovers_demand_that_the_usual_fit_misses():
    x, demand, sales, sold_out = _sold_out()
    train, test = slice(0, 2400), slice(2400, None)
    assert 0.3 < sold_out.mean() < 0.6
    label = observed.censored(sales, sold_out)
    censored = ASRRegressor(task="negative_binomial", max_epochs=10, random_state=0).fit(x[train], label[train])
    naive = ASRRegressor(task="negative_binomial", max_epochs=10, random_state=0).fit(x[train], sales[train])
    covered = lambda model: np.mean(demand[test] <= model.predict_quantiles(x[test], [0.9])[:, 0])
    assert abs(covered(censored) - 0.9) < 0.04
    assert covered(naive) < 0.82
    assert np.mean(censored.log_prob(x[test], demand[test])) > np.mean(naive.log_prob(x[test], demand[test])) + 0.05
    # log_prob of the label itself is the log mass of exact rows and the survival of sold-out ones.
    values = censored.log_prob(x[test], label[test])
    dist = censored.predict_dist(x[test])
    k = sales[test].astype(np.float32)
    manual = np.where(sold_out[test], np.asarray(dist.log_survival_left(k)), np.asarray(dist.log_prob(k)))
    assert np.allclose(values, manual, atol=1e-4)
    assert np.isclose(censored.score(x[test], label[test]), values.mean())
    # Calibrating on censored rows keeps an interval on the true demand near its promise.
    censored.calibrate(x[test][:300], label[test][:300])
    low, high = censored.predict_interval(x[test][300:], coverage=0.8).T
    inside = (demand[test][300:] >= low) & (demand[test][300:] <= high)
    assert 0.74 < inside.mean() < 0.9


def test_an_exact_label_trains_like_plain_values():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 2)).astype(np.float32)
    y = x[:, 0] * 2.0 + rng.normal(size=600)
    plain = ASRRegressor(task="gaussian", max_epochs=4, random_state=0).fit(x, y)
    exact = ASRRegressor(task="gaussian", max_epochs=4, random_state=0).fit(x, observed.exact(y))
    assert np.allclose(plain.predict(x[:50]), exact.predict(x[:50]), atol=1e-4)
    assert np.allclose(plain.log_prob(x[:50], y[:50]), exact.log_prob(x[:50], observed.exact(y[:50])), atol=1e-4)


def test_truncated_and_noisy_labels_recover_the_value_behind_them():
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    # Zero-truncated counts: only positive days were recorded.
    rate = np.exp(0.2 + 0.3 * x[:, 0])
    counts = rng.poisson(rate)
    kept = counts >= 1
    label = observed.truncated(counts[kept], lower=1.0)
    model = ASRRegressor(task="poisson", max_epochs=12, random_state=0).fit(x[kept], label)
    naive = ASRRegressor(task="poisson", max_epochs=12, random_state=0).fit(x[kept], counts[kept])
    assert abs(np.mean(model.predict(x) / rate) - 1.0) < 0.08
    assert np.mean(naive.predict(x) / rate) > 1.25
    # Known measurement noise: the head learns the spread of the value, not of the reading.
    value = 3.0 * x[:, 0] + rng.normal(size=n) * 0.5
    reading = value + rng.normal(size=n) * 1.5
    model = ASRRegressor(task="gaussian", max_epochs=15, random_state=0).fit(x, observed.noisy(reading, 1.5))
    naive = ASRRegressor(task="gaussian", max_epochs=15, random_state=0).fit(x, reading)
    spread = lambda fitted: float(np.median(np.asarray(fitted.predict_dist(x[:500]).stddev())))
    assert abs(spread(model) - 0.5) < 0.2
    assert abs(spread(naive) - np.hypot(0.5, 1.5)) < 0.2


def test_survival_on_the_log_scale_with_any_continuous_head():
    rng = np.random.default_rng(4)
    n = 3000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    log_time = 1.0 + 0.8 * x[:, 0] + rng.normal(size=n) * 0.6
    log_cutoff = rng.uniform(0.0, 2.5, size=n)
    event = log_time <= log_cutoff
    label = observed.survival(np.minimum(log_time, log_cutoff), event)
    model = ASRRegressor(task="mixture", max_epochs=12, random_state=0).fit(x, label)
    median = model.predict_quantiles(x[:400], [0.5])[:, 0]
    assert np.mean(np.abs(median - (1.0 + 0.8 * x[:400, 0]))) < 0.2


def test_observed_models_save_and_load(tmp_path):
    x, _demand, sales, sold_out = _sold_out(600, seed=5)
    model = ASRRegressor(task="mixture", max_epochs=3, random_state=0).fit(x, observed.censored(sales, sold_out))
    restored = ASRRegressor.load(model.save(tmp_path / "censored"))
    assert np.allclose(restored.predict(x[:20]), model.predict(x[:20]))
    label = observed.censored(sales[:20], sold_out[:20])
    assert np.allclose(restored.log_prob(x[:20], label), model.log_prob(x[:20], label), atol=1e-5)


def test_tasks_and_labels_that_cannot_be_scored_are_refused():
    x, _demand, sales, sold_out = _sold_out(200, seed=6)
    label = observed.censored(sales, sold_out)
    for task in ("regression", "quantile", "classification"):
        with pytest.raises(AshlarError):
            ASRRegressor(task=task, max_epochs=1).fit(x, label)
    with pytest.raises(AshlarError):
        ASRRegressor(task="flow", max_epochs=1).fit(x, observed.noisy(sales, 1.0))
    with pytest.raises(AshlarError):
        ASRRegressor(task="poisson", max_epochs=1).fit(x, observed.interval(sales - 0.5, sales + 0.5))
    model = ASRRegressor(task="poisson", max_epochs=1).fit(x, label)
    with pytest.raises(AshlarError):
        model.calibrate(x, observed.truncated(sales, lower=0.0))
    with pytest.raises(AshlarError):
        model.log_prob(x, observed.total(np.full(200, 3.0), group=np.arange(200) // 2))


# ---------------------------------------------------------------- group totals


def test_packing_keeps_each_observation_whole_in_one_batch():
    from ashlar.train import pack_units

    rng = np.random.default_rng(7)
    size = rng.integers(1, 8, size=300)
    first = np.concatenate([[0], np.cumsum(size)[:-1]])
    index, keep = pack_units(first, size, rng.permutation(300), 24)
    assert sorted(index[keep].tolist()) == list(range(int(size.sum())))
    for batch in range(index.shape[0]):
        for start, count in zip(first, size):
            rows = np.arange(start, start + count)
            slots = np.flatnonzero(np.isin(index[batch], rows) & keep[batch])
            # Whole or absent, and in consecutive slots in their own order.
            assert len(slots) in (0, count)
            if len(slots):
                assert np.array_equal(index[batch][slots], rows) and np.all(np.diff(slots) == 1)
    with pytest.raises(ValueError):
        pack_units(first, size, np.arange(300), 5)


def test_group_labels_record_their_size_and_catch_a_cut_group():
    label = observed.total([9.0, 9.0, 9.0, 4.0, 5.0, 5.0], group=[3, 3, 3, -1, 8, 8])
    assert np.array_equal(label["group_size"], [3, 3, 3, 1, 2, 2])
    assert np.array_equal(observed.group_sizes(label[:2]), [3, 3])
    observed.check_whole(label)
    with pytest.raises(AshlarError, match="GroupKFold"):
        observed.check_whole(label[:2])
    # Stacking two labels keeps their groups apart, and sharing an id by hand is caught.
    stacked = observed.concatenate([label, label])
    assert len(np.unique(stacked["group"][stacked["group"] >= 0])) == 4
    with pytest.raises(AshlarError):
        observed.check(np.concatenate([label, label]))
    # Rows of a group are brought together, keeping first-appearance order.
    scattered = observed.total([5.0, 1.0, 5.0, 7.0, 7.0, 2.0], group=[0, -1, 0, 1, 1, -1])
    order = observed.contiguous_order(scattered[[0, 1, 3, 2, 4, 5]])
    assert np.array_equal(order, [0, 3, 1, 2, 4, 5])
    first, size = observed.units(scattered[[0, 2, 1, 3, 4, 5]])
    assert np.array_equal(first, [0, 2, 3, 5]) and np.array_equal(size, [2, 1, 2, 1])
    with pytest.raises(AshlarError):
        observed.units(scattered)


def test_the_total_objective_scores_the_distribution_of_each_sum():
    from ashlar.objectives import make_observed_objective, make_total_objective

    rng = np.random.default_rng(8)
    rows, members = 7, 3
    raw = jnp.asarray(rng.normal(size=(rows, members, 2)), dtype=jnp.float32)
    # Rows 0-2 are one total, row 3 is its own, rows 4-5 a censored total, row 6 padding with no weight.
    label = observed.concatenate(
        [
            observed.total(np.full(3, 1.5), np.zeros(3)),
            observed.exact([0.25]),
            observed.grouped(observed.censored(np.full(2, 0.5), np.ones(2, dtype=bool)), np.zeros(2)),
            observed.exact([9.0]),
        ]
    )
    start = np.array([1, 0, 0, 1, 1, 0, 1], dtype=np.float32)
    y = np.concatenate([observed.fields(label), start[:, None]], axis=1)
    weight = np.array([2.0, 0, 0, 1.0, 1.0, 0, 0], dtype=np.float32)
    from ashlar.distributions import head_distribution

    dist = head_distribution("gaussian", raw)
    loc, scale = np.asarray(dist.loc, dtype=np.float64), np.asarray(dist.scale, dtype=np.float64)
    first_sum = stats.norm.logpdf(1.5, loc[:3].sum(0), np.sqrt((scale[:3] ** 2).sum(0)))
    own = stats.norm.logpdf(0.25, loc[3], scale[3])
    censored_sum = stats.norm.logsf(0.5, loc[4:6].sum(0), np.sqrt((scale[4:6] ** 2).sum(0)))
    expected = -(2.0 * first_sum.mean() + own.mean() + censored_sum.mean()) / 4.0
    got = float(make_total_objective("gaussian")(raw, jnp.asarray(y), jnp.asarray(weight)))
    assert np.isclose(got, expected, rtol=1e-5)
    grads = jax.grad(lambda r: make_total_objective("gaussian")(r, jnp.asarray(y), jnp.asarray(weight)))(raw)
    assert np.isfinite(np.asarray(grads)).all() and np.allclose(np.asarray(grads)[6], 0.0)
    # Rows of their own score what the row objective scores.
    plain = observed.exact(rng.normal(size=rows))
    y_plain = np.concatenate([observed.fields(plain), np.ones((rows, 1))], axis=1)
    ones = jnp.ones(rows)
    total = make_total_objective("gaussian")(raw, jnp.asarray(y_plain), ones)
    row = make_observed_objective("gaussian")(raw, jnp.asarray(observed.fields(plain)), ones)
    assert np.isclose(float(total), float(row), rtol=1e-6)


def _weekly(n_weeks=1000, days=7, seed=9):
    rng = np.random.default_rng(seed)
    n = n_weeks * days
    x = np.column_stack([rng.normal(size=n), rng.uniform(-1.0, 1.0, size=n)]).astype(np.float32)
    mean = 10.0 + 3.0 * x[:, 0] + 2.0 * np.sin(3.0 * x[:, 1])
    spread = 0.5 + 1.0 * (x[:, 1] > 0)
    y = mean + spread * rng.normal(size=n)
    week = np.repeat(np.arange(n_weeks), days)
    return x, y, week, mean, spread


def test_daily_distributions_are_learned_from_weekly_totals():
    x, y, week, mean, spread = _weekly()
    weekly = np.bincount(week, weights=y)[week]
    model = ASRRegressor(task="gaussian", max_epochs=30, random_state=0).fit(x, observed.total(weekly, week))
    naive = ASRRegressor(task="gaussian", max_epochs=30, random_state=0).fit(x, weekly / 7.0)
    test = slice(0, 1400)
    fitted = model.predict_dist(x[test])
    assert np.sqrt(np.mean((np.asarray(fitted.mean()) - mean[test]) ** 2)) < 0.3
    sd = np.asarray(fitted.stddev())
    high = x[test, 1] > 0
    assert sd[high].mean() - sd[~high].mean() > 0.3
    assert abs(np.mean(sd**2) - np.mean(spread[test] ** 2)) < 0.25 * np.mean(spread[test] ** 2)
    # Spreading each total evenly over its days teaches each day the week's average instead.
    assert np.sqrt(np.mean((np.asarray(naive.predict_dist(x[test]).mean()) - mean[test]) ** 2)) > 2.0
    assert np.mean(model.log_prob(x[test], y[test])) > np.mean(naive.log_prob(x[test], y[test])) + 2.0
    # Poisson rows sum to a Poisson total.
    rng = np.random.default_rng(10)
    rate = np.exp(0.5 + 0.8 * np.clip(x[:, 0], -2.0, 2.0))
    counts = rng.poisson(rate)
    weekly_counts = np.bincount(week, weights=counts)[week]
    poisson = ASRRegressor(task="poisson", max_epochs=30, random_state=0).fit(x, observed.total(weekly_counts, week))
    assert np.mean(np.abs(np.log(poisson.predict(x[test]) / rate[test]))) < 0.15


def test_totals_mix_with_exact_rows_split_whole_and_score_per_group():
    x, y, week, _mean, _spread = _weekly(n_weeks=300, seed=11)
    weekly = np.bincount(week, weights=y)[week]
    totals = week % 2 == 0
    label = observed.grouped(observed.exact(np.where(totals, weekly, y)), np.where(totals, week, -1))
    rng = np.random.default_rng(12)
    shuffled = rng.permutation(len(label))
    model = ASRRegressor(task="normal", max_epochs=4, split="temporal", random_state=0).fit(x[shuffled], label[shuffled])
    assert np.isfinite(model.history_["val_loss"]).all()
    # score is the mean log-likelihood per observation: each member sums its rows, then members mix.
    part = slice(0, 70)
    from ashlar.predict import positive_scale

    raw = np.asarray(model.predict_raw(x[part]), dtype=np.float64)
    loc = raw[..., 0, 0]
    scale = np.asarray(positive_scale(jnp.asarray(raw[..., 0, 1])), dtype=np.float64)
    values = []
    for start in range(0, 70, 7):
        rows = slice(start, start + 7)
        if totals[start]:
            member = stats.norm.logpdf(weekly[start], loc[rows].sum(0), np.sqrt((scale[rows] ** 2).sum(0)))
            values.append(np.logaddexp.reduce(member) - np.log(member.size))
        else:
            for row in range(start, start + 7):
                member = stats.norm.logpdf(y[row], loc[row], scale[row])
                values.append(np.logaddexp.reduce(member) - np.log(member.size))
    assert np.isclose(model.score(x[part], label[part]), np.mean(values), rtol=1e-4)
    with pytest.raises(AshlarError, match="GroupKFold"):
        model.score(x[:3], label[:3])
    with pytest.raises(AshlarError, match="GroupKFold"):
        ASRRegressor(task="normal", max_epochs=1).fit(x[:3], label[:3])


def test_totals_that_cannot_be_trained_are_refused():
    x, y, week, _mean, _spread = _weekly(n_weeks=40, seed=13)
    label = observed.total(np.bincount(week, weights=y)[week], week)
    for kwargs in (
        {"task": "mixture"},
        {"task": "negative_binomial"},
        {"task": "poisson", "head_kwargs": {"zero_inflated": True}},
        {"task": "gaussian", "learning_rate_schedule": "surface"},
    ):
        with pytest.raises(AshlarError):
            ASRRegressor(max_epochs=1, **kwargs).fit(x, observed.total(np.round(np.abs(label["lower"])), week))
    with pytest.raises(AshlarError, match="split group"):
        ASRRegressor(task="gaussian", max_epochs=1, split="group").fit(x, label, groups=np.arange(len(label)) % 3)
    model = ASRRegressor(task="gaussian", max_epochs=1, random_state=0).fit(x, label)
    with pytest.raises(AshlarError):
        model.log_prob(x, label)
