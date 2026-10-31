"""Joint labels: a cell per target column, exact or an interval, scored by the joint heads' box likelihood."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

from ashlar import ASRRegressor, observed
from ashlar.distributions import (
    AffineTransformed,
    AutoregressiveFlow,
    Independent,
    LowRankMultivariateNormal,
    Mixture,
    Normal,
    autoregressive_context_width,
    box_points,
    joint_mixture_distribution,
    joint_normal_distribution,
)
from ashlar.errors import AshlarError, ShapeError
from ashlar.objectives import make_joint_observed_objective

INF = np.inf


# ---------------------------------------------------------------- labels


def test_joint_labels_are_built_cell_by_cell():
    values = np.arange(12.0).reshape(4, 3)
    flags = np.zeros((4, 3), dtype=bool)
    flags[1, 2] = True
    label = observed.censored(values, flags)
    assert label.shape == (4, 3) and label.dtype == observed.DTYPE
    assert label["upper"][1, 2] == INF and label["upper"][1, 1] == values[1, 1]
    assert observed.fields(label).shape == (4, 3, 6)
    assert np.array_equal(observed.rounded(values, [1.0, 2.0, 0.5])["lower"][0], [-0.5, 0.0, 1.75])
    assert observed.interval(values - 1.0, values + 1.0).shape == (4, 3)
    stacked = observed.stack([observed.censored(values[:, 0], flags[:, 0]), values[:, 1], observed.interval(values[:, 2], INF)])
    assert stacked.shape == (4, 3) and np.all(stacked["upper"][:, 2] == INF)
    assert observed.concatenate([label, stacked]).shape == (8, 3)
    assert np.allclose(observed.affine(label, [1.0, 2.0, 3.0], 2.0)["lower"][0], [-0.5, -0.5, -0.5])
    assert observed.representative(label).shape == (4, 3)
    assert observed.as_observed(values).shape == (4, 3)


@pytest.mark.parametrize(
    "build",
    [
        lambda label: observed.truncated(label, lower=0.0),
        lambda label: observed.noisy(label, 1.0),
        lambda label: observed.grouped(label, np.zeros(4)),
        lambda label: observed.stack([label]),
        lambda label: observed.concatenate([label, np.zeros(4)]),
        lambda label: observed.stack([np.zeros(4), np.zeros(3)]),
        lambda label: observed.censored(np.zeros((4, 3)), np.zeros(4, dtype=bool)),
    ],
)
def test_what_a_joint_label_cannot_say_is_refused(build):
    with pytest.raises((AshlarError, ShapeError)):
        build(observed.exact(np.zeros((4, 3))))


def test_scikit_learn_splits_a_joint_label():
    from sklearn.model_selection import KFold, train_test_split

    label = observed.censored(np.arange(20.0).reshape(10, 2), np.arange(20).reshape(10, 2) % 3 == 0)
    train, test = train_test_split(label, test_size=0.3, random_state=0)
    assert train.shape == (7, 2) and test.shape == (3, 2)
    for fold, held in KFold(3).split(np.zeros(10)):
        assert label[fold].shape == (len(fold), 2) and label[held].shape == (len(held), 2)


# ---------------------------------------------------------------- the box likelihood


def test_box_points_are_centred_inside_the_cube():
    points = np.asarray(box_points(256, 6))
    assert points.shape == (256, 6) and (points > 0).all() and (points < 1).all()
    assert np.allclose(np.sort(points[:, 0]), (np.arange(256) + 0.5) / 256)
    for count in (1, 7, 24, 100):
        grid = np.asarray(box_points(count, 8))
        assert (grid > 0).all() and (grid < 1).all()


def _normal():
    rng = np.random.default_rng(0)
    loc, factor, diag = rng.normal(size=3), 0.8 * rng.normal(size=(3, 2)), rng.uniform(0.2, 1.0, size=3)
    return loc, factor @ factor.T + np.diag(diag), LowRankMultivariateNormal(loc, factor, diag)


def _reference(loc, cov, lower, upper):
    """``p(exact cells) * P(coarse cells in their box | exact)`` from the conditional normal, in float64."""
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    exact = lower == upper
    known, unknown = np.flatnonzero(exact), np.flatnonzero(~exact)
    log_value, mean, cov_c = 0.0, loc[unknown], cov[np.ix_(unknown, unknown)]
    if len(known):
        log_value = stats.multivariate_normal(loc[known], cov[np.ix_(known, known)]).logpdf(lower[known])
        gain = cov[np.ix_(unknown, known)] @ np.linalg.inv(cov[np.ix_(known, known)])
        mean = loc[unknown] + gain @ (lower[known] - loc[known])
        cov_c = cov[np.ix_(unknown, unknown)] - gain @ cov[np.ix_(known, unknown)]
    if len(unknown) == 1:
        sd = np.sqrt(cov_c[0, 0])
        return log_value + np.log(stats.norm.cdf(upper[unknown][0], mean[0], sd) - stats.norm.cdf(lower[unknown][0], mean[0], sd))
    if len(unknown) > 1:
        return log_value + np.log(stats.multivariate_normal(mean, cov_c).cdf(upper[unknown], lower_limit=lower[unknown]))
    return log_value


CLOSED = [
    ([0.2, 0.2, -INF], [0.2, 0.2, 0.5]),
    ([0.1, -1.0, 0.4], [0.1, 1.0, 0.4]),
    ([0.3, -0.2, 1.1], [0.3, -0.2, 1.1]),
    ([-INF, 0.4, -INF], [INF, 0.4, 1.5]),
    ([-INF, -INF, -INF], [INF, INF, INF]),
]
SAMPLED = [
    ([-0.5, -INF, 0.3], [0.5, 1.0, INF]),
    ([0.0, 1.0, -INF], [0.0, INF, 0.0]),
    ([2.5, 2.0, -INF], [INF, INF, INF]),
]


@pytest.mark.slow
def test_joint_normal_box_is_exact_for_one_coarse_column_and_close_for_several():
    loc, cov, dist = _normal()
    for lower, upper in CLOSED:
        expected = _reference(loc, cov, lower, upper)
        for count in (1, 256):
            assert float(dist.log_box(jnp.asarray(lower), jnp.asarray(upper), box_points(count, 3))) == pytest.approx(expected, abs=2e-5)
    assert float(dist.log_box(jnp.asarray(CLOSED[2][0]), jnp.asarray(CLOSED[2][1]))) == pytest.approx(
        float(dist.log_prob(jnp.asarray(CLOSED[2][0]))), abs=1e-5
    )
    for lower, upper in SAMPLED:
        expected = _reference(loc, cov, lower, upper)
        assert float(dist.log_box(jnp.asarray(lower), jnp.asarray(upper))) == pytest.approx(expected, abs=2e-3)
        assert float(dist.log_box(jnp.asarray(lower), jnp.asarray(upper), box_points(32, 3))) == pytest.approx(expected, abs=1e-2)


@pytest.mark.slow
def test_box_gradients_stay_finite_on_open_sides_and_far_tails():
    loc, _cov, _dist = _normal()
    rng = np.random.default_rng(1)
    factor, diag = jnp.asarray(0.8 * rng.normal(size=(3, 2))), jnp.asarray(rng.uniform(0.2, 1.0, size=3))

    def score(loc, factor, diag, lower, upper):
        return LowRankMultivariateNormal(loc, factor, diag).log_box(lower, upper, box_points(32, 3))

    for lower, upper in [*CLOSED, *SAMPLED, ([40.0, -INF, 0.0], [INF, -35.0, 0.0])]:
        value = score(jnp.asarray(loc), factor, diag, jnp.asarray(lower), jnp.asarray(upper))
        grads = jax.grad(score, argnums=(0, 1, 2))(jnp.asarray(loc), factor, diag, jnp.asarray(lower), jnp.asarray(upper))
        assert np.isfinite(float(value)) and all(bool(jnp.all(jnp.isfinite(part))) for part in grads)


@pytest.mark.slow
def test_mixtures_affine_maps_and_independent_columns_compose():
    rng = np.random.default_rng(2)
    components = LowRankMultivariateNormal(rng.normal(size=(4, 3)), 0.5 * rng.normal(size=(4, 3, 2)), rng.uniform(0.3, 1.0, size=(4, 3)))
    logits = jnp.asarray(rng.normal(size=4))
    mixture = Mixture(components, logits)
    lower, upper = jnp.asarray([0.1, -INF, -0.5]), jnp.asarray([0.1, 0.3, INF])
    each = components.log_box(lower, upper)
    assert float(mixture.log_box(lower, upper)) == pytest.approx(float(jax.nn.logsumexp(each + jax.nn.log_softmax(logits))), abs=1e-5)
    loc, scale = jnp.asarray([10.0, -2.0, 3.0]), jnp.asarray([2.0, 0.5, 4.0])
    moved = AffineTransformed(components, loc, scale)
    in_y = moved.log_box(loc + scale * lower, loc + scale * upper)
    assert np.allclose(in_y, each - jnp.log(scale[0]), atol=1e-5)
    marginals = Normal(jnp.asarray([0.0, 1.0, 2.0]), jnp.asarray([1.0, 2.0, 0.5]))
    expected = marginals.log_prob(lower)[0] + marginals.log_interval(-INF, 0.3)[1] + marginals.log_interval(-0.5, INF)[2]
    assert float(Independent(marginals).log_box(lower, upper)) == pytest.approx(float(expected), abs=1e-5)


def test_the_joint_mixture_layout_and_density():
    rng = np.random.default_rng(3)
    components, d, rank = 3, 2, 1
    flat = jnp.asarray(rng.normal(size=(5, components * (1 + d * (2 + rank)))))
    dist = joint_mixture_distribution(flat, d, components)
    assert dist.batch_shape == (5,) and dist.event_shape == (2,)
    parts = joint_normal_distribution(flat[:, components:].reshape(5, components, -1), d)
    value = jnp.asarray(rng.normal(size=(5, 2)))
    expected = jax.nn.logsumexp(parts.log_prob(value[:, None, :]) + jax.nn.log_softmax(flat[:, :components]), axis=-1)
    assert np.allclose(dist.log_prob(value), expected, atol=1e-5)
    with pytest.raises(ShapeError):
        joint_mixture_distribution(flat[:, :-1], d, components)


def _flow(seed=3, depth=1):
    d, hidden = 2, 16
    width = autoregressive_context_width(d, hidden)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    context = 0.6 * jax.random.normal(keys[0], (1, 1, depth * width))
    w_in = jax.random.normal(keys[1], (1, depth, hidden, d))
    w_out = 0.4 * jax.random.normal(keys[2], (1, depth, width - hidden, hidden))
    return AutoregressiveFlow(context, w_in, w_out, d)


def _trapezoid(values, grid):
    return jnp.sum(0.5 * (values[1:] + values[:-1]) * (grid[1:] - grid[:-1]))


@pytest.mark.slow
def test_flow_box_matches_quadrature_of_its_density():
    flow = _flow()

    def density(first, second):
        grid = jnp.stack(jnp.broadcast_arrays(jnp.asarray(first, jnp.float32), jnp.asarray(second, jnp.float32)), axis=-1)
        return jnp.exp(flow.log_prob(grid.reshape(-1, 1, 2))).reshape(grid.shape[:-1])

    def box(lower, upper, count=256):
        return float(flow.log_box(jnp.asarray(lower)[None, None], jnp.asarray(upper)[None, None], box_points(count, 2))[0, 0])

    second = jnp.linspace(-1.0, 0.8, 20001)
    assert box([0.3, -1.0], [0.3, 0.8], 1) == pytest.approx(float(jnp.log(_trapezoid(density(0.3, second), second))), abs=1e-4)
    first = jnp.linspace(-1.0, 0.5, 20001)
    assert box([-1.0, 0.2], [0.5, 0.2]) == pytest.approx(float(jnp.log(_trapezoid(density(first, 0.2), first))), abs=5e-4)
    first, second = jnp.linspace(-0.5, 0.7, 1501), jnp.linspace(-9.0, 0.4, 1501)
    inner = jax.vmap(lambda row: _trapezoid(row, second))(density(first[:, None], second[None, :]))
    assert box([-0.5, -INF], [0.7, 0.4]) == pytest.approx(float(jnp.log(_trapezoid(inner, first))), abs=5e-4)
    point = jnp.asarray([[[0.1, -0.3]]])
    assert box([0.1, -0.3], [0.1, -0.3], 1) == pytest.approx(float(flow.log_prob(point)[0, 0]), abs=1e-5)


@pytest.mark.slow
def test_flow_box_gradients_and_its_one_block_limit():
    flow = _flow()

    def score(context, w_in, w_out, lower, upper):
        return AutoregressiveFlow(context, w_in, w_out, 2).log_box(lower, upper, box_points(32, 2)).sum()

    for lower, upper in [([-0.5, -INF], [0.7, 0.4]), ([1.0, 0.5], [INF, INF]), ([30.0, -INF], [INF, -25.0])]:
        bounds = jnp.asarray(lower)[None, None], jnp.asarray(upper)[None, None]
        grads = jax.grad(score, argnums=(0, 1, 2))(flow.context, flow.w_in, flow.w_out, *bounds)
        assert all(bool(jnp.all(jnp.isfinite(part))) for part in grads)
    with pytest.raises(NotImplementedError):
        _flow(depth=2).log_box(jnp.zeros((1, 1, 2)), jnp.ones((1, 1, 2)))


@pytest.mark.slow
def test_the_joint_observed_objective_is_the_mean_member_box():
    rng = np.random.default_rng(4)
    raw = jnp.asarray(rng.normal(size=(6, 2, 3 * 4)))
    label = observed.censored(rng.normal(size=(6, 3)), rng.random((6, 3)) < 0.4)
    y = jnp.asarray(observed.fields(label), dtype=jnp.float32)
    loss = make_joint_observed_objective(3, "normal", points=64)(raw, y, jnp.ones(6))
    box = joint_normal_distribution(raw, 3).log_box(y[:, None, :, 0], y[:, None, :, 1], box_points(64, 3))
    assert float(loss) == pytest.approx(float(-jnp.mean(box)), rel=1e-6)


# ---------------------------------------------------------------- the estimator


def _shelves(n, seed):
    """Three products whose demands share a daily swing; each sells out when demand passes its stock."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    common = rng.normal(size=n)
    demand = 10.0 + 3.0 * np.stack(
        [x[:, 0] + 0.8 * common, -x[:, 1] + 0.8 * common, 0.5 * x[:, 0] * x[:, 1] + 0.6 * common], axis=1
    ) + 1.5 * rng.normal(size=(n, 3))
    stock = np.quantile(demand, 0.7, axis=0) + rng.normal(size=(n, 3))
    sold_out = demand >= stock
    return x, demand, np.minimum(demand, stock), sold_out


@pytest.mark.slow
@pytest.mark.parametrize("task", ["multivariate_gaussian", "multivariate_mixture", "multivariate_flow"])
def test_a_censored_joint_fit_recovers_the_demand_behind_the_sales(task):
    x, demand, sales, sold_out = _shelves(1600, seed=5)
    train, test = slice(0, 1200), slice(1200, None)
    label = observed.censored(sales, sold_out)
    censored = ASRRegressor(task=task, max_epochs=15, random_state=0).fit(x[train], label[train])
    naive = ASRRegressor(task=task, max_epochs=15, random_state=0).fit(x[train], sales[train])
    assert censored.log_prob(x[test], demand[test]).mean() > naive.log_prob(x[test], demand[test]).mean() + 0.5
    errors = np.abs(censored.predict(x[test]) - demand[test]).mean(), np.abs(naive.predict(x[test]) - demand[test]).mean()
    assert errors[0] < errors[1]
    values = censored.log_prob(x[test], label[test])
    assert values.shape == (400,) and np.isfinite(values).all()
    assert censored.score(x[test], label[test]) == pytest.approx(values.mean(), rel=1e-6)


@pytest.mark.slow
def test_the_joint_mixture_task_predicts_samples_and_saves(tmp_path):
    from ashlar import ASRModel

    x, demand, sales, sold_out = _shelves(600, seed=6)
    label = observed.censored(sales, sold_out)
    model = ASRRegressor(task="multivariate_mixture", head_kwargs={"components": 3}, max_epochs=3, random_state=0)
    model.fit(x[:500], label[:500], validation=(x[500:], label[500:]))
    dist = model.predict_dist(x[500:])
    assert dist.batch_shape == (100,) and dist.event_shape == (3,)
    assert np.allclose(np.asarray(dist.mean()), model.predict(x[500:]), atol=1e-3)
    assert model.sample(x[500:504], n=7).shape == (7, 4, 3)
    assert model.predict_quantiles(x[500:504], [0.1, 0.9]).shape == (4, 3, 2)
    assert model.predict_uncertainty(x[500:504])["total"].shape == (4, 3)
    restored = ASRModel.load(model.save(tmp_path / "mixture"))
    assert np.allclose(restored.log_prob(x[500:], label[500:]), model.log_prob(x[500:], label[500:]), atol=1e-4)
    exact = model.log_prob(x[500:], observed.exact(demand[500:]))
    assert np.allclose(exact, model.log_prob(x[500:], demand[500:]), atol=1e-4)


def test_joint_labels_that_cannot_be_trained_are_refused():
    x, _demand, sales, sold_out = _shelves(200, seed=7)
    label = observed.censored(sales, sold_out)
    with pytest.raises(AshlarError):
        ASRRegressor(task="multivariate_gaussian", max_epochs=1).fit(x, observed.censored(sales[:, 0], sold_out[:, 0]))
    with pytest.raises(AshlarError):
        ASRRegressor(task="gaussian", max_epochs=1).fit(x, label)
    with pytest.raises(AshlarError):
        ASRRegressor(task="multivariate_flow", head_kwargs={"depth": 2}, max_epochs=1).fit(x, label)
    with pytest.raises(ShapeError):
        ASRRegressor(task="multivariate_gaussian", n_events=2, max_epochs=1).fit(x, label)
    model = ASRRegressor(task="multivariate_gaussian", max_epochs=1).fit(x, label)
    with pytest.raises(AshlarError):
        model.calibrate(x, label)
    with pytest.raises(AshlarError):
        model.log_prob(x, label[:, :2])
    deep = ASRRegressor(task="multivariate_flow", head_kwargs={"depth": 2}, max_epochs=1).fit(x, observed.exact(sales))
    assert np.allclose(deep.log_prob(x, observed.exact(sales)), deep.log_prob(x, sales), atol=1e-4)
