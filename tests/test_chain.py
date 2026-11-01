"""Chained outcomes of different kinds: counts, amounts, classes, and continuous values in one joint."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import ASRRegressor, observed
from ashlar.chain import Chain, ChainHead, chain_columns, column_distribution, resolve_columns
from ashlar.errors import AshlarError, ShapeError

INF = np.inf
COLUMNS = ["poisson", "hurdle", "categorical", "gaussian"]


def _table(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    count = rng.poisson(np.exp(-0.5 + 0.7 * x[:, 0]))
    amount = np.where(count > 0, np.exp(1.0 + 0.5 * np.minimum(count, 3) + 0.3 * x[:, 1] + 0.4 * rng.normal(size=n)), 0.0)
    kind = (rng.random(n) < 1.0 / (1.0 + np.exp(-(x[:, 2] + 1.5 * (count > 0) - 0.5)))).astype(float)
    value = 0.5 * np.log1p(amount) + kind + rng.normal(size=n)
    return x, np.stack([count, amount, kind, value], axis=1).astype(float)


def _head(columns, key=0, k=2, d_in=5):
    columns = resolve_columns(chain_columns(columns), np.array([[1.0, 2.0, 1.0, 0.3], [0.0, 0.0, 0.0, -0.2]])[:, : len(columns)])
    return ChainHead(d_in, k, n_events=len(columns), n_classes=2, key=jax.random.PRNGKey(key), columns=columns), columns


def _random_chain(columns=COLUMNS, rows=4, k=2, seed=0, noise=0.3):
    head, columns = _head(columns, seed, k)
    # Perturb the weights so every term of the conditioner matters.
    leaves, tree = jax.tree.flatten(head)
    keys = jax.random.split(jax.random.PRNGKey(seed + 1), len(leaves))
    leaves = [leaf + noise * jax.random.normal(key, leaf.shape) for leaf, key in zip(leaves, keys)]
    head = jax.tree.unflatten(tree, leaves)
    hidden = jax.random.normal(jax.random.PRNGKey(seed + 2), (rows, k, 5))
    return Chain(*head(hidden, key=None), columns), columns


def test_columns_are_read_and_checked():
    columns = chain_columns(["poisson", {"kind": "hurdle", "severity": "gamma"}, {"kind": "categorical", "classes": 3}])
    assert [column.kind for column in columns] == ["poisson", "hurdle", "categorical"]
    assert columns[1].severity == "gamma" and columns[2].classes == 3 and columns[2].width == 3
    assert chain_columns([column.to_json() for column in columns]) == columns
    for bad in (None, "poisson", [], ["nope"], [{"kind": "poisson", "bins": 3}], [{"kind": "hurdle", "severity": "beta"}]):
        with pytest.raises(AshlarError):
            chain_columns(bad)
    with pytest.raises(AshlarError, match="whole numbers"):
        resolve_columns(chain_columns(["poisson"]), np.array([[0.5]]))
    with pytest.raises(AshlarError, match="non-negative"):
        resolve_columns(chain_columns(["hurdle"]), np.array([[-1.0]]))
    resolved = resolve_columns(chain_columns(["categorical"]), np.array([[0.0], [3.0]]))
    assert resolved[0].classes == 4


def test_joint_density_is_the_sum_of_the_conditionals():
    dist, columns = _random_chain()
    value = jnp.asarray([[2.0, 7.5, 1.0, 0.3], [0.0, 0.0, 0.0, -1.0], [1.0, 3.0, 0.0, 2.0], [3.0, 20.0, 1.0, 0.0]])
    value = value[:, None, :]
    total = 0.0
    for index, column in enumerate(columns):
        total = total + dist.conditional(index, value).log_prob(value[..., index])
    assert np.allclose(dist.log_prob(value), total, atol=1e-5)
    # The first column reads nothing before it: changing the later columns leaves its conditional alone.
    other = value.at[..., 1:].set(5.0)
    assert np.allclose(dist.conditional(0, value).log_prob(value[..., 0]), dist.conditional(0, other).log_prob(value[..., 0]))
    # A later column reads the ones before it, and only those.
    earlier = value.at[..., 0].set(4.0)
    assert not np.allclose(dist.conditional(1, value).log_prob(value[..., 1]), dist.conditional(1, earlier).log_prob(value[..., 1]))
    assert np.allclose(dist.conditional(1, value).log_prob(value[..., 1]), dist.conditional(1, other.at[..., 1].set(value[..., 1])).log_prob(value[..., 1]))


def test_the_mass_sums_to_one_over_counts_and_classes():
    dist, columns = _random_chain(["poisson", "categorical"], rows=3, noise=0.05)
    counts, classes = np.meshgrid(np.arange(60.0), np.arange(float(columns[1].classes)), indexing="ij")
    grid = jnp.asarray(np.stack([counts.ravel(), classes.ravel()], axis=-1))[:, None, None, :]
    mass = np.exp(np.asarray(dist.log_prob(grid))).sum(axis=0)
    assert np.allclose(mass, 1.0, atol=1e-4)


def test_box_likelihood_sums_classes_and_scores_the_last_column_interval():
    dist, columns = _random_chain(["poisson", "categorical", "gaussian"], rows=3)
    values = jnp.asarray([[1.0, 0.0, 0.2], [0.0, 1.0, -0.5], [2.0, 1.0, 1.0]])[:, None, :]
    # Every cell exact: the density.
    assert np.allclose(dist.log_box(values, values), dist.log_prob(values), atol=1e-5)
    # The class unknown, the last value censored: the sum over classes of p(class) P(value >= v | class).
    lower = values.at[..., 1].set(-INF)
    upper = values.at[..., 1].set(INF).at[..., 2].set(INF)
    expected = 0.0
    for code in range(columns[1].classes):
        filled = values.at[..., 1].set(code)
        part = dist.conditional(0, filled).log_prob(filled[..., 0]) + dist.conditional(1, filled).log_prob(filled[..., 1])
        part = part + dist.conditional(2, filled).log_survival(filled[..., 2])
        expected = expected + np.exp(np.asarray(part))
    assert np.allclose(dist.log_box(lower, upper), np.log(expected), atol=1e-5)
    # The last column left open leaves the marginal of the others.
    lower = values.at[..., 2].set(-INF)
    upper = values.at[..., 2].set(INF)
    marginal = dist.conditional(0, values).log_prob(values[..., 0]) + dist.conditional(1, values).log_prob(values[..., 1])
    assert np.allclose(dist.log_box(lower, upper), marginal, atol=1e-5)


def test_draws_follow_the_conditionals():
    dist, columns = _random_chain(["poisson", "hurdle"], rows=2, k=1)
    draws = np.asarray(dist.sample(jax.random.PRNGKey(3), (20_000,)))
    assert draws.shape == (20_000, 2, 1, 2)
    count = column_distribution(columns[0], dist._params(0, dist.context, jnp.zeros((2, 1, 2))))
    assert np.allclose(draws[..., 0].mean(axis=0), np.asarray(count.mean()), rtol=0.05)
    assert np.all(draws[..., 0] == np.round(draws[..., 0])) and np.all(draws[..., 1] >= 0)


def test_chain_fits_predicts_and_saves(tmp_path):
    x, y = _table(600)
    model = ASRRegressor(task="chain", head_kwargs={"columns": COLUMNS}, max_epochs=3, random_state=0).fit(x, y)
    assert model.chain_columns_[2].classes == 2
    assert np.allclose(model.target_mean_[:3], 0.0) and np.allclose(model.target_std_[:3], 1.0)
    assert model.target_std_[3] != 1.0
    values = model.log_prob(x[:20], y[:20])
    assert values.shape == (20,) and np.isfinite(values).all()
    draws = model.sample(x[:5], n=7)
    assert draws.shape == (7, 5, 4)
    assert np.all(draws[..., 0] == np.round(draws[..., 0])) and np.all(np.isin(draws[..., 2], [0.0, 1.0]))
    assert model.predict(x[:5]).shape == (5, 4)
    assert model.predict_interval(x[:5], target="sum", n_samples=50).shape == (5, 2)
    path = model.save(tmp_path / "chain")
    loaded = ASRRegressor.load(path)
    assert np.allclose(loaded.log_prob(x[:20], y[:20]), values, atol=1e-5)
    assert loaded._chain_columns() == model.chain_columns_


def test_chain_refuses_what_it_cannot_score():
    x, y = _table(200)
    with pytest.raises(AshlarError, match="columns"):
        ASRRegressor(task="chain").fit(x, y)
    with pytest.raises(AshlarError, match="log-likelihood"):
        ASRRegressor(task="chain", head_kwargs={"columns": COLUMNS}, objective="energy").fit(x, y)
    with pytest.raises((AshlarError, ShapeError)):
        ASRRegressor(task="chain", head_kwargs={"columns": COLUMNS[:3]}, max_epochs=1).fit(x, y)
    bad = y.copy()
    bad[::2, 0] = 0.5
    with pytest.raises(AshlarError, match="whole numbers"):
        ASRRegressor(task="chain", head_kwargs={"columns": COLUMNS}, max_epochs=1).fit(x, bad)
    # A coarse cell before the last column is refused unless its column is categorical.
    label = observed.stack(
        [observed.censored(y[:, 0], np.ones(len(y), bool)), observed.exact(y[:, 1]), observed.exact(y[:, 2]), observed.exact(y[:, 3])]
    )
    with pytest.raises(AshlarError, match="before the last column"):
        ASRRegressor(task="chain", head_kwargs={"columns": COLUMNS}, max_epochs=1).fit(x, label)


@pytest.mark.slow
def test_chain_learns_that_the_amount_follows_the_count():
    x, y = _table(4000, seed=1)
    pair = y[:, :2]
    model = ASRRegressor(task="chain", head_kwargs={"columns": ["poisson", "hurdle"]}, random_state=0).fit(x[:3000], pair[:3000])
    draws = model.sample(x[3000:], n=200, seed=0)
    # An amount is zero exactly when there was no claim.
    agree = np.mean((draws[..., 0] == 0) == (draws[..., 1] == 0))
    assert agree > 0.97
    independent = ASRRegressor(task="poisson", random_state=0).fit(x[:3000], pair[:3000, 0])
    amount = ASRRegressor(task="hurdle", random_state=0).fit(x[:3000], pair[:3000, 1])
    separate = independent.log_prob(x[3000:], pair[3000:, 0]) + amount.log_prob(x[3000:], pair[3000:, 1])
    assert model.log_prob(x[3000:], pair[3000:]).mean() > separate.mean() + 0.1


@pytest.mark.slow
def test_chain_trains_on_unknown_classes_and_a_censored_last_column():
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(size=(n, 2)).astype(np.float32)
    cause = (rng.random(n) < 1.0 / (1.0 + np.exp(-x[:, 0]))).astype(float)
    time = np.where(cause == 1, 0.5, -0.5) + 0.5 * x[:, 1] + 0.5 * rng.normal(size=n)
    limit = rng.normal(0.5, 1.0, size=n)
    seen = time <= limit
    label = observed.stack(
        [
            observed.interval(np.where(seen, cause, -INF), np.where(seen, cause, INF)),
            observed.censored(np.minimum(time, limit), ~seen),
        ]
    )
    model = ASRRegressor(task="chain", head_kwargs={"columns": ["categorical", "gaussian"]}, random_state=0).fit(x, label)
    draws = model.sample(x, n=100, seed=1)
    # The share of each cause is learned from every row, censored ones included.
    assert abs(draws[..., 0].mean() - cause.mean()) < 0.05
    assert np.isfinite(model.log_prob(x[:50], label[:50])).all()
