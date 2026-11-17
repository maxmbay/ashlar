"""ASRDressed: cross-fitted networks and label readers, pooled, smoothed and corrected into one distribution."""

from __future__ import annotations

import copy
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr, ndtri

import ashlar
from ashlar import ASRDressed, observed
from ashlar.dressed import FLOOR, LEVELS, _Columns, _floor_share, _solve, atoms_crps
from ashlar.errors import AshlarError
from micro import TABM_NET

NETWORK = {"backbone_kwargs": TABM_NET, "num_embedding_dim": 4, "num_embedding_bins": 8, "max_epochs": 3}
READER = {"heads": 2, "key_size": 4, "width": 8, "max_steps": 60}


def _table(n=260, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = (X["a"] + np.sin(2.0 * X["b"]) + 0.3 * rng.normal(size=n)).to_numpy()
    return X, y


@pytest.fixture(scope="module")
def density():
    X, y = _table()
    model = ASRDressed(members=2, network=NETWORK, reader=READER).fit(X.iloc[:200], y[:200])
    return model, X.iloc[200:], y[200:]


def _forced(model, gate, weight, floor=FLOOR):
    """The fitted model with another gate, pool weight and wide-normal share, to reach both sides of the gate."""
    other = copy.copy(model)
    other.gate_, other.pool_weight_, other.floor_ = gate, weight, floor
    return other


def test_atoms_crps_matches_the_definition():
    rng = np.random.default_rng(0)
    atoms, masses, y = rng.normal(size=(5, 6)), rng.dirichlet(np.ones(6), size=5), rng.normal(size=5)
    first = (masses * np.abs(atoms - y[:, None])).sum(axis=1)
    spread = (masses[:, :, None] * masses[:, None, :] * np.abs(atoms[:, :, None] - atoms[:, None, :])).sum(axis=(1, 2))
    assert np.allclose(atoms_crps(atoms, masses, y), first - 0.5 * spread)


def test_the_wide_normal_keeps_its_share_unless_it_would_cost_too_much():
    y = np.random.default_rng(0).normal(size=4000)
    assert _floor_share(y, 0.0, 1.0, 0.2, None) == FLOOR
    assert _floor_share(y, 0.0, 1.0, 0.2, 0.01) == FLOOR  # an ordinary table: 1% costs far less than 1% of its CRPS
    sharp = _floor_share(y, 0.0, 1.0, 1e-5, 0.01)  # a target the columns nearly determine
    wide = 1.0 / np.sqrt(np.pi)  # a standard normal's CRPS on its own draws
    assert np.isclose(sharp, np.sqrt(0.01 * 1e-5 / wide), rtol=0.05) and sharp < FLOOR


def test_columns_reads_numbers_as_they_are_and_other_columns_by_code():
    train = pd.DataFrame({"n": [1.0, np.nan, 3.0], "s": ["b", "a", None], "i": pd.array([1, None, 3], dtype="Int64")})
    columns = _Columns(train)
    assert np.array_equal(columns(train), [[1.0, 1.0, 1.0], [np.nan, 0.0, np.nan], [3.0, np.nan, 3.0]], equal_nan=True)
    new = pd.DataFrame({"n": [2.0], "s": ["z"], "i": pd.array([2], dtype="Int64")})
    assert np.array_equal(columns(new), [[2.0, np.nan, 2.0]], equal_nan=True)
    assert np.array_equal(_Columns(np.eye(2))(np.eye(2)), np.eye(2))
    with pytest.raises(AshlarError, match="DataFrame"):
        _Columns(np.eye(2))(np.asarray([["a", "b"]], dtype=object))


def _bracket(rows, low=-50.0, high=50.0):
    """What the quantile solver reads of its pools: the readers' spread and the ends of each row's bracket."""
    return [SimpleNamespace(reader=SimpleNamespace(spread=1.0, lo=np.full(rows, low), hi=np.full(rows, high)))]


def _normal_density(t):
    return np.exp(-0.5 * t**2) / np.sqrt(2 * np.pi)


def test_the_quantile_solver_leaves_a_density_that_is_far_too_large():
    """A standard normal whose density reads 50 above 8, where its CDF is 1: a corrected pool reads such a density
    beyond its network's float32 reach. The first Newton step from the far lower tail lands there and every step after
    it is too short to leave. Releases up to 0.1.1 returned 665.45 and 495.52 for the first two levels."""
    u, start = np.array([[0.001, 0.0025, 0.5]]), np.array([[-5.0, -5.0, 0.0]])

    def evaluate(i, j, y):
        t = y[i, j]
        return ndtr(t), np.where(t > 8.0, 50.0, _normal_density(t))

    q = _solve(_bracket(1, -1000.0, 1000.0), u, start, evaluate, 3e-7)
    assert np.abs(ndtr(q) - u).max() <= 3e-7
    assert np.allclose(q, ndtri(u), atol=1e-3)


def test_the_quantile_solver_ends_a_pass_from_side_to_side_of_a_steep_rise():
    """A narrow normal on a wide one, and a level inside the narrow one's rise. Newton steps land on either side of
    the rise in turn, each just inside the last, and the bracket stops closing. Releases up to 0.1.1 returned -0.0855
    from each of these starts, where the CDF is 0.202 from the level."""

    def cdf(t):
        return 0.8 * ndtr(t / 0.5) + 0.2 * ndtr(t / 0.04)

    starts = np.array([-1.0, -0.45, -0.3, -0.15])
    u = np.full((len(starts), 1), cdf(0.02))

    def evaluate(i, j, y):
        t = y[i, j]
        return cdf(t), 0.8 * _normal_density(t / 0.5) / 0.5 + 0.2 * _normal_density(t / 0.04) / 0.04

    q = _solve(_bracket(len(starts)), u, starts[:, None], evaluate, 3e-7)
    assert np.abs(cdf(q) - u).max() <= 3e-7
    assert np.abs(q - 0.02).max() < 1e-6


def test_the_quantile_solver_says_where_it_has_no_cdf_to_read():
    u, start = np.array([[0.3, 0.5]]), np.zeros((1, 2))

    def evaluate(i, j, y):
        t = y[i, j]
        return np.where(j == 0, np.nan, ndtr(t)), _normal_density(t)

    with pytest.warns(RuntimeWarning, match="not a number at 1 of the quantiles"):
        q = _solve(_bracket(1), u, start, evaluate, 3e-7)
    assert abs(ndtr(q[0, 1]) - 0.5) <= 3e-7 and q[0, 0] == 0.0  # the other level is solved; this one is its bracket's middle


def test_a_dressed_density_fits_and_predicts_one_distribution(density):
    model, X, y = density
    assert model.values_ is None and len(model.members_) == 2
    assert 0.0 <= model.pool_weight_ <= 1.0 and 0.0 <= model.gate_ <= 1.0
    assert model.bandwidth_factor_ in (1.0, 0.5, 0.25, 0.125, 2.0) and model.fallback_factor_ in (1.0, 0.5, 0.25, 0.125, 2.0)
    assert len(model.history_["gate_curve"]) == 21 and len(model.history_["reader_steps"]) == 2
    dist = model.predict_dist(X)
    assert dist.batch_shape == (60,) and dist.values is None
    q = model.predict_quantiles(X, [0.1, 0.5, 0.9])
    assert q.shape == (60, 3) and np.all(np.isfinite(q)) and np.all(np.diff(q, axis=1) > 0)
    assert np.allclose(dist.quantile(0.5), q[:, 1])
    log_density = model.log_prob(X, y)
    assert log_density.shape == (60,) and np.all(np.isfinite(log_density))
    assert np.allclose(log_density, dist.log_prob(y))
    assert model.predict(X).shape == (60,) and np.isfinite(model.score(X, y))
    assert model.sample(X, n=7, seed=1).shape == (7, 60)
    crps = dist.crps(y)
    assert crps.shape == (60,) and np.all(crps > 0)
    with pytest.raises(AshlarError, match="few values"):
        dist.probabilities()
    with pytest.raises(AshlarError, match="between 0 and 1"):
        dist.quantile([0.0, 0.5])


@pytest.mark.slow
@pytest.mark.parametrize(("gate", "weight", "floor"), [(0.0, 0.5, FLOOR), (0.6, 0.5, FLOOR), (1.0, 1.0, FLOOR), (1.0, 0.0, 1e-4)])
def test_the_dressed_density_is_exact_on_both_sides_of_the_gate(density, gate, weight, floor):
    model, X, y = density
    dist = _forced(model, gate, weight, floor).predict_dist(X.iloc[:4])
    q = dist.quantile(LEVELS)
    assert np.all(np.diff(q, axis=0) >= 0)
    assert np.abs(dist.cdf(q) - LEVELS[:, None]).max() < 5e-6
    grid = np.linspace(y.min() - 6.0, y.max() + 6.0, 3001)[:, None] * np.ones((1, 4))
    pdf, cdf = np.exp(dist.log_prob(grid)), dist.cdf(grid)
    assert np.allclose(np.trapezoid(pdf, grid, axis=0), 1.0, atol=2e-3)
    assert np.all(cdf[0] < 1e-4) and np.all(cdf[-1] > 1.0 - 1e-4) and np.all(np.diff(cdf, axis=0) >= -1e-6)
    assert np.abs(pdf - np.gradient(cdf, grid[:, 0], axis=0))[5:-5].max() < 2e-2 * pdf.max()


@pytest.mark.slow
def test_a_dressed_model_pickles(density):
    model, X, _ = density
    again = pickle.loads(pickle.dumps(model))
    assert np.allclose(again.predict_quantiles(X.iloc[:5]), model.predict_quantiles(X.iloc[:5]))


@pytest.mark.slow
def test_a_target_with_few_values_gets_a_probability_for_each_value():
    X, y = _table(seed=1)
    X["kind"] = np.where(X["c"] > 0, "up", "down")  # a text column is read by the networks and, by code, the readers
    y = np.round(np.clip(y, -2.0, 2.0))
    model = ashlar.dressed(members=3, reader=READER, **NETWORK).fit(X.iloc[:200], y[:200])
    assert isinstance(model, ASRDressed) and np.array_equal(model.values_, np.unique(y[:200]))
    values = model.values_
    for other in (model, _forced(model, 0.0, 0.5), _forced(model, 1.0, 0.5)):
        dist = other.predict_dist(X.iloc[200:])
        proba = dist.probabilities()
        assert proba.shape == (60, len(values)) and np.allclose(proba.sum(axis=1), 1.0) and np.all(proba >= 0.0)
        assert np.allclose(dist.mean(), proba @ values)
        at = np.searchsorted(values, y[200:])
        assert np.allclose(dist.log_prob(y[200:]), np.log(proba[np.arange(60), at]))
        assert np.all(np.isneginf(dist.log_prob(np.full(60, 0.5))))
        assert np.allclose(dist.cdf(np.full(60, values[-1])), 1.0) and np.allclose(dist.cdf(np.full(60, values[0] - 1.0)), 0.0)
        assert np.all(np.isin(dist.quantile([0.1, 0.5, 0.9]), values))
        assert np.all(np.isin(dist.sample(5), values))
        assert np.allclose(dist.crps(y[200:]), atoms_crps(np.broadcast_to(values, proba.shape), proba, y[200:]))
    assert ASRDressed(members=3, few_values=0).few_values == 0


def test_scikit_learn_reads_a_dressed_model_as_a_regressor():
    from sklearn.base import clone, is_regressor

    model = ashlar.dressed(learning_rate=1e-3)
    assert is_regressor(model) and is_regressor(clone(model))
    assert clone(model).get_params()["network"] == {"learning_rate": 1e-3}


def test_dressed_refuses_what_it_does_not_support():
    X, y = _table(n=120)
    model = ASRDressed(members=3, network=NETWORK, reader=READER)
    with pytest.raises(AshlarError, match="sample weights"):
        model.fit(X, y, sample_weight=np.ones(120))
    with pytest.raises(AshlarError, match="censored"):
        model.fit(X, observed.censored(y, y > 0))
    with pytest.raises(AshlarError, match="one target column"):
        model.fit(X, np.stack([y, y], axis=1))
    with pytest.raises(AshlarError, match="numeric target"):
        model.fit(X, np.where(y > 0, "a", "b"))
    with pytest.raises(AshlarError, match="at least 60 rows"):
        model.fit(X.iloc[:50], y[:50])
    with pytest.raises(AshlarError, match="finite"):
        model.fit(X, np.where(np.arange(120) == 3, np.nan, y))
    with pytest.raises(AshlarError, match="members"):
        ASRDressed(members=1).fit(X, y)
    with pytest.raises(AshlarError, match="leave them out"):
        ASRDressed(members=3, network={"task": "mixture"}).fit(X, y)
    with pytest.raises(AshlarError, match="not fitted"):
        model.predict_dist(X)
