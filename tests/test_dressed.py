"""ASRDressed: cross-fitted networks and label readers, pooled, smoothed and corrected into one distribution."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pandas as pd
import pytest

import ashlar
from ashlar import ASRDressed, observed
from ashlar.dressed import LEVELS, _Columns, atoms_crps
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


def _forced(model, gate, weight):
    """The fitted model with another gate and pool weight, to reach both sides of the gate."""
    other = copy.copy(model)
    other.gate_, other.pool_weight_ = gate, weight
    return other


def test_atoms_crps_matches_the_definition():
    rng = np.random.default_rng(0)
    atoms, masses, y = rng.normal(size=(5, 6)), rng.dirichlet(np.ones(6), size=5), rng.normal(size=5)
    first = (masses * np.abs(atoms - y[:, None])).sum(axis=1)
    spread = (masses[:, :, None] * masses[:, None, :] * np.abs(atoms[:, :, None] - atoms[:, None, :])).sum(axis=(1, 2))
    assert np.allclose(atoms_crps(atoms, masses, y), first - 0.5 * spread)


def test_columns_reads_numbers_as_they_are_and_other_columns_by_code():
    train = pd.DataFrame({"n": [1.0, np.nan, 3.0], "s": ["b", "a", None], "i": pd.array([1, None, 3], dtype="Int64")})
    columns = _Columns(train)
    assert np.array_equal(columns(train), [[1.0, 1.0, 1.0], [np.nan, 0.0, np.nan], [3.0, np.nan, 3.0]], equal_nan=True)
    new = pd.DataFrame({"n": [2.0], "s": ["z"], "i": pd.array([2], dtype="Int64")})
    assert np.array_equal(columns(new), [[2.0, np.nan, 2.0]], equal_nan=True)
    assert np.array_equal(_Columns(np.eye(2))(np.eye(2)), np.eye(2))
    with pytest.raises(AshlarError, match="DataFrame"):
        _Columns(np.eye(2))(np.asarray([["a", "b"]], dtype=object))


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
@pytest.mark.parametrize(("gate", "weight"), [(0.0, 0.5), (0.6, 0.5), (1.0, 1.0), (1.0, 0.0)])
def test_the_dressed_density_is_exact_on_both_sides_of_the_gate(density, gate, weight):
    model, X, y = density
    dist = _forced(model, gate, weight).predict_dist(X.iloc[:4])
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
