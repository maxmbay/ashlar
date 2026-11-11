"""The label reader: a predictive distribution read from the training labels, and the mixture scores built on it."""

from __future__ import annotations

import numpy as np

from ashlar.label_reader import POOL_WEIGHTS, LabelReader, mixture_crps, mixture_quantiles, pool_weight

SMALL = {"heads": 2, "key_size": 4, "width": 8, "max_steps": 200}


def _direct_crps(atoms, masses, y):
    """CRPS of a weighted empirical distribution by its definition: E|X - y| - E|X - X'| / 2."""
    first = (masses * np.abs(atoms - y[:, None])).sum(axis=1)
    spread = (masses[:, :, None] * masses[:, None, :] * np.abs(atoms[:, :, None] - atoms[:, None, :])).sum(axis=(1, 2))
    return first - 0.5 * spread


def test_mixture_crps_matches_the_definition_at_every_weight():
    rng = np.random.default_rng(0)
    first = (rng.normal(size=(6, 5)), rng.dirichlet(np.ones(5), size=6))
    second = (rng.normal(size=(6, 7)), rng.dirichlet(np.ones(7), size=6))
    y = rng.normal(size=6)
    got = mixture_crps(first, second, y)
    for w, value in zip(POOL_WEIGHTS[[0, 7, 20]], got[[0, 7, 20]], strict=True):
        atoms = np.concatenate([first[0], second[0]], axis=1)
        masses = np.concatenate([w * first[1], (1 - w) * second[1]], axis=1)
        assert np.isclose(value, _direct_crps(atoms, masses, y).mean())
    weight, scores = pool_weight(first, second, y)
    assert weight == POOL_WEIGHTS[int(np.argmin(scores))] and np.allclose(scores, got)


def test_mixture_quantiles_are_the_smallest_atoms_reaching_each_level():
    atoms = np.asarray([[0.0, 1.0, 2.0, 3.0]])
    masses = np.asarray([[0.25, 0.25, 0.25, 0.25]])
    q = mixture_quantiles((atoms, masses), (atoms, masses), 1.0, [0.1, 0.25, 0.5, 0.9])
    assert np.array_equal(q, [[0.0, 0.0, 1.0, 3.0]])


def test_the_reader_reads_a_linear_target_from_its_labels():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(260, 3))
    X[::17, 2] = np.nan  # a missing cell reads as the column's middle
    y = 2.0 * X[:, 0] - X[:, 1] + 0.01 * rng.normal(size=260)
    reader = LabelReader(random_state=0, **SMALL).fit(X[:160], y[:160], validation=(X[160:200], y[160:200]))
    atoms, masses = reader.atoms(X[200:])
    assert atoms.shape == masses.shape == (60, 200)  # the validation rows have joined the context
    assert np.allclose(masses.sum(axis=1), 1.0) and np.all(masses >= 0.0)
    assert reader.atoms(X[200:], context="train")[0].shape == (60, 160)
    mean = (atoms * masses).sum(axis=1)
    assert np.abs(mean - y[200:]).mean() < 0.25 * np.abs(y[200:] - y[:200].mean()).mean()
