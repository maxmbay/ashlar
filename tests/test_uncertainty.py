"""Member disagreement as epistemic uncertainty."""

import numpy as np
import pytest

from ashlar import ASRClassifier, ASROrdinalClassifier, ASRRegressor
from ashlar.errors import AshlarError

TABM = {"backbone": "tabm", "backbone_kwargs": {"n_blocks": 2, "d_block": 64, "k": 8}}


def _gap(n=1200, seed=0):
    """Training rows on x in [-2, -0.5] and [0.5, 2] only; the noise grows with |x|."""
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-2, -0.5, n // 2), rng.uniform(0.5, 2, n // 2)])
    X = np.stack([x, rng.normal(size=n)], axis=1).astype(np.float32)
    y = np.sin(2 * x) + 0.1 * np.abs(x) * rng.normal(size=n)
    return X, y


@pytest.mark.parametrize("task", ["regression", "gaussian", "mixture", "quantile"])
def test_uncertainty_has_one_value_per_row_and_adds_up(task):
    X, y = _gap()
    model = ASRRegressor(task=task, max_epochs=6, **TABM).fit(X, y)
    parts = model.predict_uncertainty(X[:100], n_samples=50)
    assert parts["epistemic"].shape == (100,)
    assert np.all(parts["epistemic"] >= 0)
    if task in {"regression", "quantile"}:
        assert "aleatoric" not in parts
        assert np.allclose(parts["total"], parts["epistemic"])
    else:
        assert np.all(parts["aleatoric"] > 0)
        assert np.allclose(np.square(parts["total"]), np.square(parts["epistemic"]) + np.square(parts["aleatoric"]))


def test_members_disagree_more_far_from_the_data():
    X, y = _gap()
    model = ASRRegressor(task="gaussian", max_epochs=30, **TABM).fit(X, y)
    inside = np.stack([np.linspace(-1.5, 1.5, 40), np.zeros(40)], axis=1)[np.abs(np.linspace(-1.5, 1.5, 40)) > 0.6]
    outside = np.stack([np.linspace(4.0, 6.0, 20), np.zeros(20)], axis=1)
    near = model.predict_uncertainty(inside.astype(np.float32))["epistemic"].mean()
    far = model.predict_uncertainty(outside.astype(np.float32))["epistemic"].mean()
    assert far > near


def test_single_member_backbone_has_no_disagreement():
    X, y = _gap(400)
    parts = ASRRegressor(task="gaussian", max_epochs=2).fit(X, y).predict_uncertainty(X[:20], n_samples=20)
    assert np.allclose(parts["epistemic"], 0.0)


@pytest.mark.parametrize("cls", [ASRClassifier, ASROrdinalClassifier])
def test_class_uncertainty_is_mutual_information(cls):
    X, y = _gap()
    labels = np.digitize(y, [-0.5, 0.5])
    parts = cls(max_epochs=4, **TABM).fit(X, labels).predict_uncertainty(X[:50])
    assert np.all(parts["total"] <= np.log(3) + 1e-9)
    assert np.allclose(parts["epistemic"] + parts["aleatoric"], parts["total"])
    assert np.all(parts["epistemic"] >= 0)


def test_tree_tasks_are_refused():
    from ashlar import ASROutcomeTree

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2)).astype(np.float32)
    labels = np.where(X[:, 0] > 0, "a", "b")
    model = ASROutcomeTree({"a": None, "b": None}, max_epochs=1).fit(X, labels)
    with pytest.raises(AshlarError):
        model.predict_uncertainty(X)
