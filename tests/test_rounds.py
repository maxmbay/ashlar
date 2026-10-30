import numpy as np

from ashlar import DEFAULT_RECIPE, ASRRegressor
from ashlar.rounds import (
    HISTOGRAM_LEARNING_RATE,
    HISTOGRAM_MAX_ITER,
    HISTOGRAM_MAX_LEAF_NODES,
    HISTOGRAM_PATIENCE,
    LEAF_MAX_STAGES,
    LEAF_PATIENCE,
    LEAF_RIDGE,
    LEAF_SHRINKAGE,
    NEIGHBOR_EMBED_DIM,
    NEIGHBOR_LEARNING_RATE,
    NEIGHBOR_MAX_EPOCHS,
    NEIGHBOR_PATIENCE,
    ASRHistogramBooster,
    ASRLearnedNeighborhood,
    ASRLinearLeafBooster,
)


def test_accuracy_counts_each_row():
    from ashlar.rounds import _accuracy

    score = _accuracy(np.array([0, 1, 0]), np.array([0, 0, 0]))
    assert abs(score - (2 / 3)) < 1e-12


def test_residual_default_stays_a_regressor():
    model = ASRRegressor()
    assert model.task == "regression"
    assert model.standardize_target is True
    assert model.backbone == "residual_mlp"
    assert DEFAULT_RECIPE["task"] == "regression"
    assert DEFAULT_RECIPE["standardize_target"] is True
    assert HISTOGRAM_MAX_ITER == 300
    assert HISTOGRAM_PATIENCE == 20
    assert HISTOGRAM_LEARNING_RATE == 0.1
    assert HISTOGRAM_MAX_LEAF_NODES == 31
    assert LEAF_MAX_STAGES == 200
    assert LEAF_PATIENCE == 20
    assert LEAF_SHRINKAGE == 0.1
    assert LEAF_RIDGE == 1.0
    assert NEIGHBOR_MAX_EPOCHS == 80
    assert NEIGHBOR_PATIENCE == 16
    assert NEIGHBOR_LEARNING_RATE == 1e-3
    assert NEIGHBOR_EMBED_DIM == 32


def test_histogram_learns_a_step_and_an_ordered_score():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(600, 4))
    labels = (rows[:, 0] > 0).astype(np.int32)
    model = ASRHistogramBooster(random_state=0, max_iter=40, patience=8)
    model.fit(rows, labels, "classification")
    fresh = rng.normal(size=(200, 4))
    predicted = model.predict(fresh)
    assert float(np.mean(predicted == (fresh[:, 0] > 0).astype(np.int32))) > 0.85

    ordered = np.clip(np.rint(rows[:, 1] * 1.4 + 3), 1, 5).astype(np.int32)
    ordinal = ASRHistogramBooster(random_state=0, max_iter=40, patience=8)
    ordinal.fit(rows, ordered, "ordinal")
    held = rng.normal(size=(200, 4))
    truth = np.clip(np.rint(held[:, 1] * 1.4 + 3), 1, 5).astype(np.float64)
    assert _r2(truth, ordinal.predict_score(held)) > 0.45


def test_linear_leaf_learns_a_sign_interaction():
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(800, 3))
    labels = np.where(rows[:, 1] > 0, rows[:, 0], -rows[:, 0])
    model = ASRLinearLeafBooster(random_state=0, max_stages=30, patience=8)
    model.fit(rows, labels, "regression")
    fresh = rng.normal(size=(400, 3))
    truth = np.where(fresh[:, 1] > 0, fresh[:, 0], -fresh[:, 0])
    assert _r2(truth, model.predict_score(fresh)) > 0.85
    assert model.n_stages_ >= 1


def test_learned_neighborhood_separates_two_blobs():
    rng = np.random.default_rng(2)
    left = rng.normal(loc=-2.0, size=(250, 4))
    right = rng.normal(loc=2.0, size=(250, 4))
    rows = np.vstack([left, right])
    labels = np.array([0] * 250 + [1] * 250)
    model = ASRLearnedNeighborhood(random_state=0, max_epochs=12, patience=6, batch_size=64)
    model.fit(rows, labels, "classification")
    fresh = np.vstack([rng.normal(loc=-2.0, size=(80, 4)), rng.normal(loc=2.0, size=(80, 4))])
    truth = np.array([0] * 80 + [1] * 80)
    assert float(np.mean(model.predict(fresh) == truth)) > 0.9


def _r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    total = float(np.sum(np.square(y_true - y_true.mean())))
    residual = float(np.sum(np.square(y_true - y_pred)))
    return 1.0 - residual / total
