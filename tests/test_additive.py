import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ashlar import DEFAULT_RECIPE, ASRRegressor
from ashlar.additive import (
    BASIS_WIDTH,
    EMBED_DIM,
    PAIR_CAP,
    RIDGE_GRID,
    AdditiveBody,
    ASRAdditive,
    ASRDiagonalNeighborhood,
    ASRGatedAdditive,
    NeighborhoodBody,
    _fit_ridge,
)


def test_residual_default_stays_a_regressor():
    model = ASRRegressor()
    assert model.task == "regression"
    assert model.standardize_target is True
    assert model.backbone == "residual_mlp"
    assert DEFAULT_RECIPE["standardize_target"] is True
    assert DEFAULT_RECIPE["task"] == "regression"
    assert PAIR_CAP == 10
    assert RIDGE_GRID == (1e-4, 1e-2, 1.0, 100.0)
    assert EMBED_DIM == 8
    assert BASIS_WIDTH == 9


def test_ridge_solution_is_a_critical_point():
    rng = np.random.default_rng(0)
    design = rng.normal(size=(40, 3))
    y = design @ np.array([1.5, -2.0, 0.25]) + 0.3
    coef, intercept = _fit_ridge(design, y, ridge=0.1)
    n_rows, n_columns = design.shape
    residual = design @ coef + intercept - y
    slope_grad = (2.0 / n_rows) * design.T @ residual + (2.0 * 0.1 / n_columns) * coef
    intercept_grad = (2.0 / n_rows) * residual.sum()
    assert abs(intercept_grad) < 1e-8
    assert np.max(np.abs(slope_grad)) < 1e-8


def test_additive_body_has_a_nonzero_coefficient_gradient():
    coefficients = jnp.zeros((3, 1))
    body = AdditiveBody(coefficients, jnp.zeros((1,)), (), ())
    basis = jnp.asarray(np.array([[1.0, 0.0, -1.0], [0.5, 2.0, 0.25]]))
    target = jnp.asarray(np.array([1.0, -1.0]))

    def loss(weights):
        updated = eqx.tree_at(lambda module: module.coefficients, body, weights)
        prediction = updated.logits(basis)[:, 0]
        return jnp.mean((prediction - target) ** 2)

    gradient = jax.grad(loss)(coefficients)
    assert float(jnp.linalg.norm(gradient)) > 0


def test_embedding_gradient_reaches_the_width_eight_table():
    table = jnp.zeros((2, EMBED_DIM))
    projection = jnp.asarray(np.ones((EMBED_DIM, 1)))
    body = AdditiveBody(jnp.zeros((1, 1)), jnp.zeros((1,)), (table,), (projection,))
    basis = jnp.ones((2, 1))
    levels = (jnp.asarray(np.array([0, 1])),)
    target = jnp.asarray(np.array([1.0, -1.0]))

    def loss(weights):
        updated = eqx.tree_at(lambda module: module.embeddings, body, (weights,))
        prediction = updated.logits(basis, levels)[:, 0]
        return jnp.mean((prediction - target) ** 2)

    gradient = jax.grad(loss)(table)
    assert gradient.shape == (2, EMBED_DIM)
    assert float(jnp.linalg.norm(gradient)) > 0


def test_pair_selection_recovers_a_product_and_stops_at_the_cap():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(360, 4))
    target = 3.0 * rows[:, 0] * rows[:, 1] + 0.05 * rng.normal(size=360)
    model = ASRAdditive(random_state=0).fit(rows, target, "regression")
    assert len(model.pairs_) <= PAIR_CAP
    assert (0, 1) in model.pairs_
    assert model.predict_score(rows).shape == (360,)
    assert model.n_fit_ + model.n_val_ == 360


def test_ordinal_expectation_uses_the_observed_class_values():
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(240, 2))
    labels = np.where(rows[:, 0] + rows[:, 1] > 0, 7.0, 3.0)
    model = ASRAdditive(random_state=0).fit(rows, labels, "ordinal")
    score = model.predict_score(rows)
    assert score.min() >= 2.5
    assert score.max() <= 7.5
    assert set(np.asarray(model.classes_).tolist()) == {3.0, 7.0}


def test_categorical_embedding_is_width_eight_and_changes_the_fit():
    rng = np.random.default_rng(2)
    import pandas as pd

    frame = pd.DataFrame(
        {
            "x": rng.normal(size=180),
            "group": np.where(rng.normal(size=180) > 0, "a", "b"),
        }
    )
    labels = (frame["group"] == "a").astype(np.int32)
    model = ASRAdditive(random_state=0).fit(frame, labels.to_numpy(), "classification")
    assert model.embedding_tables_[0].shape == (3, EMBED_DIM)
    prediction = model.predict(frame)
    assert accuracy(labels.to_numpy(), prediction) > 0.8


def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def test_gate_and_neighborhood_fit_a_binary_table():
    rng = np.random.default_rng(3)
    rows = rng.normal(size=(280, 2))
    labels = (rows[:, 0] > 0).astype(np.int32)
    gate = ASRGatedAdditive(random_state=0).fit(rows, labels, "classification")
    assert gate.kind_ == "feature"
    assert gate.predict(rows).shape == (280,)
    neighborhood = ASRDiagonalNeighborhood(random_state=0).fit(rows, labels, "classification")
    assert neighborhood.scales_.shape == (2,)
    assert neighborhood.predict(rows).shape == (280,)
    body = NeighborhoodBody(
        scales=jnp.ones((2,)),
        bandwidth=jnp.asarray(1.0),
        reference=jnp.asarray(rows[:20]),
        values=jnp.asarray(labels[:20].astype(np.float64)),
    )

    def loss(scales):
        updated = eqx.tree_at(lambda module: module.scales, body, scales)
        return jnp.mean((updated.predict(jnp.asarray(rows[20:40])) - labels[20:40]) ** 2)

    gradient = jax.grad(loss)(jnp.ones((2,)))
    assert float(jnp.linalg.norm(gradient)) > 0
