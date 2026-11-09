"""A cooldown after early stopping: the best checkpoint, annealed to a zero rate (issue #89)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

from ashlar import ASRRegressor
from ashlar.errors import AshlarError
from ashlar.train import _rate_at, cooldown_optimizer
from micro import MICRO


def _rows(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    return X, X["a"].to_numpy() + 0.3 * rng.normal(size=n)


# A high constant rate on a few hundred rows overfits within a few dozen epochs, so early stopping fires.
OVERFIT = {**MICRO, "task": "gaussian", "max_epochs": 60, "patience": 3, "learning_rate": 2e-2, "learning_rate_schedule": "constant"}


def _adam(state):
    """The AdamW moments in an optimizer state."""
    nodes = jax.tree_util.tree_leaves(state, is_leaf=lambda node: isinstance(node, optax.ScaleByAdamState))
    return next(node for node in nodes if isinstance(node, optax.ScaleByAdamState))


def _steps(model) -> int:
    """AdamW updates the fitted model's optimizer state has taken."""
    return int(_adam(model.opt_state_).count)


def test_the_cooldown_rate_falls_from_its_start_to_zero():
    optimizer = cooldown_optimizer(0.01, 4, weight_decay=0.0)
    params = {"w": jnp.zeros(2)}
    state = optimizer.init(params)
    moves = []
    for _ in range(6):
        updates, state = optimizer.update({"w": jnp.ones(2)}, state, params)
        moves.append(float(jnp.abs(updates["w"][0])))
    # Adam's normalized step is about one in size, so each move is the rate: 0.01, 0.0075, 0.005, 0.0025, then 0.
    assert np.allclose(moves[:4], [0.01, 0.0075, 0.005, 0.0025], rtol=1e-3)
    assert moves[4] == moves[5] == 0.0


def test_the_starting_rate_is_the_schedules_at_the_checkpoint():
    schedule = optax.linear_schedule(1.0, 0.0, 10)
    state = optax.adamw(schedule).init({"w": jnp.zeros(1)})
    assert _rate_at(0.3, state) == 0.3
    assert _rate_at(schedule, state) == 1.0
    four = (_adam(state)._replace(count=jnp.asarray(4, dtype=jnp.int32)), *state[1:])
    assert np.isclose(_rate_at(schedule, four), 0.6)


@pytest.mark.slow
def test_a_cooldown_anneals_the_best_checkpoint_and_records_it():
    X, y = _rows()
    plain = ASRRegressor(**OVERFIT).fit(X, y)
    cooled = ASRRegressor(**OVERFIT, cooldown_epochs=3).fit(X, y)
    history = cooled.history_
    best = history["best_epoch"]
    # Early stopping fired: the last three epochs did not improve, and both fits pick the same checkpoint.
    assert best == plain.history_["best_epoch"] == len(history["val_loss"]) - 3
    assert history["cooldown_epochs"] == 3 and len(history["cooldown_val_loss"]) == 3
    assert len(history["cooldown_train_loss"]) == 3
    assert np.isclose(history["cooldown_start_rate"], 2e-2)
    assert "cooldown_epochs" not in plain.history_
    # 240 training rows make one minibatch an epoch: the plain fit stops at the best epoch's update count,
    # and the cooldown carries those moments three epochs on.
    assert _steps(plain) == best and _steps(cooled) == best + 3
    assert not np.allclose(plain.predict(X), cooled.predict(X))


@pytest.mark.slow
def test_patience_zero_still_cools_the_best_epoch_and_no_validation_means_no_cooldown():
    X, y = _rows()
    full = ASRRegressor(**{**OVERFIT, "max_epochs": 12, "patience": 0}, cooldown_epochs=2).fit(X, y)
    assert len(full.history_["val_loss"]) == 12
    assert full.history_["best_epoch"] <= 12 and len(full.history_["cooldown_val_loss"]) == 2
    bare = ASRRegressor(**{**OVERFIT, "max_epochs": 5}, validation_size=0.0, cooldown_epochs=2).fit(X, y)
    assert "best_epoch" not in bare.history_ and "cooldown_epochs" not in bare.history_
    assert _steps(bare) == 5 * 2  # 300 rows at a batch of 256: two minibatches an epoch


@pytest.mark.slow
def test_a_cooled_model_saves_loads_and_resumes(tmp_path):
    X, y = _rows()
    model = ASRRegressor(**OVERFIT, cooldown_epochs=2).fit(X, y)
    restored = type(model).load(model.save(tmp_path / "model"))
    assert restored.cooldown_epochs == 2
    assert np.allclose(restored.predict(X), model.predict(X), atol=1e-6)
    restored.fit(X, y, resume=True)
    assert restored.history_["cooldown_val_loss"]


def test_cooldown_epochs_must_be_a_non_negative_integer():
    X, y = _rows(40)
    for bad in (-1, 1.5, "3"):
        with pytest.raises(AshlarError, match="cooldown_epochs"):
            ASRRegressor(**MICRO, max_epochs=1, cooldown_epochs=bad).fit(X, y)
