import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar import DEFAULT_RECIPE, ASRRegressor
from ashlar.errors import AshlarError
from ashlar.forecast import (
    FORECAST_D_MODEL,
    FORECAST_HEADS,
    FORECAST_HORIZON,
    FORECAST_LAYERS,
    FORECAST_LEARNING_RATE,
    FORECAST_LOOKBACK,
    ASRHistoryForecaster,
    ASRForecaster,
    _forecast_windows,
    _history_batch,
    _history_mse,
    _history_origins,
    _horizon_targets,
    _init_forecast,
    _init_history,
    _mse,
    _pack_history,
)


def test_forecast_leaves_the_residual_default():
    model = ASRRegressor()
    assert model.task == "regression"
    assert model.standardize_target is True
    assert model.backbone == "residual_mlp"
    assert DEFAULT_RECIPE["task"] == "regression"
    assert DEFAULT_RECIPE["standardize_target"] is True
    assert FORECAST_LOOKBACK == 48
    assert FORECAST_HORIZON == 12
    assert FORECAST_D_MODEL == 32
    assert FORECAST_HEADS == 4
    assert FORECAST_LAYERS == 2
    assert FORECAST_LEARNING_RATE == 1e-3


def test_forecast_windows_do_not_include_their_target():
    series = np.arange(200, dtype=np.float64)
    origins, inputs, targets = _forecast_windows(series, 48, 12)
    assert inputs.shape[1] == 48 and targets.shape[1] == 12
    assert np.array_equal(inputs[0], series[:48])
    assert np.array_equal(targets[0], series[48:60])
    val_start = int(len(series) * 0.9)
    train = origins + 12 <= val_start
    assert targets[train].max() < val_start
    assert inputs[train].max() < val_start


def test_forecast_loss_is_differentiable():
    net = _init_forecast(jax.random.PRNGKey(0), 16, 4)
    windows = jnp.asarray(np.random.default_rng(0).normal(size=(6, 16)), dtype=jnp.float32)
    target = jnp.asarray(np.random.default_rng(1).normal(size=(6, 4)), dtype=jnp.float32)
    loss, grads = eqx.filter_value_and_grad(_mse)(net, windows, target)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert float(jnp.sum(jnp.abs(grads.position))) > 0.0
    pred = jax.vmap(lambda window: net(window))(windows)
    assert abs(float(loss) - float(jnp.mean(jnp.square(pred - target)))) < 1e-5


def test_forecast_learns_a_sine_and_rejects_a_short_series():
    time = np.arange(900, dtype=np.float64)
    series = np.sin(2.0 * np.pi * time / 24.0)
    cut = int(len(series) * 0.8)
    model = ASRForecaster(random_state=0, max_epochs=20, patience=6).fit(series[:cut])
    _origins, inputs, targets = _forecast_windows(series, model.lookback, model.horizon)
    test = _origins >= cut
    pred = model.predict(inputs[test])
    mae = float(np.mean(np.abs(pred - targets[test])))
    mean_mae = float(np.mean(np.abs(targets[test] - series[:cut].mean())))
    assert mae < 0.25
    assert mae < mean_mae
    try:
        ASRForecaster(max_epochs=1, patience=1).fit(np.arange(20, dtype=np.float64))
    except AshlarError:
        pass
    else:
        raise AssertionError("a short series was accepted")


def test_full_history_prefix_contains_every_earlier_observation():
    series = np.arange(200, dtype=np.float64)
    recent, patches, within, valid, _ages = _pack_history(series, np.array([180]), 48, max_patches=4)
    pieces = []
    for slot in range(patches.shape[1]):
        if valid[0, slot] <= 0:
            continue
        pieces.append(patches[0, slot, within[0, slot] > 0])
    pieces.append(recent[0])
    flat = np.concatenate(pieces)
    assert np.array_equal(flat, series[:180])
    assert int(valid[0].sum()) == 3
    assert int(within.sum()) == 132


def test_full_history_loss_uses_the_distant_past():
    net = _init_history(jax.random.PRNGKey(0), 16, 4)
    origin = np.array([80])
    left = np.zeros(80, dtype=np.float32)
    right = np.zeros(80, dtype=np.float32)
    left[:8] = 3.0
    packed_left = [jnp.asarray(part[0]) for part in _pack_history(left, origin, 16, max_patches=5)]
    packed_right = [jnp.asarray(part[0]) for part in _pack_history(right, origin, 16, max_patches=5)]
    pred_left = np.asarray(net(*packed_left))
    pred_right = np.asarray(net(*packed_right))
    assert np.max(np.abs(pred_left - pred_right)) > 1e-5
    series = np.linspace(-1.0, 1.0, 80, dtype=np.float32)
    origins = np.array([40, 56, 64, 70])
    packed = [jnp.asarray(part) for part in _pack_history(series, origins, 16, max_patches=4)]
    target = jnp.asarray(np.random.default_rng(1).normal(size=(4, 4)), dtype=jnp.float32)
    loss, grads = eqx.filter_value_and_grad(_history_mse)(net, *packed, target)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert float(jnp.sum(jnp.abs(grads.patch_weight))) > 0.0
    assert float(jnp.sum(jnp.abs(grads.position))) > 0.0
    pred = _history_batch(net, *packed)
    assert abs(float(loss) - float(jnp.mean(jnp.square(pred - target)))) < 1e-5


def test_full_history_uses_the_same_origins_as_the_window():
    series = np.arange(400, dtype=np.float64)
    origins, inputs, targets = _forecast_windows(series, 48, 12)
    assert np.array_equal(origins, _history_origins(len(series), 48, 12))
    val_start = int(len(series) * 0.9)
    train = origins + 12 <= val_start
    origin = int(origins[train][-1])
    recent, patches, within, valid, _ages = _pack_history(series, np.array([origin]), 48, max_patches=8)
    pieces = [patches[0, slot, within[0, slot] > 0] for slot in range(patches.shape[1]) if valid[0, slot] > 0]
    pieces.append(recent[0])
    flat = np.concatenate(pieces)
    assert np.array_equal(flat, series[:origin])
    assert np.array_equal(inputs[train][-1], series[origin - 48 : origin])
    assert np.array_equal(targets[train][-1], series[origin : origin + 12])
    assert origin + 12 <= val_start


def test_full_history_learns_a_sine_and_rejects_a_short_series():
    time = np.arange(900, dtype=np.float64)
    series = np.sin(2.0 * np.pi * time / 24.0)
    cut = int(len(series) * 0.8)
    model = ASRHistoryForecaster(random_state=0, max_epochs=20, patience=6).fit(series[:cut])
    origins = _history_origins(len(series), model.lookback, model.horizon)
    test = origins >= cut
    pred = model.predict(series, origins[test])
    targets = _horizon_targets(series, origins[test], model.horizon)
    mae = float(np.mean(np.abs(pred - targets)))
    mean_mae = float(np.mean(np.abs(targets - series[:cut].mean())))
    assert mae < 0.25
    assert mae < mean_mae
    assert model.lookback == FORECAST_LOOKBACK
    try:
        ASRHistoryForecaster(max_epochs=1, patience=1).fit(np.arange(20, dtype=np.float64))
    except AshlarError:
        pass
    else:
        raise AssertionError("a short series was accepted")


def test_forecasters_save_and_load_without_pickle(tmp_path):
    series = np.sin(np.arange(260) / 5.0) + 0.1 * np.random.default_rng(0).normal(size=260)
    for cls in (ASRForecaster, ASRHistoryForecaster):
        for head in ("mse", "mixture"):
            model = cls(lookback=16, horizon=4, max_epochs=2, patience=1, head=head).fit(series)
            directory = tmp_path / f"{cls.__name__}-{head}"
            restored = cls.load(model.save(directory))
            assert sorted(path.name for path in directory.iterdir()) == ["metadata.json", "model.eqx"]
            assert restored.best_epoch_ == model.best_epoch_ and restored.n_obs_ == model.n_obs_
            assert np.allclose(model.predict(series[-40:]), restored.predict(series[-40:]))
            if head == "mixture":
                future = series[-4:]
                assert np.allclose(model.log_prob(series[-44:-4], future), restored.log_prob(series[-44:-4], future))
    with pytest.raises(AshlarError):
        ASRHistoryForecaster.load(tmp_path / "ASRForecaster-mse")


def test_forecaster_save_requires_fit(tmp_path):
    with pytest.raises(AshlarError):
        ASRForecaster().save(tmp_path / "unfitted")
