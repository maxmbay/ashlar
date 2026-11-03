import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ashlar.distributional import (
    FLOW_BOUND,
    HISTOGRAM_BINS,
    HISTOGRAM_LOCATION_WEIGHT,
    QUANTILE_LEVELS,
    decode,
    distributional_loss,
    flow_forward,
    flow_inverse,
    flow_log_density,
    flow_output_count,
    flow_parameter_count,
    histogram_bin,
    histogram_edges,
    log_density,
    output_width,
    point_values,
    sample_values,
)
from ashlar.errors import AshlarError, ShapeError
from ashlar.heads import DistributionHead
from ashlar.forecast import (
    ASRHistoryForecaster,
    ASRForecaster,
    _forecast_batch,
    _forecast_objective,
    _forecast_objective_jit,
    _forecast_windows,
    _history_objective,
    _history_objective_jit,
    _history_origins,
    _horizon_targets,
    _init_forecast,
    _init_history,
    _mse,
    _pack_history,
    _patches_needed,
    _scale,
)
from ashlar.predict import normal_log_prob

HEADS = ("quantile", "gaussian", "mixture", "flow", "histogram")


def _windows(batch: int, lookback: int, horizon: int, seed: int):
    rng = np.random.default_rng(seed)
    windows = jnp.asarray(rng.normal(size=(batch, lookback)), dtype=jnp.float32)
    target = jnp.asarray(rng.normal(size=(batch, horizon)) * 0.5, dtype=jnp.float32)
    return windows, target


def _sine(n: int, seed: int = 0) -> np.ndarray:
    time = np.arange(n, dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, 0.05, size=n)
    return np.sin(2.0 * np.pi * time / 24.0) + noise


def _window_loss(net, series, mean: float, std: float) -> float:
    _origins, inputs, targets = _forecast_windows(series, net.lookback, net.horizon)
    windows = jnp.asarray(_scale(inputs, mean, std))
    target = jnp.asarray(_scale(targets, mean, std))
    return float(_forecast_objective_jit(net, windows, target))


def _history_loss(net, series, mean: float, std: float) -> float:
    origins = _history_origins(len(series), net.lookback, net.horizon)
    scaled = _scale(series, mean, std)
    capacity = _patches_needed(origins, net.lookback)
    total = 0.0
    count = 0
    for start in range(0, len(origins), 64):
        chunk = origins[start : start + 64]
        packed = [jnp.asarray(part) for part in _pack_history(scaled, chunk, net.lookback, capacity)]
        target = jnp.asarray(_horizon_targets(scaled, chunk, net.horizon))
        value = float(_history_objective_jit(net, *packed, target))
        total += value * int(len(chunk))
        count += int(len(chunk))
    return total / count


@pytest.mark.slow
def test_default_forecast_head_is_squared_error():
    assert ASRForecaster().head == "mse"
    assert ASRHistoryForecaster().head == "mse"
    assert QUANTILE_LEVELS == (0.1, 0.5, 0.9)
    assert HISTOGRAM_LOCATION_WEIGHT < 1.0
    try:
        ASRForecaster(head="student")
    except AshlarError:
        pass
    else:
        raise AssertionError("an unknown forecast head was accepted")
    net = _init_forecast(jax.random.PRNGKey(0), 12, 2)
    windows, target = _windows(4, 12, 2, seed=0)
    squared = _mse(net, windows, target)
    objective = _forecast_objective(net, windows, target)
    assert abs(float(squared - objective)) < 1e-6


def test_pinball_and_gaussian_match_their_formulas():
    quantiles = jnp.zeros((1, 1, 3), dtype=jnp.float32)
    under = distributional_loss("quantile", quantiles, jnp.ones((1, 1), dtype=jnp.float32))
    assert abs(float(under) - float(np.mean(QUANTILE_LEVELS))) < 1e-6
    over = distributional_loss("quantile", jnp.ones((1, 1, 3), dtype=jnp.float32), jnp.zeros((1, 1), dtype=jnp.float32))
    assert abs(float(over) - float(np.mean([1.0 - level for level in QUANTILE_LEVELS]))) < 1e-6
    loc = jnp.zeros((2, 3), dtype=jnp.float32)
    scale = jnp.ones((2, 3), dtype=jnp.float32)
    target = jnp.full((2, 3), 0.4, dtype=jnp.float32)
    loss = distributional_loss("gaussian", (loc, scale), target)
    manual = jnp.mean(-normal_log_prob(target, loc, scale))
    assert abs(float(loss - manual)) < 1e-6


@pytest.mark.slow
def test_spline_identity_roundtrip_and_density():
    width = flow_parameter_count()
    raw = jnp.zeros((width,), dtype=jnp.float32)
    grid = jnp.linspace(-2.5, 2.5, 7, dtype=jnp.float32)
    forward = jax.vmap(lambda value: flow_forward(raw, value))(grid)
    backward, logdet = jax.vmap(lambda value: flow_inverse(raw, value))(grid)
    assert np.max(np.abs(np.asarray(forward - grid))) < 1e-5
    assert np.max(np.abs(np.asarray(backward - grid))) < 1e-5
    assert np.max(np.abs(np.asarray(logdet))) < 1e-4
    base = jnp.asarray(0.4, dtype=jnp.float32)
    drawn = flow_forward(raw, base)
    recovered, _logdet = flow_inverse(raw, drawn)
    assert abs(float(recovered - base)) < 1e-5
    outside = jnp.asarray(FLOW_BOUND + 1.0, dtype=jnp.float32)
    tail, tail_logdet = flow_inverse(raw, outside)
    assert abs(float(tail - outside)) < 1e-6
    assert abs(float(tail_logdet)) < 1e-6
    warped = jax.random.normal(jax.random.PRNGKey(2), (width,), dtype=jnp.float32) * jnp.asarray(0.4, dtype=jnp.float32)
    start = jnp.asarray(0.3, dtype=jnp.float32)
    image = flow_forward(warped, start)
    inverse, density_logdet = flow_inverse(warped, image)
    slope = jax.grad(lambda value: flow_forward(warped, value))(start)
    assert abs(float(inverse - start)) < 1e-4
    assert abs(float(density_logdet + jnp.log(jnp.abs(slope)))) < 1e-4
    density = flow_log_density(warped, image)
    grad = jax.grad(lambda parameters: flow_log_density(parameters, image))(warped)
    assert np.isfinite(float(density))
    assert np.isfinite(np.asarray(grad)).all()
    assert float(jnp.sum(jnp.abs(grad))) > 0.0
    edges = jnp.asarray([-5.0, -4.0, 0.0, 4.0, 6.0], dtype=jnp.float32)
    assert np.array_equal(np.asarray(histogram_bin(edges)), np.asarray([0, 0, 8, 15, 15]))


@pytest.mark.slow
def test_flow_shift_and_scale_give_each_row_its_own_spread_and_tail():
    from scipy.stats import norm

    count = flow_parameter_count()
    raw = jnp.zeros((3, flow_output_count()), dtype=jnp.float32)
    # An identity spline: each row is exactly a normal at its own shift and scale.
    raw = raw.at[:, count].set(jnp.asarray([0.0, 2.0, -1.0]))
    raw = raw.at[:, count + 1].set(jnp.asarray([0.0, 1.5, -2.0]))
    scale = np.log1p(np.exp(np.asarray(raw[:, count + 1]) + np.log(np.expm1(1.0 - 1e-3)))) + 1e-3
    assert np.isclose(scale[0], 1.0, atol=1e-6)
    value = jnp.asarray([0.5, 4.0, -1.2], dtype=jnp.float32)
    expected = norm.logpdf(np.asarray(value), loc=[0.0, 2.0, -1.0], scale=scale)
    assert np.allclose(flow_log_density(raw, value), expected, atol=1e-4)
    base = jnp.asarray([0.3, -0.7, 1.1], dtype=jnp.float32)
    drawn = flow_forward(raw, base)
    assert np.allclose(drawn, np.asarray([0.0, 2.0, -1.0]) + scale * np.asarray(base), atol=1e-5)
    recovered, logdet = flow_inverse(raw, drawn)
    assert np.allclose(recovered, base, atol=1e-5)
    assert np.allclose(logdet, -np.log(scale), atol=1e-5)
    # Past the spline's range the tail is the row's own normal tail, not a shared one.
    from ashlar.distributions import Flow

    dist = Flow(raw)
    threshold = jnp.asarray([3.0, 3.0, 3.0], dtype=jnp.float32)
    tail = 1.0 - np.asarray(dist.cdf(threshold))
    assert np.allclose(tail, norm.sf(3.0, loc=[0.0, 2.0, -1.0], scale=scale), atol=1e-5)
    assert np.allclose(dist.mean(), [0.0, 2.0, -1.0], atol=1e-3)
    assert np.allclose(dist.variance(), np.square(scale), rtol=1e-3)


def test_flow_reads_a_bare_spline_as_shift_zero_scale_one():
    count = flow_parameter_count()
    spline = jax.random.normal(jax.random.PRNGKey(3), (4, count), dtype=jnp.float32) * 0.5
    wide = jnp.concatenate([spline, jnp.zeros((4, 2), dtype=jnp.float32)], axis=-1)
    value = jnp.asarray([-4.0, -0.5, 0.7, 3.5], dtype=jnp.float32)
    assert np.allclose(flow_log_density(spline, value), flow_log_density(wide, value), atol=1e-5)
    base = jnp.asarray([-3.5, -0.2, 0.4, 2.0], dtype=jnp.float32)
    assert np.allclose(flow_forward(spline, base), flow_forward(wide, base), atol=1e-5)
    # A flow projection saved before the shift and scale existed still decodes.
    legacy = decode("flow", spline.reshape(2, 2 * count), 2)
    assert legacy.shape == (2, 2, count)
    with pytest.raises(ShapeError):
        decode("flow", jnp.zeros((2, 2 * count + 1)), 2)


@pytest.mark.slow
@pytest.mark.parametrize("head", HEADS)
def test_distributional_loss_is_finite_and_reaches_the_head(head):
    horizon = 3
    net = _init_forecast(jax.random.PRNGKey(0), 12, horizon, head=head)
    windows, target = _windows(6, 12, horizon, seed=1)
    loss, grads = eqx.filter_value_and_grad(_forecast_objective)(net, windows, target)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    assert np.isfinite(float(loss))
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert float(jnp.sum(jnp.abs(grads.out_weight))) > 0.0
    assert float(jnp.sum(jnp.abs(grads.out_bias))) > 0.0
    decoded = decode(head, _forecast_batch(net, windows), horizon)
    _assert_parameters(head, decoded, target)


@pytest.mark.slow
@pytest.mark.parametrize("head", HEADS)
def test_full_history_loss_reaches_the_distributional_head(head):
    horizon = 2
    net = _init_history(jax.random.PRNGKey(0), 12, horizon, head=head)
    series = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    origins = np.asarray([32, 44, 52, 58])
    packed = [jnp.asarray(part) for part in _pack_history(series, origins, 12, max_patches=4)]
    target = jnp.asarray(np.random.default_rng(1).normal(size=(4, horizon)) * 0.4, dtype=jnp.float32)
    loss, grads = eqx.filter_value_and_grad(_history_objective)(net, *packed, target)
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    assert np.isfinite(float(loss))
    assert all(np.isfinite(leaf).all() for leaf in leaves)
    assert float(jnp.sum(jnp.abs(grads.out_weight))) > 0.0
    assert float(jnp.sum(jnp.abs(grads.out_bias))) > 0.0
    assert float(jnp.sum(jnp.abs(grads.patch_weight))) > 0.0


@pytest.mark.slow
@pytest.mark.parametrize("head", HEADS)
def test_window_fit_lowers_the_distributional_loss(head):
    series = _sine(360)
    fresh = _init_forecast(jax.random.PRNGKey(0), 16, 1, head=head)
    model = ASRForecaster(
        lookback=16, horizon=1, head=head, max_epochs=5, patience=5, batch_size=64, random_state=0
    ).fit(series)
    before = _window_loss(fresh, series, model.mean_, model.std_)
    after = _window_loss(model.model_, series, model.mean_, model.std_)
    assert after < before, f"{head} loss did not fall: {before} -> {after}"
    assert np.isfinite(model.selection_loss_)
    _origins, inputs, targets = _forecast_windows(series, model.lookback, model.horizon)
    pred = model.predict(inputs[:8])
    assert pred.shape == (8, 1)
    assert np.isfinite(pred).all()
    mae = float(np.mean(np.abs(pred - targets[:8])))
    assert np.isfinite(mae)
    single = model.predict(series[-16:])
    assert single.shape == (1,)
    _assert_fitted_api(model, series[-16:], single)


@pytest.mark.slow
@pytest.mark.parametrize("head", HEADS)
def test_full_history_fit_lowers_the_distributional_loss(head):
    series = _sine(220, seed=1)
    fresh = _init_history(jax.random.PRNGKey(0), 12, 1, head=head)
    model = ASRHistoryForecaster(
        lookback=12, horizon=1, head=head, max_epochs=4, patience=4, batch_size=32, random_state=0
    ).fit(series)
    before = _history_loss(fresh, series, model.mean_, model.std_)
    after = _history_loss(model.model_, series, model.mean_, model.std_)
    assert after < before, f"full-history {head} loss did not fall: {before} -> {after}"
    origins = _history_origins(len(series), model.lookback, model.horizon)
    pred = model.predict(series, origins[:4])
    assert pred.shape == (4, 1)
    assert np.isfinite(pred).all()
    mae = float(np.mean(np.abs(pred - _horizon_targets(series, origins[:4], 1))))
    assert np.isfinite(mae)
    origin = np.asarray([len(series)])
    single = model.predict(series, origin)
    assert single.shape == (1, 1)
    assert model.predict(series).shape == (1,)
    _assert_fitted_api(model, series, single, origins=origin)


def _assert_parameters(head, decoded, target):
    point = point_values(head, decoded)
    assert point.shape == target.shape
    assert np.isfinite(np.asarray(point)).all()
    if head == "quantile":
        assert bool(jnp.all(decoded[..., 0] <= decoded[..., 1]))
        assert bool(jnp.all(decoded[..., 1] <= decoded[..., 2]))
    elif head == "gaussian":
        _loc, scale = decoded
        assert bool(jnp.all(scale > 0))
    elif head == "mixture":
        _loc, scale, weight = decoded
        assert bool(jnp.all(scale > 0))
        assert np.allclose(np.asarray(jnp.sum(weight, axis=-1)), 1.0, atol=1e-5)
    elif head == "histogram":
        probabilities = jax.nn.softmax(decoded, axis=-1)
        assert probabilities.shape[-1] == HISTOGRAM_BINS
        assert np.allclose(np.asarray(jnp.sum(probabilities, axis=-1)), 1.0, atol=1e-5)
    elif head == "flow":
        assert decoded.shape[-1] == flow_output_count()


def _assert_fitted_api(model, history, single, origins=None):
    if model.head == "quantile":
        if origins is None:
            quantiles = model.quantiles(history)
        else:
            quantiles = model.quantiles(history, origins)
        assert quantiles.shape == single.shape + (3,)
        assert np.all(quantiles[..., 0] <= quantiles[..., 1])
        assert np.all(quantiles[..., 1] <= quantiles[..., 2])
        assert np.allclose(single, quantiles[..., 1])
        return
    if model.head == "histogram":
        if origins is None:
            probabilities, centers = model.bin_probabilities(history)
        else:
            probabilities, centers = model.bin_probabilities(history, origins)
        assert probabilities.shape == single.shape + (HISTOGRAM_BINS,)
        assert centers.shape == (HISTOGRAM_BINS,)
        assert np.allclose(probabilities.sum(axis=-1), 1.0, atol=1e-5)
        return
    if origins is None:
        density = model.log_prob(history, single)
        drawn = model.sample(history, key=0, n=2)
    else:
        density = model.log_prob(history, single, origins)
        drawn = model.sample(history, key=0, n=2, origins=origins)
    assert density.shape == single.shape
    assert np.isfinite(density).all()
    assert drawn.shape == (2,) + single.shape
    assert np.isfinite(drawn).all()
    if model.head == "gaussian":
        if origins is None:
            loc, scale = model.gaussian(history)
        else:
            loc, scale = model.gaussian(history, origins)
        assert loc.shape == single.shape and scale.shape == single.shape
        assert np.allclose(loc, single)
        assert np.all(scale > 0.0)


@pytest.mark.slow
def test_histogram_bin_count_is_configurable():
    """A wider histogram head keeps every density law, at finer resolution."""
    bins = 48
    assert output_width("histogram", 1, bins) == bins
    edges = np.asarray(histogram_edges(jnp.float32, bins))
    assert edges.shape == (bins + 1,)
    assert np.isclose(edges[0], -4.0) and np.isclose(edges[-1], 4.0)

    rng = np.random.default_rng(0)
    flat = jnp.asarray(rng.normal(size=(5, bins)), dtype=jnp.float32)
    decoded = decode("histogram", flat, 1)
    assert decoded.shape == (5, 1, bins)

    target = jnp.asarray(rng.normal(size=(5, 1)) * 0.5, dtype=jnp.float32)
    assert np.isfinite(float(distributional_loss("histogram", decoded, target)))
    assert np.isfinite(np.asarray(log_density("histogram", decoded, target))).all()
    draws = sample_values("histogram", decoded, jax.random.PRNGKey(0), 32)
    assert draws.shape == (32, 5, 1)
    assert np.all(np.abs(np.asarray(draws)) <= 4.0)

    # Each bin is narrower, so the index of a value moves with the count.
    value = jnp.asarray([3.9], dtype=jnp.float32)
    assert int(histogram_bin(value, bins)[0]) == bins - 1
    assert int(histogram_bin(value, HISTOGRAM_BINS)[0]) == HISTOGRAM_BINS - 1

    with pytest.raises(AshlarError):
        output_width("histogram", 1, 1)


def test_distribution_head_rejects_bins_for_other_kinds():
    key = jax.random.PRNGKey(0)
    head = DistributionHead(8, 2, n_events=1, n_classes=1, key=key, kind="histogram", bins=32)
    hidden = jnp.zeros((3, 2, 8), dtype=jnp.float32)
    assert head(hidden, key=None).shape[-1] == 32
    with pytest.raises(ShapeError):
        DistributionHead(8, 2, n_events=1, n_classes=1, key=key, kind="gaussian", bins=32)
