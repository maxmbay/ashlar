"""Direct multi-step forecasting with a small transformer.

``ASRForecaster`` reads one univariate window of ``FORECAST_LOOKBACK``
steps. Learned positions keep that order, which the feature-set transformer
deliberately does not. The last token emits the next ``horizon`` values. A
default XGBoost forecast on the same windows uses those lags as flat features.

``ASRHistoryForecaster`` reads the whole prefix before the forecast origin.
The last ``FORECAST_LOOKBACK`` observations are one token each. Every older
observation sits in exactly one non-overlapping patch of that same length, and
each patch is a linear map to the model width. A short last patch is
zero-filled and masked. Positions are sinusoidal ages counted back from the
origin, plus a learned offset on the recent tokens. Recent tokens attend to
each other and to every patch. The target horizon is not part of the prefix.

``head`` selects the training objective. ``mse`` is squared error and remains
the default. ``quantile``, ``gaussian``, ``mixture``, ``flow``, and
``histogram`` train a pinball loss, a heteroscedastic normal, a five-component
normal mixture, a rational-quadratic spline flow, and a fixed-bin histogram.
``predict`` is still one point on the original scale, so absolute error stays
available: the quantile median, the gaussian mean, the mixture mean, the flow
median, or the histogram mean.

The residual regression default does not use this module.
"""

from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module, field, nn

from ashlar._compat import renamed
from ashlar._version import __version__
from ashlar.distributional import (
    FLOW_LAYOUT,
    bare_spline_width,
    check_output_head,
    decode,
    distributional_loss,
    histogram_centers,
    initial_bias,
    initial_weight_scale,
    log_density,
    output_width,
    point_values,
    sample_values,
)
from ashlar.distributions import AffineTransformed, head_distribution
from ashlar.errors import AshlarError, ShapeError
from ashlar.io import load_leaves, save_leaves

FORECAST_LOOKBACK = 48
FORECAST_HORIZON = 12
FORECAST_D_MODEL = 32
FORECAST_HEADS = 4
FORECAST_LAYERS = 2
FORECAST_LEARNING_RATE = 1e-3
FORECAST_BATCH_SIZE = 128
FORECAST_MAX_EPOCHS = 30
FORECAST_PATIENCE = 8
FORECAST_VAL_FRACTION = 0.1
FORECAST_TEST_FRACTION = 0.2
_CHECKPOINT_AVERAGE = 5
_MLP_FACTOR = 2


def _layernorm(values: jax.Array, scale: jax.Array, bias: jax.Array) -> jax.Array:
    mean = jnp.mean(values, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(values - mean), axis=-1, keepdims=True)
    normed = (values - mean) * jax.lax.rsqrt(variance + jnp.asarray(1e-5, dtype=values.dtype))
    return normed * scale + bias


class _TimeBlock(Module):
    """Pre-norm attention over a lookback, then a SiLU feed-forward layer."""

    attention: nn.MultiheadAttention
    norm1_scale: jax.Array
    norm1_bias: jax.Array
    mlp_up: jax.Array
    mlp_up_bias: jax.Array
    mlp_down: jax.Array
    mlp_down_bias: jax.Array
    norm2_scale: jax.Array
    norm2_bias: jax.Array

    def __call__(self, tokens: jax.Array) -> jax.Array:
        normed = _layernorm(tokens, self.norm1_scale, self.norm1_bias)
        attended = self.attention(normed, normed, normed, key=None)
        hidden = tokens + attended
        normed = _layernorm(hidden, self.norm2_scale, self.norm2_bias)
        mid = jax.nn.silu(normed @ self.mlp_up.T + self.mlp_up_bias)
        return hidden + mid @ self.mlp_down.T + self.mlp_down_bias


class ForecastNet(Module):
    """Encoder over one lookback window. The last token emits the horizon."""

    in_weight: jax.Array
    in_bias: jax.Array
    position: jax.Array
    blocks: tuple[_TimeBlock, ...]
    out_weight: jax.Array
    out_bias: jax.Array
    lookback: int = field(static=True)
    horizon: int = field(static=True)
    head: str = field(static=True, default="mse")

    def __call__(self, window: jax.Array) -> jax.Array:
        if window.ndim != 1 or window.shape[0] != self.lookback:
            raise ShapeError(f"forecast window must have shape ({self.lookback},); received {window.shape}.")
        tokens = window[:, None] * self.in_weight + self.in_bias + self.position
        for block in self.blocks:
            tokens = block(tokens)
        return tokens[-1] @ self.out_weight + self.out_bias


def _output_projection(key, d_model: int, head: str, horizon: int):
    check_output_head(head)
    width = output_width(head, horizon)
    if head == "mse":
        scale = np.float32(np.sqrt(1.0 / d_model))
    else:
        scale = np.float32(initial_weight_scale(head, d_model))
    weight = jax.random.normal(key, (d_model, width), dtype=jnp.float32) * scale
    return weight, initial_bias(head, horizon)


def _init_forecast(key, lookback: int, horizon: int, head: str = "mse") -> ForecastNet:
    check_output_head(head)
    if lookback < 2 or horizon < 1:
        raise AshlarError("lookback must be at least 2 and horizon at least 1.")
    if FORECAST_D_MODEL % FORECAST_HEADS != 0:
        raise AshlarError("forecast width must be divisible by the head count.")
    keys = jax.random.split(key, 4 + FORECAST_LAYERS)
    d_model = FORECAST_D_MODEL
    hidden = d_model * _MLP_FACTOR
    blocks = []
    for index in range(FORECAST_LAYERS):
        attn_key, up_key, down_key = jax.random.split(keys[4 + index], 3)
        blocks.append(
            _TimeBlock(
                attention=nn.MultiheadAttention(
                    FORECAST_HEADS, d_model, dropout_p=0.0, inference=True, key=attn_key
                ),
                norm1_scale=jnp.ones((d_model,), dtype=jnp.float32),
                norm1_bias=jnp.zeros((d_model,), dtype=jnp.float32),
                mlp_up=jax.random.normal(up_key, (hidden, d_model), dtype=jnp.float32) * np.float32(np.sqrt(2.0 / d_model)),
                mlp_up_bias=jnp.zeros((hidden,), dtype=jnp.float32),
                mlp_down=jax.random.normal(down_key, (d_model, hidden), dtype=jnp.float32) * np.float32(np.sqrt(2.0 / hidden)),
                mlp_down_bias=jnp.zeros((d_model,), dtype=jnp.float32),
                norm2_scale=jnp.ones((d_model,), dtype=jnp.float32),
                norm2_bias=jnp.zeros((d_model,), dtype=jnp.float32),
            )
        )
    out_weight, out_bias = _output_projection(keys[2], d_model, head, horizon)
    return ForecastNet(
        in_weight=jax.random.normal(keys[0], (d_model,), dtype=jnp.float32) * np.float32(0.5),
        in_bias=jnp.zeros((d_model,), dtype=jnp.float32),
        position=jax.random.normal(keys[1], (lookback, d_model), dtype=jnp.float32) * np.float32(0.02),
        blocks=tuple(blocks),
        out_weight=out_weight,
        out_bias=out_bias,
        lookback=int(lookback),
        horizon=int(horizon),
        head=head,
    )


def _forecast_batch(model: ForecastNet, windows: jax.Array) -> jax.Array:
    return eqx.filter_vmap(lambda net, window: net(window), in_axes=(None, 0))(model, windows)


def _mse(model: ForecastNet, windows: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(_forecast_batch(model, windows) - target))


def _forecast_objective(model: ForecastNet, windows: jax.Array, target: jax.Array) -> jax.Array:
    if model.head == "mse":
        return _mse(model, windows, target)
    flat = _forecast_batch(model, windows)
    return distributional_loss(model.head, decode(model.head, flat, model.horizon), target)


def _make_step(learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def step(model, opt_state, windows, target):
        loss, grads = eqx.filter_value_and_grad(_forecast_objective)(model, windows, target)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return optimizer, step


@eqx.filter_jit
def _apply(model: ForecastNet, windows: jax.Array) -> jax.Array:
    return _forecast_batch(model, windows)


@eqx.filter_jit
def _apply_point(model: ForecastNet, windows: jax.Array) -> jax.Array:
    flat = _forecast_batch(model, windows)
    return point_values(model.head, decode(model.head, flat, model.horizon))


@eqx.filter_jit
def _apply_gaussian(model: ForecastNet, windows: jax.Array):
    flat = _forecast_batch(model, windows)
    return decode("gaussian", flat, model.horizon)


@eqx.filter_jit
def _apply_log_prob(model: ForecastNet, windows: jax.Array, value: jax.Array) -> jax.Array:
    flat = _forecast_batch(model, windows)
    return log_density(model.head, decode(model.head, flat, model.horizon), value)


@eqx.filter_jit
def _apply_quantiles(model: ForecastNet, windows: jax.Array) -> jax.Array:
    flat = _forecast_batch(model, windows)
    return decode("quantile", flat, model.horizon)


@eqx.filter_jit
def _apply_bins(model: ForecastNet, windows: jax.Array) -> jax.Array:
    flat = _forecast_batch(model, windows)
    return jax.nn.softmax(decode("histogram", flat, model.horizon), axis=-1)


@eqx.filter_jit
def _apply_sample(model: ForecastNet, windows: jax.Array, key, n: int) -> jax.Array:
    flat = _forecast_batch(model, windows)
    return sample_values(model.head, decode(model.head, flat, model.horizon), key, n)


@eqx.filter_jit
def _forecast_objective_jit(model: ForecastNet, windows: jax.Array, target: jax.Array) -> jax.Array:
    return _forecast_objective(model, windows, target)


def _forecast_dist(head: str, flat, horizon: int, mean: float, std: float, single: bool):
    base = head_distribution(head, flat, horizon)
    if horizon == 1:
        base = jax.tree.map(lambda leaf: leaf[:, None, ...], base)
    dist = AffineTransformed(base, mean, std)
    return dist[0] if single else dist


def _as_key(key):
    if isinstance(key, (int, np.integer)):
        return jax.random.PRNGKey(int(key))
    return key


def _observed_horizon(value, single: bool, horizon: int, batch: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if single:
        if array.shape != (horizon,):
            raise AshlarError(f"value must have shape ({horizon},); received {array.shape}.")
        array = array[None, :]
    elif array.shape != (batch, horizon):
        raise AshlarError(f"value must have shape ({batch}, {horizon}); received {array.shape}.")
    if not np.isfinite(array).all():
        raise AshlarError("value must be finite.")
    return np.ascontiguousarray(array)


_DENSITY_HEADS = ("gaussian", "mixture", "flow", "histogram", "student_t")


def _snapshot(model):
    arrays, static = eqx.partition(model, eqx.is_array)
    arrays = jax.tree.map(lambda array: jnp.array(array), arrays)
    return eqx.combine(arrays, static)


def _average(models):
    arrays, static = eqx.partition(models[0], eqx.is_array)
    _leaves, treedef = jax.tree.flatten(arrays)
    collected = [jax.tree.leaves(eqx.partition(model, eqx.is_array)[0]) for model in models]
    averaged = [sum(column) / len(column) for column in zip(*collected, strict=True)]
    return eqx.combine(jax.tree.unflatten(treedef, averaged), static)


def _series(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise AshlarError(f"a forecast series must be one-dimensional; received shape {array.shape}.")
    if len(array) == 0 or not np.isfinite(array).all():
        raise AshlarError("a forecast series must be non-empty and finite.")
    return np.ascontiguousarray(array)


def _moments(values: np.ndarray) -> tuple[float, float]:
    mean = float(values.mean())
    std = float(values.std())
    if std < 1e-6:
        std = 1.0
    return mean, std


def _forecast_windows(series: np.ndarray, lookback: int, horizon: int):
    """Windows whose target is the ``horizon`` steps after the lookback.

    ``origins[i]`` is the first target index. The input is
    ``series[origin - lookback:origin]`` and does not include the target.
    """
    series = _series(series)
    if len(series) < lookback + horizon:
        raise AshlarError(
            f"a series of length {len(series)} cannot fill lookback {lookback} and horizon {horizon}."
        )
    view = np.lib.stride_tricks.sliding_window_view(series, lookback + horizon)
    origins = np.arange(lookback, lookback + len(view))
    return origins, np.ascontiguousarray(view[:, :lookback]), np.ascontiguousarray(view[:, lookback:])


def _scale(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return ((values - mean) / std).astype(np.float32)


class ASRForecaster:
    """Univariate direct multi-step forecast. History in, horizon out.

    Fit receives the series observed so far. The last ``val_fraction`` of that
    series is a chronological holdout used only to choose the epoch. The fit
    then starts again from the same initialization and trains that many epochs
    on every window in the series it was given. ``predict`` reads a lookback
    and returns one point on the original scale. Absolute error of that point
    is the summary. ``head="mse"`` trains squared error. The other heads train
    a pinball loss or their own likelihood, and the point is the median, the
    gaussian mean, the mixture mean, the flow median, or the histogram mean.
    """

    def __init__(
        self,
        lookback: int = FORECAST_LOOKBACK,
        horizon: int = FORECAST_HORIZON,
        random_state: int = 0,
        verbose: bool = False,
        max_epochs: int = FORECAST_MAX_EPOCHS,
        patience: int = FORECAST_PATIENCE,
        batch_size: int = FORECAST_BATCH_SIZE,
        val_fraction: float = FORECAST_VAL_FRACTION,
        head: str = "mse",
    ):
        if lookback < 2 or horizon < 1 or max_epochs < 1 or patience < 1 or batch_size < 1:
            raise AshlarError("lookback, horizon, epochs, patience, and batch size must be positive.")
        if not 0.0 < float(val_fraction) < 0.5:
            raise AshlarError("validation fraction must be between 0 and 0.5.")
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.head = check_output_head(head)

    def fit(self, y):
        series = _series(y)
        origins, inputs, targets = _forecast_windows(series, self.lookback, self.horizon)
        val_start = int(len(series) * (1.0 - self.val_fraction))
        train = origins + self.horizon <= val_start
        val = (origins >= val_start) & (origins + self.horizon <= len(series))
        if int(train.sum()) < 2 or int(val.sum()) < 1:
            raise AshlarError("the series is too short for a chronological forecast split.")
        self.n_train_ = int(train.sum())
        self.n_val_ = int(val.sum())
        mean, std = _moments(series[:val_start])
        best_epoch, selection = self._select(inputs, targets, train, val, mean, std)
        self.best_epoch_ = int(best_epoch)
        mean, std = _moments(series)
        model = self._run(_scale(inputs, mean, std), _scale(targets, mean, std))
        self.model_ = model
        self.mean_ = float(mean)
        self.std_ = float(std)
        self.selection_loss_ = float(selection)
        if self.head == "mse":
            self.selection_mse_ = float(selection)
        self.n_obs_ = int(len(series))
        return self

    def predict(self, history) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before predict.")
        windows, single = self._standardized_windows(history)
        apply = _apply if self.head == "mse" else _apply_point
        pieces = []
        for start in range(0, len(windows), 512):
            pieces.append(np.asarray(apply(self.model_, jnp.asarray(windows[start : start + 512]))))
        restored = np.concatenate(pieces).astype(np.float64) * self.std_ + self.mean_
        if single:
            return restored[0]
        return restored

    def predict_dist(self, history):
        """The forecast distribution, in the units of the series.

        An ``ashlar.distributions.Distribution`` with batch shape
        ``(windows, horizon)``, or ``(horizon,)`` for one history: each step
        is its own marginal, and nothing is claimed about how the steps move
        together. The ``mse`` head has no distribution; the quantile head
        answers ``quantile`` at its three levels only.
        """
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before predict_dist.")
        if self.head == "mse":
            raise AshlarError("the mse head predicts a point and has no distribution.")
        windows, single = self._standardized_windows(history)
        flat = jnp.concatenate(
            [_apply(self.model_, jnp.asarray(windows[start : start + 512])) for start in range(0, len(windows), 512)]
        )
        return _forecast_dist(self.head, flat, self.horizon, self.mean_, self.std_, single)

    def gaussian(self, history):
        """Gaussian location and scale on the original scale."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before gaussian.")
        if self.head != "gaussian":
            raise AshlarError("location and scale are available on the gaussian head.")
        windows, single = self._standardized_windows(history)
        locations = []
        scales = []
        for start in range(0, len(windows), 512):
            loc, scale = _apply_gaussian(self.model_, jnp.asarray(windows[start : start + 512]))
            locations.append(np.asarray(loc))
            scales.append(np.asarray(scale))
        loc = np.concatenate(locations).astype(np.float64) * self.std_ + self.mean_
        scale = np.concatenate(scales).astype(np.float64) * self.std_
        if single:
            return loc[0], scale[0]
        return loc, scale

    def log_prob(self, history, value) -> np.ndarray:
        """Per-step log density of ``value`` on the original scale.

        ``value`` has the shape ``predict`` returns. The quantile and
        squared-error heads have no density.
        """
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before log_prob.")
        if self.head not in _DENSITY_HEADS:
            raise AshlarError(f"the {self.head} head has no log density.")
        windows, single = self._standardized_windows(history)
        observed = _observed_horizon(value, single, self.horizon, len(windows))
        standardized = _scale(observed, self.mean_, self.std_)
        pieces = []
        for start in range(0, len(windows), 512):
            stop = start + 512
            pieces.append(
                np.asarray(
                    _apply_log_prob(
                        self.model_,
                        jnp.asarray(windows[start:stop]),
                        jnp.asarray(standardized[start:stop]),
                    )
                )
            )
        logged = np.concatenate(pieces).astype(np.float64) - np.log(self.std_)
        if single:
            return logged[0]
        return logged

    def sample(self, history, key, n: int = 1) -> np.ndarray:
        """Draw ``n`` horizons. Shape ``(n, horizon)`` or ``(n, batch, horizon)``.

        A flow draw is one forward pass from a standard normal through the spline.
        """
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before sample.")
        if self.head not in _DENSITY_HEADS:
            raise AshlarError(f"the {self.head} head has no sampler.")
        count = int(n)
        if count < 1:
            raise AshlarError("the number of samples must be positive.")
        windows, single = self._standardized_windows(history)
        pieces = []
        sample_key = _as_key(key)
        for start in range(0, len(windows), 512):
            sample_key, chunk_key = jax.random.split(sample_key)
            pieces.append(
                np.asarray(
                    _apply_sample(self.model_, jnp.asarray(windows[start : start + 512]), chunk_key, count)
                )
            )
        drawn = np.concatenate(pieces, axis=1).astype(np.float64) * self.std_ + self.mean_
        if single:
            return drawn[:, 0, :]
        return drawn

    def quantiles(self, history) -> np.ndarray:
        """Quantile forecasts at 0.1, 0.5, and 0.9, on the original scale."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before quantiles.")
        if self.head != "quantile":
            raise AshlarError("quantiles are available on the quantile head.")
        windows, single = self._standardized_windows(history)
        pieces = []
        for start in range(0, len(windows), 512):
            pieces.append(np.asarray(_apply_quantiles(self.model_, jnp.asarray(windows[start : start + 512]))))
        restored = np.concatenate(pieces).astype(np.float64) * self.std_ + self.mean_
        if single:
            return restored[0]
        return restored

    def bin_probabilities(self, history):
        """Histogram probabilities and the fixed bin centers on the original scale."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before bin_probabilities.")
        if self.head != "histogram":
            raise AshlarError("bin probabilities are available on the histogram head.")
        windows, single = self._standardized_windows(history)
        pieces = []
        for start in range(0, len(windows), 512):
            pieces.append(np.asarray(_apply_bins(self.model_, jnp.asarray(windows[start : start + 512]))))
        probabilities = np.concatenate(pieces).astype(np.float64)
        centers = np.asarray(histogram_centers(), dtype=np.float64) * self.std_ + self.mean_
        if single:
            return probabilities[0], centers
        return probabilities, centers

    def save(self, path: str | Path) -> str:
        """Write ``metadata.json`` and the network weights to the directory ``path``."""
        return _save_forecaster(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ASRForecaster":
        return _load_forecaster(cls, path, _init_forecast)

    def _standardized_windows(self, history):
        array = np.asarray(history, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        if array.ndim != 2:
            raise AshlarError(f"history must have shape (lookback,) or (batch, lookback); received {array.shape}.")
        if array.shape[1] < self.lookback:
            raise AshlarError(f"history has {array.shape[1]} steps; lookback is {self.lookback}.")
        if not np.isfinite(array).all():
            raise AshlarError("history must be finite.")
        windows = _scale(array[:, -self.lookback :], self.mean_, self.std_)
        return windows, single

    def _select(self, inputs, targets, train, val, mean: float, std: float):
        scaled_x = _scale(inputs, mean, std)
        scaled_y = _scale(targets, mean, std)
        self._reset()

        def init():
            return _init_forecast(jax.random.PRNGKey(self.random_state), self.lookback, self.horizon, self.head)

        def step_epoch(state, epoch):
            return self._epoch(state, scaled_x[train], scaled_y[train], epoch)

        def score(state):
            if self.head == "mse":
                return self._mse_numpy(state, scaled_x[val], scaled_y[val])
            return self._objective_numpy(state, scaled_x[val], scaled_y[val])

        metric = "val_mse" if self.head == "mse" else "val_loss"
        return _train_until(init, step_epoch, score, self.max_epochs, self.patience, self.verbose, metric=metric)

    def _run(self, scaled_x, scaled_y):
        self._reset()

        def init():
            return _init_forecast(jax.random.PRNGKey(self.random_state), self.lookback, self.horizon, self.head)

        def step_epoch(state, epoch):
            return self._epoch(state, scaled_x, scaled_y, epoch)

        return _train_for(init, step_epoch, self.best_epoch_)

    def _epoch(self, model, inputs, targets, epoch: int):
        if self._opt_state is None:
            optimizer, step = _make_step(FORECAST_LEARNING_RATE)
            self._opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            self._step = step
        generator = np.random.default_rng(self.random_state + epoch)
        order = generator.permutation(len(inputs))
        for start in range(0, len(order), self.batch_size):
            chosen = order[start : start + self.batch_size]
            if len(chosen) < 1:
                continue
            model, self._opt_state, _loss = self._step(
                model,
                self._opt_state,
                jnp.asarray(inputs[chosen]),
                jnp.asarray(targets[chosen]),
            )
        return model

    def _mse_numpy(self, model, inputs, targets) -> float:
        pieces = []
        for start in range(0, len(inputs), 512):
            pieces.append(np.asarray(_apply(model, jnp.asarray(inputs[start : start + 512]))))
        pred = np.concatenate(pieces)
        return float(np.mean(np.square(pred - targets)))

    def _objective_numpy(self, model, inputs, targets) -> float:
        total = 0.0
        count = 0
        for start in range(0, len(inputs), 256):
            x = jnp.asarray(inputs[start : start + 256])
            y = jnp.asarray(targets[start : start + 256])
            value = float(_forecast_objective_jit(model, x, y))
            total += value * int(x.shape[0])
            count += int(x.shape[0])
        return total / count

    def _reset(self) -> None:
        self._opt_state = None
        self._step = None


_FORECAST_PARAMS = (
    "lookback",
    "horizon",
    "random_state",
    "verbose",
    "max_epochs",
    "patience",
    "batch_size",
    "val_fraction",
    "head",
)
_FORECAST_FITTED = (
    "n_train_",
    "n_val_",
    "best_epoch_",
    "mean_",
    "std_",
    "selection_loss_",
    "selection_mse_",
    "n_obs_",
    "max_patches_",
)


def _save_forecaster(forecaster, path) -> str:
    """Save a fitted forecaster as JSON metadata plus Equinox leaves. No pickle."""
    if not hasattr(forecaster, "model_"):
        raise AshlarError("fit the forecaster before save.")
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": __version__,
        "distribution": "ashlar-tabular",
        "import": "ashlar",
        "estimator": type(forecaster).__name__,
        "params": {name: getattr(forecaster, name) for name in _FORECAST_PARAMS},
        "fitted": {name: getattr(forecaster, name) for name in _FORECAST_FITTED if hasattr(forecaster, name)},
        "flow_layout": FLOW_LAYOUT,
    }
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2))
    save_leaves(directory / "model.eqx", forecaster.model_)
    return str(directory)


def _load_forecaster(cls, path, init):
    """Rebuild the network skeleton from the saved shape, then read the weights into it."""
    directory = Path(path)
    payload = json.loads((directory / "metadata.json").read_text())
    if payload.get("estimator") != cls.__name__:
        raise AshlarError(f"{directory} holds a {payload.get('estimator')}; load it with that class.")
    forecaster = cls(**payload["params"])
    for name, value in payload["fitted"].items():
        setattr(forecaster, name, value)
    template = init(jax.random.PRNGKey(0), forecaster.lookback, forecaster.horizon, forecaster.head)
    if forecaster.head == "flow" and payload.get("flow_layout") != FLOW_LAYOUT:
        # Saved before the flow head's per-row shift and scale: the bare spline, read with shift 0 and scale 1.
        width = bare_spline_width(template.out_weight.shape[-1], forecaster.horizon)
        template = eqx.tree_at(
            lambda model: (model.out_weight, model.out_bias), template, (template.out_weight[:, :width], template.out_bias[:width])
        )
    forecaster.model_ = load_leaves(directory / "model.eqx", template)
    return forecaster


def _train_until(
    init, step_epoch, score, max_epochs: int, patience: int, verbose: bool, label: str = "forecast", metric: str = "val_mse"
):
    state = init()
    history = []
    best_epoch = 1
    best_score = float("inf")
    wait = 0
    for epoch in range(1, max_epochs + 1):
        state = step_epoch(state, epoch)
        history.append(_snapshot(state))
        history = history[-_CHECKPOINT_AVERAGE:]
        value = float(score(_average(history)))
        if verbose:
            print(f"  {label} epoch {epoch} {metric} {value:.6f}", flush=True)
        if value < best_score:
            best_score = value
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_epoch, best_score


def _train_for(init, step_epoch, epochs: int):
    state = init()
    history = []
    for epoch in range(1, epochs + 1):
        state = step_epoch(state, epoch)
        history.append(_snapshot(state))
        history = history[-_CHECKPOINT_AVERAGE:]
    return _average(history)


def _sinusoid(ages: jax.Array, d_model: int) -> jax.Array:
    """Sinusoidal position of each token's age, counted back from the origin."""
    half = d_model // 2
    steps = jnp.arange(half, dtype=jnp.float32)
    freq = jnp.exp(-jnp.log(jnp.asarray(10000.0, dtype=jnp.float32)) * steps / jnp.asarray(float(half), dtype=jnp.float32))
    angles = ages.astype(jnp.float32)[:, None] * freq[None, :]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class _HistoryBlock(Module):
    """Self-attention over the recent tokens, then cross-attention into the prefix."""

    self_attention: nn.MultiheadAttention
    cross_attention: nn.MultiheadAttention
    norm1_scale: jax.Array
    norm1_bias: jax.Array
    norm2_scale: jax.Array
    norm2_bias: jax.Array
    mlp_up: jax.Array
    mlp_up_bias: jax.Array
    mlp_down: jax.Array
    mlp_down_bias: jax.Array
    norm3_scale: jax.Array
    norm3_bias: jax.Array

    def __call__(self, tokens: jax.Array, memory: jax.Array, mask: jax.Array) -> jax.Array:
        normed = _layernorm(tokens, self.norm1_scale, self.norm1_bias)
        tokens = tokens + self.self_attention(normed, normed, normed, key=None)
        normed = _layernorm(tokens, self.norm2_scale, self.norm2_bias)
        tokens = tokens + self.cross_attention(normed, memory, memory, mask=mask, key=None)
        normed = _layernorm(tokens, self.norm3_scale, self.norm3_bias)
        mid = jax.nn.silu(normed @ self.mlp_up.T + self.mlp_up_bias)
        return tokens + mid @ self.mlp_down.T + self.mlp_down_bias


class FullHistoryNet(Module):
    """Recent tokens at full resolution, older history as fixed-length patches.

    ``recent`` has shape ``(lookback,)``. ``patches`` has shape
    ``(n_patches, lookback)`` and ``within`` is the matching 0/1 mask of real
    positions inside each patch. ``valid`` marks real patches. ``ages`` is the
    number of steps from the end of each patch back to the origin. A learned
    null token keeps cross-attention defined when the prefix has no older patch.
    """

    in_weight: jax.Array
    in_bias: jax.Array
    position: jax.Array
    patch_weight: jax.Array
    mask_weight: jax.Array
    patch_bias: jax.Array
    null_token: jax.Array
    blocks: tuple[_HistoryBlock, ...]
    out_weight: jax.Array
    out_bias: jax.Array
    lookback: int = field(static=True)
    horizon: int = field(static=True)
    head: str = field(static=True, default="mse")

    def __call__(
        self,
        recent: jax.Array,
        patches: jax.Array,
        within: jax.Array,
        valid: jax.Array,
        ages: jax.Array,
    ) -> jax.Array:
        if recent.ndim != 1 or recent.shape[0] != self.lookback:
            raise ShapeError(f"recent forecast tokens must have shape ({self.lookback},); received {recent.shape}.")
        if patches.ndim != 2 or patches.shape[1] != self.lookback or patches.shape != within.shape:
            raise ShapeError(
                f"history patches must have shape (n_patches, {self.lookback}); received {patches.shape}."
            )
        if valid.shape != (patches.shape[0],) or ages.shape != (patches.shape[0],):
            raise ShapeError("patch validity and ages must have one entry per patch.")
        d_model = self.in_weight.shape[0]
        recent_age = jnp.arange(self.lookback, 0, -1, dtype=jnp.float32)
        tokens = recent[:, None] * self.in_weight + self.in_bias + self.position + _sinusoid(recent_age, d_model)
        memory = patches @ self.patch_weight.T + within @ self.mask_weight.T + self.patch_bias
        memory = memory + _sinusoid(ages, d_model)
        memory = jnp.concatenate([self.null_token[None, :], memory], axis=0)
        keep = jnp.concatenate(
            [jnp.ones((1,), dtype=bool), valid.astype(jnp.float32) > jnp.asarray(0.5, dtype=jnp.float32)]
        )
        mask = jnp.broadcast_to(keep[None, :], (self.lookback, keep.shape[0]))
        for block in self.blocks:
            tokens = block(tokens, memory, mask)
        return tokens[-1] @ self.out_weight + self.out_bias


def _init_history_block(key, d_model: int) -> _HistoryBlock:
    self_key, cross_key, up_key, down_key = jax.random.split(key, 4)
    hidden = d_model * _MLP_FACTOR

    def _ones():
        return jnp.ones((d_model,), dtype=jnp.float32)

    def _zeros():
        return jnp.zeros((d_model,), dtype=jnp.float32)

    return _HistoryBlock(
        self_attention=nn.MultiheadAttention(FORECAST_HEADS, d_model, dropout_p=0.0, inference=True, key=self_key),
        cross_attention=nn.MultiheadAttention(FORECAST_HEADS, d_model, dropout_p=0.0, inference=True, key=cross_key),
        norm1_scale=_ones(),
        norm1_bias=_zeros(),
        norm2_scale=_ones(),
        norm2_bias=_zeros(),
        mlp_up=jax.random.normal(up_key, (hidden, d_model), dtype=jnp.float32) * np.float32(np.sqrt(2.0 / d_model)),
        mlp_up_bias=jnp.zeros((hidden,), dtype=jnp.float32),
        mlp_down=jax.random.normal(down_key, (d_model, hidden), dtype=jnp.float32) * np.float32(np.sqrt(2.0 / hidden)),
        mlp_down_bias=jnp.zeros((d_model,), dtype=jnp.float32),
        norm3_scale=_ones(),
        norm3_bias=_zeros(),
    )


def _init_history(key, lookback: int, horizon: int, head: str = "mse") -> FullHistoryNet:
    check_output_head(head)
    if lookback < 2 or horizon < 1:
        raise AshlarError("lookback must be at least 2 and horizon at least 1.")
    if FORECAST_D_MODEL % FORECAST_HEADS != 0 or FORECAST_D_MODEL % 2 != 0:
        raise AshlarError("forecast width must be divisible by the head count and by 2.")
    keys = jax.random.split(key, 6 + FORECAST_LAYERS)
    d_model = FORECAST_D_MODEL
    blocks = tuple(_init_history_block(keys[6 + index], d_model) for index in range(FORECAST_LAYERS))
    scale = np.float32(np.sqrt(1.0 / lookback))
    out_weight, out_bias = _output_projection(keys[5], d_model, head, horizon)
    return FullHistoryNet(
        in_weight=jax.random.normal(keys[0], (d_model,), dtype=jnp.float32) * np.float32(0.5),
        in_bias=jnp.zeros((d_model,), dtype=jnp.float32),
        position=jax.random.normal(keys[1], (lookback, d_model), dtype=jnp.float32) * np.float32(0.02),
        patch_weight=jax.random.normal(keys[2], (d_model, lookback), dtype=jnp.float32) * scale,
        mask_weight=jax.random.normal(keys[3], (d_model, lookback), dtype=jnp.float32) * np.float32(0.02),
        patch_bias=jnp.zeros((d_model,), dtype=jnp.float32),
        null_token=jax.random.normal(keys[4], (d_model,), dtype=jnp.float32) * np.float32(0.02),
        blocks=blocks,
        out_weight=out_weight,
        out_bias=out_bias,
        lookback=int(lookback),
        horizon=int(horizon),
        head=head,
    )


def _history_batch(model: FullHistoryNet, recent, patches, within, valid, ages) -> jax.Array:
    def one(net, recent_row, patch_row, within_row, valid_row, age_row):
        return net(recent_row, patch_row, within_row, valid_row, age_row)

    return eqx.filter_vmap(one, in_axes=(None, 0, 0, 0, 0, 0))(model, recent, patches, within, valid, ages)


def _history_mse(model: FullHistoryNet, recent, patches, within, valid, ages, target) -> jax.Array:
    return jnp.mean(jnp.square(_history_batch(model, recent, patches, within, valid, ages) - target))


def _history_objective(model: FullHistoryNet, recent, patches, within, valid, ages, target) -> jax.Array:
    if model.head == "mse":
        return _history_mse(model, recent, patches, within, valid, ages, target)
    flat = _history_batch(model, recent, patches, within, valid, ages)
    return distributional_loss(model.head, decode(model.head, flat, model.horizon), target)


def _make_history_step(learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def step(model, opt_state, recent, patches, within, valid, ages, target):
        loss, grads = eqx.filter_value_and_grad(_history_objective)(
            model, recent, patches, within, valid, ages, target
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return optimizer, step


def _history_origins(n: int, lookback: int, horizon: int) -> np.ndarray:
    if n < lookback + horizon:
        raise AshlarError(f"a series of length {n} cannot fill lookback {lookback} and horizon {horizon}.")
    return np.arange(lookback, n - horizon + 1, dtype=np.int64)


def _patches_needed(origins: np.ndarray, lookback: int) -> int:
    older = np.asarray(origins, dtype=np.int64) - int(lookback)
    if older.size == 0 or int(older.max()) <= 0:
        return 1
    return max(1, int((int(older.max()) + lookback - 1) // lookback))


def _patch_grid(series: np.ndarray, lookback: int) -> np.ndarray:
    n_grid = int(np.ceil(len(series) / lookback))
    padded = np.zeros((n_grid * lookback,), dtype=np.float32)
    padded[: len(series)] = np.asarray(series, dtype=np.float32)
    return padded.reshape(n_grid, lookback)


def _pack_history(series: np.ndarray, origins: np.ndarray, lookback: int, max_patches: int, grid=None):
    """Pack each origin's prefix without including the horizon that follows it.

    Full patches tile ``series[:origin - lookback]`` from the start. A leftover
    tail is one zero-filled patch. The recent window is ``series[origin -
    lookback:origin]``, oldest to newest.
    """
    series = np.asarray(series)
    origins = np.asarray(origins, dtype=np.int64).reshape(-1)
    if origins.size == 0:
        raise AshlarError("at least one forecast origin is required.")
    if int(origins.min()) < lookback or int(origins.max()) > len(series):
        raise AshlarError("forecast origins must lie between the lookback and the series length.")
    if max_patches < 1:
        raise AshlarError("the patch capacity must be positive.")
    if int(_patches_needed(origins, lookback)) > max_patches:
        raise AshlarError("the patch capacity is smaller than the longest prefix.")
    if grid is None:
        grid = _patch_grid(series, lookback)
    values = np.asarray(series, dtype=np.float32)
    batch = len(origins)
    older_end = origins - int(lookback)
    n_full = older_end // int(lookback)
    remainder = older_end - n_full * int(lookback)
    slot = np.arange(max_patches, dtype=np.int64)[None, :]
    full_valid = slot < n_full[:, None]
    patches = grid[np.where(full_valid, slot, 0)].astype(np.float32, copy=True)
    patches[~full_valid] = 0.0
    within = np.zeros((batch, max_patches, lookback), dtype=np.float32)
    within[full_valid] = 1.0
    valid = full_valid.astype(np.float32)
    ages = np.where(full_valid, origins[:, None] - (slot + 1) * int(lookback), 0).astype(np.float32)
    has_partial = remainder > 0
    if np.any(has_partial):
        rows = np.flatnonzero(has_partial)
        slots = n_full[rows]
        positions = np.arange(lookback, dtype=np.int64)
        source = n_full[rows, None] * int(lookback) + positions[None, :]
        partial_within = positions[None, :] < remainder[rows, None]
        gathered = values[np.clip(source, 0, len(values) - 1)]
        gathered = np.where(partial_within, gathered, 0.0).astype(np.float32)
        patches[rows, slots] = gathered
        within[rows, slots] = partial_within.astype(np.float32)
        valid[rows, slots] = 1.0
        ages[rows, slots] = np.float32(lookback)
    recent_index = origins[:, None] - int(lookback) + np.arange(lookback, dtype=np.int64)[None, :]
    recent = values[recent_index]
    return recent, patches, within, valid, ages


class ASRHistoryForecaster:
    """Univariate direct multi-step forecast that reads the whole prefix.

    Fit receives the series observed so far. The last ``val_fraction`` of that
    series is a chronological holdout used only to choose the epoch. The fit
    then starts again from the same initialization and trains that many epochs
    on every origin in the series it was given. ``predict`` reads a prefix and
    returns one point on the original scale. Absolute error of that point is
    the summary. ``head="mse"`` trains squared error. The other heads train
    a pinball loss or their own likelihood, and the point is the
    median, the gaussian mean, the mixture mean, the flow median, or the
    histogram mean. Passing ``origins`` forecasts each of those cut points;
    omitting it forecasts the horizon after the last observation.
    """

    def __init__(
        self,
        lookback: int = FORECAST_LOOKBACK,
        horizon: int = FORECAST_HORIZON,
        random_state: int = 0,
        verbose: bool = False,
        max_epochs: int = FORECAST_MAX_EPOCHS,
        patience: int = FORECAST_PATIENCE,
        batch_size: int = FORECAST_BATCH_SIZE,
        val_fraction: float = FORECAST_VAL_FRACTION,
        head: str = "mse",
    ):
        if lookback < 2 or horizon < 1 or max_epochs < 1 or patience < 1 or batch_size < 1:
            raise AshlarError("lookback, horizon, epochs, patience, and batch size must be positive.")
        if not 0.0 < float(val_fraction) < 0.5:
            raise AshlarError("validation fraction must be between 0 and 0.5.")
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.head = check_output_head(head)

    def fit(self, y):
        series = _series(y)
        origins = _history_origins(len(series), self.lookback, self.horizon)
        val_start = int(len(series) * (1.0 - self.val_fraction))
        train = origins + self.horizon <= val_start
        val = (origins >= val_start) & (origins + self.horizon <= len(series))
        if int(train.sum()) < 2 or int(val.sum()) < 1:
            raise AshlarError("the series is too short for a chronological forecast split.")
        self.n_train_ = int(train.sum())
        self.n_val_ = int(val.sum())
        self.max_patches_ = _patches_needed(origins, self.lookback)
        mean, std = _moments(series[:val_start])
        best_epoch, selection = self._select(series, origins, train, val, mean, std)
        self.best_epoch_ = int(best_epoch)
        mean, std = _moments(series)
        model = self._run(_scale(series, mean, std), origins)
        self.model_ = model
        self.mean_ = float(mean)
        self.std_ = float(std)
        self.selection_loss_ = float(selection)
        if self.head == "mse":
            self.selection_mse_ = float(selection)
        self.n_obs_ = int(len(series))
        return self

    def predict(self, history, origins=None) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before predict.")
        scaled, origins, capacity, single = self._prepare_origins(history, origins)
        apply = _history_apply if self.head == "mse" else _history_point
        pieces = []
        for start in range(0, len(origins), 128):
            chunk = origins[start : start + 128]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            pieces.append(np.asarray(apply(self.model_, *[jnp.asarray(part) for part in packed])))
        restored = np.concatenate(pieces).astype(np.float64) * self.std_ + self.mean_
        if single:
            return restored[0]
        return restored

    def predict_dist(self, history, origins=None):
        """The forecast distribution, in the units of the series.

        An ``ashlar.distributions.Distribution`` with batch shape
        ``(windows, horizon)``, or ``(horizon,)`` for one history: each step
        is its own marginal, and nothing is claimed about how the steps move
        together. The ``mse`` head has no distribution; the quantile head
        answers ``quantile`` at its three levels only.
        """
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before predict_dist.")
        if self.head == "mse":
            raise AshlarError("the mse head predicts a point and has no distribution.")
        scaled, origins, capacity, single = self._prepare_origins(history, origins)
        pieces = []
        for start in range(0, len(origins), 128):
            packed = _pack_history(scaled, origins[start : start + 128], self.lookback, capacity)
            pieces.append(_history_apply(self.model_, *[jnp.asarray(part) for part in packed]))
        return _forecast_dist(self.head, jnp.concatenate(pieces), self.horizon, self.mean_, self.std_, single)

    def gaussian(self, history, origins=None):
        """Gaussian location and scale on the original scale."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before gaussian.")
        if self.head != "gaussian":
            raise AshlarError("location and scale are available on the gaussian head.")
        scaled, origins, capacity, single = self._prepare_origins(history, origins)
        locations = []
        scales = []
        for start in range(0, len(origins), 128):
            chunk = origins[start : start + 128]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            loc, scale = _history_gaussian(self.model_, *[jnp.asarray(part) for part in packed])
            locations.append(np.asarray(loc))
            scales.append(np.asarray(scale))
        loc = np.concatenate(locations).astype(np.float64) * self.std_ + self.mean_
        scale = np.concatenate(scales).astype(np.float64) * self.std_
        if single:
            return loc[0], scale[0]
        return loc, scale

    def log_prob(self, history, value, origins=None) -> np.ndarray:
        """Per-step log density of ``value`` on the original scale."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before log_prob.")
        if self.head not in _DENSITY_HEADS:
            raise AshlarError(f"the {self.head} head has no log density.")
        scaled, origins, capacity, single = self._prepare_origins(history, origins)
        observed = _observed_horizon(value, single, self.horizon, len(origins))
        standardized = _scale(observed, self.mean_, self.std_)
        pieces = []
        for start in range(0, len(origins), 128):
            stop = start + 128
            chunk = origins[start:stop]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            pieces.append(
                np.asarray(
                    _history_log_prob(
                        self.model_,
                        *[jnp.asarray(part) for part in packed],
                        jnp.asarray(standardized[start:stop]),
                    )
                )
            )
        logged = np.concatenate(pieces).astype(np.float64) - np.log(self.std_)
        if single:
            return logged[0]
        return logged

    def sample(self, history, key, n: int = 1, origins=None) -> np.ndarray:
        """Draw ``n`` horizons. A flow draw is one spline forward pass."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before sample.")
        if self.head not in _DENSITY_HEADS:
            raise AshlarError(f"the {self.head} head has no sampler.")
        count = int(n)
        if count < 1:
            raise AshlarError("the number of samples must be positive.")
        scaled, origins, capacity, single = self._prepare_origins(history, origins)
        pieces = []
        sample_key = _as_key(key)
        for start in range(0, len(origins), 128):
            sample_key, chunk_key = jax.random.split(sample_key)
            chunk = origins[start : start + 128]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            pieces.append(
                np.asarray(
                    _history_sample(self.model_, *[jnp.asarray(part) for part in packed], chunk_key, count)
                )
            )
        drawn = np.concatenate(pieces, axis=1).astype(np.float64) * self.std_ + self.mean_
        if single:
            return drawn[:, 0, :]
        return drawn

    def quantiles(self, history, origins=None) -> np.ndarray:
        """Quantile forecasts at 0.1, 0.5, and 0.9, on the original scale."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before quantiles.")
        if self.head != "quantile":
            raise AshlarError("quantiles are available on the quantile head.")
        scaled, origins, capacity, single = self._prepare_origins(history, origins)
        pieces = []
        for start in range(0, len(origins), 128):
            chunk = origins[start : start + 128]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            pieces.append(np.asarray(_history_quantiles(self.model_, *[jnp.asarray(part) for part in packed])))
        restored = np.concatenate(pieces).astype(np.float64) * self.std_ + self.mean_
        if single:
            return restored[0]
        return restored

    def bin_probabilities(self, history, origins=None):
        """Histogram probabilities and the fixed bin centers on the original scale."""
        if not hasattr(self, "model_"):
            raise AshlarError("fit the forecaster before bin_probabilities.")
        if self.head != "histogram":
            raise AshlarError("bin probabilities are available on the histogram head.")
        scaled, origins, capacity, single = self._prepare_origins(history, origins)
        pieces = []
        for start in range(0, len(origins), 128):
            chunk = origins[start : start + 128]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            pieces.append(np.asarray(_history_bins(self.model_, *[jnp.asarray(part) for part in packed])))
        probabilities = np.concatenate(pieces).astype(np.float64)
        centers = np.asarray(histogram_centers(), dtype=np.float64) * self.std_ + self.mean_
        if single:
            return probabilities[0], centers
        return probabilities, centers

    def save(self, path: str | Path) -> str:
        """Write ``metadata.json`` and the network weights to the directory ``path``."""
        return _save_forecaster(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ASRHistoryForecaster":
        return _load_forecaster(cls, path, _init_history)

    def _prepare_origins(self, history, origins):
        series = _series(history)
        single = origins is None
        if single:
            if len(series) < self.lookback:
                raise AshlarError(f"history has {len(series)} steps; lookback is {self.lookback}.")
            origins = np.asarray([len(series)], dtype=np.int64)
        else:
            origins = np.asarray(origins, dtype=np.int64).reshape(-1)
            if origins.size == 0:
                raise AshlarError("at least one forecast origin is required.")
        scaled = _scale(series, self.mean_, self.std_)
        capacity = _patches_needed(origins, self.lookback)
        return scaled, origins, capacity, single

    def _select(self, series, origins, train, val, mean: float, std: float):
        scaled = _scale(series, mean, std)
        self._reset()

        def init():
            return _init_history(jax.random.PRNGKey(self.random_state), self.lookback, self.horizon, self.head)

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, origins[train], epoch)

        def score(state):
            if self.head == "mse":
                return self._mse_numpy(state, scaled, origins[val])
            return self._objective_numpy(state, scaled, origins[val])

        metric = "val_mse" if self.head == "mse" else "val_loss"
        return _train_until(
            init,
            step_epoch,
            score,
            self.max_epochs,
            self.patience,
            self.verbose,
            label="full-history",
            metric=metric,
        )

    def _run(self, scaled, origins):
        self._reset()

        def init():
            return _init_history(jax.random.PRNGKey(self.random_state), self.lookback, self.horizon, self.head)

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, origins, epoch)

        return _train_for(init, step_epoch, self.best_epoch_)

    def _epoch(self, model, scaled, origins, epoch: int):
        if self.verbose:
            print(f"  full-history epoch {epoch} training {len(origins)} origins", flush=True)
        if self._opt_state is None:
            optimizer, step = _make_history_step(FORECAST_LEARNING_RATE)
            self._opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            self._step = step
        generator = np.random.default_rng(self.random_state + epoch)
        order = generator.permutation(len(origins))
        grid = _patch_grid(scaled, self.lookback)
        capacity = self.max_patches_
        for start in range(0, len(order), self.batch_size):
            chosen = order[start : start + self.batch_size]
            if len(chosen) < 1:
                continue
            packed = _pack_history(scaled, origins[chosen], self.lookback, capacity, grid=grid)
            target = _horizon_targets(scaled, origins[chosen], self.horizon)
            model, self._opt_state, _loss = self._step(
                model,
                self._opt_state,
                *[jnp.asarray(part) for part in packed],
                jnp.asarray(target),
            )
        return model

    def _mse_numpy(self, model, scaled, origins) -> float:
        capacity = _patches_needed(origins, self.lookback)
        squares = []
        weights = []
        for start in range(0, len(origins), 128):
            chunk = origins[start : start + 128]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            pred = np.asarray(_history_apply(model, *[jnp.asarray(part) for part in packed]))
            target = _horizon_targets(scaled, chunk, self.horizon)
            squares.append(np.square(pred - target).sum())
            weights.append(target.size)
        return float(sum(squares) / sum(weights))

    def _objective_numpy(self, model, scaled, origins) -> float:
        capacity = _patches_needed(origins, self.lookback)
        total = 0.0
        count = 0
        for start in range(0, len(origins), 64):
            chunk = origins[start : start + 64]
            packed = _pack_history(scaled, chunk, self.lookback, capacity)
            target = _horizon_targets(scaled, chunk, self.horizon)
            value = float(
                _history_objective_jit(model, *[jnp.asarray(part) for part in packed], jnp.asarray(target))
            )
            total += value * int(len(chunk))
            count += int(len(chunk))
        return total / count

    def _reset(self) -> None:
        self._opt_state = None
        self._step = None


def _horizon_targets(series: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    index = np.asarray(origins, dtype=np.int64)[:, None] + np.arange(horizon, dtype=np.int64)[None, :]
    if int(index.max()) >= len(series):
        raise AshlarError("a forecast origin does not have a full horizon inside the series.")
    return np.asarray(series, dtype=np.float32)[index]


@eqx.filter_jit
def _history_apply(model: FullHistoryNet, recent, patches, within, valid, ages) -> jax.Array:
    return _history_batch(model, recent, patches, within, valid, ages)


def _history_flat(model: FullHistoryNet, recent, patches, within, valid, ages) -> jax.Array:
    return _history_batch(model, recent, patches, within, valid, ages)


@eqx.filter_jit
def _history_point(model: FullHistoryNet, recent, patches, within, valid, ages) -> jax.Array:
    flat = _history_flat(model, recent, patches, within, valid, ages)
    return point_values(model.head, decode(model.head, flat, model.horizon))


@eqx.filter_jit
def _history_gaussian(model: FullHistoryNet, recent, patches, within, valid, ages):
    flat = _history_flat(model, recent, patches, within, valid, ages)
    return decode("gaussian", flat, model.horizon)


@eqx.filter_jit
def _history_log_prob(model: FullHistoryNet, recent, patches, within, valid, ages, value) -> jax.Array:
    flat = _history_flat(model, recent, patches, within, valid, ages)
    return log_density(model.head, decode(model.head, flat, model.horizon), value)


@eqx.filter_jit
def _history_quantiles(model: FullHistoryNet, recent, patches, within, valid, ages) -> jax.Array:
    flat = _history_flat(model, recent, patches, within, valid, ages)
    return decode("quantile", flat, model.horizon)


@eqx.filter_jit
def _history_bins(model: FullHistoryNet, recent, patches, within, valid, ages) -> jax.Array:
    flat = _history_flat(model, recent, patches, within, valid, ages)
    return jax.nn.softmax(decode("histogram", flat, model.horizon), axis=-1)


@eqx.filter_jit
def _history_sample(model: FullHistoryNet, recent, patches, within, valid, ages, key, n: int) -> jax.Array:
    flat = _history_flat(model, recent, patches, within, valid, ages)
    return sample_values(model.head, decode(model.head, flat, model.horizon), key, n)


@eqx.filter_jit
def _history_objective_jit(model: FullHistoryNet, recent, patches, within, valid, ages, target) -> jax.Array:
    return _history_objective(model, recent, patches, within, valid, ages, target)


__getattr__ = renamed(
    __name__,
    globals(),
    {
        "TimeSeriesForecaster": "ASRForecaster",
        "FullHistoryForecaster": "ASRHistoryForecaster",
    },
)
