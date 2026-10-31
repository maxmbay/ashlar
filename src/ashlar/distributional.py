"""Distributional forecast heads for a continuous horizon.

Each head reads the same last-token features as the squared-error forecast and
trains on its own objective. The point returned by ``predict`` is only a
summary for absolute error:

- ``quantile``: pinball loss at 0.1, 0.5, and 0.9. The median is the point.
  Positive softplus gaps keep the three levels in that order.
- ``gaussian``: heteroscedastic normal. The scale is ``softplus(raw) + 1e-4``,
  the same map as the tabular normal head, and the loss is the normal
  negative log-density. The mean is the point.
- ``mixture``: five normals. Component scales use that same positive map,
  weights are a softmax, and the loss is the mixture negative log-density.
  The point is the weighted mean.
- ``flow``: one rational-quadratic spline per horizon step, with linear tails,
  pushed forward from a standard normal, then shifted and scaled per row:
  ``y = shift + scale * spline(z)``. The spline sets the shape inside
  ``[-3, 3]`` and the shift and scale where that lies, so each row has its
  own spread and its own tails. The loss is the exact change-of-variables
  log-density. A draw is one forward pass. The point is the median, the
  image of zero.
- ``histogram``: softmax over fixed bins on the standardized axis, plus a
  small squared error of the probability-weighted bin center. The point is
  that center.
- ``student_t``: location, scale, and degrees of freedom, for heavy tails.
  The scale uses the same positive map and the degrees of freedom are
  ``2 + softplus(raw)``, so the variance stays finite. The location is the
  point.

Two count heads read the raw target, not a standardized one, and only the
tabular estimator takes them:

- ``poisson``: a log rate. The loss is the Poisson negative log-mass.
- ``negative_binomial``: a log mean and a dispersion ``softplus(raw) + 1e-4``,
  the variance being ``mean + dispersion * mean**2``.

Either count head can be zero-inflated: one more output is the logit of a
structural zero, mixed with the count distribution. The point is the mean.

Squared error stays the ``mse`` head. Losses average over the batch and the
horizon. Quantile loss also averages the three levels.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ashlar.errors import AshlarError, ShapeError
from ashlar.predict import normal_log_prob, positive_scale

QUANTILE_LEVELS = (0.1, 0.5, 0.9)
MIXTURE_COMPONENTS = 5
FLOW_BINS = 8
FLOW_BOUND = 3.0
FLOW_MIN_BIN = 1e-3
FLOW_MIN_DERIVATIVE = 1e-3
# Floor of the flow head's per-row scale, as in the joint flow's columns.
FLOW_MIN_SCALE = 1e-3
# ``softplus(raw + this) + FLOW_MIN_SCALE`` is 1 at ``raw = 0``, so a new head starts at the unit scale.
FLOW_SCALE_OFFSET = math.log(math.expm1(1.0 - FLOW_MIN_SCALE))
# Saved with a model whose flow head ends in a shift and a scale. A save without it holds the bare spline.
FLOW_LAYOUT = "spline_shift_scale"
HISTOGRAM_BINS = 16
HISTOGRAM_LIMIT = 4.0
HISTOGRAM_LOCATION_WEIGHT = 0.1
STUDENT_T_MIN_DF = 2.0
STUDENT_T_INITIAL_DF = 5.0
COUNT_LOG_LIMIT = 20.0
ZERO_INITIAL_LOGIT = -2.0
OUTPUT_HEADS = ("mse", "quantile", "gaussian", "mixture", "flow", "histogram", "student_t")
COUNT_HEADS = ("poisson", "negative_binomial")
TABLE_HEADS = (*OUTPUT_HEADS, *COUNT_HEADS)
_COUNT_WIDTH = {"poisson": 1, "negative_binomial": 2}
_FLOW_DERIVATIVE_BETA = math.log(2.0) / (1.0 - FLOW_MIN_DERIVATIVE)


def check_output_head(head: str) -> str:
    """Return ``head`` when it names a forecast output. Raise otherwise."""
    if head not in OUTPUT_HEADS:
        raise AshlarError(
            "forecast head must be 'mse', 'quantile', 'gaussian', 'mixture', 'flow', 'histogram', "
            f"or 'student_t'; received {head!r}."
        )
    return head


def check_head(head: str) -> str:
    """Return ``head`` when it names a forecast or a tabular output, the count heads included."""
    if head not in TABLE_HEADS:
        raise AshlarError(f"head must be one of {', '.join(TABLE_HEADS)}; received {head!r}.")
    return head


def flow_parameter_count() -> int:
    """Unconstrained spline parameters per horizon step.

    Bin widths, bin heights, and one interior derivative per bin. The two
    boundary derivatives stay fixed at 1 so the linear tails meet the spline.
    The joint flow's columns use this spline and add their own shift and scale.
    """
    return 3 * FLOW_BINS - 1


def flow_output_count() -> int:
    """Parameters per horizon step of the ``flow`` head: the spline's, then a shift and a raw scale.

    A projection of only ``flow_parameter_count()`` per step, as saved before
    the shift and scale existed, still reads as a spline with shift 0 and
    scale 1.
    """
    return flow_parameter_count() + 2


def bare_spline_width(width: int, horizon: int) -> int:
    """The width a ``flow`` projection of ``width`` had before the per-row shift and scale."""
    if int(width) != int(horizon) * flow_output_count():
        raise ShapeError(f"a flow projection over {int(horizon)} steps has width {int(horizon) * flow_output_count()}; received {width}.")
    return int(horizon) * flow_parameter_count()


def flow_affine(raw):
    """Split flow parameters ``(..., P)`` into the spline, the shift, and the scale.

    ``P`` is ``flow_output_count()``, or ``flow_parameter_count()`` for a bare
    spline, whose shift is 0 and scale 1.
    """
    raw = jnp.asarray(raw)
    count = flow_parameter_count()
    if raw.shape[-1] == count:
        return raw, jnp.zeros(raw.shape[:-1], dtype=raw.dtype), jnp.ones(raw.shape[:-1], dtype=raw.dtype)
    if raw.shape[-1] != count + 2:
        raise ShapeError(f"flow parameters end in {count} or {count + 2} values; received {raw.shape}.")
    offset = jnp.asarray(FLOW_SCALE_OFFSET, dtype=raw.dtype)
    scale = jax.nn.softplus(raw[..., count + 1] + offset) + jnp.asarray(FLOW_MIN_SCALE, dtype=raw.dtype)
    return raw[..., :count], raw[..., count], scale


def output_width(head: str, horizon: int, bins: int = HISTOGRAM_BINS, zero_inflated: bool = False) -> int:
    """Width of the flat last-token projection for this head.

    ``bins`` is the histogram head's bin count and is ignored by every other
    head. More bins buy resolution: the density a histogram head can draw is
    piecewise constant on its bins. ``zero_inflated`` adds the structural-zero
    logit to a count head.
    """
    check_head(head)
    if zero_inflated and head not in COUNT_HEADS:
        raise AshlarError(f"zero inflation is for the count heads; the {head} head does not take it.")
    if horizon < 1:
        raise AshlarError("horizon must be at least 1.")
    if head == "mse":
        return int(horizon)
    if head == "quantile":
        return int(horizon) * 3
    if head == "gaussian":
        return int(horizon) * 2
    if head == "mixture":
        return int(horizon) * MIXTURE_COMPONENTS * 3
    if head == "flow":
        return int(horizon) * flow_output_count()
    if head == "student_t":
        return int(horizon) * 3
    if head in COUNT_HEADS:
        return int(horizon) * (_COUNT_WIDTH[head] + int(bool(zero_inflated)))
    return int(horizon) * check_histogram_bins(bins)


def initial_weight_scale(head: str, d_model: int) -> float:
    """Std of the output projection.

    The flow and the mixture start near a fixed bias (an identity spline, or
    component means spread across the standardized axis), so their projection
    is kept small. The other heads use the same ``1/sqrt(width)`` scale as the
    squared-error forecast.
    """
    check_head(head)
    if d_model < 1:
        raise AshlarError("forecast width must be positive.")
    if head in {"flow", "mixture", *COUNT_HEADS}:
        return 0.01
    return float(math.sqrt(1.0 / d_model))


def initial_bias(
    head: str, horizon: int, bins: int = HISTOGRAM_BINS, *, zero_inflated: bool = False, log_mean: float = 0.0
) -> jax.Array:
    """Output bias. Mixture means start on a grid, and quantile levels at the
    standard normal's 0.1, 0.5, and 0.9. A Student-t starts at 5 degrees of
    freedom. A count head starts at ``exp(log_mean)``, unit dispersion, and a
    small structural zero. Every other head starts at 0."""
    width = output_width(head, horizon, bins, zero_inflated)
    if head == "student_t":
        df_raw = math.log(math.expm1(STUDENT_T_INITIAL_DF - STUDENT_T_MIN_DF))
        return jnp.tile(jnp.asarray([0.0, 0.0, df_raw], dtype=jnp.float32), int(horizon))
    if head in COUNT_HEADS:
        row = [float(log_mean)]
        if head == "negative_binomial":
            row.append(math.log(math.expm1(1.0)))
        if zero_inflated:
            row.append(ZERO_INITIAL_LOGIT)
        return jnp.tile(jnp.asarray(row, dtype=jnp.float32), int(horizon))
    if head == "quantile":
        z = 1.2815515655446004
        gap = math.log(math.expm1(z))
        raw = jnp.tile(jnp.asarray([-z, gap, gap], dtype=jnp.float32), int(horizon))
        return raw
    if head != "mixture":
        return jnp.zeros((width,), dtype=jnp.float32)
    means = jnp.linspace(-2.0, 2.0, MIXTURE_COMPONENTS, dtype=jnp.float32)
    raw = jnp.zeros((int(horizon), MIXTURE_COMPONENTS, 3), dtype=jnp.float32)
    raw = raw.at[..., 0].set(means)
    return raw.reshape((width,))


def check_histogram_bins(bins: int) -> int:
    """Return ``bins`` when it is a usable bin count. Raise otherwise."""
    count = int(bins)
    if count < 2:
        raise AshlarError(f"a histogram head needs at least 2 bins; received {bins!r}.")
    return count


def histogram_edges(dtype=jnp.float32, bins: int = HISTOGRAM_BINS) -> jax.Array:
    """Fixed standardized bin edges, inclusive of both ends."""
    return jnp.linspace(-HISTOGRAM_LIMIT, HISTOGRAM_LIMIT, check_histogram_bins(bins) + 1, dtype=dtype)


def histogram_centers(dtype=jnp.float32, bins: int = HISTOGRAM_BINS) -> jax.Array:
    edges = histogram_edges(dtype, bins)
    return 0.5 * (edges[:-1] + edges[1:])


def histogram_bin(value: jax.Array, bins: int = HISTOGRAM_BINS) -> jax.Array:
    """Bin index of a standardized target. Values outside the edges use the end bins."""
    count = check_histogram_bins(bins)
    left = histogram_edges(value.dtype, count)[:-1]
    index = jnp.sum(value[..., None] >= left, axis=-1) - 1
    return jnp.clip(index, 0, count - 1).astype(jnp.int32)


def decode(head: str, flat: jax.Array, horizon: int, bins: int | None = None):
    """Turn a flat projection ``(batch, width)`` into constrained parameters.

    ``quantile`` returns levels ``(batch, horizon, 3)``. ``gaussian`` returns
    ``(loc, scale)`` each ``(batch, horizon)``. ``mixture`` returns
    ``(loc, scale, weight)`` each ``(batch, horizon, component)``. ``flow``
    returns unconstrained spline parameters ``(batch, horizon, parameters)``.
    ``histogram`` returns logits ``(batch, horizon, bin)``. ``student_t``
    returns ``(loc, scale, df)``, ``poisson`` returns ``(mean, zero)``, and
    ``negative_binomial`` returns ``(mean, dispersion, zero)``, each
    ``(batch, horizon)``; ``zero`` is the structural-zero probability, 0 for a
    head that is not zero-inflated.

    A histogram projection carries its own bin count in its width, and a count
    projection its zero inflation, so ``bins`` is only needed to check a width
    that should have been built differently.
    """
    check_head(head)
    if head == "mse":
        raise AshlarError("the squared-error head has no distributional parameters.")
    flat = jnp.asarray(flat)
    if head == "histogram" and bins is None and flat.ndim == 2 and flat.shape[-1] % int(horizon) == 0:
        bins = flat.shape[-1] // int(horizon)
    zero_inflated = (
        head in COUNT_HEADS and flat.ndim == 2 and flat.shape[-1] == int(horizon) * (_COUNT_WIDTH[head] + 1)
    )
    width = output_width(head, horizon, HISTOGRAM_BINS if bins is None else bins, zero_inflated)
    if head == "flow" and flat.ndim == 2 and flat.shape[-1] == int(horizon) * flow_parameter_count():
        # A flow head saved before the per-row shift and scale: the bare spline.
        width = flat.shape[-1]
    if flat.ndim != 2 or flat.shape[-1] != width:
        raise ShapeError(
            f"{head} projection must have shape (batch, {width}); received {flat.shape}."
        )
    batch = flat.shape[0]
    if head == "quantile":
        raw = flat.reshape(batch, int(horizon), 3)
        low = raw[..., 0]
        mid = low + jax.nn.softplus(raw[..., 1])
        high = mid + jax.nn.softplus(raw[..., 2])
        return jnp.stack((low, mid, high), axis=-1)
    if head == "gaussian":
        raw = flat.reshape(batch, int(horizon), 2)
        return raw[..., 0], positive_scale(raw[..., 1])
    if head == "mixture":
        raw = flat.reshape(batch, int(horizon), MIXTURE_COMPONENTS, 3)
        return raw[..., 0], positive_scale(raw[..., 1]), jax.nn.softmax(raw[..., 2], axis=-1)
    if head == "flow":
        return flat.reshape(batch, int(horizon), width // int(horizon))
    if head == "student_t":
        raw = flat.reshape(batch, int(horizon), 3)
        df = jax.nn.softplus(raw[..., 2]) + jnp.asarray(STUDENT_T_MIN_DF, dtype=raw.dtype)
        return raw[..., 0], positive_scale(raw[..., 1]), df
    if head in COUNT_HEADS:
        raw = flat.reshape(batch, int(horizon), width // int(horizon))
        mean = jnp.exp(jnp.clip(raw[..., 0], -COUNT_LOG_LIMIT, COUNT_LOG_LIMIT))
        zero = jax.nn.sigmoid(raw[..., -1]) if zero_inflated else jnp.zeros_like(mean)
        if head == "poisson":
            return mean, zero
        return mean, positive_scale(raw[..., 1]), zero
    return flat.reshape(batch, int(horizon), width // int(horizon))


def point_values(head: str, decoded) -> jax.Array:
    """Standardized point forecast. Shape ``(batch, horizon)``. A count head's point is its raw mean."""
    check_head(head)
    if head == "student_t":
        return decoded[0]
    if head in COUNT_HEADS:
        return (1.0 - decoded[-1]) * decoded[0]
    if head == "quantile":
        return decoded[..., 1]
    if head == "gaussian":
        loc, _scale = decoded
        return loc
    if head == "mixture":
        loc, _scale, weight = decoded
        return jnp.sum(weight * loc, axis=-1)
    if head == "flow":
        zeros = jnp.zeros(decoded.shape[:-1], dtype=decoded.dtype)
        return flow_forward(decoded, zeros)
    if head == "histogram":
        probabilities = jax.nn.softmax(decoded, axis=-1)
        return probabilities @ histogram_centers(decoded.dtype, decoded.shape[-1])
    raise AshlarError("the squared-error head is already a point forecast.")


def distributional_loss(head: str, decoded, target: jax.Array) -> jax.Array:
    """Mean training loss. ``target`` is standardized ``(batch, horizon)``."""
    check_head(head)
    if head in {"student_t", *COUNT_HEADS}:
        return jnp.mean(-log_density(head, decoded, target))
    target = jnp.asarray(target)
    if head == "quantile":
        if decoded.ndim != 3 or decoded.shape[-1] != 3 or decoded.shape[:-1] != target.shape:
            raise ShapeError(
                f"quantile parameters must have shape {target.shape} + (3,); received {getattr(decoded, 'shape', None)}."
            )
        levels = jnp.asarray(QUANTILE_LEVELS, dtype=target.dtype)
        gap = target[..., None] - decoded
        pinball = jnp.maximum(levels * gap, (levels - 1.0) * gap)
        return jnp.mean(pinball)
    if head == "gaussian":
        loc, scale = decoded
        _same_shape(loc, target, "gaussian location")
        return jnp.mean(-normal_log_prob(target, loc, scale))
    if head == "mixture":
        loc, scale, weight = decoded
        if loc.shape != scale.shape or loc.shape != weight.shape or loc.shape[:-1] != target.shape:
            raise ShapeError(
                "mixture parameters must share shape "
                f"{target.shape} + ({MIXTURE_COMPONENTS},); received {getattr(loc, 'shape', None)}."
            )
        log_component = normal_log_prob(target[..., None], loc, scale) + _log_weight(weight)
        return jnp.mean(-jax.nn.logsumexp(log_component, axis=-1))
    if head == "flow":
        _flow_shape(decoded, target)
        return jnp.mean(-flow_log_density(decoded, target))
    if head == "histogram":
        if decoded.ndim != 3 or decoded.shape[-1] < 2 or decoded.shape[:-1] != target.shape:
            raise ShapeError(
                f"histogram logits must have shape {target.shape} + (bins,) with at least 2 bins; "
                f"received {getattr(decoded, 'shape', None)}."
            )
        count = decoded.shape[-1]
        log_probability = jax.nn.log_softmax(decoded, axis=-1)
        chosen = jnp.take_along_axis(log_probability, histogram_bin(target, count)[..., None], axis=-1)[..., 0]
        expected = jax.nn.softmax(decoded, axis=-1) @ histogram_centers(decoded.dtype, count)
        squared = jnp.square(expected - target)
        return jnp.mean(-chosen + jnp.asarray(HISTOGRAM_LOCATION_WEIGHT, dtype=target.dtype) * squared)
    raise AshlarError("the squared-error head does not use the distributional loss.")


def _log_weight(weight):
    """Log of mixture weights. A weight that underflows to 0 gives a large
    negative number, not ``-inf``, so its gradient stays finite."""
    return jnp.log(jnp.maximum(weight, jnp.asarray(1e-30, dtype=weight.dtype)))


def row_loss(head: str, decoded, target: jax.Array) -> jax.Array:
    """Per-step training loss, same shape as ``target``. Its mean is ``distributional_loss``."""
    check_head(head)
    target = jnp.asarray(target)
    if head == "quantile":
        levels = jnp.asarray(QUANTILE_LEVELS, dtype=target.dtype)
        gap = target[..., None] - decoded
        return jnp.mean(jnp.maximum(levels * gap, (levels - 1.0) * gap), axis=-1)
    if head == "histogram":
        count = decoded.shape[-1]
        log_probability = jax.nn.log_softmax(decoded, axis=-1)
        chosen = jnp.take_along_axis(log_probability, histogram_bin(target, count)[..., None], axis=-1)[..., 0]
        expected = jax.nn.softmax(decoded, axis=-1) @ histogram_centers(decoded.dtype, count)
        squared = jnp.square(expected - target)
        return -chosen + jnp.asarray(HISTOGRAM_LOCATION_WEIGHT, dtype=target.dtype) * squared
    return -log_density(head, decoded, target)


def log_density(head: str, decoded, value: jax.Array) -> jax.Array:
    """Per-step log density of standardized ``value``. Same shape as ``value``.

    A count head reads the raw count and returns its log mass.
    """
    check_head(head)
    value = jnp.asarray(value)
    if head == "student_t":
        loc, scale, df = decoded
        _same_shape(loc, value, "student_t location")
        return student_t_log_prob(value, loc, scale, df)
    if head in COUNT_HEADS:
        _same_shape(decoded[0], value, f"{head} mean")
        if head == "poisson":
            mean, zero = decoded
            log_count = poisson_log_prob(value, mean)
        else:
            mean, dispersion, zero = decoded
            log_count = negative_binomial_log_prob(value, mean, dispersion)
        return zero_inflated_log_prob(value, log_count, zero)
    if head == "gaussian":
        loc, scale = decoded
        _same_shape(loc, value, "gaussian location")
        return normal_log_prob(value, loc, scale)
    if head == "mixture":
        loc, scale, weight = decoded
        if loc.shape[:-1] != value.shape:
            raise ShapeError(
                f"mixture log density expected value shape {loc.shape[:-1]}; received {value.shape}."
            )
        log_component = normal_log_prob(value[..., None], loc, scale) + _log_weight(weight)
        return jax.nn.logsumexp(log_component, axis=-1)
    if head == "flow":
        _flow_shape(decoded, value)
        return flow_log_density(decoded, value)
    if head == "histogram":
        if decoded.shape[:-1] != value.shape or decoded.shape[-1] < 2:
            raise ShapeError(
                f"histogram log density expected value shape {decoded.shape[:-1]}; received {value.shape}."
            )
        count = decoded.shape[-1]
        log_probability = jax.nn.log_softmax(decoded, axis=-1)
        chosen = jnp.take_along_axis(log_probability, histogram_bin(value, count)[..., None], axis=-1)[..., 0]
        edges = histogram_edges(value.dtype, count)
        width = edges[1] - edges[0]
        return chosen - jnp.log(width)
    raise AshlarError(f"the {head} head has no log density.")


def sample_values(head: str, decoded, key, n: int) -> jax.Array:
    """``n`` standardized draws. Shape ``(n, batch, horizon)``.

    The flow draw is one rational-quadratic forward pass from standard normal
    noise. The gaussian, mixture, and histogram draws are one sample from
    those densities. A quantile head has no draw.
    """
    check_head(head)
    count = int(n)
    if count < 1:
        raise AshlarError("the number of samples must be positive.")
    if head == "student_t":
        loc, scale, df = decoded
        noise = jax.random.t(key, df, (count,) + loc.shape, dtype=loc.dtype)
        return loc[None, ...] + scale[None, ...] * noise
    if head in COUNT_HEADS:
        mean, zero = decoded[0], decoded[-1]
        rate_key, count_key, zero_key = jax.random.split(key, 3)
        shape = (count,) + mean.shape
        rate = jnp.broadcast_to(mean, shape)
        if head == "negative_binomial":
            size = 1.0 / decoded[1]
            rate = rate * jax.random.gamma(rate_key, jnp.broadcast_to(size, shape), dtype=mean.dtype) / size
        draws = jax.random.poisson(count_key, rate, shape).astype(mean.dtype)
        structural = jax.random.bernoulli(zero_key, jnp.broadcast_to(zero, shape))
        return jnp.where(structural, jnp.zeros_like(draws), draws)
    if head == "gaussian":
        loc, scale = decoded
        noise = jax.random.normal(key, (count,) + loc.shape, dtype=loc.dtype)
        return loc[None, ...] + scale[None, ...] * noise
    if head == "mixture":
        loc, scale, weight = decoded
        component_key, noise_key = jax.random.split(key)
        choices = jax.random.categorical(
            component_key, _log_weight(weight), axis=-1, shape=(count,) + loc.shape[:-1]
        )
        chosen_loc = jnp.take_along_axis(jnp.broadcast_to(loc, (count,) + loc.shape), choices[..., None], axis=-1)[
            ..., 0
        ]
        chosen_scale = jnp.take_along_axis(
            jnp.broadcast_to(scale, (count,) + scale.shape), choices[..., None], axis=-1
        )[..., 0]
        noise = jax.random.normal(noise_key, chosen_loc.shape, dtype=loc.dtype)
        return chosen_loc + chosen_scale * noise
    if head == "flow":
        noise = jax.random.normal(key, (count,) + decoded.shape[:-1], dtype=decoded.dtype)
        return jax.vmap(lambda noise_row: flow_forward(decoded, noise_row))(noise)
    if head == "histogram":
        bin_key, uniform_key = jax.random.split(key)
        choices = jax.random.categorical(bin_key, decoded, axis=-1, shape=(count,) + decoded.shape[:-1])
        edges = histogram_edges(decoded.dtype, decoded.shape[-1])
        left = jnp.take(edges[:-1], choices)
        right = jnp.take(edges[1:], choices)
        uniform = jax.random.uniform(uniform_key, choices.shape, dtype=decoded.dtype)
        return left + uniform * (right - left)
    raise AshlarError(f"the {head} head has no sampler.")


def student_t_log_prob(value, loc, scale, df):
    """Elementwise Student-t log density with location, scale, and degrees of freedom."""
    z = (value - loc) / scale
    return (
        jax.scipy.special.gammaln(0.5 * (df + 1.0))
        - jax.scipy.special.gammaln(0.5 * df)
        - 0.5 * jnp.log(df * jnp.pi)
        - jnp.log(scale)
        - 0.5 * (df + 1.0) * jnp.log1p(z * z / df)
    )


def poisson_log_prob(value, mean):
    """Elementwise Poisson log mass of a count."""
    return value * jnp.log(mean) - mean - jax.scipy.special.gammaln(value + 1.0)


def negative_binomial_log_prob(value, mean, dispersion):
    """Negative binomial log mass with mean ``mean`` and variance ``mean + dispersion * mean**2``."""
    size = 1.0 / dispersion
    log_total = jnp.log(size + mean)
    return (
        jax.scipy.special.gammaln(value + size)
        - jax.scipy.special.gammaln(size)
        - jax.scipy.special.gammaln(value + 1.0)
        + size * (jnp.log(size) - log_total)
        + value * (jnp.log(mean) - log_total)
    )


def zero_inflated_log_prob(value, log_count, zero):
    """Mix a count log mass with a structural zero of probability ``zero``."""
    log_keep = jnp.log1p(-jnp.minimum(zero, 1.0 - 1e-6))
    at_zero = jnp.logaddexp(_log_weight(zero), log_keep + log_count)
    return jnp.where(value == 0, at_zero, log_keep + log_count)


def _same_shape(left, right, name: str) -> None:
    if left.shape != right.shape:
        raise ShapeError(f"{name} must have shape {right.shape}; received {left.shape}.")


def _flow_shape(raw, value) -> None:
    widths = (flow_output_count(), flow_parameter_count())
    if raw.ndim != value.ndim + 1 or raw.shape[:-1] != value.shape or raw.shape[-1] not in widths:
        raise ShapeError(
            f"flow parameters must have shape {value.shape} + ({flow_output_count()},); received {raw.shape}."
        )


def _positive_bin_fractions(raw):
    fractions = jax.nn.softmax(raw, axis=-1)
    return FLOW_MIN_BIN + (1.0 - FLOW_MIN_BIN * FLOW_BINS) * fractions


def _flow_geometry(raw):
    """Knots on ``[-bound, bound]`` and positive derivatives, boundary ones fixed at 1."""
    count = FLOW_BINS
    widths_raw = raw[..., :count]
    heights_raw = raw[..., count : 2 * count]
    derivative_raw = raw[..., 2 * count :]
    span = jnp.asarray(2.0 * FLOW_BOUND, dtype=raw.dtype)
    bound = jnp.asarray(FLOW_BOUND, dtype=raw.dtype)

    def knots(fractions):
        cumulative = jnp.cumsum(fractions, axis=-1)
        zeros = jnp.zeros_like(fractions[..., :1])
        cumulative = jnp.concatenate([zeros, cumulative], axis=-1)
        cumulative = cumulative * span - bound
        cumulative = cumulative.at[..., 0].set(-bound)
        cumulative = cumulative.at[..., -1].set(bound)
        sizes = cumulative[..., 1:] - cumulative[..., :-1]
        return cumulative, sizes

    cumwidths, widths = knots(_positive_bin_fractions(widths_raw))
    cumheights, heights = knots(_positive_bin_fractions(heights_raw))
    beta = jnp.asarray(_FLOW_DERIVATIVE_BETA, dtype=raw.dtype)
    floor = jnp.asarray(FLOW_MIN_DERIVATIVE, dtype=raw.dtype)
    interior = floor + jnp.logaddexp(jnp.zeros_like(derivative_raw), beta * derivative_raw) / beta
    ones = jnp.ones_like(interior[..., :1])
    derivatives = jnp.concatenate([ones, interior, ones], axis=-1)
    return cumwidths, widths, cumheights, heights, derivatives


def _bin_index(knots, inputs):
    edges = knots.at[..., -1].add(jnp.asarray(1e-6, dtype=knots.dtype))
    index = jnp.sum(inputs[..., None] >= edges, axis=-1) - 1
    return jnp.clip(index, 0, FLOW_BINS - 1).astype(jnp.int32)


def _gather(values, index):
    return jnp.take_along_axis(values, index[..., None], axis=-1)[..., 0]


def _spline_pieces(inputs, cumwidths, widths, cumheights, heights, derivatives, inverse: bool):
    if inverse:
        index = _bin_index(cumheights, inputs)
    else:
        index = _bin_index(cumwidths, inputs)
    input_cumwidths = _gather(cumwidths, index)
    input_bin_widths = _gather(widths, index)
    input_cumheights = _gather(cumheights, index)
    input_heights = _gather(heights, index)
    delta = heights / widths
    input_delta = _gather(delta, index)
    input_derivatives = _gather(derivatives, index)
    input_derivatives_plus_one = _gather(derivatives[..., 1:], index)
    return (
        input_cumwidths,
        input_bin_widths,
        input_cumheights,
        input_heights,
        input_delta,
        input_derivatives,
        input_derivatives_plus_one,
    )


def _rational_quadratic(theta, input_cumheights, input_heights, input_delta, input_derivatives, input_derivatives_plus_one):
    theta_one_minus_theta = theta * (1.0 - theta)
    numerator = input_heights * (input_delta * jnp.square(theta) + input_derivatives * theta_one_minus_theta)
    denominator = input_delta + (
        (input_derivatives + input_derivatives_plus_one - 2.0 * input_delta) * theta_one_minus_theta
    )
    outputs = input_cumheights + numerator / jnp.maximum(denominator, jnp.asarray(1e-6, dtype=denominator.dtype))
    derivative_numerator = jnp.square(input_delta) * (
        input_derivatives_plus_one * jnp.square(theta)
        + 2.0 * input_delta * theta_one_minus_theta
        + input_derivatives * jnp.square(1.0 - theta)
    )
    safe_numerator = jnp.maximum(derivative_numerator, jnp.asarray(1e-6, dtype=derivative_numerator.dtype))
    safe_denominator = jnp.maximum(denominator, jnp.asarray(1e-6, dtype=denominator.dtype))
    logabsdet = jnp.log(safe_numerator) - 2.0 * jnp.log(safe_denominator)
    return outputs, logabsdet


def _spline_forward_core(inputs, cumwidths, widths, cumheights, heights, derivatives):
    (
        input_cumwidths,
        input_bin_widths,
        input_cumheights,
        input_heights,
        input_delta,
        input_derivatives,
        input_derivatives_plus_one,
    ) = _spline_pieces(inputs, cumwidths, widths, cumheights, heights, derivatives, inverse=False)
    theta = (inputs - input_cumwidths) / input_bin_widths
    return _rational_quadratic(
        theta,
        input_cumheights,
        input_heights,
        input_delta,
        input_derivatives,
        input_derivatives_plus_one,
    )


def _spline_inverse_core(inputs, cumwidths, widths, cumheights, heights, derivatives):
    (
        input_cumwidths,
        input_bin_widths,
        input_cumheights,
        input_heights,
        input_delta,
        input_derivatives,
        input_derivatives_plus_one,
    ) = _spline_pieces(inputs, cumwidths, widths, cumheights, heights, derivatives, inverse=True)
    shift = inputs - input_cumheights
    slope_gap = input_derivatives + input_derivatives_plus_one - 2.0 * input_delta
    quadratic_a = shift * slope_gap + input_heights * (input_delta - input_derivatives)
    quadratic_b = input_heights * input_derivatives - shift * slope_gap
    quadratic_c = -input_delta * shift
    discriminant = jnp.maximum(jnp.square(quadratic_b) - 4.0 * quadratic_a * quadratic_c, 0.0)
    root = (2.0 * quadratic_c) / (-quadratic_b - jnp.sqrt(discriminant))
    outputs = root * input_bin_widths + input_cumwidths
    _forward, logabsdet = _rational_quadratic(
        root,
        input_cumheights,
        input_heights,
        input_delta,
        input_derivatives,
        input_derivatives_plus_one,
    )
    return outputs, -logabsdet


def _flow_map(raw, inputs, inverse: bool):
    bound = jnp.asarray(FLOW_BOUND, dtype=inputs.dtype)
    inside = (inputs >= -bound) & (inputs <= bound)
    clipped = jnp.clip(inputs, -bound, bound)
    geometry = _flow_geometry(raw)
    if inverse:
        mapped, logdet = _spline_inverse_core(clipped, *geometry)
    else:
        mapped, logdet = _spline_forward_core(clipped, *geometry)
    outputs = jnp.where(inside, mapped, inputs)
    logdet = jnp.where(inside, logdet, jnp.zeros_like(inputs))
    return outputs, logdet


def flow_forward(raw, base) -> jax.Array:
    """Push a standard-normal sample through the conditional spline, then its shift and scale.

    ``raw`` ends in ``flow_output_count()`` values, or in
    ``flow_parameter_count()`` for the bare spline.
    """
    spline, shift, scale = flow_affine(raw)
    outputs, _logdet = _flow_map(spline, base, inverse=False)
    return shift + scale * outputs


def flow_inverse(raw, value):
    """Map a standardized observation to the base noise and ``log|dz/dy|``."""
    spline, shift, scale = flow_affine(raw)
    base, logdet = _flow_map(spline, (value - shift) / scale, inverse=True)
    return base, logdet - jnp.log(scale)


def flow_log_density(raw, value) -> jax.Array:
    """Exact standardized log density under the conditional spline."""
    base, logdet = flow_inverse(raw, value)
    log_base = -0.5 * (jnp.log(2.0 * jnp.pi) + jnp.square(base))
    return log_base + logdet
