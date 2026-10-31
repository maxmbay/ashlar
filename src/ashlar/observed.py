"""Labels that say what was observed about ``y``, and the likelihood of each.

A plain ``y`` says every value was seen exactly. Many tables know less: a
sale capped by the stock on the shelf, an income reported as a bracket, a
reading below a detection limit, a customer still active when the data was
pulled, a price rounded to the dollar, a measurement with known error, rows
recorded only because their value fell inside a range, or a total recorded
for a group of rows instead of each row's own value. Each is a statement
about where ``y`` lies, and the likelihood of a row is the probability the
model gives that statement (Heitjan and Rubin, 1991):

- ``exact(y)``: the value itself. The log density, or the log mass for a count.
- ``interval(lower, upper)``: ``P(lower <= Y <= upper)``, a closed interval;
  ``-inf`` or ``inf`` leaves a side open.
- ``censored(y, flag)``: where ``flag`` is set, only ``Y >= y`` is known, as
  for a day that sold out (``side="left"`` for ``Y <= y``, as for a reading
  below a detection limit). Elsewhere ``y`` is exact.
- ``survival(time, event, entry=None)``: an event at ``time`` where ``event``
  is set and right-censoring at ``time`` elsewhere; with ``entry``, each row
  was only seen because its time passed ``entry`` (delayed entry).
- ``rounded(y, step)``: ``y`` is the value rounded to a multiple of ``step``.
- ``truncated(label, lower, upper)``: rows were recorded only when the value
  fell inside ``[lower, upper]``, so the likelihood is divided by
  ``P(lower <= Y <= upper)``.
- ``noisy(label, scale)``: what was recorded is ``Y`` plus independent normal
  noise of standard deviation ``scale``. Bounds and truncation apply to the
  recorded value.
- ``total(values, group)``: rows that share a ``group`` id were observed only
  through the sum of their values, which ``values`` holds on every row of
  the group. ``grouped(label, group)`` does the same for any label, so a
  group's total can itself be censored or rounded.

Every function returns a NumPy structured array with one record per row, so
scikit-learn splits, indexes, and cross-validates it like any other ``y``.
Pass it as ``y`` to ``fit``, ``log_prob``, ``score``, and ``calibrate``;
``predict_dist`` still describes ``Y`` itself, so its quantiles, draws, and
intervals are for the true value, not for what was recorded. Labels compose:
``truncated(censored(sales, sold_out), lower=1)`` is a right-censored count
that is only recorded when positive.

The fields are ``lower`` and ``upper`` (equal for an exact value),
``truncate_lower`` and ``truncate_upper`` (``-inf`` and ``inf`` for none),
``noise`` (0 for none), and ``group`` (``-1`` for a row that is its own
observation).

The JAX functions ``log_likelihood`` and ``randomized_pit`` take any
``ashlar.distributions.Distribution``, so a custom objective can train on
the same labels: the training ``y`` it receives is the ``(rows, 6)`` array of
``fields(label)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ashlar.errors import AshlarError, ShapeError

FIELDS = ("lower", "upper", "truncate_lower", "truncate_upper", "noise", "group")
DTYPE = np.dtype(
    [
        ("lower", np.float64),
        ("upper", np.float64),
        ("truncate_lower", np.float64),
        ("truncate_upper", np.float64),
        ("noise", np.float64),
        ("group", np.int64),
    ]
)


def _column(values, name: str, n: int | None = None) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AshlarError(f"{name} must be numeric.") from exc
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim == 0 and n is not None:
        array = np.full(n, float(array))
    if array.ndim != 1:
        raise ShapeError(f"{name} must be one value per row; received shape {array.shape}.")
    if n is not None and len(array) != n:
        raise ShapeError(f"{name} has {len(array)} rows; expected {n}.")
    return array


def _flags(values, name: str, n: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.shape != (n,):
        raise ShapeError(f"{name} must be one flag per row; received shape {array.shape}.")
    if array.dtype != bool:
        if not np.isin(array, (0, 1)).all():
            raise AshlarError(f"{name} must hold booleans or 0 and 1.")
        array = array.astype(bool)
    return array


def _new(lower, upper) -> np.ndarray:
    label = np.empty(len(lower), dtype=DTYPE)
    label["lower"] = lower
    label["upper"] = upper
    label["truncate_lower"] = -np.inf
    label["truncate_upper"] = np.inf
    label["noise"] = 0.0
    label["group"] = -1
    return check(label)


def exact(values) -> np.ndarray:
    """Each row's value, seen exactly."""
    values = _column(values, "values")
    return _new(values, values)


def interval(lower, upper) -> np.ndarray:
    """Each row's value lies in the closed interval ``[lower, upper]``. ``-inf`` or ``inf`` leaves a side open."""
    lower = _column(lower, "lower")
    return _new(lower, _column(upper, "upper", len(lower)))


def censored(values, censored, side: str = "right") -> np.ndarray:
    """Exact values, except where ``censored`` is set: there only ``Y >= value`` (``side="right"``) or ``Y <= value`` is known.

    For a count such as units sold out of a stock of 12, ``Y >= 12``
    includes 12 itself, as it should.
    """
    values = _column(values, "values")
    flag = _flags(censored, "censored", len(values))
    if side == "right":
        return _new(values, np.where(flag, np.inf, values))
    if side == "left":
        return _new(np.where(flag, -np.inf, values), values)
    raise AshlarError(f"side must be 'right' or 'left'; received {side!r}.")


def survival(time, event, entry=None) -> np.ndarray:
    """An event at ``time`` where ``event`` is set, and still waiting at ``time`` elsewhere (right-censored).

    ``entry`` is when each row came under observation: a row is only in the
    table because its time passed its entry, so its likelihood is
    truncated at ``entry`` from below.
    """
    time = _column(time, "time")
    label = censored(time, ~_flags(event, "event", len(time)))
    if entry is None:
        return label
    return truncated(label, lower=_column(entry, "entry", len(time)))


def rounded(values, step) -> np.ndarray:
    """``values`` are the true values rounded to the nearest multiple of ``step``: each lies within ``step / 2``.

    For counts, write the whole numbers each reported value stands for
    with ``interval``, since rounding a count leaves a closed range of integers.
    """
    values = _column(values, "values")
    step = _column(step, "step", len(values))
    if not np.isfinite(step).all() or (step <= 0).any():
        raise AshlarError("step must be positive and finite.")
    return _new(values - 0.5 * step, values + 0.5 * step)


def truncated(label, lower=-np.inf, upper=np.inf) -> np.ndarray:
    """Rows were recorded only when the value fell inside ``[lower, upper]``; plain values are read as exact."""
    label = as_observed(label).copy()
    n = len(label)
    label["truncate_lower"] = np.maximum(label["truncate_lower"], _column(lower, "lower", n))
    label["truncate_upper"] = np.minimum(label["truncate_upper"], _column(upper, "upper", n))
    return check(label)


def noisy(label, scale) -> np.ndarray:
    """What was recorded is the value plus independent normal noise of standard deviation ``scale``.

    The model's distribution is for the value itself; the likelihood is that
    of the recorded value, so the head learns to subtract the noise it is
    told about. Bounds and truncation apply to the recorded value. The
    ``gaussian``, ``mixture``, and ``normal`` heads support it, since a
    normal plus normal noise stays in closed form.
    """
    label = as_observed(label).copy()
    scale = _column(scale, "scale", len(label))
    if not np.isfinite(scale).all() or (scale < 0).any():
        raise AshlarError("noise scale must be non-negative and finite.")
    label["noise"] = np.sqrt(np.square(label["noise"]) + np.square(scale))
    return check(label)


def grouped(label, group) -> np.ndarray:
    """Rows that share a ``group`` id were observed only through the sum of their values, as ``label`` says.

    Every row of a group carries the same observation of the group's sum.
    A group id of ``-1`` leaves a row as its own observation. The
    ``gaussian``, ``normal``, and ``poisson`` heads support totals, because a
    sum of their rows has a closed form.
    """
    label = as_observed(label).copy()
    try:
        ids = np.asarray(group)
        if ids.ndim == 2 and ids.shape[1] == 1:
            ids = ids[:, 0]
        if ids.shape != (len(label),):
            raise ShapeError(f"group must be one id per row; received shape {ids.shape}.")
        if not np.issubdtype(ids.dtype, np.integer):
            if not np.all(np.asarray(ids, dtype=np.float64) == np.round(np.asarray(ids, dtype=np.float64))):
                raise AshlarError("group ids must be whole numbers.")
        ids = ids.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise AshlarError("group ids must be whole numbers.") from exc
    if (ids < -1).any():
        raise AshlarError("group ids must be -1 (no group) or non-negative.")
    label["group"] = ids
    return check(label)


def total(values, group) -> np.ndarray:
    """Each group's total, seen exactly. ``values`` holds the group's total on every one of its rows."""
    return grouped(exact(values), group)


def concatenate(labels) -> np.ndarray:
    """Stack labels row-wise. Plain values are read as exact."""
    return check(np.concatenate([as_observed(label) for label in labels]))


def is_observed(y) -> bool:
    """Whether ``y`` is a label from this module."""
    dtype = getattr(y, "dtype", None)
    return dtype is not None and dtype.names is not None and set(FIELDS) <= set(dtype.names)


def as_observed(y) -> np.ndarray:
    """``y`` itself if it is a label from this module; otherwise its values, read as exact."""
    if is_observed(y):
        array = np.asarray(y)
        if array.dtype != DTYPE:
            converted = np.empty(array.shape, dtype=DTYPE)
            for name in FIELDS:
                converted[name] = array[name]
            array = converted
        if array.ndim != 1:
            raise ShapeError(f"an observed label is one record per row; received shape {array.shape}.")
        return array
    return exact(y)


def check(label) -> np.ndarray:
    """Raise ``AshlarError`` unless every record describes a possible observation. Returns the label."""
    if not is_observed(label):
        raise AshlarError("expected a label from ashlar.observed.")
    label = np.asarray(label)
    if label.ndim != 1:
        raise ShapeError(f"an observed label is one record per row; received shape {label.shape}.")
    lower, upper = label["lower"], label["upper"]
    t_lower, t_upper = label["truncate_lower"], label["truncate_upper"]
    numbers = (lower, upper, t_lower, t_upper, label["noise"])
    if any(np.isnan(field).any() for field in numbers):
        raise AshlarError("an observed label has no missing fields; use interval(-inf, inf) for a row with no information.")
    if (lower > upper).any():
        raise AshlarError("an observed interval needs lower <= upper.")
    if (lower == np.inf).any() or (upper == -np.inf).any():
        raise AshlarError("an observed interval needs a finite lower bound or -inf, and a finite upper bound or inf.")
    if (t_lower >= t_upper).any():
        raise AshlarError("a truncation range needs truncate_lower < truncate_upper.")
    if (t_lower > lower).any() or (upper > t_upper).any():
        raise AshlarError("an observation must lie inside its truncation range.")
    noise = label["noise"]
    if not np.isfinite(noise).all() or (noise < 0).any():
        raise AshlarError("noise must be a non-negative, finite standard deviation.")
    group = label["group"]
    if (group < -1).any():
        raise AshlarError("group ids must be -1 (no group) or non-negative.")
    members = group >= 0
    if members.any():
        ids, first = np.unique(group[members], return_index=True)
        rows = np.flatnonzero(members)
        lead = rows[first][np.searchsorted(ids, group[members])]
        for name in FIELDS[:5]:
            values = label[name]
            if not np.array_equal(values[members], values[lead]):
                raise AshlarError(f"every row of a group carries the group's observation; {name} differs within a group.")
    return label


def fields(label) -> np.ndarray:
    """The ``(rows, 6)`` float array of ``lower, upper, truncate_lower, truncate_upper, noise, group``."""
    label = as_observed(label)
    return np.stack([np.asarray(label[name], dtype=np.float64) for name in FIELDS], axis=1)


def group_sizes(label) -> np.ndarray:
    """Rows in each row's group, 1 for a row with no group."""
    label = as_observed(label)
    group = label["group"]
    sizes = np.ones(len(label), dtype=np.int64)
    members = group >= 0
    if members.any():
        ids, inverse, counts = np.unique(group[members], return_inverse=True, return_counts=True)
        sizes[members] = counts[inverse]
    return sizes


def has_groups(label) -> bool:
    return bool((as_observed(label)["group"] >= 0).any())


def is_truncated(label) -> bool:
    label = as_observed(label)
    return bool(np.isfinite(label["truncate_lower"]).any() or np.isfinite(label["truncate_upper"]).any())


def is_noisy(label) -> bool:
    return bool((as_observed(label)["noise"] > 0).any())


def is_exact(label) -> np.ndarray:
    label = as_observed(label)
    return label["lower"] == label["upper"]


def representative(label) -> np.ndarray:
    """One number per row that stands for its observation, for scaling and starting values.

    The value itself if exact; the midpoint of a finite interval; the finite
    bound of a one-sided one; ``nan`` for no information. A group's
    number is spread evenly over its rows.
    """
    label = as_observed(label)
    lower, upper = label["lower"], label["upper"]
    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    value = np.full(len(label), np.nan)
    both = finite_lower & finite_upper
    value[both] = 0.5 * (lower[both] + upper[both])
    value[finite_lower & ~finite_upper] = lower[finite_lower & ~finite_upper]
    value[~finite_lower & finite_upper] = upper[~finite_lower & finite_upper]
    return value / group_sizes(label)


def affine(label, loc: float, scale: float) -> np.ndarray:
    """The label of ``(Y - loc) / scale``. A group's sum of ``n`` rows moves by ``n * loc``."""
    label = as_observed(label).copy()
    shift = float(loc) * group_sizes(label)
    for name in ("lower", "upper", "truncate_lower", "truncate_upper"):
        label[name] = (label[name] - shift) / float(scale)
    label["noise"] = label["noise"] / float(scale)
    return label


def log_likelihood(dist, lower, upper, truncate_lower=None, truncate_upper=None, noise=None):
    """Log-likelihood of each row's observation under ``dist``. A JAX function.

    ``lower == upper`` is an exact value and scores its log density (log
    mass for a count); otherwise the log probability of the closed interval
    ``[lower, upper]``. With ``truncate_lower`` or ``truncate_upper``, the
    log probability of the truncation range is subtracted. With ``noise``,
    ``dist`` is first widened by normal noise of that standard deviation.
    Arguments broadcast against ``dist.batch_shape``.
    """
    if noise is not None:
        dist = dist.convolve_normal(jnp.asarray(noise, dtype=jnp.float32))
    lower = jnp.asarray(lower, dtype=jnp.float32)
    upper = jnp.asarray(upper, dtype=jnp.float32)
    exact_row = lower == upper
    point = dist.log_prob(jnp.where(exact_row, lower, 0.0))
    coarse = dist.log_interval(jnp.where(exact_row, -jnp.inf, lower), jnp.where(exact_row, jnp.inf, upper))
    result = jnp.where(exact_row, point, coarse)
    if truncate_lower is not None or truncate_upper is not None:
        low = -jnp.inf if truncate_lower is None else jnp.asarray(truncate_lower, dtype=jnp.float32)
        high = jnp.inf if truncate_upper is None else jnp.asarray(truncate_upper, dtype=jnp.float32)
        result = result - dist.log_interval(low, high)
    return result


def randomized_pit(dist, lower, upper, key):
    """A probability integral transform for each observation, uniform when the model is calibrated.

    Uniform between ``P(Y < lower)`` and ``P(Y <= upper)``: the CDF value
    itself for an exact value, spread over the jump at a count, and spread
    over everything the label allows for a censored or interval one. It is
    uniform when the censoring is independent of the value given the
    features. A JAX function.
    """
    low, high = dist.cdf_interval(lower, upper)
    return low + jax.random.uniform(key, jnp.shape(high), dtype=high.dtype) * (high - low)
