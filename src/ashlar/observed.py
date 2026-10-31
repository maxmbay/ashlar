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
``noise`` (0 for none), ``group`` (``-1`` for a row that is its own
observation), and ``group_size``, the rows the group's total covers, so a
split that cuts a group is caught instead of scoring part of a sum against
the whole total.

A joint target, ``y`` with one column per outcome, takes a label with one
record per row and column, shaped ``(rows, columns)``: ``exact``,
``interval``, ``censored``, and ``rounded`` take tables as well as single
columns, and ``stack`` puts one-column labels side by side. Each cell is
exact or an interval, so a row of several products' sales can be exact for
some and censored for the ones that sold out. Truncation, noise, and group
totals are for a one-column label.

The JAX functions ``log_likelihood`` and ``randomized_pit`` take any
``ashlar.distributions.Distribution``, so a custom objective can train on
the same labels: the training ``y`` it receives is the ``(rows, 6)`` array of
``fields(label)``, or ``(rows, columns, 6)`` for a joint label.
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
        ("group_size", np.int64),
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


def _cells(values, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """One value per row, or per row and column for a joint target; with ``shape``, broadcast to it."""
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise AshlarError(f"{name} must be numeric.") from exc
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if shape is None:
        if array.ndim not in (1, 2):
            raise ShapeError(
                f"{name} must be one value per row, or one per row and column for a joint target; received shape {array.shape}."
            )
        return array
    try:
        return np.array(np.broadcast_to(array, shape), dtype=np.float64)
    except ValueError as exc:
        raise ShapeError(f"{name} has shape {array.shape}; expected {shape}.") from exc


def _flags(values, name: str, shape) -> np.ndarray:
    shape = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
    array = np.asarray(values)
    if array.ndim == 2 and array.shape[1] == 1 and len(shape) == 1:
        array = array[:, 0]
    if array.shape != shape:
        unit = "one flag per row" if len(shape) == 1 else "one flag per row and column"
        raise ShapeError(f"{name} must be {unit}, shape {shape}; received shape {array.shape}.")
    if array.dtype != bool:
        if not np.isin(array, (0, 1)).all():
            raise AshlarError(f"{name} must hold booleans or 0 and 1.")
        array = array.astype(bool)
    return array


def _one_column(label, what: str) -> None:
    if np.ndim(label) != 1:
        raise AshlarError(f"{what} is for a label of one column; a joint label takes exact, interval, censored, and rounded cells.")


def _new(lower, upper) -> np.ndarray:
    label = np.empty(np.shape(lower), dtype=DTYPE)
    label["lower"] = lower
    label["upper"] = upper
    label["truncate_lower"] = -np.inf
    label["truncate_upper"] = np.inf
    label["noise"] = 0.0
    label["group"] = -1
    label["group_size"] = 1
    return check(label)


def exact(values) -> np.ndarray:
    """Each row's value, seen exactly. A ``(rows, columns)`` table makes a joint label."""
    values = _cells(values, "values")
    return _new(values, values)


def interval(lower, upper) -> np.ndarray:
    """Each row's value lies in the closed interval ``[lower, upper]``. ``-inf`` or ``inf`` leaves a side open.

    Tables of bounds make a joint label, one interval per row and column.
    """
    lower = _cells(lower, "lower")
    return _new(lower, _cells(upper, "upper", lower.shape))


def censored(values, censored, side: str = "right") -> np.ndarray:
    """Exact values, except where ``censored`` is set: there only ``Y >= value`` (``side="right"``) or ``Y <= value`` is known.

    For a count such as units sold out of a stock of 12, ``Y >= 12``
    includes 12 itself, as it should. A ``(rows, columns)`` table of values
    and flags makes a joint label, censored cell by cell.
    """
    values = _cells(values, "values")
    flag = _flags(censored, "censored", values.shape)
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
    time = _cells(time, "time")
    label = censored(time, ~_flags(event, "event", time.shape))
    if entry is None:
        return label
    return truncated(label, lower=_column(entry, "entry", len(time)))


def rounded(values, step) -> np.ndarray:
    """``values`` are the true values rounded to the nearest multiple of ``step``: each lies within ``step / 2``.

    For counts, write the whole numbers each reported value stands for
    with ``interval``, since rounding a count leaves a closed range of integers.
    For a joint label, ``step`` can be one number, one per column, or one per cell.
    """
    values = _cells(values, "values")
    step = _cells(step, "step", values.shape)
    if not np.isfinite(step).all() or (step <= 0).any():
        raise AshlarError("step must be positive and finite.")
    return _new(values - 0.5 * step, values + 0.5 * step)


def truncated(label, lower=-np.inf, upper=np.inf) -> np.ndarray:
    """Rows were recorded only when the value fell inside ``[lower, upper]``; plain values are read as exact."""
    label = as_observed(label).copy()
    _one_column(label, "truncation")
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
    _one_column(label, "noise")
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
    sum of their rows has a closed form: the rows are independent given the
    features, so a sum of normals is normal and a sum of Poisson counts is
    Poisson. ``fit`` keeps each group in one minibatch and on one side of
    its validation split, and a group's weight is the mean of its rows'.
    """
    label = as_observed(label).copy()
    _one_column(label, "a group total")
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
    label["group_size"] = _rows_per_group(ids)
    return check(label)


def total(values, group) -> np.ndarray:
    """Each group's total, seen exactly. ``values`` holds the group's total on every one of its rows."""
    return grouped(exact(values), group)


def concatenate(labels) -> np.ndarray:
    """Stack labels row-wise. Plain values are read as exact.

    Group ids are renumbered so that groups from different labels stay apart.
    Joint labels stack too, when they have the same columns.
    """
    parts = []
    offset = 0
    for label in labels:
        part = as_observed(label).copy()
        members = part["group"] >= 0
        if members.any():
            part["group"][members] += offset
            offset = int(part["group"][members].max()) + 1
        parts.append(part)
    if len({part.shape[1:] for part in parts}) > 1:
        raise ShapeError(f"labels to concatenate need the same columns; received shapes {[part.shape for part in parts]}.")
    return check(np.concatenate(parts))


def stack(labels) -> np.ndarray:
    """A joint label from one-column labels, one per target column, shape ``(rows, columns)``.

    ``stack([observed.censored(sales_a, sold_out_a), observed.censored(sales_b, sold_out_b)])``
    is two products' sales, each censored on the days it sold out. Plain
    values are read as exact. The joint heads train on it.
    """
    parts = [as_observed(label) for label in labels]
    if not parts:
        raise AshlarError("stack needs at least one label.")
    for part in parts:
        _one_column(part, "stack")
        if is_truncated(part) or is_noisy(part) or has_groups(part):
            raise AshlarError(
                "a joint label takes exact, interval, censored, and rounded cells; truncation, noise, and group totals are for a one-column label."
            )
    if len({len(part) for part in parts}) > 1:
        raise ShapeError(f"labels to stack need the same rows; received {[len(part) for part in parts]}.")
    return check(np.stack(parts, axis=1))


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
            names = array.dtype.names
            if "group_size" in names:
                converted["group_size"] = array["group_size"]
            else:
                converted["group_size"] = _rows_per_group(array["group"]) if array.ndim == 1 else 1
            array = converted
        if array.ndim not in (1, 2):
            raise ShapeError(f"an observed label is one record per row, or per row and column; received shape {array.shape}.")
        return array
    return exact(y)


def check(label) -> np.ndarray:
    """Raise ``AshlarError`` unless every record describes a possible observation. Returns the label."""
    if not is_observed(label):
        raise AshlarError("expected a label from ashlar.observed.")
    label = np.asarray(label)
    if label.ndim not in (1, 2):
        raise ShapeError(f"an observed label is one record per row, or per row and column; received shape {label.shape}.")
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
    if label.ndim == 2:
        if np.isfinite(t_lower).any() or np.isfinite(t_upper).any() or (noise > 0).any() or (group >= 0).any():
            raise AshlarError(
                "a joint label takes exact, interval, censored, and rounded cells; truncation, noise, and group totals are for a one-column label."
            )
        if "group_size" in label.dtype.names and (label["group_size"] != 1).any():
            raise AshlarError("group_size is 1 for a row of its own and at least 1 in a group.")
        return label
    members = group >= 0
    size = label["group_size"] if "group_size" in label.dtype.names else _rows_per_group(group)
    if (size[~members] != 1).any() or (size < 1).any():
        raise AshlarError("group_size is 1 for a row of its own and at least 1 in a group.")
    if members.any():
        ids, first = np.unique(group[members], return_index=True)
        rows = np.flatnonzero(members)
        lead = rows[first][np.searchsorted(ids, group[members])]
        for name in (*FIELDS[:5], "group_size"):
            values = size if name == "group_size" else label[name]
            if not np.array_equal(values[members], values[lead]):
                raise AshlarError(f"every row of a group carries the group's observation; {name} differs within a group.")
        if (_rows_per_group(group)[members] > size[members]).any():
            raise AshlarError("a group has more rows than its total covers; two groups share an id.")
    return label


def _rows_per_group(group) -> np.ndarray:
    """Rows present in each row's group, 1 for a row of its own."""
    group = np.asarray(group)
    sizes = np.ones(len(group), dtype=np.int64)
    members = group >= 0
    if members.any():
        _ids, inverse, counts = np.unique(group[members], return_inverse=True, return_counts=True)
        sizes[members] = counts[inverse]
    return sizes


def check_whole(label) -> np.ndarray:
    """Raise ``AshlarError`` unless every group has all the rows its total covers. Returns the label."""
    label = as_observed(label)
    present = _rows_per_group(label["group"])
    cut = present < label["group_size"]
    if cut.any():
        row = int(np.flatnonzero(cut)[0])
        raise AshlarError(
            f"group {int(label['group'][row])} covers {int(label['group_size'][row])} rows and only {int(present[row])} are here, "
            "so its total would be scored against part of its sum. Keep each group's rows together, "
            "for example with GroupKFold on the group ids."
        )
    return label


def fields(label) -> np.ndarray:
    """The ``(rows, 6)`` float array of ``lower, upper, truncate_lower, truncate_upper, noise, group``.

    ``(rows, columns, 6)`` for a joint label.
    """
    label = as_observed(label)
    return np.stack([np.asarray(label[name], dtype=np.float64) for name in FIELDS], axis=-1)


def group_sizes(label) -> np.ndarray:
    """Rows each row's group total covers, 1 for a row with no group."""
    return np.asarray(as_observed(label)["group_size"], dtype=np.int64)


def has_groups(label) -> bool:
    return bool((as_observed(label)["group"] >= 0).any())


def contiguous_order(label) -> np.ndarray:
    """A row order that puts the rows of each group next to each other.

    Observations keep the order of their first row, and the rows of a group
    keep their own order.
    """
    group = as_observed(label)["group"]
    key = np.where(group >= 0, group, -1 - np.arange(len(group)))
    _keys, first, inverse = np.unique(key, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first, kind="stable")] = np.arange(len(first))
    return np.argsort(rank[inverse], kind="stable")


def units(label) -> tuple[np.ndarray, np.ndarray]:
    """The first row and the row count of each observation: a group, or a row of its own.

    The rows of each group must be adjacent, as ``contiguous_order`` leaves them.
    """
    group = as_observed(label)["group"]
    n = len(group)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    starts = np.ones(n, dtype=bool)
    starts[1:] = (group[1:] != group[:-1]) | (group[1:] < 0)
    first = np.flatnonzero(starts)
    ids = group[first]
    ids = ids[ids >= 0]
    if len(np.unique(ids)) != len(ids):
        raise AshlarError("the rows of each group must be adjacent; order them with ashlar.observed.contiguous_order.")
    return first.astype(np.int64), np.diff(np.append(first, n)).astype(np.int64)


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
    value = np.full(label.shape, np.nan)
    both = finite_lower & finite_upper
    value[both] = 0.5 * (lower[both] + upper[both])
    value[finite_lower & ~finite_upper] = lower[finite_lower & ~finite_upper]
    value[~finite_lower & finite_upper] = upper[~finite_lower & finite_upper]
    return value / group_sizes(label)


def affine(label, loc, scale) -> np.ndarray:
    """The label of ``(Y - loc) / scale``. A group's sum of ``n`` rows moves by ``n * loc``.

    For a joint label, ``loc`` and ``scale`` can be one number per column.
    """
    label = as_observed(label).copy()
    loc, scale = np.asarray(loc, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    shift = loc * group_sizes(label)
    for name in ("lower", "upper", "truncate_lower", "truncate_upper"):
        label[name] = (label[name] - shift) / scale
    label["noise"] = label["noise"] / scale
    return label


def log_likelihood(dist, lower, upper, truncate_lower=None, truncate_upper=None, noise=None, points=None):
    """Log-likelihood of each row's observation under ``dist``. A JAX function.

    ``lower == upper`` is an exact value and scores its log density (log
    mass for a count); otherwise the log probability of the closed interval
    ``[lower, upper]``. With ``truncate_lower`` or ``truncate_upper``, the
    log probability of the truncation range is subtracted. With ``noise``,
    ``dist`` is first widened by normal noise of that standard deviation.
    Arguments broadcast against ``dist.batch_shape``.

    For a joint ``dist``, the bounds have a last axis of columns and the
    result is ``dist.log_box(lower, upper, points)``: the density of the
    exact cells times the probability of the coarse ones given them.
    """
    if dist.event_shape:
        if truncate_lower is not None or truncate_upper is not None or noise is not None:
            raise NotImplementedError("a joint label takes exact and interval cells; truncation and noise are for one column.")
        return dist.log_box(jnp.asarray(lower, dtype=jnp.float32), jnp.asarray(upper, dtype=jnp.float32), points)
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
