"""Split-conformal calibration of a fitted distribution head.

A density head is calibrated in probability space. Each held-out row gives its
probability integral transform ``u = F(y | x)``, read from draws of the
predictive distribution and randomized on ties so a discrete head still gives a
uniform ``u``. A central interval at nominal level ``c`` covers ``y`` exactly
when ``|2u - 1| <= c``, so the conformal quantile of those scores is the
nominal level that covers the requested fraction of new rows. This is
distributional conformal prediction (Chernozhukov, Wüthrich, and Zhu, 2021).
The same PIT values recalibrate a single quantile level: the level ``t`` maps
to the ``t``-quantile of the PIT values (Kuleshov, Fenner, and Ermon, 2018).

The quantile head has three levels and no draws, so it is calibrated by
conformalized quantile regression (Romano, Patterson, and Candès, 2019) on its
0.1 and 0.9 levels: the score is how far ``y`` falls outside that interval.

Marginal coverage holds when the calibration rows are exchangeable with the
rows you predict, and only then. Rows used to fit the model are not.

A joint head is calibrated column by column, and on chosen functions of its
columns, such as their sum. Each is a univariate PIT table of its own: a
column's PIT under its marginal, and a function's PIT under the distribution
of that function of the joint outcome. That distribution is exact for a
weighted sum under the normal-based joint heads (``Distribution.project``) and
read from joint draws otherwise. Calibrating a function of the joint outcome
is the same distributional conformal step, applied to that one number.

``recalibrated_cdf`` turns a model's own ``P(Y <= t)`` into the share of
calibration rows whose PIT was at most that much, so an exceedance
probability can be recalibrated as well as an interval.
"""

from __future__ import annotations

import json
import math
import numbers

import numpy as np

from ashlar.errors import AshlarError, ShapeError

MIN_CALIBRATION_ROWS = 20
# The function of a joint outcome every joint calibration includes.
SUM = "sum"


def pit_values(draws: np.ndarray, y: np.ndarray, seed: int = 0, rng=None) -> np.ndarray:
    """Randomized PIT of ``y`` under ``draws`` of shape ``(n, rows)``. One value per row in ``(0, 1)``.

    ``y`` is ranked among the ``n`` draws with ties broken at random, and the
    rank is spread uniformly over its ``1 / (n + 1)`` slot. When ``y`` is a
    draw from the same distribution the result is exactly uniform, so the
    scores carry no grid from the finite number of draws. ``rng``, a NumPy
    generator, replaces ``seed`` when several tables are drawn in turn.
    """
    draws = np.asarray(draws, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if draws.ndim != 2 or draws.shape[1] != y.shape[0]:
        raise ShapeError(f"draws must have shape (n, {y.shape[0]}); received {draws.shape}.")
    n = draws.shape[0]
    below = np.sum(draws < y[None, :], axis=0)
    tied = np.sum(draws == y[None, :], axis=0)
    rng = np.random.default_rng(int(seed)) if rng is None else rng
    rank = below + np.floor(rng.uniform(size=y.shape[0]) * (tied + 1))
    return (rank + rng.uniform(size=y.shape[0])) / (n + 1)


def conformal_quantile(scores, coverage: float) -> float:
    """The ``ceil((n + 1) c)``-th smallest score. ``inf`` when ``n`` is too small for ``c``."""
    scores = np.sort(np.asarray(scores, dtype=np.float64).reshape(-1))
    coverage = check_coverage(coverage)
    rank = math.ceil((scores.size + 1) * coverage)
    if rank > scores.size:
        return math.inf
    return float(scores[rank - 1])


def check_coverage(coverage) -> float:
    value = float(coverage)
    if not 0.0 < value < 1.0:
        raise AshlarError(f"coverage must be strictly between 0 and 1; received {coverage!r}.")
    return value


def check_rows(n_rows: int) -> None:
    if int(n_rows) < MIN_CALIBRATION_ROWS:
        raise AshlarError(
            f"calibration needs at least {MIN_CALIBRATION_ROWS} held-out rows; received {int(n_rows)}."
        )


def pit_calibration(pit) -> dict:
    """Saved calibration of a density head: its sorted PIT values."""
    values = np.sort(np.clip(np.asarray(pit, dtype=np.float64).reshape(-1), 0.0, 1.0))
    check_rows(values.size)
    return {"kind": "pit", "values": [float(value) for value in values]}


def cqr_calibration(low, high, y) -> dict:
    """Saved calibration of the quantile head: sorted CQR scores on the scale of ``y``."""
    low = np.asarray(low, dtype=np.float64).reshape(-1)
    high = np.asarray(high, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    check_rows(y.size)
    scores = np.sort(np.maximum(low - y, y - high))
    return {"kind": "cqr", "values": [float(value) for value in scores]}


def draw_interval(draws: np.ndarray, low: float, high: float) -> np.ndarray:
    """Interval per row from ``draws`` of shape ``(n, rows)``, shape ``(rows, 2)``.

    The ends are the order statistics that bracket the levels in the PIT of
    ``pit_values``: a value whose PIT lies in ``[low, high]`` is always inside,
    so the PIT coverage the calibration measured carries over to the interval.
    A level outside what ``n`` draws resolve takes the smallest or largest draw.
    """
    ordered = np.sort(np.asarray(draws, dtype=np.float64), axis=0)
    n = ordered.shape[0]
    first = min(max(math.floor(float(low) * (n + 1)), 1), n)
    last = min(max(math.ceil(float(high) * (n + 1)), 1), n)
    return np.stack((ordered[first - 1], ordered[last - 1]), axis=1)


def interval_levels(calibration: dict, coverage: float) -> tuple[float, float]:
    """Nominal quantile levels of the calibrated central interval at ``coverage``.

    When too few rows were held out for ``coverage``, the levels are 0 and 1:
    the widest interval the draws give.
    """
    if calibration["kind"] != "pit":
        raise AshlarError("interval levels come from a density head's PIT calibration.")
    scores = np.abs(2.0 * np.asarray(calibration["values"], dtype=np.float64) - 1.0)
    nominal = min(conformal_quantile(scores, coverage), 1.0)
    return 0.5 * (1.0 - nominal), 0.5 * (1.0 + nominal)


def recalibrated_levels(calibration: dict, levels) -> tuple[float, ...]:
    """Each requested level mapped to the nominal level that held that fraction of calibration rows."""
    if calibration["kind"] != "pit":
        raise AshlarError("recalibrated levels come from a density head's PIT calibration.")
    values = np.asarray(calibration["values"], dtype=np.float64)
    return tuple(float(np.quantile(values, float(level))) for level in levels)


def cqr_margin(calibration: dict, coverage: float) -> float:
    if calibration["kind"] != "cqr":
        raise AshlarError("a CQR margin comes from the quantile head's calibration.")
    return conformal_quantile(calibration["values"], coverage)


def draw_quantiles(draws: np.ndarray, levels) -> np.ndarray:
    """The order statistic of ``draws`` ``(n, rows)`` at each level, shape ``(rows, len(levels))``.

    The ``ceil(level (n + 1))``-th smallest draw, the value whose randomized
    PIT among the draws (``pit_values``) brackets ``level``.
    """
    ordered = np.sort(np.asarray(draws, dtype=np.float64), axis=0)
    n = ordered.shape[0]
    index = [min(max(math.ceil(float(level) * (n + 1)), 1), n) - 1 for level in np.atleast_1d(levels)]
    return ordered[index].T


def recalibrated_cdf(calibration: dict, probability) -> np.ndarray:
    """``P(Y <= t)`` after calibration, from the model's own ``F(t)``.

    The share of calibration rows whose PIT was at most ``F(t)``: the
    empirical CDF of the PIT values, joined linearly from ``(0, 0)`` to
    ``(1, 1)``, so a model that was right keeps its probabilities, an
    overconfident one has them pulled toward the middle, and a probability
    strictly between 0 and 1 stays there (Kuleshov, Fenner, and Ermon, 2018).
    """
    if calibration["kind"] != "pit":
        raise AshlarError("a recalibrated probability comes from a density head's PIT calibration.")
    values = np.sort(np.asarray(calibration["values"], dtype=np.float64))
    unique, counts = np.unique(values, return_counts=True)
    xs = np.concatenate([[0.0], unique, [1.0]])
    ys = np.concatenate([[0.0], np.cumsum(counts) / (values.size + 1.0), [1.0]])
    keep = np.concatenate([np.diff(xs) > 0.0, [True]])
    return np.interp(np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0), xs[keep], ys[keep])


def target_spec(target, n_columns: int) -> tuple[str, dict]:
    """``(name, spec)`` of one function of a joint outcome's ``n_columns`` columns.

    - ``"sum"``: the total, ``{"weights": ones}``.
    - a weight per column: that weighted sum, named by its weights.
    - a function of the last axis, ``f(y)`` with ``y`` shaped ``(..., columns)``
      and returning ``(...)``: named by its ``__name__``, which is how a saved
      calibration finds it again, so it must be a ``def``, not a ``lambda``.
    - any other string: the name of a function calibrated earlier, ``{"name": ...}``.
    """
    if isinstance(target, str):
        if target == SUM:
            return SUM, {"weights": np.ones(int(n_columns))}
        return target, {"name": target}
    if callable(target):
        name = getattr(target, "__name__", None)
        if not name or name == "<lambda>":
            raise AshlarError(
                "a function target is known by its name, which a saved calibration keeps; "
                "define it with def instead of lambda."
            )
        if name == SUM:
            raise AshlarError("'sum' names the built-in total; give your function another name.")
        return name, {"function": target}
    weights = np.asarray(target, dtype=np.float64)
    if weights.shape != (int(n_columns),) or not np.isfinite(weights).all() or not np.any(weights):
        raise AshlarError(
            f"a target is 'sum', a named function, or one finite weight per column ({int(n_columns)}), "
            f"not all zero; received {target!r}."
        )
    return weights_name(weights), {"weights": weights}


def weights_name(weights) -> str:
    """The name a weighted sum's calibration is saved under: its weights."""
    return "weights " + json.dumps([float(value) for value in np.asarray(weights, dtype=np.float64).reshape(-1)])


def joint_targets(target, n_columns: int) -> dict:
    """``{name: spec}`` of every function a joint calibration covers: the sum, then each of ``target``.

    ``target`` is ``None``, one target (see ``target_spec``), or a list of
    them. A list of plain numbers is one weighted sum.
    """
    if target is None:
        items = []
    elif isinstance(target, (list, tuple)) and target and not all(isinstance(value, numbers.Real) for value in target):
        items = list(target)
    else:
        items = [target]
    specs = {SUM: {"weights": np.ones(int(n_columns))}}
    for item in items:
        name, spec = target_spec(item, n_columns)
        if "name" in spec:
            raise AshlarError(f"calibrate needs the function itself for {name!r}, not its name.")
        specs[name] = spec
    return specs


def joint_calibration(columns, targets: dict) -> dict:
    """Saved calibration of a joint head: a PIT table per column and per function of the columns.

    ``columns`` is ``(rows, columns)`` of PIT values; ``targets`` maps each
    function's name to ``(spec, pit)``.
    """
    columns = np.asarray(columns, dtype=np.float64)
    tables = {}
    for name, (spec, pit) in targets.items():
        table = pit_calibration(np.clip(pit, 0.0, 1.0))
        if "weights" in spec:
            table["weights"] = [float(value) for value in np.asarray(spec["weights"]).reshape(-1)]
        else:
            table["function"] = name
        tables[name] = table
    return {
        "kind": "joint",
        "columns": [pit_calibration(np.clip(columns[:, index], 0.0, 1.0)) for index in range(columns.shape[1])],
        "targets": tables,
    }
