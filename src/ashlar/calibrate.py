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
"""

from __future__ import annotations

import math

import numpy as np

from ashlar.errors import AshlarError, ShapeError

MIN_CALIBRATION_ROWS = 20


def pit_values(draws: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Randomized PIT of ``y`` under ``draws`` of shape ``(n, rows)``. One value per row in ``(0, 1)``.

    ``y`` is ranked among the ``n`` draws with ties broken at random, and the
    rank is spread uniformly over its ``1 / (n + 1)`` slot. When ``y`` is a
    draw from the same distribution the result is exactly uniform, so the
    scores carry no grid from the finite number of draws.
    """
    draws = np.asarray(draws, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if draws.ndim != 2 or draws.shape[1] != y.shape[0]:
        raise ShapeError(f"draws must have shape (n, {y.shape[0]}); received {draws.shape}.")
    n = draws.shape[0]
    below = np.sum(draws < y[None, :], axis=0)
    tied = np.sum(draws == y[None, :], axis=0)
    rng = np.random.default_rng(int(seed))
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
