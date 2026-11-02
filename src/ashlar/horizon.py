"""Forecasting as a table: each forecast origin is a row, and the horizon is a joint outcome.

``horizon_frame(data, target=..., lookback=..., horizon=...)`` turns a long
table of one or many series into rows any ``ASRRegressor`` can fit. Each row
is one series at one origin, the last step it has seen. Its features are the
target's last ``lookback`` values, the same lags of any ``past`` covariates
(known only once they happen, such as the number of customers), the values of
the ``known`` covariates at each step ahead (promotions, holidays, prices,
the calendar), and the series' ``static`` columns. Its outcome is the next
``horizon`` values of the target, one column per step.

A joint head then forecasts the whole path at once, without feeding draws back:
``multivariate_gaussian``, ``multivariate_mixture``, or ``multivariate_flow``
gives the joint distribution of the steps, with an exact log density, so its
draws are paths with the right dependence between steps. A function of the
path, such as the peak over the horizon, the total, or the step at which the
cumulative demand passes the stock, is read from those draws, and
``calibrate(X, Y, target=fn)`` calibrates it on held-out origins (split
conformal on the function's probability integral transform), so
``predict_interval(X, 0.8, target=fn)`` and ``predict_exceedance(X, t,
target=fn)`` hold on new origins exchangeable with those.

``ASRForecaster`` and ``ASRHistoryForecaster`` remain small one-series models
with per-step marginals; this module is the route for covariates, many
series, and paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ashlar.errors import AshlarError

SCALES = ("mean",)


@dataclass(frozen=True)
class HorizonFrame:
    """The rows of ``horizon_frame``.

    ``X`` is a DataFrame with a row per series and origin. ``Y`` is
    ``(rows, horizon)``, the target at each step ahead, ``NaN`` past the end
    of the data. ``index`` has the series (``series``), the origin's time or
    position (``origin``), and the divisor the target was scaled by
    (``scale``, 1 without ``scale``). ``target_columns`` names the columns of
    ``Y``.
    """

    X: pd.DataFrame
    Y: np.ndarray
    index: pd.DataFrame
    target_columns: tuple[str, ...]

    def unscale(self, values) -> np.ndarray:
        """``values`` in the target's own units: times each row's scale on the row axis.

        ``values`` is ``(rows, ...)`` or, for draws, ``(draws, rows, ...)``.
        """
        values = np.asarray(values, dtype=np.float64)
        scale = self.index["scale"].to_numpy(np.float64)
        rows = len(scale)
        if values.ndim >= 1 and values.shape[0] == rows:
            return values * scale.reshape((rows,) + (1,) * (values.ndim - 1))
        if values.ndim >= 2 and values.shape[1] == rows:
            return values * scale.reshape((1, rows) + (1,) * (values.ndim - 2))
        raise AshlarError(f"values must have the {rows} rows on their first or second axis; received {values.shape}.")


def _column(part: pd.DataFrame, name: str) -> np.ndarray:
    values = part[name].to_numpy()
    if values.dtype == object or not np.issubdtype(values.dtype, np.number) or np.issubdtype(values.dtype, np.bool_):
        return values.astype(object)
    return values.astype(np.float64)


def _shift(values: np.ndarray, positions: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """``values[positions + offset]`` per row and offset, missing where that falls outside the series."""
    index = positions[:, None] + offsets[None, :]
    inside = (index >= 0) & (index < len(values))
    taken = values[np.clip(index, 0, len(values) - 1)]
    if values.dtype == object:
        out = np.empty(taken.shape, dtype=object)
        out[...] = taken
        out[~inside] = None
        return out
    return np.where(inside, taken, np.nan)


def horizon_frame(
    data,
    *,
    target: str,
    lookback: int,
    horizon: int,
    series: str | None = None,
    time: str | None = None,
    known=(),
    past=(),
    static=(),
    stride: int = 1,
    start=None,
    end=None,
    complete: bool = True,
    scale: str | None = None,
) -> HorizonFrame:
    """One row per series and forecast origin, the next ``horizon`` target values its outcome.

    ``data`` is a long table, one row per series and step: a pandas
    DataFrame or anything ``pd.DataFrame`` takes. ``series`` names the
    column that tells the series apart (none: one series), and ``time`` the
    column to order each series by (none: the rows' order). Steps are rows:
    a series with gaps should be reindexed first. An origin is a step with
    ``lookback`` steps of history up to and including it, taken every
    ``stride`` steps, and only between ``start`` and ``end`` (on ``time``,
    or positions without it), both included. ``complete=True`` keeps only
    origins with all ``horizon`` steps ahead observed, in the table and not
    missing, as for training and scoring; ``False`` also keeps the last
    origins, with ``NaN`` in ``Y`` past the data or where the target is
    missing, as for forecasting rows whose known covariates are in the table
    but whose target is not. A missing value in the lookback stays ``NaN``,
    which the model reads as a missing feature.

    Feature columns: ``{target}_lag{j}`` for ``j = 1 .. lookback``, ``lag1``
    being the value at the origin; ``{column}_lag{j}`` for each ``past``
    column; ``{column}_step{h}`` for each ``known`` column and ``h = 1 ..
    horizon``, its value at that step ahead; and each ``static`` column, read
    from the series' first row. ``scale="mean"`` divides the target's lags
    and ``Y`` by the mean absolute value of the row's lags (1 where that is
    0), adds their log as ``{target}_log_scale``, and keeps the divisor in
    ``index["scale"]``, so one model can read series of very different sizes;
    ``unscale`` puts draws back in the target's units.
    """
    lookback, horizon, stride = int(lookback), int(horizon), int(stride)
    if lookback < 1 or horizon < 1 or stride < 1:
        raise AshlarError("lookback, horizon, and stride must be positive.")
    if scale is not None and scale not in SCALES:
        raise AshlarError(f"scale must be None or one of {SCALES}; received {scale!r}.")
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    known, past, static = list(known), list(past), list(static)
    needed = [target, *known, *past, *static] + [name for name in (series, time) if name is not None]
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise AshlarError(f"columns {missing} are not in data.")
    overlap = set(known) & set(past) | ({target} & set(known + past + static))
    if overlap:
        raise AshlarError(f"columns {sorted(overlap)} are listed twice among the target, known, past, and static columns.")
    groups = [(None, frame)] if series is None else frame.groupby(series, sort=False)
    lags = np.arange(0, -lookback, -1)
    steps = np.arange(1, horizon + 1)
    pieces, outcomes, indices = [], [], []
    for key, part in groups:
        if time is not None:
            part = part.sort_values(time, kind="stable")
        part = part.reset_index(drop=True)
        n = len(part)
        last = n - 1 - horizon if complete else n - 1
        positions = np.arange(lookback - 1, last + 1, stride)
        clock = part[time].to_numpy() if time is not None else np.arange(n)
        if positions.size and start is not None:
            positions = positions[clock[positions] >= start]
        if positions.size and end is not None:
            positions = positions[clock[positions] <= end]
        if positions.size == 0:
            continue
        values = part[target].to_numpy(np.float64)
        future = _shift(values, positions, steps)
        if complete:
            observed = ~np.isnan(future).any(axis=1)
            positions, future = positions[observed], future[observed]
            if positions.size == 0:
                continue
        history = _shift(values, positions, lags)
        divisor = np.ones(positions.size)
        if scale == "mean":
            seen = ~np.isnan(history)
            level = np.where(seen, np.abs(history), 0.0).sum(axis=1) / np.maximum(seen.sum(axis=1), 1)
            divisor = np.where(level > 0, level, 1.0)
            history = history / divisor[:, None]
            future = future / divisor[:, None]
        columns = {f"{target}_lag{j}": history[:, j - 1] for j in range(1, lookback + 1)}
        if scale == "mean":
            columns[f"{target}_log_scale"] = np.log(divisor)
        for name in past:
            shifted = _shift(_column(part, name), positions, lags)
            columns.update({f"{name}_lag{j}": shifted[:, j - 1] for j in range(1, lookback + 1)})
        for name in known:
            shifted = _shift(_column(part, name), positions, steps)
            columns.update({f"{name}_step{h}": shifted[:, h - 1] for h in range(1, horizon + 1)})
        for name in static:
            value = part[name].iloc[0]
            columns[name] = np.full(positions.size, value, dtype=object if isinstance(value, str) else None)
        pieces.append(pd.DataFrame(columns))
        outcomes.append(future)
        indices.append(pd.DataFrame({"series": key, "origin": clock[positions], "scale": divisor}))
    if not pieces:
        raise AshlarError("no series has an origin with the lookback, horizon, and bounds asked for.")
    X = pd.concat(pieces, ignore_index=True)
    return HorizonFrame(
        X=X,
        Y=np.concatenate(outcomes, axis=0),
        index=pd.concat(indices, ignore_index=True),
        target_columns=tuple(f"{target}_step{h}" for h in range(1, horizon + 1)),
    )


def peak(y):
    """The largest step of a path: ``y`` is ``(..., steps)``. A ``target`` for ``calibrate``."""
    return y.max(axis=-1)


def stockout(stock: float, name: str | None = None):
    """A path function: the first step at which the cumulative total passes ``stock``, ``steps + 1`` if none does.

    ``model.calibrate(X, Y, target=stockout(500.0))`` and then
    ``predict_exceedance(X, h, target=...)`` is the probability that the
    stock lasts beyond step ``h``.
    """
    stock = float(stock)

    def stockout_step(y):
        import jax.numpy as jnp

        total = jnp.cumsum(jnp.asarray(y), axis=-1)
        passed = total > stock
        first = jnp.argmax(passed, axis=-1) + 1
        return jnp.where(jnp.any(passed, axis=-1), first, total.shape[-1] + 1).astype(total.dtype)

    stockout_step.__name__ = name or f"stockout_{stock:g}"
    return stockout_step
