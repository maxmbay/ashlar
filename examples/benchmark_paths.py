"""Forecast paths with covariates: Rossmann daily sales, 12 days ahead, from a lag frame and a joint head.

The cheap test of #43. Each row is one store at one forecast origin
(``ashlar.horizon.horizon_frame``): the last 28 days of ``log1p`` sales, and
for each of the next 12 days whether the store is open, runs a promotion,
or has a state or school holiday, and its weekday; the store's type,
assortment, competition distance, and long-running promotion are static.
The outcome is the next 12 days of ``log1p`` sales, a joint outcome. Closed
days stay in, with no sales, so a path is what the store sells day by day.

Splits by time, with no target day shared between them: origins up to 16
February 2015 train (every third day), origins from 28 February to 19 May
2015 calibrate (every third day), and every origin from 31 May 2015 to the
end of the data (19 July, so the 12 days fit) is scored, on a random sample
of ``stores`` stores (300 by default, a new sample per seed).

Models:

- ``joint``: ``multivariate_gaussian`` (rank 4) with the covariates.
- ``joint_marginals``: ``joint``'s own marginals with the dependence between
  days dropped (``Independent(dist.marginals())``), scored raw.
- ``independent``: ``multivariate_gaussian`` with ``rank=0``, a model with
  independent days, fitted and calibrated like the others.
- ``joint_flow``: ``multivariate_flow`` with the covariates.
- ``no_covariates``: the joint normal from the 28 lags alone.
- ``chronos2`` (stage ``chronos``, optional): Chronos-2 zero-shot with the
  known covariates (open, promotion, school and state holiday), from each
  store's whole history before the origin. It gives quantiles per day, not
  paths, so it is scored per day only.
- ``deepar`` (stage ``deepar``, optional): DeepAR from GluonTS, trained on
  the same days with the same covariates and the store's type and
  assortment for 500 epochs of 50 batches, whose autoregressive draws are
  paths, scored raw.

Draws of ``log1p`` sales are capped at twice the largest day of sales in the
training rows before they become sales. A normal on log sales can give a row
unlike any it trained on a spread so wide that its upper draws are billions a
day: store 876, which closed for refurbishment in July 2015 after its best
weeks, got a standard deviation of 31 on log sales on its first closed days,
and its rows alone decided the mean daily loss of the first run.

Per day, in sales: the mean quantile loss over levels 0.1 to 0.9 (twice the
pinball loss, an approximation of the CRPS), the median over row-days of that
loss, the mean without the cap, and the 80% interval's coverage.
Per path, in sales: the 12-day total and the peak day, each with its CRPS
(from 400 draws), its 80% interval's coverage and width, and the Brier score
and expected calibration error (10 bins) of an exceedance,
``P(total > 1.1 × the last 12 days' total)`` and ``P(peak > the largest day
of the last 28)``. The ashlar models' paths are read raw from the draws and
after ``calibrate(..., target=[total, peak_sales])`` on the calibration
origins.

Stage ``recalibrate`` asks which calibration carries over to the summer.
It fits ``joint``, ``independent``, and ``joint_flow`` again, reads the PIT
of every origin from 28 February on once (each day, the total, and the peak,
as ``calibrate`` stores them, from 1,000 draws), and scores the same
forecasts under each calibration in ``STRATEGIES``: the fit stage's spring
origins, alone or Mondrian by the calendar of the 12 days ahead (a public
holiday, else a school holiday, else neither) or by store type; the last 28
origins observed at the first forecast; at each origin, the last 28 origins
observed by then, alone or by calendar (rolling); and every origin observed
by then, weighted with a half-life of 14 days. A calibration's intervals,
quantiles, and exceedance probabilities are read as ``predict_interval``,
``predict_quantiles``, and ``predict_exceedance`` read them, from 1,000
draws of each path; the weighted one puts the forecast row's own weight at
infinity (Barber, Candès, Ramdas, and Tibshirani, 2023). It also saves each
row's PIT and its draws' order statistics, from which stage ``adaptive``
scores intervals whose level follows each quantity's own misses, as they
come in 12 days late (quantile tracking, Angelopoulos, Candès, and
Tibshirani, 2023), at several step sizes.

Data: the fev benchmark's ``rossmann/1D`` (``autogluon/fev_datasets`` on
Hugging Face), saved as ``$ASHLAR_BENCH/rossmann-1D.parquet``
(``/tmp/ashlar-bench`` by default).

    python examples/benchmark_paths.py fit seeds=0,1,2
    python examples/benchmark_paths.py chronos seeds=0,1,2   # needs chronos-forecasting
    python examples/benchmark_paths.py deepar seeds=0,1,2    # needs gluonts[torch]
    python examples/benchmark_paths.py recalibrate seeds=0,1,2
    python examples/benchmark_paths.py adaptive seeds=0,1,2
    python examples/benchmark_paths.py report seeds=0,1,2
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

from ashlar import ASRRegressor
from ashlar.calibrate import MIN_GROUP_ROWS
from ashlar.distributions import Independent
from ashlar.horizon import horizon_frame, peak

DATA = Path(os.environ.get("ASHLAR_BENCH", "/tmp/ashlar-bench"))
ROOT = DATA / "paths"
LOOKBACK = 28
HORIZON = 12
LEVELS = np.round(np.arange(0.1, 0.91, 0.1), 2)
N_DRAWS = 400
TRAIN_END = pd.Timestamp("2015-02-16")
CAL_START, CAL_END = pd.Timestamp("2015-02-28"), pd.Timestamp("2015-05-19")
TEST_START = pd.Timestamp("2015-05-31")
# Draws of log1p sales are capped before expm1 at this many times the largest day of sales in the training rows.
CAP_FACTOR = 2.0
# The uncapped scores only keep draws finite: about 485 million a day.
GUARD = 20.0
KNOWN = ["open", "promo", "state_holiday", "school_holiday", "weekday"]
STATIC = ["store_type", "assortment", "competition", "promo2"]
# DeepAR's training loss moved by under 1% over its last 20 epochs at 500, on every seed; at 50 it was still falling fast.
DEEPAR_EPOCHS = 500
# Numeric known covariates for Chronos-2 and DeepAR, which read the calendar themselves.
REFERENCE_COVARIATES = ["open", "promo", "school_holiday", "holiday"]
MODELS = {
    "joint": dict(task="multivariate_gaussian", head_kwargs={"rank": 4}),
    "independent": dict(task="multivariate_gaussian", head_kwargs={"rank": 0}),
    "joint_flow": dict(task="multivariate_flow"),
    "no_covariates": dict(task="multivariate_gaussian", head_kwargs={"rank": 4}),
}


def cap_of(train) -> float:
    """The cap on draws of ``log1p`` sales: ``CAP_FACTOR`` times the largest day of sales in the training rows."""
    return float(np.log1p(CAP_FACTOR * np.expm1(np.nanmax(train.Y))))


def path_targets(cap: float):
    """The path functions to calibrate, from ``log1p`` sales capped at ``cap``: the total and the peak day, in sales."""
    import jax.numpy as jnp

    def total(y):
        return jnp.sum(jnp.expm1(jnp.minimum(jnp.asarray(y), cap)), axis=-1)

    def peak_sales(y):
        return jnp.expm1(jnp.minimum(peak(jnp.asarray(y)), cap))

    return total, peak_sales


def long_table(stores: int, seed: int = 0) -> pd.DataFrame:
    """Every day of ``stores`` random Rossmann stores, closed days included, one row per store and day.

    A day a store has no row for becomes a row with its sales and covariates
    missing, so that steps are days.
    """
    raw = pd.read_parquet(DATA / "rossmann-1D.parquet")
    chosen = np.random.default_rng(seed).choice(len(raw), size=min(stores, len(raw)), replace=False)
    raw = raw.iloc[np.sort(chosen)]
    daily = ["timestamp", "DayOfWeek", "Sales", "Open", "Promo", "StateHoliday", "SchoolHoliday"]
    frame = raw.explode(daily, ignore_index=True)
    holiday = frame["StateHoliday"].astype(str)
    table = pd.DataFrame(
        {
            "store": frame["id"].astype(str).to_numpy(dtype=object),
            "day": pd.to_datetime(frame["timestamp"]),
            "sales": np.log1p(frame["Sales"].astype(float).clip(lower=0.0).to_numpy()),
            "open": frame["Open"].astype(float).to_numpy(),
            "promo": frame["Promo"].astype(float).to_numpy(),
            "state_holiday": holiday.to_numpy(dtype=object),
            "holiday": (~holiday.isin(["0", "0.0"])).astype(float).to_numpy(),
            "school_holiday": frame["SchoolHoliday"].astype(float).to_numpy(),
            "weekday": frame["DayOfWeek"].astype(float).astype(int).astype(str).to_numpy(dtype=object),
            "store_type": frame["StoreType"].astype(str).to_numpy(dtype=object),
            "assortment": frame["Assortment"].astype(str).to_numpy(dtype=object),
            "competition": np.log1p(frame["CompetitionDistance"].astype(float).to_numpy()),
            "promo2": frame["Promo2"].astype(float).to_numpy(),
        }
    )
    days = pd.date_range(table["day"].min(), table["day"].max(), freq="D")
    grid = pd.MultiIndex.from_product([table["store"].unique(), days], names=["store", "day"])
    table = table.set_index(["store", "day"]).reindex(grid).reset_index()
    for name in STATIC:
        table[name] = table.groupby("store", sort=False)[name].transform("first")
    return table


def frames(data: pd.DataFrame, covariates: bool):
    common = dict(target="sales", lookback=LOOKBACK, horizon=HORIZON, series="store", time="day")
    if covariates:
        common.update(known=KNOWN, static=STATIC)
    train = horizon_frame(data, end=TRAIN_END, stride=3, **common)
    calibration = horizon_frame(data, start=CAL_START, end=CAL_END, stride=3, **common)
    test = horizon_frame(data, start=TEST_START, **common)
    return train, calibration, test


def quantile_loss(quantiles: np.ndarray, y: np.ndarray) -> float:
    """Twice the mean pinball loss over ``LEVELS``; ``quantiles`` is ``(..., levels)``."""
    error = y[..., None] - quantiles
    return float(np.mean(2.0 * np.maximum(LEVELS * error, (LEVELS - 1.0) * error)))


def crps_draws(draws: np.ndarray, y: np.ndarray) -> float:
    """CRPS from draws ``(n, rows)``: ``E|X - y| - E|X - X'| / 2``, with the draws split in two halves."""
    half = draws.shape[0] // 2
    spread = np.mean(np.abs(draws[:half] - draws[half : 2 * half]), axis=0)
    return float(np.mean(np.mean(np.abs(draws - y[None]), axis=0) - 0.5 * spread))


def calibration_error(probability: np.ndarray, event: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: the row-weighted gap between mean probability and event rate per bin."""
    which = np.clip((probability * bins).astype(int), 0, bins - 1)
    gaps = [np.sum(which == b) * abs(probability[which == b].mean() - event[which == b].mean()) for b in range(bins) if np.any(which == b)]
    return float(np.sum(gaps) / len(probability))


def daily_quantiles(quantiles: np.ndarray, truth: np.ndarray) -> dict:
    """Per-day scores in sales from quantiles ``(rows, steps, levels)`` at ``LEVELS``."""
    error = truth[..., None] - quantiles
    loss = np.mean(2.0 * np.maximum(LEVELS * error, (LEVELS - 1.0) * error), axis=-1)
    return {
        "quantile_loss": float(loss.mean()),
        "median_loss": float(np.median(loss)),
        "coverage": float(np.mean((quantiles[..., 0] <= truth) & (truth <= quantiles[..., -1]))),
        "by_step": loss.mean(axis=0).tolist(),
        "median_by_step": np.median(loss, axis=0).tolist(),
    }


def _sales_quantiles(draws: np.ndarray, cap: float) -> np.ndarray:
    return np.moveaxis(np.quantile(np.expm1(np.minimum(draws, cap)), LEVELS, axis=0), 0, -1)


def daily_draws(draws: np.ndarray, y_log: np.ndarray, cap: float) -> dict:
    """Per-day scores in sales from draws of ``log1p`` sales ``(n, rows, steps)``, capped at ``cap``."""
    truth = np.expm1(y_log)
    scores = daily_quantiles(_sales_quantiles(draws, cap), truth)
    scores["uncapped_quantile_loss"] = quantile_loss(_sales_quantiles(draws, GUARD), truth)
    return scores


def path_scores(draws: np.ndarray, y_log: np.ndarray, lags_log: np.ndarray, cap: float, calibrated=None) -> dict:
    """Scores of the total and the peak from draws of ``log1p`` paths ``(n, rows, steps)``, capped at ``cap``.

    ``calibrated`` maps ``"total"`` and ``"peak"`` to a function of the
    rows' thresholds that returns the calibrated 80% interval ``(rows, 2)``
    and exceedance probability ``(rows,)``.
    """
    sales = np.expm1(np.minimum(draws, cap))
    truth = np.expm1(y_log)
    recent = np.expm1(lags_log)
    out = {}
    for name, function, threshold in (
        ("total", lambda paths: paths.sum(axis=-1), 1.1 * np.nansum(recent[:, :HORIZON], axis=1)),
        ("peak", lambda paths: paths.max(axis=-1), np.nanmax(recent, axis=1)),
    ):
        values = function(sales)
        observed = function(truth)
        low, high = np.quantile(values, [0.1, 0.9], axis=0)
        probability = np.mean(values > threshold[None], axis=0)
        event = (observed > threshold).astype(float)
        out[name] = {
            "crps": crps_draws(values, observed),
            "coverage": float(np.mean((low <= observed) & (observed <= high))),
            "width": float(np.mean(high - low)),
            "brier": float(np.mean(np.square(probability - event))),
            "ece": calibration_error(probability, event),
            "event_rate": float(event.mean()),
            "mean_probability": float(probability.mean()),
        }
        if calibrated is not None:
            interval, exceed = calibrated[name](threshold)
            out[name]["calibrated_coverage"] = float(np.mean((interval[:, 0] <= observed) & (observed <= interval[:, 1])))
            out[name]["calibrated_width"] = float(np.mean(interval[:, 1] - interval[:, 0]))
            out[name]["calibrated_brier"] = float(np.mean(np.square(exceed - event)))
            out[name]["calibrated_ece"] = calibration_error(exceed, event)
            out[name]["calibrated_mean_probability"] = float(np.mean(exceed))
    return out


def _lags(frame) -> np.ndarray:
    return frame.X[[f"sales_lag{j}" for j in range(1, LOOKBACK + 1)]].to_numpy(np.float64)


def run_fit(stores: int, seed: int, models, epochs):
    ROOT.mkdir(parents=True, exist_ok=True)
    data = long_table(stores, seed)
    for name in models:
        path = ROOT / f"{name}-{stores}-{seed}.json"
        if path.exists():
            continue
        train, calibration, test = frames(data, covariates=name != "no_covariates")
        cap = cap_of(train)
        total, peak_sales = path_targets(cap)
        settings = dict(MODELS[name], random_state=seed)
        if epochs is not None:
            settings["max_epochs"] = epochs
        started = time.perf_counter()
        model = ASRRegressor(**settings).fit(train.X, train.Y)
        seconds = time.perf_counter() - started
        model.calibrate(calibration.X, calibration.Y, target=[total, peak_sales])
        draws = np.asarray(model.sample(test.X, n=N_DRAWS, seed=seed), dtype=np.float64)

        def calibrated(function):
            def read(threshold):
                interval = np.asarray(model.predict_interval(test.X, 0.8, target=function), dtype=np.float64)
                exceed = np.asarray(model.predict_exceedance(test.X, threshold, target=function), dtype=np.float64)
                return interval, exceed

            return read

        result = {
            "seconds": seconds,
            "cap": cap,
            "train_rows": int(len(train.X)),
            "calibration_rows": int(len(calibration.X)),
            "test_rows": int(len(test.X)),
            "nll": float(-np.mean(np.asarray(model.log_prob(test.X, test.Y)))),
            "daily": daily_draws(draws, test.Y, cap),
            "paths": path_scores(draws, test.Y, _lags(test), cap, {"total": calibrated(total), "peak": calibrated(peak_sales)}),
        }
        if name == "joint":
            import jax

            loose = Independent(model.predict_dist(test.X).marginals())
            loose = np.asarray(loose.sample(jax.random.PRNGKey(seed), (N_DRAWS,)), dtype=np.float64)
            result["joint_marginals"] = {"daily": daily_draws(loose, test.Y, cap), "paths": path_scores(loose, test.Y, _lags(test), cap)}
        path.write_text(json.dumps(result))
        print(name, stores, seed, json.dumps(result), flush=True)


def _quantile_columns(forecast: pd.DataFrame, levels) -> list:
    named = {}
    for column in forecast.columns:
        try:
            named[round(float(column), 4)] = column
        except (TypeError, ValueError):
            continue
    return [named[round(float(level), 4)] for level in levels]


def run_chronos(stores: int, seed: int):
    """Chronos-2 zero-shot with the known covariates, quantiles per day at each test origin."""
    from chronos import BaseChronosPipeline

    path = ROOT / f"chronos2-{stores}-{seed}.json"
    if path.exists():
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    data = long_table(stores, seed)
    train, _calibration, test = frames(data, covariates=True)
    cap = cap_of(train)
    pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    table = data.assign(target=np.expm1(data["sales"]))
    table[REFERENCE_COVARIATES] = table[REFERENCE_COVARIATES].fillna(0.0)
    levels = [float(level) for level in LEVELS]
    quantiles = np.full((len(test.X), HORIZON, len(levels)), np.nan)
    series = test.index["series"].to_numpy()
    started = time.perf_counter()
    for origin, rows in test.index.groupby("origin").indices.items():
        origin = pd.Timestamp(origin)
        names = series[rows]
        chosen = table["store"].isin(names)
        context = table[chosen & (table["day"] <= origin)][["store", "day", "target", *REFERENCE_COVARIATES]]
        ahead = chosen & (table["day"] > origin) & (table["day"] <= origin + pd.Timedelta(days=HORIZON))
        forecast = pipeline.predict_df(
            context, future_df=table[ahead][["store", "day", *REFERENCE_COVARIATES]], prediction_length=HORIZON,
            quantile_levels=levels, id_column="store", timestamp_column="day", target="target",
        )
        forecast = forecast.sort_values(["store", "day"], kind="stable")
        values = forecast[_quantile_columns(forecast, levels)].to_numpy(np.float64).reshape(-1, HORIZON, len(levels))
        position = {store: index for index, store in enumerate(forecast["store"].to_numpy()[::HORIZON])}
        quantiles[rows] = values[[position[name] for name in names]]
    # The same cap as the draws of the other models, though Chronos-2's quantiles of sales come nowhere near it.
    quantiles = np.minimum(quantiles, np.expm1(cap))
    result = {
        "seconds": time.perf_counter() - started,
        "cap": cap,
        "test_rows": int(len(test.X)),
        "daily": daily_quantiles(quantiles, np.expm1(test.Y)),
    }
    path.write_text(json.dumps(result))
    print("chronos2", stores, seed, json.dumps(result), flush=True)


def run_deepar(stores: int, seed: int, epochs):
    """DeepAR (GluonTS, PyTorch) on ``log1p`` sales with the known covariates; its draws are paths."""
    import torch
    from gluonts.dataset.common import ListDataset
    from gluonts.torch.model.deepar import DeepAREstimator

    path = ROOT / f"deepar-{stores}-{seed}.json"
    if path.exists():
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    data = long_table(stores, seed)
    train_frame, _calibration, test = frames(data, covariates=True)
    cap = cap_of(train_frame)
    start = data["day"].min()
    kinds = {name: sorted(data[name].dropna().unique()) for name in ("store_type", "assortment")}
    series = {}
    for store, part in data.groupby("store", sort=False):
        part = part.sort_values("day")
        series[store] = {
            "target": part["sales"].to_numpy(np.float32),
            "dynamic": part[REFERENCE_COVARIATES].fillna(0.0).to_numpy(np.float32).T,
            "cat": [kinds[name].index(part[name].iloc[0]) for name in kinds],
            "real": [float(np.nan_to_num(part["competition"].iloc[0])), float(np.nan_to_num(part["promo2"].iloc[0]))],
        }

    def entry(store, end, ahead):
        values = series[store]
        return {
            "start": pd.Period(start, freq="D"),
            "target": values["target"][: end + 1],
            "feat_dynamic_real": values["dynamic"][:, : end + 1 + ahead],
            "feat_static_cat": values["cat"],
            "feat_static_real": values["real"],
            "item_id": store,
        }

    def position(day):
        return int((pd.Timestamp(day) - start).days)

    # The last day the ashlar models train on: the 12th step of the last training origin.
    train = ListDataset([entry(store, position(TRAIN_END) + HORIZON, 0) for store in series], freq="D")
    estimator = DeepAREstimator(
        freq="D", prediction_length=HORIZON, context_length=LOOKBACK,
        num_feat_dynamic_real=len(REFERENCE_COVARIATES), num_feat_static_cat=len(kinds), num_feat_static_real=2,
        cardinality=[len(values) for values in kinds.values()],
        trainer_kwargs={"max_epochs": epochs or DEEPAR_EPOCHS, "accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False},
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    predictor = estimator.train(train)
    seconds = time.perf_counter() - started
    rows = ListDataset(
        [entry(store, position(origin), HORIZON) for store, origin in zip(test.index["series"], test.index["origin"])], freq="D"
    )
    draws = np.stack([forecast.samples for forecast in predictor.predict(rows, num_samples=N_DRAWS)], axis=1).astype(np.float64)
    result = {
        "seconds": seconds,
        "cap": cap,
        "test_rows": int(len(test.X)),
        "daily": daily_draws(draws, test.Y, cap),
        "paths": path_scores(draws, test.Y, _lags(test), cap),
    }
    path.write_text(json.dumps(result))
    print("deepar", stores, seed, json.dumps(result), flush=True)


# Stage ``recalibrate``: which calibration rows, and which groups of them, carry over to the summer's origins.
# A window of recent origins, one a day.
WINDOW = 28
# The weighted calibration's half-life, in days of origin.
HALF_LIFE = 14.0
# Levels whose pinball losses average to the CRPS of a path function after calibration.
GRID = np.arange(1, 100) / 100.0
CALIBRATED_DRAWS = 1000
RECALIBRATED = ("joint", "independent", "joint_flow")
STRATEGIES = {
    "raw": "no calibration",
    "spring": "28 February to 19 May, every third origin: the fit stage's calibration",
    "spring_calendar": "the same rows, Mondrian by the calendar of the 12 days ahead",
    "spring_store_type": "the same rows, Mondrian by store type",
    "latest": "the last 28 origins whose 12 days are observed at the first forecast, 22 April to 19 May",
    "rolling": "at each origin, the last 28 origins whose 12 days are observed by then",
    "rolling_calendar": "rolling, Mondrian by the calendar of the 12 days ahead",
    "weighted": "at each origin, every origin from 28 February whose 12 days are observed, with a half-life of 14 days",
}
PATHS = ("total", "peak")


def calendar_labels(X) -> np.ndarray:
    """Each row's calendar over its 12 days ahead: a public holiday, else a school holiday, else neither."""
    public = np.zeros(len(X), dtype=bool)
    school = np.zeros(len(X), dtype=bool)
    for step in range(1, HORIZON + 1):
        public |= ~X[f"state_holiday_step{step}"].astype(str).isin(["0", "0.0", "nan"]).to_numpy()
        school |= np.nan_to_num(X[f"school_holiday_step{step}"].to_numpy(np.float64)) > 0
    return np.where(public, "public", np.where(school, "school", "neither")).astype(object)


def _table_levels(table, coverage: float) -> tuple[float, float]:
    """``interval_levels`` of a table ``(sorted PIT values, weights or None)``; nominal without one.

    Weighted, the conformal quantile puts the forecast row's own weight, 1,
    at infinity (Barber, Candès, Ramdas, and Tibshirani, 2023), which with
    equal weights is the ``ceil((n + 1) c)``-th smallest score.
    """
    from ashlar.calibrate import interval_levels

    if table is None:
        return 0.5 * (1.0 - coverage), 0.5 * (1.0 + coverage)
    values, weights = table
    if weights is None:
        return interval_levels({"kind": "pit", "values": values}, coverage)
    scores = np.abs(2.0 * values - 1.0)
    order = np.argsort(scores, kind="stable")
    mass = np.cumsum(weights[order]) / (weights.sum() + 1.0)
    reached = np.flatnonzero(mass >= coverage)
    nominal = 1.0 if reached.size == 0 else min(float(scores[order][reached[0]]), 1.0)
    return 0.5 * (1.0 - nominal), 0.5 * (1.0 + nominal)


def _table_quantiles(table, levels) -> np.ndarray:
    """``recalibrated_levels`` of a table: the PIT values' quantile at each level, weighted when it has weights."""
    levels = np.asarray(levels, dtype=np.float64)
    if table is None:
        return levels
    values, weights = table
    if weights is None:
        return np.quantile(values, levels)
    # Plotting positions that reduce to numpy's linear quantile when the weights are equal.
    cumulative = np.cumsum(weights)
    positions = (cumulative - weights) / max(cumulative[-1] - weights[-1], 1e-12)
    return np.interp(levels, positions, values)


def _table_cdf(table, probability) -> np.ndarray:
    """``recalibrated_cdf`` of a table: the share of calibration weight at or below the model's own ``F(t)``."""
    from ashlar.calibrate import recalibrated_cdf

    probability = np.asarray(probability, dtype=np.float64)
    if table is None:
        return probability
    values, weights = table
    if weights is None:
        return recalibrated_cdf({"kind": "pit", "values": values}, probability)
    unique, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weights)
    xs = np.concatenate([[0.0], unique, [1.0]])
    ys = np.concatenate([[0.0], np.cumsum(mass) / (weights.sum() + 1.0), [1.0]])
    keep = np.concatenate([np.diff(xs) > 0.0, [True]])
    return np.interp(np.clip(probability, 0.0, 1.0), xs[keep], ys[keep])


def _order_statistic(ordered: np.ndarray, levels: np.ndarray, rows: np.ndarray, upper: bool = True) -> np.ndarray:
    """The draw at each level, as ``draw_quantiles`` (``upper``) or the lower end of ``draw_interval``.

    ``ordered`` is ``(n, rows)``, sorted on the first axis; ``levels`` is
    ``(levels,)``; the result is ``(levels, len(rows))``.
    """
    n = ordered.shape[0]
    scaled = np.asarray(levels, dtype=np.float64) * (n + 1)
    index = np.clip(np.ceil(scaled) if upper else np.floor(scaled), 1, n).astype(np.int64) - 1
    return ordered[index][:, rows]


def _pinball(quantiles: np.ndarray, truth: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over ``levels`` of ``quantiles`` ``(levels, rows)``, per row."""
    error = truth[None, :] - quantiles
    levels = levels[:, None]
    return np.mean(2.0 * np.maximum(levels * error, (levels - 1.0) * error), axis=0)


def _score_rows(out: dict, rows: np.ndarray, tables: dict, block: dict, cap: float) -> None:
    """Fill ``out`` for ``rows`` of a block (indices into the block) read with ``tables`` (``None``: raw)."""
    for step in range(HORIZON):
        table = None if tables is None else tables["columns"][step]
        ordered = block["daily"][..., step]
        truth = block["truth"][rows, step]
        low, high = _table_levels(table, 0.8)
        ends = np.expm1(np.minimum(np.concatenate([
            _order_statistic(ordered, np.array([low]), rows, upper=False),
            _order_statistic(ordered, np.array([high]), rows),
        ]), cap))
        quantiles = np.expm1(np.minimum(_order_statistic(ordered, _table_quantiles(table, LEVELS), rows), cap))
        out["daily_cover"][rows, step] = (ends[0] <= truth) & (truth <= ends[1])
        out["daily_width"][rows, step] = ends[1] - ends[0]
        out["daily_loss"][rows, step] = _pinball(quantiles, truth, LEVELS)
    for name in PATHS:
        table = None if tables is None else tables[name]
        ordered = block[name]
        observed = block[f"{name}_truth"][rows]
        low, high = _table_levels(table, 0.8)
        lower = _order_statistic(ordered, np.array([low]), rows, upper=False)[0]
        upper = _order_statistic(ordered, np.array([high]), rows)[0]
        out[f"{name}_cover"][rows] = (lower <= observed) & (observed <= upper)
        out[f"{name}_width"][rows] = upper - lower
        out[f"{name}_probability"][rows] = 1.0 - _table_cdf(table, block[f"{name}_below"][rows])
        out[f"{name}_crps"][rows] = _pinball(_order_statistic(ordered, _table_quantiles(table, GRID), rows), observed, GRID)


def _tables(scores: dict, rows: np.ndarray, weights=None) -> dict:
    """The PIT table of every column and path function from the calibration ``rows``, sorted, with their weights."""

    def one(pit):
        values = np.clip(np.asarray(pit, dtype=np.float64)[rows], 0.0, 1.0)
        if weights is None:
            return np.sort(values), None
        order = np.argsort(values, kind="stable")
        return values[order], np.asarray(weights, dtype=np.float64)[order]

    return {
        "columns": [one(scores["columns"][:, step]) for step in range(HORIZON)],
        "total": one(scores["total"]),
        "peak": one(scores["peak"]),
        "rows": int(rows.size),
    }


def _calibration_rows(strategy: str, origin: np.ndarray, at, spring: np.ndarray):
    """The pool rows that calibrate the forecasts made at origin ``at``, and their weights (``None``: equal)."""
    day = np.timedelta64(1, "D")
    seen = np.datetime64(at) - HORIZON * day
    if strategy.startswith("spring"):
        return np.flatnonzero(spring), None
    if strategy == "latest":
        last = np.datetime64(CAL_END)
        return np.flatnonzero((origin <= last) & (origin > last - WINDOW * day)), None
    if strategy.startswith("rolling"):
        return np.flatnonzero((origin <= seen) & (origin > seen - WINDOW * day)), None
    if strategy == "weighted":
        rows = np.flatnonzero(origin <= seen)
        age = (seen - origin[rows]) / day
        return rows, np.power(0.5, age / HALF_LIFE)
    raise ValueError(strategy)


def _grouping(strategy: str, calendar: np.ndarray, store_type: np.ndarray):
    if strategy.endswith("_calendar"):
        return calendar
    if strategy.endswith("_store_type"):
        return store_type
    return None


def _events(Y_log: np.ndarray, lags_log: np.ndarray) -> dict:
    """Each row's observed total and peak in sales, their thresholds, and whether each passed it."""
    truth = np.expm1(Y_log)
    recent = np.expm1(lags_log)
    out = {
        "total_truth": truth.sum(axis=1),
        "peak_truth": truth.max(axis=1),
        "total_threshold": 1.1 * np.nansum(recent[:, :HORIZON], axis=1),
        "peak_threshold": np.nanmax(recent, axis=1),
    }
    for name in PATHS:
        out[f"{name}_event"] = (out[f"{name}_truth"] > out[f"{name}_threshold"]).astype(float)
    return out


def _summary(out: dict, events: dict, weeks: np.ndarray) -> dict:
    """Mean scores over the test rows, and the 80% intervals' coverage by week of origin."""
    result = {
        "daily": {
            "coverage": float(out["daily_cover"].mean()),
            "width": float(out["daily_width"].mean()),
            "quantile_loss": float(out["daily_loss"].mean()),
            "median_loss": float(np.median(out["daily_loss"])),
        },
        "by_week": {"daily": [float(out["daily_cover"][weeks == week].mean()) for week in np.unique(weeks)]},
    }
    for name in PATHS:
        probability, event = out[f"{name}_probability"], events[f"{name}_event"]
        result[name] = {
            "coverage": float(out[f"{name}_cover"].mean()),
            "width": float(out[f"{name}_width"].mean()),
            "crps": float(out[f"{name}_crps"].mean()),
            "brier": float(np.mean(np.square(probability - event))),
            "ece": calibration_error(probability, event),
            "mean_probability": float(probability.mean()),
            "event_rate": float(event.mean()),
        }
        result["by_week"][name] = [float(out[f"{name}_cover"][weeks == week].mean()) for week in np.unique(weeks)]
    return result


def run_recalibrate(stores: int, seed: int, models, epochs):
    """Fit each model once, then score the summer's forecasts under every calibration in ``STRATEGIES``.

    Every row from 28 February on gets its PIT under each column, the total,
    and the peak once (``calibrate``'s own scores, 1,000 draws a row); a
    calibration is the table of some of those rows, read for each forecast
    with ``ashlar.calibrate``'s steps on 1,000 draws of its path, as
    ``predict_interval`` and ``predict_exceedance`` read them.
    """
    ROOT.mkdir(parents=True, exist_ok=True)
    data = long_table(stores, seed)
    common = dict(target="sales", lookback=LOOKBACK, horizon=HORIZON, series="store", time="day", known=KNOWN, static=STATIC)
    for name in models:
        path = ROOT / f"recalibrate-{name}-{stores}-{seed}.json"
        archive = path.with_suffix(".npz")
        if path.exists() and archive.exists():
            continue
        train, calibration, _test = frames(data, covariates=True)
        pool = horizon_frame(data, start=CAL_START, **common)
        cap = cap_of(train)
        total, peak_sales = path_targets(cap)
        settings = dict(MODELS[name], random_state=seed)
        if epochs is not None:
            settings["max_epochs"] = epochs
        started = time.perf_counter()
        model = ASRRegressor(**settings).fit(train.X, train.Y)
        fitted = time.perf_counter()
        raw, _functions = model._calibration_scores(pool.X, pool.Y, CALIBRATED_DRAWS, seed, [total, peak_sales])
        scores = {"columns": raw["columns"], "total": raw["targets"]["total"][1], "peak": raw["targets"]["peak_sales"][1]}
        scored = time.perf_counter()
        origin = pd.to_datetime(pool.index["origin"]).to_numpy().astype("datetime64[D]")
        spring_days = pd.to_datetime(calibration.index["origin"]).to_numpy().astype("datetime64[D]")
        spring = np.isin(origin, spring_days)
        calendar = calendar_labels(pool.X)
        store_type = pool.X["store_type"].astype(str).to_numpy(dtype=object)
        events = _events(pool.Y, _lags(pool))
        test = np.flatnonzero(origin >= np.datetime64(TEST_START))
        days = np.unique(origin[test])
        outs = {
            strategy: {
                "daily_cover": np.zeros((test.size, HORIZON), dtype=bool),
                "daily_width": np.zeros((test.size, HORIZON)),
                "daily_loss": np.zeros((test.size, HORIZON)),
                **{f"{path_name}_{kind}": np.zeros(test.size, dtype=bool if kind == "cover" else np.float64)
                   for path_name in PATHS for kind in ("cover", "width", "probability", "crps")},
            }
            for strategy in STRATEGIES
        }
        base = {strategy: np.zeros((test.size, 2)) for strategy in ("spring", "rolling")}
        calibration_rows = {strategy: [] for strategy in STRATEGIES}
        position = {row: index for index, row in enumerate(test)}
        grids = {
            "daily_grid": np.zeros((test.size, HORIZON, GRID.size), dtype=np.float32),
            "total_grid": np.zeros((test.size, GRID.size), dtype=np.float32),
            "peak_grid": np.zeros((test.size, GRID.size), dtype=np.float32),
            "total_below": np.zeros(test.size),
            "peak_below": np.zeros(test.size),
        }
        drawn = np.asarray(model.sample(pool.X.iloc[test], n=CALIBRATED_DRAWS, seed=seed), dtype=np.float32)
        for at in days:
            rows = np.flatnonzero(origin == at)
            draws = drawn[:, [position[row] for row in rows]]
            sales = np.expm1(np.minimum(draws.astype(np.float64), cap))
            block = {
                "daily": np.sort(draws, axis=0),
                "truth": np.expm1(pool.Y[rows]),
                "total": np.sort(sales.sum(axis=-1), axis=0),
                "peak": np.sort(sales.max(axis=-1), axis=0),
            }
            for path_name in PATHS:
                block[f"{path_name}_truth"] = events[f"{path_name}_truth"][rows]
                block[f"{path_name}_below"] = np.mean(block[path_name] <= events[f"{path_name}_threshold"][rows][None], axis=0)
            where = np.asarray([position[row] for row in rows])
            every = np.arange(rows.size)
            for step in range(HORIZON):
                grid = _order_statistic(block["daily"][..., step], GRID, every)
                grids["daily_grid"][where, step] = np.expm1(np.minimum(grid, cap)).T
            for path_name in PATHS:
                grids[f"{path_name}_grid"][where] = _order_statistic(block[path_name], GRID, every).T
                grids[f"{path_name}_below"][where] = block[f"{path_name}_below"]
            for strategy in STRATEGIES:
                part = {key: value[where] for key, value in outs[strategy].items()}
                local = np.arange(rows.size)
                if strategy == "raw":
                    _score_rows(part, local, None, block, cap)
                else:
                    chosen, weights = _calibration_rows(strategy, origin, at, spring)
                    calibration_rows[strategy].append(int(chosen.size))
                    pooled = _tables(scores, chosen, weights)
                    labels = _grouping(strategy, calendar, store_type)
                    if labels is None:
                        _score_rows(part, local, pooled, block, cap)
                    else:
                        for label in np.unique(labels[rows]):
                            members = chosen[labels[chosen] == label]
                            table = _tables(scores, members) if members.size >= MIN_GROUP_ROWS else pooled
                            _score_rows(part, np.flatnonzero(labels[rows] == label), table, block, cap)
                for key, value in part.items():
                    outs[strategy][key][where] = value
            for strategy in base:
                chosen, _weights = _calibration_rows(strategy, origin, at, spring)
                for index, path_name in enumerate(PATHS):
                    base[strategy][where, index] = events[f"{path_name}_event"][chosen].mean()
        test_events = {key: value[test] for key, value in events.items()}
        weeks = ((origin[test] - origin[test].min()) / np.timedelta64(7, "D")).astype(int)
        result = {
            "fit_seconds": fitted - started,
            "score_seconds": scored - fitted,
            "seconds": time.perf_counter() - started,
            "cap": cap,
            "pool_rows": int(len(pool.X)),
            "test_rows": int(test.size),
            "calendar_share": {
                label: {"spring": float(np.mean(calendar[spring] == label)), "test": float(np.mean(calendar[test] == label))}
                for label in ("public", "school", "neither")
            },
            "calibration_rows": {strategy: float(np.mean(counts)) for strategy, counts in calibration_rows.items() if counts},
            "strategies": {strategy: _summary(out, test_events, weeks) for strategy, out in outs.items()},
            "base_rate": {
                strategy: {
                    path_name: {
                        "brier": float(np.mean(np.square(values[:, index] - test_events[f"{path_name}_event"]))),
                        "ece": calibration_error(values[:, index], test_events[f"{path_name}_event"]),
                    }
                    for index, path_name in enumerate(PATHS)
                }
                for strategy, values in base.items()
            },
        }
        np.savez(
            archive,
            origin=origin.astype(np.int64),
            test=test,
            spring=spring,
            calendar=calendar.astype(str),
            columns=scores["columns"].astype(np.float32),
            total=np.asarray(scores["total"], dtype=np.float32),
            peak=np.asarray(scores["peak"], dtype=np.float32),
            daily_truth=np.expm1(pool.Y[test]).astype(np.float32),
            **{f"{path_name}_{kind}": events[f"{path_name}_{kind}"] for path_name in PATHS for kind in ("truth", "threshold", "event")},
            **grids,
        )
        path.write_text(json.dumps(result))
        print("recalibrate", name, stores, seed, json.dumps(result), flush=True)


def report_recalibrate(stores: int, seeds) -> None:
    """The scores of the summer's forecasts under each calibration, mean over seeds, one table per model."""
    for name in RECALIBRATED:
        runs = _runs(f"recalibrate-{name}", stores, seeds)
        if not runs:
            continue
        print(f"\n{name}: calibration of the 31 May to 19 July origins, {len(runs)} seeds\n")
        print("| Calibration | Rows | Daily 80% | Daily loss | Total 80% | Total CRPS | Total Brier | Peak 80% | Peak width | Peak CRPS | Peak Brier | Peak ECE |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for strategy in STRATEGIES:
            cells = [run["strategies"][strategy] for run in runs]

            def mean(part, key, cells=cells):
                return float(np.mean([cell[part][key] for cell in cells]))

            count = np.mean([run["calibration_rows"].get(strategy, 0.0) for run in runs])
            print(
                f"| {strategy} | {count:,.0f} | {mean('daily', 'coverage'):.1%} | {mean('daily', 'quantile_loss'):,.1f} | "
                f"{mean('total', 'coverage'):.1%} | {mean('total', 'crps'):,.0f} | {mean('total', 'brier'):.4f} | "
                f"{mean('peak', 'coverage'):.1%} | {mean('peak', 'width'):,.0f} | {mean('peak', 'crps'):,.0f} | "
                f"{mean('peak', 'brier'):.4f} | {mean('peak', 'ece'):.3f} |"
            )
        for strategy in ("spring", "rolling"):
            brier = np.mean([run["base_rate"][strategy]["peak"]["brier"] for run in runs])
            total = np.mean([run["base_rate"][strategy]["total"]["brier"] for run in runs])
            print(f"| base rate, {strategy} rows | | | | | | {total:.4f} | | | | {brier:.4f} | |")
        print("\nPeak 80% coverage by week of origin:")
        for strategy in STRATEGIES:
            weeks = np.mean([run["strategies"][strategy]["by_week"]["peak"] for run in runs], axis=0)
            print(f"  {strategy:<18}" + " ".join(f"{value:6.1%}" for value in weeks))
        shares = runs[0]["calendar_share"]
        print("\nCalendar of the 12 days ahead, share of rows (seed " + str(seeds[0]) + "): " + ", ".join(
            f"{label} {share['spring']:.0%} in spring, {share['test']:.0%} in summer" for label, share in shares.items()
        ))


# Stage ``adaptive``: each quantity's interval follows its own recent misses, scored from ``recalibrate``'s archives.
# Steps of the tracked level per origin, for a miss rate one unit off.
RATES = (0.02, 0.05, 0.1, 0.2)


def tracked_levels(scores: np.ndarray, day: np.ndarray, rate: float, coverage: float = 0.8) -> np.ndarray:
    """Each row's nominal level by quantile tracking on its PIT score ``|2u - 1|``, one level per origin.

    Origins come one a day. The level starts at ``coverage`` at the first
    origin; at each origin it moves by ``rate`` times the share of rows that
    the origin ``HORIZON`` days back, whose days are all observed by then,
    left outside its own interval, less ``1 - coverage`` (Angelopoulos,
    Candès, and Tibshirani, 2023; adaptive conformal inference, Gibbs and
    Candès, 2021). A row is inside when its score is at most its level.
    """
    offset = (day - day.min()).astype(np.int64)
    order = np.argsort(offset, kind="stable")
    bounds = np.searchsorted(offset[order], np.arange(offset.max() + 2))
    levels = np.empty(offset.max() + 1)
    level = coverage
    for k in range(levels.size):
        back = k - HORIZON
        if back >= 0 and bounds[back + 1] > bounds[back]:
            rows = order[bounds[back] : bounds[back + 1]]
            miss = float(np.mean(scores[rows] > levels[back]))
            level = float(np.clip(level + rate * (miss - (1.0 - coverage)), 0.0, 1.0))
        levels[k] = level
    return levels[offset]


def _grid_at(grid: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Each row's value at its own level from its draws' order statistics at ``GRID``, ``(rows, GRID)``, linear between."""
    position = np.clip((np.asarray(levels) - GRID[0]) / (GRID[1] - GRID[0]), 0.0, GRID.size - 1.0)
    low = np.floor(position).astype(np.int64)
    high = np.minimum(low + 1, GRID.size - 1)
    share = position - low
    rows = np.arange(grid.shape[0])
    return grid[rows, low] * (1.0 - share) + grid[rows, high] * share


def _level_scores(archive, test_levels: dict, weeks: np.ndarray) -> dict:
    """Coverage and width of the central intervals at each test row's nominal level, per quantity."""
    test = archive["test"]
    out = {"by_week": {}}
    covered = np.zeros((test.size, HORIZON), dtype=bool)
    width = np.zeros((test.size, HORIZON))
    for step in range(HORIZON):
        level = test_levels["columns"][step]
        score = np.abs(2.0 * archive["columns"][test, step] - 1.0)
        covered[:, step] = score <= level
        grid = archive["daily_grid"][:, step]
        width[:, step] = _grid_at(grid, 0.5 * (1.0 + level)) - _grid_at(grid, 0.5 * (1.0 - level))
    out["daily"] = {"coverage": float(covered.mean()), "width": float(width.mean())}
    out["by_week"]["daily"] = [float(covered[weeks == week].mean()) for week in np.unique(weeks)]
    for name in PATHS:
        level = test_levels[name]
        score = np.abs(2.0 * archive[name][test] - 1.0)
        inside = score <= level
        grid = archive[f"{name}_grid"]
        spread = _grid_at(grid, 0.5 * (1.0 + level)) - _grid_at(grid, 0.5 * (1.0 - level))
        out[name] = {"coverage": float(inside.mean()), "width": float(spread.mean())}
        out["by_week"][name] = [float(inside[weeks == week].mean()) for week in np.unique(weeks)]
    return out


def run_adaptive(stores: int, seed: int, models) -> None:
    """Score quantile tracking at each rate in ``RATES``, the spring table, and the level fixed in hindsight."""
    from ashlar.calibrate import conformal_quantile

    for name in models:
        source = ROOT / f"recalibrate-{name}-{stores}-{seed}.npz"
        if not source.exists():
            continue
        archive = dict(np.load(source))
        day, test, spring = archive["origin"], archive["test"], archive["spring"]
        weeks = (day[test] - day[test].min()) // 7
        keys = [("columns", step) for step in range(HORIZON)] + [(path_name, None) for path_name in PATHS]

        def score(key):
            values = archive[key[0]] if key[1] is None else archive[key[0]][:, key[1]]
            return np.abs(2.0 * np.asarray(values, dtype=np.float64) - 1.0)

        def levels(make):
            chosen = {"columns": [None] * HORIZON}
            for key in keys:
                value = make(score(key))
                if key[1] is None:
                    chosen[key[0]] = value
                else:
                    chosen["columns"][key[1]] = value
            return chosen

        result = {
            "spring": _level_scores(archive, levels(lambda s: np.full(test.size, min(conformal_quantile(s[spring], 0.8), 1.0))), weeks),
            "hindsight": _level_scores(archive, levels(lambda s: np.full(test.size, float(np.quantile(s[test], 0.8)))), weeks),
        }
        for rate in RATES:
            result[f"adaptive {rate}"] = _level_scores(archive, levels(lambda s, rate=rate: tracked_levels(s, day, rate)[test]), weeks)
        (ROOT / f"adaptive-{name}-{stores}-{seed}.json").write_text(json.dumps(result))
        print("adaptive", name, stores, seed, json.dumps({key: {part: value[part] for part in ("daily", "total", "peak")} for key, value in result.items()}), flush=True)


def report_adaptive(stores: int, seeds) -> None:
    """Coverage and width of the 80% intervals when each quantity's level follows its recent misses."""
    for name in RECALIBRATED:
        runs = _runs(f"adaptive-{name}", stores, seeds)
        if not runs:
            continue
        print(f"\n{name}: 80% intervals read from the PIT, {len(runs)} seeds (hindsight: one level per quantity fixed on the test rows)\n")
        print("| Calibration | Daily 80% | Daily width | Total 80% | Total width | Peak 80% | Peak width |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for key in runs[0]:
            cells = [run[key] for run in runs]

            def mean(part, field, cells=cells):
                return float(np.mean([cell[part][field] for cell in cells]))

            print(
                f"| {key} | {mean('daily', 'coverage'):.1%} | {mean('daily', 'width'):,.0f} | {mean('total', 'coverage'):.1%} | "
                f"{mean('total', 'width'):,.0f} | {mean('peak', 'coverage'):.1%} | {mean('peak', 'width'):,.0f} |"
            )
        print("\nCoverage by week of origin (daily / total / peak):")
        for key in runs[0]:
            parts = []
            for part in ("daily", "total", "peak"):
                weeks = np.mean([run[key]["by_week"][part] for run in runs], axis=0)
                parts.append(" ".join(f"{value:5.1%}" for value in weeks))
            print(f"  {key:<15} " + "  |  ".join(parts))


def _runs(name: str, stores: int, seeds) -> list[dict]:
    files = [ROOT / f"{name}-{stores}-{seed}.json" for seed in seeds]
    return [json.loads(file.read_text()) for file in files if file.exists()]


def _mean_se(values) -> str:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return f"{values.mean():,.1f}"
    return f"{values.mean():,.1f} ± {values.std(ddof=1) / np.sqrt(values.size):,.1f}"


def report(stores: int, seeds):
    rows = {}
    for name in MODELS:
        runs = _runs(name, stores, seeds)
        if runs:
            rows[name] = runs
            if name == "joint":
                rows["joint_marginals"] = [run["joint_marginals"] for run in runs]
    for name in ("chronos2", "deepar"):
        runs = _runs(name, stores, seeds)
        if runs:
            rows[name] = runs
    print(f"{stores} stores, seeds {', '.join(map(str, seeds))}; ± one standard error over seeds.\n")
    print("| Model | Daily quantile loss | Median row-day | Uncapped | Daily 80% coverage | Joint NLL | Fit seconds |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name, runs in rows.items():
        daily = [run["daily"] for run in runs]
        loss = _mean_se([day["quantile_loss"] for day in daily])
        median = f"{np.mean([day['median_loss'] for day in daily]):,.1f}"
        uncapped = _mean_se([day["uncapped_quantile_loss"] for day in daily]) if "uncapped_quantile_loss" in daily[0] else "—"
        coverage = np.mean([day["coverage"] for day in daily])
        nll = f"{np.mean([run['nll'] for run in runs]):.3f}" if "nll" in runs[0] else "—"
        seconds = f"{np.mean([run['seconds'] for run in runs]):.0f}" if "seconds" in runs[0] else "—"
        print(f"| {name} | {loss} | {median} | {uncapped} | {coverage:.1%} | {nll} | {seconds} |")
    print()
    print("| Model | Path | CRPS | 80% coverage | Width | Brier | Calibration error | Mean P, events |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for name, runs in rows.items():
        if "paths" not in runs[0]:
            continue
        for kind in ("total", "peak"):
            cells = [run["paths"][kind] for run in runs]

            def mean(key, cells=cells):
                return np.mean([cell[key] for cell in cells])

            def pair(key, form, cells=cells):
                raw = format(mean(key), form)
                return raw + (" → " + format(mean("calibrated_" + key), form) if "calibrated_" + key in cells[0] else "")

            print(
                f"| {name} | {kind} | {_mean_se([cell['crps'] for cell in cells])} | {pair('coverage', '.1%')} | "
                f"{pair('width', ',.0f')} | {pair('brier', '.4f')} | {pair('ece', '.4f')} | "
                f"{mean('mean_probability'):.3f}, {mean('event_rate'):.3f} |"
            )
    print("\nA → B: raw from the draws → after calibrate(..., target=[total, peak_sales]) on the calibration origins.")


def main(argv):
    stages = [arg for arg in argv if "=" not in arg] or ["fit", "report"]
    options = dict(arg.split("=", 1) for arg in argv if "=" in arg)
    stores = int(options.get("stores", 300))
    seeds = tuple(int(seed) for seed in options.get("seeds", "0").split(","))
    models = options.get("models", ",".join(MODELS)).split(",")
    epochs = int(options["epochs"]) if "epochs" in options else None
    for stage in stages:
        if stage == "fit":
            for seed in seeds:
                run_fit(stores, seed, models, epochs)
        elif stage == "chronos":
            for seed in seeds:
                run_chronos(stores, seed)
        elif stage == "deepar":
            for seed in seeds:
                run_deepar(stores, seed, epochs)
        elif stage == "recalibrate":
            chosen = [name for name in models if name in RECALIBRATED] if "models" in options else list(RECALIBRATED)
            for seed in seeds:
                run_recalibrate(stores, seed, chosen, epochs)
        elif stage == "adaptive":
            chosen = [name for name in models if name in RECALIBRATED] if "models" in options else list(RECALIBRATED)
            for seed in seeds:
                run_adaptive(stores, seed, chosen)
            report_adaptive(stores, seeds)
        elif stage == "report":
            report(stores, seeds)
            report_recalibrate(stores, seeds)
            report_adaptive(stores, seeds)
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
