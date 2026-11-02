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
  assortment, whose autoregressive draws are paths, scored raw.

Per day, in sales: the mean quantile loss over levels 0.1 to 0.9 (twice the
pinball loss, an approximation of the CRPS) and the 80% interval's coverage.
Per path, in sales: the 12-day total and the peak day, each with its CRPS
(from 400 draws), its 80% interval's coverage and width, and the Brier score
and expected calibration error (10 bins) of an exceedance,
``P(total > 1.1 × the last 12 days' total)`` and ``P(peak > the largest day
of the last 28)``. The ashlar models' paths are read raw from the draws and
after ``calibrate(..., target=[total, peak_sales])`` on the calibration
origins.

Data: the fev benchmark's ``rossmann/1D`` (``autogluon/fev_datasets`` on
Hugging Face), saved as ``$ASHLAR_BENCH/rossmann-1D.parquet``
(``/tmp/ashlar-bench`` by default).

    python examples/benchmark_paths.py fit seeds=0,1,2
    python examples/benchmark_paths.py chronos seeds=0,1,2   # needs chronos-forecasting
    python examples/benchmark_paths.py deepar seeds=0,1,2    # needs gluonts[torch]
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
# Draws of log1p sales are capped here before expm1, about 485 million a day, so a rare wild draw stays finite.
CEILING = 20.0
KNOWN = ["open", "promo", "state_holiday", "school_holiday", "weekday"]
STATIC = ["store_type", "assortment", "competition", "promo2"]
# Numeric known covariates for Chronos-2 and DeepAR, which read the calendar themselves.
REFERENCE_COVARIATES = ["open", "promo", "school_holiday", "holiday"]
MODELS = {
    "joint": dict(task="multivariate_gaussian", head_kwargs={"rank": 4}),
    "independent": dict(task="multivariate_gaussian", head_kwargs={"rank": 0}),
    "joint_flow": dict(task="multivariate_flow"),
    "no_covariates": dict(task="multivariate_gaussian", head_kwargs={"rank": 4}),
}


def total(y):
    """The horizon's total sales, from ``log1p`` sales."""
    import jax.numpy as jnp

    return jnp.sum(jnp.expm1(jnp.minimum(jnp.asarray(y), CEILING)), axis=-1)


def peak_sales(y):
    """The horizon's largest day of sales, from ``log1p`` sales."""
    import jax.numpy as jnp

    return jnp.expm1(jnp.minimum(peak(jnp.asarray(y)), CEILING))


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
    return {
        "quantile_loss": quantile_loss(quantiles, truth),
        "coverage": float(np.mean((quantiles[..., 0] <= truth) & (truth <= quantiles[..., -1]))),
        "by_step": [quantile_loss(quantiles[:, step], truth[:, step]) for step in range(HORIZON)],
    }


def daily_draws(draws: np.ndarray, y_log: np.ndarray) -> dict:
    """Per-day scores in sales from draws of ``log1p`` sales ``(n, rows, steps)``."""
    quantiles = np.moveaxis(np.quantile(np.expm1(np.minimum(draws, CEILING)), LEVELS, axis=0), 0, -1)
    return daily_quantiles(quantiles, np.expm1(y_log))


def path_scores(draws: np.ndarray, y_log: np.ndarray, lags_log: np.ndarray, calibrated=None) -> dict:
    """Scores of the total and the peak from draws of ``log1p`` paths ``(n, rows, steps)``.

    ``calibrated`` maps ``"total"`` and ``"peak"`` to a function of the
    rows' thresholds that returns the calibrated 80% interval ``(rows, 2)``
    and exceedance probability ``(rows,)``.
    """
    sales = np.expm1(np.minimum(draws, CEILING))
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
            "train_rows": int(len(train.X)),
            "calibration_rows": int(len(calibration.X)),
            "test_rows": int(len(test.X)),
            "nll": float(-np.mean(np.asarray(model.log_prob(test.X, test.Y)))),
            "daily": daily_draws(draws, test.Y),
            "paths": path_scores(draws, test.Y, _lags(test), {"total": calibrated(total), "peak": calibrated(peak_sales)}),
        }
        if name == "joint":
            import jax

            loose = Independent(model.predict_dist(test.X).marginals())
            loose = np.asarray(loose.sample(jax.random.PRNGKey(seed), (N_DRAWS,)), dtype=np.float64)
            result["joint_marginals"] = {"daily": daily_draws(loose, test.Y), "paths": path_scores(loose, test.Y, _lags(test))}
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
    _train, _calibration, test = frames(data, covariates=True)
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
    result = {"seconds": time.perf_counter() - started, "test_rows": int(len(test.X)), "daily": daily_quantiles(quantiles, np.expm1(test.Y))}
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
    _train, _calibration, test = frames(data, covariates=True)
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
        trainer_kwargs={"max_epochs": epochs or 50, "accelerator": "cpu", "enable_progress_bar": False, "enable_model_summary": False},
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
        "test_rows": int(len(test.X)),
        "daily": daily_draws(draws, test.Y),
        "paths": path_scores(draws, test.Y, _lags(test)),
    }
    path.write_text(json.dumps(result))
    print("deepar", stores, seed, json.dumps(result), flush=True)


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
    print("| Model | Daily quantile loss | Daily 80% coverage | Joint NLL | Fit seconds |")
    print("|---|---:|---:|---:|---:|")
    for name, runs in rows.items():
        loss = _mean_se([run["daily"]["quantile_loss"] for run in runs])
        coverage = np.mean([run["daily"]["coverage"] for run in runs])
        nll = f"{np.mean([run['nll'] for run in runs]):.3f}" if "nll" in runs[0] else "—"
        seconds = f"{np.mean([run['seconds'] for run in runs]):.0f}" if "seconds" in runs[0] else "—"
        print(f"| {name} | {loss} | {coverage:.1%} | {nll} | {seconds} |")
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
        elif stage == "report":
            report(stores, seeds)
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
