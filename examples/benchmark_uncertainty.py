"""Benchmarks for calibrated intervals, the count heads, and member uncertainty.

Three stages, each written to its own JSON file in ``/tmp/ashlar-bench``:

- ``calibration``: 80% and 90% intervals before and after ``calibrate``, for
  the default mixture and the quantile head, beside LightGBM quantile
  regression before and after the same conformal step (CQR). Each split is
  60% train, 20% calibration, 20% test.
- ``counts``: ``poisson``, ``negative_binomial``, and a zero-inflated
  negative binomial against the ``gaussian`` and ``mixture`` heads and
  LightGBM with a Poisson objective, on four count targets.
- ``uncertainty``: ``predict_uncertainty`` from a 32-member TabM. Does the
  members' disagreement rank the rows by error, and does it grow on rows
  outside the training range?
- ``groups``: the calibration tables again, each with a group column. Does
  the 80% interval hold within every group after ``calibrate``, and after
  ``calibrate(..., groups=column)``, which calibrates each group on its own?
- ``localize``: the shift test of ``uncertainty`` with a calibration slice.
  Does ``calibrate(..., localize="members")``, which calibrates rows in bins
  of member disagreement, keep the 80% interval on rows past the training
  range better than one calibration of every row?

Every table is a public GitHub mirror. Three seeds, 0, 1, and 2. A finished
seed is skipped on a rerun.

    JAX_PLATFORMS=cpu python examples/benchmark_uncertainty.py --stage all
    python examples/benchmark_uncertainty.py --stage report
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import (
    DATA_DIR,
    RAW,
    SOURCES,
    _align_categories,
    _crps,
    _fit_gbt,
    _path,
    _sample,
    _strings,
    abalone,
    california,
    diamonds,
    taxis,
)

from ashlar import ASRRegressor, tabm_piecewise

SEEDS = (0, 1, 2)
N_DRAWS = 400
SOURCES["bike_hour.csv"] = f"{RAW}/udacity/deep-learning-v2-pytorch/master/project-bikesharing/Bike-Sharing-Dataset/hour.csv"


# ---------------------------------------------------------------- data


def bike(target: str):
    """Capital Bikeshare rentals per hour, 2011-2012. ``casual`` and ``registered`` sum to ``cnt``, so both leave."""
    frame = pd.read_csv(_path("bike_hour.csv"))
    y = frame[target].to_numpy(np.float64)
    frame = frame.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
    for column in ("season", "weathersit"):
        frame[column] = frame[column].astype(str)
    return _strings(frame), y


def taxi_pickups():
    """NYC taxi pickups per half hour, July 2014 to January 2015, from the previous day and the same slot a week ago."""
    frame = pd.read_csv(_path("nyc_taxi.csv"), parse_dates=["timestamp"])
    counts = frame["value"].to_numpy(np.float64)
    stamp = frame["timestamp"]
    lags = {f"lag_{lag}": np.roll(counts, lag) for lag in (*range(1, 49), 336)}
    out = pd.DataFrame({"slot": (stamp.dt.hour * 2 + stamp.dt.minute // 30).astype(np.float64),
                        "weekday": stamp.dt.dayofweek.astype(np.float64), **lags})
    keep = slice(336, None)
    return out.iloc[keep].reset_index(drop=True), counts[keep]


def abalone_rings():
    return abalone()


CALIBRATION_TABLES = {
    "california": california,
    "diamonds_price": lambda: diamonds(log=False),
    "abalone": abalone,
    "taxi_tip": lambda: taxis("tip"),
    "bike_count": lambda: bike("cnt"),
}
COUNT_TABLES = {
    "bike_count": lambda: bike("cnt"),
    "bike_casual": lambda: bike("casual"),
    "taxi_pickups": taxi_pickups,
    "abalone_rings": abalone_rings,
}
# The column whose groups each calibration table is scored within, and calibrated by in the groups stage.
GROUP_COLUMNS = {
    "california": "ocean_proximity",
    "diamonds_price": "cut",
    "abalone": "sex",
    "taxi_tip": "color",
    "bike_count": "hr",
}
UNCERTAINTY_TABLES = {
    # The column held back above its 80th percentile for the shift test.
    "california": (california, "median_income"),
    "diamonds_price": (lambda: _sample(*diamonds(log=False), 20_000), "carat"),
    "bike_count": (lambda: bike("cnt"), "temp"),
}


# ---------------------------------------------------------------- shared


def _three_way(x, y, seed):
    """60% train, 20% calibration, 20% test."""
    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4, random_state=seed)
    x_cal, x_test, y_cal, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=seed)
    return x_train, x_cal, x_test, y_train, y_cal, y_test


def _interval(lo, hi, y):
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "width": float(np.mean(hi - lo))}


def _conformal(scores, coverage):
    scores = np.sort(np.asarray(scores))
    rank = math.ceil((len(scores) + 1) * coverage)
    return math.inf if rank > len(scores) else float(scores[rank - 1])


def _lgbm_quantiles(x_train, y_train, levels, seed):
    models = []
    for tau in levels:
        model, frame = _fit_gbt("lightgbm", "regression", True, seed, x_train, y_train, objective="quantile", alpha=tau)
        models.append(model)
    return lambda x: np.stack([m.predict(_align_categories(frame, x)) for m in models], axis=1)


# ---------------------------------------------------------------- stage: calibration


def stage_calibration(results):
    out = results.setdefault("calibration", {})
    for name, loader in _chosen(CALIBRATION_TABLES).items():
        x, y = loader()
        rows = out.setdefault(name, {"rows": int(len(y)), "runs": {}})
        for seed in SEEDS:
            if str(seed) in rows["runs"]:
                continue
            x_train, x_cal, x_test, y_train, y_cal, y_test = _three_way(x, y, seed)
            run = {}
            started = time.perf_counter()
            mixture = ASRRegressor(task="mixture", random_state=seed).fit(x_train, y_train)
            for coverage in (0.8, 0.9):
                raw = mixture.predict_interval(x_test, coverage, n_samples=N_DRAWS, seed=seed, calibrated=False)
                run[f"mixture_raw_{coverage}"] = _interval(raw[:, 0], raw[:, 1], y_test)
            mixture.calibrate(x_cal, y_cal, n_samples=N_DRAWS, seed=seed)
            for coverage in (0.8, 0.9):
                cal = mixture.predict_interval(x_test, coverage, n_samples=N_DRAWS, seed=seed)
                run[f"mixture_cal_{coverage}"] = _interval(cal[:, 0], cal[:, 1], y_test)
            run["mixture_seconds"] = time.perf_counter() - started

            quantile = ASRRegressor(task="quantile", random_state=seed).fit(x_train, y_train)
            raw = quantile.predict_interval(x_test, calibrated=False)
            run["quantile_raw_0.8"] = _interval(raw[:, 0], raw[:, 1], y_test)
            cal = quantile.calibrate(x_cal, y_cal).predict_interval(x_test)
            run["quantile_cal_0.8"] = _interval(cal[:, 0], cal[:, 1], y_test)

            predict = _lgbm_quantiles(x_train, y_train, (0.1, 0.9), seed)
            raw = predict(x_test)
            run["lightgbm_raw_0.8"] = _interval(raw[:, 0], raw[:, 1], y_test)
            levels = predict(x_cal)
            margin = _conformal(np.maximum(levels[:, 0] - y_cal, y_cal - levels[:, 1]), 0.8)
            run["lightgbm_cal_0.8"] = _interval(raw[:, 0] - margin, raw[:, 1] + margin, y_test)
            rows["runs"][str(seed)] = run
            print(name, seed, {k: round(v["coverage"], 3) for k, v in run.items() if isinstance(v, dict)}, flush=True)
            _save(results)


# ---------------------------------------------------------------- stage: groups


def _by_group(lo, hi, y, groups, min_rows=50):
    """Coverage and width within each group with at least ``min_rows`` test rows."""
    out = {}
    for key in sorted(set(groups.tolist()), key=str):
        rows = groups == key
        if rows.sum() >= min_rows:
            out[str(key)] = {**_interval(lo[rows], hi[rows], y[rows]), "rows": int(rows.sum())}
    return out


def stage_groups(results):
    out = results.setdefault("groups", {})
    for name, loader in _chosen(CALIBRATION_TABLES).items():
        x, y = loader()
        column = GROUP_COLUMNS[name]
        rows = out.setdefault(name, {"rows": int(len(y)), "column": column, "runs": {}})
        for seed in SEEDS:
            if str(seed) in rows["runs"]:
                continue
            x_train, x_cal, x_test, y_train, y_cal, y_test = _three_way(x, y, seed)
            groups = x_test[column].to_numpy()
            run = {"calibration_rows": {str(k): int(v) for k, v in x_cal[column].value_counts().items()}}
            started = time.perf_counter()
            model = ASRRegressor(task="mixture", random_state=seed).fit(x_train, y_train)
            raw = model.predict_interval(x_test, 0.8, n_samples=N_DRAWS, seed=seed, calibrated=False)
            run["raw"] = _by_group(raw[:, 0], raw[:, 1], y_test, groups)
            model.calibrate(x_cal, y_cal, n_samples=N_DRAWS, seed=seed)
            pooled = model.predict_interval(x_test, 0.8, n_samples=N_DRAWS, seed=seed)
            run["pooled"] = _by_group(pooled[:, 0], pooled[:, 1], y_test, groups)
            run["pooled_all"] = _interval(pooled[:, 0], pooled[:, 1], y_test)
            model.calibrate(x_cal, y_cal, n_samples=N_DRAWS, seed=seed, groups=column)
            grouped = model.predict_interval(x_test, 0.8, n_samples=N_DRAWS, seed=seed)
            run["grouped"] = _by_group(grouped[:, 0], grouped[:, 1], y_test, groups)
            run["grouped_all"] = _interval(grouped[:, 0], grouped[:, 1], y_test)
            run["own_tables"] = sorted(model.calibration_["groups"])
            run["seconds"] = time.perf_counter() - started
            rows["runs"][str(seed)] = run
            spread = {
                key: (min(v["coverage"] for v in run[key].values()), max(v["coverage"] for v in run[key].values()))
                for key in ("raw", "pooled", "grouped")
            }
            print(name, seed, {key: tuple(round(v, 3) for v in value) for key, value in spread.items()}, flush=True)
            _save(results)


# ---------------------------------------------------------------- stage: localize


def stage_localize(results):
    out = results.setdefault("localize", {})
    for name, (loader, shift_column) in _chosen(UNCERTAINTY_TABLES).items():
        x, y = loader()
        rows = out.setdefault(name, {"rows": int(len(y)), "shift_column": shift_column, "runs": {}})
        for seed in SEEDS:
            if str(seed) in rows["runs"]:
                continue
            # Train and calibrate below the column's 80th percentile; test inside and above it.
            column = x[shift_column].to_numpy(np.float64)
            inside = column <= np.quantile(column, 0.8)
            xi, yi = x[inside].reset_index(drop=True), y[inside]
            x_fit, x_cal, x_in, y_fit, y_cal, y_in = _three_way(xi, yi, seed)
            x_out, y_out = x[~inside].reset_index(drop=True), y[~inside]
            started = time.perf_counter()
            model = tabm_piecewise(task="gaussian", random_state=seed).fit(x_fit, y_fit)
            run = {}
            for label, calibrate in (("raw", None), ("pooled", {}), ("members", {"localize": "members"})):
                if calibrate is not None:
                    model.calibrate(x_cal, y_cal, n_samples=N_DRAWS, seed=seed, **calibrate)
                for where, xs, ys in (("in_range", x_in, y_in), ("above_range", x_out, y_out)):
                    interval = model.predict_interval(xs, 0.8, n_samples=N_DRAWS, seed=seed, calibrated=calibrate is not None)
                    run[f"{label}_{where}"] = _interval(interval[:, 0], interval[:, 1], ys)
            run["edges"] = model.calibration_["by"]["members"]
            bins = model._row_groups(x_out, model.calibration_, None)
            run["above_range_bins"] = {str(k): int(v) for k, v in zip(*np.unique(bins, return_counts=True))}
            run["seconds"] = time.perf_counter() - started
            rows["runs"][str(seed)] = run
            print(name, seed, {k: round(v["coverage"], 3) for k, v in run.items() if isinstance(v, dict) and "coverage" in v}, flush=True)
            _save(results)


# ---------------------------------------------------------------- stage: counts


COUNT_MODELS = {
    "gaussian": {"task": "gaussian"},
    "mixture": {"task": "mixture"},
    "poisson": {"task": "poisson"},
    "negative_binomial": {"task": "negative_binomial"},
    "zinb": {"task": "negative_binomial", "head_kwargs": {"zero_inflated": True}},
}


def _count_scores(draws, mean, y, log_mass=None):
    """``draws`` is ``(rows, n)``. Interval levels from the draws, integer-valued for a count model."""
    lo, hi = np.quantile(draws, [0.1, 0.9], axis=1, method="inverted_cdf")
    scores = {
        "crps": _crps(draws, y),
        "rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
        "zero_share": float(np.mean(draws == 0)),
        **{f"{key}80": value for key, value in _interval(lo, hi, y).items()},
    }
    if log_mass is not None:
        scores["nll"] = float(-np.mean(log_mass))
    return scores


def stage_counts(results):
    out = results.setdefault("counts", {})
    for name, loader in _chosen(COUNT_TABLES).items():
        x, y = loader()
        rows = out.setdefault(name, {"rows": int(len(y)), "mean": float(np.mean(y)), "var": float(np.var(y)),
                                     "zeros": float(np.mean(y == 0)), "runs": {}})
        for seed in SEEDS:
            if str(seed) in rows["runs"]:
                continue
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
            run = {}
            for label, params in COUNT_MODELS.items():
                started = time.perf_counter()
                model = ASRRegressor(random_state=seed, **params).fit(x_train, y_train)
                draws = model.sample(x_test, n=N_DRAWS, seed=seed).T
                counted = params["task"] in {"poisson", "negative_binomial"}
                run[label] = _count_scores(draws, model.predict(x_test), y_test,
                                           model.log_prob(x_test, y_test) if counted else None)
                run[label]["seconds"] = time.perf_counter() - started
            started = time.perf_counter()
            model, frame = _fit_gbt("lightgbm", "regression", True, seed, x_train, y_train, objective="poisson")
            rate = np.maximum(model.predict(_align_categories(frame, x_test)), 1e-9)
            draws = np.random.default_rng(seed).poisson(rate[:, None], size=(len(rate), N_DRAWS)).astype(float)
            run["lightgbm_poisson"] = _count_scores(draws, rate, y_test, stats.poisson.logpmf(y_test, rate))
            run["lightgbm_poisson"]["seconds"] = time.perf_counter() - started
            rows["runs"][str(seed)] = run
            print(name, seed, {k: round(v["crps"], 3) for k, v in run.items()}, flush=True)
            _save(results)


# ---------------------------------------------------------------- stage: uncertainty


def _selective_rmse(error, score, keep=0.5):
    order = np.argsort(score)
    kept = order[: int(len(order) * keep)]
    return float(np.sqrt(np.mean(np.square(error[kept]))))


def stage_uncertainty(results):
    out = results.setdefault("uncertainty", {})
    for name, (loader, shift_column) in _chosen(UNCERTAINTY_TABLES).items():
        x, y = loader()
        rows = out.setdefault(name, {"rows": int(len(y)), "shift_column": shift_column, "runs": {}})
        for seed in SEEDS:
            if str(seed) in rows["runs"]:
                continue
            run = {}
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
            model = tabm_piecewise(task="gaussian", random_state=seed).fit(x_train, y_train)
            parts = model.predict_uncertainty(x_test, n_samples=200, seed=seed)
            error = model.predict(x_test) - y_test
            random_score = np.random.default_rng(seed).uniform(size=len(error))
            run["random"] = {"rmse_kept_half": _selective_rmse(error, random_score)}
            for key in ("epistemic", "aleatoric", "total"):
                run[key] = {
                    "spearman_abs_error": float(stats.spearmanr(parts[key], np.abs(error)).statistic),
                    "rmse_kept_half": _selective_rmse(error, parts[key]),
                    "mean": float(np.mean(parts[key])),
                }
            run["rmse_all"] = float(np.sqrt(np.mean(np.square(error))))

            # Shift: train below the column's 80th percentile, then compare held-out rows inside and above it.
            column = x[shift_column].to_numpy(np.float64)
            inside = column <= np.quantile(column, 0.8)
            xi, yi = x[inside].reset_index(drop=True), y[inside]
            x_fit, x_in, y_fit, y_in = train_test_split(xi, yi, test_size=0.2, random_state=seed)
            x_out, y_out = x[~inside].reset_index(drop=True), y[~inside]
            shifted = tabm_piecewise(task="gaussian", random_state=seed).fit(x_fit, y_fit)
            for label, xs, ys in (("in_range", x_in, y_in), ("above_range", x_out, y_out)):
                parts = shifted.predict_uncertainty(xs, n_samples=200, seed=seed)
                err = shifted.predict(xs) - ys
                run[f"shift_{label}"] = {
                    "rmse": float(np.sqrt(np.mean(np.square(err)))),
                    "epistemic": float(np.mean(parts["epistemic"])),
                    "aleatoric": float(np.mean(parts["aleatoric"])),
                }
            rows["runs"][str(seed)] = run
            print(name, seed, json.dumps({k: v for k, v in run.items() if k.startswith("shift")}), flush=True)
            _save(results)


# ---------------------------------------------------------------- report


def _mean_sd(values, digits):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std():.{digits}f}"


def _runs(stage):
    """One stage's tables, from its file and every per-table file."""
    merged = {}
    for path in sorted(DATA_DIR.glob(f"uq-{stage}*.json")):
        for name, rows in json.loads(path.read_text())[stage].items():
            merged.setdefault(name, {**rows, "runs": {}})["runs"].update(rows["runs"])
    return merged


def stage_report(_results):
    print("## Calibration\n")
    for name, table in _runs("calibration").items():
        runs = list(table["runs"].values())
        print(f"### {name}, {table['rows']:,} rows, {len(runs)} splits\n")
        print("| Method | Coverage | Width |\n|---|---:|---:|")
        for key in runs[0]:
            if isinstance(runs[0][key], dict):
                print(f"| {key} | {_mean_sd([r[key]['coverage'] for r in runs], 3)} | {_mean_sd([r[key]['width'] for r in runs], 3)} |")
        print()
    print("## Counts\n")
    for name, table in _runs("counts").items():
        runs = list(table["runs"].values())
        print(f"### {name}, {table['rows']:,} rows, mean {table['mean']:.2f}, variance {table['var']:.1f}, "
              f"{table['zeros']:.0%} zeros, {len(runs)} splits\n")
        print("| Model | CRPS | NLL | RMSE | 80% coverage | 80% width | Share of draws at 0 |\n|---|---:|---:|---:|---:|---:|---:|")
        for key in runs[0]:
            nll = _mean_sd([r[key]["nll"] for r in runs], 3) if "nll" in runs[0][key] else "—"
            print(f"| {key} | {_mean_sd([r[key]['crps'] for r in runs], 3)} | {nll} | {_mean_sd([r[key]['rmse'] for r in runs], 3)} | "
                  f"{_mean_sd([r[key]['coverage80'] for r in runs], 3)} | {_mean_sd([r[key]['width80'] for r in runs], 2)} | "
                  f"{_mean_sd([r[key]['zero_share'] for r in runs], 3)} |")
        print()
    print("## Uncertainty\n")
    for name, table in _runs("uncertainty").items():
        runs = list(table["runs"].values())
        print(f"### {name}, {table['rows']:,} rows, shift on {table['shift_column']}, {len(runs)} splits\n")
        print(f"RMSE on every test row: {_mean_sd([r['rmse_all'] for r in runs], 3)}\n")
        print("| Rank rows by | Spearman with abs error | RMSE of the half it is surest of |\n|---|---:|---:|")
        for key in ("epistemic", "aleatoric", "total"):
            print(f"| {key} | {_mean_sd([r[key]['spearman_abs_error'] for r in runs], 3)} | {_mean_sd([r[key]['rmse_kept_half'] for r in runs], 3)} |")
        print(f"| random | — | {_mean_sd([r['random']['rmse_kept_half'] for r in runs], 3)} |\n")
        print("| Held-out rows | RMSE | Mean epistemic | Mean aleatoric |\n|---|---:|---:|---:|")
        for label in ("in_range", "above_range"):
            key = f"shift_{label}"
            print(f"| {label} | {_mean_sd([r[key]['rmse'] for r in runs], 3)} | {_mean_sd([r[key]['epistemic'] for r in runs], 3)} | "
                  f"{_mean_sd([r[key]['aleatoric'] for r in runs], 3)} |")
        print()


def report_groups():
    print("## Coverage within groups\n")
    for name, table in _runs("groups").items():
        runs = list(table["runs"].values())
        print(f"### {name}, {table['rows']:,} rows, groups by {table['column']}, {len(runs)} splits\n")
        print("| Group | Test rows | Raw | Calibrated on every row | Calibrated per group |\n|---|---:|---:|---:|---:|")
        keys = sorted({key for run in runs for key in run["pooled"]}, key=str)
        for key in keys:
            have = [run for run in runs if key in run["pooled"]]
            cells = [_mean_sd([run[stage][key]["coverage"] for run in have], 3) for stage in ("raw", "pooled", "grouped")]
            print(f"| {key} | {int(np.mean([run['pooled'][key]['rows'] for run in have]))} | " + " | ".join(cells) + " |")
        for stage in ("raw", "pooled", "grouped"):
            worst = [max(abs(v["coverage"] - 0.8) for v in run[stage].values()) for run in runs]
            print(f"\n{stage}: largest gap from 80% in any group {_mean_sd(worst, 3)}", end="")
        print(f"\nall rows: every-row calibration {_mean_sd([r['pooled_all']['coverage'] for r in runs], 3)}, "
              f"per group {_mean_sd([r['grouped_all']['coverage'] for r in runs], 3)}; width "
              f"{_mean_sd([r['pooled_all']['width'] for r in runs], 3)} and {_mean_sd([r['grouped_all']['width'] for r in runs], 3)}\n")
    print("## Coverage past the training range\n")
    for name, table in _runs("localize").items():
        runs = list(table["runs"].values())
        print(f"### {name}, shift on {table['shift_column']}, {len(runs)} splits\n")
        print("| Calibration | In range | Width | Above range | Width |\n|---|---:|---:|---:|---:|")
        for label in ("raw", "pooled", "members"):
            cells = []
            for where in ("in_range", "above_range"):
                cells += [_mean_sd([r[f"{label}_{where}"]["coverage"] for r in runs], 3), _mean_sd([r[f"{label}_{where}"]["width"] for r in runs], 3)]
            print(f"| {label} | " + " | ".join(cells) + " |")
        print()


# ---------------------------------------------------------------- main


TABLES: list[str] = []


def _chosen(tables: dict) -> dict:
    return {name: value for name, value in tables.items() if not TABLES or name in TABLES}


def _out_path(stage: str) -> Path:
    """With ``--tables``, a file per table set, so several processes can run one stage side by side."""
    suffix = "".join(f"-{name}" for name in TABLES)
    return DATA_DIR / f"uq-{stage}{suffix}.json"


def _save(results):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for stage, value in results.items():
        _out_path(stage).write_text(json.dumps({stage: value}, indent=1, sort_keys=True))


STAGES = {
    "calibration": stage_calibration,
    "counts": stage_counts,
    "uncertainty": stage_uncertainty,
    "groups": stage_groups,
    "localize": stage_localize,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all", "report"], default="all")
    parser.add_argument("--tables", nargs="*", default=[], help="Only these tables, written to their own file.")
    args = parser.parse_args()
    TABLES[:] = args.tables
    if args.stage == "report":
        stage_report(None)
        report_groups()
        return
    for name, stage in STAGES.items():
        if args.stage in {name, "all"}:
            path = _out_path(name)
            results = json.loads(path.read_text()) if path.exists() else {}
            stage(results)
            _save(results)


if __name__ == "__main__":
    main()
