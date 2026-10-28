"""Score a window-locked transformer and a full-history transformer against lag XGBoost.

Both tests use the same five series, the same chronological cut, and the same
forecast origins. Lookback is 48 and horizon is 12, in each series' own step.
The last 20% is the test. Each transformer is fit on the prefix before that
cut. The last 10% of that prefix chooses the epoch, then the network is
reinitialized and trained for that many epochs on every pre-test origin.
``MultiOutputRegressor(XGBRegressor())`` with no arguments is fit on the 48 raw
lags of those same pre-test origins. It has no early stopping.

The window-locked transformer sees only those 48 lags. The full-history
transformer sees the entire prefix before the origin. Test MAE is the
comparison: lower or equal beats or matches. RMSE and R² are recorded on the
same points. This does not change the residual regression default and does not
refit earlier result files.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_forecast.py
"""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xgboost
from sklearn.multioutput import MultiOutputRegressor

from ashlar import tabular
from ashlar.forecast import (
    FORECAST_BATCH_SIZE,
    FORECAST_D_MODEL,
    FORECAST_HEADS,
    FORECAST_HORIZON,
    FORECAST_LAYERS,
    FORECAST_LEARNING_RATE,
    FORECAST_LOOKBACK,
    FORECAST_MAX_EPOCHS,
    FORECAST_PATIENCE,
    FORECAST_TEST_FRACTION,
    FORECAST_VAL_FRACTION,
    FullHistoryForecaster,
    TimeSeriesForecaster,
    _forecast_windows,
    _history_batch,
    _history_mse,
    _history_origins,
    _init_history,
    _pack_history,
)
from ashlar.recipes import DEFAULT_RECIPE

DATA_DIR = Path("/tmp/ashlar-bench/forecast")
OUT_PATH = Path("/tmp/ashlar-bench/forecast-results.json")

DATASETS = (
    {
        "name": "monthly-sunspots",
        "column": "Sunspots",
        "sampling": "month",
        "source": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
        "rows": 2820,
        "description": "Monthly sunspot counts. Columns Month and Sunspots, file order, no missing values.",
    },
    {
        "name": "daily-min-temperatures",
        "column": "Temp",
        "sampling": "day",
        "source": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
        "rows": 3650,
        "description": "Daily minimum temperature in Melbourne. Columns Date and Temp, file order, no missing values.",
    },
    {
        "name": "etth1-ot",
        "column": "OT",
        "sampling": "hour",
        "source": "https://github.com/zhouhaoyi/ETDataset",
        "rows": 17420,
        "description": "ETTh1 oil temperature, column OT. Hourly. The other load channels are not used.",
    },
    {
        "name": "bike-hour-cnt",
        "column": "cnt",
        "sampling": "hour",
        "source": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
        "rows": 17379,
        "description": "UCI Bike Sharing hourly rental count, column cnt, file order. Missing hours are not filled. Weather and calendar columns are not used.",
    },
    {
        "name": "electricity-client0",
        "column": "0",
        "sampling": "hour",
        "source": "https://github.com/laiguokun/multivariate-time-series-data",
        "rows": 26304,
        "description": "Electricity load, first client only (column 0 of the Lai et al. matrix). The other 320 clients are not used.",
    },
)


def _load(spec: dict) -> np.ndarray:
    name = spec["name"]
    if name == "monthly-sunspots":
        frame = pd.read_csv(DATA_DIR / "monthly-sunspots.csv")
        values = frame["Sunspots"].to_numpy(dtype=np.float64)
    elif name == "daily-min-temperatures":
        frame = pd.read_csv(DATA_DIR / "daily-min-temperatures.csv")
        values = frame["Temp"].to_numpy(dtype=np.float64)
    elif name == "etth1-ot":
        frame = pd.read_csv(DATA_DIR / "ETTh1.csv")
        values = frame["OT"].to_numpy(dtype=np.float64)
    elif name == "bike-hour-cnt":
        with zipfile.ZipFile(DATA_DIR / "bike.zip") as archive:
            frame = pd.read_csv(archive.open("hour.csv"))
        if not frame["instant"].is_monotonic_increasing:
            raise RuntimeError("bike instant is not monotonic; refusing to reorder the series.")
        values = frame["cnt"].to_numpy(dtype=np.float64)
    elif name == "electricity-client0":
        values = np.loadtxt(DATA_DIR / "electricity.txt.gz", delimiter=",", usecols=0, dtype=np.float64)
        if values.shape != (26304,) or values[0] != 14.0:
            raise RuntimeError(f"electricity column 0 has shape {values.shape} and first value {values[0]}.")
    else:
        raise RuntimeError(name)
    if values.shape != (spec["rows"],) or not np.isfinite(values).all():
        raise RuntimeError(f"{name} is not a finite series of length {spec['rows']}.")
    return np.ascontiguousarray(values)


def _scores(pred: np.ndarray, target: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise RuntimeError(f"forecast predictions have shape {pred.shape} against {target.shape}.")
    error = pred - target
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    total = float(np.sum(np.square(target - target.mean())))
    r2 = float("nan") if total == 0.0 else float(1.0 - np.sum(np.square(error)) / total)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _load_payload() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"datasets": {}}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _wins(payload: dict, key: str) -> list[str]:
    names = []
    for name, row in payload["datasets"].items():
        model = row.get(key)
        baseline = row.get("xgboost")
        if model is None or baseline is None:
            continue
        if model["mae"] <= baseline["mae"]:
            names.append(name)
    return names


def _check_constants() -> None:
    model = tabular()
    if model.task != "regression" or model.standardize_target is not True or model.backbone != "residual_mlp":
        raise RuntimeError("the residual regression default changed.")
    if DEFAULT_RECIPE["task"] != "regression" or DEFAULT_RECIPE["standardize_target"] is not True:
        raise RuntimeError("DEFAULT_RECIPE changed.")
    if DEFAULT_RECIPE["backbone"] != "residual_mlp" or DEFAULT_RECIPE["categorical_embedding_dim"] != 8:
        raise RuntimeError("DEFAULT_RECIPE changed.")
    expected = {
        "lookback": 48,
        "horizon": 12,
        "d_model": 32,
        "heads": 4,
        "layers": 2,
        "learning_rate": 1e-3,
        "batch_size": 128,
        "max_epochs": 30,
        "patience": 8,
        "val_fraction": 0.1,
        "test_fraction": 0.2,
    }
    actual = {
        "lookback": FORECAST_LOOKBACK,
        "horizon": FORECAST_HORIZON,
        "d_model": FORECAST_D_MODEL,
        "heads": FORECAST_HEADS,
        "layers": FORECAST_LAYERS,
        "learning_rate": FORECAST_LEARNING_RATE,
        "batch_size": FORECAST_BATCH_SIZE,
        "max_epochs": FORECAST_MAX_EPOCHS,
        "patience": FORECAST_PATIENCE,
        "val_fraction": FORECAST_VAL_FRACTION,
        "test_fraction": FORECAST_TEST_FRACTION,
    }
    if actual != expected:
        raise RuntimeError(f"forecast constants changed: {actual}")


def _probe() -> dict:
    net = _init_history(jax.random.PRNGKey(0), 16, 4)
    series = np.linspace(-1.0, 1.0, 80, dtype=np.float32)
    origins = np.array([40, 56, 64, 70], dtype=np.int64)
    packed = [jnp.asarray(part) for part in _pack_history(series, origins, 16, max_patches=4)]
    target = jnp.asarray(np.random.default_rng(1).normal(size=(4, 4)), dtype=jnp.float32)
    loss, grads = eqx.filter_value_and_grad(_history_mse)(net, *packed, target)
    pred = _history_batch(net, *packed)
    gap = abs(float(loss) - float(jnp.mean(jnp.square(pred - target))))
    leaves = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    finite = all(np.isfinite(leaf).all() for leaf in leaves)
    patch_grad = float(jnp.sum(jnp.abs(grads.patch_weight)))
    position_grad = float(jnp.sum(jnp.abs(grads.position)))
    if not finite or gap > 1e-5 or patch_grad <= 0.0 or position_grad <= 0.0:
        raise RuntimeError("full-history probe failed.")
    left = np.zeros(80, dtype=np.float32)
    right = np.zeros(80, dtype=np.float32)
    left[:8] = 3.0
    origin = np.array([80], dtype=np.int64)
    pred_left = np.asarray(net(*[jnp.asarray(part[0]) for part in _pack_history(left, origin, 16, max_patches=5)]))
    pred_right = np.asarray(net(*[jnp.asarray(part[0]) for part in _pack_history(right, origin, 16, max_patches=5)]))
    if float(np.max(np.abs(pred_left - pred_right))) <= 1e-5:
        raise RuntimeError("full-history probe ignored the distant past.")
    return {
        "differentiable": True,
        "loss_matches_mse": True,
        "patch_grad": patch_grad,
        "position_grad": position_grad,
        "distant_past_changes_prediction": True,
        "residual_default_task": "regression",
        "residual_default_standardize_target": True,
        "residual_default_backbone": "residual_mlp",
        "categorical_embedding_dim": 8,
    }


def _fit_split(series: np.ndarray):
    n = len(series)
    test_start = int(n * (1.0 - FORECAST_TEST_FRACTION))
    origins, inputs, targets = _forecast_windows(series, FORECAST_LOOKBACK, FORECAST_HORIZON)
    if not np.array_equal(origins, _history_origins(n, FORECAST_LOOKBACK, FORECAST_HORIZON)):
        raise RuntimeError("window and full-history origins differ.")
    pre = origins + FORECAST_HORIZON <= test_start
    test = (origins >= test_start) & (origins + FORECAST_HORIZON <= n)
    if int(test.sum()) < 1 or int(pre.sum()) < 2:
        raise RuntimeError("the chronological forecast split is empty.")
    fit = series[:test_start]
    val_start = int(len(fit) * (1.0 - FORECAST_VAL_FRACTION))
    fit_origins = origins[pre]
    expect_train = int(np.sum(fit_origins + FORECAST_HORIZON <= val_start))
    expect_val = int(np.sum((fit_origins >= val_start) & (fit_origins + FORECAST_HORIZON <= test_start)))
    return {
        "test_start": int(test_start),
        "val_start": int(val_start),
        "origins": origins,
        "inputs": inputs,
        "targets": targets,
        "pre": pre,
        "test": test,
        "expect_train": expect_train,
        "expect_val": expect_val,
        "mean_baseline_mae": float(np.mean(np.abs(targets[test] - fit.mean()))),
    }


def _check_counts(model, split: dict, label: str) -> None:
    if model.n_train_ != split["expect_train"] or model.n_val_ != split["expect_val"]:
        raise RuntimeError(
            f"{label} split counts {model.n_train_}/{model.n_val_} "
            f"do not match {split['expect_train']}/{split['expect_val']}."
        )


def _model_record(model, pred, target, seconds: float) -> dict:
    row = _scores(pred, target)
    row.update(
        {
            "seconds": float(seconds),
            "best_epoch": int(model.best_epoch_),
            "selection_mse": float(model.selection_mse_),
            "n_train": int(model.n_train_),
            "n_val": int(model.n_val_),
        }
    )
    return row


def main() -> None:
    _check_constants()
    payload = _load_payload()
    protocol = {
        "lookback": FORECAST_LOOKBACK,
        "horizon": FORECAST_HORIZON,
        "test_fraction": FORECAST_TEST_FRACTION,
        "val_fraction": FORECAST_VAL_FRACTION,
        "d_model": FORECAST_D_MODEL,
        "heads": FORECAST_HEADS,
        "layers": FORECAST_LAYERS,
        "learning_rate": FORECAST_LEARNING_RATE,
        "batch_size": FORECAST_BATCH_SIZE,
        "max_epochs": FORECAST_MAX_EPOCHS,
        "patience": FORECAST_PATIENCE,
        "metric": "mae",
        "match_rule": "transformer_mae <= xgboost_mae",
        "xgb": "sklearn.multioutput.MultiOutputRegressor(xgboost.XGBRegressor())",
        "window": "last 48 observations, one token each, learned positions",
        "full_history": (
            "entire prefix; last 48 steps are tokens; older steps are non-overlapping "
            "patches of 48; sinusoidal age from the origin plus a learned recent offset"
        ),
    }
    if "protocol" in payload and payload["protocol"] != protocol:
        raise RuntimeError("forecast-results.json was written with a different protocol.")
    payload["protocol"] = protocol
    if "probe" not in payload:
        print("probe", flush=True)
        payload["probe"] = _probe()
        _save(payload)
    datasets = payload.setdefault("datasets", {})
    for spec in DATASETS:
        name = spec["name"]
        row = datasets.setdefault(name, {})
        row["description"] = spec["description"]
        row["source"] = spec["source"]
        row["column"] = spec["column"]
        row["sampling"] = spec["sampling"]
        if "xgboost" in row and "window" in row and "full_history" in row:
            print(f"skip {name}", flush=True)
            continue
        print(f"load {name} {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", flush=True)
        series = _load(spec)
        split = _fit_split(series)
        row["rows"] = int(len(series))
        row["test_start"] = split["test_start"]
        row["val_start"] = split["val_start"]
        row["n_pretest_origins"] = int(split["pre"].sum())
        row["n_test_origins"] = int(split["test"].sum())
        row["n_test_points"] = int(split["test"].sum() * FORECAST_HORIZON)
        row["mean_baseline_mae"] = split["mean_baseline_mae"]
        target = split["targets"][split["test"]]
        if "xgboost" not in row:
            print(f"xgboost {name}", flush=True)
            started = time.perf_counter()
            regressor = MultiOutputRegressor(xgboost.XGBRegressor())
            regressor.fit(split["inputs"][split["pre"]], split["targets"][split["pre"]])
            pred = np.asarray(regressor.predict(split["inputs"][split["test"]]))
            row["xgboost"] = _scores(pred, target)
            row["xgboost"]["seconds"] = float(time.perf_counter() - started)
            row["xgboost"]["n_train"] = int(split["pre"].sum())
            _save(payload)
            print(f"xgboost {name} mae {row['xgboost']['mae']:.6f} seconds {row['xgboost']['seconds']:.1f}", flush=True)
        if "window" not in row:
            print(f"window {name}", flush=True)
            started = time.perf_counter()
            model = TimeSeriesForecaster(verbose=True).fit(series[: split["test_start"]])
            _check_counts(model, split, "window")
            pred = model.predict(split["inputs"][split["test"]])
            row["window"] = _model_record(model, pred, target, time.perf_counter() - started)
            _save(payload)
            print(f"window {name} mae {row['window']['mae']:.6f} epoch {row['window']['best_epoch']}", flush=True)
        if "full_history" not in row:
            print(f"full-history {name}", flush=True)
            started = time.perf_counter()
            model = FullHistoryForecaster(verbose=True).fit(series[: split["test_start"]])
            _check_counts(model, split, "full-history")
            pred = model.predict(series, split["origins"][split["test"]])
            row["full_history"] = _model_record(model, pred, target, time.perf_counter() - started)
            _save(payload)
            print(
                f"full-history {name} mae {row['full_history']['mae']:.6f} epoch {row['full_history']['best_epoch']}",
                flush=True,
            )
        payload["window_beats_or_matches"] = _wins(payload, "window")
        payload["full_history_beats_or_matches"] = _wins(payload, "full_history")
        _save(payload)
    payload["window_beats_or_matches"] = _wins(payload, "window")
    payload["full_history_beats_or_matches"] = _wins(payload, "full_history")
    _save(payload)
    print("window beats or matches", payload.get("window_beats_or_matches"), flush=True)
    print("full-history beats or matches", payload.get("full_history_beats_or_matches"), flush=True)
    print("EXIT:0", flush=True)


if __name__ == "__main__":
    main()
