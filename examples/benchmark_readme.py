"""The numbers in the README: Ashlar beside XGBoost and LightGBM.

Four stages, each written to its own key of one JSON file:

- ``point``: default ``TabularEstimator()`` or ``classifier()`` against default
  XGBoost and LightGBM, and against early-stopped versions of both that see the
  same validation rows Ashlar holds out. Five public tables, three random
  80/20 splits each.
- ``dist``: the forecast heads (``gaussian``, ``mixture``, ``flow``,
  ``histogram``, ``quantile``) attached to a table through a custom head,
  against XGBoost and LightGBM quantile regression and a gradient-boosted mean
  with a constant Gaussian residual. Pinball loss, 80% coverage, and a
  sample-based CRPS on held-out rows.
- ``embed``: learned categorical embeddings against one-hot inside Ashlar,
  and against native categorical splits in LightGBM and XGBoost, on a table
  whose two strongest columns are taxi zones with ~200 levels each.
- ``forecast``: the six forecast heads on Melbourne daily minimum
  temperatures, beside LightGBM on the same lag window.

Every table is downloaded from a public GitHub mirror into ``/tmp/ashlar-bench``.
Each stage writes ``readme-<stage>.json`` there, so stages can run side by side.
A finished seed is skipped on a rerun.

    JAX_PLATFORMS=cpu python examples/benchmark_readme.py --stage all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

import jax
import jax.numpy as jnp
import lightgbm
import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

from ashlar import TabularEstimator, TimeSeriesForecaster, classifier
from ashlar.distributional import (
    QUANTILE_LEVELS,
    decode,
    log_density,
    point_values,
    sample_values,
)
from ashlar.forecast import _forecast_windows

sys.path.insert(0, str(Path(__file__).resolve().parent))

from table_heads import flatten_members, table_head

DATA_DIR = Path("/tmp/ashlar-bench")
SEEDS = (0, 1, 2)
VALIDATION = 0.2
RAW = "https://raw.githubusercontent.com"
SOURCES = {
    "california.csv": f"{RAW}/ageron/handson-ml2/master/datasets/housing/housing.csv",
    "diamonds.csv": f"{RAW}/mwaskom/seaborn-data/master/diamonds.csv",
    "taxis.csv": f"{RAW}/mwaskom/seaborn-data/master/taxis.csv",
    "abalone.csv": f"{RAW}/jbrownlee/Datasets/master/abalone.csv",
    "adult.csv": f"{RAW}/jbrownlee/Datasets/master/adult-all.csv",
    "ames.csv": f"{RAW}/wblakecannon/ames/master/data/housing.csv",
    "temperatures.csv": f"{RAW}/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    "flights.csv": f"{RAW}/byuidatascience/data4python4ds/master/data-raw/flights/flights.csv",
    "flights-3m.parquet": f"{RAW}/vega/vega-datasets/main/data/flights-3m.parquet",
}
N_DRAWS = 400
Z80 = 1.2815515655446004


# ---------------------------------------------------------------- data


def _path(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCES[name], path)
    return path


def _strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Object columns as strings, missing as None, so every model reads the same table."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object or str(frame[column].dtype) in {"str", "string"}:
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    return frame


def california():
    frame = pd.read_csv(_path("california.csv"))
    y = frame.pop("median_house_value").to_numpy(np.float64) / 1e5
    return _strings(frame), y


def diamonds(log: bool = True):
    frame = pd.read_csv(_path("diamonds.csv"))
    price = frame.pop("price").to_numpy(np.float64)
    return _strings(frame), (np.log(price) if log else price / 1e3)


def abalone():
    columns = ["sex", "length", "diameter", "height", "whole", "shucked", "viscera", "shell", "rings"]
    frame = pd.read_csv(_path("abalone.csv"), header=None, names=columns)
    y = frame.pop("rings").to_numpy(np.float64)
    return _strings(frame), y


def ames():
    frame = pd.read_csv(_path("ames.csv"), index_col=0)
    frame = frame.drop(columns=["Order", "PID"])
    y = np.log(frame.pop("SalePrice").to_numpy(np.float64))
    return _strings(frame), y


def adult():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital", "occupation",
        "relationship", "race", "sex", "gain", "loss", "hours", "country", "income",
    ]
    frame = pd.read_csv(_path("adult.csv"), header=None, names=columns, na_values="?", skipinitialspace=True)
    y = (frame.pop("income").str.strip() == ">50K").astype(np.int64).to_numpy()
    return _strings(frame), y


def _sample(frame, y, n, seed=0):
    rows = np.random.default_rng(seed).choice(len(y), size=n, replace=False)
    return frame.iloc[rows].reset_index(drop=True), y[rows]


def flights():
    """NYC 2013 departures, arrival delay in minutes, from what is known before pushback.

    Departure delay, actual times, and air time are left out: they would give the answer away.
    """
    frame = pd.read_csv(_path("flights.csv"))
    frame = frame[frame["arr_delay"].notna()].reset_index(drop=True)
    date = pd.to_datetime(frame[["year", "month", "day"]])
    out = pd.DataFrame(
        {
            "month": frame["month"].astype(np.float64),
            "day_of_year": date.dt.dayofyear.astype(np.float64),
            "weekday": date.dt.dayofweek.astype(np.float64),
            "departure": (frame["sched_dep_time"] // 100 + frame["sched_dep_time"] % 100 / 60).astype(np.float64),
            "arrival": (frame["sched_arr_time"] // 100 + frame["sched_arr_time"] % 100 / 60).astype(np.float64),
            "distance": frame["distance"].astype(np.float64),
            "carrier": frame["carrier"],
            "origin": frame["origin"],
            "dest": frame["dest"],
            "tailnum": frame["tailnum"],
        }
    )
    return _strings(out), frame["arr_delay"].to_numpy(np.float64)


def flights_3m():
    """Three million US flights, January to June 2001, arrival delay in minutes.

    The source has only the scheduled time, distance, origin, and destination.
    Needs ``pyarrow`` to read the parquet file.
    """
    frame = pd.read_parquet(_path("flights-3m.parquet"))
    date = frame["date"]
    out = pd.DataFrame(
        {
            "month": date.dt.month.astype(np.float64),
            "day_of_year": date.dt.dayofyear.astype(np.float64),
            "weekday": date.dt.dayofweek.astype(np.float64),
            "departure": (date.dt.hour + date.dt.minute / 60).astype(np.float64),
            "distance": frame["distance"].astype(np.float64),
            "origin": frame["origin"],
            "dest": frame["destination"],
        }
    )
    return _strings(out), frame["delay"].to_numpy(np.float64)


def taxis(target: str):
    """NYC taxi trips. ``tip`` drops payment and total, which reveal it."""
    frame = pd.read_csv(_path("taxis.csv"), parse_dates=["pickup", "dropoff"])
    frame = frame.dropna(subset=["pickup_zone", "dropoff_zone"]).reset_index(drop=True)
    pickup = frame["pickup"]
    minutes = (frame["dropoff"] - pickup).dt.total_seconds().to_numpy() / 60.0
    out = pd.DataFrame(
        {
            "hour": pickup.dt.hour.astype(np.float64),
            "weekday": pickup.dt.dayofweek.astype(np.float64),
            "passengers": frame["passengers"].astype(np.float64),
            "color": frame["color"],
            "pickup_zone": frame["pickup_zone"],
            "dropoff_zone": frame["dropoff_zone"],
            "pickup_borough": frame["pickup_borough"],
            "dropoff_borough": frame["dropoff_borough"],
        }
    )
    if target == "duration":
        keep = (minutes > 0.5) & (minutes < 120)
        return _strings(out[keep].reset_index(drop=True)), np.log(minutes[keep])
    out["distance"] = frame["distance"].astype(np.float64)
    out["fare"] = frame["fare"].astype(np.float64)
    out["tolls"] = frame["tolls"].astype(np.float64)
    return _strings(out), frame["tip"].to_numpy(np.float64)


# ---------------------------------------------------------------- baselines


def _gbt_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype("category")
    return frame


def _align_categories(train: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    other = other.copy()
    for column in train.columns:
        if isinstance(train[column].dtype, pd.CategoricalDtype):
            categories = train[column].cat.categories
            values = other[column].astype(object)
            # A category the training rows never saw is missing to the trees, as it is index 0 to Ashlar.
            other[column] = pd.Categorical(values.where(values.isin(categories)), categories=categories)
    return other


def _gbt_split(x_train, y_train, seed):
    """The same 80/20 inner split Ashlar uses is not exposed, so draw one with the same size."""
    return train_test_split(x_train, y_train, test_size=VALIDATION, random_state=seed)


def _xgb(task, early: bool, seed: int, **extra):
    params = dict(enable_categorical=True, tree_method="hist", random_state=seed, n_jobs=1, **extra)
    if early:
        params.update(n_estimators=4000, learning_rate=0.05, early_stopping_rounds=100)
    return (xgboost.XGBClassifier if task == "classification" else xgboost.XGBRegressor)(**params)


def _lgbm(task, early: bool, seed: int, **extra):
    params = dict(random_state=seed, n_jobs=1, verbose=-1, **extra)
    if early:
        params.update(n_estimators=4000, learning_rate=0.05)
    return (lightgbm.LGBMClassifier if task == "classification" else lightgbm.LGBMRegressor)(**params)


def _fit_gbt(kind, task, early, seed, x_train, y_train, **extra):
    frame = _gbt_frame(x_train)
    model = (_xgb if kind == "xgboost" else _lgbm)(task, early, seed, **extra)
    if not early:
        model.fit(frame, y_train)
        return model, frame
    fit_x, val_x, fit_y, val_y = _gbt_split(frame, y_train, seed)
    if kind == "xgboost":
        model.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], verbose=False)
    else:
        model.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], callbacks=[lightgbm.early_stopping(100, verbose=False)])
    return model, frame


# ---------------------------------------------------------------- stage: point


POINT_TABLES = {
    "california": ("regression", california),
    "diamonds": ("regression", diamonds),
    "abalone": ("regression", abalone),
    "ames": ("regression", ames),
    "adult": ("classification", adult),
}


def _score(task, y, point, proba=None):
    if task == "regression":
        return {"rmse": float(np.sqrt(np.mean((point - y) ** 2)))}
    return {"accuracy": float(accuracy_score(y, point)), "log_loss": float(log_loss(y, proba))}


def stage_point(results):
    out = results.setdefault("point", {})
    for name, (task, loader) in POINT_TABLES.items():
        x, y = loader()
        rows = out.setdefault(name, {"task": task, "rows": int(len(y)), "columns": int(x.shape[1]), "runs": {}})
        for seed in SEEDS:
            key = str(seed)
            if key in rows["runs"]:
                continue
            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=seed, stratify=y if task == "classification" else None
            )
            run = {}
            started = time.perf_counter()
            model = (classifier if task == "classification" else TabularEstimator)(random_state=seed)
            model.fit(x_train, y_train)
            point = np.asarray(model.predict(x_test))
            proba = np.asarray(model.predict_proba(x_test))[:, 1] if task == "classification" else None
            run["ashlar"] = {**_score(task, y_test, point, proba), "seconds": time.perf_counter() - started}
            for kind in ("xgboost", "lightgbm"):
                for early in (False, True):
                    started = time.perf_counter()
                    gbt, frame = _fit_gbt(kind, task, early, seed, x_train, y_train)
                    test_frame = _align_categories(frame, x_test)
                    point = gbt.predict(test_frame)
                    proba = gbt.predict_proba(test_frame)[:, 1] if task == "classification" else None
                    label = f"{kind}{'_early' if early else ''}"
                    run[label] = {**_score(task, y_test, point, proba), "seconds": time.perf_counter() - started}
            rows["runs"][key] = run
            print(name, seed, json.dumps({k: round(v[next(iter(v))], 4) for k, v in run.items()}), flush=True)
            _save(results)


# ---------------------------------------------------------------- distributional heads on a table


def fit_distribution(kind, x_train, z_train, seed, **overrides):
    """``z_train`` is the standardized target. The fixed histogram bins assume that axis."""
    return table_head(kind, random_state=seed, **overrides).fit(x_train, z_train)


def distribution_scores(kind, model, x_test, y_test, mean, std, seed):
    raw = jnp.asarray(model.predict_raw(x_test))
    flat, batch, members = flatten_members(raw)
    decoded = decode(kind, flat, 1)
    z_test = (y_test - mean) / std
    scores = {}
    if kind == "quantile":
        levels = np.asarray(decoded).reshape(batch, members, 3).mean(axis=1) * std + mean
        point = levels[:, 1]
    else:
        draws = np.asarray(sample_values(kind, decoded, jax.random.PRNGKey(seed), N_DRAWS))
        draws = draws.reshape(N_DRAWS, batch, members).transpose(1, 0, 2).reshape(batch, -1) * std + mean
        levels = np.quantile(draws, QUANTILE_LEVELS, axis=1).T
        point = np.asarray(point_values(kind, decoded)).reshape(batch, members).mean(axis=1) * std + mean
        scores["crps"] = _crps(draws, y_test)
        repeated = jnp.repeat(jnp.asarray(z_test, dtype=flat.dtype), members)[:, None]
        logp = np.asarray(log_density(kind, decoded, repeated)).reshape(batch, members)
        logp = np.logaddexp.reduce(logp, axis=1) - np.log(members) - np.log(std)
        scores["nll"] = float(-np.mean(logp))
    scores.update(_interval_scores(levels, y_test, point))
    return scores


def _crps(draws, y):
    """Sample CRPS: E|X - y| - E|X - X'| / 2, the second term from sorted draws."""
    draws = np.sort(draws, axis=1)
    n = draws.shape[1]
    first = np.mean(np.abs(draws - y[:, None]), axis=1)
    weights = 2.0 * np.arange(1, n + 1) - n - 1
    second = np.sum(weights * draws, axis=1) / (n * n)
    return float(np.mean(first - second))


def _pinball(levels, y):
    tau = np.asarray(QUANTILE_LEVELS)
    gap = y[:, None] - levels
    return float(np.mean(np.maximum(tau * gap, (tau - 1.0) * gap)))


def _interval_scores(levels, y, point):
    levels = np.sort(levels, axis=1)
    covered = (y >= levels[:, 0]) & (y <= levels[:, 2])
    return {
        "pinball": _pinball(levels, y),
        "coverage80": float(np.mean(covered)),
        "width80": float(np.mean(levels[:, 2] - levels[:, 0])),
        "rmse": float(np.sqrt(np.mean((point - y) ** 2))),
        "mae": float(np.mean(np.abs(point - y))),
    }


def gbt_distribution(kind, x_train, y_train, x_test, y_test, seed):
    """Quantile regression at 0.1, 0.5, 0.9, plus a mean with a constant normal residual."""
    out = {}
    levels = []
    for tau in QUANTILE_LEVELS:
        if kind == "xgboost":
            model, frame = _fit_gbt(kind, "regression", True, seed, x_train, y_train,
                                    objective="reg:quantileerror", quantile_alpha=tau)
        else:
            model, frame = _fit_gbt(kind, "regression", True, seed, x_train, y_train, objective="quantile", alpha=tau)
        levels.append(model.predict(_align_categories(frame, x_test)))
    levels = np.stack(levels, axis=1)
    out[f"{kind}_quantile"] = _interval_scores(levels, y_test, levels[:, 1])

    model, frame = _fit_gbt(kind, "regression", True, seed, x_train, y_train)
    fit_x, val_x, _fit_y, val_y = _gbt_split(frame, y_train, seed)
    sigma = float(np.std(val_y - model.predict(val_x)))
    loc = model.predict(_align_categories(frame, x_test))
    normal_levels = loc[:, None] + sigma * np.array([-Z80, 0.0, Z80])
    scores = _interval_scores(normal_levels, y_test, loc)
    scores["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y_test - loc) / sigma) ** 2))
    draws = loc[:, None] + sigma * np.random.default_rng(seed).standard_normal((len(loc), N_DRAWS))
    scores["crps"] = _crps(draws, y_test)
    out[f"{kind}_normal"] = scores
    return out


DIST_TABLES = {
    # Zero-inflated: a third of tips are 0 (cash), the rest a skewed bump. Payment is hidden.
    "taxi_tip": lambda: taxis("tip"),
    # Censored at 5.00001 (x100k): a point mass at the cap beside a smooth body.
    "california": california,
    # Price in thousands, not logged: right skew and a spread that grows with carat.
    # A fixed 12,000-row sample keeps six neural fits per split to CPU minutes.
    "diamonds_price": lambda: _sample(*diamonds(log=False), 12_000),
    # Ring counts: close to a heteroscedastic normal.
    "abalone": abalone,
}
DIST_HEADS = ("gaussian", "mixture", "flow", "histogram", "quantile")


def stage_dist(results):
    out = results.setdefault("dist", {})
    for name, loader in DIST_TABLES.items():
        x, y = loader()
        rows = out.setdefault(name, {"rows": int(len(y)), "runs": {}})
        for seed in SEEDS:
            key = str(seed)
            if key in rows["runs"]:
                continue
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
            mean, std = float(np.mean(y_train)), float(np.std(y_train))
            run = {}
            started = time.perf_counter()
            point = TabularEstimator(random_state=seed).fit(x_train, y_train).predict(x_test)
            run["ashlar_mse"] = {
                "rmse": float(np.sqrt(np.mean((point - y_test) ** 2))),
                "mae": float(np.mean(np.abs(point - y_test))),
                "seconds": time.perf_counter() - started,
            }
            for kind in DIST_HEADS:
                started = time.perf_counter()
                model = fit_distribution(kind, x_train, (y_train - mean) / std, seed)
                run[f"ashlar_{kind}"] = distribution_scores(kind, model, x_test, y_test, mean, std, seed)
                run[f"ashlar_{kind}"]["seconds"] = time.perf_counter() - started
            for kind in ("xgboost", "lightgbm"):
                run.update(gbt_distribution(kind, x_train, y_train, x_test, y_test, seed))
            rows["runs"][key] = run
            print(name, seed, json.dumps({k: round(v.get("crps", v.get("pinball", v["rmse"])), 4) for k, v in run.items()}), flush=True)
            _save(results)


# ---------------------------------------------------------------- stage: large

# 260k training rows. Every Ashlar head gets the same larger batch and step, so an
# epoch is 500 steps instead of 4,000. Chosen before any test score was seen.
LARGE = {"batch_size": 512, "learning_rate": 2e-3, "max_epochs": 40, "patience": 6}
LARGE_SEEDS = (0, 1, 2)


# Three million rows. A bigger batch again, and every score on a fixed 100,000-row
# sample of the test split, so 400 draws per row fit in memory.
MILLION = {"batch_size": 2048, "learning_rate": 2e-3, "max_epochs": 30, "patience": 4}
MILLION_TEST_ROWS = 100_000


def stage_large(results):
    _large(results, "flights", flights, LARGE, None, ("carrier", "origin", "dest", "tailnum"))


def stage_million(results):
    _large(results, "flights_3m", flights_3m, MILLION, MILLION_TEST_ROWS, ("origin", "dest"))


def _large(results, name, loader, settings, test_rows, categorical):
    out = results.setdefault(_stage_of(name), {})
    x, y = loader()
    rows = out.setdefault(name, {"rows": int(len(y)), "runs": {}, "settings": settings, "test_rows": test_rows})
    rows["levels"] = {c: int(x[c].nunique()) for c in categorical}
    for seed in ONLY_SEED or LARGE_SEEDS:
        key = str(seed)
        if key in rows["runs"]:
            continue
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        if test_rows is not None:
            keep = np.random.default_rng(seed).choice(len(y_test), size=test_rows, replace=False)
            x_test, y_test = x_test.iloc[keep].reset_index(drop=True), y_test[keep]
        mean, std = float(np.mean(y_train)), float(np.std(y_train))
        run = {}
        started = time.perf_counter()
        point = TabularEstimator(random_state=seed, **settings).fit(x_train, y_train).predict(x_test)
        run["ashlar_mse"] = {
            "rmse": float(np.sqrt(np.mean((point - y_test) ** 2))),
            "mae": float(np.mean(np.abs(point - y_test))),
            "seconds": time.perf_counter() - started,
        }
        print(name, seed, "mse", round(run["ashlar_mse"]["mae"], 3), flush=True)
        for kind in DIST_HEADS:
            started = time.perf_counter()
            model = fit_distribution(kind, x_train, (y_train - mean) / std, seed, **settings)
            run[f"ashlar_{kind}"] = distribution_scores(kind, model, x_test, y_test, mean, std, seed)
            run[f"ashlar_{kind}"]["seconds"] = time.perf_counter() - started
            print(name, seed, kind, json.dumps({k: round(v, 3) for k, v in run[f"ashlar_{kind}"].items()}), flush=True)
        for kind in ("xgboost", "lightgbm"):
            started = time.perf_counter()
            run.update(gbt_distribution(kind, x_train, y_train, x_test, y_test, seed))
            print(name, seed, kind, round(time.perf_counter() - started, 1), flush=True)
        rows["runs"][key] = run
        print(name, seed, json.dumps({k: round(v.get("crps", v.get("pinball", v["rmse"])), 4) for k, v in run.items()}), flush=True)
        _save(results)


def _stage_of(name):
    return "million" if name == "flights_3m" else "large"


# ---------------------------------------------------------------- stage: embeddings


# The default stops after a handful of epochs on this small, wide table. This
# setting was picked by hand on split 0 and is reported on all three splits.
LONGER = {"learning_rate": 3e-3, "max_epochs": 100, "patience": 20}


def stage_embed(results):
    out = results.setdefault("embed", {})
    x, y = taxis("duration")
    rows = out.setdefault("taxi_duration", {"rows": int(len(y)), "runs": {}})
    rows["levels"] = {c: int(x[c].nunique()) for c in ("pickup_zone", "dropoff_zone")}
    for seed in SEEDS:
        key = str(seed)
        if key in rows["runs"]:
            continue
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        run = {}
        for label, kwargs in (
            ("ashlar_embedding", {}),
            ("ashlar_onehot", {"categorical_embeddings": []}),
            ("ashlar_embedding_longer", LONGER),
            ("ashlar_embedding16_longer", {**LONGER, "categorical_embedding_dim": 16}),
            ("ashlar_onehot_longer", {**LONGER, "categorical_embeddings": []}),
        ):
            started = time.perf_counter()
            model = TabularEstimator(random_state=seed, **kwargs).fit(x_train, y_train)
            run[label] = _score("regression", y_test, model.predict(x_test))
            run[label]["seconds"] = time.perf_counter() - started
        for kind in ("xgboost", "lightgbm"):
            started = time.perf_counter()
            gbt, frame = _fit_gbt(kind, "regression", True, seed, x_train, y_train)
            run[f"{kind}_early"] = _score("regression", y_test, gbt.predict(_align_categories(frame, x_test)))
            run[f"{kind}_early"]["seconds"] = time.perf_counter() - started
        rows["runs"][key] = run
        print("taxi_duration", seed, json.dumps({k: round(v["rmse"], 4) for k, v in run.items()}), flush=True)
        _save(results)


# ---------------------------------------------------------------- stage: forecast


FORECAST_LOOKBACK = 28


def _lag_frame(inputs):
    return pd.DataFrame(inputs, columns=[f"lag{FORECAST_LOOKBACK - i}" for i in range(FORECAST_LOOKBACK)])


def stage_forecast(results):
    out = results.setdefault("forecast", {})
    values = pd.read_csv(_path("temperatures.csv"))["Temp"].to_numpy(np.float64)
    cut = int(len(values) * 0.8)
    origins, inputs, targets = _forecast_windows(values, FORECAST_LOOKBACK, 1)
    test = origins >= cut
    x_test, y_test = inputs[test], targets[test][:, 0]
    rows = out.setdefault("temperatures", {"rows": int(len(values)), "test_steps": int(test.sum()), "runs": {}})
    for seed in SEEDS:
        key = str(seed)
        if key in rows["runs"]:
            continue
        run = {}
        for head in ("mse", "gaussian", "mixture", "flow", "histogram", "quantile"):
            started = time.perf_counter()
            model = TimeSeriesForecaster(lookback=FORECAST_LOOKBACK, horizon=1, head=head, random_state=seed)
            model.fit(values[:cut])
            point = np.asarray(model.predict(x_test)).reshape(-1)
            scores = {"rmse": float(np.sqrt(np.mean((point - y_test) ** 2))), "mae": float(np.mean(np.abs(point - y_test)))}
            if head == "quantile":
                levels = np.asarray(model.quantiles(x_test)).reshape(-1, 3)
                scores.update(_interval_scores(levels, y_test, point))
            elif head != "mse":
                draws = np.asarray(model.sample(x_test, jax.random.PRNGKey(seed), N_DRAWS)).reshape(N_DRAWS, -1).T
                levels = np.quantile(draws, QUANTILE_LEVELS, axis=1).T
                scores.update(_interval_scores(levels, y_test, point))
                scores["crps"] = _crps(draws, y_test)
                scores["nll"] = float(-np.mean(np.asarray(model.log_prob(x_test, y_test[:, None]))))
            scores["seconds"] = time.perf_counter() - started
            run[f"ashlar_{head}"] = scores
        train = origins < cut
        lag_train, y_train = _lag_frame(inputs[train]), targets[train][:, 0]
        lag_test = _lag_frame(x_test)
        model, _ = _fit_gbt("lightgbm", "regression", True, seed, lag_train, y_train)
        point = model.predict(lag_test)
        levels = []
        for tau in QUANTILE_LEVELS:
            q, _ = _fit_gbt("lightgbm", "regression", True, seed, lag_train, y_train, objective="quantile", alpha=tau)
            levels.append(q.predict(lag_test))
        run["lightgbm_lags"] = _interval_scores(np.stack(levels, axis=1), y_test, point)
        rows["runs"][key] = run
        print("temperatures", seed, json.dumps({k: round(v.get("crps", v["mae"]), 4) for k, v in run.items()}), flush=True)
        _save(results)


# ---------------------------------------------------------------- report


LABELS = {
    "ashlar": "Ashlar, default",
    "xgboost": "XGBoost, default",
    "xgboost_early": "XGBoost, early-stopped",
    "lightgbm": "LightGBM, default",
    "lightgbm_lags": "LightGBM on the same lags",
    "lightgbm_early": "LightGBM, early-stopped",
    "ashlar_mse": "Ashlar, squared error",
    "ashlar_gaussian": "Ashlar, `gaussian`",
    "ashlar_mixture": "Ashlar, `mixture`",
    "ashlar_flow": "Ashlar, `flow`",
    "ashlar_histogram": "Ashlar, `histogram`",
    "ashlar_quantile": "Ashlar, `quantile`",
    "xgboost_quantile": "XGBoost, three quantile models",
    "lightgbm_quantile": "LightGBM, three quantile models",
    "xgboost_normal": "XGBoost mean + constant normal",
    "lightgbm_normal": "LightGBM mean + constant normal",
    "ashlar_embedding": "Ashlar, 8-d embeddings, default",
    "ashlar_onehot": "Ashlar, one-hot, default",
    "ashlar_embedding_longer": "Ashlar, 8-d embeddings, longer",
    "ashlar_embedding16_longer": "Ashlar, 16-d embeddings, longer",
    "ashlar_onehot_longer": "Ashlar, one-hot, longer",
}


def _cell(runs, method, metric, digits=3):
    values = [run[method][metric] for run in runs.values() if method in run and metric in run[method]]
    if not values:
        return "—"
    mean = float(np.mean(values))
    if metric == "coverage80":
        return f"{mean:.0%}"
    if metric == "seconds":
        return f"{mean:.0f}" if mean >= 10 else f"{mean:.1f}"
    if len(values) == 1:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {float(np.std(values)):.{digits}f}"


def _table(runs, metrics, headers):
    methods = list(next(iter(runs.values())))
    lines = ["| Model | " + " | ".join(headers) + " |", "|---|" + "---:|" * len(headers)]
    for method in methods:
        cells = [_cell(runs, method, metric) for metric in metrics]
        lines.append(f"| {LABELS.get(method, method)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


SECTIONS = {
    "point": "Point accuracy",
    "dist": "Distributions on smaller tables",
    "large": "Distributions on large tables",
    "million": "Distributions on large tables",
    "embed": "Learned embeddings",
    "forecast": "Forecast heads",
}
TITLES = {
    "california": "California housing, median value in $100k",
    "diamonds": "Diamonds, log price",
    "abalone": "Abalone, rings",
    "ames": "Ames housing, log sale price",
    "adult": "Adult income, over $50k",
    "taxi_tip": "NYC taxi tips, $, payment type hidden",
    "diamonds_price": "Diamonds, price in $1,000, 12,000-row sample",
    "flights": "NYC flights 2013, arrival delay in minutes",
    "flights_3m": "US flights 2001, arrival delay in minutes, scored on 100,000 test rows",
    "taxi_duration": "NYC taxi trips, log minutes, from zones without distance",
    "temperatures": "Melbourne daily minimum temperature, °C, one day ahead",
}


def stage_report(_results):
    """Markdown tables for docs/benchmarks.md: mean ± standard deviation over the splits."""
    section = None
    for stage in ("point", "dist", "large", "million", "embed", "forecast"):
        for name, rows in _merged(stage).items():
            runs = rows["runs"]
            if stage == "forecast":
                # Older forecast files named the lag baseline plain "lightgbm".
                runs = {k: {("lightgbm_lags" if m == "lightgbm" else m): v for m, v in run.items()} for k, run in runs.items()}
            if SECTIONS[stage] != section:
                section = SECTIONS[stage]
                print(f"\n## {section}")
            splits = f"{len(runs)} split{'s' if len(runs) > 1 else ''}"
            print(f"\n### {TITLES.get(name, name)}\n\n{rows.get('rows'):,} rows, {splits}.\n")
            if stage == "point" and rows["task"] == "classification":
                print(_table(runs, ["accuracy", "log_loss", "seconds"], ["Accuracy", "Log loss", "Seconds"]))
            elif stage in {"point", "embed"}:
                print(_table(runs, ["rmse", "seconds"], ["RMSE", "Seconds"]))
            else:
                print(_table(
                    runs,
                    ["crps", "pinball", "coverage80", "width80", "nll", "mae", "seconds"],
                    ["CRPS", "Pinball", "80% coverage", "80% width", "NLL", "MAE", "Seconds"],
                ))


# ---------------------------------------------------------------- main


ONLY_SEED: tuple[int, ...] = ()


def _out_path(stage: str) -> Path:
    suffix = f"-{ONLY_SEED[0]}" if ONLY_SEED and stage in {"large", "million"} else ""
    return DATA_DIR / f"readme-{stage}{suffix}.json"


def _merged(stage: str):
    """One stage's results, with the runs of every per-seed file folded in."""
    merged = {}
    for path in sorted(DATA_DIR.glob(f"readme-{stage}*.json")):
        for name, rows in json.loads(path.read_text())[stage].items():
            target = merged.setdefault(name, {**rows, "runs": {}})
            target["runs"].update(rows["runs"])
    return merged


def _save(results):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for stage, value in results.items():
        _out_path(stage).write_text(json.dumps({stage: value}, indent=1, sort_keys=True))


STAGES = {
    "point": stage_point,
    "dist": stage_dist,
    "large": stage_large,
    "million": stage_million,
    "embed": stage_embed,
    "forecast": stage_forecast,
}
REPORT = "report"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all", REPORT], default="all")
    parser.add_argument("--seed", type=int, default=None, help="large and million stages: run this split alone")
    args = parser.parse_args()
    global ONLY_SEED
    ONLY_SEED = () if args.seed is None else (args.seed,)
    if args.stage == REPORT:
        stage_report(None)
        return
    for name, stage in STAGES.items():
        if args.stage in {name, "all"}:
            path = _out_path(name)
            results = json.loads(path.read_text()) if path.exists() else {}
            stage(results)
            _save(results)


if __name__ == "__main__":
    main()
