"""Ashlar beside LightGBM, XGBoost, and CatBoost on large public tables.

Every table comes from OpenML (or the vega flights mirror) and is capped at
2,000,000 rows by a fixed random sample. Each run is one (table, model, seed)
triple: the seed draws the 80/20 test split, every model holds out the same
20% of the training rows for early stopping, and the scores are on the test
rows. Runs append to one JSON-lines file, so several processes can share a
machine, and a finished run is skipped on a rerun.

Regression tables report RMSE and MAE for the point models, and CRPS, NLL,
80% coverage and width for the distributional ones. Classification tables
report accuracy, log loss, and AUC (one-vs-rest for more than two classes).

    JAX_PLATFORMS=cpu python examples/benchmark_large.py --tables higgs --models lightgbm ashlar --seeds 0 1 2
    python examples/benchmark_large.py --report
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
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

CACHE = Path(os.environ.get("ASHLAR_BENCH_CACHE", Path.home() / ".cache" / "ashlar-bench"))
OUT = CACHE / "large.jsonl"
MAX_ROWS = 2_000_000
VALIDATION = 0.2
N_DRAWS = 200
Z80 = 1.2815515655446004
LEVELS = (0.1, 0.5, 0.9)
GBM_THREADS = int(os.environ.get("GBM_THREADS", "1"))
# The distributional scores sample at most this many test rows, with a fixed seed.
DIST_ROWS = 100_000


# ---------------------------------------------------------------- data


def _openml(did: int):
    import openml

    openml.config.set_root_cache_directory(str(CACHE / "openml"))
    dataset = openml.datasets.get_dataset(did, download_data=True, download_qualities=False)
    X, y, _categorical, _names = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
    return X, y


def _cap(frame, y, seed=0):
    if len(y) <= MAX_ROWS:
        return frame.reset_index(drop=True), np.asarray(y)
    rows = np.sort(np.random.default_rng(seed).choice(len(y), size=MAX_ROWS, replace=False))
    return frame.iloc[rows].reset_index(drop=True), np.asarray(y)[rows]


def _strings(frame):
    """Category and object columns become strings; missing stays None. Every model reads the same table."""
    frame = frame.copy()
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.CategoricalDtype) or dtype == object or str(dtype) in {"str", "string", "bool"}:
            values = frame[column].astype(object)
            frame[column] = values.where(values.notna(), None).map(lambda v: v if v is None else str(v))
        else:
            frame[column] = frame[column].astype(np.float64)
    return frame


def zurich():
    """Delay of Zurich trams and buses in seconds, from weather and the time of day (OpenML 45034)."""
    X, y = _openml(45034)
    return _cap(_strings(X), np.asarray(y, dtype=np.float64))


def airlines_delay():
    """US departure delay in minutes, 10M flights 1987-2008 (OpenML 42728)."""
    X, y = _openml(42728)
    return _cap(_strings(X), np.asarray(y, dtype=np.float64))


def taxi_tip():
    """NYC green-taxi tip in dollars, December 2016 (OpenML 42729)."""
    X, y = _openml(42729)
    return _cap(_strings(X), np.asarray(y, dtype=np.float64))


def year():
    """Release year of a song from 90 timbre features (OpenML 44027, YearPredictionMSD)."""
    X, y = _openml(44027)
    return _cap(_strings(X), np.asarray(y, dtype=np.float64))


def flights_3m():
    from benchmark_readme import flights_3m as load

    frame, y = load()
    return _cap(frame, y)


def higgs():
    """Signal against background in particle collisions (OpenML 45664, a 5.1M-row subset of HIGGS)."""
    X, y = _openml(45664)
    return _cap(_strings(X), np.asarray(y).astype(np.int64))


def covertype():
    """Forest cover type, 7 classes (OpenML 1596)."""
    X, y = _openml(1596)
    return _cap(_strings(X), np.asarray(y).astype(str))


def airlines():
    """Whether a US flight is delayed (OpenML 1169)."""
    X, y = _openml(1169)
    return _cap(_strings(X), np.asarray(y).astype(str))


def click():
    """Whether a search ad is clicked, KDD Cup 2012 track 2 (OpenML 1218). Ids are numbers here."""
    X, y = _openml(1218)
    return _cap(_strings(X), np.asarray(y).astype(str))


TABLES = {
    "zurich": ("regression", zurich),
    "airlines_delay": ("regression", airlines_delay),
    "flights_3m": ("regression", flights_3m),
    "taxi_tip": ("regression", taxi_tip),
    "year": ("regression", year),
    "higgs": ("classification", higgs),
    "covertype": ("classification", covertype),
    "airlines": ("classification", airlines),
    "click": ("classification", click),
}


def load(table):
    path = CACHE / "frames" / f"{table}.parquet"
    if path.exists():
        frame = pd.read_parquet(path)
        y = frame.pop("__target__").to_numpy()
        return _strings(frame), y
    frame, y = TABLES[table][1]()
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.assign(__target__=y).to_parquet(path)
    return frame, y


def splits(table, seed):
    frame, y = load(table)
    task = TABLES[table][0]
    stratify = y if task == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(frame, y, test_size=0.2, random_state=seed, stratify=stratify)
    return task, x_train.reset_index(drop=True), x_test.reset_index(drop=True), y_train, y_test


# ---------------------------------------------------------------- scores


def crps_samples(draws, y):
    """Sample CRPS per row set: E|X - y| - E|X - X'| / 2. ``draws`` is ``(rows, n)``."""
    draws = np.sort(draws, axis=1)
    n = draws.shape[1]
    first = np.mean(np.abs(draws - y[:, None]), axis=1)
    weights = 2.0 * np.arange(1, n + 1) - n - 1
    second = np.sum(weights * draws, axis=1) / (n * n)
    return float(np.mean(first - second))


def crps_normal(loc, scale, y):
    from scipy.stats import norm

    z = (y - loc) / scale
    return float(np.mean(scale * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))))


def interval(levels, y):
    levels = np.sort(levels, axis=1)
    tau = np.asarray(LEVELS)
    gap = y[:, None] - levels
    return {
        "pinball": float(np.mean(np.maximum(tau * gap, (tau - 1.0) * gap))),
        "coverage80": float(np.mean((y >= levels[:, 0]) & (y <= levels[:, 2]))),
        "width80": float(np.mean(levels[:, 2] - levels[:, 0])),
    }


def point_scores(point, y):
    return {"rmse": float(np.sqrt(np.mean((point - y) ** 2))), "mae": float(np.mean(np.abs(point - y)))}


def class_scores(proba, classes, y):
    out = {
        "accuracy": float(accuracy_score(y, np.asarray(classes)[np.argmax(proba, axis=1)])),
        "log_loss": float(log_loss(y, proba, labels=classes)),
    }
    if len(classes) == 2:
        out["auc"] = float(roc_auc_score((np.asarray(y) == classes[1]).astype(int), proba[:, 1]))
    else:
        out["auc"] = float(roc_auc_score(y, proba, labels=classes, multi_class="ovr"))
    return out


def dist_rows(n, seed):
    if n <= DIST_ROWS:
        return np.arange(n)
    return np.sort(np.random.default_rng(seed).choice(n, size=DIST_ROWS, replace=False))


# ---------------------------------------------------------------- trees


def _tree_frame(frame):
    frame = frame.copy()
    for column in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[column].dtype):
            frame[column] = frame[column].astype(object).astype("category")
    return frame


def _align(train, other):
    other = other.copy()
    for column in train.columns:
        if isinstance(train[column].dtype, pd.CategoricalDtype):
            values = other[column].astype(object)
            other[column] = pd.Categorical(values.where(values.isin(train[column].cat.categories)),
                                           categories=train[column].cat.categories)
    return other


def fit_tree(kind, task, x_train, y_train, seed, classes=None):
    """Early-stopped at learning rate 0.05, up to 4,000 trees, 100 rounds of patience."""
    frame = _tree_frame(x_train)
    fit_x, val_x, fit_y, val_y = train_test_split(frame, y_train, test_size=VALIDATION, random_state=seed)
    if task == "classification":
        fit_y = np.searchsorted(classes, fit_y)
        val_y = np.searchsorted(classes, val_y)
    if kind == "lightgbm":
        import lightgbm

        cls = lightgbm.LGBMClassifier if task == "classification" else lightgbm.LGBMRegressor
        model = cls(n_estimators=4000, learning_rate=0.05, random_state=seed, n_jobs=GBM_THREADS, verbose=-1)
        model.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], callbacks=[lightgbm.early_stopping(100, verbose=False)])
    elif kind == "xgboost":
        import xgboost

        cls = xgboost.XGBClassifier if task == "classification" else xgboost.XGBRegressor
        model = cls(n_estimators=4000, learning_rate=0.05, early_stopping_rounds=100, enable_categorical=True,
                    tree_method="hist", max_cat_to_onehot=1, random_state=seed, n_jobs=GBM_THREADS)
        model.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], verbose=False)
    else:
        import catboost

        cats = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]

        def prep(f):
            f = f.copy()
            for c in cats:
                f[c] = f[c].astype(object).where(f[c].notna(), "__missing__").astype(str)
            return f

        cls = catboost.CatBoostClassifier if task == "classification" else catboost.CatBoostRegressor
        extra = {} if task == "classification" else {"loss_function": "RMSE"}
        model = cls(iterations=4000, learning_rate=0.05, od_type="Iter", od_wait=100, random_seed=seed,
                    thread_count=GBM_THREADS, verbose=False, cat_features=cats, **extra)
        model.fit(prep(fit_x), fit_y, eval_set=(prep(val_x), val_y), use_best_model=True)
        model._prep = prep
    return model, frame, (val_x, val_y)


def tree_predict(model, frame, x, proba=False):
    x = _align(frame, _tree_frame(x))
    if hasattr(model, "_prep"):
        x = model._prep(x)
    return model.predict_proba(x) if proba else model.predict(x)


def run_tree(kind, table, seed):
    task, x_train, x_test, y_train, y_test = splits(table, seed)
    start = time.time()
    classes = np.unique(y_train) if task == "classification" else None
    model, frame, (val_x, val_y) = fit_tree(kind, task, x_train, y_train, seed, classes)
    seconds = time.time() - start
    record = {"seconds": round(seconds, 1), "trees": int(_tree_count(model))}
    if task == "classification":
        record.update(class_scores(tree_predict(model, frame, x_test, proba=True), classes, y_test))
        return record
    loc = tree_predict(model, frame, x_test)
    record.update(point_scores(loc, y_test))
    # The distribution a tree regressor offers without another library: its mean
    # and a normal whose width is the residual standard deviation on the validation rows.
    val_pred = model.predict(model._prep(val_x)) if hasattr(model, "_prep") else model.predict(val_x)
    sigma = float(np.std(val_y - val_pred))
    rows = dist_rows(len(y_test), seed)
    yd, ld = y_test[rows], loc[rows]
    record["crps"] = crps_normal(ld, sigma, yd)
    record["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((yd - ld) / sigma) ** 2))
    record.update(interval(ld[:, None] + sigma * np.array([-Z80, 0.0, Z80]), yd))
    return record


def _tree_count(model):
    for name in ("best_iteration_", "best_iteration"):
        value = getattr(model, name, None)
        if value is not None and not callable(value):
            return int(value) + 1
    if hasattr(model, "get_best_iteration"):
        return int(model.get_best_iteration()) + 1
    return -1


def run_catboost_uncertainty(table, seed):
    """CatBoost's own distributional regression: a mean and a variance per row (RMSEWithUncertainty)."""
    import catboost

    task, x_train, x_test, y_train, y_test = splits(table, seed)
    start = time.time()
    frame = _tree_frame(x_train)
    cats = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]

    def prep(f):
        f = f.copy()
        for c in cats:
            f[c] = f[c].astype(object).where(f[c].notna(), "__missing__").astype(str)
        return f

    fit_x, val_x, fit_y, val_y = train_test_split(frame, y_train, test_size=VALIDATION, random_state=seed)
    model = catboost.CatBoostRegressor(iterations=4000, learning_rate=0.05, od_type="Iter", od_wait=100,
                                       random_seed=seed, thread_count=GBM_THREADS, verbose=False,
                                       cat_features=cats, loss_function="RMSEWithUncertainty")
    model.fit(prep(fit_x), fit_y, eval_set=(prep(val_x), val_y), use_best_model=True)
    seconds = time.time() - start
    pred = model.predict(prep(_align(frame, _tree_frame(x_test))))
    loc, var = pred[:, 0], pred[:, 1]
    scale = np.sqrt(np.maximum(var, 1e-12))
    record = {"seconds": round(seconds, 1), "trees": int(model.get_best_iteration()) + 1}
    record.update(point_scores(loc, y_test))
    rows = dist_rows(len(y_test), seed)
    yd, ld, sd = y_test[rows], loc[rows], scale[rows]
    record["crps"] = crps_normal(ld, sd, yd)
    record["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((yd - ld) / sd) ** 2))
    record.update(interval(ld[:, None] + sd[:, None] * np.array([-Z80, 0.0, Z80]), yd))
    return record


# ---------------------------------------------------------------- ashlar


def ashlar_params(task, n_train, overrides):
    params = dict(overrides)
    if task == "classification":
        params.setdefault("task", "classification")
    return params


def run_ashlar(table, seed, *, head=None, overrides=None):
    from ashlar import TabularEstimator, classifier

    task, x_train, x_test, y_train, y_test = splits(table, seed)
    overrides = dict(overrides or {})
    start = time.time()
    if task == "classification":
        model = classifier(random_state=seed, **overrides)
    else:
        model = TabularEstimator(task=head or "regression", random_state=seed, **overrides)
    model.fit(x_train, y_train)
    seconds = time.time() - start
    record = {"seconds": round(seconds, 1), "epochs": len(model.history_["train_loss"])}
    if task == "classification":
        record.update(class_scores(model.predict_proba(x_test), model.classes_, y_test))
        return record
    record.update(point_scores(model.predict(x_test), y_test))
    if head in (None, "regression"):
        return record
    rows = dist_rows(len(y_test), seed)
    xd, yd = x_test.iloc[rows].reset_index(drop=True), y_test[rows]
    if head == "quantile":
        record.update(interval(model.predict_quantiles(xd), yd))
        return record
    draws = model.sample(xd, n=N_DRAWS, seed=seed).T
    record["crps"] = crps_samples(draws, yd)
    record["nll"] = float(-np.mean(model.log_prob(xd, yd)))
    record.update(interval(np.quantile(draws, LEVELS, axis=1).T, yd))
    return record


# ---------------------------------------------------------------- other neural packages

# PyTabKit's tuned-default recipes: RealMLP (Holzmüller et al. 2024), TabM
# (Gorishniy et al. 2025), FT-Transformer and MLP-PLR (Gorishniy et al. 2021, 2022).
# Each holds out the same share of training rows for early stopping.
NEURAL = {
    "realmlp": "RealMLP_TD",
    "tabm_pytabkit": "TabM_D",
    "ftt": "FTT_D",
    "mlp_plr": "MLP_PLR_D",
}
NN_THREADS = int(os.environ.get("NN_THREADS", "4"))


def run_neural(name, table, seed):
    import pytabkit

    task, x_train, x_test, y_train, y_test = splits(table, seed)
    suffix = "Classifier" if task == "classification" else "Regressor"
    cls = getattr(pytabkit, f"{NEURAL[name]}_{suffix}")
    frame = _tree_frame(x_train)
    # PyTabKit refuses missing numbers, so they get the training median, as Ashlar's default does.
    medians = {c: frame[c].median() for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c].dtype)}
    frame = frame.fillna(medians)
    start = time.time()
    model = cls(device="cpu", random_state=seed, n_threads=NN_THREADS, val_fraction=VALIDATION)
    model.fit(frame, y_train)
    seconds = time.time() - start
    test = _align(frame, _tree_frame(x_test)).fillna(medians)
    record = {"seconds": round(seconds, 1)}
    if task == "classification":
        record.update(class_scores(model.predict_proba(test), model.classes_, y_test))
    else:
        record.update(point_scores(np.asarray(model.predict(test)).reshape(-1), y_test))
    return record


# ---------------------------------------------------------------- driver


def model_runner(name):
    """``lightgbm``, ``xgboost``, ``catboost``, ``catboost_uncertainty``, ``ashlar``, or ``ashlar_<head>``."""
    if name in {"lightgbm", "xgboost", "catboost"}:
        return lambda table, seed, overrides: run_tree(name, table, seed)
    if name in NEURAL:
        return lambda table, seed, overrides: run_neural(name, table, seed)
    if name == "catboost_uncertainty":
        return lambda table, seed, overrides: run_catboost_uncertainty(table, seed)
    if name == "ashlar":
        return lambda table, seed, overrides: run_ashlar(table, seed, overrides=overrides)
    if name.startswith("ashlar_"):
        head = name.split("_", 1)[1]
        return lambda table, seed, overrides: run_ashlar(table, seed, head=head, overrides=overrides)
    raise SystemExit(f"unknown model {name}")


def done_runs(out):
    if not out.exists():
        return set()
    seen = set()
    for line in out.read_text().splitlines():
        row = json.loads(line)
        seen.add((row["table"], row["model"], row.get("variant", ""), row["seed"]))
    return seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tables", nargs="+", default=list(TABLES))
    parser.add_argument("--models", nargs="+", default=["lightgbm", "xgboost", "catboost", "ashlar"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--variant", default="", help="a label for --params, kept in the output")
    parser.add_argument("--params", default="{}", help="JSON overrides for the Ashlar estimator")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--prepare", action="store_true", help="download and cache the tables, then stop")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        for table in args.tables:
            frame, y = load(table)
            print(table, frame.shape, flush=True)
        return
    overrides = json.loads(args.params)
    for table in args.tables:
        task = TABLES[table][0]
        for name in args.models:
            if task == "classification" and (name.startswith("ashlar_") or name == "catboost_uncertainty"):
                continue
            for seed in args.seeds:
                key = (table, name, args.variant, seed)
                if key in done_runs(out):
                    continue
                record = model_runner(name)(table, seed, overrides)
                record = {"table": table, "model": name, "variant": args.variant, "seed": seed, **record}
                with out.open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
                print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
