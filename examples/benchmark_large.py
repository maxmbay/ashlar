"""Ashlar beside LightGBM, XGBoost, and CatBoost on large public tables.

Every table comes from OpenML and is capped at
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
    """US departure delay in minutes, 10M flights 1987-2008 (OpenML 42728).

    Day of month is dropped: with a random split it lets a model memorize
    single bad-weather days. The source has no year, so a time split is not
    possible; month and weekday stay as seasonal features.
    """
    X, y = _openml(42728)
    return _cap(_strings(X.drop(columns=["DayofMonth"])), np.asarray(y, dtype=np.float64))


def taxi_tip():
    """NYC green-taxi tip in dollars, December 2016 (OpenML 42729)."""
    X, y = _openml(42729)
    return _cap(_strings(X), np.asarray(y, dtype=np.float64))


def year():
    """Release year of a song from 90 timbre features (OpenML 44027, YearPredictionMSD)."""
    X, y = _openml(44027)
    return _cap(_strings(X), np.asarray(y, dtype=np.float64))


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


# Calendar-day groups of the current training rows, or None for an iid table.
TRAIN_GROUPS = None


def splits(table, seed):
    global TRAIN_GROUPS
    frame, y = load(table)
    task = TABLES[table][0]
    if "__group__" in frame.columns:
        from sklearn.model_selection import GroupShuffleSplit

        groups = frame.pop("__group__").to_numpy()
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(frame, y, groups))
        TRAIN_GROUPS = groups[train]
        return (task, frame.iloc[train].reset_index(drop=True), frame.iloc[test].reset_index(drop=True),
                y[train], y[test])
    TRAIN_GROUPS = None
    stratify = y if task == "classification" else None
    x_train, x_test, y_train, y_test = train_test_split(frame, y, test_size=0.2, random_state=seed, stratify=stratify)
    return task, x_train.reset_index(drop=True), x_test.reset_index(drop=True), y_train, y_test


def inner_split(frame, y, seed):
    """The early-stopping slice of the training rows: whole days when the table has them."""
    if TRAIN_GROUPS is None:
        return train_test_split(frame, y, test_size=VALIDATION, random_state=seed)
    from sklearn.model_selection import GroupShuffleSplit

    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=VALIDATION, random_state=seed).split(frame, y, TRAIN_GROUPS))
    return frame.iloc[fit], frame.iloc[val], y[fit], y[val]


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
    fit_x, val_x, fit_y, val_y = inner_split(frame, y_train, seed)
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


def fit_catboost_uncertainty(x_train, y_train, seed, path=None):
    """CatBoost's own distributional regression, early-stopped like the trees: a mean and a variance per row.

    Returns the model and ``predict(x) -> (loc, scale)``. With ``path``, a
    model saved there is loaded instead of fitted, and a new fit is saved there.
    """
    import catboost

    frame = _tree_frame(x_train)
    cats = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]

    def prep(f):
        f = f.copy()
        for c in cats:
            f[c] = f[c].astype(object).where(f[c].notna(), "__missing__").astype(str)
        return f

    if path is not None and Path(path).exists():
        model = catboost.CatBoostRegressor()
        model.load_model(str(path))
    else:
        fit_x, val_x, fit_y, val_y = inner_split(frame, y_train, seed)
        model = catboost.CatBoostRegressor(iterations=4000, learning_rate=0.05, od_type="Iter", od_wait=100,
                                           random_seed=seed, thread_count=GBM_THREADS, verbose=False,
                                           cat_features=cats, loss_function="RMSEWithUncertainty")
        model.fit(prep(fit_x), fit_y, eval_set=(prep(val_x), val_y), use_best_model=True)
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            model.save_model(str(path))

    def predict(x):
        pred = model.predict(prep(_align(frame, _tree_frame(x))))
        return pred[:, 0], np.sqrt(np.maximum(pred[:, 1], 1e-12))

    return model, predict


def run_catboost_uncertainty(table, seed):
    """CatBoost's own distributional regression: a mean and a variance per row (RMSEWithUncertainty)."""
    task, x_train, x_test, y_train, y_test = splits(table, seed)
    start = time.time()
    model, predict = fit_catboost_uncertainty(x_train, y_train, seed)
    seconds = time.time() - start
    loc, scale = predict(x_test)
    record = {"seconds": round(seconds, 1), "trees": int(model.get_best_iteration()) + 1}
    record.update(point_scores(loc, y_test))
    rows = dist_rows(len(y_test), seed)
    yd, ld, sd = y_test[rows], loc[rows], scale[rows]
    record["crps"] = crps_normal(ld, sd, yd)
    record["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((yd - ld) / sd) ** 2))
    record.update(interval(ld[:, None] + sd[:, None] * np.array([-Z80, 0.0, Z80]), yd))
    return record


# ---------------------------------------------------------------- distributional trees

# LightGBMLSS (März 2023) grows one LightGBM tree per distribution parameter
# each round, on the gradient and Hessian of the negative log-likelihood.
# NGBoost (Duan et al. 2020) fits a small tree to each parameter's natural
# gradient. Both pin older torch, NumPy, pandas, and scikit-learn than the
# rest of this script, so they run from their own environment:
#
#   python -m venv lss && lss/bin/pip install lightgbmlss==0.6.1 ngboost==0.5.11 lightgbm==4.7.0 \
#       torch==2.9.1 pyro-ppl==1.9.2 numpy==2.3.5 pandas==2.2.3 scipy==1.16.3 scikit-learn==1.7.2 pyarrow
#   GBM_THREADS=6 lss/bin/python examples/benchmark_large.py --tables airlines_delay \
#       --models lightgbmlss_gaussian lightgbmlss_studentt lightgbmlss_mixture ngboost --seeds 0 1 2
#
# On macOS, torch and LightGBM each bring an OpenMP runtime (libomp). With
# both loaded, LightGBMLSS segfaults the first time torch runs an operation on
# several threads inside LightGBM's objective, so ``run_lightgbmlss`` imports
# torch first: LightGBM then uses the libomp that torch already loaded.
#
# The target is standardized with the fitting rows' mean and spread, and each
# family starts from its unconditional fit (``initialize=True``); otherwise
# LightGBMLSS starts every parameter at 0.5, far from a delay in minutes.
# LightGBM's ``max_delta_step=0.5`` caps each tree's step on the raw
# parameters: without it the mixture's Newton steps diverge within a few
# rounds, and it changes the Gaussian and the Student-t very little.
# Scores are in the target's units: CRPS in closed form (Gaussian, Student-t,
# and the Gaussian mixture), NLL, and the 0.1, 0.5, 0.9 quantiles in closed
# form or by bisection on the mixture's CDF.
LSS_MIXTURE_COMPONENTS = 3
NGBOOST_ROWS = 200_000
PREDICTIONS = CACHE / "predictions"


def _normal_abs(mu, var):
    """E|X| for X ~ N(mu, var)."""
    from scipy.stats import norm

    sd = np.sqrt(var)
    z = mu / sd
    return mu * (2 * norm.cdf(z) - 1) + 2 * sd * norm.pdf(z)


def crps_mixture(weight, loc, scale, y):
    """Closed-form CRPS of a Gaussian mixture per row (Grimit et al. 2006). Arrays are ``(rows, components)``."""
    first = np.sum(weight * _normal_abs(y[:, None] - loc, scale**2), axis=1)
    second = np.zeros(len(y))
    for i in range(loc.shape[1]):
        for j in range(loc.shape[1]):
            second += weight[:, i] * weight[:, j] * _normal_abs(loc[:, i] - loc[:, j], scale[:, i] ** 2 + scale[:, j] ** 2)
    return float(np.mean(first - 0.5 * second))


def crps_student_t(df, loc, scale, y):
    """Closed-form CRPS of a Student-t with ``df > 1`` (Jordan, Krüger and Lerch 2019)."""
    from scipy.special import beta
    from scipy.stats import t as student

    z = (y - loc) / scale
    tail = 2 * np.sqrt(df) * beta(0.5, df - 0.5) / ((df - 1) * beta(0.5, df / 2) ** 2)
    per_row = scale * (z * (2 * student.cdf(z, df) - 1) + 2 * student.pdf(z, df) * (df + z * z) / (df - 1) - tail)
    return float(np.mean(per_row))


def mixture_quantiles(weight, loc, scale, levels=LEVELS, steps=80):
    """Quantiles of each row's Gaussian mixture by bisection on its CDF, ``(rows, len(levels))``."""
    from scipy.stats import norm

    out = []
    for level in levels:
        lo = np.min(loc - 12 * scale, axis=1)
        hi = np.max(loc + 12 * scale, axis=1)
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            below = np.sum(weight * norm.cdf((mid[:, None] - loc) / scale), axis=1) < level
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        out.append(0.5 * (lo + hi))
    return np.stack(out, axis=1)


def mixture_nll(weight, loc, scale, y):
    from scipy.special import logsumexp

    log = np.log(np.maximum(weight, 1e-300)) - 0.5 * ((y[:, None] - loc) / scale) ** 2 - np.log(scale * math.sqrt(2 * math.pi))
    return float(-np.mean(logsumexp(log, axis=1)))


def _save_predictions(table, name, seed, **arrays):
    """Each scored row's predicted parameters, kept outside the repository for figures that need them."""
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDICTIONS / f"{table}-{name}-seed{seed}.npz", **arrays)


def run_lightgbmlss(family, table, seed):
    """LightGBMLSS with a Gaussian, a Student-t, or a mixture of Gaussians, early-stopped like the trees."""
    import torch  # before LightGBM: on macOS it then reuses torch's libomp instead of loading a second one

    import lightgbm
    from lightgbmlss.distributions.Gaussian import Gaussian
    from lightgbmlss.distributions.Mixture import Mixture
    from lightgbmlss.distributions.StudentT import StudentT
    from lightgbmlss.model import LightGBMLSS

    torch.set_num_threads(GBM_THREADS)
    task, x_train, x_test, y_train, y_test = splits(table, seed)
    start = time.time()
    frame = _tree_frame(x_train)
    fit_x, val_x, fit_y, val_y = inner_split(frame, y_train, seed)
    center, spread = float(np.mean(fit_y)), float(np.std(fit_y))
    if family == "gaussian":
        dist = Gaussian(stabilization="None", response_fn="exp", loss_fn="nll", initialize=True)
    elif family == "studentt":
        dist = StudentT(stabilization="None", response_fn="exp", loss_fn="nll", initialize=True)
    elif family == "mixture":
        dist = Mixture(Gaussian(response_fn="exp"), M=LSS_MIXTURE_COMPONENTS, hessian_mode="individual", tau=1.0,
                       initialize=True)
    else:
        raise SystemExit(f"unknown LightGBMLSS family {family}")
    model = LightGBMLSS(dist)
    train = lightgbm.Dataset(fit_x, label=(fit_y - center) / spread, free_raw_data=False)
    valid = lightgbm.Dataset(val_x, label=(val_y - center) / spread, reference=train, free_raw_data=False)
    params = {"learning_rate": 0.05, "max_delta_step": 0.5, "num_threads": GBM_THREADS, "verbose": -1}
    model.train(params, train, num_boost_round=4000, valid_sets=[valid],
                callbacks=[lightgbm.early_stopping(100, verbose=False)])
    seconds = time.time() - start
    test = _align(frame, _tree_frame(x_test))
    raw = model.predict(test, pred_type="parameters", n_samples=1)
    record = {"seconds": round(seconds, 1), "trees": int(model.booster.best_iteration)}
    rows = dist_rows(len(y_test), seed)
    yd = y_test[rows]
    if family == "mixture":
        m = LSS_MIXTURE_COMPONENTS
        loc = center + spread * raw[[f"loc_{i}" for i in range(1, m + 1)]].to_numpy(np.float64)
        scale = spread * raw[[f"scale_{i}" for i in range(1, m + 1)]].to_numpy(np.float64)
        noisy = raw[[f"mix_prob_{i}" for i in range(1, m + 1)]].to_numpy(np.float64)
        noisy = noisy / noisy.sum(axis=1, keepdims=True)
        # LightGBMLSS adds Gumbel noise to the weights when it predicts (a Gumbel softmax with a fixed
        # seed), so two equal rows get different mixtures. The scores use the plain softmax of the
        # predicted weight logits, which scores better; the noisy weights' scores are kept beside them.
        score = model.booster.predict(test, raw_score=True).reshape(-1, model.dist.n_dist_param)
        logits = score[:, 2 * m:] + np.asarray(model.start_values, dtype=np.float64)[None, 2 * m:]
        weight = np.exp(logits - logits.max(axis=1, keepdims=True))
        weight = weight / weight.sum(axis=1, keepdims=True)
        record.update(point_scores(np.sum(weight * loc, axis=1), y_test))
        w, lo, sc = weight[rows], loc[rows], scale[rows]
        record["crps"] = crps_mixture(w, lo, sc, yd)
        record["nll"] = mixture_nll(w, lo, sc, yd)
        record.update(interval(mixture_quantiles(w, lo, sc), yd))
        record["crps_gumbel_weights"] = crps_mixture(noisy[rows], lo, sc, yd)
        record["nll_gumbel_weights"] = mixture_nll(noisy[rows], lo, sc, yd)
        _save_predictions(table, f"lightgbmlss_{family}", seed, rows=rows, weight=w.astype(np.float32),
                          loc=lo.astype(np.float32), scale=sc.astype(np.float32))
        return record
    loc = center + spread * raw["loc"].to_numpy(np.float64)
    scale = spread * raw["scale"].to_numpy(np.float64)
    record.update(point_scores(loc, y_test))
    ld, sd = loc[rows], scale[rows]
    if family == "gaussian":
        record["crps"] = crps_normal(ld, sd, yd)
        record["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((yd - ld) / sd) ** 2))
        record.update(interval(ld[:, None] + sd[:, None] * np.array([-Z80, 0.0, Z80]), yd))
        _save_predictions(table, f"lightgbmlss_{family}", seed, rows=rows, loc=ld.astype(np.float32),
                          scale=sd.astype(np.float32))
        return record
    from scipy.stats import t as student

    df = raw["df"].to_numpy(np.float64)[rows]
    record["df_median"] = float(np.median(df))
    record["crps"] = crps_student_t(df, ld, sd, yd)
    record["nll"] = float(-np.mean(student.logpdf((yd - ld) / sd, df) - np.log(sd)))
    record.update(interval(ld[:, None] + sd[:, None] * np.stack([student.ppf(q, df) for q in LEVELS], axis=1), yd))
    _save_predictions(table, f"lightgbmlss_{family}", seed, rows=rows, df=df.astype(np.float32),
                      loc=ld.astype(np.float32), scale=sd.astype(np.float32))
    return record


def run_ngboost(table, seed, overrides=None):
    """NGBoost with a normal, on a 200,000-row sample of the training rows: depth-3 trees on the natural gradient.

    Categories become their integer codes (unseen ones -1) and missing numbers the training median, since
    scikit-learn's trees read only numbers. ``overrides`` may set ``learning_rate`` (0.05, as the trees) and
    ``min_samples_leaf`` (NGBoost's default tree takes 1). The record keeps the spread of the predicted scales.
    """
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor

    overrides = overrides or {}

    task, x_train, x_test, y_train, y_test = splits(table, seed)
    start = time.time()
    frame = _tree_frame(x_train)
    keep = np.sort(np.random.default_rng(seed).choice(len(y_train), size=min(NGBOOST_ROWS, len(y_train)), replace=False))
    medians = {c: frame[c].median() for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c].dtype)}

    def numbers(f):
        f = _align(frame, f)
        out = np.empty(f.shape, dtype=np.float64)
        for k, column in enumerate(f.columns):
            if isinstance(f[column].dtype, pd.CategoricalDtype):
                out[:, k] = f[column].cat.codes.to_numpy()
            else:
                out[:, k] = f[column].fillna(medians[column]).to_numpy(np.float64)
        return out

    sample, sample_y = numbers(frame.iloc[keep]), y_train[keep]
    fit_x, val_x, fit_y, val_y = train_test_split(sample, sample_y, test_size=VALIDATION, random_state=seed)
    # NGBoost's default base learner, with the smallest leaf as an option.
    tree = DecisionTreeRegressor(criterion="friedman_mse", max_depth=3, random_state=seed,
                                 min_samples_leaf=int(overrides.get("min_samples_leaf", 1)))
    model = NGBRegressor(Dist=Normal, Base=tree, n_estimators=4000, learning_rate=float(overrides.get("learning_rate", 0.05)),
                         random_state=seed, verbose=False)
    model.fit(fit_x, fit_y, X_val=val_x, Y_val=val_y, early_stopping_rounds=100)
    seconds = time.time() - start
    best = getattr(model, "best_val_loss_itr", None)
    params = model.pred_dist(numbers(_tree_frame(x_test)), max_iter=best).params
    loc, scale = np.asarray(params["loc"], np.float64), np.asarray(params["scale"], np.float64)
    record = {"seconds": round(seconds, 1), "trees": int(best + 1) if best is not None else len(model.base_models),
              "training_rows": int(len(keep))}
    record.update(point_scores(loc, y_test))
    rows = dist_rows(len(y_test), seed)
    yd, ld, sd = y_test[rows], loc[rows], scale[rows]
    record["crps"] = crps_normal(ld, sd, yd)
    record["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * sd**2) + 0.5 * ((yd - ld) / sd) ** 2))
    record.update(interval(ld[:, None] + sd[:, None] * np.array([-Z80, 0.0, Z80]), yd))
    record["scale_quantiles"] = {str(q): float(np.quantile(sd, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)}
    _save_predictions(table, "ngboost" + ("-" + "-".join(f"{k}{v}" for k, v in sorted(overrides.items())) if overrides else ""),
                      seed, rows=rows, loc=ld.astype(np.float32), scale=sd.astype(np.float32))
    return record


# ---------------------------------------------------------------- ashlar


def ashlar_params(task, n_train, overrides):
    params = dict(overrides)
    if task == "classification":
        params.setdefault("task", "classification")
    return params


def run_ashlar(table, seed, *, head=None, overrides=None, recipe=None):
    from ashlar import ASRModel, ASRClassifier, tabm_piecewise

    task, x_train, x_test, y_train, y_test = splits(table, seed)
    overrides = dict(overrides or {})
    start = time.time()
    if recipe == "tabm_piecewise":
        model = tabm_piecewise(task="classification" if task == "classification" else head or "regression",
                               random_state=seed, **overrides)
    elif task == "classification":
        model = ASRClassifier(random_state=seed, **overrides)
    else:
        model = ASRModel(task=head or "regression", random_state=seed, **overrides)
    if TRAIN_GROUPS is None:
        model.fit(x_train, y_train)
    else:
        model.set_params(split="group").fit(x_train, y_train, groups=TRAIN_GROUPS)
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
    """``lightgbm``, ``xgboost``, ``catboost``, ``catboost_uncertainty``, ``lightgbmlss_<family>``,
    ``ngboost``, a PyTabKit name, ``ashlar``, or ``ashlar_<head>``."""
    if name in {"lightgbm", "xgboost", "catboost"}:
        return lambda table, seed, overrides: run_tree(name, table, seed)
    if name in NEURAL:
        return lambda table, seed, overrides: run_neural(name, table, seed)
    if name == "catboost_uncertainty":
        return lambda table, seed, overrides: run_catboost_uncertainty(table, seed)
    if name.startswith("lightgbmlss_"):
        family = name.split("_", 1)[1]
        return lambda table, seed, overrides: run_lightgbmlss(family, table, seed)
    if name == "ngboost":
        return lambda table, seed, overrides: run_ngboost(table, seed, overrides)
    if name == "ashlar":
        return lambda table, seed, overrides: run_ashlar(table, seed, overrides=overrides)
    if name == "ashlar_tabm":
        return lambda table, seed, overrides: run_ashlar(table, seed, overrides=overrides, recipe="tabm_piecewise")
    if name.startswith("ashlar_"):
        head = name.split("_", 1)[1]
        return lambda table, seed, overrides: run_ashlar(table, seed, head=head, overrides=overrides)
    raise SystemExit(f"unknown model {name}")


LABELS = {
    "ashlar": "Ashlar, default",
    "ashlar_tabm": "Ashlar, `tabm_piecewise()`",
    "ashlar_mixture": "Ashlar, `task=\"mixture\"`",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "catboost_uncertainty": "CatBoost, `RMSEWithUncertainty`",
    "lightgbmlss_gaussian": "LightGBMLSS, Gaussian",
    "lightgbmlss_studentt": "LightGBMLSS, Student-t",
    "lightgbmlss_mixture": f"LightGBMLSS, mixture of {LSS_MIXTURE_COMPONENTS} Gaussians",
    "ngboost": "NGBoost, normal, 200,000 training rows",
    "realmlp": "RealMLP (pytabkit)",
    "tabm_pytabkit": "TabM (pytabkit)",
    "ftt": "FT-Transformer (pytabkit)",
    "mlp_plr": "MLP-PLR (pytabkit)",
}


TITLES = {
    "zurich": "Zurich public transport, delay, 2,000,000 of 5,465,575 rows",
    "airlines_delay": "Airline delay, departure delay in minutes, 2,000,000 of 10,000,000 rows",
    "taxi_tip": "NYC green taxi, tip in $, 581,835 rows",
    "year": "Year prediction (MSD), release year, 515,345 rows",
    "higgs": "Higgs, signal against background, 2,000,000 of 5,100,000 rows",
    "covertype": "Covertype, 7 forest types, 581,012 rows",
    "airlines": "Airlines, delayed or not, 539,383 rows",
    "click": "Click prediction, clicked or not, 1,997,410 rows",
}
METRIC_NAMES = {
    "rmse": "RMSE", "mae": "MAE", "crps": "CRPS", "nll": "NLL", "coverage80": "80% coverage",
    "width80": "80% width", "accuracy": "Accuracy", "auc": "AUC", "log_loss": "Log loss",
}


def report(out, variant_for=None, seeds=None):
    """Markdown tables: mean ± std over the seeds that ran, per table and model; seconds from seed 0."""
    variant_for = variant_for or {}
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    frame = pd.DataFrame(rows)
    frame["variant"] = frame["variant"].fillna("")
    keep = frame.apply(lambda r: r["variant"] == variant_for.get(r["model"], ""), axis=1)
    frame = frame[keep]
    if seeds is not None:
        frame = frame[frame["seed"].isin(seeds)]
    for table, (task, _loader) in TABLES.items():
        part = frame[frame["table"] == table]
        if part.empty:
            continue
        metrics = (["accuracy", "auc", "log_loss"] if task == "classification"
                   else ["rmse", "mae", "crps", "nll", "coverage80", "width80"])
        print(f"\n### {TITLES.get(table, table)}\n")
        print("| Model | Splits | " + " | ".join(METRIC_NAMES[m] for m in metrics) + " | Seconds |")
        print("|---|---:|" + "---:|" * (len(metrics) + 1))
        for model in LABELS:
            group = part[part["model"] == model]
            if group.empty:
                continue
            cells = []
            for metric in metrics:
                values = group[metric].dropna() if metric in group else pd.Series(dtype=float)
                if values.empty:
                    cells.append("—")
                elif metric == "coverage80":
                    cells.append(f"{values.mean():.0%}")
                elif len(values) == 1:
                    cells.append(f"{values.iloc[0]:.4g}")
                else:
                    cells.append(f"{values.mean():.4g} ± {values.std(ddof=0):.2g}")
            # Seed 0 ran each tree on one thread and seeds 1 and 2 on six, so the seconds are seed 0's.
            first = group[group["seed"] == 0]
            seconds = (first if not first.empty else group)["seconds"].mean()
            print(f"| {LABELS[model]} | {len(group)} | " + " | ".join(cells) + f" | {seconds:.0f} |")


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
    parser.add_argument("--report", action="store_true", help="print markdown tables of the results, then stop")
    args = parser.parse_args()
    out = Path(args.out)
    if args.report:
        report(out, {"ashlar": "v3", "ashlar_mixture": "v3"}, seeds=args.seeds)
        return
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
            distributional = name.startswith(("ashlar_", "lightgbmlss_")) or name in ("catboost_uncertainty", "ngboost")
            if task == "classification" and distributional:
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
