"""CPU benchmark: Ashlar default, Ashlar search, and default XGBoost.

Same outer split for every model. The split is the Grinsztajn medium-size
procedure (appendix A.3 / ``data_to_train_test``): train fraction
min(10000 / n, 0.70), then 30% of the remainder for the paper validation
partition and 70% for the test partition, ``numpy.random.RandomState(0)``.
One fold, because the resulting test partition is larger than 6000 rows.

The paper validation partition is scored and is not used to train or to
choose hyperparameters. Ashlar's default recipe holds out its own 20% of
the train partition. XGBoost is constructed with no arguments and therefore
trains on every train row.

Run:

    JAX_PLATFORMS=cpu python examples/benchmark.py
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import ashlar
from ashlar import ASRClassifier, ASRRegressor

DATA_DIR = Path("/tmp/ashlar-bench")
OUT_PATH = DATA_DIR / "jax-results.json"

DATASETS = (
    {
        "name": "houses",
        "task": "regression",
        "metric": "r2",
        "path": DATA_DIR / "reg_num-houses.csv",
        "source": "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/houses.csv",
        "openml_transformed": "https://www.openml.org/d/44138",
        "openml_original": "https://www.openml.org/d/537",
        "description": (
            "Grinsztajn numerical-regression houses table. "
            "This is the transformed California Housing regression matrix, not the raw sklearn target in units of $100,000."
        ),
    },
    {
        "name": "california",
        "task": "classification",
        "metric": "accuracy",
        "path": DATA_DIR / "clf_num-california.csv",
        "source": "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num/california.csv",
        "openml_transformed": "https://www.openml.org/d/45028",
        "openml_original": "https://www.dcc.fc.up.pt/ltorgo/Regression/cal_housing.html",
        "description": (
            "Grinsztajn numerical-classification california table. "
            "Binary label price_above_median on a transformed California Housing feature matrix."
        ),
    },
)


def grinsztajn_medium_split(frame: pd.DataFrame, rng: np.random.RandomState):
    """Match ``data_to_train_test`` for a medium benchmark (max 10_000 train rows)."""
    n_rows = frame.shape[0]
    train_prop = 0.70
    val_test_prop = 0.3
    max_train_samples = 10000
    max_val_samples = 50000
    max_test_samples = 50000
    train_set_prop = min(max_train_samples / n_rows, train_prop)
    train, held = train_test_split(frame, train_size=train_set_prop, random_state=rng)
    val, test = train_test_split(held, train_size=val_test_prop, random_state=rng)
    if len(val) > max_val_samples:
        val = val.iloc[:max_val_samples]
    if len(test) > max_test_samples:
        test = test.iloc[:max_test_samples]
    return train, val, test, {
        "protocol": "grinsztajn_medium_data_to_train_test",
        "train_prop_cap": train_prop,
        "max_train_samples": max_train_samples,
        "val_fraction_of_remainder": val_test_prop,
        "random_state": "numpy.random.RandomState(0)",
        "n_folds": 1,
        "fold_rule": "test rows > 6000 implies one fold (appendix A.3)",
        "train_fraction_used": train_set_prop,
        "n_rows": int(n_rows),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "n_features": int(frame.shape[1] - 1),
    }


def regression_scores(y_true, y_pred) -> dict:
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    jax.block_until_ready(pred)
    return {
        "r2": float(r2_score(true, pred)),
        "rmse_transformed": float(np.sqrt(np.mean((true - pred) ** 2))),
    }


def classification_scores(y_true, y_pred) -> dict:
    pred = np.asarray(y_pred).reshape(-1)
    true = np.asarray(y_true).reshape(-1)
    jax.block_until_ready(pred)
    if pred.dtype.kind == "f":
        pred = pred.astype(np.int64)
    true = true.astype(np.int64)
    return {"accuracy": float(np.mean(pred == true))}


def score(task: str, y_true, y_pred) -> dict:
    if task == "regression":
        return regression_scores(y_true, y_pred)
    return classification_scores(y_true, y_pred)


def timed_fit_predict(model, train, val, test, target: str, task: str, fit_kwargs=None) -> dict:
    fit_kwargs = {} if fit_kwargs is None else fit_kwargs
    x_train = train.drop(columns=[target])
    y_train = train[target].to_numpy()
    started = time.perf_counter()
    model.fit(x_train, y_train, **fit_kwargs)
    fit_seconds = time.perf_counter() - started
    predict_started = time.perf_counter()
    test_pred = model.predict(test.drop(columns=[target]))
    val_pred = model.predict(val.drop(columns=[target]))
    jax.block_until_ready(test_pred)
    jax.block_until_ready(val_pred)
    predict_seconds = time.perf_counter() - predict_started
    return {
        "test": score(task, test[target], test_pred),
        "paper_validation": score(task, val[target], val_pred),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "wall_seconds": fit_seconds + predict_seconds,
    }


def ashlar_details(model) -> dict:
    history = getattr(model, "history_", {}) or {}
    losses = history.get("val_loss", [])
    details = {"epochs_ran": int(len(losses)), "search": model.search_}
    return details


def run_dataset(spec: dict) -> dict:
    frame = pd.read_csv(spec["path"])
    target = frame.columns[-1]
    rng = np.random.RandomState(0)
    train, val, test, split_info = grinsztajn_medium_split(frame, rng)
    split_info["target"] = target
    split_info["columns"] = list(frame.columns)
    cache = DATA_DIR / f"split-{spec['name']}.csv"
    tagged = pd.concat(
        [
            train.assign(_partition="train"),
            val.assign(_partition="val"),
            test.assign(_partition="test"),
        ],
        ignore_index=True,
    )
    tagged.to_csv(cache, index=False)
    print(f"== {spec['name']} rows train/val/test {split_info['n_train']}/{split_info['n_val']}/{split_info['n_test']}", flush=True)

    if spec["task"] == "regression":
        default_model = ASRRegressor()
        search_model = ASRRegressor()
    else:
        default_model = ASRClassifier()
        search_model = ASRClassifier()

    print("  ashlar default", flush=True)
    default = timed_fit_predict(default_model, train, val, test, target, spec["task"])
    default["details"] = ashlar_details(default_model)
    default["recipe"] = "ASRRegressor()" if spec["task"] == "regression" else "ASRClassifier()"
    print("   ", default["test"], "wall", round(default["wall_seconds"], 2), flush=True)

    print("  ashlar search", flush=True)
    searched = timed_fit_predict(search_model, train, val, test, target, spec["task"], {"search": True})
    searched["details"] = ashlar_details(search_model)
    searched["recipe"] = default["recipe"] + " fit(search=True) with DEFAULT_SEARCH"
    print("   ", searched["test"], "wall", round(searched["wall_seconds"], 2), flush=True)

    print("  xgboost defaults", flush=True)
    if spec["task"] == "regression":
        booster = xgboost.XGBRegressor()
    else:
        booster = xgboost.XGBClassifier()
    xgb = timed_fit_predict(booster, train, val, test, target, spec["task"])
    xgb["params"] = booster.get_params()
    print("   ", xgb["test"], "wall", round(xgb["wall_seconds"], 2), flush=True)

    return {
        "name": spec["name"],
        "task": spec["task"],
        "metric": spec["metric"],
        "description": spec["description"],
        "source": spec["source"],
        "openml_transformed": spec["openml_transformed"],
        "openml_original": spec["openml_original"],
        "split": split_info,
        "split_cache": str(cache),
        "models": {"ashlar_default": default, "ashlar_search": searched, "xgboost_defaults": xgb},
    }


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_platforms": os.environ.get("JAX_PLATFORMS"),
        },
        "versions": {
            "ashlar": ashlar.__version__,
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__ if hasattr(jax, "lib") else None,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "python": platform.python_version(),
        },
        "datasets": [run_dataset(spec) for spec in DATASETS],
    }
    try:
        import jaxlib

        payload["versions"]["jaxlib"] = jaxlib.__version__
    except Exception as exc:  # pragma: no cover
        payload["versions"]["jaxlib_error"] = str(exc)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
