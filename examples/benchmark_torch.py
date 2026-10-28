"""DeepTab and PyTabKit fits on the splits written by examples/benchmark.py.

Shared CPU ceiling: ``max_epochs`` / ``n_epochs`` 50, the same cap as Ashlar's
default recipe. Architecture, learning rate, batch size, and early-stopping
patience stay at each library's own defaults. This is not those libraries'
published training budget.

Run with the environment that has DeepTab and PyTabKit installed:

    python examples/benchmark_torch.py
"""

from __future__ import annotations

import json
import os
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATA_DIR = Path("/tmp/ashlar-bench")
SPLIT_NAMES = ("houses", "california")
EPOCH_CAP = 50
OUT_PATH = DATA_DIR / "torch-results.json"


def scores(task: str, y_true, y_pred) -> dict:
    pred = np.asarray(y_pred).reshape(-1)
    true = np.asarray(y_true).reshape(-1)
    if task == "regression":
        pred = pred.astype(np.float64)
        true = true.astype(np.float64)
        return {
            "r2": float(r2_score(true, pred)),
            "rmse_transformed": float(np.sqrt(np.mean((true - pred) ** 2))),
        }
    if pred.dtype.kind == "f":
        pred = np.rint(pred).astype(np.int64)
    true = true.astype(np.int64)
    return {"accuracy": float(np.mean(pred == true))}


def load_split(name: str):
    frame = pd.read_csv(DATA_DIR / f"split-{name}.csv")
    target = [column for column in frame.columns if column != "_partition"][-1]
    parts = {}
    for part in ("train", "val", "test"):
        chunk = frame.loc[frame["_partition"] == part].drop(columns=["_partition"])
        parts[part] = chunk
    task = "regression" if name == "houses" else "classification"
    return parts, target, task


def timed(model, parts, target: str, task: str) -> dict:
    train = parts["train"]
    started = time.perf_counter()
    model.fit(train.drop(columns=[target]), train[target].to_numpy())
    fit_seconds = time.perf_counter() - started
    predict_started = time.perf_counter()
    test_pred = model.predict(parts["test"].drop(columns=[target]))
    val_pred = model.predict(parts["val"].drop(columns=[target]))
    predict_seconds = time.perf_counter() - predict_started
    return {
        "test": scores(task, parts["test"][target], test_pred),
        "paper_validation": scores(task, parts["val"][target], val_pred),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "wall_seconds": fit_seconds + predict_seconds,
        "epoch_cap": EPOCH_CAP,
    }


def try_model(label: str, factory, parts, target, task) -> dict:
    print(f"  {label}", flush=True)
    try:
        model = factory()
        result = timed(model, parts, target, task)
        result["status"] = "ok"
        print("   ", result["test"], "wall", round(result["wall_seconds"], 2), flush=True)
        return result
    except Exception:
        message = traceback.format_exc()
        print(message, flush=True)
        return {"status": "failed", "error": message}


def main():
    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__}
    failures = []
    try:
        import importlib.metadata as metadata

        versions["torch"] = metadata.version("torch")
        versions["deeptab"] = metadata.version("deeptab")
        versions["pytabkit"] = metadata.version("pytabkit")
        versions["pytorch-lightning"] = metadata.version("pytorch-lightning")
    except Exception as exc:
        failures.append({"stage": "versions", "error": str(exc)})

    deeptab_ok = True
    pytabkit_ok = True
    try:
        from deeptab.models import MLPClassifier, MLPRegressor
    except Exception:
        deeptab_ok = False
        failures.append({"stage": "import deeptab", "error": traceback.format_exc()})
    try:
        from pytabkit import (
            RealMLP_TD_Classifier,
            RealMLP_TD_Regressor,
            TabM_D_Classifier,
            TabM_D_Regressor,
        )
    except Exception:
        pytabkit_ok = False
        failures.append({"stage": "import pytabkit", "error": traceback.format_exc()})

    datasets = []
    for name in SPLIT_NAMES:
        parts, target, task = load_split(name)
        print(f"== {name}", flush=True)
        models = {}
        if deeptab_ok:
            checkpoint = DATA_DIR / f"deeptab-ckpts-{name}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            common = dict(max_epochs=EPOCH_CAP, random_state=0, checkpoint_path=str(checkpoint))
            if task == "regression":
                models["deeptab_mlp"] = try_model(
                    "deeptab MLPRegressor",
                    lambda: _FitProxy(MLPRegressor(), common),
                    parts,
                    target,
                    task,
                )
            else:
                models["deeptab_mlp"] = try_model(
                    "deeptab MLPClassifier",
                    lambda: _FitProxy(MLPClassifier(), common),
                    parts,
                    target,
                    task,
                )
            models["deeptab_mlp"]["settings"] = {
                "class": "MLPRegressor" if task == "regression" else "MLPClassifier",
                "architecture": "library default MLPConfig",
                "trainer": "library default TrainerConfig except max_epochs=50 and random_state=0",
                "passed_fit_kwargs": common,
            }
        if pytabkit_ok:
            nn_kwargs = dict(device="cpu", random_state=0, n_epochs=EPOCH_CAP, n_cv=1, n_refit=0, n_repeats=1, verbosity=0)
            if task == "regression":
                models["pytabkit_realmlp_td"] = try_model(
                    "pytabkit RealMLP_TD_Regressor",
                    lambda: RealMLP_TD_Regressor(**nn_kwargs),
                    parts,
                    target,
                    task,
                )
                models["pytabkit_tabm_d"] = try_model(
                    "pytabkit TabM_D_Regressor",
                    lambda: TabM_D_Regressor(**nn_kwargs),
                    parts,
                    target,
                    task,
                )
            else:
                models["pytabkit_realmlp_td"] = try_model(
                    "pytabkit RealMLP_TD_Classifier",
                    lambda: RealMLP_TD_Classifier(**nn_kwargs),
                    parts,
                    target,
                    task,
                )
                models["pytabkit_tabm_d"] = try_model(
                    "pytabkit TabM_D_Classifier",
                    lambda: TabM_D_Classifier(**nn_kwargs),
                    parts,
                    target,
                    task,
                )
            for key in ("pytabkit_realmlp_td", "pytabkit_tabm_d"):
                models[key]["settings"] = {
                    "device": "cpu",
                    "n_epochs": EPOCH_CAP,
                    "n_cv": 1,
                    "n_refit": 0,
                    "n_repeats": 1,
                    "other_hyperparameters": "library defaults, including val_fraction=0.2",
                }
        datasets.append({"name": name, "task": task, "target": target, "models": models})

    payload = {
        "versions": versions,
        "hardware": {"platform": platform.platform(), "cpu_count": os.cpu_count(), "cuda": _cuda()},
        "epoch_cap": EPOCH_CAP,
        "datasets": datasets,
        "failures": failures,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print("wrote", OUT_PATH, flush=True)


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class _FitProxy:
    """Forward fit kwargs that DeepTab accepts on fit, not on the constructor."""

    def __init__(self, model, fit_kwargs):
        self.model = model
        self.fit_kwargs = fit_kwargs

    def fit(self, x, y):
        return self.model.fit(x, y, **self.fit_kwargs)

    def predict(self, x):
        return self.model.predict(x)


if __name__ == "__main__":
    main()
