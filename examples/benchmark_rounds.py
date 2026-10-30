"""Score three nonlinear models on the five XGBoost cuts.

Does not refit XGBoost. Does not change the residual regression default.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_rounds.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from ashlar.recipes import DEFAULT_RECIPE
from ashlar.rounds import ASRHistogramBooster, ASRLearnedNeighborhood, ASRLinearLeafBooster

OUT_PATH = Path("/tmp/ashlar-bench/rounds-results.json")
WINE_PATH = Path("/tmp/ashlar-bench/extra-regression-splits/wine_quality.npz")
CALIFORNIA_PATH = Path("/tmp/ashlar-bench/split-california.csv")
COVERAGE = Path("/tmp/ashlar-bench/coverage-splits")

CUTS = {
    "wine_expected": 0.48996919003701056,
    "wine_regression": 0.4284963911647759,
    "electricity": 0.8543548063415614,
    "eye_movements": 0.642901813633521,
    "california": 0.8968296614723267,
}
ORDER = ("wine_expected", "wine_regression", "electricity", "eye_movements", "california")
MODELS = (
    ("histogram", ASRHistogramBooster),
    ("linear_leaf", ASRLinearLeafBooster),
    ("learned_neighborhood", ASRLearnedNeighborhood),
)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {name: {} for name, _cls in MODELS}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _metric(task: str, y_true, y_pred) -> float:
    if task == "classification":
        return float(accuracy_score(np.asarray(y_true), np.asarray(y_pred)))
    return float(r2_score(np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)))


def _wine():
    arrays = np.load(WINE_PATH, allow_pickle=False)
    return {
        "x_train": np.asarray(arrays["x_train"], dtype=np.float64),
        "y_train": np.asarray(arrays["y_train"]),
        "x_val": np.asarray(arrays["x_val"], dtype=np.float64),
        "y_val": np.asarray(arrays["y_val"]),
        "x_test": np.asarray(arrays["x_test"], dtype=np.float64),
        "y_test": np.asarray(arrays["y_test"]),
    }


def _coverage(name: str):
    arrays = np.load(COVERAGE / f"{name}.npz", allow_pickle=False)
    return {
        "x_train": np.asarray(arrays["x_train"], dtype=np.float64),
        "y_train": np.asarray(arrays["y_train"]),
        "x_val": np.asarray(arrays["x_val"], dtype=np.float64),
        "y_val": np.asarray(arrays["y_val"]),
        "x_test": np.asarray(arrays["x_test"], dtype=np.float64),
        "y_test": np.asarray(arrays["y_test"]),
    }


def _california():
    frame = pd.read_csv(CALIFORNIA_PATH)
    target = "price_above_median"
    parts = {}
    for split, key in (("train", "train"), ("val", "val"), ("test", "test")):
        rows = frame.loc[frame["_partition"] == split]
        features = rows.drop(columns=[target, "_partition"])
        parts[f"x_{key}"] = features.to_numpy(dtype=np.float64)
        parts[f"y_{key}"] = rows[target].to_numpy()
    return parts


def _dataset(name: str):
    if name == "wine_expected":
        return _wine(), "ordinal"
    if name == "wine_regression":
        return _wine(), "regression"
    if name == "california":
        return _california(), "classification"
    return _coverage(name), "classification"


def _detail(model) -> dict:
    detail = {
        "n_fit": int(model.n_fit_),
        "n_val": int(model.n_val_),
        "selection_metric": float(model.selection_metric_),
    }
    if hasattr(model, "best_iter_"):
        detail["best_iter"] = int(model.best_iter_)
    if hasattr(model, "best_stages_"):
        detail["best_stages"] = int(model.best_stages_)
        detail["n_stages"] = int(model.n_stages_)
    if hasattr(model, "best_epoch_"):
        detail["best_epoch"] = int(model.best_epoch_)
    return detail


def _fit(family: str, cls, name: str, payload: dict) -> None:
    payload.setdefault(family, {})
    if name in payload[family]:
        print(f"skip {family} {name}", flush=True)
        return
    parts, task = _dataset(name)
    print(f"start {family} {name} task {task}", flush=True)
    started = time.perf_counter()
    model = cls(random_state=0, verbose=True).fit(parts["x_train"], parts["y_train"], task)
    if name == "wine_expected" and (model.n_fit_ != 3637 or model.n_val_ != 910):
        raise SystemExit(f"{family} wine internal split was {model.n_fit_}/{model.n_val_}, expected 3637/910")
    prediction = model.predict(parts["x_test"]) if task == "classification" else model.predict_score(parts["x_test"])
    paper = model.predict(parts["x_val"]) if task == "classification" else model.predict_score(parts["x_val"])
    score = _metric(task, parts["y_test"], prediction)
    row = {
        "test_score": score,
        "paper_val_score": _metric(task, parts["y_val"], paper),
        "cut": CUTS[name],
        "cleared": bool(score >= CUTS[name]),
        "task": task,
        "wall_seconds": time.perf_counter() - started,
    }
    row.update(_detail(model))
    payload[family][name] = row
    _save(payload)
    print(
        f"done {family} {name} test {score:.6f} cut {CUTS[name]:.6f} cleared {row['cleared']}",
        flush=True,
    )


def main() -> None:
    if DEFAULT_RECIPE["task"] != "regression" or DEFAULT_RECIPE["standardize_target"] is not True:
        raise SystemExit("residual regression default changed")
    payload = _load()
    for family, cls in MODELS:
        for name in ORDER:
            _fit(family, cls, name, payload)
    final = []
    for family, _cls in MODELS:
        for name in ORDER:
            row = payload[family][name]
            final.append({"family": family, "name": name, "test_score": row["test_score"], "cleared": row["cleared"]})
    payload["final"] = final
    _save(payload)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
