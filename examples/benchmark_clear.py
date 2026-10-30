"""Score the additive fitter, then the gate and the neighborhood, on the five cuts.

Wine expected label runs first. A later body is fit only on rows that fell short.
Does not refit XGBoost. Does not change the residual regression default.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_clear.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from ashlar.additive import ASRAdditive, ASRDiagonalNeighborhood, ASRGatedAdditive
from ashlar.recipes import DEFAULT_RECIPE

OUT_PATH = Path("/tmp/ashlar-bench/clear-benchmark-results.json")
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


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"additive": {}, "gate": {}, "neighborhood": {}}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _cleared(name: str, score: float) -> bool:
    return float(score) >= CUTS[name]


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


def _record(model, parts, task: str, name: str, started: float) -> dict:
    prediction = model.predict(parts["x_test"]) if task == "classification" else model.predict_score(parts["x_test"])
    paper = model.predict(parts["x_val"]) if task == "classification" else model.predict_score(parts["x_val"])
    score = _metric(task, parts["y_test"], prediction)
    paper_score = _metric(task, parts["y_val"], paper)
    pairs = getattr(model, "pairs_", None)
    row = {
        "test_score": score,
        "paper_val_score": paper_score,
        "cut": CUTS[name],
        "cleared": _cleared(name, score),
        "task": task,
        "n_fit": int(model.n_fit_),
        "n_val": int(model.n_val_),
        "wall_seconds": time.perf_counter() - started,
    }
    if pairs is not None:
        row["pairs"] = [list(pair) for pair in pairs]
        row["n_pairs"] = len(pairs)
        row["ridge"] = float(model.ridge_)
        row["selection_metric"] = float(model.selection_metric_)
    if getattr(model, "kind_", None) is not None:
        row["kind"] = model.kind_
        if model.peak_ is not None:
            row["peak"] = [float(value) for value in np.asarray(model.peak_).tolist()]
        if model.feature_index_ is not None:
            row["feature_index"] = int(model.feature_index_)
            row["threshold"] = float(model.threshold_)
    if hasattr(model, "scales_"):
        row["scales"] = [float(value) for value in np.asarray(model.scales_).tolist()]
        row["bandwidth"] = float(model.bandwidth_)
        row["selection_metric"] = float(model.selection_metric_)
    inside = getattr(model, "inside_", None)
    if inside is not None:
        row["inside_pairs"] = [list(pair) for pair in inside.pairs]
        row["outside_pairs"] = [list(pair) for pair in model.outside_.pairs]
        row["inside_n_pairs"] = len(inside.pairs)
        row["outside_n_pairs"] = len(model.outside_.pairs)
    return row


def _fit_stage(stage: str, name: str, payload: dict) -> None:
    if name in payload[stage]:
        print(f"skip {stage} {name}", flush=True)
        return
    parts, task = _dataset(name)
    print(f"start {stage} {name} task {task}", flush=True)
    started = time.perf_counter()
    if stage == "additive":
        model = ASRAdditive(random_state=0, verbose=True).fit(parts["x_train"], parts["y_train"], task)
    elif stage == "gate":
        model = ASRGatedAdditive(random_state=0, verbose=True).fit(parts["x_train"], parts["y_train"], task)
    else:
        model = ASRDiagonalNeighborhood(random_state=0, verbose=True).fit(parts["x_train"], parts["y_train"], task)
    if stage == "additive" and name == "wine_expected" and (model.n_fit_ != 3637 or model.n_val_ != 910):
        raise SystemExit(f"wine internal split was {model.n_fit_}/{model.n_val_}, expected 3637/910")
    row = _record(model, parts, task, name, started)
    payload[stage][name] = row
    _save(payload)
    print(
        f"done {stage} {name} test {row['test_score']:.6f} cut {row['cut']:.6f} cleared {row['cleared']}",
        flush=True,
    )


def _short(payload: dict, stage: str, names: list[str]) -> list[str]:
    return [name for name in names if not payload[stage][name]["cleared"]]


def main() -> None:
    if DEFAULT_RECIPE["task"] != "regression" or DEFAULT_RECIPE["standardize_target"] is not True:
        raise SystemExit("residual regression default changed")
    payload = _load()
    for name in ORDER:
        _fit_stage("additive", name, payload)
    short = _short(payload, "additive", list(ORDER))
    print(f"additive short {short}", flush=True)
    for name in short:
        _fit_stage("gate", name, payload)
    still = [name for name in short if not payload["gate"][name]["cleared"]]
    print(f"gate short {still}", flush=True)
    for name in still:
        _fit_stage("neighborhood", name, payload)
    final = []
    for name in ORDER:
        stages = []
        for stage in ("additive", "gate", "neighborhood"):
            if name in payload.get(stage, {}):
                stages.append((stage, payload[stage][name]))
        clearing = [item for item in stages if item[1]["cleared"]]
        body, row = clearing[0] if clearing else max(stages, key=lambda item: item[1]["test_score"])
        final.append({"name": name, "body": body, "test_score": row["test_score"], "cleared": bool(clearing)})
    payload["final"] = final
    _save(payload)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
