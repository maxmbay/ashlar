"""Score three differentiable priors on the open XGBoost cuts.

Wine is not fit. Does not refit XGBoost. Does not change the residual regression default.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_priors.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ashlar.priors import BartPriorEnsemble, GradientSparseNet, VariationalPairwise
from ashlar.recipes import DEFAULT_RECIPE

OUT_PATH = Path("/tmp/ashlar-bench/prior-results.json")
CALIFORNIA_PATH = Path("/tmp/ashlar-bench/split-california.csv")
COVERAGE = Path("/tmp/ashlar-bench/coverage-splits")

CUTS = {
    "electricity": 0.8543548063415614,
    "eye_movements": 0.642901813633521,
    "california": 0.8968296614723267,
}
ORDER = ("electricity", "eye_movements", "california")
EXPECTED_SPLIT = {
    "electricity": (8000, 2000),
    "eye_movements": (4260, 1065),
    "california": (8000, 2000),
}
MODELS = (
    ("bart_prior", BartPriorEnsemble),
    ("variational_pairwise", VariationalPairwise),
    ("gradient_penalty", GradientSparseNet),
)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {name: {} for name, _cls in MODELS}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


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
    if name == "california":
        return _california()
    return _coverage(name)


def _detail(model) -> dict:
    detail = {
        "n_fit": int(model.n_fit_),
        "n_val": int(model.n_val_),
        "selection_metric": float(model.selection_metric_),
        "best_epoch": int(model.best_epoch_),
    }
    if hasattr(model, "feature_peak_"):
        detail["feature_peak"] = float(model.feature_peak_)
        detail["gate_margin"] = float(model.gate_margin_)
        detail["root_split"] = float(model.root_split_)
        detail["child_split"] = float(model.child_split_)
    return detail


def _fit(family: str, cls, name: str, payload: dict) -> None:
    payload.setdefault(family, {})
    if name in payload[family]:
        print(f"skip {family} {name}", flush=True)
        return
    parts = _dataset(name)
    print(f"start {family} {name}", flush=True)
    started = time.perf_counter()
    model = cls(random_state=0, verbose=True).fit(parts["x_train"], parts["y_train"], "classification")
    expected = EXPECTED_SPLIT[name]
    if (model.n_fit_, model.n_val_) != expected:
        raise SystemExit(f"{family} {name} split was {model.n_fit_}/{model.n_val_}, expected {expected[0]}/{expected[1]}")
    prediction = model.predict(parts["x_test"])
    paper = model.predict(parts["x_val"])
    score = float(accuracy_score(np.asarray(parts["y_test"]), np.asarray(prediction)))
    row = {
        "test_score": score,
        "paper_val_score": float(accuracy_score(np.asarray(parts["y_val"]), np.asarray(paper))),
        "cut": CUTS[name],
        "cleared": bool(score >= CUTS[name]),
        "wall_seconds": time.perf_counter() - started,
    }
    row.update(_detail(model))
    payload[family][name] = row
    _save(payload)
    print(f"done {family} {name} test {score:.6f} cut {CUTS[name]:.6f} cleared {row['cleared']}", flush=True)


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
