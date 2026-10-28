"""Winning houses recipe on three other Grinsztajn numerical regression tables.

One ``RandomState(0)`` medium split each. Ashlar is not the residual-MLP default.
Classification is not run.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_extra_regression.py --stage ashlar
    /tmp/deeptab-venv/bin/python examples/benchmark_extra_regression.py --stage tabm
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

DATA_DIR = Path("/tmp/ashlar-bench")
OUT_PATH = DATA_DIR / "extra-regression-results.json"
SPLIT_DIR = DATA_DIR / "extra-regression-splits"
DATASETS = (
    {
        "name": "wine_quality",
        "file": "reg_num-wine_quality.csv",
        "source": "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/wine_quality.csv",
        "openml": 287,
        "target": "quality",
    },
    {
        "name": "cpu_act",
        "file": "reg_num-cpu_act.csv",
        "source": "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/cpu_act.csv",
        "openml": 197,
        "target": "usr",
    },
    {
        "name": "sulfur",
        "file": "reg_num-sulfur.csv",
        "source": "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/reg_num/sulfur.csv",
        "openml": 23515,
        "target": "y1",
    },
)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"seed": 0, "datasets": []}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _row(payload: dict, name: str) -> dict:
    for item in payload["datasets"]:
        if item["name"] == name:
            return item
    item = {"name": name}
    payload["datasets"].append(item)
    return item


def _scores(y_true, y_pred) -> dict:
    from sklearn.metrics import r2_score

    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return {
        "r2": float(r2_score(true, pred)),
        "rmse_transformed": float(np.sqrt(np.mean((true - pred) ** 2))),
    }


def _prepare(spec: dict):
    import benchmark

    frame = pd.read_csv(DATA_DIR / spec["file"])
    if frame.columns[-1] != spec["target"]:
        raise SystemExit(f"{spec['name']} last column is {frame.columns[-1]!r}, expected {spec['target']!r}")
    train, val, test, provenance = benchmark.grinsztajn_medium_split(frame, np.random.RandomState(0))
    target = spec["target"]
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    columns = list(train.drop(columns=[target]).columns)
    np.savez(
        SPLIT_DIR / f"{spec['name']}.npz",
        x_train=train.drop(columns=[target]).to_numpy(),
        y_train=train[target].to_numpy(),
        x_val=val.drop(columns=[target]).to_numpy(),
        y_val=val[target].to_numpy(),
        x_test=test.drop(columns=[target]).to_numpy(),
        y_test=test[target].to_numpy(),
        columns=np.array(columns),
    )
    return train, val, test, target, provenance


def stage_ashlar() -> None:
    import xgboost

    from ashlar import tabm

    payload = _load()
    for spec in DATASETS:
        row = _row(payload, spec["name"])
        if "ashlar" in row and "xgboost" in row:
            print(f"skip {spec['name']}", flush=True)
            continue
        train, val, test, target, provenance = _prepare(spec)
        x_train = train.drop(columns=[target])
        y_train = train[target].to_numpy()
        x_test = test.drop(columns=[target])
        y_test = test[target].to_numpy()
        x_val = val.drop(columns=[target])
        y_val = val[target].to_numpy()
        row.update(
            {
                "source": spec["source"],
                "openml": spec["openml"],
                "target": target,
                "n_rows": int(provenance["n_rows"]),
                "n_train": int(len(train)),
                "n_paper_val": int(len(val)),
                "n_test": int(len(test)),
                "n_features": int(x_train.shape[1]),
                "feature_names": list(x_train.columns),
            }
        )
        model = tabm(
            numeric_scaling="quantile_tabr",
            standardize_target=True,
            share_training_batches=False,
            max_epochs=50,
            random_state=0,
        )
        print(f"start ashlar {spec['name']}", flush=True)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        test_scores = _scores(y_test, model.predict(x_test))
        val_scores = _scores(y_val, model.predict(x_val))
        elapsed = time.perf_counter() - started
        history = [float(value) for value in model.history_["val_loss"]]
        row["ashlar"] = {
            "test_r2": test_scores["r2"],
            "test_rmse": test_scores["rmse_transformed"],
            "paper_val_r2": val_scores["r2"],
            "epochs_ran": len(history),
            "best_val_loss": min(history) if history else None,
            "wall_seconds": elapsed,
            "n_quantiles": int(model.numeric_transform_.quantiles.shape[0]),
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "optimizer": "adamw",
        }
        print(f"ashlar {spec['name']} r2={test_scores['r2']:.6f} wall={elapsed:.1f}", flush=True)
        booster = xgboost.XGBRegressor()
        xgb_started = time.perf_counter()
        booster.fit(x_train, y_train)
        xgb_scores = _scores(y_test, booster.predict(x_test))
        row["xgboost"] = {
            "test_r2": xgb_scores["r2"],
            "test_rmse": xgb_scores["rmse_transformed"],
            "wall_seconds": time.perf_counter() - xgb_started,
            "config": "XGBRegressor() defaults",
        }
        print(f"xgb {spec['name']} r2={xgb_scores['r2']:.6f}", flush=True)
        _save(payload)
    _save(payload)


def stage_tabm() -> None:
    from pytabkit import TabM_D_Regressor

    payload = _load()
    for spec in DATASETS:
        row = _row(payload, spec["name"])
        if "tabm_d" in row:
            print(f"skip tabm {spec['name']}", flush=True)
            continue
        arrays = np.load(SPLIT_DIR / f"{spec['name']}.npz", allow_pickle=False)
        columns = [str(name) for name in arrays["columns"].tolist()]
        x_train = pd.DataFrame(arrays["x_train"], columns=columns)
        x_test = pd.DataFrame(arrays["x_test"], columns=columns)
        x_val = pd.DataFrame(arrays["x_val"], columns=columns)
        model = TabM_D_Regressor(
            device="cpu",
            random_state=0,
            n_epochs=50,
            n_cv=1,
            n_refit=0,
            n_repeats=1,
            verbosity=0,
        )
        print(f"start tabm {spec['name']}", flush=True)
        started = time.perf_counter()
        model.fit(x_train, arrays["y_train"])
        test_scores = _scores(arrays["y_test"], model.predict(x_test))
        val_scores = _scores(arrays["y_val"], model.predict(x_val))
        elapsed = time.perf_counter() - started
        row["tabm_d"] = {
            "test_r2": test_scores["r2"],
            "test_rmse": test_scores["rmse_transformed"],
            "paper_val_r2": val_scores["r2"],
            "wall_seconds": elapsed,
            "config": "TabM_D_Regressor(device=cpu, random_state=0, n_epochs=50, n_cv=1, n_refit=0, n_repeats=1, verbosity=0)",
        }
        print(f"tabm {spec['name']} r2={test_scores['r2']:.6f} wall={elapsed:.1f}", flush=True)
        _save(payload)
    _save(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("ashlar", "tabm"), required=True)
    args = parser.parse_args()
    if args.stage == "ashlar":
        stage_ashlar()
    else:
        stage_tabm()


if __name__ == "__main__":
    main()
