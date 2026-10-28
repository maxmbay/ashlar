"""TabM-D on the five houses redraws already scored for Ashlar and XGBoost.

Dump the splits with the Ashlar environment, then fit with the PyTabKit
environment. This does not fit Ashlar.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_tabm_d_seeds.py --dump-splits
    /tmp/deeptab-venv/bin/python examples/benchmark_tabm_d_seeds.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("/tmp/ashlar-bench")
SPLIT_DIR = DATA_DIR / "tabm-d-seed-splits"
OUT_PATH = DATA_DIR / "tabm-d-seeds-results.json"
SEEDS = (0, 1, 2, 3, 4)


def dump_splits() -> None:
    import benchmark

    frame = pd.read_csv(DATA_DIR / "reg_num-houses.csv")
    target = frame.columns[-1]
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        train, val, test, _ = benchmark.grinsztajn_medium_split(frame, np.random.RandomState(seed))
        path = SPLIT_DIR / f"seed-{seed}.npz"
        np.savez(
            path,
            x_train=train.drop(columns=[target]).to_numpy(),
            y_train=train[target].to_numpy(),
            x_val=val.drop(columns=[target]).to_numpy(),
            y_val=val[target].to_numpy(),
            x_test=test.drop(columns=[target]).to_numpy(),
            y_test=test[target].to_numpy(),
            columns=np.array(list(train.drop(columns=[target]).columns)),
        )
        print(f"dumped seed={seed} train={len(train)} val={len(val)} test={len(test)}", flush=True)


def _load_payload() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {
        "model": "TabM_D_Regressor",
        "kwargs": {
            "device": "cpu",
            "n_epochs": 50,
            "n_cv": 1,
            "n_refit": 0,
            "n_repeats": 1,
            "verbosity": 0,
            "random_state": "seed",
        },
        "seeds": list(SEEDS),
        "runs": [],
    }


def _scores(y_true, y_pred) -> dict:
    from sklearn.metrics import r2_score

    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return {
        "r2": float(r2_score(true, pred)),
        "rmse_transformed": float(np.sqrt(np.mean((true - pred) ** 2))),
    }


def _summarize(runs: list[dict]) -> dict:
    scores = np.array([row["test_r2"] for row in runs], dtype=np.float64)
    return {
        "n": int(len(runs)),
        "ddof": 1,
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)) if len(scores) > 1 else None,
    }


def fit_seeds() -> None:
    from pytabkit import TabM_D_Regressor

    payload = _load_payload()
    done = {int(row["seed"]) for row in payload["runs"]}
    for seed in SEEDS:
        if seed in done:
            print(f"skip seed={seed}", flush=True)
            continue
        arrays = np.load(SPLIT_DIR / f"seed-{seed}.npz", allow_pickle=False)
        columns = [str(name) for name in arrays["columns"].tolist()]
        x_train = pd.DataFrame(arrays["x_train"], columns=columns)
        x_test = pd.DataFrame(arrays["x_test"], columns=columns)
        x_val = pd.DataFrame(arrays["x_val"], columns=columns)
        y_train = arrays["y_train"]
        model = TabM_D_Regressor(
            device="cpu",
            random_state=seed,
            n_epochs=50,
            n_cv=1,
            n_refit=0,
            n_repeats=1,
            verbosity=0,
        )
        print(f"start seed={seed}", flush=True)
        started = time.perf_counter()
        model.fit(x_train, y_train)
        test_scores = _scores(arrays["y_test"], model.predict(x_test))
        val_scores = _scores(arrays["y_val"], model.predict(x_val))
        elapsed = time.perf_counter() - started
        row = {
            "seed": seed,
            "random_state": seed,
            "test_r2": test_scores["r2"],
            "test_rmse": test_scores["rmse_transformed"],
            "paper_val_r2": val_scores["r2"],
            "wall_seconds": elapsed,
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
        }
        print(f"seed={seed} r2={test_scores['r2']:.6f} wall={elapsed:.1f}", flush=True)
        payload["runs"].append(row)
        payload["runs"].sort(key=lambda item: item["seed"])
        if len(payload["runs"]) >= 2:
            payload["summary"] = _summarize(payload["runs"])
        OUT_PATH.write_text(json.dumps(payload, indent=2))
    payload["summary"] = _summarize(payload["runs"])
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(payload["summary"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-splits", action="store_true")
    args = parser.parse_args()
    if args.dump_splits:
        dump_splits()
    else:
        fit_seeds()


if __name__ == "__main__":
    main()
