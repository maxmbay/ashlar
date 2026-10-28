"""Five seeds of the 0.8321 houses recipe against default XGBoost.

Each seed redraws the Grinsztajn medium train/paper-val/test partition with
``numpy.random.RandomState(seed)`` and fits ``tabm`` with the same integer as
``random_state``. The paper validation and test rows of that seed are not used
to train or to select the model. ``quantile_tabr`` noise stays at its fixed
generator seed of 0.

California is not run here: ``standardize_target`` and
``share_training_batches=False`` are regression-only.

Does not change the residual-MLP default. Does not overwrite jax-results.json.

Run:

    JAX_PLATFORMS=cpu python examples/benchmark_tabm_seeds.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
import xgboost

import benchmark
from ashlar import tabm

OUT_PATH = Path("/tmp/ashlar-bench/tabm-seeds-results.json")
SEEDS = (0, 1, 2, 3, 4)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {
        "recipe": "quantile_tabr_label_zscore_unshared",
        "seeds": list(SEEDS),
        "split_protocol": "grinsztajn_medium_data_to_train_test",
        "note": (
            "Each seed passes numpy.random.RandomState(seed) into "
            "grinsztajn_medium_split. The provenance string inside that helper "
            "always says RandomState(0); the seed field on each row is the one used."
        ),
        "runs": [],
    }


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _one(frame: pd.DataFrame, seed: int) -> dict:
    train, val, test, provenance = benchmark.grinsztajn_medium_split(
        frame, np.random.RandomState(seed)
    )
    target = frame.columns[-1]
    x_train = train.drop(columns=[target])
    y_train = train[target].to_numpy()
    x_test = test.drop(columns=[target])
    y_test = test[target].to_numpy()
    x_val = val.drop(columns=[target])
    y_val = val[target].to_numpy()

    model = tabm(
        numeric_scaling="quantile_tabr",
        standardize_target=True,
        share_training_batches=False,
        max_epochs=50,
        random_state=seed,
    )
    print(f"start seed={seed}", flush=True)
    started = time.perf_counter()
    model.fit(x_train, y_train)
    paper_val = benchmark.regression_scores(y_val, model.predict(x_val))
    test_scores = benchmark.regression_scores(y_test, model.predict(x_test))
    elapsed = time.perf_counter() - started
    val_loss = [float(value) for value in model.history_["val_loss"]]

    booster = xgboost.XGBRegressor()
    xgb_started = time.perf_counter()
    booster.fit(x_train, y_train)
    xgb_test = benchmark.regression_scores(y_test, booster.predict(x_test))
    xgb_elapsed = time.perf_counter() - xgb_started

    row = {
        "seed": seed,
        "n_train": int(len(train)),
        "n_paper_val": int(len(val)),
        "n_test": int(len(test)),
        "provenance_label": provenance["random_state"],
        "ashlar": {
            "test_r2": test_scores["r2"],
            "test_rmse": test_scores["rmse_transformed"],
            "paper_val_r2": paper_val["r2"],
            "epochs_ran": len(val_loss),
            "best_val_loss": min(val_loss) if val_loss else None,
            "last_val_loss": val_loss[-1] if val_loss else None,
            "wall_seconds": elapsed,
            "n_quantiles": int(model.numeric_transform_.quantiles.shape[0]),
            "target_mean": float(model.target_mean_),
            "target_std": float(model.target_std_),
            "learning_rate": model.learning_rate,
            "weight_decay": model.weight_decay,
            "share_training_batches": model.share_training_batches,
            "k": int(model.core_.backbone.k),
        },
        "xgboost": {
            "test_r2": xgb_test["r2"],
            "test_rmse": xgb_test["rmse_transformed"],
            "wall_seconds": xgb_elapsed,
            "config": "XGBRegressor() defaults",
        },
    }
    print(
        f"seed={seed} ashlar_r2={test_scores['r2']:.6f} xgb_r2={xgb_test['r2']:.6f} "
        f"epochs={row['ashlar']['epochs_ran']} wall={elapsed:.1f}",
        flush=True,
    )
    return row


def _summarize(runs: list[dict]) -> dict:
    ashlar = np.array([row["ashlar"]["test_r2"] for row in runs], dtype=np.float64)
    xgb = np.array([row["xgboost"]["test_r2"] for row in runs], dtype=np.float64)
    return {
        "n": int(len(runs)),
        "ddof": 1,
        "ashlar_mean": float(ashlar.mean()),
        "ashlar_std": float(ashlar.std(ddof=1)),
        "xgboost_mean": float(xgb.mean()),
        "xgboost_std": float(xgb.std(ddof=1)),
        "ashlar_minus_xgboost_mean": float(ashlar.mean() - xgb.mean()),
        "paired_deltas": [float(a - b) for a, b in zip(ashlar, xgb)],
        "ashlar_wins_seeds": int(np.sum(ashlar > xgb)),
    }


def main():
    frame = pd.read_csv(benchmark.DATA_DIR / "reg_num-houses.csv")
    payload = _load()
    done = {int(row["seed"]) for row in payload["runs"]}
    for seed in SEEDS:
        if seed in done:
            print(f"skip seed={seed}", flush=True)
            continue
        payload["runs"].append(_one(frame, seed))
        payload["runs"].sort(key=lambda row: row["seed"])
        if len(payload["runs"]) >= 2:
            payload["summary"] = _summarize(payload["runs"])
        _save(payload)
    payload["summary"] = _summarize(payload["runs"])
    _save(payload)
    summary = payload["summary"]
    print(
        f"mean ashlar={summary['ashlar_mean']:.6f} std={summary['ashlar_std']:.6f} "
        f"xgb={summary['xgboost_mean']:.6f} std={summary['xgboost_std']:.6f} "
        f"delta={summary['ashlar_minus_xgboost_mean']:.6f}",
        flush=True,
    )
    print(OUT_PATH)


if __name__ == "__main__":
    main()
