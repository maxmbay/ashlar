"""Houses-only TabM borrowings. Does not change the residual-MLP default.

``tabm()`` with the scored TabM-D epoch cap of 50. AdamW stays the optimizer.
One run uses ``quantile_tabr``, one uses a train-only label z-score inverted
in ``predict``, and one uses both. Does not overwrite jax-results.json.

Run:

    JAX_PLATFORMS=cpu python examples/benchmark_tabm_borrow.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pandas as pd

import benchmark
from ashlar import tabm

OUT_PATH = Path("/tmp/ashlar-bench/tabm-borrow-results.json")

RUNS = (
    {"name": "quantile_tabr", "numeric_scaling": "quantile_tabr", "max_epochs": 50},
    {"name": "label_zscore", "standardize_target": True, "max_epochs": 50},
    {
        "name": "quantile_tabr_and_label_zscore",
        "numeric_scaling": "quantile_tabr",
        "standardize_target": True,
        "max_epochs": 50,
    },
)


def _houses():
    frame = pd.read_csv(benchmark.DATA_DIR / "reg_num-houses.csv")
    rng = np.random.RandomState(0)
    train, val, test, provenance = benchmark.grinsztajn_medium_split(frame, rng)
    target = frame.columns[-1]
    return train, val, test, target, provenance


def _fit_score(train, val, test, target, name, **overrides):
    model = tabm(**overrides)
    print(f"start {name}", flush=True)
    started = time.perf_counter()
    model.fit(train.drop(columns=[target]), train[target].to_numpy())
    paper_val = benchmark.regression_scores(val[target].to_numpy(), model.predict(val.drop(columns=[target])))
    test_scores = benchmark.regression_scores(test[target].to_numpy(), model.predict(test.drop(columns=[target])))
    jax.block_until_ready(test_scores["r2"])
    elapsed = time.perf_counter() - started
    val_loss = [float(value) for value in model.history_["val_loss"]]
    n_quantiles = None
    if model.numeric_transform_.scaling == "quantile_tabr":
        n_quantiles = int(model.numeric_transform_.quantiles.shape[0])
    row = {
        "name": name,
        "overrides": overrides,
        "learning_rate": model.learning_rate,
        "weight_decay": model.weight_decay,
        "feature_dropout": model.feature_dropout,
        "numeric_scaling": model.numeric_scaling,
        "standardize_target": model.standardize_target,
        "target_mean": None if model.target_mean_ is None else float(model.target_mean_),
        "target_std": None if model.target_std_ is None else float(model.target_std_),
        "n_quantiles": n_quantiles,
        "batch_size": model.batch_size,
        "patience": model.patience,
        "max_epochs": model.max_epochs,
        "backbone": model.backbone,
        "backbone_kwargs": model.backbone_kwargs,
        "optimizer": "adamw",
        "epochs_ran": len(val_loss),
        "best_val_loss": min(val_loss) if val_loss else None,
        "last_val_loss": val_loss[-1] if val_loss else None,
        "paper_val": paper_val,
        "test": test_scores,
        "wall_seconds": elapsed,
    }
    print(f"{name} r2={row['test']['r2']:.6f} epochs={row['epochs_ran']} knots={n_quantiles}", flush=True)
    return row


def main():
    train, val, test, target, provenance = _houses()
    rows = [_fit_score(train, val, test, target, **run) for run in RUNS]
    OUT_PATH.write_text(json.dumps({"split": provenance, "runs": rows}, indent=2))
    print(OUT_PATH)


if __name__ == "__main__":
    main()
