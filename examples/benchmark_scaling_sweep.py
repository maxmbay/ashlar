"""Houses-only probes. Does not overwrite the published jax-results.json.

1. Quantile numeric map on the regularized residual MLP. Learning rate,
   feature dropout, weight decay, epochs, and backbone stay at the default.
2. Dropout and activation sweep on the same split with mean/std scaling.
   Feature dropout is 0 or 0.1. Activation is relu (the default), gelu, or silu.

Run:

    JAX_PLATFORMS=cpu python examples/benchmark_scaling_sweep.py
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
from ashlar import ASRRegressor

OUT_PATH = Path("/tmp/ashlar-bench/scaling-sweep-results.json")
BASELINE_R2 = 0.7322


def _houses():
    frame = pd.read_csv(benchmark.DATA_DIR / "reg_num-houses.csv")
    rng = np.random.RandomState(0)
    train, val, test, provenance = benchmark.grinsztajn_medium_split(frame, rng)
    target = frame.columns[-1]
    return train, val, test, target, provenance


def _fit_score(train, val, test, target, **overrides):
    model = ASRRegressor(**overrides)
    started = time.perf_counter()
    model.fit(train.drop(columns=[target]), train[target].to_numpy())
    paper_val = benchmark.regression_scores(
        val[target].to_numpy(), model.predict(val.drop(columns=[target]))
    )
    test_scores = benchmark.regression_scores(
        test[target].to_numpy(), model.predict(test.drop(columns=[target]))
    )
    jax.block_until_ready(test_scores["r2"])
    elapsed = time.perf_counter() - started
    val_loss = [float(value) for value in model.history_["val_loss"]]
    return {
        "overrides": overrides,
        "numeric_scaling": model.numeric_scaling,
        "feature_dropout": model.feature_dropout,
        "activation": model.core_.backbone.activation,
        "learning_rate": model.learning_rate,
        "weight_decay": model.weight_decay,
        "max_epochs": model.max_epochs,
        "patience": model.patience,
        "epochs_ran": len(val_loss),
        "best_val_loss": min(val_loss) if val_loss else None,
        "last_val_loss": val_loss[-1] if val_loss else None,
        "n_quantiles": None
        if model.numeric_transform_.scaling != "quantile"
        else int(model.numeric_transform_.quantiles.shape[0]),
        "paper_val": paper_val,
        "test": test_scores,
        "wall_seconds": elapsed,
        "delta_from_0_7322": float(test_scores["r2"] - BASELINE_R2),
    }


def main():
    train, val, test, target, provenance = _houses()
    quantile = _fit_score(train, val, test, target, numeric_scaling="quantile")
    quantile["hypothesis_failed"] = abs(quantile["test"]["r2"] - BASELINE_R2) <= 0.01
    sweep = []
    for dropout in (0.0, 0.1):
        for activation in ("relu", "gelu", "silu"):
            row = _fit_score(
                train,
                val,
                test,
                target,
                feature_dropout=dropout,
                backbone_kwargs={"activation": activation},
            )
            sweep.append(row)
            print(
                f"dropout={dropout} activation={activation} r2={row['test']['r2']:.6f}",
                flush=True,
            )
    print(
        f"quantile r2={quantile['test']['r2']:.6f} failed={quantile['hypothesis_failed']}",
        flush=True,
    )
    payload = {
        "split": provenance,
        "baseline_r2": BASELINE_R2,
        "quantile": quantile,
        "sweep": sweep,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(OUT_PATH)


if __name__ == "__main__":
    main()
