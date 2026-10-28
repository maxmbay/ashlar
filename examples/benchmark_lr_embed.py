"""Houses-only learning-rate and embedding probes. AdamW throughout.

The base is the no-argument recipe: SiLU residual MLP, feature dropout 0,
weight decay 1e-4, mean/std, learning rate 1e-3, constant schedule.
Each run changes one of those. Does not overwrite jax-results.json.

Run:

    JAX_PLATFORMS=cpu python examples/benchmark_lr_embed.py
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
from ashlar import tabular

OUT_PATH = Path("/tmp/ashlar-bench/lr-embed-results.json")
BASELINE_R2 = 0.7615

RUNS = (
    {"name": "lr_3e-3_constant", "learning_rate": 3e-3, "learning_rate_schedule": "constant"},
    {"name": "lr_3e-4_constant", "learning_rate": 3e-4, "learning_rate_schedule": "constant"},
    {"name": "lr_1e-4_constant", "learning_rate": 1e-4, "learning_rate_schedule": "constant"},
    {"name": "lr_3e-3_linear", "learning_rate": 3e-3, "learning_rate_schedule": "linear"},
    {"name": "linear_relu_embeddings", "num_embeddings": "linear_relu"},
)


def _houses():
    frame = pd.read_csv(benchmark.DATA_DIR / "reg_num-houses.csv")
    rng = np.random.RandomState(0)
    train, val, test, provenance = benchmark.grinsztajn_medium_split(frame, rng)
    target = frame.columns[-1]
    return train, val, test, target, provenance


def _fit_score(train, val, test, target, name, **overrides):
    model = tabular(**overrides)
    started = time.perf_counter()
    model.fit(train.drop(columns=[target]), train[target].to_numpy())
    paper_val = benchmark.regression_scores(val[target].to_numpy(), model.predict(val.drop(columns=[target])))
    test_scores = benchmark.regression_scores(test[target].to_numpy(), model.predict(test.drop(columns=[target])))
    jax.block_until_ready(test_scores["r2"])
    elapsed = time.perf_counter() - started
    val_loss = [float(value) for value in model.history_["val_loss"]]
    row = {
        "name": name,
        "overrides": overrides,
        "learning_rate": model.learning_rate,
        "learning_rate_schedule": model.learning_rate_schedule,
        "weight_decay": model.weight_decay,
        "feature_dropout": model.feature_dropout,
        "numeric_scaling": model.numeric_scaling,
        "num_embeddings": model.num_embeddings,
        "num_embedding_dim": model.num_embedding_dim,
        "activation": model.core_.backbone.activation,
        "optimizer": "adamw",
        "epochs_ran": len(val_loss),
        "best_val_loss": min(val_loss) if val_loss else None,
        "last_val_loss": val_loss[-1] if val_loss else None,
        "paper_val": paper_val,
        "test": test_scores,
        "wall_seconds": elapsed,
        "delta_from_0_7615": float(test_scores["r2"] - BASELINE_R2),
    }
    print(
        f"{name} r2={row['test']['r2']:.6f} lr={row['learning_rate']} schedule={row['learning_rate_schedule']} "
        f"emb={row['num_embeddings']} act={row['activation']} drop={row['feature_dropout']}",
        flush=True,
    )
    return row


def main():
    train, val, test, target, provenance = _houses()
    rows = [_fit_score(train, val, test, target, **run) for run in RUNS]
    OUT_PATH.write_text(json.dumps({"split": provenance, "baseline_r2": BASELINE_R2, "runs": rows}, indent=2))
    print(OUT_PATH)


if __name__ == "__main__":
    main()
