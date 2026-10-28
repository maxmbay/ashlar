"""One houses fit: the 0.8251 TabM setup with a distinct shuffle per member.

Does not change the residual-MLP default. Does not overwrite jax-results.json.

Run:

    JAX_PLATFORMS=cpu python examples/benchmark_tabm_unshared.py
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

OUT_PATH = Path("/tmp/ashlar-bench/tabm-unshared-results.json")


def main():
    frame = pd.read_csv(benchmark.DATA_DIR / "reg_num-houses.csv")
    train, val, test, provenance = benchmark.grinsztajn_medium_split(frame, np.random.RandomState(0))
    target = frame.columns[-1]
    model = tabm(
        numeric_scaling="quantile_tabr",
        standardize_target=True,
        share_training_batches=False,
        max_epochs=50,
    )
    print("start unshared", flush=True)
    started = time.perf_counter()
    model.fit(train.drop(columns=[target]), train[target].to_numpy())
    paper_val = benchmark.regression_scores(val[target].to_numpy(), model.predict(val.drop(columns=[target])))
    test_scores = benchmark.regression_scores(test[target].to_numpy(), model.predict(test.drop(columns=[target])))
    jax.block_until_ready(test_scores["r2"])
    elapsed = time.perf_counter() - started
    val_loss = [float(value) for value in model.history_["val_loss"]]
    payload = {
        "split": provenance,
        "name": "quantile_tabr_label_zscore_unshared",
        "share_training_batches": model.share_training_batches,
        "numeric_scaling": model.numeric_scaling,
        "standardize_target": model.standardize_target,
        "n_quantiles": int(model.numeric_transform_.quantiles.shape[0]),
        "learning_rate": model.learning_rate,
        "weight_decay": model.weight_decay,
        "optimizer": "adamw",
        "max_epochs": model.max_epochs,
        "patience": model.patience,
        "epochs_ran": len(val_loss),
        "best_val_loss": min(val_loss) if val_loss else None,
        "last_val_loss": val_loss[-1] if val_loss else None,
        "paper_val": paper_val,
        "test": test_scores,
        "wall_seconds": elapsed,
        "backbone_share": model.core_.backbone.share_training_batches,
        "k": int(model.core_.backbone.k),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"r2={test_scores['r2']:.6f} epochs={payload['epochs_ran']}", flush=True)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
