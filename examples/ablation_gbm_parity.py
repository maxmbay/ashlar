"""Ablations behind the new defaults: which change buys what, one at a time.

Each config is a dict of ``TabularEstimator`` arguments. A run is one
(dataset, config, seed) triple, appended as one JSON line, so runs can go side
by side and a finished run is skipped. Same 80/20 splits as
``benchmark_readme.py``.

    JAX_PLATFORMS=cpu python examples/ablation_gbm_parity.py --data california --configs default wide
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ashlar import TabularEstimator
from benchmark_readme import adult, california, diamonds

WIDE = {"n_blocks": 3, "d_block": 256, "dropout": 0.1}
TABM = {"n_blocks": 3, "d_block": 256, "dropout": 0.1, "k": 32}
TABM_EMB = {"n_blocks": 2, "d_block": 256, "dropout": 0.1, "k": 32, "start_scaling_init": "normal"}
LONG = {"max_epochs": 100, "patience": 16}

CONFIGS = {
    # The defaults before this round: two blocks of 64, no numeric embeddings,
    # batch 64, constant 1e-3, 50 epochs, patience 10.
    "v0": {"backbone_kwargs": {"n_blocks": 2, "d_block": 64, "dropout": 0.0}, "num_embeddings": None,
           "batch_size": 64, "max_epochs": 50, "patience": 10, "learning_rate": 1e-3,
           "learning_rate_schedule": "constant"},
    "default": {},
    "bs256": {"batch_size": 256, "learning_rate": 2e-3},
    "wide": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 1e-3, **LONG},
    "wide_cosine": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3,
                    "learning_rate_schedule": "cosine", **LONG},
    "wide_ple": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 1e-3,
                 "num_embeddings": "piecewise", "num_embedding_dim": 16, **LONG},
    "wide_periodic": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 1e-3,
                      "num_embeddings": "periodic", "num_embedding_dim": 16, **LONG},
    "tabm": {"backbone": "tabm", "backbone_kwargs": TABM, "batch_size": 256, "learning_rate": 2e-3,
             "weight_decay": 0.0, **LONG},
    "tabm_ple": {"backbone": "tabm", "backbone_kwargs": TABM_EMB, "batch_size": 256, "learning_rate": 2e-3,
                 "weight_decay": 0.0, "num_embeddings": "piecewise", "num_embedding_dim": 16, **LONG},
    "tabm_periodic": {"backbone": "tabm", "backbone_kwargs": TABM_EMB, "batch_size": 256, "learning_rate": 2e-3,
                      "weight_decay": 0.0, "num_embeddings": "periodic", "num_embedding_dim": 16, **LONG},
    "tabm_ple_cosine": {"backbone": "tabm", "backbone_kwargs": TABM_EMB, "batch_size": 256,
                        "learning_rate": 2e-3, "learning_rate_schedule": "cosine", "weight_decay": 0.0,
                        "num_embeddings": "piecewise", "num_embedding_dim": 16, **LONG},
    "tabm_ple_quantile": {"backbone": "tabm", "backbone_kwargs": TABM_EMB, "batch_size": 256,
                          "learning_rate": 2e-3, "weight_decay": 0.0, "numeric_scaling": "quantile",
                          "num_embeddings": "piecewise", "num_embedding_dim": 16, **LONG},
    # Variations on the wide MLP with piecewise embeddings.
    "ple_cosine": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3,
                   "learning_rate_schedule": "cosine", "num_embeddings": "piecewise", "num_embedding_dim": 16, **LONG},
    "ple_cosine_512": {"backbone_kwargs": {**WIDE, "d_block": 512}, "batch_size": 256, "learning_rate": 1e-3,
                       "learning_rate_schedule": "cosine", "num_embeddings": "piecewise", "num_embedding_dim": 16,
                       **LONG},
    "ple_cosine_drop0": {"backbone_kwargs": {**WIDE, "dropout": 0.0}, "batch_size": 256, "learning_rate": 2e-3,
                         "learning_rate_schedule": "cosine", "num_embeddings": "piecewise", "num_embedding_dim": 16,
                         **LONG},
    "ple_cosine_drop2": {"backbone_kwargs": {**WIDE, "dropout": 0.2}, "batch_size": 256, "learning_rate": 2e-3,
                         "learning_rate_schedule": "cosine", "num_embeddings": "piecewise", "num_embedding_dim": 16,
                         **LONG},
    "ple_cosine_bs1024": {"backbone_kwargs": WIDE, "batch_size": 1024, "learning_rate": 4e-3,
                          "learning_rate_schedule": "cosine", "num_embeddings": "piecewise", "num_embedding_dim": 16,
                          **LONG},
    "ple_cosine_d32": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3,
                       "learning_rate_schedule": "cosine", "num_embeddings": "piecewise", "num_embedding_dim": 32,
                       **LONG},
    "ple_cosine_wd": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3, "weight_decay": 1e-2,
                      "learning_rate_schedule": "cosine", "num_embeddings": "piecewise", "num_embedding_dim": 16,
                      **LONG},
    "ple_cosine_quantile": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3,
                            "numeric_scaling": "quantile", "learning_rate_schedule": "cosine",
                            "num_embeddings": "piecewise", "num_embedding_dim": 16, **LONG},
    # Nonlinear embeddings: a ReLU on the piecewise embedding, and a hidden layer per categorical column.
    "ple_relu_cosine": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3,
                        "learning_rate_schedule": "cosine", "num_embeddings": "piecewise_relu",
                        "num_embedding_dim": 16, **LONG},
    "ple_cosine_cathidden": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3,
                             "learning_rate_schedule": "cosine", "num_embeddings": "piecewise",
                             "num_embedding_dim": 16, "categorical_embedding_hidden": True,
                             "categorical_embedding_dim": 16, **LONG},
    "ple_cosine_cat16": {"backbone_kwargs": WIDE, "batch_size": 256, "learning_rate": 2e-3,
                         "learning_rate_schedule": "cosine", "num_embeddings": "piecewise",
                         "num_embedding_dim": 16, "categorical_embedding_dim": 16, **LONG},
}

DATA = {
    "california": ("regression", california),
    "diamonds": ("regression", diamonds),
    "adult": ("classification", adult),
}


def run(data: str, config: str, seed: int, out: Path, extra: dict | None = None):
    task, loader = DATA[data]
    frame, y = loader()
    x_train, x_test, y_train, y_test = train_test_split(frame, y, test_size=0.2, random_state=seed)
    params = dict(CONFIGS[config], **(extra or {}))
    if task == "classification":
        params.update(task="classification", standardize_target=False, prediction_policy="mean_probabilities")
    start = time.time()
    model = TabularEstimator(random_state=seed, **params).fit(x_train, y_train)
    seconds = time.time() - start
    record = {"data": data, "config": config, "seed": seed, "seconds": round(seconds, 1),
              "epochs": len(model.history_["train_loss"])}
    if task == "classification":
        proba = model.predict_proba(x_test)
        record["accuracy"] = float(accuracy_score(y_test, model.classes_[proba.argmax(1)]))
        record["log_loss"] = float(log_loss(y_test, proba, labels=model.classes_))
    else:
        record["rmse"] = float(np.sqrt(np.mean((model.predict(x_test) - y_test) ** 2)))
    with out.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="+", default=list(DATA))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", default="/tmp/ashlar-bench/ablation.jsonl")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            row = json.loads(line)
            done.add((row["data"], row["config"], row["seed"]))
    for data in args.data:
        for config in args.configs:
            for seed in args.seeds:
                if (data, config, seed) not in done:
                    run(data, config, seed, out)


if __name__ == "__main__":
    main()
