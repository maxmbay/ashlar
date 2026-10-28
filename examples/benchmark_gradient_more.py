"""Score the unchanged gradient-penalty classifier on three more tables.

bank-marketing, MagicTelescope, and credit are numerical classification tables
from the same Grinsztajn release as electricity, eye_movements, and california.
The network is GradientSparseNet with its scored defaults. Default
XGBClassifier is fit on the same outer split. Paper-validation is recorded and
not used to train or to select. Does not change the residual regression default
and does not refit earlier result files.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_more.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xgboost
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark
from ashlar.priors import (
    GRADIENT_BATCH_SIZE,
    GRADIENT_LAMBDA,
    GRADIENT_LEARNING_RATE,
    GRADIENT_MAX_EPOCHS,
    GRADIENT_PATIENCE,
    GRADIENT_WIDTH,
    GradientSparseNet,
    _init_sparse,
    _sparse_logits,
    _sparse_loss,
)
from ashlar.recipes import DEFAULT_RECIPE

DATA_DIR = Path("/tmp/ashlar-bench")
OUT_PATH = DATA_DIR / "gradient-more-results.json"
HF = "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main"

DATASETS = (
    {
        "name": "bank-marketing",
        "file": "clf_num-bank-marketing.csv",
        "source": f"{HF}/clf_num/bank-marketing.csv",
        "openml_original": 1461,
        "openml_transformed": 44126,
        "description": (
            "Grinsztajn numerical-classification bank-marketing table. "
            "Binary label, 7 numeric features, same feature count as electricity."
        ),
    },
    {
        "name": "MagicTelescope",
        "file": "clf_num-MagicTelescope.csv",
        "source": f"{HF}/clf_num/MagicTelescope.csv",
        "openml_original": 1120,
        "openml_transformed": 44125,
        "description": (
            "Grinsztajn numerical-classification MagicTelescope table. "
            "Binary label, 10 numeric features, between eye_movements and california in size."
        ),
    },
    {
        "name": "credit",
        "file": "clf_num-credit.csv",
        "source": f"{HF}/clf_num/credit.csv",
        "openml_original": None,
        "openml_transformed": 44089,
        "description": (
            "Grinsztajn numerical-classification credit table. "
            "Binary SeriousDlqin2yrs, 10 numeric features, nearest unused row count to california."
        ),
    },
)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"datasets": {}}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _class_index(y_train):
    classes = np.unique(np.asarray(y_train))
    mapping = {value: index for index, value in enumerate(classes.tolist())}

    def transform(y):
        encoded = []
        for value in np.asarray(y).reshape(-1).tolist():
            if value not in mapping:
                raise KeyError(value)
            encoded.append(mapping[value])
        return np.asarray(encoded, dtype=np.int32)

    return classes, transform


def _probe() -> dict:
    model = _init_sparse(jax.random.PRNGKey(0), 4)
    rows = jnp.asarray(np.random.default_rng(0).normal(size=(8, 4)), dtype=jnp.float32)
    target = jnp.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=jnp.float32)
    loss, grads = eqx.filter_value_and_grad(_sparse_loss)(model, rows, target)
    parameter_grads = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    logit = _sparse_logits(model, rows)
    likelihood = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, target))

    def one_logit(row):
        return _sparse_logits(model, row[None, :])[0]

    gradient = jax.vmap(jax.grad(one_logit))(rows)
    penalty = GRADIENT_LAMBDA * jnp.mean(jnp.abs(gradient))
    assembled = likelihood + penalty
    finite_parameters = all(np.isfinite(leaf).all() for leaf in parameter_grads)
    finite_input = bool(np.isfinite(np.asarray(gradient)).all())
    matches = abs(float(loss) - float(assembled)) < 1e-5
    penalty_present = float(jnp.mean(jnp.abs(gradient))) > 0.0
    differentiable = bool(
        finite_parameters and finite_input and matches and penalty_present and np.isfinite(float(loss))
    )
    if not differentiable:
        raise SystemExit("gradient-penalty loss is not differentiable")
    return {
        "differentiable": True,
        "loss": "mean sigmoid binary cross-entropy plus lambda times mean absolute input gradient",
        "activation": "silu",
        "hard_splits": False,
        "gradient_lambda": float(GRADIENT_LAMBDA),
        "width": int(GRADIENT_WIDTH),
        "learning_rate": float(GRADIENT_LEARNING_RATE),
        "batch_size": int(GRADIENT_BATCH_SIZE),
        "max_epochs": int(GRADIENT_MAX_EPOCHS),
        "patience": int(GRADIENT_PATIENCE),
        "probe_loss": float(loss),
        "probe_penalty": float(penalty),
        "probe_input_gradient_finite": True,
        "probe_parameter_gradient_finite": True,
    }


def _parts(spec: dict):
    frame = pd.read_csv(DATA_DIR / spec["file"])
    target = frame.columns[-1]
    features = frame.drop(columns=[target])
    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise SystemExit(f"{spec['name']} has a non-finite feature")
    if features.shape[1] < 1 or frame[target].nunique() != 2:
        raise SystemExit(f"{spec['name']} is not a binary numeric table")
    train, val, test, split_info = benchmark.grinsztajn_medium_split(frame, np.random.RandomState(0))
    split_info["target"] = str(target)
    split_info["source"] = spec["source"]
    split_info["openml_transformed"] = spec["openml_transformed"]
    split_info["openml_original"] = spec["openml_original"]
    split_info["description"] = spec["description"]
    split_info["class_counts"] = {str(key): int(value) for key, value in frame[target].value_counts().items()}

    def pack(part):
        return part.drop(columns=[target]).to_numpy(dtype=np.float64), part[target].to_numpy()

    x_train, y_train = pack(train)
    x_val, y_val = pack(val)
    x_test, y_test = pack(test)
    return split_info, x_train, y_train, x_val, y_val, x_test, y_test


def _gradient(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    started = time.perf_counter()
    model = GradientSparseNet(random_state=0, verbose=True)
    model.fit(x_train, y_train, "classification")
    test_pred = model.predict(x_test)
    val_pred = model.predict(x_val)
    jax.block_until_ready(test_pred)
    return {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "paper_val_accuracy": float(accuracy_score(y_val, val_pred)),
        "best_epoch": int(model.best_epoch_),
        "selection_metric": float(model.selection_metric_),
        "n_fit": int(model.n_fit_),
        "n_internal_val": int(model.n_val_),
        "wall_seconds": time.perf_counter() - started,
        "config": (
            "GradientSparseNet(random_state=0): width 128, SiLU, "
            "binary cross-entropy plus 0.01 times mean absolute input gradient, "
            "Adam 1e-3, batch 256, patience 16, cap 80"
        ),
    }


def _xgboost(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    classes, transform = _class_index(y_train)
    started = time.perf_counter()
    booster = xgboost.XGBClassifier()
    booster.fit(x_train, transform(y_train))
    test_pred = classes[np.asarray(booster.predict(x_test), dtype=int)]
    val_pred = classes[np.asarray(booster.predict(x_val), dtype=int)]
    return {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "paper_val_accuracy": float(accuracy_score(y_val, val_pred)),
        "wall_seconds": time.perf_counter() - started,
        "config": "XGBClassifier() with no arguments, labels mapped to 0..K-1 and predictions mapped back",
    }


def main() -> None:
    if DEFAULT_RECIPE["task"] != "regression" or DEFAULT_RECIPE["standardize_target"] is not True:
        raise SystemExit("residual regression default changed")
    if GRADIENT_LAMBDA != 1e-2 or GRADIENT_LEARNING_RATE != 1e-3 or GRADIENT_WIDTH != 128:
        raise SystemExit("gradient-penalty defaults changed")
    if GRADIENT_MAX_EPOCHS != 80 or GRADIENT_PATIENCE != 16 or GRADIENT_BATCH_SIZE != 256:
        raise SystemExit("gradient-penalty schedule changed")
    payload = _load()
    if "probe" not in payload:
        payload["probe"] = _probe()
        payload["residual_default"] = {
            "task": DEFAULT_RECIPE["task"],
            "standardize_target": DEFAULT_RECIPE["standardize_target"],
            "backbone": DEFAULT_RECIPE["backbone"],
        }
        _save(payload)
        print("probe differentiable", payload["probe"]["differentiable"], flush=True)
    payload.setdefault("datasets", {})
    for spec in DATASETS:
        row = payload["datasets"].setdefault(spec["name"], {})
        if "gradient_penalty" in row and "xgboost_defaults" in row:
            print(f"skip {spec['name']}", flush=True)
            continue
        split_info, x_train, y_train, x_val, y_val, x_test, y_test = _parts(spec)
        row["split"] = split_info
        print(
            f"== {spec['name']} {split_info['n_train']}/{split_info['n_val']}/{split_info['n_test']} "
            f"features {split_info['n_features']}",
            flush=True,
        )
        if "gradient_penalty" not in row:
            print("  gradient penalty", flush=True)
            row["gradient_penalty"] = _gradient(x_train, y_train, x_val, y_val, x_test, y_test)
            print("   ", row["gradient_penalty"]["test_accuracy"], flush=True)
            _save(payload)
        if "xgboost_defaults" not in row:
            print("  xgboost defaults", flush=True)
            row["xgboost_defaults"] = _xgboost(x_train, y_train, x_val, y_val, x_test, y_test)
            print("   ", row["xgboost_defaults"]["test_accuracy"], flush=True)
            _save(payload)
    final = []
    for spec in DATASETS:
        row = payload["datasets"][spec["name"]]
        final.append(
            {
                "name": spec["name"],
                "gradient_penalty": row["gradient_penalty"]["test_accuracy"],
                "xgboost_defaults": row["xgboost_defaults"]["test_accuracy"],
            }
        )
    payload["final"] = final
    payload["differentiable"] = True
    _save(payload)
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
