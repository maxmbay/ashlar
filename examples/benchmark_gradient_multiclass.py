"""Score the multiclass gradient-penalty classifier on five public tables.

The two-class path is unchanged. Three or more classes use one logit per class,
softmax cross-entropy, and an L1 penalty on the input Jacobian. Default
``XGBClassifier`` is fit on the same outer split. The split is
``grinsztajn_medium_split`` with ``RandomState(0)``. Paper-validation is
recorded and not used to train or to select. Does not change the residual
regression default and does not refit earlier result files.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_multiclass.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import xgboost
from sklearn.datasets import fetch_openml
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
    _init_sparse_multi,
    _multi_logits,
    _multi_loss,
)
from ashlar.recipes import DEFAULT_RECIPE

DATA_DIR = Path("/tmp/ashlar-bench")
OUT_PATH = DATA_DIR / "gradient-multiclass-results.json"

DATASETS = (
    {
        "name": "letter",
        "openml": 6,
        "source": "https://www.openml.org/d/6",
        "description": "OpenML letter. 26 classes, 16 numeric features.",
    },
    {
        "name": "pendigits",
        "openml": 32,
        "source": "https://www.openml.org/d/32",
        "description": "OpenML pendigits. 10 classes, 16 numeric features.",
    },
    {
        "name": "optdigits",
        "openml": 28,
        "source": "https://www.openml.org/d/28",
        "description": "OpenML optdigits. 10 classes, 64 numeric features.",
    },
    {
        "name": "satimage",
        "openml": 182,
        "source": "https://www.openml.org/d/182",
        "description": "OpenML satimage. 6 classes, 36 numeric features.",
    },
    {
        "name": "waveform-5000",
        "openml": 60,
        "source": "https://www.openml.org/d/60",
        "description": "OpenML waveform-5000. 3 classes, 40 numeric features.",
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
    model = _init_sparse_multi(jax.random.PRNGKey(0), 4, 3)
    rows = jnp.asarray(np.random.default_rng(0).normal(size=(8, 4)), dtype=jnp.float32)
    target = jnp.asarray([0, 1, 2, 0, 1, 2, 1, 0], dtype=jnp.int32)
    loss, grads = eqx.filter_value_and_grad(_multi_loss)(model, rows, target)
    parameter_grads = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    logits = _multi_logits(model, rows)
    likelihood = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target))

    def one_logits(row):
        return _multi_logits(model, row[None, :])[0]

    jacobian = jax.vmap(jax.jacobian(one_logits))(rows)
    penalty = GRADIENT_LAMBDA * jnp.mean(jnp.abs(jacobian))
    finite_parameters = all(np.isfinite(leaf).all() for leaf in parameter_grads)
    finite_input = bool(np.isfinite(np.asarray(jacobian)).all())
    matches = abs(float(loss) - float(likelihood + penalty)) < 1e-5
    penalty_present = float(jnp.mean(jnp.abs(jacobian))) > 0.0
    if not (finite_parameters and finite_input and matches and penalty_present and np.isfinite(float(loss))):
        raise SystemExit("multiclass gradient-penalty loss is not differentiable")
    return {
        "differentiable": True,
        "loss": "mean softmax cross-entropy plus lambda times mean absolute input Jacobian of the class logits",
        "activation": "silu",
        "two_class_path": "unchanged one-logit sigmoid loss",
        "gradient_lambda": float(GRADIENT_LAMBDA),
        "width": int(GRADIENT_WIDTH),
        "learning_rate": float(GRADIENT_LEARNING_RATE),
        "batch_size": int(GRADIENT_BATCH_SIZE),
        "max_epochs": int(GRADIENT_MAX_EPOCHS),
        "patience": int(GRADIENT_PATIENCE),
        "probe_loss": float(loss),
        "probe_penalty": float(penalty),
        "probe_input_jacobian_finite": True,
        "probe_parameter_gradient_finite": True,
    }


def _frame(spec: dict) -> pd.DataFrame:
    cache = DATA_DIR / f"openml-{spec['openml']}-{spec['name']}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    bunch = fetch_openml(
        data_id=spec["openml"],
        as_frame=True,
        parser="liac-arff",
        data_home=str(DATA_DIR / "openml-cache"),
    )
    features = bunch.data.apply(pd.to_numeric, errors="coerce")
    if int(features.isna().sum().sum()) != 0:
        raise SystemExit(f"{spec['name']} has a non-numeric feature")
    target_name = str(bunch.target.name or "target")
    if target_name in features.columns:
        target_name = "target"
    frame = features.copy()
    frame[target_name] = np.asarray(bunch.target)
    frame.to_csv(cache, index=False)
    return frame


def _parts(spec: dict):
    frame = _frame(spec)
    target = frame.columns[-1]
    numeric = frame.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    if int(numeric.isna().sum().sum()) != 0 or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise SystemExit(f"{spec['name']} has a non-finite feature")
    n_classes = int(frame[target].nunique())
    if numeric.shape[1] < 1 or n_classes < 3:
        raise SystemExit(f"{spec['name']} is not a multiclass numeric table")
    prepared = numeric.copy()
    prepared[target] = frame[target].to_numpy()
    train, val, test, split_info = benchmark.grinsztajn_medium_split(prepared, np.random.RandomState(0))
    if int(train[target].nunique()) != n_classes:
        raise SystemExit(f"{spec['name']} train split dropped a class")
    split_info["target"] = str(target)
    split_info["source"] = spec["source"]
    split_info["openml"] = spec["openml"]
    split_info["description"] = spec["description"]
    split_info["n_classes"] = n_classes
    split_info["class_counts"] = {str(key): int(value) for key, value in frame[target].value_counts().items()}

    def pack(part):
        return part.drop(columns=[target]).to_numpy(dtype=np.float64), part[target].to_numpy()

    return (split_info, *pack(train), *pack(val), *pack(test))


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
        "n_classes": int(model.n_classes_),
        "wall_seconds": time.perf_counter() - started,
        "config": (
            "GradientSparseNet(random_state=0): width 128, SiLU, one logit per class, "
            "softmax cross-entropy plus 0.01 times mean absolute input Jacobian, "
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
            f"features {split_info['n_features']} classes {split_info['n_classes']}",
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
    matched = []
    for spec in DATASETS:
        row = payload["datasets"][spec["name"]]
        gradient = row["gradient_penalty"]["test_accuracy"]
        booster = row["xgboost_defaults"]["test_accuracy"]
        beats_or_matches = bool(gradient >= booster)
        final.append(
            {
                "name": spec["name"],
                "gradient_penalty": gradient,
                "xgboost_defaults": booster,
                "beats_or_matches": beats_or_matches,
            }
        )
        if beats_or_matches:
            matched.append(spec["name"])
    payload["final"] = final
    payload["beats_or_matches"] = matched
    payload["differentiable"] = True
    _save(payload)
    print(json.dumps({"final": final, "beats_or_matches": matched}, indent=2), flush=True)


if __name__ == "__main__":
    main()
