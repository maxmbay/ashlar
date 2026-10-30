"""Score the unchanged gradient-penalty classifier on fifteen numerical tables.

These are public binary classification tables that were not in the earlier
gradient-penalty runs. Ten come from the Grinsztajn ``clf_num`` release.
phoneme and kdd_ipums_la_97-small are served from an older commit of that
release. Five are OpenML numerical tables. The split is
``grinsztajn_medium_split`` with ``RandomState(0)``. Default ``XGBClassifier``
is fit on the same outer split. Paper-validation is recorded and not used to
train or to select. Does not change the residual regression default and does
not refit earlier result files.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_suite.py
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
    ASRGradientSparseNet,
    _init_sparse,
    _sparse_logits,
    _sparse_loss,
)
from ashlar.recipes import DEFAULT_RECIPE

DATA_DIR = Path("/tmp/ashlar-bench")
OUT_PATH = DATA_DIR / "gradient-suite-results.json"
HF = "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main/clf_num"
HF_OLDER = (
    "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/"
    "f918b61498213f463fc183c6e070b8cf9fb7c146/clf_num"
)

DATASETS = (
    {
        "name": "house_16H",
        "file": "clf_num-house_16H.csv",
        "source": f"{HF}/house_16H.csv",
        "openml_original": 821,
        "openml_transformed": 44123,
        "description": "Grinsztajn numerical-classification house_16H table. Binary label, 16 numeric features.",
    },
    {
        "name": "heloc",
        "file": "clf_num-heloc.csv",
        "source": f"{HF}/heloc.csv",
        "openml_original": None,
        "openml_transformed": 45026,
        "description": "Grinsztajn numerical-classification heloc table. Binary RiskPerformance, 22 numeric features.",
    },
    {
        "name": "default-of-credit-card-clients",
        "file": "clf_num-default-of-credit-card-clients.csv",
        "source": f"{HF}/default-of-credit-card-clients.csv",
        "openml_original": 42477,
        "openml_transformed": 45020,
        "description": "Grinsztajn numerical-classification credit-card default table. Binary label, 20 numeric features.",
    },
    {
        "name": "Bioresponse",
        "file": "clf_num-Bioresponse.csv",
        "source": f"{HF}/Bioresponse.csv",
        "openml_original": 4134,
        "openml_transformed": 45019,
        "description": "Grinsztajn numerical-classification Bioresponse table. Binary label, 419 numeric features.",
    },
    {
        "name": "pol",
        "file": "clf_num-pol.csv",
        "source": f"{HF}/pol.csv",
        "openml_original": 722,
        "openml_transformed": 44122,
        "description": (
            "Grinsztajn numerical-classification pol table, OpenML 44122. "
            "Binary label, 26 numeric features. This is not the regression pol file, OpenML 44133."
        ),
    },
    {
        "name": "Diabetes130US",
        "file": "clf_num-Diabetes130US.csv",
        "source": f"{HF}/Diabetes130US.csv",
        "openml_original": 4541,
        "openml_transformed": 45022,
        "description": "Grinsztajn numerical-classification Diabetes130US table. Binary readmitted label, 7 numeric features.",
    },
    {
        "name": "MiniBooNE",
        "file": "clf_num-MiniBooNE.csv",
        "source": f"{HF}/MiniBooNE.csv",
        "openml_original": 41150,
        "openml_transformed": 44128,
        "description": "Grinsztajn numerical-classification MiniBooNE table. Binary signal label, 50 numeric features.",
    },
    {
        "name": "jannis",
        "file": "clf_num-jannis.csv",
        "source": f"{HF}/jannis.csv",
        "openml_original": 41168,
        "openml_transformed": 45021,
        "description": "Grinsztajn numerical-classification jannis table. The released file is binary, 54 numeric features.",
    },
    {
        "name": "phoneme",
        "file": "clf_num-phoneme.csv",
        "source": f"{HF_OLDER}/phoneme.csv",
        "openml_original": None,
        "openml_transformed": None,
        "description": "Grinsztajn numerical-classification phoneme table from the older clf_num snapshot. Binary label, 6 numeric features.",
    },
    {
        "name": "kdd_ipums_la_97-small",
        "file": "clf_num-kdd_ipums_la_97-small.csv",
        "source": f"{HF_OLDER}/kdd_ipums_la_97-small.csv",
        "openml_original": None,
        "openml_transformed": None,
        "description": "Grinsztajn numerical-classification kdd_ipums_la_97-small table from the older clf_num snapshot. Binary label, 21 numeric features.",
    },
    {
        "name": "eeg-eye-state",
        "openml": 1471,
        "source": "https://www.openml.org/d/1471",
        "description": "OpenML eeg-eye-state. Binary eye state, 14 numeric features.",
    },
    {
        "name": "mozilla4",
        "openml": 1046,
        "source": "https://www.openml.org/d/1046",
        "description": "OpenML mozilla4. Binary label, 5 numeric features.",
    },
    {
        "name": "spambase",
        "openml": 44,
        "source": "https://www.openml.org/d/44",
        "description": "OpenML spambase. Binary spam label, 57 numeric features.",
    },
    {
        "name": "twonorm",
        "openml": 1507,
        "source": "https://www.openml.org/d/1507",
        "description": "OpenML twonorm. Binary label, 20 numeric features.",
    },
    {
        "name": "qsar-biodeg",
        "openml": 1494,
        "source": "https://www.openml.org/d/1494",
        "description": "OpenML qsar-biodeg. Binary biodegradation label, 41 numeric features.",
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
    finite_parameters = all(np.isfinite(leaf).all() for leaf in parameter_grads)
    finite_input = bool(np.isfinite(np.asarray(gradient)).all())
    matches = abs(float(loss) - float(likelihood + penalty)) < 1e-5
    penalty_present = float(jnp.mean(jnp.abs(gradient))) > 0.0
    if not (finite_parameters and finite_input and matches and penalty_present and np.isfinite(float(loss))):
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


def _openml_frame(spec: dict) -> pd.DataFrame:
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


def _frame(spec: dict) -> pd.DataFrame:
    if "openml" in spec:
        return _openml_frame(spec)
    return pd.read_csv(DATA_DIR / spec["file"])


def _parts(spec: dict):
    frame = _frame(spec)
    target = frame.columns[-1]
    features = frame.drop(columns=[target])
    numeric = features.apply(pd.to_numeric, errors="coerce")
    if int(numeric.isna().sum().sum()) != 0 or not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise SystemExit(f"{spec['name']} has a non-finite feature")
    if numeric.shape[1] < 1 or frame[target].nunique() != 2:
        raise SystemExit(f"{spec['name']} is not a binary numeric table")
    prepared = numeric.copy()
    prepared[target] = frame[target].to_numpy()
    train, val, test, split_info = benchmark.grinsztajn_medium_split(prepared, np.random.RandomState(0))
    if train[target].nunique() != 2:
        raise SystemExit(f"{spec['name']} train split lost a class")
    split_info["target"] = str(target)
    split_info["source"] = spec["source"]
    split_info["openml_original"] = spec.get("openml_original", spec.get("openml"))
    split_info["openml_transformed"] = spec.get("openml_transformed")
    split_info["description"] = spec["description"]
    split_info["class_counts"] = {str(key): int(value) for key, value in frame[target].value_counts().items()}

    def pack(part):
        return part.drop(columns=[target]).to_numpy(dtype=np.float64), part[target].to_numpy()

    return (split_info, *pack(train), *pack(val), *pack(test))


def _gradient(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    started = time.perf_counter()
    model = ASRGradientSparseNet(random_state=0, verbose=True)
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
            "ASRGradientSparseNet(random_state=0): width 128, SiLU, "
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
