"""Score learned categorical embeddings on the gradient-penalty classifier.

Numeric matrices keep the existing loss. Categorical columns use one learned
table each: index 0 is unknown, known levels are 1..V in first-seen order, and
the table is initialized at scale 0.02 with width 8. The penalty is the mean
absolute Jacobian of the logits with respect to the concatenation of z-scored
numerics and those embedding vectors. Default ``XGBClassifier`` sees the same
outer split, with train-only one-hot categoricals and unseen levels filled as
0. The split is ``grinsztajn_medium_split`` with ``RandomState(0)``.
Paper-validation is recorded and not used to train or to select. Does not
change the residual regression default and does not refit earlier result files.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_gradient_categorical.py
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
    GRADIENT_EMBED_DIM,
    GRADIENT_EMBED_SCALE,
    GRADIENT_LAMBDA,
    GRADIENT_LEARNING_RATE,
    GRADIENT_MAX_EPOCHS,
    GRADIENT_PATIENCE,
    GRADIENT_WIDTH,
    ASRGradientSparseNet,
    _embedded_features,
    _embedded_loss,
    _embedded_multi_loss,
    _init_embedded,
    _multi_logits,
    _sparse_logits,
)
from ashlar.recipes import DEFAULT_RECIPE

DATA_DIR = Path("/tmp/ashlar-bench")
OUT_PATH = DATA_DIR / "gradient-categorical-results.json"

DATASETS = (
    {
        "name": "credit-g",
        "openml": 31,
        "source": "https://www.openml.org/d/31",
        "description": "OpenML credit-g. Binary credit risk, 7 numeric and 13 categorical columns, no missing values.",
    },
    {
        "name": "cmc",
        "openml": 23,
        "source": "https://www.openml.org/d/23",
        "description": "OpenML cmc. Three contraceptive-method classes, 2 numeric and 7 categorical columns, no missing values.",
    },
    {
        "name": "car",
        "openml": 40975,
        "source": "https://www.openml.org/d/40975",
        "description": "OpenML car. Four acceptability classes, 6 categorical columns, no numeric columns.",
    },
    {
        "name": "mushroom",
        "openml": 24,
        "source": "https://www.openml.org/d/24",
        "description": "OpenML mushroom. Binary edible or poisonous, 22 categorical columns. stalk-root has missing values.",
    },
    {
        "name": "adult",
        "openml": 1590,
        "source": "https://www.openml.org/d/1590",
        "description": "OpenML adult. Binary income, 6 numeric and 8 categorical columns. workclass, occupation, and native-country have missing values.",
    },
)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"datasets": {}}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _numeric_feature(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


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
    vocabs = (("red", "blue"),)
    model = _init_embedded(jax.random.PRNGKey(0), 1 + GRADIENT_EMBED_DIM, vocabs, 2, False)
    numeric = jnp.asarray(np.random.default_rng(0).normal(size=(4, 1)), dtype=jnp.float32)
    indices = jnp.asarray([[1], [2], [0], [1]], dtype=jnp.int32)
    target = jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=jnp.float32)
    loss, grads = eqx.filter_value_and_grad(_embedded_loss)(model, numeric, indices, target)
    parameter_grads = [np.asarray(leaf) for leaf in jax.tree.leaves(grads)]
    features = _embedded_features(model, numeric, indices)
    logit = _sparse_logits(model, features)
    likelihood = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, target))

    def one_logit(row):
        return _sparse_logits(model, row[None, :])[0]

    gradient = jax.vmap(jax.grad(one_logit))(features)
    penalty = GRADIENT_LAMBDA * jnp.mean(jnp.abs(gradient))
    multi = _init_embedded(jax.random.PRNGKey(1), GRADIENT_EMBED_DIM, (("a", "b", "c"),), 3, True)
    multi_index = jnp.asarray([[1], [2], [3], [0]], dtype=jnp.int32)
    multi_target = jnp.asarray([0, 1, 2, 1], dtype=jnp.int32)
    empty = jnp.zeros((4, 0), dtype=jnp.float32)
    multi_loss, multi_grads = eqx.filter_value_and_grad(_embedded_multi_loss)(multi, empty, multi_index, multi_target)
    multi_features = _embedded_features(multi, empty, multi_index)
    multi_logits = _multi_logits(multi, multi_features)
    multi_likelihood = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(multi_logits, multi_target))

    def one_logits(row):
        return _multi_logits(multi, row[None, :])[0]

    jacobian = jax.vmap(jax.jacobian(one_logits))(multi_features)
    multi_penalty = GRADIENT_LAMBDA * jnp.mean(jnp.abs(jacobian))
    finite_parameters = all(np.isfinite(leaf).all() for leaf in parameter_grads)
    finite_embeddings = bool(np.isfinite(np.asarray(grads.embeddings[0])).all())
    embedding_grad = float(jnp.sum(jnp.abs(grads.embeddings[0]))) > 0.0
    finite_input = bool(np.isfinite(np.asarray(gradient)).all())
    matches = abs(float(loss) - float(likelihood + penalty)) < 1e-5
    multi_finite = all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(multi_grads))
    multi_matches = abs(float(multi_loss) - float(multi_likelihood + multi_penalty)) < 1e-5
    if not (
        finite_parameters
        and finite_embeddings
        and embedding_grad
        and finite_input
        and matches
        and multi_finite
        and multi_matches
        and np.isfinite(float(loss))
        and float(jnp.mean(jnp.abs(gradient))) > 0.0
    ):
        raise SystemExit("categorical gradient-penalty loss is not differentiable")
    return {
        "differentiable": True,
        "embedding": (
            "one table per categorical column, shape (1 + n_levels, 8), "
            "normal initialization times 0.02, index 0 unknown, known levels 1..V in first-seen order"
        ),
        "loss": (
            "cross-entropy plus lambda times the mean absolute Jacobian of the logits "
            "with respect to z-scored numerics concatenated with the embedding vectors"
        ),
        "activation": "silu",
        "numeric_path": "unchanged when there is no categorical column",
        "gradient_lambda": float(GRADIENT_LAMBDA),
        "width": int(GRADIENT_WIDTH),
        "embed_dim": int(GRADIENT_EMBED_DIM),
        "embed_scale": float(GRADIENT_EMBED_SCALE),
        "learning_rate": float(GRADIENT_LEARNING_RATE),
        "batch_size": int(GRADIENT_BATCH_SIZE),
        "max_epochs": int(GRADIENT_MAX_EPOCHS),
        "patience": int(GRADIENT_PATIENCE),
        "probe_loss": float(loss),
        "probe_penalty": float(penalty),
        "probe_input_jacobian_finite": True,
        "probe_parameter_gradient_finite": True,
        "probe_embedding_gradient_finite": True,
        "probe_multiclass_loss": float(multi_loss),
        "probe_multiclass_penalty": float(multi_penalty),
    }


def _frame(spec: dict) -> pd.DataFrame:
    cache = DATA_DIR / f"openml-{spec['openml']}-{spec['name']}-categorical.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    bunch = fetch_openml(
        data_id=spec["openml"],
        as_frame=True,
        parser="liac-arff",
        data_home=str(DATA_DIR / "openml-cache"),
    )
    features = bunch.data.copy()
    target_name = str(getattr(bunch.target, "name", None) or "target")
    if target_name in features.columns:
        target_name = "target"
    target = bunch.target.to_numpy() if hasattr(bunch.target, "to_numpy") else np.asarray(bunch.target)
    features[target_name] = np.asarray(target)
    features.to_pickle(cache)
    return features


def _parts(spec: dict):
    frame = _frame(spec)
    target = frame.columns[-1]
    features = frame.drop(columns=[target])
    categorical = [column for column in features.columns if not _numeric_feature(features[column])]
    numeric = [column for column in features.columns if _numeric_feature(features[column])]
    if not categorical:
        raise SystemExit(f"{spec['name']} has no categorical column")
    if numeric:
        nums = features[numeric].apply(pd.to_numeric, errors="coerce")
        if int(nums.isna().sum().sum()) != 0 or not np.isfinite(nums.to_numpy(dtype=np.float64)).all():
            raise SystemExit(f"{spec['name']} has a non-finite numeric column")
    n_classes = int(pd.Series(frame[target]).nunique(dropna=False))
    if n_classes < 2:
        raise SystemExit(f"{spec['name']} has fewer than two classes")
    train, val, test, split_info = benchmark.grinsztajn_medium_split(frame, np.random.RandomState(0))
    if int(pd.Series(train[target]).nunique(dropna=False)) != n_classes:
        raise SystemExit(f"{spec['name']} train split dropped a class")
    split_info["target"] = str(target)
    split_info["source"] = spec["source"]
    split_info["openml"] = spec["openml"]
    split_info["description"] = spec["description"]
    split_info["n_classes"] = n_classes
    split_info["n_numeric"] = len(numeric)
    split_info["n_categorical"] = len(categorical)
    split_info["numeric_columns"] = [str(column) for column in numeric]
    split_info["categorical_columns"] = [str(column) for column in categorical]
    split_info["class_counts"] = {
        str(key): int(value) for key, value in pd.Series(frame[target]).value_counts(dropna=False).items()
    }
    split_info["categorical_missing"] = {str(column): int(features[column].isna().sum()) for column in categorical}

    def pack(part):
        return part.drop(columns=[target]), part[target].to_numpy()

    return (split_info, *pack(train), *pack(val), *pack(test))


def _gradient(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    started = time.perf_counter()
    model = ASRGradientSparseNet(random_state=0, verbose=True)
    model.fit(x_train, y_train, "classification")
    if not model.categorical_:
        raise SystemExit("categorical columns were fit as numeric")
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
        "embed_dim": int(model.embed_dim_),
        "vocab_sizes": [int(len(vocab)) for vocab in model.categories_],
        "input_width": int(model.model_.w1.shape[0]),
        "embedding_shapes": [list(np.asarray(table).shape) for table in model.model_.embeddings],
        "wall_seconds": time.perf_counter() - started,
        "config": (
            "ASRGradientSparseNet(random_state=0): width 128, SiLU, embedding dim 8, "
            "init scale 0.02, index 0 unknown, known levels 1..V in first-seen order, "
            "z-scored numerics concatenated with embeddings, cross-entropy plus 0.01 "
            "times the mean absolute Jacobian of the logits with respect to that vector, "
            "Adam 1e-3, batch 256, patience 16, cap 80"
        ),
    }


def _one_hot(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    categorical = [column for column in train.columns if not _numeric_feature(train[column])]
    encoded = pd.get_dummies(train, columns=categorical, dtype=np.float64)

    def align(frame: pd.DataFrame) -> np.ndarray:
        other = pd.get_dummies(frame, columns=categorical, dtype=np.float64)
        return other.reindex(columns=encoded.columns, fill_value=0.0).to_numpy(dtype=np.float64)

    return encoded.to_numpy(dtype=np.float64), align(val), align(test)


def _xgboost(x_train, y_train, x_val, y_val, x_test, y_test) -> dict:
    classes, transform = _class_index(y_train)
    train_x, val_x, test_x = _one_hot(x_train, x_val, x_test)
    started = time.perf_counter()
    booster = xgboost.XGBClassifier()
    booster.fit(train_x, transform(y_train))
    test_pred = classes[np.asarray(booster.predict(test_x), dtype=int)]
    val_pred = classes[np.asarray(booster.predict(val_x), dtype=int)]
    return {
        "test_accuracy": float(accuracy_score(y_test, test_pred)),
        "paper_val_accuracy": float(accuracy_score(y_val, val_pred)),
        "n_columns": int(train_x.shape[1]),
        "wall_seconds": time.perf_counter() - started,
        "config": (
            "XGBClassifier() with no arguments. Categorical columns use train-only one-hot; "
            "unseen and missing levels are 0. Numeric columns are not standardized. "
            "Labels are mapped to 0..K-1 and predictions are mapped back."
        ),
    }


def main() -> None:
    if DEFAULT_RECIPE["task"] != "regression" or DEFAULT_RECIPE["standardize_target"] is not True:
        raise SystemExit("residual regression default changed")
    if DEFAULT_RECIPE["categorical_embedding_dim"] != 8:
        raise SystemExit("residual embedding width changed")
    if GRADIENT_LAMBDA != 1e-2 or GRADIENT_LEARNING_RATE != 1e-3 or GRADIENT_WIDTH != 128:
        raise SystemExit("gradient-penalty defaults changed")
    if GRADIENT_MAX_EPOCHS != 80 or GRADIENT_PATIENCE != 16 or GRADIENT_BATCH_SIZE != 256:
        raise SystemExit("gradient-penalty schedule changed")
    if GRADIENT_EMBED_DIM != 8 or GRADIENT_EMBED_SCALE != 0.02:
        raise SystemExit("gradient embedding defaults changed")
    payload = _load()
    if "probe" not in payload:
        payload["probe"] = _probe()
        payload["residual_default"] = {
            "task": DEFAULT_RECIPE["task"],
            "standardize_target": DEFAULT_RECIPE["standardize_target"],
            "backbone": DEFAULT_RECIPE["backbone"],
            "categorical_embedding_dim": DEFAULT_RECIPE["categorical_embedding_dim"],
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
            f"numeric {split_info['n_numeric']} categorical {split_info['n_categorical']} "
            f"classes {split_info['n_classes']}",
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
