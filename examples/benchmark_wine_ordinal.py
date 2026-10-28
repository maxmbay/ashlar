"""Wine quality as an ordinal classifier on the existing medium split.

Scores the expected label with test R². Does not refit the regression runs.
Does not change the residual-MLP default.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_wine_ordinal.py
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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from ashlar import ordinal, tabm

BIN_SOURCES = ("alcohol", "volatile.acidity", "sulphates")
N_BINS = 8

SPLIT_PATH = Path("/tmp/ashlar-bench/extra-regression-splits/wine_quality.npz")
OUT_PATH = Path("/tmp/ashlar-bench/wine-ordinal-results.json")


def _r2(y_true, y_pred) -> float:
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(r2_score(true, pred))


def _frame(values, columns):
    return pd.DataFrame(values, columns=columns)


def _fit_ordinal(factory, x_train, y_train, x_test, y_test):
    model = factory()
    started = time.perf_counter()
    model.fit(x_train, y_train)
    score = model.predict_score(x_test)
    labels = model.predict(x_test)
    elapsed = time.perf_counter() - started
    history = [float(value) for value in model.history_["val_loss"]]
    row = {
        "test_r2_expected": _r2(y_test, score),
        "test_r2_mode": _r2(y_test, labels),
        "epochs_ran": len(history),
        "best_val_loss": min(history) if history else None,
        "wall_seconds": elapsed,
        "classes": [float(value) for value in np.asarray(model.classes_).tolist()],
        "task": model.task,
        "backbone": model.backbone if isinstance(model.backbone, str) else model.backbone.__name__,
        "numeric_scaling": model.numeric_scaling,
        "learning_rate": model.learning_rate,
        "weight_decay": model.weight_decay,
    }
    if "chosen_learning_rate" in model.history_:
        row["chosen_learning_rate"] = float(model.history_["chosen_learning_rate"])
        row["surface_curvature"] = float(model.history_["surface_curvature"])
        row["surface_grad_rms"] = float(model.history_["surface_grad_rms"])
        row["surface_n_train"] = int(model.history_["surface_n_train"])
        row["surface_warmup_steps"] = int(model.history_["surface_warmup_steps"])
        row["surface_probe_rate"] = float(model.history_["surface_probe_rate"])
    prior = getattr(model, "class_prior_", None)
    if prior is not None:
        row["class_prior"] = [float(value) for value in np.asarray(prior, dtype=np.float64).tolist()]
    if model.categorical_embeddings:
        row["categorical_embeddings"] = [str(name) for name in model.categorical_embeddings]
    projection = getattr(model.core_, "bottleneck", None)
    if projection is not None:
        row["bottleneck_width"] = int(projection.width)
    if model.histogram_variance is not None:
        row["histogram_variance"] = float(model.histogram_variance)
        row["head"] = model.head if isinstance(model.head, str) else model.head.__name__
    if model.expectation_weight is not None:
        row["expectation_weight"] = float(model.expectation_weight)
        row["loss"] = "ordinal negative log-likelihood + expectation_weight * (probabilities @ class_values - y)^2"
    histogram = getattr(model.core_, "soft_histogram", None)
    if histogram is not None:
        row["soft_stages"] = int(histogram.n_stages)
        row["soft_shrinkage"] = float(histogram.shrinkage)
    return row


def _bin_labels(values, edges):
    index = np.digitize(np.asarray(values, dtype=np.float64), np.asarray(edges, dtype=np.float64), right=False)
    return np.array([f"b{int(item)}" for item in index], dtype=object)


def _with_quantile_bins(x_train, y_train, x_test):
    """Eight train-only quantile bins of the three strongest linear chemicals.

    Edges come from the same internal training rows ``fit`` will train on.
    The paper-validation rows are not in ``x_train``. The test rows do not
    set the edges. Wine has no stored categorical column.
    """
    inner, _held = train_test_split(
        np.arange(len(x_train)),
        test_size=0.2,
        random_state=0,
        stratify=np.asarray(y_train).reshape(-1),
    )
    train = x_train.copy()
    test = x_test.copy()
    edges = {}
    names = []
    for column in BIN_SOURCES:
        source = np.asarray(x_train.iloc[inner][column], dtype=np.float64)
        cuts = np.unique(np.quantile(source, np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]))
        name = f"{column}_bin"
        train[name] = _bin_labels(x_train[column], cuts)
        test[name] = _bin_labels(x_test[column], cuts)
        edges[column] = [float(value) for value in cuts.tolist()]
        names.append(name)
    return train, test, names, edges


def _load_rows() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text()).get("rows", {})
    return {}


def _save(rows: dict) -> None:
    OUT_PATH.write_text(json.dumps({"split": str(SPLIT_PATH), "rows": rows}, indent=2))


def main():
    arrays = np.load(SPLIT_PATH, allow_pickle=False)
    columns = [str(name) for name in arrays["columns"].tolist()]
    x_train = _frame(arrays["x_train"], columns)
    x_test = _frame(arrays["x_test"], columns)
    y_train = arrays["y_train"]
    y_test = arrays["y_test"]
    rows = _load_rows()

    if "ordinal_residual" in rows:
        print("skip ordinal residual", flush=True)
    else:
        print("start ordinal residual", flush=True)
        rows["ordinal_residual"] = _fit_ordinal(
            lambda: ordinal(max_epochs=50, random_state=0),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_residual"], flush=True)
        _save(rows)

    if "ordinal_tabm" in rows:
        print("skip ordinal tabm", flush=True)
    else:
        print("start ordinal tabm", flush=True)
        rows["ordinal_tabm"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_tabm"], flush=True)
        _save(rows)

    if "ordinal_tabm_weight_decay" in rows:
        print("skip ordinal tabm weight decay", flush=True)
    else:
        print("start ordinal tabm weight decay", flush=True)
        rows["ordinal_tabm_weight_decay"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
                weight_decay=1.0,
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_tabm_weight_decay"], flush=True)
        _save(rows)

    if "ordinal_tabm_surface_lr" in rows:
        print("skip ordinal tabm surface lr", flush=True)
    else:
        print("start ordinal tabm surface lr", flush=True)
        rows["ordinal_tabm_surface_lr"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
                learning_rate_schedule="surface",
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_tabm_surface_lr"], flush=True)
        _save(rows)

    if "ordinal_tabm_prior_embed" in rows:
        print("skip ordinal tabm prior embed", flush=True)
    else:
        print("start ordinal tabm prior embed", flush=True)
        x_train_bins, x_test_bins, embedded, edges = _with_quantile_bins(x_train, y_train, x_test)
        row = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
                head_kwargs={"class_prior": "empirical"},
                categorical_embeddings=embedded,
            ),
            x_train_bins,
            y_train,
            x_test_bins,
            y_test,
        )
        row["bin_edges"] = edges
        row["bin_count"] = N_BINS
        rows["ordinal_tabm_prior_embed"] = row
        print(rows["ordinal_tabm_prior_embed"], flush=True)
        _save(rows)

    if "ordinal_tabm_prior_embed_patience0" in rows:
        print("skip ordinal tabm prior embed patience 0", flush=True)
    else:
        print("start ordinal tabm prior embed patience 0", flush=True)
        x_train_bins, x_test_bins, embedded, edges = _with_quantile_bins(x_train, y_train, x_test)

        def _run(max_epochs: int) -> dict:
            model = tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=max_epochs,
                patience=0,
                random_state=0,
                head_kwargs={"class_prior": "empirical"},
                categorical_embeddings=list(embedded),
            )
            started = time.perf_counter()
            model.fit(x_train_bins, y_train)
            score = model.predict_score(x_test_bins)
            labels = model.predict(x_test_bins)
            val_loss = [float(value) for value in model.history_["val_loss"]]
            best_epoch = int(np.argmin(val_loss)) + 1 if val_loss else None
            prior = getattr(model, "class_prior_", None)
            print(
                f"patience0 epochs={len(val_loss)} best_epoch={best_epoch} "
                f"val_tail={val_loss[-5:]} r2={_r2(y_test, score):.6f}",
                flush=True,
            )
            return {
                "test_r2_expected": _r2(y_test, score),
                "test_r2_mode": _r2(y_test, labels),
                "epochs_ran": len(val_loss),
                "best_epoch": best_epoch,
                "best_val_loss": min(val_loss) if val_loss else None,
                "val_loss": val_loss,
                "wall_seconds": time.perf_counter() - started,
                "classes": [float(value) for value in np.asarray(model.classes_).tolist()],
                "task": model.task,
                "backbone": model.backbone if isinstance(model.backbone, str) else model.backbone.__name__,
                "numeric_scaling": model.numeric_scaling,
                "learning_rate": model.learning_rate,
                "weight_decay": model.weight_decay,
                "patience": 0,
                "max_epochs": max_epochs,
                "class_prior": None if prior is None else [float(value) for value in np.asarray(prior).tolist()],
                "categorical_embeddings": [str(name) for name in embedded],
                "bin_edges": edges,
                "bin_count": N_BINS,
                "checkpoint": "lowest validation loss; patience 0 trains the full budget",
            }

        first = _run(50)
        # Still falling at epoch 50 when that loss is below the loss at epoch 40.
        falling = len(first["val_loss"]) >= 50 and first["val_loss"][49] < first["val_loss"][39]
        print(
            f"val epoch40={first['val_loss'][39] if len(first['val_loss']) >= 50 else None} "
            f"epoch50={first['val_loss'][49] if len(first['val_loss']) >= 50 else None} "
            f"falling={falling}",
            flush=True,
        )
        if falling:
            print("validation loss still falling at 50; continuing to 100", flush=True)
            full = _run(100)
            full["continued_to_100"] = True
            full["val_loss_epoch_40"] = first["val_loss"][39]
            full["val_loss_epoch_50"] = first["val_loss"][49]
            full["test_r2_expected_at_50"] = first["test_r2_expected"]
            full["first_50_val_loss_matches"] = bool(
                len(full["val_loss"]) >= 50 and np.allclose(full["val_loss"][:50], first["val_loss"])
            )
            rows["ordinal_tabm_prior_embed_patience0"] = full
        else:
            first["continued_to_100"] = False
            rows["ordinal_tabm_prior_embed_patience0"] = first
        print(rows["ordinal_tabm_prior_embed_patience0"]["test_r2_expected"], flush=True)
        _save(rows)

    if "ordinal_tabm_softmax" in rows:
        print("skip ordinal tabm softmax", flush=True)
    else:
        print("start ordinal tabm softmax", flush=True)
        rows["ordinal_tabm_softmax"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                head="softmax",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_tabm_softmax"], flush=True)
        _save(rows)

    if "ordinal_tabm_bottleneck" in rows:
        print("skip ordinal tabm bottleneck", flush=True)
    else:
        print("start ordinal tabm bottleneck", flush=True)
        rows["ordinal_tabm_bottleneck"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
                bottleneck=True,
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_tabm_bottleneck"], flush=True)
        _save(rows)

    if "ordinal_tabm_histogram" in rows:
        print("skip ordinal tabm histogram", flush=True)
    else:
        print("start ordinal tabm histogram", flush=True)
        rows["ordinal_tabm_histogram"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                head="softmax",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
                histogram_variance=0.5,
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_tabm_histogram"], flush=True)
        _save(rows)

    if "ordinal_tabm_soft_stages" in rows:
        print("skip ordinal tabm soft stages", flush=True)
    else:
        print("start ordinal tabm soft stages", flush=True)
        rows["ordinal_tabm_soft_stages"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
                soft_stages=8,
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        print(rows["ordinal_tabm_soft_stages"], flush=True)
        _save(rows)

    if "ordinal_tabm_expectation" in rows:
        print("skip ordinal tabm expectation", flush=True)
    else:
        print("start ordinal tabm expectation", flush=True)
        rows["ordinal_tabm_expectation"] = _fit_ordinal(
            lambda: tabm(
                task="ordinal",
                prediction_policy="mean_probabilities",
                numeric_scaling="quantile_tabr",
                max_epochs=50,
                random_state=0,
                expectation_weight=1.0,
            ),
            x_train,
            y_train,
            x_test,
            y_test,
        )
        rows["ordinal_tabm_expectation"]["config"] = (
            "tabm(task=ordinal, prediction_policy=mean_probabilities, "
            "numeric_scaling=quantile_tabr, max_epochs=50, random_state=0, expectation_weight=1)"
        )
        print(rows["ordinal_tabm_expectation"], flush=True)
        _save(rows)

    if "xgboost_multiclass_expected" in rows:
        print("skip xgboost multiclass", flush=True)
        print(OUT_PATH, flush=True)
        return
    # This XGBoost build expects class ids 0..K-1. Map back to the quality values.
    classes = np.unique(np.asarray(y_train))
    index_of = {value: index for index, value in enumerate(classes.tolist())}
    y_index = np.array([index_of[value] for value in np.asarray(y_train).tolist()], dtype=np.int32)
    booster = xgboost.XGBClassifier()
    started = time.perf_counter()
    booster.fit(x_train, y_index)
    probabilities = np.asarray(booster.predict_proba(x_test), dtype=np.float64)
    expected = probabilities @ classes.astype(np.float64)
    mode = classes[np.asarray(booster.predict(x_test), dtype=int)]
    rows["xgboost_multiclass_expected"] = {
        "test_r2_expected": _r2(y_test, expected),
        "test_r2_mode": _r2(y_test, mode),
        "wall_seconds": time.perf_counter() - started,
        "classes": [float(value) for value in classes.tolist()],
        "config": "XGBClassifier() defaults on class ids 0..K-1, expected score mapped back to quality",
    }
    print(rows["xgboost_multiclass_expected"], flush=True)
    _save(rows)
    print(OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
