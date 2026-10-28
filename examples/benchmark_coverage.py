"""Current defaults on more public tables, one Grinsztajn medium split.

Residual MLP, the regression ``tabm()`` recipe where it runs, ``classifier()``
or ``ordinal()`` when that target fits, default XGBoost, and TabM-D at the
50-epoch cap. Finished keys are left in place. After the predictor default
changed, no-argument models on the older tables are written to ``predictor``
and ``ordinal_predictor``. A string-categorical table is scored from scratch.
Does not overwrite earlier result files.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_coverage.py --stage ashlar
    /tmp/deeptab-venv/bin/python examples/benchmark_coverage.py --stage tabm
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

DATA_DIR = Path("/tmp/ashlar-bench")
OUT_PATH = DATA_DIR / "coverage-results.json"
SPLIT_DIR = DATA_DIR / "coverage-splits"
HF = "https://huggingface.co/datasets/inria-soda/tabular-benchmark/resolve/main"

# Regression TabM call already used on houses, wine, cpu_act, and sulfur.
TABM_REGRESSION = dict(
    numeric_scaling="quantile_tabr",
    standardize_target=True,
    share_training_batches=False,
    max_epochs=50,
    random_state=0,
)
# Ordinal TabM call already used on wine quality. No regression-only flags.
TABM_ORDINAL = dict(
    task="ordinal",
    prediction_policy="mean_probabilities",
    numeric_scaling="quantile_tabr",
    max_epochs=50,
    random_state=0,
)
TABM_D = dict(
    device="cpu",
    random_state=0,
    n_epochs=50,
    n_cv=1,
    n_refit=0,
    n_repeats=1,
    verbosity=0,
)

DATASETS = (
    {
        "name": "yprop_4_1",
        "task": "regression",
        "ordinal": False,
        "file": "reg_num-yprop_4_1.csv",
        "source": f"{HF}/reg_num/yprop_4_1.csv",
        "openml": 416,
        "openml_transformed": 45032,
        "target": "oz252",
    },
    {
        "name": "Ailerons",
        "task": "regression",
        "ordinal": False,
        "file": "reg_num-Ailerons.csv",
        "source": f"{HF}/reg_num/Ailerons.csv",
        "openml": 296,
        "openml_transformed": 44137,
        "target": "goal",
    },
    {
        "name": "Bike_Sharing_Demand",
        "task": "regression",
        "ordinal": False,
        "file": "reg_num-Bike_Sharing_Demand.csv",
        "source": f"{HF}/reg_num/Bike_Sharing_Demand.csv",
        "openml": 42712,
        "openml_transformed": 44142,
        "target": "count",
    },
    {
        "name": "electricity",
        "task": "classification",
        "ordinal": False,
        "file": "clf_num-electricity.csv",
        "source": f"{HF}/clf_num/electricity.csv",
        "openml": 151,
        "openml_transformed": 44120,
        "target": "class",
    },
    {
        "name": "eye_movements",
        "task": "classification",
        "ordinal": False,
        "file": "clf_num-eye_movements.csv",
        "source": f"{HF}/clf_num/eye_movements.csv",
        "openml": 1044,
        "openml_transformed": 44130,
        "target": "label",
    },
    {
        "name": "pol",
        "task": "regression",
        "ordinal": True,
        "file": "reg_num-pol.csv",
        "source": f"{HF}/reg_num/pol.csv",
        "openml": 201,
        "openml_transformed": 44133,
        "target": "foo",
    },
    {
        "name": "abalone",
        "task": "regression",
        "ordinal": True,
        "file": "reg_cat-abalone.csv",
        "source": f"{HF}/reg_cat/abalone.csv",
        "openml": 42726,
        "openml_transformed": 45042,
        "target": "Classnumberofrings",
    },
    {
        "name": "diamonds",
        "task": "regression",
        "ordinal": False,
        "file": "seaborn-diamonds.csv",
        "source": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv",
        "openml": None,
        "openml_transformed": None,
        "target": "price",
    },
)


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"seed": 0, "datasets": []}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _row(payload: dict, name: str) -> dict:
    for item in payload["datasets"]:
        if item["name"] == name:
            return item
    item = {"name": name}
    payload["datasets"].append(item)
    return item


def _spec(name: str) -> dict:
    for spec in DATASETS:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def _target_summary(values) -> dict:
    series = pd.Series(np.asarray(values).reshape(-1))
    counts = series.value_counts(dropna=False)
    top_value = counts.index[0]
    summary = {
        "nunique": int(series.nunique(dropna=False)),
        "top_value": top_value.item() if hasattr(top_value, "item") else top_value,
        "top_count": int(counts.iloc[0]),
        "top_fraction": float(counts.iloc[0] / len(series)),
    }
    if summary["nunique"] <= 40:
        ordered = sorted(counts.items(), key=lambda item: (str(type(item[0])), item[0]))
        summary["counts"] = [
            {"value": key.item() if hasattr(key, "item") else key, "count": int(count)} for key, count in ordered
        ]
    return summary


def _frame(spec: dict) -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / spec["file"])
    target = spec["target"]
    if target not in frame.columns:
        raise SystemExit(f"{spec['name']} has no target column {target!r}")
    if frame.columns[-1] != target:
        frame = frame[[column for column in frame.columns if column != target] + [target]]
    return frame


def _numeric_feature(series: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series))


def _categorical_columns(frame: pd.DataFrame) -> list[str]:
    return [name for name in frame.columns if not _numeric_feature(frame[name])]


def _cat_arrays(frame: pd.DataFrame, columns: list[str]):
    width = 1
    for name in columns:
        observed = frame.loc[~frame[name].isna(), name].astype(str)
        if len(observed):
            width = max(width, int(observed.str.len().max()))
    text = np.empty((len(frame), len(columns)), dtype=f"U{width}")
    missing = np.zeros((len(frame), len(columns)), dtype=np.bool_)
    for index, name in enumerate(columns):
        series = frame[name]
        mask = series.isna().to_numpy()
        missing[:, index] = mask
        values = series.astype(object).to_numpy()
        for row, (flag, value) in enumerate(zip(mask, values, strict=True)):
            text[row, index] = "" if flag else str(value)
    return text, missing


def _save_split(path: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target: str) -> None:
    features = {
        "train": train.drop(columns=[target]),
        "val": val.drop(columns=[target]),
        "test": test.drop(columns=[target]),
    }
    columns = list(features["train"].columns)
    categorical = _categorical_columns(features["train"])
    payload = {
        "y_train": _label_array(train[target].to_numpy()),
        "y_val": _label_array(val[target].to_numpy()),
        "y_test": _label_array(test[target].to_numpy()),
        "columns": np.array(columns),
    }
    if not categorical:
        payload["x_train"] = features["train"].to_numpy(dtype=np.float64)
        payload["x_val"] = features["val"].to_numpy(dtype=np.float64)
        payload["x_test"] = features["test"].to_numpy(dtype=np.float64)
    else:
        numeric = [name for name in columns if name not in categorical]
        payload["num_columns"] = np.array(numeric)
        payload["cat_columns"] = np.array(categorical)
        for split, frame in features.items():
            payload[f"x_{split}_num"] = (
                frame[numeric].to_numpy(dtype=np.float64) if numeric else np.zeros((len(frame), 0), dtype=np.float64)
            )
            text, missing = _cat_arrays(frame, categorical)
            payload[f"x_{split}_cat"] = text
            payload[f"x_{split}_cat_missing"] = missing
    np.savez(path, **payload)


def _prepare(spec: dict):
    import benchmark

    frame = _frame(spec)
    train, val, test, provenance = benchmark.grinsztajn_medium_split(frame, np.random.RandomState(0))
    target = spec["target"]
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    path = SPLIT_DIR / f"{spec['name']}.npz"
    if not path.exists():
        _save_split(path, train, val, test, target)
    return train, val, test, provenance


def _label_array(values):
    array = np.asarray(values).reshape(-1)
    if array.dtype == object:
        return np.asarray(array, dtype=str)
    return array


def _features_from_split(arrays, split: str) -> pd.DataFrame:
    columns = [str(name) for name in arrays["columns"].tolist()]
    if "x_train" in arrays.files:
        return pd.DataFrame(arrays[f"x_{split}"], columns=columns)
    numeric_names = [str(name) for name in arrays["num_columns"].tolist()]
    categorical_names = [str(name) for name in arrays["cat_columns"].tolist()]
    data = {}
    numeric = np.asarray(arrays[f"x_{split}_num"])
    for index, name in enumerate(numeric_names):
        data[name] = numeric[:, index]
    text = np.asarray(arrays[f"x_{split}_cat"])
    missing = np.asarray(arrays[f"x_{split}_cat_missing"])
    for index, name in enumerate(categorical_names):
        values = text[:, index].astype(object)
        values[missing[:, index]] = None
        data[name] = values
    return pd.DataFrame({name: data[name] for name in columns})


def _load_split(spec: dict):
    arrays = np.load(SPLIT_DIR / f"{spec['name']}.npz", allow_pickle=False)
    return {
        "x_train": _features_from_split(arrays, "train"),
        "x_val": _features_from_split(arrays, "val"),
        "x_test": _features_from_split(arrays, "test"),
        "y_train": arrays["y_train"],
        "y_val": arrays["y_val"],
        "y_test": arrays["y_test"],
    }


def _r2(y_true, y_pred) -> float:
    from sklearn.metrics import r2_score

    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(r2_score(true, pred))


def _rmse(y_true, y_pred) -> float:
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def _accuracy(y_true, y_pred) -> float:
    true = np.asarray(y_true).reshape(-1).tolist()
    pred = np.asarray(y_pred).reshape(-1).tolist()
    if len(true) != len(pred):
        raise RuntimeError(f"accuracy length mismatch {len(true)} vs {len(pred)}")
    return float(np.mean([left == right for left, right in zip(true, pred, strict=True)]))


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


def _history(model) -> dict:
    history = [float(value) for value in model.history_["val_loss"]]
    payload = {
        "epochs_ran": len(history),
        "best_val_loss": min(history) if history else None,
    }
    transform = getattr(model, "numeric_transform_", None)
    quantiles = getattr(transform, "quantiles", None)
    if quantiles is not None and getattr(transform, "scaling", None) in {"quantile", "quantile_tabr"}:
        payload["n_quantiles"] = int(quantiles.shape[0])
    payload["numeric_scaling"] = str(model.numeric_scaling)
    payload["categorical_embeddings"] = None if model.categorical_embeddings is None else [str(name) for name in model.categorical_embeddings]
    embedded = getattr(model, "embedded_columns_", None)
    if embedded is not None:
        payload["embedded_columns"] = [str(name) for name in embedded]
    return payload


def _timed(fit_predict):
    started = time.perf_counter()
    try:
        scores = fit_predict()
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_seconds": time.perf_counter() - started,
        }
    scores["wall_seconds"] = time.perf_counter() - started
    return scores


def _regression_scores(y_true, y_pred) -> dict:
    return {"metric": "r2", "test": _r2(y_true, y_pred), "rmse": _rmse(y_true, y_pred)}


def _ashlar_regression(model, parts) -> dict:
    model.fit(parts["x_train"], parts["y_train"])
    test_pred = model.predict(parts["x_test"])
    val_pred = model.predict(parts["x_val"])
    import jax

    jax.block_until_ready(test_pred)
    scores = _regression_scores(parts["y_test"], test_pred)
    scores["paper_val"] = _r2(parts["y_val"], val_pred)
    scores.update(_history(model))
    return scores


def _ashlar_accuracy(model, parts) -> dict:
    model.fit(parts["x_train"], parts["y_train"])
    test_pred = model.predict(parts["x_test"])
    val_pred = model.predict(parts["x_val"])
    import jax

    jax.block_until_ready(test_pred)
    return {
        "metric": "accuracy",
        "test": _accuracy(parts["y_test"], test_pred),
        "paper_val": _accuracy(parts["y_val"], val_pred),
        **_history(model),
    }


def _ashlar_expected(model, parts) -> dict:
    model.fit(parts["x_train"], parts["y_train"])
    test_expected = model.predict_score(parts["x_test"])
    val_expected = model.predict_score(parts["x_val"])
    test_mode = model.predict(parts["x_test"])
    val_mode = model.predict(parts["x_val"])
    import jax

    jax.block_until_ready(test_expected)
    return {
        "metric": "r2_expected",
        "test": _r2(parts["y_test"], test_expected),
        "paper_val": _r2(parts["y_val"], val_expected),
        "test_mode": _r2(parts["y_test"], test_mode),
        "paper_val_mode": _r2(parts["y_val"], val_mode),
        "classes": [float(value) for value in np.asarray(model.classes_, dtype=np.float64).tolist()],
        **_history(model),
    }


def _expected_from_proba(proba, classes, y_true) -> dict:
    order = np.asarray(classes)
    expected = np.asarray(proba, dtype=np.float64) @ order.astype(np.float64)
    mode = order[np.argmax(np.asarray(proba, dtype=np.float64), axis=1)]
    return {
        "metric": "r2_expected",
        "test": _r2(y_true, expected),
        "test_mode": _r2(y_true, mode),
        "classes": [float(value) for value in order.astype(np.float64).tolist()],
    }


def _baseline_matrices(parts: dict):
    """Train-only one-hot for XGBoost and TabM-D. Numeric frames are unchanged."""
    train = parts["x_train"]
    categorical = _categorical_columns(train)
    if not categorical:
        return parts["x_train"], parts["x_val"], parts["x_test"], False
    train_encoded = pd.get_dummies(train, columns=categorical, dtype=np.float64)

    def align(frame):
        encoded = pd.get_dummies(frame, columns=categorical, dtype=np.float64)
        return encoded.reindex(columns=train_encoded.columns, fill_value=0.0)

    return train_encoded, align(parts["x_val"]), align(parts["x_test"]), True


def stage_ashlar() -> None:
    import xgboost

    from ashlar import classifier, ordinal, tabular, tabm

    payload = _load()
    for spec in DATASETS:
        row = _row(payload, spec["name"])
        had_residual = "residual" in row
        had_ordinal = "ordinal_residual" in row
        train, val, test, provenance = _prepare(spec)
        target = spec["target"]
        parts = {
            "x_train": train.drop(columns=[target]),
            "x_val": val.drop(columns=[target]),
            "x_test": test.drop(columns=[target]),
            "y_train": _label_array(train[target].to_numpy()),
            "y_val": _label_array(val[target].to_numpy()),
            "y_test": _label_array(test[target].to_numpy()),
        }
        row.update(
            {
                "source": spec["source"],
                "openml": spec["openml"],
                "openml_transformed": spec["openml_transformed"],
                "target": target,
                "task": spec["task"],
                "ordinal": bool(spec["ordinal"]),
                "n_rows": int(provenance["n_rows"]),
                "n_train": int(len(train)),
                "n_paper_val": int(len(val)),
                "n_test": int(len(test)),
                "n_features": int(parts["x_train"].shape[1]),
                "target_summary": _target_summary(frame_target(spec)),
                "provenance": provenance,
            }
        )
        _save(payload)
        if spec["task"] == "classification":
            if "residual" not in row:
                print(f"start classifier {spec['name']}", flush=True)
                row["residual"] = _timed(lambda: _ashlar_accuracy(classifier(), parts))
                row["residual"]["config"] = "classifier()"
                print(row["residual"], flush=True)
                _save(payload)
        else:
            if "residual" not in row:
                print(f"start tabular {spec['name']}", flush=True)
                row["residual"] = _timed(lambda: _ashlar_regression(tabular(), parts))
                row["residual"]["config"] = "tabular()"
                print(row["residual"], flush=True)
                _save(payload)
            if "tabm_regression" not in row:
                print(f"start tabm regression {spec['name']}", flush=True)
                row["tabm_regression"] = _timed(lambda: _ashlar_regression(tabm(**TABM_REGRESSION), parts))
                row["tabm_regression"]["config"] = (
                    "tabm(numeric_scaling=quantile_tabr, standardize_target=True, "
                    "share_training_batches=False, max_epochs=50, random_state=0)"
                )
                print(row["tabm_regression"], flush=True)
                _save(payload)
        if spec["ordinal"]:
            if "ordinal_residual" not in row:
                print(f"start ordinal {spec['name']}", flush=True)
                row["ordinal_residual"] = _timed(lambda: _ashlar_expected(ordinal(), parts))
                row["ordinal_residual"]["config"] = "ordinal()"
                print(row["ordinal_residual"], flush=True)
                _save(payload)
            if "ordinal_tabm" not in row:
                print(f"start ordinal tabm {spec['name']}", flush=True)
                row["ordinal_tabm"] = _timed(lambda: _ashlar_expected(tabm(**TABM_ORDINAL), parts))
                row["ordinal_tabm"]["config"] = (
                    "tabm(task=ordinal, prediction_policy=mean_probabilities, "
                    "numeric_scaling=quantile_tabr, max_epochs=50, random_state=0)"
                )
                print(row["ordinal_tabm"], flush=True)
                _save(payload)
        if "xgboost" not in row:
            print(f"start xgboost {spec['name']}", flush=True)
            row["xgboost"] = _timed(lambda: _xgboost(xgboost, spec, parts))
            print(row["xgboost"], flush=True)
            _save(payload)
        if spec["ordinal"] and "xgboost_expected" not in row:
            print(f"start xgboost expected {spec['name']}", flush=True)
            row["xgboost_expected"] = _timed(lambda: _xgboost_expected(xgboost, parts))
            print(row["xgboost_expected"], flush=True)
            _save(payload)
        if had_residual and "predictor" not in row:
            print(f"start predictor default {spec['name']}", flush=True)
            if spec["task"] == "classification":
                row["predictor"] = _timed(lambda: _ashlar_accuracy(classifier(), parts))
                row["predictor"]["config"] = "classifier()"
            else:
                row["predictor"] = _timed(lambda: _ashlar_regression(tabular(), parts))
                row["predictor"]["config"] = "tabular()"
            print(row["predictor"], flush=True)
            _save(payload)
        elif not had_residual and "residual" in row and "predictor" not in row:
            row["predictor"] = json.loads(json.dumps(row["residual"]))
            _save(payload)
        if spec["ordinal"] and had_ordinal and "ordinal_predictor" not in row:
            print(f"start ordinal predictor default {spec['name']}", flush=True)
            row["ordinal_predictor"] = _timed(lambda: _ashlar_expected(ordinal(), parts))
            row["ordinal_predictor"]["config"] = "ordinal()"
            print(row["ordinal_predictor"], flush=True)
            _save(payload)
    _save(payload)


def frame_target(spec: dict):
    return _frame(spec)[spec["target"]].to_numpy()


def _xgboost(xgboost, spec: dict, parts: dict) -> dict:
    x_train, x_val, x_test, encoded = _baseline_matrices(parts)
    host = " on a train-only one-hot of categorical columns; unseen levels are 0" if encoded else ""
    if spec["task"] == "classification":
        classes, transform = _class_index(parts["y_train"])
        booster = xgboost.XGBClassifier()
        booster.fit(x_train, transform(parts["y_train"]))
        test_pred = classes[np.asarray(booster.predict(x_test), dtype=int)]
        val_pred = classes[np.asarray(booster.predict(x_val), dtype=int)]
        return {
            "metric": "accuracy",
            "test": _accuracy(parts["y_test"], test_pred),
            "paper_val": _accuracy(parts["y_val"], val_pred),
            "config": "XGBClassifier() on class ids 0..K-1, predictions mapped back" + host,
        }
    booster = xgboost.XGBRegressor()
    booster.fit(x_train, parts["y_train"])
    test_pred = booster.predict(x_test)
    val_pred = booster.predict(x_val)
    scores = _regression_scores(parts["y_test"], test_pred)
    scores["paper_val"] = _r2(parts["y_val"], val_pred)
    scores["config"] = "XGBRegressor()" + host
    return scores


def _xgboost_expected(xgboost, parts: dict) -> dict:
    x_train, x_val, x_test, encoded = _baseline_matrices(parts)
    classes, transform = _class_index(parts["y_train"])
    booster = xgboost.XGBClassifier()
    booster.fit(x_train, transform(parts["y_train"]))
    test = _expected_from_proba(booster.predict_proba(x_test), classes, parts["y_test"])
    paper = _expected_from_proba(booster.predict_proba(x_val), classes, parts["y_val"])
    test["paper_val"] = paper["test"]
    test["paper_val_mode"] = paper["test_mode"]
    host = " on a train-only one-hot of categorical columns; unseen levels are 0" if encoded else ""
    test["config"] = (
        "XGBClassifier() on class ids 0..K-1, expected score mapped back to the label values" + host
    )
    return test


def stage_tabm() -> None:
    from pytabkit import TabM_D_Classifier, TabM_D_Regressor

    payload = _load()
    for spec in DATASETS:
        row = _row(payload, spec["name"])
        if "n_train" not in row:
            raise SystemExit(f"{spec['name']} has no split metadata. Run --stage ashlar first.")
        parts = _load_split(spec)
        if spec["task"] == "classification":
            if "tabm_d" not in row:
                print(f"start tabm-d classifier {spec['name']}", flush=True)
                row["tabm_d"] = _timed(lambda: _tabm_d_accuracy(TabM_D_Classifier, parts))
                print(row["tabm_d"], flush=True)
                _save(payload)
        else:
            if "tabm_d" not in row:
                print(f"start tabm-d regressor {spec['name']}", flush=True)
                row["tabm_d"] = _timed(lambda: _tabm_d_regression(TabM_D_Regressor, parts))
                print(row["tabm_d"], flush=True)
                _save(payload)
        if spec["ordinal"] and "tabm_d_expected" not in row:
            print(f"start tabm-d expected {spec['name']}", flush=True)
            row["tabm_d_expected"] = _timed(lambda: _tabm_d_expected(TabM_D_Classifier, parts))
            print(row["tabm_d_expected"], flush=True)
            _save(payload)
    _save(payload)


def _tabm_d_regression(factory, parts: dict) -> dict:
    x_train, x_val, x_test, encoded = _baseline_matrices(parts)
    host = " on a train-only one-hot of categorical columns; unseen levels are 0" if encoded else ""
    model = factory(**TABM_D)
    model.fit(x_train, parts["y_train"])
    test_pred = model.predict(x_test)
    val_pred = model.predict(x_val)
    scores = _regression_scores(parts["y_test"], test_pred)
    scores["paper_val"] = _r2(parts["y_val"], val_pred)
    scores["config"] = (
        "TabM_D_Regressor(device=cpu, random_state=0, n_epochs=50, n_cv=1, n_refit=0, n_repeats=1, verbosity=0)"
        + host
    )
    return scores


def _tabm_d_accuracy(factory, parts: dict) -> dict:
    x_train, x_val, x_test, encoded = _baseline_matrices(parts)
    host = " on a train-only one-hot of categorical columns; unseen levels are 0" if encoded else ""
    classes, transform = _class_index(parts["y_train"])
    model = factory(**TABM_D)
    model.fit(x_train, transform(parts["y_train"]))
    test_pred = classes[np.asarray(model.predict(x_test), dtype=int)]
    val_pred = classes[np.asarray(model.predict(x_val), dtype=int)]
    return {
        "metric": "accuracy",
        "test": _accuracy(parts["y_test"], test_pred),
        "paper_val": _accuracy(parts["y_val"], val_pred),
        "config": (
            "TabM_D_Classifier(device=cpu, random_state=0, n_epochs=50, n_cv=1, n_refit=0, n_repeats=1, verbosity=0) "
            "on class ids 0..K-1" + host
        ),
    }


def _tabm_d_expected(factory, parts: dict) -> dict:
    x_train, x_val, x_test, encoded = _baseline_matrices(parts)
    classes, transform = _class_index(parts["y_train"])
    model = factory(**TABM_D)
    model.fit(x_train, transform(parts["y_train"]))
    test = _align_expected(model, x_test, classes, parts["y_test"])
    paper = _align_expected(model, x_val, classes, parts["y_val"])
    test["paper_val"] = paper["test"]
    test["paper_val_mode"] = paper["test_mode"]
    host = " on a train-only one-hot of categorical columns; unseen levels are 0" if encoded else ""
    test["config"] = (
        "TabM_D_Classifier(device=cpu, random_state=0, n_epochs=50, n_cv=1, n_refit=0, n_repeats=1, verbosity=0) "
        "on class ids 0..K-1, expected score mapped back to the label values" + host
    )
    return test


def _align_expected(model, features, classes, y_true) -> dict:
    proba = np.asarray(model.predict_proba(features), dtype=np.float64)
    model_classes = [int(value) for value in np.asarray(model.classes_).tolist()]
    order = [model_classes.index(index) for index in range(len(classes))]
    return _expected_from_proba(proba[:, order], classes, y_true)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("ashlar", "tabm"), required=True)
    args = parser.parse_args()
    if args.stage == "ashlar":
        stage_ashlar()
    else:
        stage_tabm()


if __name__ == "__main__":
    main()
