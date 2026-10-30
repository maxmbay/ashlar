"""Score ``tabm(task="classification")`` on the classification splits.

Shared minibatches, no label z-score, robust numeric scale. The 50-epoch
cap matches the other neural rows. Does not refit XGBoost or TabM-D.
Writes ``/tmp/ashlar-bench/tabm-classification-results.json``.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_tabm_classification.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import benchmark_coverage as coverage
from ashlar import ASRClassifier, tabm

OUT_PATH = Path("/tmp/ashlar-bench/tabm-classification-results.json")
CALIFORNIA_SPLIT = Path("/tmp/ashlar-bench/split-california.csv")
CONFIG = 'tabm(task="classification", max_epochs=50, random_state=0)'


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"config": CONFIG, "datasets": []}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _row(payload: dict, name: str) -> dict:
    for item in payload["datasets"]:
        if item["name"] == name:
            return item
    item = {"name": name}
    payload["datasets"].append(item)
    return item


def _legal(model) -> None:
    if model.task != "classification":
        raise SystemExit("tabm classification task was not classification")
    if model.standardize_target:
        raise SystemExit("tabm classification turned on the label z-score")
    if model.share_training_batches is not True:
        raise SystemExit("tabm classification required unshared batches")
    if model.prediction_policy != "mean_probabilities":
        raise SystemExit("tabm classification did not use mean_probabilities")
    if model.numeric_scaling != "robust":
        raise SystemExit("tabm classification turned on a quantile map")


def _parts(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    return {
        "x_train": train.drop(columns=[target]),
        "x_val": val.drop(columns=[target]),
        "x_test": test.drop(columns=[target]),
        "y_train": coverage._label_array(train[target].to_numpy()),
        "y_val": coverage._label_array(val[target].to_numpy()),
        "y_test": coverage._label_array(test[target].to_numpy()),
    }


def _score_tabm(parts: dict) -> dict:
    model = tabm(task="classification", max_epochs=50, random_state=0)
    _legal(model)
    scores = coverage._timed(lambda: coverage._ashlar_accuracy(model, parts))
    scores["config"] = CONFIG
    scores["standardize_target"] = bool(model.standardize_target)
    scores["share_training_batches"] = bool(model.share_training_batches)
    scores["prediction_policy"] = str(model.prediction_policy)
    return scores


def _score_classifier(parts: dict) -> dict:
    model = ASRClassifier()
    scores = coverage._timed(lambda: coverage._ashlar_accuracy(model, parts))
    scores["config"] = "ASRClassifier()"
    return scores


def _coverage(payload: dict, name: str) -> None:
    row = _row(payload, name)
    if "tabm_classification" in row:
        return
    spec = coverage._spec(name)
    print(f"start {name}", flush=True)
    train, val, test, provenance = coverage._prepare(spec)
    target = spec["target"]
    row.update(
        {
            "n_train": int(len(train)),
            "n_paper_val": int(len(val)),
            "n_test": int(len(test)),
            "n_rows": int(provenance["n_rows"]),
            "n_features": int(train.shape[1] - 1),
            "provenance": provenance,
        }
    )
    row["tabm_classification"] = _score_tabm(_parts(train, val, test, target))
    print(row["tabm_classification"], flush=True)
    _save(payload)


def _california(payload: dict) -> None:
    row = _row(payload, "california")
    frame = pd.read_csv(CALIFORNIA_SPLIT)
    target = "price_above_median"
    train = frame.loc[frame["_partition"] == "train"].drop(columns=["_partition"])
    val = frame.loc[frame["_partition"] == "val"].drop(columns=["_partition"])
    test = frame.loc[frame["_partition"] == "test"].drop(columns=["_partition"])
    parts = _parts(train, val, test, target)
    row.update(
        {
            "n_train": int(len(train)),
            "n_paper_val": int(len(val)),
            "n_test": int(len(test)),
            "n_features": int(parts["x_train"].shape[1]),
            "split_cache": str(CALIFORNIA_SPLIT),
        }
    )
    if "classifier" not in row:
        print("start california classifier", flush=True)
        row["classifier"] = _score_classifier(parts)
        print(row["classifier"], flush=True)
        _save(payload)
    if "tabm_classification" not in row:
        print("start california tabm", flush=True)
        row["tabm_classification"] = _score_tabm(parts)
        print(row["tabm_classification"], flush=True)
        _save(payload)


def main() -> None:
    payload = _load()
    for name in ("electricity", "eye_movements"):
        _coverage(payload, name)
    _california(payload)
    _save(payload)


if __name__ == "__main__":
    main()
