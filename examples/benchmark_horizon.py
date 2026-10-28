"""Rescore ``tabular()`` after regression labels are z-scored by default.

Same coverage splits and the cached houses split. Does not refit
classification, ordinal, XGBoost, or TabM. Writes
``/tmp/ashlar-bench/horizon-results.json``.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_horizon.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

import benchmark_coverage as coverage
from ashlar import tabular

OUT_PATH = Path("/tmp/ashlar-bench/horizon-results.json")
HOUSES_SPLIT = Path("/tmp/ashlar-bench/split-houses.csv")


def _load() -> dict:
    if OUT_PATH.exists():
        return json.loads(OUT_PATH.read_text())
    return {"standardize_target": True, "datasets": []}


def _save(payload: dict) -> None:
    OUT_PATH.write_text(json.dumps(payload, indent=2))


def _row(payload: dict, name: str) -> dict:
    for item in payload["datasets"]:
        if item["name"] == name:
            return item
    item = {"name": name}
    payload["datasets"].append(item)
    return item


def _score_frame(
    name: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    *,
    standardize_target: bool = True,
) -> dict:
    parts = {
        "x_train": train.drop(columns=[target]),
        "x_val": val.drop(columns=[target]),
        "x_test": test.drop(columns=[target]),
        "y_train": coverage._label_array(train[target].to_numpy()),
        "y_val": coverage._label_array(val[target].to_numpy()),
        "y_test": coverage._label_array(test[target].to_numpy()),
    }
    model = tabular(standardize_target=standardize_target)
    if model.standardize_target is not standardize_target:
        raise SystemExit("tabular() did not keep the requested label scale")
    scores = coverage._timed(lambda: coverage._ashlar_regression(model, parts))
    scores["config"] = "tabular()" if standardize_target else 'tabular(standardize_target=False)'
    fitted = model if model.standardize_target else None
    if fitted is not None and getattr(fitted, "target_std_", None) is not None:
        scores["target_mean"] = float(fitted.target_mean_)
        scores["target_std"] = float(fitted.target_std_)
    return scores


def main() -> None:
    payload = _load()
    for spec in coverage.DATASETS:
        if spec["task"] != "regression":
            continue
        row = _row(payload, spec["name"])
        if "unit_target" in row:
            continue
        print(f"start {spec['name']}", flush=True)
        train, val, test, provenance = coverage._prepare(spec)
        target = spec["target"]
        row.update(
            {
                "n_train": int(len(train)),
                "n_paper_val": int(len(val)),
                "n_test": int(len(test)),
                "n_rows": int(provenance["n_rows"]),
                "provenance": provenance,
            }
        )
        row["unit_target"] = _score_frame(spec["name"], train, val, test, target)
        print(row["unit_target"], flush=True)
        _save(payload)

    houses = _row(payload, "houses")
    if "unit_target" not in houses:
        print("start houses", flush=True)
        frame = pd.read_csv(HOUSES_SPLIT)
        target = "medianhousevalue"
        train = frame.loc[frame["_partition"] == "train"].drop(columns=["_partition"])
        val = frame.loc[frame["_partition"] == "val"].drop(columns=["_partition"])
        test = frame.loc[frame["_partition"] == "test"].drop(columns=["_partition"])
        houses.update({"n_train": int(len(train)), "n_paper_val": int(len(val)), "n_test": int(len(test))})
        houses["unit_target"] = _score_frame("houses", train, val, test, target)
        print(houses["unit_target"], flush=True)
        _save(payload)
    if "robust_only" not in houses:
        print("start houses robust only", flush=True)
        frame = pd.read_csv(HOUSES_SPLIT)
        target = "medianhousevalue"
        train = frame.loc[frame["_partition"] == "train"].drop(columns=["_partition"])
        val = frame.loc[frame["_partition"] == "val"].drop(columns=["_partition"])
        test = frame.loc[frame["_partition"] == "test"].drop(columns=["_partition"])
        houses["robust_only"] = _score_frame("houses", train, val, test, target, standardize_target=False)
        print(houses["robust_only"], flush=True)
        _save(payload)
    _save(payload)


if __name__ == "__main__":
    main()
