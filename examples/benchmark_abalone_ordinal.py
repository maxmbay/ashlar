"""Score ``ASROrdinalClassifier()`` on the existing abalone split.

Classes with fewer than two outer-training rows stay in the training
partition. Does not refit the residual regressor, XGBoost, or TabM-D.
Writes ``/tmp/ashlar-bench/abalone-ordinal-results.json``.

    JAX_PLATFORMS=cpu /tmp/gnet-venv/bin/python examples/benchmark_abalone_ordinal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

import benchmark_coverage as coverage
from ashlar import ASROrdinalClassifier

OUT_PATH = Path("/tmp/ashlar-bench/abalone-ordinal-results.json")


def _rare_levels(labels) -> list:
    values, counts = np.unique(np.asarray(labels).reshape(-1), return_counts=True)
    rare = values[counts < 2]
    return [value.item() if hasattr(value, "item") else value for value in rare]


def main() -> None:
    if OUT_PATH.exists():
        payload = json.loads(OUT_PATH.read_text())
        if "ordinal" in payload:
            print("skip abalone ordinal", payload["ordinal"].get("test"), flush=True)
            return
    else:
        payload = {}
    spec = coverage._spec("abalone")
    parts = coverage._load_split(spec)
    rare = _rare_levels(parts["y_train"])
    print("rare outer-train levels", rare, flush=True)
    model = ASROrdinalClassifier()
    if model.task != "ordinal" or model.backbone != "residual_mlp" or model.expectation_weight is not None:
        raise SystemExit("ASROrdinalClassifier() is no longer the residual ordinal recipe")
    scores = coverage._timed(lambda: coverage._ashlar_expected(model, parts))
    scores["config"] = "ASROrdinalClassifier()"
    scores["rare_train_levels"] = rare
    scores["rule"] = (
        "a class with fewer than two rows stays in the training partition; "
        "validation_size is applied to the remaining rows"
    )
    payload = {
        "name": "abalone",
        "n_train": int(len(parts["y_train"])),
        "n_paper_val": int(len(parts["y_val"])),
        "n_test": int(len(parts["y_test"])),
        "ordinal": scores,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(scores, flush=True)


if __name__ == "__main__":
    main()
