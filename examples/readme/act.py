"""Act on the distribution: an order at every cost ratio from one fit, and what drives the spread.

``compute`` runs the evidence (each stage is skipped when its results file
exists) and writes the small artifacts the figure reads:

- ``benchmark_decide.py sweep``: on hourly bike rentals, the newsvendor cost
  of one mixture fit's order read at eight critical ratios from 0.5 to 0.95,
  against LightGBM's quantile regression refitted at each ratio, three splits.
- ``benchmark_explain.py``: mean absolute attributions per feature, Ashlar's
  ``explain`` of the mean, median, 80% width, and P(rentals > 500) against
  SHAP of LightGBM's mean, of its 0.9 minus 0.1 quantile models, and of a
  classifier of rentals > 500, three splits.

``draw`` renders ``act.png`` and the ``act`` thumbnail from the artifacts alone.

    python examples/readme/act.py compute
    python examples/readme/act.py draw
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXAMPLES))

import style as k  # noqa: E402

FIGURE = "act"
CACHE = Path("/tmp/ashlar-bench/readme")
DECIDE = CACHE / "decide.json"
EXPLAIN = CACHE / "explain.json"


def _run(path: Path, *command: str):
    if path.exists():
        print(f"using {path}")
        return
    print("running", " ".join(command), flush=True)
    subprocess.run([sys.executable, *command], check=True, cwd=EXAMPLES.parent)


def _stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sd": float(values.std()), "min": float(values.min()),
            "max": float(values.max()), "runs": values.tolist()}


# ---------------------------------------------------------------- compute


def _sweep():
    rows = json.loads(DECIDE.read_text())["sweep"]
    levels = sorted({row["level"] for row in rows})
    out = {"levels": levels, "splits": len({row["seed"] for row in rows}), "methods": {}}
    for method in ("lightgbm", "lightgbm_early"):
        out["methods"][method] = {
            "pct": [_stats([row["pct_vs_ashlar"][method] for row in rows if row["level"] == level]) for level in levels],
            "service": [_stats([row["service"][method] for row in rows if row["level"] == level]) for level in levels],
        }
    out["methods"]["ashlar"] = {
        "service": [_stats([row["service"]["ashlar"] for row in rows if row["level"] == level]) for level in levels],
        "cost": [_stats([row["cost"]["ashlar"] for row in rows if row["level"] == level]) for level in levels],
    }
    return out


def _attributions():
    from scipy import stats

    results = json.loads(EXPLAIN.read_text())["shap"]
    columns = list(results[0]["importance"])
    features = list(results[0]["importance"]["ashlar_mean"])
    out = {"features": features, "splits": len(results), "rows": results[0]["rows"], "columns": {}}
    for column in columns:
        per_seed = np.array([[r["importance"][column][f] for f in features] for r in results])
        ranks = np.array([stats.rankdata(-values) for values in per_seed])
        out["columns"][column] = {
            "mean_abs": per_seed.mean(axis=0),
            "sd": per_seed.std(axis=0),
            "rank": ranks.mean(axis=0),
            "prediction": float(np.mean([r["prediction"][column] for r in results])),
        }
    out["crossed"] = _stats([r["crossed"]["test_rows"] for r in results])
    return out


def compute():
    CACHE.mkdir(parents=True, exist_ok=True)
    _run(DECIDE, "examples/benchmark_decide.py", "sweep", "--seeds", "0", "1", "2", "--out", str(DECIDE))
    _run(EXPLAIN, "examples/benchmark_explain.py", "--seeds", "0", "1", "2", "--out", str(EXPLAIN))
    k.save_json(FIGURE, "sweep", _sweep())
    k.save_json(FIGURE, "attributions", _attributions())


# ---------------------------------------------------------------- draw


def draw():
    raise SystemExit("draw is written once the style kit is on dev")


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
