"""Intervals that hold: within every hour of the day, and when cells go blank.

``compute`` runs the evidence (each stage is skipped when its results file
exists) and writes the small artifacts the figures read:

- ``benchmark_uncertainty.py --stage groups --tables bike_count``: coverage of
  the 80% interval within each hour of the day on hourly bike rentals, for
  Ashlar's mixture (raw, one calibration, a table per hour) and for LightGBM's
  0.1 and 0.9 quantile models (raw, CQR with one margin, CQR with a margin per
  hour), three splits.
- ``benchmark_uncertainty.py --stage calibration``: coverage and width after
  calibration on five tables, Ashlar against CQR, three splits.
- ``benchmark_inputs.py mask``: coverage, width, and pinball as 0 to 50% of
  the test rows' numeric cells go blank, three splits.
- One California test row (split 0), its predictive density as 0, 1, 2, and 3
  of its cells go blank, under ``with_uncertain_inputs(32)`` and the median fill.

``draw`` renders ``intervals.png``, ``intervals-missing.png``, and the
``intervals`` thumbnail from the artifacts alone.

    python examples/readme/intervals.py compute
    python examples/readme/intervals.py draw
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

FIGURE = "intervals"
BENCH = Path("/tmp/ashlar-bench")
CACHE = BENCH / "readme"
GROUPS = BENCH / "uq-groups-bike_count.json"
CALIBRATION_TABLES = ("california", "diamonds_price", "bike_count", "abalone", "taxi_tip")
CALIBRATION = BENCH / f"uq-calibration-{'-'.join(CALIBRATION_TABLES)}.json"
INPUTS = CACHE / "inputs.json"
BAND = (0.75, 0.85)
# The California row's cells, blanked one more at a time in this order.
BLANK_ORDER = ("median_income", "latitude", "longitude")


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


def _by_hour():
    table = json.loads(GROUPS.read_text())["groups"]["bike_count"]
    runs = [table["runs"][seed] for seed in sorted(table["runs"])]
    hours = sorted({key for run in runs for key in run["pooled"]}, key=int)
    out = {"hours": [int(h) for h in hours], "splits": len(runs), "band": BAND, "methods": {}}
    for method in ("raw", "pooled", "grouped", "cqr_raw", "cqr_pooled", "cqr_grouped"):
        per_hour = [_stats([run[method][h]["coverage"] for run in runs]) for h in hours]
        means = np.array([cell["mean"] for cell in per_hour])
        out["methods"][method] = {
            "coverage": per_hour,
            "width": [float(np.mean([run[method][h]["width"] for run in runs])) for h in hours],
            "outside": int(np.sum((means < BAND[0]) | (means > BAND[1]))),
            "rms": float(np.sqrt(np.mean([(run[method][h]["coverage"] - 0.8) ** 2 for run in runs for h in hours]))),
            "furthest_per_split": _stats([max(abs(run[method][h]["coverage"] - 0.8) for h in hours) for run in runs]),
        }
    for method, key in (("pooled", "pooled_all"), ("grouped", "grouped_all"),
                        ("cqr_pooled", "cqr_pooled_all"), ("cqr_grouped", "cqr_grouped_all")):
        out["methods"][method]["all"] = {
            "coverage": _stats([run[key]["coverage"] for run in runs]),
            "width": _stats([run[key]["width"] for run in runs]),
        }
    out["test_rows"] = [int(np.mean([run["pooled"][h]["rows"] for run in runs])) for h in hours]
    return out


def _widths():
    tables = json.loads(CALIBRATION.read_text())["calibration"]
    out = {}
    for name in CALIBRATION_TABLES:
        runs = [tables[name]["runs"][seed] for seed in sorted(tables[name]["runs"])]
        out[name] = {"rows": tables[name]["rows"], "splits": len(runs)}
        for method in ("mixture_raw_0.8", "mixture_cal_0.8", "quantile_cal_0.8", "lightgbm_raw_0.8", "lightgbm_cal_0.8"):
            out[name][method] = {
                "coverage": _stats([run[method]["coverage"] for run in runs]),
                "width": _stats([run[method]["width"] for run in runs]),
            }
        ratio = [run["mixture_cal_0.8"]["width"] / run["lightgbm_cal_0.8"]["width"] for run in runs]
        out[name]["width_ratio"] = _stats(ratio)
    return out


def _missing():
    rows = json.loads(INPUTS.read_text())["mask"]
    out = {}
    for table in ("california", "bike"):
        out[table] = {}
        for method in ("median", "draws", "lightgbm", "lightgbm_masked"):
            cells = {}
            for rate in sorted({row["rate"] for row in rows}):
                group = [row for row in rows if row["table"] == table and row["method"] == method and row["rate"] == rate]
                if not group:
                    continue
                cell = {metric: _stats([row[metric] for row in group]) for metric in ("coverage", "width", "pinball")}
                cell["by_blanks"] = {
                    count: _stats([row["by_blanks"][count]["coverage"] for row in group if count in row["by_blanks"]])
                    for count in ("0", "1", "2", "3")
                    if any(count in row["by_blanks"] for row in group)
                }
                cells[f"{rate:g}"] = cell
            out[table][method] = cells
    return out


def _one_row():
    """One California test row's density as 0 to 3 of its cells go blank, split 0, the mask stage's model."""
    from benchmark_inputs import DRAWS, _fit_split

    model, (_, _, X_test, _, _, y_test), _ = _fit_split("california", 0)
    complete = np.flatnonzero(X_test.notna().all(axis=1).to_numpy())
    # The complete test row whose value is nearest the median of the test values, the first such row on a tie.
    index = int(complete[np.argmin(np.abs(y_test[complete] - np.median(y_test)))])
    row = X_test.iloc[[index]]
    grid = np.linspace(0.0, 5.2, 521)
    views = {"draws": model.with_uncertain_inputs(DRAWS, seed=0), "median": model}
    out = {"row": {name: (None if value != value else value) for name, value in row.iloc[0].items()},
           "y": float(y_test[index]), "index": index, "grid": grid, "blanked": list(BLANK_ORDER), "steps": []}
    for count in range(len(BLANK_ORDER) + 1):
        blank = row.copy()
        blank[list(BLANK_ORDER[:count])] = np.nan
        step = {"blank": count}
        for name, view in views.items():
            dist = view.predict_dist(blank)
            density = np.exp(np.asarray(dist.log_prob(grid[:, None]), dtype=np.float64)[:, 0])
            lower, upper = view.predict_interval(blank, 0.8)[0]
            step[name] = {"density": density, "interval": [float(lower), float(upper)],
                          "median": float(view.predict_quantiles(blank, [0.5])[0, 0])}
        out["steps"].append(step)
        print(count, {name: step[name]["interval"] for name in views}, flush=True)
    return out


def compute():
    CACHE.mkdir(parents=True, exist_ok=True)
    _run(GROUPS, "examples/benchmark_uncertainty.py", "--stage", "groups", "--tables", "bike_count")
    _run(CALIBRATION, "examples/benchmark_uncertainty.py", "--stage", "calibration", "--tables", *CALIBRATION_TABLES)
    _run(INPUTS, "examples/benchmark_inputs.py", "mask", "--seeds", "0", "1", "2", "--out", str(INPUTS))
    k.save_json(FIGURE, "by_hour", _by_hour())
    k.save_json(FIGURE, "widths", _widths())
    k.save_json(FIGURE, "missing", _missing())
    k.save_json(FIGURE, "one_row", _one_row(), digits=5)


# ---------------------------------------------------------------- draw


def draw():
    raise SystemExit("draw is written once the style kit is on dev")


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
