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
    # What the 256-step integral misses, as a share of how far each target moved from the baseline row.
    out["gap"] = {column: _stats([r["gap"][column]["missed"] / r["gap"][column]["moved"] for r in results])
                  for column in results[0]["gap"]}
    return out


def compute():
    CACHE.mkdir(parents=True, exist_ok=True)
    _run(DECIDE, "examples/benchmark_decide.py", "sweep", "--seeds", "0", "1", "2", "--out", str(DECIDE))
    _run(EXPLAIN, "examples/benchmark_explain.py", "--seeds", "0", "1", "2", "--out", str(EXPLAIN))
    k.save_json(FIGURE, "sweep", _sweep())
    k.save_json(FIGURE, "attributions", _attributions())


# ---------------------------------------------------------------- draw

NAMES = {
    "hr": "hour", "workingday": "working day", "yr": "year", "weathersit": "weather", "temp": "temperature",
    "season": "season", "weekday": "weekday", "mnth": "month", "hum": "humidity", "atemp": "feels-like",
    "windspeed": "wind", "holiday": "holiday",
}
# (group, column, colour): LightGBM with SHAP on the left of each pair, Ashlar's explain on the right
MATRIX = (
    ("mean", "lightgbm_mean", k.CLAY),
    ("mean", "ashlar_mean", k.TEAL),
    ("80% width", "lightgbm_width", k.CLAY),
    ("80% width", "ashlar_width", k.TEAL),
    ("P(rentals > 500)", "lightgbm_tail", k.CLAY),
    ("P(rentals > 500)", "ashlar_tail", k.TEAL),
)
RATIO_TICKS = ((1, "1:1", "0.5"), (1.5, "1.5:1", "0.6"), (7 / 3, "2.3:1", "0.7"), (4, "4:1", "0.8"),
               (9, "9:1", "0.9"), (19, "19:1", "0.95"))


def _best(data):
    """LightGBM's better setting at each ratio, as a percentage above Ashlar's cost."""
    return np.minimum([c["mean"] for c in data["methods"]["lightgbm"]["pct"]],
                      [c["mean"] for c in data["methods"]["lightgbm_early"]["pct"]])


def _shares(data, column):
    values = np.asarray(data["columns"][column]["mean_abs"], dtype=float)
    return values / values.sum()


def _riser(data):
    """The feature whose share of Ashlar's attribution grows most from the mean to the 80% width."""
    mean, width = _shares(data, "ashlar_mean"), _shares(data, "ashlar_width")
    i = int(np.argmax(width - mean))
    return data["features"][i], mean[i], width[i]


def _sweep_panel(ax, data):
    x = np.log(np.asarray(data["levels"]) / (1 - np.asarray(data["levels"])))
    ends = {}
    for method, color, dash in (("lightgbm", k.SLATE, (0, (5, 3))), ("lightgbm_early", k.CLAY, "-")):
        cells = data["methods"][method]["pct"]
        mean = np.array([c["mean"] for c in cells])
        lo, hi = np.array([c["min"] for c in cells]), np.array([c["max"] for c in cells])
        ax.fill_between(x, lo, hi, color=k.mix(color, k.CARD, 0.82), lw=0, zorder=1)
        ax.plot(x, mean, color=color, lw=2.4, linestyle=dash, zorder=3)
        ax.scatter(x, mean, s=26, color=color, edgecolor=k.CARD, lw=0.9, zorder=4)
        ends[method] = mean
    left = x[0] - 0.42
    ax.plot([left, x[-1]], [0, 0], color=k.TEAL, lw=3.0, solid_capstyle="butt", zorder=2)
    k.code(ax, x[0] - 0.2, -1.6, "Ashlar: one fit, predict_order(X, under, over)", k.TEAL, ha="left", va="top")
    for method, color, label in (("lightgbm_early", k.CLAY, "early-stopped"), ("lightgbm", k.SLATE, "400 trees")):
        end = ends[method][-1]
        k.rich(ax, x[-1] + 0.1, end, [(k.signed(end, 0), color, k.F.value), ("  " + label, color, k.F.label)],
               transform=ax.transData, va="center")
        start = ends[method][0]
        k.value(ax, x[0], start + 1.4, k.signed(start, 1), color, ha="center", va="bottom")
    ax.set_xlim(left, x[-1] + 1.45)
    ax.set_ylim(-6.5, 35)
    ax.set_xticks([np.log(r) for r, _, _ in RATIO_TICKS], [f"{a}\n{b}" for _, a, b in RATIO_TICKS])
    ax.set_yticks([0, 10, 20, 30], ["0%", "+10%", "+20%", "+30%"])
    ax.spines["bottom"].set_bounds(left, x[-1] + 0.12)
    k.axis_names(ax, x="cost of a unit short to a unit over, and the critical ratio",
                 y="LightGBM's cost above Ashlar's, refitted at each ratio")
    k.panel_title(ax, "The further into the tail, the more a refit costs")


def _matrix_panel(ax, data):
    features = data["features"]
    order = sorted(range(len(features)), key=lambda i: -data["columns"]["ashlar_mean"]["mean_abs"][i])
    n = len(order)
    xs = np.array([0.0, 1.0, 2.7, 3.7, 5.4, 6.4])
    ax.set_xlim(-0.55, 6.95)
    ax.set_ylim(-0.75, n + 0.9)
    fig = ax.figure
    box = ax.get_position()
    pitch = box.height * fig.get_figheight() / (n + 1.65) * 72
    unit = box.width * fig.get_figwidth() / 7.5 * 72
    shares = {column: _shares(data, column) for _, column, _ in MATRIX}
    largest = max(float(s.max()) for s in shares.values())
    diameter = min(1.18 * pitch, 0.98 * unit)
    riser, _, _ = _riser(data)
    r = order.index(features.index(riser))
    ax.axhspan(n - 1 - r - 0.5, n - 1 - r + 0.5, color=k.TEAL_WASH, lw=0, zorder=0)
    for row, i in enumerate(order):
        y = n - 1 - row
        ax.axhline(y, color=k.GRID, lw=0.8, zorder=1)
        ax.text(-0.75, y, NAMES.get(features[i], features[i]), fontproperties=k.F.label, color=k.INK, ha="right",
                va="center", clip_on=False)
    for j, (_, column, color) in enumerate(MATRIX):
        share = shares[column][order]
        ys = n - 1 - np.arange(n)
        ax.scatter(np.full(n, xs[j]), ys, s=diameter ** 2 * share / largest, color=color, alpha=0.92,
                   edgecolor=k.CARD, lw=0.8, zorder=3)
    for column, x in (("ashlar_mean", xs[1]), ("ashlar_width", xs[3])):
        share = shares[column][features.index(riser)]
        radius = diameter * np.sqrt(share / largest) / 2 / unit
        k.value(ax, x + radius + 0.12, n - 1 - r, k.pct(share), k.TEAL, size=12.5, va="center", zorder=5)
    for g, group in enumerate(("mean", "80% width", "P(rentals > 500)")):
        ax.text(xs[2 * g] + 0.5, n + 0.35, group, fontproperties=k.F.label, color=k.INK, ha="center",
                va="bottom")
    ax.text(xs[0] + 0.3, n - 0.35, "LightGBM", fontproperties=k.F.small, color=k.CLAY, ha="right", va="bottom")
    ax.text(xs[1] - 0.3, n - 0.35, "Ashlar", fontproperties=k.F.small, color=k.TEAL, ha="left", va="bottom")
    ax.set_xticks([])
    ax.set_yticks([])


def draw():
    sweep = k.load_json(FIGURE, "sweep")
    attributions = k.load_json(FIGURE, "attributions")
    best = _best(sweep)
    riser, mean_share, width_share = _riser(attributions)
    fig = k.card(
        f"One fit, every cost ratio: LightGBM refitted per ratio costs up to {best.max():.0f}% more.",
        f"Hourly bike rentals, {sweep['splits']} splits. Left: the newsvendor cost of each order on held-out hours. "
        f"Right: mean absolute attribution on {k.num(attributions['rows'])} held-out hours, circle area the share "
        "of each column; Ashlar's explain beside SHAP of LightGBM's mean, of its 0.9 minus 0.1 quantile models, "
        "and of a classifier.",
        height=8.9, footer=2,
    )
    _sweep_panel(k.panel(fig, (0.0, 0.0, 0.5, 1.0), lpad=0.8, rpad=0.25, tail=1.05, head=0.8), sweep)
    right = k.panel(fig, (0.55, 0.0, 1.0, 1.0), lpad=1.3, rpad=0.05, tail=0.15, head=0.8, frame="")
    _matrix_panel(right, attributions)
    k.panel_title(right, f"{NAMES.get(riser, riser).capitalize()}: {k.pct(width_share)} of the width, "
                         f"{k.pct(mean_share)} of the mean")
    crossed = attributions["crossed"]["mean"]
    k.footnote(fig, "LightGBM needs a model per question: two quantile models for the width (the 0.9 model fell "
                    f"below the 0.1 model on {100 * crossed:.2f}% of hours) and a classifier for each threshold. "
                    "Decision-aware training was tried and did not pay (issue #32).", line=0)
    k.save(fig, "act")
    _thumb(sweep)


def _thumb(sweep):
    x = np.log(np.asarray(sweep["levels"]) / (1 - np.asarray(sweep["levels"])))
    best = _best(sweep)

    def mark(ax):
        ax.fill_between(x, 0, best, color=k.CLAY_PALE, lw=0, zorder=1)
        ax.plot(x, best, color=k.CLAY, lw=3.0, zorder=3)
        ax.plot([x[0], x[-1]], [0, 0], color=k.TEAL, lw=4.0, solid_capstyle="butt", zorder=4)
        ax.set_xlim(x[0] - 0.05, x[-1] + 0.05)
        ax.set_ylim(-2, best.max() * 1.08)

    k.thumb("act", "Any cost ratio", [("LightGBM refits cost up to ", k.MUTED), (f"{best.max():.0f}% more", k.CLAY)],
            mark)


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
