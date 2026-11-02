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
    for method in ("raw", "pooled", "grouped", "cqr_raw", "cqr_pooled", "cqr_scaled", "cqr_grouped"):
        per_hour = [_stats([run[method][h]["coverage"] for run in runs]) for h in hours]
        means = np.array([cell["mean"] for cell in per_hour])
        out["methods"][method] = {
            "coverage": per_hour,
            "width": [float(np.mean([run[method][h]["width"] for run in runs])) for h in hours],
            "outside": int(np.sum((means < BAND[0]) | (means > BAND[1]))),
            "rms": float(np.sqrt(np.mean([(run[method][h]["coverage"] - 0.8) ** 2 for run in runs for h in hours]))),
            "furthest_per_split": _stats([max(abs(run[method][h]["coverage"] - 0.8) for h in hours) for run in runs]),
        }
    for method, key in (("pooled", "pooled_all"), ("grouped", "grouped_all"), ("cqr_pooled", "cqr_pooled_all"),
                        ("cqr_scaled", "cqr_scaled_all"), ("cqr_grouped", "cqr_grouped_all")):
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
        if name not in tables:
            print(f"no calibration results for {name} yet")
            continue
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
    cells = {name: (None if isinstance(value, float) and np.isnan(value) else value) for name, value in row.iloc[0].items()}
    out = {"row": cells, "y": float(y_test[index]), "index": index, "grid": grid, "blanked": list(BLANK_ORDER), "steps": []}
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

MISSING_LINES = (
    # method, colour, line style, label
    ("median", k.SLATE, (0, (5, 3)), "median fill"),
    ("lightgbm", k.CLAY, "-", "LightGBM given NaN"),
    ("lightgbm_masked", k.CLAY, (0, (1, 2.2)), "LightGBM trained with blanks"),
    ("draws", k.TEAL, "-", "with_uncertain_inputs(32)"),
)


def _band(ax, lo=BAND[0], hi=BAND[1], label: bool = False):
    ax.axhspan(lo, hi, color=k.TEAL_WASH, lw=0, zorder=0)
    if label:
        ax.text(0.97, lo + 0.006, "75% to 85%", fontproperties=k.F.small, color=k.TEAL, ha="right", va="bottom",
                transform=ax.get_yaxis_transform(), zorder=1)


def _end_labels(ax, ends, x, gap):
    """Value labels at the right end of each line, nudged apart so none overlap: ``ends`` is [(y, text, colour)]."""
    ends = sorted(ends)
    placed = []
    for y, text, color in ends:
        y_text = y if not placed or y - placed[-1] >= gap else placed[-1] + gap
        placed.append(y_text)
        k.value(ax, x, y_text, text, color, va="center", clip_on=False)


def _coverage_panel(ax, cells, title, band_label=False):
    ends = []
    for method, color, dash, _label in MISSING_LINES:
        rates = sorted(cells[method], key=float)
        x = [100 * float(r) for r in rates]
        y = [cells[method][r]["coverage"]["mean"] for r in rates]
        lw = 3.2 if method == "draws" else 2.3
        z = 6 if method == "draws" else 4
        ax.plot(x, y, color=color, lw=lw, linestyle=dash, zorder=z)
        ax.scatter(x, y, s=40 if method == "draws" else 24, color=color, edgecolor=k.CARD, lw=1.0, zorder=z + 1)
        ends.append((y[-1], k.pct(y[-1]), color))
    _end_labels(ax, ends, 52.5, 0.032)
    _band(ax, label=band_label)
    ax.set_xlim(-2, 50)
    ax.set_ylim(0.3, 0.92)
    ax.set_xticks([0, 10, 30, 50], ["0", "10", "30", "50%"])
    ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ["40%", "50%", "60%", "70%", "80%", "90%"])
    k.panel_title(ax, title)
    k.axis_names(ax, x="numeric cells blank", y="inside the 80% interval")


def _k_dollars(width):
    """$100k units as text; the dollar sign is escaped so matplotlib does not read a pair as math."""
    return f"\\${100 * width:.0f}k"


def _one_row_panel(ax, data):
    grid = np.asarray(data["grid"])
    steps = data["steps"][::-1]  # the most blanks at the bottom, in front
    dens = [np.asarray(step["draws"]["density"]) for step in steps]
    inner = [tuple(step["draws"]["interval"]) for step in steps]
    bases = k.ridgeline(ax, grid, dens, inner=inner, gap=1.55, height=1.45, scale="shared")
    names = ["all cells known", "income blank", "+ latitude", "+ longitude"][::-1]
    for base, step, name in zip(bases, steps, names):
        lo, hi = step["median"]["interval"]
        ax.plot([lo, hi], [base - 0.24] * 2, color=k.SLATE, lw=3.6, solid_capstyle="butt", zorder=200)
        ax.text(-0.04, base + 0.05, name, fontproperties=k.F.label, color=k.INK, ha="right", va="bottom",
                transform=ax.get_yaxis_transform())
        dlo, dhi = step["draws"]["interval"]
        ax.text(1.03, base + 0.16, _k_dollars(dhi - dlo), fontproperties=k.F.value, color=k.TEAL, ha="left",
                va="center", transform=ax.get_yaxis_transform())
        ax.text(1.03, base - 0.24, _k_dollars(hi - lo), fontproperties=k.F.value, color=k.SLATE, ha="left",
                va="center", transform=ax.get_yaxis_transform())
    top = bases[-1] + 1.6
    ax.text(1.03, bases[-1] + 0.62, "80% width", fontproperties=k.F.small, color=k.MUTED, ha="left",
            va="bottom", transform=ax.get_yaxis_transform())
    ax.plot([data["y"]] * 2, [-0.5, top + 0.35], color=k.INK, lw=1.4, zorder=300)
    ax.text(data["y"] - 0.07, top + 0.35, f"what happened, {_k_dollars(data['y'])}", fontproperties=k.F.label,
            color=k.INK, ha="right", va="top")
    ax.set_xlim(0, 4.5)
    ax.set_ylim(-0.6, top + 0.45)
    k.style(ax, frame="b")
    ax.set_yticks([])
    ax.set_xticks([0, 1, 2, 3, 4], ["\\$0", "\\$100k", "\\$200k", "\\$300k", "\\$400k"])
    k.panel_title(ax, "One row as its cells go blank")
    k.axis_names(ax, x="median house value")


def draw_missing():
    missing = k.load_json(FIGURE, "missing")
    row = k.load_json(FIGURE, "one_row")
    cal, bike = missing["california"], missing["bike"]
    draws30 = cal["draws"]["0.3"]["coverage"]["mean"]
    fig = k.card(
        f"With 30% of the cells blank, the 80% interval still holds {k.pct(draws30)}.",
        "California housing and hourly bike rentals, mean of three splits. Fitted and calibrated on complete rows; "
        "at test time numeric cells are blanked at random. LightGBM: quantile models with CQR on the same rows.",
        height=8.4, footer=3,
    )
    common = {"lpad": 0.72, "rpad": 0.62, "head": 0.62}
    _coverage_panel(k.panel(fig, (0.0, 0.0, 0.25, 1.0), **common), cal, "California housing", band_label=True)
    _coverage_panel(k.panel(fig, (0.275, 0.0, 0.525, 1.0), **common), bike, "Hourly bike rentals")
    _one_row_panel(k.panel(fig, (0.565, 0.0, 1.0, 1.0), lpad=1.5, rpad=0.95, head=0.62), row)
    k.legend_row(fig, [(("line" if dash == "-" else "dashed" if dash[1][0] == 5 else "dotted"), color, label)
                       for _m, color, dash, label in MISSING_LINES[::-1]], line=2)
    masked, draws = cal["lightgbm_masked"]["0.3"]["width"]["mean"], cal["draws"]["0.3"]["width"]["mean"]
    k.footnote(fig, f"LightGBM trained and calibrated with the same share of its cells blank holds 80% and, on "
                    f"California at 30%, is narrower ({_k_dollars(masked)} against {_k_dollars(draws)}), but it needs "
                    f"the share in advance and a fit per share. Right: one California row; teal its 80% interval, "
                    f"slate the median fill's.", line=0)
    k.save(fig, "intervals-missing")


STRIPS = (
    # row title, Ashlar's method, the baselines' methods with colour
    ("Raw", "raw", (("cqr_raw", k.CLAY),)),
    ("One calibration of every row", "pooled", (("cqr_pooled", k.CLAY), ("cqr_scaled", k.SLATE))),
    ("A table per hour", "grouped", (("cqr_grouped", k.CLAY),)),
)
WIDTH_TABLES = (("california", "California housing"), ("diamonds_price", "Diamond prices"),
                ("bike_count", "Bike rentals"), ("abalone", "Abalone rings"))


def _strip(ax, data, title, ours, theirs, *, last: bool):
    hours = np.asarray(data["hours"])
    methods = data["methods"]
    _band(ax)
    ax.axhline(0.8, color=k.TEAL, lw=0.8, alpha=0.5, zorder=1)
    series = [(ours, k.TEAL)] + list(theirs)
    offsets = np.linspace(-0.24, 0.24, len(series)) if len(series) > 1 else [0.0]
    for (method, color), offset in zip(series, offsets):
        cells = methods[method]["coverage"]
        mean = np.array([c["mean"] for c in cells])
        lo, hi = np.array([c["min"] for c in cells]), np.array([c["max"] for c in cells])
        x = hours + offset
        ax.vlines(x, lo, hi, color=k.mix(color, k.CARD, 0.55), lw=1.5, zorder=3)
        ax.scatter(x, mean, s=30, color=color, edgecolor=k.CARD, lw=0.8, zorder=4)
    ax.set_xlim(-0.7, 23.7)
    ax.set_ylim(0.52, 1.0)
    ax.set_yticks([0.6, 0.8, 1.0], ["60%", "80%", "100%"])
    if last:
        ax.set_xticks([0, 6, 12, 18, 23], ["0:00", "6:00", "12:00", "18:00", "23:00"])
        k.axis_names(ax, x="hour of the day")
    else:
        ax.set_xticks([])
    ax.text(0.0, 1.0, title, transform=k._above(ax, 0.06), fontproperties=k.F.label, color=k.INK, ha="left",
            va="bottom")
    # Hours outside 75% to 85% (mean over the splits) and the mean width over all rows, in the right margin.
    for i, (method, color) in enumerate(series):
        width = methods[method].get("all", {}).get("width")
        text = f"{methods[method]['outside']} off" + (f" · {width['mean']:.0f} wide" if width else "")
        ax.text(1.03, 0.84 - 0.29 * i, text, transform=ax.transAxes, fontproperties=k.F.value, color=color,
                ha="left", va="center")


def _widths_panel(ax, data):
    tables = [t for t in WIDTH_TABLES if t[0] in data]
    n = len(tables)
    for i, (table, name) in enumerate(tables):
        y = n - 1 - i
        cell = data[table]
        ratio = cell["width_ratio"]
        ax.plot([1.0, 1.0], [y - 0.2, y + 0.2], color=k.CLAY, lw=3.0, solid_capstyle="butt", zorder=2)
        ax.hlines(y, ratio["min"], ratio["max"], color=k.mix(k.TEAL, k.CARD, 0.5), lw=2.0, zorder=3)
        ax.scatter([ratio["mean"]], [y], s=90, color=k.TEAL, edgecolor=k.CARD, lw=1.0, zorder=4)
        side = 1 if ratio["mean"] >= 1.0 else -1
        k.value(ax, ratio["max" if side > 0 else "min"] + 0.012 * side, y, k.signed(100 * (ratio["mean"] - 1), 0),
                k.TEAL, ha="left" if side > 0 else "right", va="center")
        ours, theirs = cell["mixture_cal_0.8"], cell["lightgbm_cal_0.8"]
        ax.text(0.765, y + 0.46, name, fontproperties=k.F.label, color=k.INK, ha="left", va="bottom")
        ax.text(0.765, y + 0.42, f"coverage {k.pct(ours['coverage']['mean'], 1)}, CQR {k.pct(theirs['coverage']['mean'], 1)}",
                fontproperties=k.F.small, color=k.MUTED, ha="left", va="top")
    ax.set_ylim(-0.6, n - 0.05)
    ax.set_yticks([])
    ax.set_xlim(0.76, 1.12)
    ax.set_xticks([0.8, 0.9, 1.0, 1.1], ["80%", "90%", "100%", "110%"])
    ax.text(1.0, -0.3, "CQR", fontproperties=k.F.label, color=k.CLAY, ha="center", va="top")
    k.axis_names(ax, x="Ashlar's 80% width, CQR's = 100%")


def draw_hours():
    data = k.load_json(FIGURE, "by_hour")
    widths = k.load_json(FIGURE, "widths")
    m = data["methods"]
    narrower = sum(widths[t]["width_ratio"]["mean"] < 1.0 for t, _ in WIDTH_TABLES if t in widths)
    compared = sum(t in widths for t, _ in WIDTH_TABLES)
    fig = k.card(
        f"One calibration leaves {m['pooled']['outside']} of 24 hours outside 75% to 85%; CQR leaves "
        f"{m['cqr_pooled']['outside']}.",
        f"Left: hourly bike rentals, coverage of the 80% interval within each hour of the day (about "
        f"{int(np.mean(data['test_rows'])):d} held-out rows an hour), mean of {data['splits']} splits, whiskers "
        "their range; \"off\" counts the hours outside 75% to 85%, widths in rentals. Right: the width after "
        "calibration on four tables against LightGBM CQR's, at the coverage under each name.",
        height=9.2, footer=1,
    )
    n = len(STRIPS)
    box = fig.box
    extra = 0.65 / (box.height - box.top - box.bottom)  # the last strip's room for hour ticks and the axis name
    h = (1.0 - extra) / n
    for i, (title, ours, theirs) in enumerate(STRIPS):
        y1 = 1.0 - i * h
        y0 = 0.0 if i == n - 1 else 1.0 - (i + 1) * h
        ax = k.panel(fig, (0.0, y0, 0.67, y1), head=0.38, tail=0.75 if i == n - 1 else 0.1, lpad=0.75, rpad=1.75)
        _strip(ax, data, title, ours, theirs, last=i == n - 1)
    right = k.panel(fig, (0.79, 0.0, 1.0, 1.0), head=0.38, tail=0.75, lpad=0.0, rpad=0.1, frame="b")
    _widths_panel(right, widths)
    k.panel_title(right, f"Narrower than CQR on {narrower} of {compared}", raise_by=0.06)
    k.legend_row(fig, [("dot", k.TEAL, "Ashlar"), ("dot", k.CLAY, "LightGBM quantile models: raw, CQR, CQR per hour"),
                       ("dot", k.SLATE, "CQR, margin scaled by width"), ("patch", k.TEAL_WASH, "75% to 85%")],
                 line=0)
    k.save(fig, "intervals")


def draw_thumb():
    data = k.load_json(FIGURE, "by_hour")
    hours = np.asarray(data["hours"])
    methods = data["methods"]

    def mark(ax):
        ax.axhspan(*BAND, color=k.TEAL_WASH, lw=0, zorder=0)
        for method, color, offset in (("cqr_pooled", k.CLAY, 0.2), ("pooled", k.TEAL, -0.2)):
            mean = [c["mean"] for c in methods[method]["coverage"]]
            ax.scatter(hours + offset, mean, s=46, color=color, edgecolor=k.CARD, lw=0.9, zorder=3)
        ax.set_xlim(-0.8, 23.8)
        ax.set_ylim(0.66, 1.0)

    inside = 24 - methods["pooled"]["outside"]
    k.thumb("intervals", "Intervals that hold", [(f"{inside} of 24", k.TEAL), (" hours on target, against ", k.MUTED),
                                                 (f"{24 - methods['cqr_pooled']['outside']}", k.CLAY)], mark)


def draw():
    draw_hours()
    draw_missing()
    draw_thumb()


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
