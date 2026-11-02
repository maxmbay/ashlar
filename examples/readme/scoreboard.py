"""The scoreboard: Ashlar against the best gradient-boosted trees on the eight large tables.

Reads the large-table benchmark's results (``examples/benchmark_large.py``, shipped from the Mac to
``examples/readme/artifacts/large/large.jsonl``) and draws ``docs/images/readme/scoreboard.png``:

- Left: Ashlar's default against the best of LightGBM, XGBoost, and CatBoost on each table, as the
  change in test error (RMSE, or 1 − accuracy for the classification tables).
- Right: the predicted distribution against the best tree's. On the regression tables,
  ``ASRRegressor(task="mixture")``'s CRPS against the best tree distribution: LightGBM, XGBoost, and
  CatBoost with a normal of one width, CatBoost ``RMSEWithUncertainty``, LightGBMLSS, and NGBoost.
  On the classification tables, the default's log loss against the trees'.

The best tree is the one with the lowest mean over the seeds; each seed's change is against it, the
dot is the mean change and the whisker one standard deviation across seeds. A dot is teal or clay
when the whole whisker is on one side of zero, and grey when it crosses zero (or, with one split,
when the change is under half a percent).

    python examples/readme/scoreboard.py draw
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import style as k

RESULTS = k.ARTIFACTS / "large" / "large.jsonl"
VARIANT = {"ashlar": "v3", "ashlar_mixture": "v3"}  # the recorded Ashlar runs
TREES = ("lightgbm", "xgboost", "catboost")
TREE_DISTRIBUTIONS = TREES + ("catboost_uncertainty", "lightgbmlss_gaussian", "lightgbmlss_studentt",
                              "lightgbmlss_mixture", "ngboost")
NAMES = {
    "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
    "catboost_uncertainty": "CatBoost RMSEWithUncertainty", "lightgbmlss_gaussian": "LightGBMLSS Gaussian",
    "lightgbmlss_studentt": "LightGBMLSS Student-t", "lightgbmlss_mixture": "LightGBMLSS mixture",
    "ngboost": "NGBoost",
}
GROUPS = (  # (group, ((table, name on the figure, rows), ...)), regression first
    ("Regression", (("airlines_delay", "Airline delay", "2M"), ("year", "Year (MSD)", "515k"),
                    ("zurich", "Zurich transport", "2M"), ("taxi_tip", "NYC green taxi tips", "582k"))),
    ("Classification", (("higgs", "Higgs", "2M"), ("covertype", "Covertype", "581k"),
                        ("click", "Click prediction", "2M"), ("airlines", "Airlines", "539k"))),
)
TIE = 0.5  # percent: with one split, a smaller change is drawn as a tie


def load(path: Path = RESULTS) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("variant", "") == VARIANT.get(r["model"], "")]


def _by_seed(rows, table, model, metric):
    out = {}
    for r in rows:
        if r["table"] == table and r["model"] == model and r.get(metric) is not None:
            out[r["seed"]] = float(r[metric])
    return out


def compare(rows, table, ours, theirs, metric, *, error=False) -> dict | None:
    """Per-seed change of ``ours`` against the best of ``theirs`` (lowest mean over the shared seeds)."""
    def value(model):
        got = _by_seed(rows, table, model, metric)
        return {s: (1.0 - v if error else v) for s, v in got.items()}

    mine = value(ours)
    candidates = {m: value(m) for m in theirs}
    candidates = {m: v for m, v in candidates.items() if v}
    if not mine or not candidates:
        return None
    # Compare on the seeds every candidate and ours share; a candidate with fewer seeds is dropped
    # rather than shrinking the comparison, unless it is the only one.
    seeds = set(mine)
    full = {m: v for m, v in candidates.items() if seeds <= set(v)}
    pool = full or candidates
    seeds = sorted(seeds.intersection(*[set(v) for v in pool.values()]))
    if not seeds:
        return None
    means = {m: np.mean([v[s] for s in seeds]) for m, v in pool.items()}
    best = min(means, key=means.get)
    change = np.array([(mine[s] - pool[best][s]) / pool[best][s] for s in seeds]) * 100
    return {"table": table, "best": best, "seeds": seeds, "change": change, "mean": float(change.mean()),
            "sd": float(change.std(ddof=0)), "ours": float(np.mean([mine[s] for s in seeds])),
            "theirs": float(means[best]), "metric": metric}


def results(rows):
    """``(error, distribution)``: one comparison per table for each panel."""
    error, distribution = {}, {}
    for group, tables in GROUPS:
        for table, _, _ in tables:
            if group == "Regression":
                error[table] = compare(rows, table, "ashlar", TREES, "rmse")
                distribution[table] = compare(rows, table, "ashlar_mixture", TREE_DISTRIBUTIONS, "crps")
            else:
                error[table] = compare(rows, table, "ashlar", TREES, "accuracy", error=True)
                distribution[table] = compare(rows, table, "ashlar", TREES, "log_loss")
    return ({t: v for t, v in error.items() if v}, {t: v for t, v in distribution.items() if v})


def tied(item) -> bool:
    lo, hi = item["mean"] - item["sd"], item["mean"] + item["sd"]
    if len(item["seeds"]) == 1:
        return abs(item["mean"]) < TIE
    return lo <= 0 <= hi


def _percent(x: float) -> str:
    return k.signed(x, 2 if abs(x) < 0.1 else 1)


def _cell(ax, y, item, pad):
    """One table's change in one panel: a line from zero, the whisker, the dot, and the number."""
    mean, sd = item["mean"], item["sd"]
    lo, hi = mean - sd, mean + sd
    color = k.SLATE if tied(item) else k.TEAL if mean < 0 else k.CLAY
    ax.plot([0, mean], [y, y], color=color, lw=2.4, solid_capstyle="butt", zorder=3)
    if len(item["seeds"]) > 1 and sd > 0:
        ax.plot([lo, hi], [y, y], color=color, lw=1.3, zorder=4)
        for end in (lo, hi):
            ax.plot([end, end], [y - 0.13, y + 0.13], color=color, lw=1.3, zorder=4)
    ax.scatter([mean], [y], s=110, color=color, edgecolor=k.CARD, linewidth=1.6, zorder=5)
    if mean <= 0:
        k.value(ax, min(lo, mean) - pad, y, _percent(mean), color, ha="right")
    else:
        k.value(ax, max(hi, mean) + pad, y, _percent(mean), color, ha="left")


def _span(items, floor):
    reach = max([abs(i["mean"]) + i["sd"] for i in items] + [floor])
    return reach * 1.7


def splits_text(items) -> str:
    counts = sorted({len(item["seeds"]) for item in items})
    words = {1: "one", 2: "two", 3: "three"}
    if counts == [1]:
        return "one split per table so far"
    if len(counts) == 1:
        return f"{words.get(counts[0], counts[0])} splits per table"
    return f"{words.get(counts[0], counts[0])} to {words.get(counts[-1], counts[-1])} splits per table so far"


def draw() -> None:
    rows = load()
    error, distribution = results(rows)
    fig = k.card(
        "Level with the trees on accuracy. Ahead on distributions.",
        "Eight OpenML tables of half a million to two million rows, defaults on every side, "
        f"{splits_text(list(error.values()) + list(distribution.values()))}. Each dot is Ashlar's change "
        "against the best tree on that table; left of zero, Ashlar is better.",
        height=9.3,
        footer=3,
    )
    left, right = k.panels(fig, 1, 2, wspace=0.55, widths=[1.0, 1.0], head=0.95, tail=0.5, lpad=2.75, rpad=0.25)[0]
    positions, heads, y = {}, [], 0.0
    for group, tables in GROUPS:
        heads.append((group, y - 0.72))
        for table, _, _ in tables:
            positions[table] = y
            y += 1.0
        y += 0.7
    bottom = y - 0.7
    panels = (
        (left, error, "Error against the best tree", "RMSE, or 1 − accuracy for classification",
         0.9, 4.0),
        (right, distribution, "Distribution against the best tree's",
         "CRPS of task=\"mixture\", or log loss for classification", 1.2, 8.0),
    )
    for ax, items, title, subtitle, pad, floor in panels:
        span = _span(list(items.values()), floor)
        for table, item in items.items():
            _cell(ax, positions[table], item, pad * span / 20)
        k.style(ax, frame="")
        ax.set_xlim(-span, span)
        ax.set_ylim(bottom - 0.45, -1.2)
        for _, tables in GROUPS:
            ax.plot([0, 0], [positions[tables[0][0]] - 0.5, positions[tables[-1][0]] + 0.5], color=k.INK, lw=1.1,
                    zorder=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 1.0, title, transform=k._above(ax, 0.42), fontproperties=k.F.panel, color=k.INK, ha="center",
                va="bottom")
        ax.text(0.5, 1.0, subtitle, transform=k._above(ax, 0.14), fontproperties=k.F.small, color=k.MUTED,
                ha="center", va="bottom")
        gap = span * 0.04
        ax.text(-gap, bottom - 0.05, "← Ashlar better", color=k.TEAL, fontproperties=k.F.label, ha="right", va="top")
        ax.text(gap, bottom - 0.05, "trees better →", color=k.CLAY, fontproperties=k.F.label, ha="left", va="top")
    for group, tables in GROUPS:
        for table, name, size in tables:
            k.rich(left, -0.03, positions[table], [(name, k.INK, k.F.label), (f"  {size}", k.MUTED, k.F.small)],
                   transform=left.get_yaxis_transform(), ha="right", va="center")
    for group, gy in heads:
        left.text(-0.03, gy + 0.18, group.upper(), transform=left.get_yaxis_transform(), fontproperties=k.F.small,
                  color=k.MUTED, ha="right", va="center")
    k.footnote(fig, "The dot is the mean change over the splits and the whisker one standard deviation; grey where it "
                    "crosses zero. Tips are a spike at $0 and a long tail: CatBoost's per-row normal wins their CRPS, "
                    "Ashlar's mixture their likelihood. Ashlar trains in minutes on a laptop CPU, where LightGBM takes "
                    "seconds to minutes. Every number is in docs/benchmarks.md, Large tables.", line=0)
    k.save(fig, "scoreboard")
    for panel, items in (("error", error), ("distribution", distribution)):
        for item in items.values():
            print(panel, item["table"], NAMES.get(item["best"], item["best"]), item["seeds"], round(item["mean"], 2),
                  round(item["sd"], 2), round(item["ours"], 5), round(item["theirs"], 5))


if __name__ == "__main__":
    k.main(draw=draw)
