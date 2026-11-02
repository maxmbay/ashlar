"""The scoreboard: Ashlar against the best gradient-boosted trees on the eight large tables.

Reads the large-table benchmark's results (``examples/benchmark_large.py``, shipped from the Mac to
``examples/readme/artifacts/large/large.jsonl``) and draws ``docs/images/readme/scoreboard.png``:

- Left: Ashlar's default against the best of LightGBM, XGBoost, and CatBoost on each table, as the
  change in test error (RMSE, or 1 − accuracy for the classification tables).
- Right: the predicted distribution against the best tree's. On the regression tables,
  ``ASRRegressor(task="mixture")``'s CRPS against the best tree distribution: LightGBM, XGBoost, and
  CatBoost with a normal of one width, CatBoost ``RMSEWithUncertainty``, and LightGBMLSS (a Gaussian, a
  Student-t, and a mixture of three Gaussians).
  On the classification tables, the default's log loss against the trees'.

The best tree is the one with the lowest mean over the seeds that ours and every tree share, so a
model with fewer seeds can still be the best; each seed the two share is a change against it, the dot is
the mean change and the whisker one standard deviation across seeds. A dot is teal or clay
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
                              "lightgbmlss_mixture")  # NGBoost is held, as in the benchmark's report
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
DOT_RADIUS = (110 ** 0.5 / 2 + 1.6) / 72  # inches: the dot (s=110) and its edge


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
    candidates = {m: v for m, v in ((m, value(m)) for m in theirs) if set(v) & set(mine)}
    if not candidates:
        return None
    # The best candidate is picked on the seeds ours and every candidate share, so that one with fewer
    # seeds than the others can still be the best; ours is then compared with it on every seed the two share.
    common = set(mine).intersection(*[set(v) for v in candidates.values()])

    def mean_over(model):
        seeds = common or set(mine) & set(candidates[model])
        return np.mean([candidates[model][s] for s in seeds])

    best = min(candidates, key=mean_over)
    seeds = sorted(set(mine) & set(candidates[best]))
    theirs_values = [candidates[best][s] for s in seeds]
    change = np.array([(mine[s] - candidates[best][s]) / candidates[best][s] for s in seeds]) * 100
    return {"table": table, "best": best, "seeds": seeds, "change": change, "mean": float(change.mean()),
            "sd": float(change.std(ddof=0)), "ours": float(np.mean([mine[s] for s in seeds])),
            "theirs": float(np.mean(theirs_values)), "metric": metric,
            "ours_sd": float(np.std([mine[s] for s in seeds])), "theirs_sd": float(np.std(theirs_values))}


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
    # A whisker shorter than the dot's radius would only show its caps at the dot's edge: leave it out.
    inches = ax.get_position().width * ax.figure.get_figwidth()
    radius = DOT_RADIUS * (ax.get_xlim()[1] - ax.get_xlim()[0]) / inches
    whisker = len(item["seeds"]) > 1 and sd > radius
    if whisker:
        ax.plot([lo, hi], [y, y], color=color, lw=1.3, zorder=4)
        for end in (lo, hi):
            ax.plot([end, end], [y - 0.13, y + 0.13], color=color, lw=1.3, zorder=4)
    ax.scatter([mean], [y], s=110, color=color, edgecolor=k.CARD, linewidth=1.6, zorder=5)
    if not whisker:
        lo = hi = mean
    if mean <= 0:
        k.value(ax, lo - pad, y, _percent(mean), color, ha="right")
    else:
        k.value(ax, hi + pad, y, _percent(mean), color, ha="left")


def _span(items, floor):
    reach = max([abs(i["mean"]) + i["sd"] for i in items] + [floor])
    return reach * 1.7


WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def headline(error, distribution) -> str:
    """The card's title, in numbers: the README's heading above the figure says the claim in words.

    The tables whose error is within half a percent of the best tree's, the ones ahead by more, and the
    range of the CRPS gain on the regression tables where the mixture is ahead.
    """
    level = [abs(v["mean"]) for v in error.values() if abs(v["mean"]) < TIE]
    ahead = [v for v in error.values() if v["mean"] <= -TIE]
    regression = {table for table, _, _ in GROUPS[0][1]}
    gains = sorted(-v["mean"] for t, v in distribution.items() if t in regression and v["mean"] <= -TIE)
    parts = []
    if level:
        parts.append(f"Within {np.ceil(max(level) * 10 - 1e-9) / 10:g}% on {WORDS[len(level)]} tables")
    if ahead:
        parts.append(f"ahead on {WORDS[len(ahead)]}" if level else f"Ahead on {WORDS[len(ahead)]} tables")
    first = ", ".join(parts) or "Against the best tree on each table"
    if not gains:
        return first + "."
    low, high = round(gains[0]), round(gains[-1])
    ranges = [f"{high}%"] if low == high else [f"{low}% to {high}%", f"{low}–{high}%"]
    fonts = k.setup()
    room = k.WIDTH - 2 * 0.62 - 0.2  # the card's text width, less a little air
    for span in ranges:
        text = f"{first}; CRPS {span} lower on {WORDS[len(gains)]}."
        if k.text_width(text, fonts.title) <= room:
            return text
    return first + "."


POSSESSIVE = {  # the best tree distribution, as the footnote names it
    "lightgbm": "LightGBM's normal of one width", "xgboost": "XGBoost's normal of one width",
    "catboost": "CatBoost's normal of one width", "catboost_uncertainty": "CatBoost's RMSEWithUncertainty",
    "lightgbmlss_gaussian": "LightGBMLSS's Gaussian", "lightgbmlss_studentt": "LightGBMLSS's Student-t",
    "lightgbmlss_mixture": "LightGBMLSS's mixture of three normals",
}
SHORT = {"airlines_delay": "airline delay", "year": "release years", "zurich": "Zurich transport",
         "taxi_tip": "taxi tips"}


def _and(items: list[str]) -> str:
    if len(items) < 3:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def best_distributions(distribution) -> str:
    """One sentence naming the best tree distribution on each regression table, in the figure's order."""
    order, by_model = [], {}
    for table, _, _ in GROUPS[0][1]:
        if table in distribution:
            best = distribution[table]["best"]
            if best not in by_model:
                order.append(best)
            by_model.setdefault(best, []).append(SHORT[table])
    if not order:
        return ""
    parts = [f"{POSSESSIVE[model]} on {_and(by_model[model])}" for model in order]
    if len(parts) == 2 and not any("," in part for part in parts):
        joined = f"{parts[0]}, and {parts[1]}"
    else:
        joined = "; ".join(parts)
    return f"The best tree distribution is {joined}."


def splits_text(items) -> str:
    counts = sorted({len(item["seeds"]) for item in items})
    if counts == [1]:
        return "one split per table so far"
    if len(counts) == 1:
        return f"{WORDS.get(counts[0], counts[0])} splits per table"
    return f"{WORDS.get(counts[0], counts[0])} to {WORDS.get(counts[-1], counts[-1])} splits per table so far"


def draw() -> None:
    rows = load()
    error, distribution = results(rows)
    fig = k.card(
        headline(error, distribution),
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
        ax.set_xlim(-span, span)
        ax.set_ylim(bottom - 0.45, -1.2)
        for table, item in items.items():
            _cell(ax, positions[table], item, pad * span / 20)
        k.style(ax, frame="")
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
                    f"crosses zero. {best_distributions(distribution)} Ashlar trains in minutes on a laptop CPU, where "
                    "LightGBM takes seconds to minutes. Every number is in docs/benchmarks.md, Large tables.", line=0)
    k.save(fig, "scoreboard")
    for panel, items in (("error", error), ("distribution", distribution)):
        for item in items.values():
            print(panel, item["table"], NAMES.get(item["best"], item["best"]), item["seeds"], round(item["mean"], 2),
                  round(item["sd"], 2), round(item["ours"], 5), round(item["theirs"], 5))


ROWS = {"airlines_delay": "2,000,000", "year": "515,345", "zurich": "2,000,000", "taxi_tip": "581,835",
        "higgs": "2,000,000", "covertype": "581,012", "click": "1,997,410", "airlines": "539,383"}
TARGETS = {"airlines_delay": "delay in minutes", "year": "release year", "zurich": "delay", "taxi_tip": "tip in $",
           "higgs": "signal", "covertype": "7 forest types", "click": "click", "airlines": "delayed"}


def _entry(value: float, sd: float, seeds: int, digits: int, bold: bool, name: str = "") -> str:
    """A table cell: the mean (bold when it is the better one), ± the sd over the splits, and the model's name."""
    text = f"{value:.{digits}f}".replace("-", "−")
    text = f"**{text}**" if bold else text
    if seeds > 1:
        text += f" ± {sd:.1g}"
    return text + (f" ({name})" if name else "")


def _bold(got: dict, digits: int, ours_better: bool) -> tuple[bool, bool]:
    """Which cell to bold: the better one, or neither when the two print the same."""
    if f"{got['ours']:.{digits}f}" == f"{got['theirs']:.{digits}f}":
        return False, False
    return ours_better, not ours_better


def tables() -> None:
    """The README's two tables, as means ± sd over the splits: point accuracy, then the distributions."""
    rows = load()
    print("| Table | Rows | Target, metric | Ashlar | Best tree |\n|---|---:|---|---:|---:|")
    for group, entries in GROUPS:
        for table, name, _ in entries:
            regression = group == "Regression"
            got = compare(rows, table, "ashlar", TREES, "rmse" if regression else "accuracy", error=not regression)
            if got is None:
                continue
            ours_better = got["ours"] < got["theirs"]  # an error either way: RMSE, or 1 - accuracy
            if not regression:
                got["ours"], got["theirs"] = 1.0 - got["ours"], 1.0 - got["theirs"]
            digits = (2 if got["ours"] > 10 else 3) if regression else 3
            ours_bold, theirs_bold = _bold(got, digits, ours_better)
            print(f"| {name} | {ROWS[table]} | {TARGETS[table]}, {'RMSE ↓' if regression else 'accuracy ↑'} | "
                  f"{_entry(got['ours'], got['ours_sd'], len(got['seeds']), digits, ours_bold)} | "
                  f"{_entry(got['theirs'], got['theirs_sd'], len(got['seeds']), digits, theirs_bold, NAMES[got['best']])} |")
    print()
    print("| Table | Score | Ashlar | Best tree distribution |\n|---|---|---:|---:|")
    for group, entries in GROUPS:
        for table, name, _ in entries:
            regression = group == "Regression"
            # CRPS only: on a target with a spike (airline delay's on-time minute, taxi tips' $0) the mixture's
            # NLL turns on how narrow it makes the spike, so it is not a comparison (docs/benchmarks.md).
            scores = (("crps", "CRPS"),) if regression else (("log_loss", "Log loss"),)
            for n, (metric, label) in enumerate(scores):
                got = compare(rows, table, "ashlar_mixture" if regression else "ashlar",
                              TREE_DISTRIBUTIONS if regression else TREES, metric)
                if got is None:
                    continue
                digits = 2 if regression else 3
                ours_bold, theirs_bold = _bold(got, digits, got["ours"] < got["theirs"])
                title = f"{name}, {TARGETS[table]}" if regression else name
                print(f"| {title if n == 0 else ''} | {label} | "
                      f"{_entry(got['ours'], got['ours_sd'], len(got['seeds']), digits, ours_bold)} | "
                      f"{_entry(got['theirs'], got['theirs_sd'], len(got['seeds']), digits, theirs_bold, NAMES[got['best']])} |")


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["tables"]:
        tables()
    else:
        k.main(draw=draw)
