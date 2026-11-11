"""A dressed model against one network, on dev tables it was not designed on.

``compute`` copies the scores of the 46 held-out fits (folds 1 and 2 of 23 tables) from the dev-set results
(``ASHLAR_DRESSED_RESULTS``, a JSON-lines file with one record per table and fold); ``draw`` plots, per table, the
dressed model's CRPS and the undressed pool's, each relative to one network alone. docs/dressed.md has the test.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import style as k

RESULTS = Path(os.environ.get("ASHLAR_DRESSED_RESULTS", Path.home() / "projects/ScoringBench/runs/devset/density.jsonl"))
NAMES = {"557_analcatdata_apnea1": "apnea1", "560_bodyfat": "bodyfat", "690_visualizing_galaxy": "galaxy",
         "666_rmftsa_ladata": "ladata", "1595_poker": "poker", "519_vinnie": "vinnie", "nikuradse_1": "nikuradse",
         "557_analcatdata_apnea1_rated6": "apnea1, 6 ratings", "cars_rated7": "cars, 7 ratings",
         "auction_verification_rated5": "auction_verification, 5 ratings", "grid_stability_rated10": "grid_stability, 10 ratings",
         "geographical_origin_of_music": "origin_of_music", "naval_propulsion_plant": "naval_propulsion"}
EDGE = 0.4  # ratios below this are drawn at the left edge with their value written out


def compute():
    rows = [json.loads(line) for line in RESULTS.read_text().splitlines()]
    keep = ("table", "fold", "kind", "rows", "network", "bagged", "dressed", "log_dressed", "log_bagged_network", "gate")
    k.save_json("dressed", "devset", [{name: r[name] for name in keep} for r in rows if r["fold"] in (1, 2)])


def draw():
    rows = k.load_json("dressed", "devset")
    tables = sorted({r["table"] for r in rows}, key=lambda t: np.mean([np.log(r["dressed"] / r["network"]) for r in rows if r["table"] == t]))
    ratio = np.asarray([r["dressed"] / r["network"] for r in rows])
    better, worse = int((ratio < 0.99).sum()), int((ratio > 1.01).sum())
    fig = k.card("A dressed model against one network alone, on 46 fits it was not designed on.",
                 "CRPS of ashlar.dressed() divided by the CRPS of one of its own networks, on folds 1 and 2 of 23 "
                 "regression tables (at most 2,400 training rows). Left of the line the dressed model is better.",
                 height=10.9, footer=4)
    ax = k.panel(fig, lpad=3.3, tail=0.75, head=0.2)
    ax.set_xscale("log")
    ax.set_xlim(EDGE * 0.93, 1.09)
    ax.set_ylim(len(tables) - 0.4, -0.6)
    k.style(ax, frame="b", grid="x")
    ax.axvline(1.0, color=k.INK, lw=1.1, zorder=2)
    ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontproperties=k.F.tick, color=k.MUTED)
    ax.set_xticks([], minor=True)
    ax.set_yticks([])
    for y, table in enumerate(tables):
        mine = sorted((r for r in rows if r["table"] == table), key=lambda r: r["fold"])
        k.rich(ax, -0.015, y, [(NAMES.get(table, table), k.INK, k.F.label), (f"  {mine[0]['rows']:,} rows", k.MUTED, k.F.small)],
               transform=ax.get_yaxis_transform(), ha="right", va="center")
        ax.plot([EDGE * 0.93, 1.09], [y, y], color=k.GRID, lw=0.8, zorder=1)
        off = []
        for shift, r in zip((-0.16, 0.16), mine, strict=True):
            pooled, dressed = r["bagged"] / r["network"], r["dressed"] / r["network"]
            ax.plot(max(pooled, EDGE), y + shift, marker="o", ms=7.5, mfc=k.CARD, mec=k.SLATE, mew=1.6, lw=0, zorder=3)
            ax.plot(max(dressed, EDGE), y + shift, marker="o", ms=7.5, mfc=k.TEAL, mec=k.CARD, mew=0.8, lw=0, zorder=4)
            if dressed < EDGE:
                off.append(f"{dressed:.2g}")
        if off:
            k.value(ax, EDGE * 1.035, y, " and ".join(off), color=k.TEAL, size=10.5)
    k.axis_names(ax, x="CRPS relative to one network alone (log scale)")
    k.legend_row(fig, [("dot", k.TEAL, "the dressed model"), ("dot", k.SLATE, "the same five members pooled, not dressed (no density)")], line=3)
    k.footnote(fig, f"Each table has two dots of each kind, fold 1 above fold 2. The design was fixed on fold 0 before these folds were read. "
                    f"The dressed model is better than one network by more than 1% on {better} of the {len(rows)} fits and worse by more than 1% on "
                    f"{worse} (by 1.4% at most). On wave_energy, a linear target, the undressed pool is sharper still: see docs/dressed.md, Limits.", line=0)
    k.save(fig, "dressed")
    print("better", better, "worse", worse, "max", float(ratio.max()))


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
