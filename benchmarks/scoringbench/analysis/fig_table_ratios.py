"""By table, the dressed model's CRPS divided by the best other untuned model's, sorted.

One stem per table from 1 to the ratio, on a log scale. Filled teal: the dressed model is lowest of the 21 on that
table; open clay: another untuned model is lower. Reads out/by_table.csv (run by_table.py first). Writes
figures/table_ratios.pdf and .png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402
import figstyle as s  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

s.setup()
bt = pd.read_csv(c.OUT / "by_table.csv").sort_values("ratio_to_best_other").reset_index(drop=True)
x = np.arange(1, len(bt) + 1)
r = bt.ratio_to_best_other.values
first = r < 1

fig, ax = plt.subplots(figsize=(s.WIDTH, 2.6))
fig.subplots_adjust(left=0.115, right=0.975, top=0.93, bottom=0.17)
ax.vlines(x, 1, r, color=s.LIGHT, lw=0.9, zorder=1)
ax.axhline(1, color=s.INK, lw=0.6, zorder=2)
ax.scatter(x[first], r[first], s=13, color=s.TEAL, edgecolors="white", linewidths=0.4, zorder=3)
ax.scatter(x[~first], r[~first], s=15, facecolors="white", edgecolors=s.CLAY, linewidths=1.0, zorder=3)
ax.set_yscale("log")
ticks = [0.5, 0.75, 1, 1.5, 2, 2.5]
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_minor_locator(NullLocator())
ax.set_yticklabels([f"{t:g}" for t in ticks])
ax.set_ylim(0.48, 2.75)
ax.set_xlim(0, len(bt) + 1)
ax.set_xticks([1, 20, 40, 60, 80, 99])

n_first = int(first.sum())
ax.text(n_first / 2, 1.65, f"{n_first} tables: the dressed model is lowest", ha="center", va="center",
        fontproperties=s.font("Regular", s.NOTE), color=s.INK)
ax.text(len(bt) - 9, 0.62, f"{len(bt) - n_first} tables: another\nuntuned model is lower", ha="center", va="center",
        fontproperties=s.font("Regular", s.NOTE), color=s.INK, linespacing=1.15)
mono = s.font("Regular", s.NOTE, mono=True)
i = int(bt.index[bt.dataset == "Brazilian_houses"][0])
ax.text(x[i] - 1.3, r[i], "Brazilian_houses", ha="right", va="center", fontproperties=mono, color=s.MUTED)
pair = [int(bt.index[bt.dataset == name][0]) for name in ("344_mv", "visualizing_soil")]
ax.text(min(x[pair]) - 1.3, float(np.mean(r[pair])), "344_mv, visualizing_soil", ha="right", va="center", fontproperties=mono,
        color=s.MUTED)
lo = bt.iloc[0]
ax.text(x[0] + 1.3, r[0], lo.dataset, ha="left", va="center", fontproperties=s.font("Regular", s.NOTE, mono=True), color=s.MUTED)

ax.set_xlabel("The 99 tables, sorted by the ratio", fontproperties=s.font("Regular", s.LABEL))
ax.set_ylabel("CRPS ratio, dressed model\nto best other untuned", fontproperties=s.font("Regular", s.LABEL), linespacing=1.15)
s.clean(ax, left=True, grid="y")
for t in ax.get_xticklabels() + ax.get_yticklabels():
    t.set_fontproperties(s.font("Regular", s.TICK))
print(s.save(fig, "table_ratios"))
