"""Empirical coverage of central intervals minus nominal, averaged over tables, by nominal level.

Two panels, each with its own vertical scale: the 87 tables with more than 20 distinct target values in some
training fold (the dressed model fits a density) and the 12 with at most 20 in every fold (it predicts a probability
for each value). Three models: the dressed model and the two strongest other untuned models by CRPS mean rank. Reads out/coverage.csv (run
calibration.py first). Writes figures/coverage.pdf and .png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402
import figstyle as s  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

s.setup()
cov = pd.read_csv(c.OUT / "coverage.csv")
ranks = pd.read_csv(c.OUT / "untuned_crps_ranks.csv")
others = ranks[ranks.model != c.DRESSED].model.head(2).tolist()
STYLE = {c.DRESSED: (s.TEAL, "o", "Dressed model (our run)"), others[0]: (s.CLAY, "s", others[0]), others[1]: (s.SLATE, "^", others[1])}

fig, axes = plt.subplots(1, 2, figsize=(s.WIDTH, 2.55), gridspec_kw={"width_ratios": [1, 1]})
fig.subplots_adjust(left=0.1, right=0.985, top=0.8, bottom=0.17, wspace=0.22)
TITLES = {"more than 20 values": "{n} tables with more than 20 target values", "at most 20 values": "{n} tables with at most 20 target values"}
LIMITS = {"more than 20 values": (-5, 16), "at most 20 values": (-15, 55)}
for ax, subset in zip(axes, ("more than 20 values", "at most 20 values")):
    sub = cov[cov.tables == subset]
    n = int(sub.n_tables.iloc[0])
    ax.axhline(0, color=s.INK, lw=0.6, zorder=1)
    for m, (color, marker, _) in STYLE.items():
        d = sub[sub.model == m].sort_values("nominal")
        ax.plot(range(len(d)), (d.mean_coverage - d.nominal) * 100, color=color, lw=1.3, marker=marker, ms=4.2,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3 if m == c.DRESSED else 2)
    ax.set_title(TITLES[subset].format(n=n), fontproperties=s.font("Medium", s.LABEL), color=s.INK, loc="left", pad=4)
    ax.set_ylim(*LIMITS[subset])
    ax.set_xticks(range(6), ["20", "40", "60", "80", "90", "95"])  # the six levels, evenly spaced
    ax.set_xlim(-0.35, 5.35)
    ax.set_xlabel("Nominal coverage of the central interval (%)", fontproperties=s.font("Regular", s.LABEL))
    s.clean(ax, left=True, grid="y")
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontproperties(s.font("Regular", s.TICK))
axes[0].set_ylabel("Empirical minus nominal\n(percentage points)", fontproperties=s.font("Regular", s.LABEL), linespacing=1.15)
handles = [Line2D([], [], color=color, marker=marker, lw=1.3, ms=4.2, markeredgecolor="white", markeredgewidth=0.6)
           for color, marker, _ in STYLE.values()]
labels = [label for *_, label in STYLE.values()]
leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1, 0.995), ncol=3, frameon=False, handlelength=2.0,
                 columnspacing=1.6, handletextpad=0.5)
for t, m in zip(leg.get_texts(), STYLE):
    t.set_fontproperties(s.font("SemiBold", s.NOTE) if m == c.DRESSED else s.font("Regular", s.NOTE, mono=True))
print(s.save(fig, "coverage"))
