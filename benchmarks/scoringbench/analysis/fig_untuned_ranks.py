"""Mean rank of the 21 untuned models on 99 tables, with the Nemenyi critical difference marked.

Reads out/untuned_crps_ranks.csv and out/numbers.json (run untuned_crps.py first). Writes figures/untuned_ranks.pdf
and .png.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402
import figstyle as s  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

s.setup()
ranks = pd.read_csv(c.OUT / "untuned_crps_ranks.csv").sort_values("meanrank").reset_index(drop=True)
num = json.loads((c.OUT / "numbers.json").read_text())["untuned_crps"]
cd, me = num["nemenyi_cd"], num["dressed_mean_rank"]
n = len(ranks)

fig, ax = plt.subplots(figsize=(s.WIDTH, 3.55))
fig.subplots_adjust(left=0.385, right=0.975, top=0.86, bottom=0.12)
y = list(range(n))[::-1]  # best at the top

# The critical difference, measured from the dressed model's mean rank.
ax.axvspan(me, me + cd, color=s.TEAL_WASH, zorder=0, lw=0)
ax.plot([me, me + cd], [n - 0.15, n - 0.15], color=s.TEAL, lw=0.9, clip_on=False, solid_capstyle="butt")
for x in (me, me + cd):
    ax.plot([x, x], [n - 0.35, n + 0.05], color=s.TEAL, lw=0.9, clip_on=False, solid_capstyle="butt")
ax.text(me + cd / 2, n + 0.35, f"critical difference {cd:.2f} (Nemenyi, 5%)", ha="center", va="bottom",
        fontproperties=s.font("Regular", s.NOTE), color=s.INK, clip_on=False)

for yi, r in zip(y, ranks.itertuples()):
    dressed = r.model == c.DRESSED
    ax.plot([0.5, r.meanrank], [yi, yi], color=s.HAIR, lw=0.5, zorder=1)
    ax.scatter([r.meanrank], [yi], s=26 if dressed else 18, color=s.TEAL if dressed else s.GREY, zorder=3,
               edgecolors="white", linewidths=0.8)
    label = "Dressed model (our run)" if dressed else r.model
    ax.text(-0.02, yi, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontproperties=s.font("SemiBold", s.TICK) if dressed else s.font("Regular", s.TICK, mono=True),
            color=s.INK if dressed else s.MUTED)
    if dressed or r.Index == 1:
        ax.text(r.meanrank + 0.25, yi, f"{r.meanrank:.2f}", va="center", ha="left",
                fontproperties=s.font("Medium", s.NOTE), color=s.INK)

ax.set_xlim(0.5, 17.5)
ax.set_ylim(-0.7, n - 0.3)
ax.set_yticks([])
ax.set_xticks([1, 5, 10, 15])
ax.set_xlabel("Mean rank by CRPS over 99 tables (1 is best)", fontproperties=s.font("Regular", s.LABEL))
s.clean(ax, grid="x")
for t in ax.get_xticklabels():
    t.set_fontproperties(s.font("Regular", s.TICK))
print(s.save(fig, "untuned_ranks"))
