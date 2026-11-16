"""Mean ranks of all 64 models on 99 tables, one row per family, the dressed model marked.

Families are rows (position carries identity, so the figure needs no family colours); teal marks the dressed model.
Within a row, points that would overlap are stacked a little apart. Reads out/full_field_ranks.csv and
out/numbers.json (run full_field.py first). Writes figures/full_field.pdf and .png.
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
field = pd.read_csv(c.OUT / "full_field_ranks.csv")
num = json.loads((c.OUT / "numbers.json").read_text())["full_field"]
me, cd, pos = num["dressed_mean_rank"], num["nemenyi_cd"], num["dressed_position"]
ROWS = [("pretrained", "Pretrained"), ("fine-tuned", "Fine-tuned from a\npretrained model"),
        ("tuned", "Trained from scratch,\ntuned on each table"), ("untuned", "Trained from scratch,\nuntuned")]


def swarm(xs, width, step):
    """Vertical offsets so that no two points closer than ``width`` in x share a level (0, +1, -1, +2, -2, ...)."""
    placed, out = [], []
    for x in xs:
        for level in (0, 1, -1, 2, -2, 3, -3):
            if all(abs(x - px) >= width or level != pl for px, pl in placed):
                placed.append((x, level))
                out.append(level * step)
                break
    return out


fig, ax = plt.subplots(figsize=(s.WIDTH, 2.75))
fig.subplots_adjust(left=0.215, right=0.975, top=0.83, bottom=0.17)
ax.axvspan(me - cd, me + cd, color=s.TEAL_WASH, lw=0, zorder=0)
ax.axvline(me, color=s.TEAL, lw=0.8, zorder=1)
counts = field.family.value_counts().to_dict()
for i, (fam, label) in enumerate(ROWS):
    yc = len(ROWS) - 1 - i
    sub = field[field.family == fam].sort_values("meanrank")
    offsets = swarm(sub.meanrank.tolist(), width=0.95, step=0.17)
    for (r, dy) in zip(sub.itertuples(), offsets):
        dressed = r.model == c.DRESSED
        ax.scatter([r.meanrank], [yc + dy], s=24 if dressed else 15, color=s.TEAL if dressed else s.GREY,
                   edgecolors="white", linewidths=0.6, zorder=4 if dressed else 3)
    n = counts[fam] - (1 if fam == "untuned" else 0)
    ax.text(-0.015, yc, f"{label} ({n}{' + ours' if fam == 'untuned' else ''})", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontproperties=s.font("Regular", s.TICK), color=s.INK, linespacing=1.15)
    ax.axhline(yc, color=s.HAIR, lw=0.5, zorder=0)

# Direct labels: the dressed model and the two tuned models.
tuned = field[field.family == "tuned"].set_index("model")
ax.annotate(f"Dressed model, {pos}th, mean rank {me:.2f}", xy=(me, 0), xytext=(me - 1.5, -0.6),
            fontproperties=s.font("SemiBold", s.NOTE), color=s.INK, va="center", ha="right",
            arrowprops=dict(arrowstyle="-", color=s.TEAL, lw=0.6, shrinkA=0, shrinkB=3))
for m, text, dx in ((c.REALMLP_TUNED, "RealMLP", -1.0), (c.TABM_TUNED, "TabM", 1.0)):
    x = tuned.loc[m, "meanrank"]
    ax.text(x + dx, 1.28, f"{text} {x:.2f}", ha="right" if dx < 0 else "left", va="center",
            fontproperties=s.font("Regular", s.NOTE), color=s.MUTED)
ax.text(me, len(ROWS) - 0.35, f"±1 critical difference ({cd:.2f})", ha="center", va="bottom",
        fontproperties=s.font("Regular", s.NOTE), color=s.INK)

ax.set_xlim(0, 65)
ax.set_ylim(-0.95, len(ROWS) - 0.4)
ax.set_yticks([])
ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
ax.set_xlabel("Mean rank by CRPS among 64 models over 99 tables (1 is best)", fontproperties=s.font("Regular", s.LABEL))
s.clean(ax)
for t in ax.get_xticklabels():
    t.set_fontproperties(s.font("Regular", s.TICK))
print(s.save(fig, "full_field"))
