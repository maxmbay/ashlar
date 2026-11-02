"""Figure ``joint-blocks``: the logo is a joint density, and so is this one.

Two stages, run from the repository root::

    JAX_PLATFORMS=cpu python examples/readme/joint_blocks.py compute   # a minute or two on four cores
    python examples/readme/joint_blocks.py draw                        # seconds; needs the style kit's fonts

``compute`` fits ``task="multivariate_gaussian"`` on ``log1p([casual,
registered])`` of Capital Bikeshare's hourly rentals exactly as split 0 of
``examples/benchmark_joint_riders.py`` does (the same training rows, early
stopping on the same validation rows, ``random_state=0``), takes two held-out
hours, a working day's 08:00 and a Sunday's 14:00 in clear summer weather,
and writes ``examples/readme/artifacts/joint-blocks/blocks.json``: each
hour's predicted probability in every cell of one 16 × 16 grid of log riders
(from 400,000 joint draws), its predicted correlation, what happened that
hour, and the riders benchmark's numbers from
``/tmp/ashlar-bench/joint-riders.json``, which ``benchmark_joint_riders.py
run`` writes.

``draw`` stacks each hour's mass in the logo's isometric blocks on one floor.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import style as k  # noqa: E402

NAME = "joint-blocks"
SEED = 0
CELLS = 16
DRAWS = 400_000
# The floor runs from the lower 0.5% to the upper 99.5% of either hour's draws on each axis, and a margin beyond.
COVER = 0.005
MARGIN = 0.06
# (name, rows of the held-out hours that qualify): the first qualifying test hour of split 0 is drawn.
HOURS = {
    "commute": {"workingday": 1, "hr": 8, "weathersit": "1", "season": "2"},
    "leisure": {"weekday": 0, "hr": 14, "weathersit": "1", "season": "2"},
}
DAYS = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")


def compute() -> None:
    import jax
    from sklearn.model_selection import train_test_split

    from ashlar import ASRRegressor
    from benchmark_joint_riders import RESULTS, riders

    x, counts = riders()
    y = np.log1p(counts)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)
    x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=SEED)
    test = x_test.reset_index(drop=True)
    picks = {}
    for name, condition in HOURS.items():
        mask = np.ones(len(test), dtype=bool)
        for column, value in condition.items():
            mask &= (test[column] == value).to_numpy()
        picks[name] = int(np.flatnonzero(mask)[0])
    rows = test.iloc[list(picks.values())].reset_index(drop=True)
    model = ASRRegressor(task="multivariate_gaussian", random_state=SEED).fit(x_fit, y_fit, validation=(x_val, y_val))
    draws = np.asarray(model.predict_dist(rows).sample(jax.random.PRNGKey(SEED), (DRAWS,)), dtype=np.float64)
    low = np.quantile(draws, COVER, axis=0).min(axis=0)  # draws are (draws, hours, 2)
    high = np.quantile(draws, 1.0 - COVER, axis=0).max(axis=0)
    low, high = low - MARGIN * (high - low), high + MARGIN * (high - low)
    edges = [np.linspace(low[c], high[c], CELLS + 1) for c in range(2)]
    hours = {}
    for h, (name, row) in enumerate(picks.items()):
        features = test.iloc[row]
        hours[name] = {
            "row": row,
            "day": DAYS[int(features["weekday"])],
            "hour": int(features["hr"]),
            "workingday": int(features["workingday"]),
            "temp": float(features["temp"]),
            "observed": np.expm1(y_test[row]).round().astype(int),
            "observed_log1p": y_test[row],
            "mass": np.histogram2d(draws[:, h, 0], draws[:, h, 1], bins=edges)[0] / DRAWS,
            "mean_log1p": draws[:, h].mean(axis=0),
            "correlation": float(np.corrcoef(draws[:, h].T)[0, 1]),
        }
    k.save_json(NAME, "blocks", {
        "split": SEED,
        "edges_log1p": edges,
        "hours": hours,
        "total_hours": int(len(x)),
        "test_hours": int(len(test)),
        "riders": json.loads(RESULTS.read_text()) if RESULTS.exists() else None,
    })


COURSES = 12  # blocks at the tallest cell, as in the logo
THUMB_COURSES = 7
TICKS = ((10, 30, 100, 300), (150, 300, 600))  # casual, registered
# The evidence: (benchmark model, label), the first being the model drawn.
EVIDENCE = (
    ("joint", "this model, one fit"),
    ("independent + static correlation", "fitted apart + one correlation"),
    ("independent heads", "two heads fitted apart"),
    ("LightGBM quantiles of the total", "LightGBM quantiles of the total"),
    ("LightGBM quantiles per group, summed", "LightGBM per group, summed"),
)


def _cell(value: float, edges) -> float:
    """Where ``value`` riders fall along an axis of the floor, in cells from its low end."""
    return (np.log1p(value) - edges[0]) / (edges[-1] - edges[0]) * CELLS


def _floor(mass):
    """The two hours' mass summed, and each cell's colour: as in the logo, the hour that claims its mass."""
    total = mass["commute"] + mass["leisure"]
    share = np.divide(mass["commute"], total, out=np.zeros_like(total), where=total > 0)
    colors = np.empty(total.shape, dtype=object)
    for i in range(total.shape[0]):
        for j in range(total.shape[1]):
            colors[i, j] = k.mix(k.TEAL, k.CLAY, float(np.clip((share[i, j] - 0.25) / 0.5, 0.0, 1.0)))
    return total, colors


def draw() -> None:
    data = k.load_json(NAME, "blocks")
    edges = [np.asarray(e) for e in data["edges_log1p"]]
    hours = data["hours"]
    # Registered riders run from the back corner forward, most first, so the commute stands at the back and the
    # weekend afternoon in front, where the logo has its two modes.
    mass = {name: np.asarray(hours[name]["mass"])[:, ::-1] for name in ("commute", "leisure")}
    total, colors = _floor(mass)
    heights = k.stacks(total, COURSES)

    fig = k.card(
        "The logo is a joint density. So is this one.",
        f"Capital Bikeshare, {data['total_hours']:,} hours of 2011 and 2012. One fit predicts casual and registered "
        "riders together. Two held-out hours on one 16 × 16 floor of log riders: each stack is the predicted "
        "chance of its cell, and the ink dot is what happened.",
        height=9.0,
    )
    box = fig.box
    blocks_width = 0.68  # of the card, from its left edge
    ax = fig.add_axes([0.0, 0.0, blocks_width, 1.0 - (box.top - 0.1) / box.height])
    k.isometric_blocks(ax, heights, colors)
    half = CELLS * 0.866
    ax.set_xlim(-half - 3.6, half + 3.6)
    ax.set_ylim(-CELLS - 2.9, COURSES * 0.652 + 0.3)

    # Ticks and names along the two front edges.
    for axis, (sign, name) in enumerate(((-1.0, "casual riders"), (1.0, "registered riders"))):
        normal = np.array([0.5 * sign, -0.866])
        for tick in TICKS[axis]:
            c = _cell(tick, edges[axis])
            if not 0.3 <= c <= CELLS - 0.3:
                continue
            x, y = k.iso(c, CELLS) if axis == 0 else k.iso(CELLS, CELLS - c)
            ax.plot([x, x + 0.3 * normal[0]], [y, y + 0.3 * normal[1]], color=k.MUTED, lw=0.9)
            ax.text(x + 0.85 * normal[0], y + 0.85 * normal[1], f"{tick:,}", fontproperties=k.F.tick, color=k.MUTED,
                    ha="center", va="center")
        x, y = k.iso(CELLS * 0.58, CELLS) if axis == 0 else k.iso(CELLS, CELLS * 0.58)
        ax.text(x + 2.1 * normal[0], y + 2.1 * normal[1], f"{name}, log scale", fontproperties=k.F.axis, color=k.INK,
                ha="center", va="center", rotation=30.0 * sign)

    # Each hour: a label on a leader line from its tallest stack, and what happened as an ink dot.
    places = {"commute": (-1.0, 4.4, 1.3), "leisure": (1.0, 5.4, 0.4)}
    for name, color in (("commute", k.CLAY), ("leisure", k.TEAL)):
        hour = hours[name]
        i, j = np.unravel_index(np.argmax(mass[name]), mass[name].shape)
        top = k.iso(i + 0.5, j + 0.5, heights[i, j])
        side, dx, dy = places[name]
        anchor = (top[0] + side * dx, top[1] + dy)
        ax.plot([top[0], anchor[0]], [top[1], anchor[1]], color=color, lw=1.2, zorder=50)
        ha = "left" if side > 0 else "right"
        when = f"{hour['day']} {hour['hour']:02d}:00" + (", a working day" if hour["workingday"] else "")
        ax.text(anchor[0] + side * 0.25, anchor[1] + 0.05, when, fontproperties=k.F.label, color=color, ha=ha,
                va="bottom", zorder=50)
        casual, registered = hour["observed"]
        ax.text(anchor[0] + side * 0.25, anchor[1] - 0.1, f"what happened: {casual:,} casual, {registered:,} registered",
                fontproperties=k.F.small, color=k.MUTED, ha=ha, va="top", zorder=50)
        oi = int(np.clip(np.floor(_cell(casual, edges[0])), 0, CELLS - 1))
        oj = int(np.clip(CELLS - 1 - np.floor(_cell(registered, edges[1])), 0, CELLS - 1))
        dot = k.iso(oi + 0.5, oj + 0.5, heights[oi, oj])
        ax.scatter([dot[0]], [dot[1]], s=46, color=k.INK, edgecolor=k.CARD, lw=1.2, zorder=60)

    # The evidence, in the right-hand column: the total's 80% interval over the benchmark's splits.
    splits = [data["riders"][str(seed)] for seed in range(3) if str(seed) in data["riders"]]

    def mean(model, key):
        return float(np.mean([split[model][key] for split in splits]))

    inch = fig.dpi_scale_trans
    labels = [label for _model, label in EVIDENCE]
    label_width = max(k.text_width(label, k.F.label) for label in labels) + 0.3
    value_width = k.text_width("100%", k.F.value) + 0.35
    x0 = box.width - box.right - label_width - 2 * value_width
    top = box.height - box.top - 0.3
    fig.text(x0, top, "Bikes out in the hour, casual plus registered", fontproperties=k.F.panel, color=k.INK,
             ha="left", va="top", transform=inch)
    fig.text(x0, top - 0.38, f"80% interval on {splits[0]['test hours']:,} held-out hours, mean of {len(splits)} splits",
             fontproperties=k.F.small, color=k.MUTED, ha="left", va="top", transform=inch)
    y = top - 0.98
    for x, text in ((x0 + label_width + value_width, "holds"), (x0 + label_width + 2 * value_width, "width")):
        fig.text(x, y, text, fontproperties=k.F.small, color=k.MUTED, ha="right", va="center", transform=inch)
    for r, (model, label) in enumerate(EVIDENCE):
        y = top - 1.4 - 0.4 * r
        color = k.TEAL_DARK if r == 0 else k.MUTED
        fig.text(x0, y, label, fontproperties=k.F.label, color=k.INK if r == 0 else k.MUTED, ha="left", va="center",
                 transform=inch)
        fig.text(x0 + label_width + value_width, y, k.pct(mean(model, "total_coverage_80"), 0),
                 fontproperties=k.F.value, color=color, ha="right", va="center", transform=inch)
        fig.text(x0 + label_width + 2 * value_width, y, k.num(mean(model, "total_width_80")), fontproperties=k.F.value,
                 color=color, ha="right", va="center", transform=inch)
    note = ("One correlation fitted after the fact holds as often: the gain is from modelling the dependence at all. "
            "LightGBM's quantiles of the total hold too few hours.")
    fig.text(x0, box.bottom + 0.55, k.wrap(note, k.F.small, label_width + 2 * value_width),
             fontproperties=k.F.small, color=k.MUTED, ha="left", va="bottom", transform=inch, linespacing=1.5)
    k.save(fig, NAME)

    # The gallery's thumbnail: the same floor at half the resolution, and the first row of the evidence.
    coarse = {name: m.reshape(CELLS // 2, 2, CELLS // 2, 2).sum(axis=(1, 3)) for name, m in mass.items()}
    small, small_colors = _floor(coarse)
    k.thumb("outcomes", "Joint outcomes", [
        (k.pct(mean("joint", "total_coverage_80"), 0), k.TEAL),
        (" inside an 80% band, against ", k.MUTED),
        (k.pct(mean("LightGBM quantiles of the total", "total_coverage_80"), 0), k.CLAY),
    ], lambda ax: _thumb_mark(ax, k.stacks(small, THUMB_COURSES), small_colors))


def _thumb_mark(ax, heights, colors):
    """The blocks, with the axes cut to the floor and the tallest stack so that the mark fills the thumbnail."""
    k.isometric_blocks(ax, heights, colors, lw=0.6)
    g = heights.shape[0]
    tops = [k.iso(i, j, heights[i, j])[1] for i in range(g) for j in range(g) if heights[i, j] > 0]
    ax.set_xlim(-g * 0.866 - 0.2, g * 0.866 + 0.2)
    ax.set_ylim(-g - 0.2, max(tops) + 0.2)


if __name__ == "__main__":
    k.main(compute=compute, draw=draw)
