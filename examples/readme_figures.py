"""Draw the README figures that use public data.

``tip-heads.png``: the average predictive density of four heads on held-out
NYC taxi tips, over the tips themselves. ``taxi-zones.png``: the learned
pickup-zone embedding, projected to two dimensions and colored by the
median trip length from that zone.
All reuse the tables and the fit settings of ``benchmark_readme.py``.

    JAX_PLATFORMS=cpu python examples/readme_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_readme as bench
import figures as style
from ashlar import ASRModel

HEADS = (
    ("gaussian", "Gaussian"),
    ("mixture", "Mixture of five normals"),
    ("flow", "Spline flow"),
    ("histogram", "Histogram, 16 bins"),
)


def _average_density(kind, model, x_test, grid, mean, std):
    """Mean predictive density over the held-out rows at each grid value, on the scale of the target."""
    rows = len(x_test)
    return np.asarray(
        [np.mean(np.exp(model.log_prob(x_test, np.full(rows, (value - mean) / std)))) / std for value in grid]
    )


def tip_heads() -> None:
    x, y = bench.taxis("tip")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    mean, std = float(np.mean(y_train)), float(np.std(y_train))
    grid = np.linspace(-4.0, 10.0, 561)
    zero_share = float(np.mean(y_test == 0))

    width, height = 13.8, 8.4
    fig = plt.figure(figsize=(width, height), facecolor=style.WHITE)
    fig.text(0.07, 0.965, f"{zero_share:.0%} of taxi tips are exactly zero.", fontproperties=style.TITLE, color=style.INK, ha="left", va="top")
    fig.text(
        0.07,
        0.895,
        "Average predictive density on held-out trips, payment type hidden. The bars are the held-out tips.",
        fontproperties=style.CLAIM,
        color=style.MUTED,
        ha="left",
        va="top",
    )
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.855, 0.855], transform=fig.transFigure, color=style.HAIR, linewidth=1.05))
    bins = np.arange(-4.0, 10.25, 0.25)
    for index, (kind, label) in enumerate(HEADS):
        row, col = divmod(index, 2)
        ax = fig.add_axes([0.085 + col * 0.45, 0.09 + (1 - row) * 0.39, 0.39, 0.27])
        style._style_plot(ax)
        model = bench.fit_distribution(kind, x_train, (y_train - mean) / std, 0)
        density = _average_density(kind, model, x_test, grid, mean, std)
        draws = model.sample(x_test, n=200, seed=0) * std + mean
        below = float(np.mean(np.abs(draws) <= 0.25))
        ax.hist(y_test, bins=bins, density=True, color=style.HAIR, edgecolor=style.WHITE, linewidth=0.6, zorder=1)
        color = style.CLAY if kind == "gaussian" else style.TEAL
        ax.plot(grid, density, color=color, linewidth=2.2, zorder=3)
        ax.set_xlim(-4, 10)
        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        ax.set_xticks([-4, -2, 0, 2, 4, 6, 8, 10])
        style._ticks(ax)
        ax.text(0.0, 1.06, label, transform=ax.transAxes, fontproperties=style.AXIS, color=style.INK, ha="left", va="bottom")
        ax.text(
            0.99,
            0.93,
            f"{below:.0%} of the mass within 25¢ of zero",
            transform=ax.transAxes,
            fontproperties=style.TICK,
            color=style.MUTED,
            ha="right",
            va="top",
        )
        if row == 1:
            ax.set_xlabel("tip, dollars", fontproperties=style.AXIS, color=style.INK, labelpad=10)
        print(kind, f"mass_near_zero {below:.4f}", flush=True)
    print(f"zero_share {zero_share:.4f}", flush=True)
    style._save(fig, "tip-heads.png")


def taxi_zones() -> None:
    x, y = bench.taxis("duration")
    model = ASRModel(random_state=0).fit(x, y)
    names = [column.name for column in model.encoder_.embedding_columns]
    column = model.encoder_.embedding_columns[names.index("pickup_zone")]
    table = np.asarray(model.core_.categorical_embeddings.tables[names.index("pickup_zone")], dtype=np.float64)[1:]
    counts = x["pickup_zone"].value_counts()
    keep = [i for i, name in enumerate(column.categories) if counts.get(name, 0) >= 5]
    vectors = table[keep]
    labels = [column.categories[i] for i in keep]
    centered = vectors - vectors.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vh[:2].T

    minutes = np.exp(y)
    typical = {name: float(np.median(minutes[(x["pickup_zone"] == name).to_numpy()])) for name in labels}
    color = np.log(np.array([typical[name] for name in labels]))
    if np.corrcoef(plane[:, 0], color)[0, 1] < 0:
        plane[:, 0] = -plane[:, 0]
    print(f"pc1_vs_log_median_minutes {float(np.corrcoef(plane[:, 0], color)[0, 1]):.3f}", flush=True)

    fig, ax = style._chart(
        "The airports stand apart.",
        "Learned 8-d pickup-zone vectors, first two principal components. Color is the median trip from that zone.",
        height=8.0,
    )
    sizes = 24 + 3 * np.sqrt(np.array([counts[name] for name in labels]))
    points = ax.scatter(plane[:, 0], plane[:, 1], c=color, cmap=style.EFFECT, s=sizes,
                        edgecolors=style.WHITE, linewidths=1.0, zorder=3)
    for target, (dx, dy) in {
        "JFK Airport": (-20, 26),
        "LaGuardia Airport": (-20, 26),
        "Times Sq/Theatre District": (18, 18),
        "Upper East Side North": (18, -18),
    }.items():
        if target in labels:
            i = labels.index(target)
            style._callout(ax, (plane[i, 0], plane[i, 1]), dx, dy, f"{target}, {typical[target]:.0f} min",
                           ha="right" if dx < 0 else "left")
    bar = fig.colorbar(points, ax=ax, fraction=0.025, pad=0.02)
    ticks = [t for t in (5, 7, 10, 15, 20, 30, 40) if color.min() <= np.log(t) <= color.max()]
    bar.set_ticks(np.log(ticks))
    bar.set_ticklabels([f"{t} min" for t in ticks])
    bar.outline.set_visible(False)
    for label in bar.ax.get_yticklabels():
        label.set_fontproperties(style.TICK)
        label.set_color(style.MUTED)
    ax.set_xticks([])
    ax.set_yticks([])
    print(f"zones {len(labels)}", flush=True)
    style._save(fig, "taxi-zones.png")


def main() -> None:
    style._load_fonts()
    which = sys.argv[1:] or ["tips", "zones"]
    if "tips" in which:
        tip_heads()
    if "zones" in which:
        taxi_zones()


if __name__ == "__main__":
    main()
