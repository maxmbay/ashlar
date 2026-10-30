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


FARE_HEADS = (
    ("gaussian", "gaussian"),
    ("mixture", "mixture"),
    ("flow", "flow"),
    ("histogram", "histogram, 48 bins"),
)
FARE_BINS = 48
FLAT_FARE = 52.0  # the JFK flat fare, a spike no smooth head can grow on its own
# Three conditions with different shapes, and a different head wins each one.
FARE_CONDITIONS = (
    ("Manhattan", "Manhattan", "Manhattan to Manhattan"),
    ("Brooklyn", "Brooklyn", "Brooklyn to Brooklyn"),
    ("Queens", "Manhattan", "Queens to Manhattan"),
)


def fare_rows():
    """The taxi table with every column that gives the fare away removed.

    Distance, tolls, tip, total, and payment are dropped, so the fare has to
    come from where and when the trip ran.
    """
    import pandas as pd

    frame = pd.read_csv(bench._path("taxis.csv"), parse_dates=["pickup"])
    frame = frame.dropna(subset=["pickup_zone", "dropoff_zone"]).reset_index(drop=True)
    pickup = frame["pickup"]
    x = pd.DataFrame(
        {
            "hour": pickup.dt.hour.astype(np.float64),
            "weekday": pickup.dt.dayofweek.astype(np.float64),
            "passengers": frame["passengers"].astype(np.float64),
            "color": frame["color"],
            "pickup_zone": frame["pickup_zone"],
            "dropoff_zone": frame["dropoff_zone"],
            "pickup_borough": frame["pickup_borough"],
            "dropoff_borough": frame["dropoff_borough"],
        }
    )
    return bench._strings(x), frame["fare"].to_numpy(np.float64)


def fare_heads() -> None:
    """One fit per head, then the predictive density under three different conditions."""
    x, y = fare_rows()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
    mean, std = float(np.mean(y_train)), float(np.std(y_train))
    grid = np.linspace(0.0, 70.0, 561)

    models = {}
    for kind, _label in FARE_HEADS:
        overrides = {"head_kwargs": {"bins": FARE_BINS}} if kind == "histogram" else {}
        models[kind] = bench.fit_distribution(kind, x_train, (y_train - mean) / std, 0, **overrides)

    groups = []
    for pickup, dropoff, label in FARE_CONDITIONS:
        mask = ((x_test["pickup_borough"] == pickup) & (x_test["dropoff_borough"] == dropoff)).to_numpy()
        groups.append((label, x_test[mask], y_test[mask]))

    # Mean held-out log density per condition, on the scale of the fare. The best head per row is lit.
    likelihoods = []
    for label, x_group, y_group in groups:
        row = {
            kind: float(np.mean(np.asarray(models[kind].log_prob(x_group, (y_group - mean) / std)) - np.log(std)))
            for kind, _label in FARE_HEADS
        }
        likelihoods.append(row)
        print(f"{label} rows {len(y_group)} " + " ".join(f"{k} {v:+.3f}" for k, v in row.items()), flush=True)

    width, height = 16.4, 12.2
    fig = plt.figure(figsize=(width, height), facecolor=style.WHITE)
    fig.text(
        0.055,
        0.982,
        "Predictive distributions for taxi fare, three conditions and four heads.",
        fontproperties=style.TITLE,
        color=style.INK,
        ha="left",
        va="top",
    )
    for offset, line in (
        (0.947, "Taxi fare given pickup zone, dropoff zone, hour, weekday, passengers, and cab color. Distance, tolls, and the total are not features."),
        (0.918, "Each panel is the average predictive density over the held-out trips of that condition, with their fares behind it. A different head wins each row."),
    ):
        fig.text(0.055, offset, line, fontproperties=style.CLAIM, color=style.MUTED, ha="left", va="top")
    fig.add_artist(
        plt.Line2D([0.055, 0.945], [0.893, 0.893], transform=fig.transFigure, color=style.HAIR, linewidth=1.05)
    )

    bins = np.arange(0.0, 70.5, 2.0)
    rows = len(groups)
    for row, (label, x_group, y_group) in enumerate(groups):
        best = max(likelihoods[row], key=likelihoods[row].get)
        for column, (kind, head_label) in enumerate(FARE_HEADS):
            ax = fig.add_axes([0.055 + column * 0.228, 0.075 + (rows - 1 - row) * 0.268, 0.195, 0.195])
            style._style_plot(ax)
            density = _average_density(kind, models[kind], x_group, grid, mean, std)
            ax.hist(y_group, bins=bins, density=True, color=style.HAIR, edgecolor=style.WHITE, linewidth=0.6, zorder=1)
            lit = kind == best
            ax.plot(grid, density, color=style.TEAL if lit else style.MUTED, linewidth=2.3 if lit else 1.5,
                    alpha=1.0 if lit else 0.75, zorder=3)
            ax.set_xlim(0, 70)
            ax.set_ylim(0, 0.2)
            ax.set_xticks([0, 20, 40, 60])
            ax.set_yticks([])
            style._ticks(ax)
            if row == 0:
                ax.text(0.0, 1.1, head_label, transform=ax.transAxes, fontproperties=style.AXIS, color=style.INK,
                        ha="left", va="bottom")
            if column == 0:
                ax.text(-0.085, 0.5, label, transform=ax.transAxes, fontproperties=style.AXIS, color=style.INK,
                        ha="center", va="center", rotation=90)
            if row == rows - 1:
                ax.set_xlabel("fare, dollars", fontproperties=style.AXIS, color=style.INK, labelpad=9)
            ax.text(
                0.98,
                0.93,
                f"log density {likelihoods[row][kind]:+.2f}" + ("  ·  best" if lit else ""),
                transform=ax.transAxes,
                fontproperties=style.TICK,
                color=style.TEAL if lit else style.MUTED,
                ha="right",
                va="top",
            )
    flat = {
        kind: float(np.mean(np.abs(models[kind].sample(groups[-1][1], n=200, seed=0) * std + mean - FLAT_FARE) <= 2.0))
        for kind, _label in FARE_HEADS
    }
    actual = float(np.mean(np.abs(groups[-1][2] - FLAT_FARE) <= 2.0))
    print(f"actual_flat_share {actual:.4f} heads " + " ".join(f"{k} {v:.3f}" for k, v in flat.items()), flush=True)
    style._save(fig, "fare-heads.png")


FAN_LOOKBACK = 60
FAN_HORIZON = 21


def forecast_fan() -> None:
    """A 21-day fan from one origin, beside the 80% band's coverage over every test origin."""
    import pandas as pd

    from ashlar import ASRForecaster

    values = pd.read_csv(bench._path("temperatures.csv"))["Temp"].to_numpy(np.float64)
    cut = int(len(values) * 0.8)
    model = ASRForecaster(lookback=FAN_LOOKBACK, horizon=FAN_HORIZON, head="mixture", random_state=0)
    model.fit(values[:cut])

    origins = np.arange(cut, len(values) - FAN_HORIZON)
    histories = np.stack([values[origin - FAN_LOOKBACK : origin] for origin in origins])
    truths = np.stack([values[origin : origin + FAN_HORIZON] for origin in origins])
    draws = np.asarray(model.sample(histories, 0, n=400))
    low, high = np.percentile(draws, [10, 90], axis=0)
    covered = float(np.mean((truths >= low) & (truths <= high)))
    width = float(np.mean(high - low))
    print(f"coverage_80 {covered:.4f} mean_width {width:.3f}", flush=True)

    pick = int(np.argmax(origins >= cut + 400))
    fan = draws[:, pick, :]
    days = np.arange(1, FAN_HORIZON + 1)
    fig, ax = style._chart(
        f"The 80% band holds {covered:.0%} of the days it forecast.",
        "Melbourne daily minimum temperature, 21 days ahead from one origin. Bands are 400 draws from the mixture head.",
        height=7.6,
    )
    past = np.arange(-21, 1)
    ax.plot(past, values[origins[pick] - 21 : origins[pick] + 1], color=style.INK, linewidth=1.8, zorder=4)
    # The fan starts at the last observed day, so the bands leave the history rather than float beside it.
    last = float(values[origins[pick]])
    joined = np.concatenate([[0], days])
    for lo, hi, alpha in ((10, 90, 0.16), (25, 75, 0.26)):
        band_low, band_high = np.percentile(fan, [lo, hi], axis=0)
        ax.fill_between(
            joined,
            np.concatenate([[last], band_low]),
            np.concatenate([[last], band_high]),
            color=style.TEAL,
            alpha=alpha,
            linewidth=0,
            zorder=2,
        )
    ax.plot(joined, np.concatenate([[last], np.median(fan, axis=0)]), color=style.TEAL, linewidth=2.2, zorder=5)
    ax.plot(joined, np.concatenate([[last], truths[pick]]), color=style.CLAY, linewidth=1.8, zorder=6)
    style._callout(ax, (days[-1], truths[pick][-1]), -26, 22, "what happened", ha="right")
    style._callout(ax, (days[-1], float(np.median(fan, axis=0)[-1])), -26, -24, "median forecast", ha="right")
    ax.set_xlabel("days ahead", fontproperties=style.AXIS, color=style.INK, labelpad=12)
    ax.set_ylabel("°C", fontproperties=style.AXIS, color=style.INK, labelpad=12)
    ax.axvline(0, color=style.HAIR, linewidth=1.2, zorder=1)
    style._save(fig, "forecast-fan.png")


TIP_TREE = {"no_tip": None, "tip": ["under_two", "two_or_more"]}


def tip_labels(frame) -> np.ndarray:
    """Nested outcome: whether a tip happened, and if it did, how big it was."""
    tip = frame["tip"].to_numpy(np.float64)
    labels = np.where(tip <= 0.0, "no_tip", np.where(tip < 2.0, "under_two", "two_or_more"))
    return labels.astype(object)


def outcome_embeddings() -> None:
    """The zone becomes a vector, and what that vector predicts depends on the fare beside it."""
    import pandas as pd

    from ashlar import ASROutcomeTree

    frame = pd.read_csv(bench._path("taxis.csv"), parse_dates=["pickup"])
    frame = frame.dropna(subset=["pickup_zone", "dropoff_zone"]).reset_index(drop=True)
    pickup = frame["pickup"]
    x = pd.DataFrame(
        {
            "hour": pickup.dt.hour.astype(np.float64),
            "weekday": pickup.dt.dayofweek.astype(np.float64),
            "passengers": frame["passengers"].astype(np.float64),
            "distance": frame["distance"].astype(np.float64),
            "fare": frame["fare"].astype(np.float64),
            "color": frame["color"],
            "pickup_zone": frame["pickup_zone"],
            "dropoff_zone": frame["dropoff_zone"],
            "pickup_borough": frame["pickup_borough"],
            "dropoff_borough": frame["dropoff_borough"],
        }
    )
    x = bench._strings(x)
    y = tip_labels(frame)

    model = ASROutcomeTree(TIP_TREE, random_state=0).fit(x, y)
    tip_column = list(model.conditional_names(None)).index("tip")
    names = [column.name for column in model.encoder_.embedding_columns]
    index = names.index("pickup_zone")
    column = model.encoder_.embedding_columns[index]
    table = np.asarray(model.core_.categorical_embeddings.tables[index], dtype=np.float64)[1:]

    counts = x["pickup_zone"].value_counts()
    keep = [i for i, name in enumerate(column.categories) if counts.get(name, 0) >= 25]
    vectors = table[keep]
    labels = [column.categories[i] for i in keep]
    centered = vectors - vectors.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    plane = centered @ vh[:2].T

    # P(tip | zone, fare): one zone vector read beside a sweep of fares, everything else fixed.
    fares = np.linspace(4.0, 60.0, 57)

    def curve(zone: str) -> np.ndarray:
        rows = {
            "hour": np.full(len(fares), 18.0),
            "weekday": np.full(len(fares), 2.0),
            "passengers": np.full(len(fares), 1.0),
            "distance": np.full(len(fares), float(np.median(x["distance"]))),
            "fare": fares,
            "color": np.array(["yellow"] * len(fares), dtype=object),
            "pickup_zone": np.array([zone] * len(fares), dtype=object),
            "dropoff_zone": np.array(["Midtown Center"] * len(fares), dtype=object),
            "pickup_borough": np.array(["Manhattan"] * len(fares), dtype=object),
            "dropoff_borough": np.array(["Manhattan"] * len(fares), dtype=object),
        }
        return np.asarray(model.predict_conditional(rows, None))[:, tip_column]

    curves = {name: curve(name) for name in labels}
    swing = {name: float(line.max() - line.min()) for name, line in curves.items()}
    order = sorted(labels, key=lambda name: swing[name])
    tracked = [order[-1], order[-2], order[1], order[0]]
    for name in tracked:
        print(f"{name} P(tip) {curves[name][0]:.3f} at $4, {curves[name][-1]:.3f} at $60, swing {swing[name]:.3f}", flush=True)

    share = {name: float(np.mean(y[(x["pickup_zone"] == name).to_numpy()] != "no_tip")) for name in labels}
    color = np.array([share[name] for name in labels])

    fig = plt.figure(figsize=(16.4, 7.8), facecolor=style.WHITE)
    fig.text(
        0.055,
        0.968,
        "Learned pickup-zone vectors, and the fare response they produce.",
        fontproperties=style.TITLE,
        color=style.INK,
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.908,
        "An outcome tree over taxi tips: no tip, or a tip under or over $2. The zone enters as an 8-d vector,",
        fontproperties=style.CLAIM,
        color=style.MUTED,
        ha="left",
        va="top",
    )
    fig.text(
        0.055,
        0.868,
        "and what that vector does to P(tip) depends on the fare it is read beside.",
        fontproperties=style.CLAIM,
        color=style.MUTED,
        ha="left",
        va="top",
    )
    fig.add_artist(plt.Line2D([0.055, 0.945], [0.828, 0.828], transform=fig.transFigure, color=style.HAIR, linewidth=1.05))

    left = fig.add_axes([0.075, 0.12, 0.30, 0.62])
    style._style_plot(left)
    sizes = 22 + 3 * np.sqrt(np.array([counts[name] for name in labels]))
    points = left.scatter(plane[:, 0], plane[:, 1], c=color, cmap=style.EFFECT, s=sizes,
                          edgecolors=style.WHITE, linewidths=1.0, zorder=3)
    # Park each label in the nearest free corner, with a leader to its point, so four
    # callouts on a crowded cloud never land on each other.
    corners = [(0.02, 0.97, "left", "top"), (0.98, 0.97, "right", "top"),
               (0.02, 0.03, "left", "bottom"), (0.98, 0.03, "right", "bottom")]
    taken = set()
    for name in tracked:
        i = labels.index(name)
        point = left.transAxes.inverted().transform(left.transData.transform(plane[i]))
        choice = min(
            (c for c in range(len(corners)) if c not in taken),
            key=lambda c: (corners[c][0] - point[0]) ** 2 + (corners[c][1] - point[1]) ** 2,
        )
        taken.add(choice)
        cx, cy, ha, va = corners[choice]
        left.annotate(
            name,
            xy=plane[i],
            xycoords="data",
            xytext=(cx, cy),
            textcoords="axes fraction",
            fontproperties=style.CALLOUT,
            color=style.INK,
            ha=ha,
            va=va,
            arrowprops={"arrowstyle": "-", "color": style.INK, "linewidth": 0.9,
                        "shrinkA": 2, "shrinkB": 4},
            zorder=6,
        )
    left.set_xticks([])
    left.set_yticks([])
    left.text(0.0, 1.04, "pickup-zone vectors, first two components", transform=left.transAxes,
              fontproperties=style.AXIS, color=style.INK, ha="left", va="bottom")
    bar = fig.colorbar(points, ax=left, fraction=0.035, pad=0.06)
    bar.set_label("share of trips tipped", fontproperties=style.TICK, color=style.MUTED)
    bar.outline.set_visible(False)
    for text in bar.ax.get_yticklabels():
        text.set_fontproperties(style.TICK)
        text.set_color(style.MUTED)

    right = fig.add_axes([0.52, 0.12, 0.25, 0.62])
    style._style_plot(right)
    palette = {name: tone for name, tone in zip(tracked, [style.CLAY, "#8a8178", style.TEAL, "#2f6f8f"], strict=True)}
    # Label slots, top to bottom in the order the curves end, so four labels never collide.
    slots = dict(zip(sorted(tracked, key=lambda name: -curves[name][-1]), (0.93, 0.70, 0.47, 0.24), strict=True))
    for name in tracked:
        tone = palette[name]
        right.plot(fares, curves[name], color=tone, linewidth=2.2, zorder=3)
        right.text(1.06, slots[name], f"{name}\n{swing[name]:+.2f} from \\$4 to \\$60", transform=right.transAxes,
                   fontproperties=style.TICK, color=tone, ha="left", va="center")
    right.set_xlim(4, 60)
    right.set_xticks([10, 30, 50])
    style._ticks(right)
    right.set_xlabel("fare, dollars", fontproperties=style.AXIS, color=style.INK, labelpad=10)
    right.text(0.0, 1.04, "P(tip | zone, fare), other columns fixed", transform=right.transAxes,
               fontproperties=style.AXIS, color=style.INK, ha="left", va="bottom")
    style._save(fig, "outcome-embeddings.png")


def main() -> None:
    style._load_fonts()
    which = sys.argv[1:] or ["tips", "zones"]
    if "tips" in which:
        tip_heads()
    if "zones" in which:
        taxi_zones()
    if "fan" in which:
        forecast_fan()
    if "fares" in which:
        fare_heads()
    if "embeddings" in which:
        outcome_embeddings()


if __name__ == "__main__":
    main()
