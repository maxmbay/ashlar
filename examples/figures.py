"""Generate the README figures and the path diagrams.

CPU, small series, a few epochs. Writes PNGs under docs/images/.
Does not fit a benchmark and does not publish the package.

    JAX_PLATFORMS=cpu python examples/figures.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

from ashlar import ASRModel, ASRForecaster, ASROrdinalClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
# Inter TTFs. Set ASHLAR_FONT_DIR to a folder holding Inter-Regular.ttf and friends.
FONT_DIR = Path(os.environ.get("ASHLAR_FONT_DIR", "/usr/share/fonts/truetype/macos"))

WHITE = "#ffffff"
INK = "#1a1916"
MUTED = "#5e5a54"
HAIR = "#e4e0da"
TEAL = "#0f6e68"
CLAY = "#d4471c"

EFFECT = LinearSegmentedColormap.from_list(
    "ashlar_effect", [TEAL, "#8a8178", CLAY]
)

TITLE = CLAIM = AXIS = TICK = CALLOUT = None
NODE = NODE_SUB = EDGE = RANK = None


def _load_fonts() -> None:
    global TITLE, CLAIM, AXIS, TICK, CALLOUT, NODE, NODE_SUB, EDGE, RANK
    files = {name: FONT_DIR / f"Inter-{name}.ttf" for name in ("Regular", "Medium", "SemiBold", "Bold")}
    for path in files.values():
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    def face(name: str, size: float) -> FontProperties:
        path = files[name]
        if path.exists():
            return FontProperties(fname=str(path), size=size)
        return FontProperties(family="Inter", size=size)

    regular = files["Regular"]
    TITLE = face("Bold", 26)
    CLAIM = face("Regular", 15.5)
    AXIS = face("SemiBold", 15.5)
    TICK = face("Regular", 13.5)
    CALLOUT = face("SemiBold", 14)
    NODE = face("SemiBold", 13)
    NODE_SUB = face("Regular", 11.5)
    EDGE = face("Medium", 12)
    RANK = face("Medium", 12.5)
    family = font_manager.FontProperties(fname=str(regular)).get_name() if regular.exists() else "Inter"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 13,
            "text.color": INK,
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "axes.linewidth": 0,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
        }
    )


def _style_plot(ax) -> None:
    ax.set_facecolor(WHITE)
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.15)
    ax.tick_params(axis="both", which="both", length=0, pad=10, colors=MUTED, labelsize=13.5)


def _ticks(ax) -> None:
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontproperties(TICK)
        label.set_color(MUTED)


def _names(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel, fontproperties=AXIS, color=INK, labelpad=14)
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.03,
        ylabel,
        transform=ax.transAxes,
        fontproperties=AXIS,
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )


def _chart(title: str, claim: str, width: float = 13.8, height: float = 7.4):
    fig = plt.figure(figsize=(width, height), facecolor=WHITE)
    fig.text(0.07, 0.965, title, fontproperties=TITLE, color=INK, ha="left", va="top")
    fig.text(0.07, 0.875, claim, fontproperties=CLAIM, color=MUTED, ha="left", va="top")
    fig.add_artist(
        plt.Line2D(
            [0.07, 0.93],
            [0.828, 0.828],
            transform=fig.transFigure,
            color=HAIR,
            linewidth=1.05,
            clip_on=False,
        )
    )
    ax = fig.add_axes([0.085, 0.145, 0.85, 0.60])
    _style_plot(ax)
    return fig, ax


def _callout(ax, xy, dx, dy, text, color=INK, ha="left", va="center"):
    ax.annotate(
        text,
        xy=xy,
        xytext=(dx, dy),
        textcoords="offset points",
        fontproperties=CALLOUT,
        color=color,
        ha=ha,
        va=va,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 1.15, "shrinkA": 0, "shrinkB": 2},
        zorder=6,
        annotation_clip=False,
        clip_on=False,
    )


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=210, facecolor=WHITE)
    plt.close(fig)
    print(f"wrote {path}", flush=True)


def _diagram(title: str, claim: str, width: float, height: float):
    """Header in inches, so a taller figure does not pull the claim into the title."""
    fig = plt.figure(figsize=(width, height), facecolor=WHITE)
    fig.text(0.04, 1 - 0.28 / height, title, fontproperties=TITLE, color=INK, ha="left", va="top")
    fig.text(0.04, 1 - 0.78 / height, claim, fontproperties=CLAIM, color=MUTED, ha="left", va="top")
    hair = 1 - 1.22 / height
    fig.add_artist(
        plt.Line2D([0.04, 0.96], [hair, hair], transform=fig.transFigure, color=HAIR, linewidth=1.05)
    )
    left, right, top, bottom = 0.46, 0.40, 1.68, 0.36
    ax_w = width - left - right
    ax_h = height - top - bottom
    ax = fig.add_axes([left / width, bottom / height, ax_w / width, ax_h / height])
    ax.set_xlim(0, ax_w)
    ax.set_ylim(0, ax_h)
    ax.set_axis_off()
    ax.set_facecolor(WHITE)
    return fig, ax


def _node(ax, cx, cy, w, h, title, subtitle=None, rail=None):
    x = cx - w / 2
    y = cy - h / 2
    if rail == TEAL:
        fill = "#f2faf8"
    elif rail == CLAY:
        fill = "#fdf5f1"
    else:
        fill = WHITE
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.09",
            facecolor=fill,
            edgecolor=INK,
            linewidth=1.4,
            zorder=4,
            mutation_aspect=None,
        )
    )
    if rail is not None:
        ax.add_patch(
            Rectangle(
                (x + 0.08, y + 0.12),
                0.065,
                h - 0.24,
                facecolor=rail,
                edgecolor="none",
                zorder=5,
            )
        )
    nudge = 0.05 if rail is not None else 0.0
    if subtitle:
        ax.text(cx + nudge, cy + 0.13, title, fontproperties=NODE, color=INK, ha="center", va="center", zorder=6, clip_on=False)
        ax.text(
            cx + nudge,
            cy - 0.15,
            subtitle,
            fontproperties=NODE_SUB,
            color=MUTED,
            ha="center",
            va="center",
            zorder=6,
            clip_on=False,
        )
    else:
        ax.text(cx + nudge, cy, title, fontproperties=NODE, color=INK, ha="center", va="center", zorder=6, clip_on=False)
    return {
        "left": np.array([x, cy]),
        "right": np.array([x + w, cy]),
        "top": np.array([cx, y + h]),
        "bottom": np.array([cx, y]),
        "cx": cx,
        "cy": cy,
    }


def _stroke(ax, points) -> None:
    pts = [np.asarray(p, dtype=float) for p in points]
    ax.plot(
        [p[0] for p in pts],
        [p[1] for p in pts],
        color=INK,
        linewidth=1.75,
        solid_capstyle="butt",
        solid_joinstyle="miter",
        zorder=2,
    )


def _arrow(ax, points) -> None:
    pts = [np.asarray(p, dtype=float) for p in points]
    tip = pts[-1]
    direction = tip - pts[-2]
    length = float(np.linalg.norm(direction))
    if length < 1e-6:
        return
    direction = direction / length
    head = 0.18
    wing = 0.075
    end = tip - direction * head
    _stroke(ax, pts[:-1] + [end])
    perp = np.array([-direction[1], direction[0]])
    base = tip - direction * head
    ax.add_patch(
        Polygon(
            np.vstack([tip, base + perp * wing, base - perp * wing]),
            closed=True,
            facecolor=INK,
            edgecolor="none",
            zorder=5,
        )
    )


def _tag(ax, x, y, text, ha="center", va="center") -> None:
    ax.text(x, y, text, fontproperties=EDGE, color=INK, ha=ha, va=va, zorder=6, clip_on=False)


def _rank(ax, x, y, text) -> None:
    ax.text(x, y, text, fontproperties=RANK, color=MUTED, ha="left", va="bottom", zorder=6, clip_on=False)


def tabular_fit() -> None:
    rng = np.random.default_rng(0)
    n = 180
    x = rng.uniform(-2.8, 2.8, size=n).astype(np.float32)
    y = np.sin(1.35 * x) + 0.28 * x
    y = (y + rng.normal(0.0, 0.16, size=n)).astype(np.float32)
    model = ASRModel(
        max_epochs=24,
        patience=8,
        batch_size=32,
        learning_rate=6e-3,
        backbone_kwargs={"n_blocks": 2, "d_block": 48, "dropout": 0.0},
        validation_size=0.2,
        random_state=0,
    )
    model.fit({"x": x}, y)
    grid = np.linspace(-2.8, 2.8, 240, dtype=np.float32)
    predicted = np.asarray(model.predict({"x": grid}), dtype=np.float64)
    truth = np.sin(1.35 * grid) + 0.28 * grid
    print(f"tabular curve mae {float(np.mean(np.abs(predicted - truth))):.4f}", flush=True)

    fig, ax = _chart(
        "The fitted line follows the rows.",
        "The clay line is the prediction. It keeps the shape and leaves the noise.",
    )
    ax.scatter(x, y, s=36, color=TEAL, alpha=0.9, linewidths=0, zorder=2)
    ax.plot(grid, predicted, color=CLAY, linewidth=2.8, solid_capstyle="round", zorder=3)
    peak = int(np.argmax(predicted))
    _callout(ax, (grid[peak], predicted[peak]), 16, 18, "prediction", CLAY, ha="left", va="bottom")
    trough = int(np.argmin(predicted))
    nearby = np.argmin(np.abs(x - grid[trough]))
    _callout(ax, (x[nearby], y[nearby]), -8, -28, "rows", TEAL, ha="right", va="top")
    ax.set_xlim(-3.15, 3.35)
    low = min(float(y.min()), float(predicted.min()))
    high = max(float(y.max()), float(predicted.max()))
    pad = 0.22 * (high - low)
    ax.set_ylim(low - pad * 0.85, high + pad * 1.15)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-1, 0, 1])
    _names(ax, "input", "target")
    _ticks(ax)
    _save(fig, "tabular-fit.png")


def categorical_embeddings() -> None:
    effect = {
        "harbor": 1.7,
        "wharf": 1.45,
        "pier": 1.2,
        "plaza": 0.45,
        "square": 0.2,
        "court": -0.05,
        "mill": -0.4,
        "works": -0.65,
        "grove": -1.25,
        "field": -1.5,
        "meadow": -1.75,
    }
    names = list(effect)
    rng = np.random.default_rng(1)
    per = 40
    city = np.repeat(np.array(names, dtype=object), per)
    rooms = rng.normal(0.0, 1.0, size=len(city)).astype(np.float32)
    signal = np.array([effect[item] for item in city], dtype=np.float32)
    y = (0.3 * rooms + signal + rng.normal(0.0, 0.12, size=len(city))).astype(np.float32)
    order = rng.permutation(len(city))
    model = ASRModel(
        max_epochs=36,
        patience=10,
        batch_size=48,
        learning_rate=1e-2,
        weight_decay=1e-5,
        categorical_embedding_dim=2,
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "dropout": 0.0},
        validation_size=0.15,
        random_state=0,
    )
    model.fit({"rooms": rooms[order], "city": city[order]}, y[order])
    column = model.encoder_.embedding_columns[0]
    table = np.asarray(model.core_.categorical_embeddings.tables[0], dtype=np.float64)
    vectors, effects, labels = [], [], []
    for index, name in enumerate(column.categories):
        vectors.append(table[index + 1])
        effects.append(effect[name])
        labels.append(name)
    vectors = np.vstack(vectors)
    effects = np.asarray(effects)
    centered = vectors - vectors.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    direction = vh[0]
    projection = centered @ direction
    if np.corrcoef(projection, effects)[0, 1] < 0:
        direction = -direction
        projection = -projection
    print(f"embedding effect correlation {float(np.corrcoef(projection, effects)[0, 1]):.3f}", flush=True)

    height = 7.7
    width = 13.8
    fig = plt.figure(figsize=(width, height), facecolor=WHITE)
    fig.text(0.07, 0.965, "Categories with the same effect sit together.", fontproperties=TITLE, color=INK, ha="left", va="top")
    fig.text(
        0.07,
        0.875,
        "Waterfront, civic, and open ground fall into three clusters.",
        fontproperties=CLAIM,
        color=MUTED,
        ha="left",
        va="top",
    )
    fig.add_artist(
        plt.Line2D([0.07, 0.93], [0.828, 0.828], transform=fig.transFigure, color=HAIR, linewidth=1.05)
    )
    side = 4.55
    left = 1.7
    bottom = 0.95
    ax = fig.add_axes([left / width, bottom / height, side / width, side / height])
    _style_plot(ax)
    ax.scatter(
        vectors[:, 0],
        vectors[:, 1],
        c=effects,
        cmap=EFFECT,
        s=170,
        zorder=3,
        edgecolors=WHITE,
        linewidths=1.6,
        vmin=-1.75,
        vmax=1.7,
    )
    by_name = {name: vectors[index] for index, name in enumerate(labels)}
    normal = np.array([-direction[1], direction[0]])
    along = {name: float(projection[index]) for index, name in enumerate(labels)}
    ordered = sorted(labels, key=along.get)
    for rank, name in enumerate(ordered):
        sign = 1 if rank % 2 == 0 else -1
        offset = sign * normal
        ha = "left" if offset[0] >= 0 else "right"
        va = "bottom" if offset[1] >= 0 else "top"
        ax.annotate(
            name,
            by_name[name],
            xytext=(15 * offset[0], 15 * offset[1]),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontproperties=TICK,
            color=INK,
            zorder=4,
            annotation_clip=False,
        )
    span = max(float(np.ptp(vectors[:, 0])), float(np.ptp(vectors[:, 1])), 0.4)
    group_color = {"waterfront": CLAY, "civic": "#6f655c", "open ground": TEAL}
    # Anchor each name on the points it actually describes, off the diagonal.
    placements = {
        "waterfront": (["harbor", "wharf", "pier"], direction * span * 0.34 + normal * span * 0.08),
        "civic": (["plaza", "square", "court"], normal * span * 0.48),
        "open ground": (["grove", "field", "meadow"], -direction * span * 0.32 - normal * span * 0.06),
    }
    for group, (members, shift) in placements.items():
        points = np.vstack([by_name[name] for name in members])
        centroid = points.mean(axis=0)
        text_at = centroid + shift
        ha = "left" if shift[0] >= 0 else "right"
        va = "bottom" if shift[1] >= 0 else "top"
        ax.annotate(
            group,
            centroid,
            xytext=text_at,
            textcoords="data",
            fontproperties=CALLOUT,
            color=group_color[group],
            ha=ha,
            va=va,
            arrowprops={"arrowstyle": "-", "color": group_color[group], "lw": 1.15, "shrinkA": 2, "shrinkB": 2},
            zorder=5,
            annotation_clip=False,
        )
    pad = span * 1.05
    center = vectors.mean(axis=0)
    ax.set_xlim(center[0] - pad, center[0] + pad)
    ax.set_ylim(center[1] - pad, center[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("embedding axis 1", fontproperties=AXIS, color=INK, labelpad=12)
    ax.set_ylabel("")
    ax.text(
        0.0,
        1.04,
        "embedding axis 2",
        transform=ax.transAxes,
        fontproperties=AXIS,
        color=INK,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("left", "bottom", "top", "right"):
        ax.spines[spine].set_visible(False)

    ramp_left = left + side + 0.85
    ramp_bottom = bottom + 0.35
    ramp_h = side - 0.7
    cax = fig.add_axes([ramp_left / width, ramp_bottom / height, 0.18 / width, ramp_h / height])
    cax.imshow(np.linspace(0, 1, 256).reshape(-1, 1), aspect="auto", cmap=EFFECT, origin="lower")
    cax.set_axis_off()
    fig.text(
        (ramp_left + 0.34) / width,
        (ramp_bottom + ramp_h + 0.08) / height,
        "raises the target",
        fontproperties=TICK,
        color=CLAY,
        ha="left",
        va="bottom",
    )
    fig.text(
        (ramp_left + 0.34) / width,
        (ramp_bottom - 0.08) / height,
        "lowers the target",
        fontproperties=TICK,
        color=TEAL,
        ha="left",
        va="top",
    )
    fig.text(
        (ramp_left - 0.08) / width,
        (ramp_bottom + ramp_h / 2) / height,
        "effect",
        fontproperties=AXIS,
        color=INK,
        ha="right",
        va="center",
        rotation=90,
    )
    _save(fig, "categorical-embeddings.png")


def ordinal_score() -> None:
    rng = np.random.default_rng(2)
    n = 420
    x = rng.uniform(-2.2, 2.2, size=(n, 1)).astype(np.float32)
    label = (np.digitize(x[:, 0], np.array([-1.3, -0.4, 0.4, 1.3])) + 1).astype(np.int32)
    model = ASROrdinalClassifier(
        max_epochs=28,
        patience=8,
        batch_size=48,
        learning_rate=2e-2,
        backbone_kwargs={"n_blocks": 2, "d_block": 32, "dropout": 0.0},
        validation_size=0.2,
        random_state=0,
    )
    model.fit(x, label)
    score = np.asarray(model.predict_score(x), dtype=np.float64)
    print(f"ordinal mae {float(np.mean(np.abs(score - label))):.4f}", flush=True)

    fig, ax = _chart(
        "The expected score stays with the integer label.",
        "Each clay mark is the mean expected score. It sits on that integer.",
    )
    jitter = rng.uniform(-0.12, 0.12, size=n)
    ax.plot([0.55, 5.45], [0.55, 5.45], color=INK, linewidth=1.15, linestyle=(0, (4, 3.2)), zorder=1)
    ax.scatter(label + jitter, score, s=22, color=TEAL, alpha=0.55, linewidths=0, zorder=2)
    classes = np.asarray(model.classes_, dtype=np.float64)
    means = np.array([score[label == int(value)].mean() for value in classes])
    ax.scatter(classes, means, s=150, color=CLAY, edgecolors=WHITE, linewidths=2.0, zorder=4)
    _callout(ax, (classes[3], means[3]), 18, 22, "mean expected score", CLAY, ha="left", va="bottom")
    _callout(ax, (5.35, 5.35), 10, 16, "same label", INK, ha="left", va="bottom")
    cloud = (label == 2)
    _callout(
        ax,
        (2.0, float(np.median(score[cloud]))),
        -36,
        -26,
        "rows",
        TEAL,
        ha="right",
        va="top",
    )
    ax.set_xlim(0.35, 6.15)
    ax.set_ylim(0.35, 6.15)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    _names(ax, "integer label", "expected score")
    _ticks(ax)
    _save(fig, "ordinal-score.png")


def synthetic_cycle(n: int, seed: int = 0) -> np.ndarray:
    step = np.arange(n, dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, 0.18, size=n)
    return 12.0 + 4.0 * np.sin(2.0 * np.pi * step / 24.0) + noise


def quantile_forecast() -> None:
    """One-step quantile forecasts rolled across a held-out cycle."""
    lookback = 48
    holdout = 24
    series = synthetic_cycle(lookback + 24 * 10 + holdout, seed=3)
    cut = len(series) - holdout
    model = ASRForecaster(
        lookback=lookback,
        horizon=1,
        head="quantile",
        max_epochs=28,
        patience=8,
        batch_size=64,
        random_state=0,
    )
    model.fit(series[:cut])
    origins = np.arange(cut, len(series))
    windows = np.stack([series[origin - lookback : origin] for origin in origins])
    point = np.asarray(model.predict(windows), dtype=np.float64)[:, 0]
    band = np.asarray(model.quantiles(windows), dtype=np.float64)[:, 0, :]
    future = series[cut:]
    print(
        f"forecast mae {float(np.mean(np.abs(point - future))):.3f} "
        f"coverage {float(np.mean((future >= band[:, 0]) & (future <= band[:, 2]))):.3f}",
        flush=True,
    )

    shown = 72
    hist_x = np.arange(-shown, 0)
    fut_x = np.arange(0, holdout)
    history = series[cut - shown : cut]
    fig, ax = _chart(
        "The band follows the next cycle.",
        "Rolled one step at a time. The clay band covers the held-out cycle, from 0.1 to 0.9.",
    )
    ax.fill_between(fut_x, band[:, 0], band[:, 2], color=to_rgba(CLAY, 0.16), linewidth=0, zorder=1)
    ax.plot(fut_x, band[:, 0], color=to_rgba(CLAY, 0.85), linewidth=1.05, solid_capstyle="round", zorder=2)
    ax.plot(fut_x, band[:, 2], color=to_rgba(CLAY, 0.85), linewidth=1.05, solid_capstyle="round", zorder=2)
    ax.plot(hist_x, history, color=TEAL, linewidth=2.5, solid_capstyle="round", zorder=3)
    ax.plot(fut_x, future, color=TEAL, linewidth=1.7, linestyle=(0, (3.4, 2.2)), solid_capstyle="round", zorder=4)
    ax.plot(fut_x, point, color=CLAY, linewidth=2.7, solid_capstyle="round", zorder=5)
    ax.plot([0, 0], [7.2, 18.6], color=HAIR, linewidth=1.05, zorder=0)
    low = min(float(history.min()), float(band[:, 0].min()), float(future.min()))
    high = max(float(history.max()), float(band[:, 2].max()), float(future.max()))
    span = high - low
    bracket = high + span * 0.10
    ax.plot([0, holdout - 1], [bracket, bracket], color=CLAY, linewidth=1.2, solid_capstyle="butt", zorder=6)
    ax.plot([0, 0], [bracket - span * 0.035, bracket], color=CLAY, linewidth=1.2, zorder=6)
    ax.plot([holdout - 1, holdout - 1], [bracket - span * 0.035, bracket], color=CLAY, linewidth=1.2, zorder=6)
    ax.text(
        (holdout - 1) / 2,
        bracket + span * 0.025,
        "covered cycle",
        fontproperties=CALLOUT,
        color=CLAY,
        ha="center",
        va="bottom",
        zorder=6,
    )
    crest = int(np.argmax(history))
    _callout(ax, (hist_x[crest], history[crest]), -70, 16, "history", TEAL, ha="right", va="bottom")
    _callout(ax, (fut_x[-1], point[-1]), 8, 12, "point forecast", CLAY, ha="left", va="bottom")
    valley = int(np.argmin(future))
    _callout(ax, (fut_x[valley], future[valley]), -16, -22, "held-out cycle", TEAL, ha="right", va="top")
    ax.set_xlim(-shown - 1, holdout + 12)
    ax.set_ylim(low - span * 0.18, bracket + span * 0.16)
    ax.set_xticks([-60, -40, -20, 0, 20])
    _names(ax, "step from the origin", "value")
    _ticks(ax)
    _save(fig, "quantile-forecast.png")


def mixture_density() -> None:
    rng = np.random.default_rng(4)
    n = 520
    series = rng.choice(np.array([-2.0, 2.0]), size=n) + rng.normal(0.0, 0.32, size=n)
    lookback = 24
    model = ASRForecaster(
        lookback=lookback,
        horizon=1,
        head="mixture",
        max_epochs=12,
        patience=4,
        batch_size=64,
        random_state=0,
    )
    model.fit(series)
    grid = np.linspace(-4.6, 4.6, 281)
    window = series[-lookback:]
    history = np.repeat(window[None, :], len(grid), axis=0)
    log_density = np.asarray(model.log_prob(history, grid[:, None]), dtype=np.float64).reshape(-1)
    density = np.exp(log_density)
    at = lambda level: float(np.exp(model.log_prob(window, np.array([level]))[0]))
    print(f"mixture density at -2 {at(-2):.4f} at 0 {at(0):.4f} at 2 {at(2):.4f}", flush=True)

    fig, ax = _chart(
        "Both modes get the mass.",
        "One peak at −2 and one at +2. Almost nothing sits between them.",
    )
    ax.hist(
        series,
        bins=np.linspace(-4.6, 4.6, 33),
        density=True,
        color=to_rgba(TEAL, 0.78),
        linewidth=0,
        zorder=2,
    )
    ax.plot(grid, density, color=CLAY, linewidth=2.8, solid_capstyle="round", zorder=3)
    left = int(np.argmax(np.where(grid < 0, density, -np.inf)))
    right = int(np.argmax(np.where(grid > 0, density, -np.inf)))
    _callout(ax, (grid[left], density[left]), -8, 16, "mode at −2", CLAY, ha="center", va="bottom")
    _callout(ax, (grid[right], density[right]), 8, 16, "mode at +2", CLAY, ha="center", va="bottom")
    ax.text(0.0, 0.16, "no mass", fontproperties=CALLOUT, color=MUTED, ha="center", va="bottom", zorder=4)
    ax.text(-3.7, 0.38, "series", fontproperties=CALLOUT, color=TEAL, ha="right", va="center", zorder=4)
    ax.set_xlim(-5.1, 5.1)
    ax.set_ylim(0, float(density.max()) * 1.28)
    ax.set_xticks([-4, -2, 0, 2, 4])
    _names(ax, "next value", "density")
    _ticks(ax)
    _save(fig, "mixture-density.png")


def histogram_bins() -> None:
    series = synthetic_cycle(24 * 16, seed=5)
    lookback = 36
    origin = 366
    observed = series[:origin]
    model = ASRForecaster(
        lookback=lookback,
        horizon=1,
        head="histogram",
        max_epochs=14,
        patience=5,
        batch_size=64,
        random_state=0,
    )
    model.fit(observed[: origin - 24])
    window = observed[-lookback:]
    probabilities, centers = model.bin_probabilities(window)
    probabilities = np.asarray(probabilities, dtype=np.float64)[0]
    centers = np.asarray(centers, dtype=np.float64)
    target = float(series[origin])
    width = float(centers[1] - centers[0])
    bin_index = int(np.clip(np.floor((target - (centers[0] - width / 2)) / width), 0, len(centers) - 1))
    print(
        f"histogram target {target:.3f} mass {probabilities[bin_index]:.3f} "
        f"centers {centers[0]:.2f}..{centers[-1]:.2f}",
        flush=True,
    )

    colors = [CLAY if index == bin_index else TEAL for index in range(len(centers))]
    fig, ax = _chart(
        "Almost all of the probability sits on the next step.",
        "Sixteen bins. The clay bar is the bin that holds the next value.",
    )
    ax.bar(centers, probabilities, width=width * 0.78, color=colors, linewidth=0, zorder=2)
    mass = probabilities[bin_index]
    ax.text(
        centers[bin_index],
        mass + 0.035,
        "next step",
        fontproperties=CALLOUT,
        color=CLAY,
        ha="center",
        va="bottom",
        zorder=4,
    )
    others = np.flatnonzero((probabilities > 0.01) & (np.arange(len(probabilities)) != bin_index))
    if len(others):
        other = int(others[0])
        _callout(
            ax,
            (centers[other], probabilities[other]),
            -28,
            16,
            "the rest",
            TEAL,
            ha="right",
            va="bottom",
        )
    empty_x = centers[max(0, bin_index - 4)]
    ax.text(
        empty_x,
        0.45,
        "empty",
        fontproperties=CALLOUT,
        color=MUTED,
        ha="center",
        va="center",
        zorder=4,
    )
    pad = width * 1.4
    ax.set_xlim(centers[0] - pad, centers[-1] + pad)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1.0])
    _names(ax, "value", "probability")
    _ticks(ax)
    _save(fig, "histogram-bins.png")


def tabular_path() -> None:
    fig, ax = _diagram(
        "A row becomes a prediction.",
        "Numeric columns are scaled. The category becomes a row in the embedding table. Both enter the backbone.",
        15.6,
        7.5,
    )
    _rank(ax, 0.02, 4.72, "Columns")
    _rank(ax, 4.55, 4.72, "Split")
    _rank(ax, 8.85, 4.72, "Backbone")
    _rank(ax, 12.15, 4.72, "Head")

    rooms = _node(ax, 1.35, 4.05, 2.4, 0.88, "rooms", "numeric", rail=TEAL)
    age = _node(ax, 1.35, 2.45, 2.4, 0.88, "age", "numeric", rail=TEAL)
    city = _node(ax, 1.35, 0.85, 2.4, 0.88, "city", "category", rail=TEAL)
    scaled = _node(ax, 6.05, 3.25, 2.75, 0.96, "scaled numbers", "median, robust scale")
    embedded = _node(ax, 6.05, 0.85, 2.75, 0.96, "embedding table", "one vector per name")
    backbone = _node(ax, 10.15, 2.05, 2.4, 0.96, "backbone", "residual blocks")
    head = _node(ax, 13.35, 2.05, 2.15, 0.96, "head", "the prediction", rail=CLAY)

    bus = 4.05
    _stroke(ax, [rooms["right"], (bus, rooms["cy"]), (bus, scaled["cy"])])
    _arrow(ax, [(bus, scaled["cy"]), scaled["left"]])
    _tag(ax, (rooms["right"][0] + bus) / 2, rooms["cy"] + 0.14, "robust scale", va="bottom")
    _stroke(ax, [age["right"], (bus, age["cy"]), (bus, scaled["cy"])])
    _tag(ax, (age["right"][0] + bus) / 2, age["cy"] - 0.14, "robust scale", va="top")
    _arrow(ax, [city["right"], embedded["left"]])
    _tag(ax, (city["right"][0] + embedded["left"][0]) / 2, city["cy"] + 0.16, "embed", va="bottom")

    join = 8.15
    _stroke(ax, [scaled["right"], (join, scaled["cy"]), (join, backbone["cy"])])
    _arrow(ax, [(join, backbone["cy"]), backbone["left"]])
    _tag(ax, scaled["right"][0] + 0.1, scaled["cy"] + 0.14, "concatenate", ha="left", va="bottom")
    _stroke(ax, [embedded["right"], (join, embedded["cy"]), (join, backbone["cy"])])
    _tag(ax, embedded["right"][0] + 0.1, embedded["cy"] - 0.14, "concatenate", ha="left", va="top")
    _arrow(ax, [backbone["right"], head["left"]])
    _tag(ax, (backbone["right"][0] + head["left"][0]) / 2, backbone["cy"] + 0.18, "features", va="bottom")
    _save(fig, "tabular-path.png")


def forecast_path() -> None:
    fig, ax = _diagram(
        "The last token chooses the forecast.",
        "A window, or a patched full history, becomes tokens. The head then branches.",
        16.6,
        8.9,
    )
    _rank(ax, 0.02, 6.25, "History")
    _rank(ax, 5.35, 6.25, "Tokens")
    _rank(ax, 8.55, 6.25, "Head")
    _rank(ax, 12.9, 6.25, "Outputs")

    window = _node(ax, 1.5, 5.05, 2.65, 0.9, "window", "the lookback", rail=TEAL)
    history = _node(ax, 1.5, 1.7, 2.65, 0.9, "full history", "older steps in patches", rail=TEAL)
    ax.text(0.08, 3.38, "or", fontproperties=EDGE, color=MUTED, ha="left", va="center", zorder=6)

    tokens = _node(ax, 6.45, 3.38, 2.15, 0.96, "tokens", "one vector each")
    head = _node(ax, 9.55, 3.38, 1.9, 0.96, "head", "reads the last token")

    outputs = [
        ("point", "one value", "value"),
        ("quantiles", "0.1, 0.5, 0.9", "three levels"),
        ("mixture", "five normals", "five normals"),
        ("flow", "one spline", "a spline"),
        ("histogram", "16 bins", "16 bins"),
    ]
    ys = [5.55, 4.28, 3.01, 1.74, 0.47]
    nodes = []
    for (title, subtitle, _edge), y in zip(outputs, ys, strict=True):
        nodes.append(_node(ax, 14.05, float(y), 2.2, 0.8, title, subtitle, rail=CLAY))

    bus = 4.55
    _stroke(ax, [window["right"], (bus, window["cy"]), (bus, tokens["cy"])])
    _arrow(ax, [(bus, tokens["cy"]), tokens["left"]])
    _tag(ax, window["right"][0] + 0.1, window["cy"] + 0.12, "one token per step", ha="left", va="bottom")
    _stroke(ax, [history["right"], (bus, history["cy"]), (bus, tokens["cy"])])
    _tag(ax, history["right"][0] + 0.1, history["cy"] - 0.12, "patch, then recent", ha="left", va="top")
    _arrow(ax, [tokens["right"], head["left"]])
    _tag(ax, (tokens["right"][0] + head["left"][0]) / 2, tokens["cy"] + 0.18, "last token", va="bottom")

    fan = 11.15
    _stroke(ax, [head["right"], (fan, head["cy"])])
    _stroke(ax, [(fan, ys[0]), (fan, ys[-1])])
    for node, (_title, _subtitle, edge), y in zip(nodes, outputs, ys, strict=True):
        _arrow(ax, [(fan, y), node["left"]])
        _tag(ax, (fan + node["left"][0]) / 2, y + 0.14, edge, va="bottom")
    _save(fig, "forecast-path.png")


def distributional_head() -> None:
    fig, ax = _diagram(
        "One vector becomes the density.",
        "It splits into a location, a scale, and a weight. Each triple is a normal, and the weights mix them.",
        15.6,
        11.2,
    )
    _rank(ax, 0.02, 8.7, "Vector")
    _rank(ax, 4.15, 8.7, "Per component")
    _rank(ax, 8.15, 8.7, "A normal")
    _rank(ax, 11.9, 8.7, "The mix")

    lanes = [7.55, 5.85, 4.15, 2.45, 0.75]
    vector = _node(ax, 1.25, float(np.mean(lanes)), 2.15, 1.08, "head vector", "one row of the head", rail=TEAL)
    density = _node(ax, 13.15, float(np.mean(lanes)), 2.2, 1.15, "density", "sum of the five", rail=CLAY)

    bus = 2.95
    _stroke(ax, [vector["right"], (bus, vector["cy"])])
    _tag(ax, (vector["right"][0] + bus) / 2, vector["cy"] + 0.16, "split", va="bottom")
    _stroke(ax, [(bus, lanes[0] + 0.58), (bus, lanes[-1] - 0.58)])

    param_x = 5.55
    join = 7.55
    normal_x = 9.15
    mix = 11.35
    for index, lane in enumerate(lanes):
        k = index + 1
        location = _node(ax, param_x, lane + 0.58, 1.55, 0.5, f"μ{k}")
        scale = _node(ax, param_x, lane, 1.55, 0.5, f"σ{k}")
        weight = _node(ax, param_x, lane - 0.58, 1.55, 0.5, f"π{k}")
        normal = _node(ax, normal_x, lane, 1.9, 0.78, f"Normal {k}", "μ and σ")
        _arrow(ax, [(bus, location["cy"]), location["left"]])
        _arrow(ax, [(bus, scale["cy"]), scale["left"]])
        _arrow(ax, [(bus, weight["cy"]), weight["left"]])
        _tag(ax, bus + 0.12, location["cy"] + 0.08, "location", ha="left", va="bottom")
        _tag(ax, bus + 0.12, scale["cy"] + 0.08, "scale", ha="left", va="bottom")
        _tag(ax, bus + 0.12, weight["cy"] + 0.08, "weight", ha="left", va="bottom")
        _stroke(ax, [location["right"], (join, location["cy"]), (join, normal["cy"])])
        _stroke(ax, [weight["right"], (join, weight["cy"]), (join, normal["cy"])])
        _arrow(ax, [(join, normal["cy"]), normal["left"]])
        _arrow(ax, [normal["right"], (mix, normal["cy"]), (mix, density["cy"]), density["left"]])
        _tag(ax, (normal["right"][0] + mix) / 2, normal["cy"] + 0.14, f"mix  π{k}", va="bottom")
    _save(fig, "distributional-head.png")


def outcome_tree() -> None:
    """The cricket tree of ``ASROutcomeTree``: one backbone, one softmax per internal node."""
    fig, ax = _diagram(
        "One backbone, one softmax per split.",
        "A nested outcome, as a tree of labels. A leaf's probability is the product of the branches on its path.",
        15.6,
        9.0,
    )
    _rank(ax, 0.02, 6.5, "Row")
    _rank(ax, 3.9, 6.5, "Splits")
    _rank(ax, 11.1, 6.5, "Leaves")

    row = _node(ax, 1.5, 3.3, 2.65, 0.96, "the row", "over, score, batter", rail=TEAL)
    ball = _node(ax, 5.15, 3.3, 1.95, 0.9, "outcome", "3 ways")
    runs = _node(ax, 8.6, 1.95, 1.75, 0.9, "runs", "2 ways")

    tops = {"wicket": 5.35, "dot": 4.15, "four": 2.6, "six": 1.35}
    nodes = {title: _node(ax, 12.6, y, 1.9, 0.78, title, rail=CLAY) for title, y in tops.items()}

    _arrow(ax, [row["right"], ball["left"]])
    _tag(ax, (row["right"][0] + ball["left"][0]) / 2, ball["cy"] + 0.18, "one backbone", va="bottom")

    fan = 6.55
    _stroke(ax, [ball["right"], (fan, ball["cy"])])
    _stroke(ax, [(fan, tops["wicket"]), (fan, runs["cy"])])
    for title in ("wicket", "dot"):
        _arrow(ax, [(fan, tops[title]), nodes[title]["left"]])
        _tag(ax, (fan + nodes[title]["left"][0]) / 2, tops[title] + 0.16, f"P({title} | x)", va="bottom")
    _arrow(ax, [(fan, runs["cy"]), runs["left"]])
    _tag(ax, (fan + runs["left"][0]) / 2, runs["cy"] + 0.16, "P(runs | x)", va="bottom")

    split = 10.1
    _stroke(ax, [runs["right"], (split, runs["cy"])])
    _stroke(ax, [(split, tops["four"]), (split, tops["six"])])
    for title in ("four", "six"):
        _arrow(ax, [(split, tops[title]), nodes[title]["left"]])
        _tag(ax, (split + nodes[title]["left"][0]) / 2, tops[title] + 0.16, f"P({title} | runs)", va="bottom")

    for offset, text in (
        (0.34, 'ASROutcomeTree({"wicket": None, "dot": None, "runs": ["four", "six"]})'),
        (-0.18, 'predict_proba gives the leaves, predict_conditional(X, "runs") one split.'),
    ):
        ax.text(0.02, offset, text, fontproperties=EDGE, color=MUTED, ha="left", va="center", zorder=6)
    _save(fig, "outcome-tree.png")


def main() -> None:
    _load_fonts()
    selected = sys.argv[1:]
    jobs = [
        ("outcome_tree", outcome_tree),
        ("tabular_fit", tabular_fit),
        ("categorical_embeddings", categorical_embeddings),
        ("ordinal_score", ordinal_score),
        ("quantile_forecast", quantile_forecast),
        ("mixture_density", mixture_density),
        ("histogram_bins", histogram_bins),
        ("tabular_path", tabular_path),
        ("forecast_path", forecast_path),
        ("distributional_head", distributional_head),
    ]
    for name, job in jobs:
        if not selected or name in selected:
            job()


if __name__ == "__main__":
    main()
