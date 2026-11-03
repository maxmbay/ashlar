"""The grammar strip: every builder in ``ashlar.observed``, drawn as what it says about ``y``.

No data and no fit. Each glyph puts one label on the same predictive density: what the label says of ``y`` in ink,
and the part of the density the head scores, its likelihood, in teal. ``draw`` writes
``docs/images/readme/label-grammar.png`` and the gallery thumbnail ``thumbs/observed.png``; there is no ``compute``
stage.

    python examples/readme/label_grammar.py draw
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style as k  # noqa: E402

X = np.linspace(0.0, 1.0, 800)
GLYPH_W, GLYPH_H = 2.05, 1.22  # inches, the same for every glyph
MARK_Y = -0.17  # where a label's ink sits under the axis


def _gamma(x, shape: float = 3.2, scale: float = 0.13, start: float = 0.04):
    from scipy.stats import gamma

    return gamma.pdf(np.asarray(x) - start, shape, scale=scale)


PEAK = float(_gamma(X).max())


def f(x):
    """The glyph's density, a shifted gamma on [0, 1] scaled so its peak sits at height 1."""
    return _gamma(x) / PEAK


def _base(ax, top: float = 1.12, axis: bool = True):
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.36, top)
    ax.axis("off")
    if axis:
        ax.plot([0.0, 1.0], [0.0, 0.0], color=k.INK, lw=1.1, solid_capstyle="butt", zorder=2)


def _curve(ax, y=None, where=None, **kwargs):
    style_ = {"color": k.TEAL, "lw": 1.6, "zorder": 3}
    style_.update(kwargs)
    curve = f(X) if y is None else np.asarray(y, float)
    if where is not None:
        curve = np.where(where, curve, np.nan)
    ax.plot(X, curve, **style_)


def _shade(ax, low: float, high: float, y=None):
    curve = f(X) if y is None else y
    ax.fill_between(X, 0, curve, where=(X >= low) & (X <= high), color=k.TEAL_PALE, lw=0, zorder=1, interpolate=True)


def _height(ax, x: float, height: float):
    """A density read at one value: a dark teal stroke from the axis to the curve."""
    ax.plot([x, x], [0, height], color=k.TEAL_DARK, lw=2.4, solid_capstyle="butt", zorder=4)


def _dot(ax, x, y=0.0, size=36):
    ax.scatter([x], [y], s=size, color=k.INK, edgecolor=k.CARD, lw=1.3, zorder=7, clip_on=False)


def _arrow(ax, x0: float, x1: float, y: float = MARK_Y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y), zorder=6,
                arrowprops={"arrowstyle": "-|>,head_length=0.5,head_width=0.25", "color": k.INK, "lw": 1.5,
                            "shrinkA": 0, "shrinkB": 0})


def _bracket(ax, low: float, high: float, y: float = MARK_Y, serif: float = 0.065, lip: float = 0.022):
    """A closed range in ink: a rule with square ends."""
    ax.plot([low, high], [y, y], color=k.INK, lw=1.5, solid_capstyle="butt", zorder=6)
    for x, inward in ((low, 1), (high, -1)):
        ax.plot([x + inward * lip, x, x, x + inward * lip], [y + serif, y + serif, y - serif, y - serif],
                color=k.INK, lw=1.5, solid_joinstyle="miter", solid_capstyle="butt", zorder=6)


def _veil(ax, low: float, high: float, top: float = 1.08):
    """A range of values the table never shows: a warm-grey veil from the axis up."""
    ax.add_patch(Rectangle((low, -0.05), high - low, top + 0.05, facecolor=k.DATA, alpha=0.38, lw=0, zorder=5))


def _post(ax, x: float, top: float = 1.08):
    """A truncation edge: an ink post from just under the axis to the top of the veil."""
    ax.plot([x, x], [-0.07, top], color=k.INK, lw=1.3, solid_capstyle="butt", zorder=6)


def exact(ax):
    _base(ax)
    _curve(ax)
    y = 0.42
    _height(ax, y, f(y))
    _dot(ax, y)


def interval(ax):
    _base(ax)
    _shade(ax, 0.3, 0.62)
    _curve(ax)
    _bracket(ax, 0.3, 0.62)


def censored(ax):
    _base(ax)
    _shade(ax, 0.52, 1.0)
    _curve(ax)
    _dot(ax, 0.52, MARK_Y)
    _arrow(ax, 0.52, 1.0)


def rounded(ax):
    _base(ax)
    y, step = 0.5, 0.2
    _shade(ax, y - step / 2, y + step / 2)
    _curve(ax)
    _bracket(ax, y - step / 2, y + step / 2)
    _dot(ax, y, MARK_Y)


def noisy(ax):
    _base(ax)
    sd = 0.085
    kernel = np.exp(-0.5 * ((X - 0.5) / sd) ** 2)
    wide = np.convolve(f(X), kernel / kernel.sum(), mode="same")
    _curve(ax, color=k.TEAL_PALE, lw=1.4)
    _curve(ax, wide)
    y = 0.64
    _height(ax, y, float(np.interp(y, X, wide)))
    for size, tone in ((420, 0.66), (190, 0.4)):
        ax.scatter([y], [MARK_Y], s=size, color=k.mix(k.DATA, k.CARD, tone), lw=0, zorder=5, clip_on=False)
    _dot(ax, y, MARK_Y)


def survival(ax):
    _base(ax)
    entry, time = 0.22, 0.56
    _shade(ax, time, 1.0)
    _curve(ax, where=X >= entry)
    _curve(ax, where=X <= entry, color=k.TEAL_PALE, lw=1.4)
    _veil(ax, -0.03, entry)
    _post(ax, entry)
    _dot(ax, time, MARK_Y)
    _arrow(ax, time, 1.0)


def truncated(ax):
    low, high = 0.2, 0.74
    inside = (X >= low) & (X <= high)
    mass = np.trapezoid(f(X)[inside], X[inside]) / np.trapezoid(f(X), X)
    top = 1.0 / mass + 0.06
    _base(ax, top=top)
    _curve(ax, color=k.TEAL_PALE, lw=1.4)
    _curve(ax, f(X) / mass, where=inside)
    y = 0.45
    _height(ax, y, f(y) / mass)
    _veil(ax, -0.03, low, top=top - 0.03)
    _veil(ax, high, 1.03, top=top - 0.03)
    _post(ax, low, top - 0.03)
    _post(ax, high, top - 0.03)
    _dot(ax, y)


def total(ax):
    _base(ax)
    days = np.array([0.36, 0.22, 0.44, 0.3])
    width, gap = 0.1, 0.035
    xs = 0.02 + np.arange(len(days)) * (width + gap)
    for x, h in zip(xs, days, strict=True):
        ax.add_patch(Rectangle((x, 0), width, h * 1.25, facecolor="none", edgecolor=k.DATA_DARK, lw=1.2,
                               linestyle=(0, (2.5, 2)), zorder=3))
    _bracket(ax, xs[0], xs[-1] + width)
    ax.text(0.66, 0.36, "=", fontproperties=k.F.value, color=k.MUTED, ha="center", va="center")
    column, bottom = 0.78, 0.0
    for h in days * 1.25 * 0.6:
        ax.add_patch(Rectangle((column, bottom), width, h, facecolor="none", edgecolor=k.DATA_DARK, lw=1.0,
                               linestyle=(0, (2.5, 2)), zorder=3))
        bottom += h
    ax.add_patch(Rectangle((column, 0), width, bottom, facecolor="none", edgecolor=k.INK, lw=1.6, zorder=4))


def one_of(ax):
    _base(ax, axis=False)
    ax.set_ylim(-0.5, 1.12)
    names = ("cat", "dog", "bird")
    probability = np.array([0.4, 0.27, 0.33])
    xs = np.array([0.17, 0.5, 0.83])
    for x, name, p, inside in zip(xs, names, probability, (True, True, False), strict=True):
        ax.add_patch(Rectangle((x - 0.1, 0.0), 0.2, p * 2.2, facecolor=k.TEAL_PALE if inside else k.HAIR,
                               edgecolor=k.TEAL if inside else "none", lw=1.4, zorder=3))
        ax.text(x, -0.06, name, fontproperties=k.F.mono, color=k.INK if inside else k.MUTED, ha="center", va="top",
                zorder=6)
    ax.plot([0.0, 1.0], [0.0, 0.0], color=k.INK, lw=1.1, solid_capstyle="butt", zorder=4)
    _bracket(ax, xs[0] - 0.12, xs[1] + 0.12, y=-0.42, serif=0.05)


ROWS = (
    (
        (exact, "exact(y)", "the density at y"),
        (interval, "interval(lower, upper)", "P(lower ≤ Y ≤ upper)"),
        (censored, "censored(y, flag)", "P(Y ≥ y) where flagged"),
        (rounded, "rounded(y, step)", "P(Y within step / 2 of y)"),
        (noisy, "noisy(y, scale)", "density of Y + noise at y"),
    ),
    (
        (survival, "survival(time, event, entry)", "P(T ≥ time) ÷ P(T ≥ entry)"),
        (truncated, "truncated(y, lower, upper)", "density ÷ P(lower ≤ Y ≤ upper)"),
        (total, "total(values, group)", "density of the group's sum"),
        (one_of, "one_of(candidates)", "P(cat) + P(dog)"),
    ),
)
ROW_H = 2.36  # inches per row: the glyph, then the call and its likelihood under it


def draw() -> None:
    fig = k.card(
        "Label types in ashlar.observed",
        "Each label type drawn on one predicted density: what the label says about y in ink, the part of the "
        "distribution its likelihood uses in teal, and in warm grey the values the data does not show.",
        height=1.85 + 2 * ROW_H - 0.22,
    )
    box = fig.box
    width_in = box.width - box.left - box.right
    for row, glyphs in enumerate(ROWS):
        pitch = width_in / len(glyphs)
        glyph_top = box.height - box.top - row * ROW_H
        for column, (glyph, call, meaning) in enumerate(glyphs):
            left = box.left + column * pitch
            ax = fig.add_axes([left / box.width, (glyph_top - GLYPH_H) / box.height, GLYPH_W / box.width,
                               GLYPH_H / box.height])
            k.style(ax, frame="")
            glyph(ax)
            base = glyph_top - GLYPH_H - 0.36
            fig.text(left / box.width, base / box.height, call, fontproperties=k.F.mono, color=k.INK, ha="left",
                     va="baseline")
            fig.text(left / box.width, (base - 0.3) / box.height, meaning, fontproperties=k.F.small, color=k.MUTED,
                     ha="left", va="baseline")
    k.save(fig, "label-grammar")
    _thumb()


# The tile's number: four products that sell out together, simulated (docs/benchmarks.md, "Several products that
# sell out together"). The joint mixture fitted to the censored label expects every shelf empty on 11.4% of test
# days, against the 11.1% on which it happened; four censored fits, one per product, expect 0.7%.
JOINT_EMPTY = {"joint": 0.114, "seen": 0.111, "one per product": 0.007}


def _thumb() -> None:
    def mark(ax):
        ax.axis("off")
        for column, glyph in enumerate((censored, noisy, total)):
            inset = ax.inset_axes([column * 0.35, 0.0, 0.3, 1.0])
            k.style(inset, frame="")
            glyph(inset)
            for line in inset.lines:  # the tile is shown at a quarter of its pixels: heavier strokes
                line.set_linewidth(1.8 * line.get_linewidth())
            for patch in inset.patches:
                patch.set_linewidth(1.6 * patch.get_linewidth())
            for dots in inset.collections:
                if hasattr(dots, "get_sizes") and len(dots.get_sizes()):
                    dots.set_sizes(2.2 * dots.get_sizes())

    k.thumb("observed", "Observed labels",
            [("All four out: ", k.MUTED), (k.pct(JOINT_EMPTY["joint"], 1), k.TEAL),
             (f", seen {k.pct(JOINT_EMPTY['seen'], 1)}, against ", k.MUTED),
             (k.pct(JOINT_EMPTY["one per product"], 1), k.CLAY)], mark)


if __name__ == "__main__":
    k.main(draw=draw)
