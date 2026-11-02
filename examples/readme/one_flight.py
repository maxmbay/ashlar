"""One fit, one flight, every answer: what the quick start's calls return for one held-out flight.

Draws ``docs/images/readme/one-flight.png`` from the hero's artifacts
(``examples/readme/artifacts/hero-flights/``, written by ``hero_flights.py compute``):
``card.json`` holds every call's value for twelve candidate flights, and ``pool.npz``
holds their mixtures. The flight is picked by a fixed rule: the candidate with the
largest actual delay that its calibrated 80% interval still holds.

The top panel is that flight's predictive density on the hero's delay axis, with
the calibrated 80% interval solid, the point mass at zero (leaving on the minute) as a
stem, CatBoost ``RMSEWithUncertainty``'s 80% interval as a clay bar, and the delay
that happened. Under it, each call of the quick start on that one row, in order, with
its answer placed on the same axis; LightGBM's point sits beside ``predict``.

    python examples/readme/one_flight.py draw
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import hero_flights as h
import style as k

NAME = "one-flight"
X_LO, X_HI = h.X_LO, h.X_HI
DENSITY = 3.0  # the density panel's height, in lane units
LANE = 0.86  # the gap between lanes
AXIS_LEFT = 5.15  # inches from the card's left edge to the delay axis; the calls sit left of it
CLEAR = {"facecolor": k.CARD, "edgecolor": "none", "pad": 1.5}  # a label the threshold line passes behind
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()


def pick(card) -> int:
    """The candidate with the largest actual delay inside its calibrated 80% interval."""
    inside = [j for j, (lo, hi) in enumerate(card["predict_interval"]) if lo <= card["actual"][j] <= hi]
    return max(inside, key=lambda j: card["actual"][j])


def out(x: float, digits: int = 1) -> str:
    """A number as Python prints it, rounded: what the call returns."""
    return f"{x:.{digits}f}"


def draw(folder: Path = h.ARTIFACTS) -> None:
    summary, pool, card = h.load(folder)
    j = pick(card)
    i = int(np.flatnonzero(pool["is_card"])[j])
    if int(pool["row"][i]) != int(card["rows"][j]):
        raise SystemExit("pool.npz and card.json disagree on the card flights")
    col, fmt = summary["columns"], summary["clock_format"]
    x_of = lambda name: pool[f"x_{col[name]}"][i]  # noqa: E731
    actual = float(card["actual"][j])
    mean, median = float(card["predict"][j]), float(card["median"][j])
    q10, q90 = (float(v) for v in card["predict_quantiles"][j])
    p60 = float(card["predict_exceedance"][j])
    lo, hi = (float(v) for v in card["predict_interval"][j])
    draws = [float(v) for v in card["sample"][j]]
    log_prob = float(card["log_prob"][j])
    cat_loc, cat_scale = float(card["catboost_loc"][j]), float(card["catboost_scale"][j])
    cat_lo, cat_hi = cat_loc - h.bench.Z80 * cat_scale, cat_loc + h.bench.Z80 * cat_scale
    lgbm = float(card["lightgbm_point"][j])
    explained = card["explain"]
    values = np.asarray(explained["values"][j])
    top = np.argsort(-np.abs(values))[:3]
    departure = h.clock(x_of("departure"), fmt)

    fig = k.card(
        "One fit, one flight, every answer.",
        "A held-out flight. Each line is a call from the quick start on this one row, its answer placed on the "
        "same delay axis as the flight's predicted distribution.",
        height=9.2,
        footer=3,
    )
    box = fig.box
    _, content_bottom, _, content_h = box.cell()
    bottom = content_bottom + 0.9  # room under the axes for the tick labels and the axis name
    axis_w = box.width - box.right - AXIS_LEFT
    ax = fig.add_axes([AXIS_LEFT / box.width, bottom / box.height, axis_w / box.width,
                       (content_bottom + content_h - bottom) / box.height])
    k.style(ax, frame="b")
    per_u = axis_w / (h.warp(X_HI) - h.warp(X_LO))  # inches per unit of the warped axis

    def width_u(text, prop) -> float:
        return k.text_width(text, prop) / per_u

    # ---- the flight's distribution
    u = np.linspace(h.warp(X_LO), h.warp(X_HI), 1400)
    density = h.warped_density(pool, i, u)
    curve = density / density.max() * DENSITY
    _, (mass, at) = h.split_atom(pool, i)
    solid = k.mix(k.TEAL, k.CARD, 0.36)
    ax.fill_between(u, 0, curve, color=k.TEAL_PALE, lw=0, zorder=2)
    ax.fill_between(u, 0, curve, where=(u >= h.warp(lo)) & (u <= h.warp(hi)), color=solid, lw=0, zorder=3,
                    interpolate=True)
    ax.plot(u, curve, color=k.TEAL_DARK, lw=1.5, zorder=4)
    stem = mass * h.NEEDLE * DENSITY / 1.5  # the hero's stem scale, relative to a ridge's height
    ax.plot([h.warp(at)] * 2, [0, stem], color=k.TEAL_DARK, lw=2.8, solid_capstyle="butt", zorder=6)
    ax.scatter([h.warp(at)], [stem], s=40, color=k.TEAL_DARK, edgecolor=k.CARD, lw=0.9, zorder=7)

    def height_at(x):
        return float(np.interp(h.warp(x), u, curve))

    label_y = DENSITY + 0.35
    for x, label in ((median, f"median {out(median)}"), (mean, f"mean {out(mean)}")):
        ux = h.warp(x)
        ax.plot([ux, ux], [0, height_at(x)], color=k.TEAL_DARK, lw=1.3, zorder=5)
        ax.plot([ux, ux], [height_at(x) + 0.08, label_y - 0.08], color=k.TEAL_DARK, lw=0.8, ls=(0, (1, 2)), zorder=5)
        k.value(ax, ux, label_y, label, k.TEAL_DARK, ha="center", va="bottom", size=12.5)
    y_bar = -0.42
    ax.plot([h.warp(cat_lo), h.warp(cat_hi)], [y_bar, y_bar], color=k.CLAY, lw=3.0, solid_capstyle="butt", zorder=5)
    ax.text(h.warp(cat_lo), y_bar - 0.2, f"CatBoost RMSEWithUncertainty's 80% interval, {out(cat_lo, 0)} to "
            f"{out(cat_hi, 0)} minutes".replace("-", "−"), fontproperties=k.F.small, color=k.CLAY, ha="left", va="top")
    ax.scatter([h.warp(actual)], [0], s=64, color=k.INK, edgecolor=k.CARD, lw=1.2, zorder=10, clip_on=False)
    note_y = max(height_at(x) for x in np.linspace(actual, X_HI, 50)) + 0.55
    ax.plot([h.warp(actual)] * 2, [0.12, note_y - 0.1], color=k.INK, lw=0.8, ls=(0, (1, 2)), zorder=9)
    ax.text(h.warp(actual), note_y, f"it left {actual:.0f} minutes late", fontproperties=k.F.label, color=k.INK,
            ha="center", va="bottom", zorder=10, bbox=CLEAR)

    # ---- the calls, one lane each, in the quick start's order
    lanes = [
        ("predict", "model.predict(x)"),
        ("quantiles", "model.predict_quantiles(x, [0.1, 0.9])"),
        ("exceedance", "model.predict_exceedance(x, 60)"),
        ("sample", "model.sample(x, n=5)"),
        ("log_prob", "model.log_prob(x, y)"),
        ("calibrate", "model.calibrate(X_holdout, y_holdout)"),
        ("interval", "model.predict_interval(x, coverage=0.8)"),
        ("explain", 'model.explain(x, "quantile", level=0.9)'),
    ]
    code_frac = (box.left - AXIS_LEFT) / axis_w
    rows = {}
    for n, (key, call) in enumerate(lanes):
        y = -1.45 - n * LANE
        rows[key] = y
        ax.text(code_frac, y, call, transform=ax.get_yaxis_transform(), fontproperties=k.F.mono, color=k.INK,
                ha="left", va="center")
        ax.plot([h.warp(X_LO), h.warp(X_HI)], [y, y], color=k.HAIR, lw=0.8, zorder=0)
    pad = 0.08

    def val(x, y, text, ha="left", color=k.TEAL_DARK):
        k.code(ax, x, y + 0.02, text, color, bold=True, ha=ha, va="center", zorder=12)

    def dot(x, y, color=k.TEAL, size=46):
        ax.scatter([h.warp(x)], [y], s=size, color=color, edgecolor=k.CARD, lw=1.0, zorder=11)

    def bracket(a, b, y):
        ax.plot([h.warp(a), h.warp(b)], [y, y], color=k.TEAL, lw=2.4, solid_capstyle="butt", zorder=11)
        for end in (a, b):
            ax.plot([h.warp(end)] * 2, [y - 0.16, y + 0.16], color=k.TEAL, lw=2.0, solid_capstyle="butt", zorder=11)

    y = rows["predict"]
    dot(mean, y)
    val(h.warp(mean) - pad, y, out(mean), ha="right")
    dot(lgbm, y, k.SLATE)
    ax.text(h.warp(lgbm) + pad, y, f"LightGBM's point, {out(lgbm)}", fontproperties=k.F.small, color=k.SLATE,
            ha="left", va="center", zorder=10, bbox=CLEAR)

    y = rows["quantiles"]
    bracket(q10, q90, y)
    val(h.warp(q10) - pad, y, out(q10), ha="right")
    val(h.warp(q90) + pad, y, out(q90))

    y = rows["exceedance"]
    ax.annotate("", xy=(h.warp(X_HI), y), xytext=(h.warp(h.THRESHOLD), y), zorder=11,
                arrowprops={"arrowstyle": "-|>", "color": k.TEAL, "lw": 2.2, "shrinkA": 0, "shrinkB": 0,
                            "mutation_scale": 14})
    ax.plot([h.warp(h.THRESHOLD)] * 2, [y - 0.16, y + 0.16], color=k.TEAL, lw=2.0, solid_capstyle="butt", zorder=11)
    val(h.warp(h.THRESHOLD) - pad, y, out(p60, 3), ha="right")

    y = rows["sample"]
    ax.scatter(h.warp(draws), [y] * len(draws), s=46, color=k.TEAL, edgecolor=k.CARD, lw=1.0, zorder=11)
    text = "[" + ", ".join(out(d) for d in draws) + "]"
    if h.warp(max(draws)) + pad + width_u(text, k.F.mono_bold) <= h.warp(X_HI):
        val(h.warp(max(draws)) + pad, y, text)
    else:
        val(h.warp(min(draws)) - pad, y, text, ha="right")

    y = rows["log_prob"]
    dot(actual, y, k.INK)
    val(h.warp(actual) + pad, y, out(log_prob, 2))
    ax.text(h.warp(actual) - pad, y, f"y = {actual:.0f}", fontproperties=k.F.small, color=k.MUTED, ha="right",
            va="center")

    y = rows["calibrate"]
    ax.text(h.warp(X_LO), y, f"on {card['calibration_rows']:,} other held-out flights", fontproperties=k.F.small,
            color=k.MUTED, ha="left", va="center")

    y = rows["interval"]
    bracket(lo, hi, y)
    val(h.warp(lo) - pad, y, out(lo), ha="right")
    val(h.warp(hi) + pad, y, out(hi))

    y = rows["explain"]
    parts = [f"{explained['feature_names'][f]} {values[f]:+.1f}" for f in top]
    val(h.warp(X_LO), y, ",  ".join(parts))

    # The threshold of predict_exceedance, from its lane up through the distribution.
    ax.plot([h.warp(h.THRESHOLD)] * 2, [rows["exceedance"], label_y], color=k.MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
    ax.set_xlim(h.warp(X_LO), h.warp(X_HI))
    ax.set_ylim(rows["explain"] - 0.5, label_y + 0.6)
    ax.set_yticks([])
    ax.set_xticks(h.warp(h.TICKS))
    ax.set_xticklabels([k.num(t) for t in h.TICKS])
    for label in ax.get_xticklabels():
        label.set_fontproperties(k.F.tick)
    k.axis_names(ax, x="departure delay, minutes (log scale beyond half an hour)")

    # ---- the flight, beside its distribution
    ident = ax.get_yaxis_transform()
    ax.text(code_frac, DENSITY + 0.6, f"{x_of('carrier')} {x_of('origin')}→{x_of('dest')}", transform=ident,
            fontproperties=k.font("Bold", 24), color=k.INK, ha="left", va="top")
    ax.text(code_frac, DENSITY - 0.5, f"{DAYS[int(x_of('weekday')) - 1]} {departure}, "
            f"{MONTHS[int(x_of('month')) - 1]}\n{int(x_of('distance')):,} miles", transform=ident,
            fontproperties=k.F.label, color=k.MUTED, ha="left", va="top", linespacing=1.5)

    h.legend(fig, k, [("patch", solid, "calibrated 80% interval"), ("pale", k.TEAL_PALE, "tails"),
                      ("needle", k.TEAL_DARK, "the chance of leaving on the minute"),
                      ("bar", k.CLAY, "CatBoost's 80%"), ("dot", k.INK, "what happened")], line=2)
    names = explained["feature_names"]
    dep = summary["columns"]["departure"]
    base = explained["numeric_baseline"].get(dep)
    carrier_value = values[names.index(summary["columns"]["carrier"])]
    k.footnote(fig, f"explain splits the 0.9 quantile, {out(q90)} minutes, among the columns: the {departure} departure "
                    f"adds {values[names.index(dep)]:.1f} minutes against the median departure, "
                    f"{h.clock(base, fmt)}, and {x_of('carrier')} takes {abs(carrier_value):.1f} off against the "
                    "average carrier.", line=0)
    k.save(fig, NAME)


if __name__ == "__main__":
    k.main(draw=draw)
