"""The README figure kit: one look for every figure on the front page.

Every README figure is an off-white card with rounded corners on a transparent
PNG, so it reads as a soft card on a white page and as a deliberate light card
in GitHub's dark mode, as the logo does. Colours have fixed roles, so a reader
learns them once:

    TEAL   Ashlar                         TEAL_PALE   its tails, bands, fills
    CLAY   the strongest baseline          CLAY_PALE   its bands and fills
    SLATE  a second baseline or reference  DATA        observed data (bars)
    INK    what actually happened, the truth, titles

Ashlar's own outputs are never drawn on a teal-to-clay ramp, since clay means
the baseline; an ordered condition uses ``TEAL_RAMP``.

A figure script under ``examples/readme/`` has a ``compute`` stage (heavy, may
need the Mac, writes small artifacts) and a ``draw`` stage (seconds, reads them):

    import style as k

    def compute():
        ...
        k.save_json("hero-flights", "flights", {...})

    def draw():
        data = k.load_json("hero-flights", "flights")
        fig = k.card("Every flight gets its own distribution.",
                     "Airline delay, 2,000,000 flights, schedule only. ...", height=10.5, footer=1)
        ax = k.panel(fig)                    # the whole content area
        ...
        k.footnote(fig, "CRPS 9.17 against 11.46 ...")
        k.save(fig, "hero-flights")          # docs/images/readme/hero-flights.png

    if __name__ == "__main__":
        k.main(compute=compute, draw=draw)   # python examples/readme/x.py compute|draw

Fonts: Inter for text, JetBrains Mono for API names, found on the system or
under ``ASHLAR_FONT_DIR``. The kit refuses to draw without them rather than
fall back to a default face. ``python examples/readme/style.py`` draws a
specimen of the kit.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402
from matplotlib.textpath import TextPath  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "docs" / "images" / "readme"
THUMBS = IMAGES / "thumbs"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

DPI = 200
WIDTH = 13.8  # inches; GitHub shows a full-width figure at about a third of its pixels
THUMB_SIZE = (6.4, 3.4)
MAX_BYTES = 450_000


# ---------------------------------------------------------------- colour


def mix(a: str, b: str, t: float) -> str:
    """``a`` moved a share ``t`` of the way toward ``b``."""
    ca, cb = np.array(to_rgb(a)), np.array(to_rgb(b))
    return to_hex(ca + (cb - ca) * float(np.clip(t, 0.0, 1.0)))


def shade(color: str, amount: float) -> str:
    """Lighter toward white for ``amount > 0``, darker toward black for ``amount < 0``."""
    return mix(color, "#ffffff", amount) if amount > 0 else mix(color, "#000000", -amount)


CARD = "#fbfaf7"  # the logo's dark-mode card
INK = "#1b1a17"  # the logo's ink
MUTED = "#5f5a52"  # the logo's muted text
HAIR = "#e4e0da"
GRID = "#eeeae3"
FLOOR = "#e9e4dc"  # the logo's floor
FLOOR_EDGE = "#d9d2c6"

TEAL = "#147a72"  # the logo's teal: Ashlar
CLAY = "#d4502a"  # the logo's clay: the strongest baseline
SLATE = "#6e7a86"  # a second baseline or a reference tool
DATA = "#d8d1c6"  # observed data: bars, histograms
DATA_DARK = "#8a8178"  # observed data drawn as lines or points

TEAL_DARK = shade(TEAL, -0.28)
TEAL_PALE = mix(TEAL, CARD, 0.72)
TEAL_WASH = mix(TEAL, CARD, 0.88)
CLAY_DARK = shade(CLAY, -0.22)
CLAY_PALE = mix(CLAY, CARD, 0.70)
CLAY_WASH = mix(CLAY, CARD, 0.87)
SLATE_PALE = mix(SLATE, CARD, 0.70)

# An ordered condition (hour of day, a sweep): light to dark teal.
TEAL_RAMP = LinearSegmentedColormap.from_list("ashlar_teal", [mix(TEAL, CARD, 0.78), TEAL, TEAL_DARK])
# A density as colour: the card, through pale teal, to dark teal.
DENSITY = LinearSegmentedColormap.from_list("ashlar_density", [CARD, TEAL_PALE, TEAL, TEAL_DARK])


# ---------------------------------------------------------------- fonts

_SEARCH = [
    os.environ.get("ASHLAR_FONT_DIR", ""),
    "/usr/share/fonts/opentype/inter",
    "/usr/share/fonts/truetype/inter",
    "/usr/share/fonts/truetype/jetbrains-mono",
    "/usr/share/fonts/opentype/jetbrains-mono",
    str(Path.home() / ".local/share/fonts"),
    str(Path.home() / "Library/Fonts"),
    "/Library/Fonts",
    "/usr/share/fonts",
]
_FILES: dict[str, Path] = {}


def _find(stem: str) -> Path:
    if stem in _FILES:
        return _FILES[stem]
    for folder in filter(None, _SEARCH):
        base = Path(folder)
        if not base.is_dir():
            continue
        for ext in (".otf", ".ttf"):
            hits = [base / f"{stem}{ext}"] if (base / f"{stem}{ext}").exists() else list(base.rglob(f"{stem}{ext}"))
            if hits:
                _FILES[stem] = hits[0]
                return hits[0]
    raise RuntimeError(
        f"font {stem} not found. README figures need Inter and JetBrains Mono: draw them in the cloud "
        "container, which has both, or set ASHLAR_FONT_DIR to a folder that holds Inter-Regular.otf, "
        "Inter-SemiBold.otf, JetBrainsMono-Regular.ttf, and the other weights."
    )


def font(weight: str = "Regular", size: float = 12.5, mono: bool = False) -> FontProperties:
    """Inter (or JetBrains Mono) at a weight: Regular, Medium, SemiBold, Bold."""
    family = "JetBrainsMono" if mono else "Inter"
    return FontProperties(fname=str(_find(f"{family}-{weight}")), size=size)


@dataclass
class Fonts:
    title: FontProperties
    subtitle: FontProperties
    axis: FontProperties
    tick: FontProperties
    label: FontProperties
    value: FontProperties
    small: FontProperties
    panel: FontProperties
    mono: FontProperties
    mono_bold: FontProperties


F: Fonts | None = None


def setup() -> Fonts:
    """Load the fonts and set matplotlib's defaults. Called by ``card`` and ``thumb``; safe to call twice."""
    global F
    if F is not None:
        return F
    for weight in ("Regular", "Medium", "SemiBold", "Bold"):
        font_manager.fontManager.addfont(str(_find(f"Inter-{weight}")))
    for weight in ("Regular", "Medium", "SemiBold", "Bold"):
        font_manager.fontManager.addfont(str(_find(f"JetBrainsMono-{weight}")))
    F = Fonts(
        title=font("Bold", 25),
        subtitle=font("Regular", 14.5),
        axis=font("SemiBold", 14),
        tick=font("Regular", 12.5),
        label=font("Medium", 13),
        value=font("SemiBold", 13.5),
        small=font("Regular", 12),
        panel=font("SemiBold", 15),
        mono=font("Regular", 12.5, mono=True),
        mono_bold=font("SemiBold", 12.5, mono=True),
    )
    plt.rcParams.update(
        {
            "font.family": font("Regular").get_name(),
            "font.size": 12.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 1.1,
            "axes.facecolor": "none",
            "axes.unicode_minus": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 12.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "xtick.minor.size": 0,
            "ytick.minor.size": 0,
            "xtick.major.pad": 8,
            "ytick.major.pad": 8,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "svg.fonttype": "none",
            "legend.frameon": False,
        }
    )
    return F


# ---------------------------------------------------------------- text


def text_width(text: str, prop: FontProperties) -> float:
    """Width of one line of text in inches."""
    return TextPath((0, 0), text, prop=prop).get_extents().width / 72.0


def wrap(text: str, prop: FontProperties, width_in: float) -> str:
    """Break ``text`` into lines no wider than ``width_in``. Explicit newlines are kept."""
    lines = []
    for paragraph in text.split("\n"):
        words, line = paragraph.split(" "), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if line and text_width(trial, prop) > width_in:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return "\n".join(lines)


def rich(where, x: float, y: float, segments, *, transform=None, ha: str = "left", va: str = "center", zorder=10):
    """One line of text in several colours and faces: ``segments`` is ``[(text, color, FontProperties), ...]``.

    ``where`` is an axes or a figure; ``transform`` defaults to its fraction coordinates.
    """
    areas = [TextArea(t, textprops={"color": c, "fontproperties": p}) for t, c, p in segments]
    box = HPacker(children=areas, align="baseline", pad=0, sep=0)
    if transform is None:
        transform = where.transAxes if hasattr(where, "transAxes") else where.transFigure
    ax_ = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]
    ay_ = {"bottom": 0.0, "center": 0.5, "top": 1.0, "baseline": 0.2}[va]
    ab = AnnotationBbox(box, (x, y), xycoords=transform, frameon=False, box_alignment=(ax_, ay_), pad=0,
                        annotation_clip=False, zorder=zorder)
    where.add_artist(ab)
    return ab


# ---------------------------------------------------------------- the card


@dataclass
class Layout:
    """The card's content region, in inches from each edge of the figure."""

    width: float
    height: float
    left: float
    right: float
    top: float
    bottom: float

    def cell(self, x0=0.0, y0=0.0, x1=1.0, y1=1.0):
        """A rectangle of the content region given in content fractions, as inches (left, bottom, w, h)."""
        cw = self.width - self.left - self.right
        ch = self.height - self.top - self.bottom
        return self.left + x0 * cw, self.bottom + y0 * ch, (x1 - x0) * cw, (y1 - y0) * ch


def card(title: str, subtitle: str = "", *, height: float = 8.0, width: float = WIDTH, footer: int = 0,
         margin: float = 0.62, gap: float = 0.22):
    """A figure drawn on the README card, with the title and subtitle set.

    ``footer`` reserves that many lines at the bottom for ``footnote`` or ``legend_row``.
    The content region is ``fig.box``; ``panel(fig)`` fills it, leaving room for tick labels and names.
    """
    setup()
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_alpha(0.0)
    inset = 0.025
    fig.add_artist(
        FancyBboxPatch((inset, inset), width - 2 * inset, height - 2 * inset,
                       boxstyle="round,pad=0,rounding_size=0.16", transform=fig.dpi_scale_trans,
                       facecolor=CARD, edgecolor=HAIR, linewidth=1.0, zorder=-100)
    )
    top = 0.5
    fig.text(margin / width, 1 - top / height, title, fontproperties=F.title, color=INK, ha="left", va="top")
    top += 0.52
    if subtitle:
        body = wrap(subtitle, F.subtitle, width - 2 * margin)
        fig.text(margin / width, 1 - top / height, body, fontproperties=F.subtitle, color=MUTED, ha="left",
                 va="top", linespacing=1.5)
        top += (body.count("\n") + 1) * 14.5 * 1.5 / 72
    bottom = 0.34 + footer * 0.32
    fig.box = Layout(width, height, margin, margin, top + gap, bottom)
    return fig


def style(ax, frame: str = "lb", grid: str = ""):
    """Strip an axes to the house style: ``frame`` names the spines kept (l, b, r, t); ``grid`` is "", "x", "y"."""
    ax.set_facecolor("none")
    for side, name in (("left", "l"), ("bottom", "b"), ("right", "r"), ("top", "t")):
        ax.spines[side].set_visible(name in frame)
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(1.1)
    ax.tick_params(axis="both", which="both", length=0, colors=MUTED, labelsize=12.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(F.tick)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    return ax


def _axes(fig, left, bottom, w, h, head, tail, lpad, rpad, frame, grid, **kwargs):
    box = fig.box
    rect = [(left + lpad) / box.width, (bottom + tail) / box.height,
            max(w - lpad - rpad, 0.1) / box.width, max(h - head - tail, 0.1) / box.height]
    return style(fig.add_axes(rect, **kwargs), frame, grid)


def panel(fig, rect=(0.0, 0.0, 1.0, 1.0), *, head: float = 0.55, tail: float = 0.75, lpad: float = 0.65,
          rpad: float = 0.15, frame: str = "lb", grid: str = "", **kwargs):
    """An axes in a cell of the card's content region. ``rect`` is (x0, y0, x1, y1) in content fractions.

    The paddings, in inches, leave room inside the cell: ``head`` above the axes for a panel title and a
    y name, ``tail`` below for tick labels and the x name, ``lpad`` at the left for y tick labels.
    Pass 0 for any the panel does not need.
    """
    return _axes(fig, *fig.box.cell(*rect), head, tail, lpad, rpad, frame, grid, **kwargs)


def panels(fig, nrows: int = 1, ncols: int = 1, *, wspace: float = 0.3, hspace: float = 0.25, widths=None,
           heights=None, head: float = 0.55, tail: float = 0.75, lpad: float = 0.65, rpad: float = 0.15,
           frame: str = "lb", grid: str = "", **kwargs):
    """A grid of axes over the content region, as a 2-D array. ``wspace`` and ``hspace`` are inches
    between cells; the paddings are as in ``panel``, applied inside every cell."""
    box = fig.box
    left0, bottom0, cw, ch = box.cell()
    widths = np.asarray(widths if widths is not None else [1.0] * ncols, float)
    heights = np.asarray(heights if heights is not None else [1.0] * nrows, float)
    ws = widths / widths.sum() * (cw - wspace * (ncols - 1))
    hs = heights / heights.sum() * (ch - hspace * (nrows - 1))
    axes = np.empty((nrows, ncols), dtype=object)
    top = bottom0 + ch
    for r in range(nrows):
        x = left0
        for c in range(ncols):
            axes[r, c] = _axes(fig, x, top - hs[r], ws[c], hs[r], head, tail, lpad, rpad, frame, grid, **kwargs)
            x += ws[c] + wspace
        top -= hs[r] + hspace
    return axes


def _above(ax, inches: float):
    from matplotlib.transforms import offset_copy

    return offset_copy(ax.transAxes, fig=ax.figure, y=inches, units="inches")


def axis_names(ax, x: str | None = None, y: str | None = None):
    """Name the axes: x under the axis, y set horizontally just above the axis's top-left corner."""
    if x:
        ax.set_xlabel(x, fontproperties=F.axis, color=INK, labelpad=10)
    if y:
        ax.set_ylabel("")
        ax.text(0.0, 1.0, y, transform=_above(ax, 0.08), fontproperties=F.axis, color=INK, ha="left",
                va="bottom", clip_on=False)


def panel_title(ax, text: str, *, color: str = INK, ha: str = "left", raise_by: float = 0.34):
    """A short title over one panel of a multi-panel figure, above the y name."""
    x = {"left": 0.0, "center": 0.5, "right": 1.0}[ha]
    return ax.text(x, 1.0, text, transform=_above(ax, raise_by), fontproperties=F.panel, color=color, ha=ha,
                   va="bottom", clip_on=False)


def footnote(fig, text: str, *, line: int = 0):
    """Muted small text in the footer lines reserved by ``card(footer=...)``. Line 0 is the lowest."""
    box = fig.box
    body = wrap(text, F.small, box.width - box.left - box.right)
    return fig.text(box.left / box.width, (0.34 + line * 0.32) / box.height, body, fontproperties=F.small,
                    color=MUTED, ha="left", va="bottom", linespacing=1.4)


def legend_row(fig, items, *, line: int = 0, y: float | None = None, x: float | None = None,
               ncol: int | None = None, spacing: float = 2.2):
    """A row of legend keys. ``items`` is ``[(kind, color, label), ...]`` with kind in
    "patch", "pale", "line", "dashed", "dotted", "dot", "bar". Placed on footer ``line`` (0 is the lowest)."""
    handles = []
    for kind, color, label in items:
        if kind in ("patch", "pale"):
            handles.append(Patch(facecolor=color, edgecolor="none", label=label))
        elif kind == "bar":
            handles.append(Line2D([], [], color=color, lw=4.5, solid_capstyle="butt", label=label))
        elif kind == "dot":
            handles.append(Line2D([], [], color=color, marker="o", lw=0, ms=7, markeredgecolor=CARD, label=label))
        else:
            style_ = {"line": "-", "dashed": (0, (5, 3)), "dotted": (0, (1, 2.4))}[kind]
            handles.append(Line2D([], [], color=color, lw=2.2, linestyle=style_, label=label))
    box = fig.box
    x = box.left / box.width if x is None else x
    y = (0.3 + 0.32 * line) / box.height if y is None else y
    return fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x, y), ncol=ncol or len(handles),
                      frameon=False, prop=F.label, handlelength=1.8, columnspacing=spacing, labelcolor=INK,
                      borderaxespad=0, handletextpad=0.6)


def value(ax, x, y, text: str, color: str = INK, *, ha: str = "left", va: str = "center", size=None, **kwargs):
    """A number printed in the panel, in the colour of the thing it describes."""
    prop = F.value if size is None else font("SemiBold", size)
    return ax.text(x, y, text, fontproperties=prop, color=color, ha=ha, va=va, **kwargs)


def code(ax, x, y, text: str, color: str = INK, *, bold: bool = False, **kwargs):
    """An API name or call, in JetBrains Mono."""
    return ax.text(x, y, text, fontproperties=F.mono_bold if bold else F.mono, color=color, **kwargs)


# ---------------------------------------------------------------- figure forms


def ridgeline(ax, x, densities, *, gap: float = 1.0, height: float = 1.5, inner=None, color: str = TEAL,
              pale: str | None = None, edge: str | None = None, scale: str = "row", lw: float = 1.4,
              floor: float = 0.004):
    """One ridge per row of ``densities`` (rows, len(x)); row 0 sits at the bottom, in front.

    ``inner`` is a (lo, hi) per row, drawn solid with the tails pale. ``scale`` is "row" (each ridge
    to its own peak) or "shared". Returns the baseline height of each row, for bars and dots under it.
    """
    densities = np.asarray(densities, float)
    if pale is None:
        pale = TEAL_PALE if color == TEAL else mix(color, CARD, 0.72)
    edge = edge or (TEAL_DARK if color == TEAL else shade(color, -0.25))
    peak = densities.max(axis=1, keepdims=True) if scale == "row" else densities.max()
    rows = densities / np.where(peak > 0, peak, 1.0) * height
    bases = np.arange(len(rows)) * gap
    for r in range(len(rows) - 1, -1, -1):  # the back row first, so nearer ridges cover it
        z = 3 * (len(rows) - r)
        base, p = bases[r], rows[r]
        ax.fill_between(x, base, base + p, color=CARD, lw=0, zorder=z)
        ax.fill_between(x, base, base + p, color=pale, lw=0, zorder=z + 0.1)
        if inner is not None:
            lo, hi = inner[r]
            ax.fill_between(x, base, base + p, where=(x >= lo) & (x <= hi), color=color, lw=0, zorder=z + 0.2,
                            interpolate=True)
        ax.plot(x, np.where(p > floor * height, base + p, np.nan), color=edge, lw=lw, zorder=z + 0.3)
    return bases


def fan(ax, x, quantiles, *, color: str = TEAL, median: bool = True, lw: float = 1.8, alpha_max: float = 0.55,
        zorder: float = 1):
    """Nested quantile bands: ``quantiles`` is (levels, len(x)), sorted, an odd count with the median in
    the middle. Bands between symmetric levels, darker toward the median."""
    q = np.asarray(quantiles, float)
    n = len(q)
    pairs = n // 2
    for k in range(pairs):
        alpha = alpha_max * (k + 1) / pairs
        ax.fill_between(x, q[k], q[n - 1 - k], color=color, alpha=alpha * 0.55, lw=0, zorder=zorder + k * 0.01)
    if median and n % 2:
        ax.plot(x, q[n // 2], color=shade(color, -0.2), lw=lw, zorder=zorder + 1)


def iso(i, j, z=0.0, *, course: float = 0.652):
    """The logo's isometric projection: cell (i, j) at height z (in courses) to plot coordinates."""
    i, j, z = np.asarray(i, float), np.asarray(j, float), np.asarray(z, float)
    return (i - j) * 0.866, -(i + j) * 0.5 + z * course


def isometric_blocks(ax, heights, colors, *, course: float = 0.652, floor: bool = True, lw: float = 0.45,
                     edge_tone: float = -0.42, tones=(0.16, -0.12, -0.30), lift: float = 0.22):
    """Stack blocks on a G x G floor exactly as the logo does.

    ``heights`` (G, G) holds whole courses per cell; ``colors`` is one colour or a (G, G) array of colours.
    The +i face is the right-front face, the +j face the left-front one. ``tones`` shade the top, the
    left-front, and the right-front faces; ``lift`` lightens higher courses, as in the logo.
    """
    heights = np.asarray(heights, int)
    g = heights.shape[0]
    colors = np.full(heights.shape, colors, dtype=object) if isinstance(colors, str) else np.asarray(colors, object)
    zmax = max(int(heights.max()), 1)
    polys, faces, edges = [], [], []

    def quad(points):
        xs, ys = iso(*np.array(points).T, course=course)
        return np.column_stack([xs, ys])

    if floor:
        for i in range(g):
            for j in range(g):
                polys.append(quad([(i, j, 0), (i + 1, j, 0), (i + 1, j + 1, 0), (i, j + 1, 0)]))
                faces.append(FLOOR)
                edges.append(FLOOR_EDGE)
    for diagonal in range(2 * g - 1):  # back to front
        for i in range(g):
            j = diagonal - i
            if not 0 <= j < g or heights[i, j] <= 0:
                continue
            base = colors[i, j]
            for z in range(heights[i, j]):
                c = shade(base, lift * z / zmax - 0.06)
                e = shade(c, edge_tone)
                left = [(i, j + 1, z), (i + 1, j + 1, z), (i + 1, j + 1, z + 1), (i, j + 1, z + 1)]
                right = [(i + 1, j, z), (i + 1, j + 1, z), (i + 1, j + 1, z + 1), (i + 1, j, z + 1)]
                polys += [quad(left), quad(right)]
                faces += [shade(c, tones[1]), shade(c, tones[2])]
                edges += [e, e]
                if z == heights[i, j] - 1:
                    top = [(i, j, z + 1), (i + 1, j, z + 1), (i + 1, j + 1, z + 1), (i, j + 1, z + 1)]
                    polys.append(quad(top))
                    faces.append(shade(c, tones[0]))
                    edges.append(e)
    collection = PolyCollection(polys, facecolors=faces, edgecolors=edges, linewidths=lw, joinstyle="round")
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.axis("off")
    xs, ys = iso(np.array([0, g, g, 0]), np.array([0, 0, g, g]), course=course)
    ax.set_xlim(xs.min() - 1, xs.max() + 1)
    ax.set_ylim(ys.min() - 1, ys.max() + zmax * course + 1)
    return collection


def stacks(mass, courses: int = 12):
    """Whole courses per cell from a mass grid, as the logo rounds them: the tallest cell gets ``courses``."""
    mass = np.asarray(mass, float)
    return np.round(courses * mass / mass.max()).astype(int)


# ---------------------------------------------------------------- thumbnails


def thumb(slug: str, label: str, number: list | str, draw_fn, *, out: Path | None = None):
    """A gallery thumbnail: the core mark only, on the card, with no text.

    ``label`` (about three words) and ``number`` (the result against the baseline, plain text or rich
    segments ``[(text, color), ...]``) record what the thumbnail stands for. They are not drawn: the
    gallery prints its link text and caption under the image, where they stay readable on a phone,
    so baked-in text would only repeat them at a tenth of the size. ``draw_fn(ax)`` draws the mark.
    Written to ``docs/images/readme/thumbs/<slug>.png``.
    """
    setup()
    w, h = THUMB_SIZE
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_alpha(0.0)
    fig.add_artist(FancyBboxPatch((0.02, 0.02), w - 0.04, h - 0.04, boxstyle="round,pad=0,rounding_size=0.14",
                                  transform=fig.dpi_scale_trans, facecolor=CARD, edgecolor=HAIR, linewidth=1.0,
                                  zorder=-100))
    pad = 0.36
    ax = fig.add_axes([pad / w, pad / h, 1 - 2 * pad / w, 1 - 2 * pad / h])
    style(ax, frame="")
    ax.set_xticks([])
    ax.set_yticks([])
    draw_fn(ax)
    return save(fig, slug, folder=out or THUMBS)


# ---------------------------------------------------------------- saving


def save(fig, name: str, *, folder: Path | None = None, quantize: bool = True) -> Path:
    """Write ``docs/images/readme/<name>.png`` (transparent outside the card), shrunk to 256 colours."""
    folder = Path(folder or IMAGES)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    if quantize:
        from PIL import Image

        image = Image.open(path).convert("RGBA")
        image.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
    size = path.stat().st_size
    note = "" if size <= MAX_BYTES else f"  (over {MAX_BYTES // 1000} KB: simplify the figure)"
    print(f"wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path} {size / 1000:.0f} KB{note}")
    return path


# ---------------------------------------------------------------- artifacts


def artifact_dir(figure: str) -> Path:
    path = ARTIFACTS / figure
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plain(obj, digits: int):
    if isinstance(obj, dict):
        return {str(k): _plain(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v, digits) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist(), digits)
    if isinstance(obj, (np.floating, float)):
        value_ = float(obj)
        return None if not np.isfinite(value_) else float(f"{value_:.{digits}g}")
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def save_json(figure: str, name: str, obj, *, digits: int = 6) -> Path:
    """Write ``artifacts/<figure>/<name>.json``, NumPy arrays as lists, floats to ``digits`` significant figures."""
    path = artifact_dir(figure) / f"{name}.json"
    path.write_text(json.dumps(_plain(obj, digits), separators=(",", ":")) + "\n")
    print(f"wrote {path.relative_to(ROOT)} {path.stat().st_size / 1000:.0f} KB")
    return path


def load_json(figure: str, name: str):
    return json.loads((ARTIFACTS / figure / f"{name}.json").read_text())


def save_npz(figure: str, name: str, **arrays) -> Path:
    path = artifact_dir(figure) / f"{name}.npz"
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print(f"wrote {path.relative_to(ROOT)} {path.stat().st_size / 1000:.0f} KB")
    return path


def load_npz(figure: str, name: str) -> dict:
    with np.load(ARTIFACTS / figure / f"{name}.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# ---------------------------------------------------------------- numbers


def pct(x: float, digits: int = 0) -> str:
    """0.803 -> "80%"."""
    return f"{100 * x:.{digits}f}%"


def signed(x: float, digits: int = 1, unit: str = "%") -> str:
    """A change with a true minus sign: -20.04 -> "−20.0%"."""
    text = f"{x:+.{digits}f}{unit}"
    return text.replace("-", "−")


def num(x: float, digits: int = 0) -> str:
    """Thousands separators and a true minus sign: 1093.4 -> "1,093"."""
    return f"{x:,.{digits}f}".replace("-", "−")


# ---------------------------------------------------------------- scripts


def main(compute=None, draw=None):
    """``python examples/readme/<figure>.py compute|draw``: run one stage of a figure script."""
    stages = {name: fn for name, fn in (("compute", compute), ("draw", draw)) if fn is not None}
    if len(sys.argv) < 2 or sys.argv[1] not in stages:
        raise SystemExit(f"usage: python {Path(sys.argv[0]).name} {'|'.join(stages)}")
    stages[sys.argv[1]]()


# ---------------------------------------------------------------- specimen


def specimen(path: Path):
    """Every helper on one card, for checking the kit by eye."""
    from scipy.stats import norm

    setup()
    fig = card(
        "The kit: one card, fixed colours, Inter and JetBrains Mono.",
        "Teal is Ashlar, clay the strongest baseline, slate a second baseline, warm grey the data, and ink what "
        "happened. A ridgeline, a quantile fan, the logo's isometric blocks, and labels in place of legends.",
        height=8.8,
        footer=2,
    )
    axes = panels(fig, 1, 3, wspace=0.35, widths=[1.2, 1.0, 1.0])
    rng = np.random.default_rng(0)
    x = np.linspace(-30, 200, 500)
    dens, inner = [], []
    for r in range(6):
        d = 0.75 * np.exp(-0.5 * ((x - 2) / 7) ** 2) + 0.25 * (r + 1) / 6 * np.exp(-0.5 * ((x - 25 * r) / (8 + 6 * r)) ** 2)
        dens.append(d)
        cdf = np.cumsum(d) / d.sum()
        inner.append((x[np.searchsorted(cdf, 0.1)], x[np.searchsorted(cdf, 0.9)]))
    ax = axes[0, 0]
    bases = ridgeline(ax, x, dens, inner=inner, gap=1.0, height=1.55)
    for r, b in enumerate(bases):
        mid = 12 + 5 * r
        ax.plot([mid - 37, mid + 37], [b - 0.17] * 2, color=CLAY, lw=3.2, solid_capstyle="butt", zorder=200)
        ax.scatter([rng.uniform(-5, 60)], [b - 0.17], s=28, color=INK, edgecolor=CARD, lw=1.0, zorder=201)
    style(ax, frame="b")
    ax.set_yticks([])
    ax.set_xlim(-30, 200)
    ax.set_ylim(-0.5, bases[-1] + 1.8)
    axis_names(ax, x="delay, minutes")
    panel_title(ax, "ridgeline")
    ax = axes[0, 1]
    s = np.linspace(50, 150, 60)
    levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    q = np.array([6 + 0.012 * s + (0.8 + 0.004 * s) * norm.ppf(level) for level in levels])
    fan(ax, s, q)
    ax.plot(s, 6.4 + 0.01 * s + 0.15 * np.sin(s / 9), color=CLAY, lw=2.0, linestyle=(0, (5, 3)))
    value(ax, 0.98, 0.05, "0 violations", TEAL, ha="right", transform=ax.transAxes)
    style(ax, frame="lb", grid="y")
    axis_names(ax, x="bonus-malus", y="log cost")
    panel_title(ax, "fan")
    ax = axes[0, 2]
    g = 16
    u = (np.arange(g) + 0.5) / g
    uu, vv = np.meshgrid(u, u, indexing="ij")
    a = np.exp(-0.5 * ((uu - 0.66) ** 2 + (vv - 0.66) ** 2) / 0.15**2)
    b = 1.3 * np.exp(-0.5 * ((uu - 0.3) ** 2 + (vv - 0.32) ** 2) / 0.095**2)
    isometric_blocks(ax, stacks(a + b), np.where(b > a, CLAY, TEAL))
    panel_title(ax, "isometric blocks")
    box = fig.box
    rich(fig, box.left / box.width, (0.34 + 0.32 + 0.1) / box.height,
         [("80%", TEAL, F.value), (" of held-out flights inside Ashlar's 80% interval, against ", MUTED, F.label),
          ("94%", CLAY, F.value), (" for the trees.  ", MUTED, F.label),
          ("model.predict_interval(X, coverage=0.8)", INK, F.mono)], va="bottom")
    legend_row(fig, [("patch", TEAL, "Ashlar's 80% interval"), ("pale", TEAL_PALE, "its tails"),
                     ("bar", CLAY, "strongest baseline"), ("dashed", SLATE, "second baseline"),
                     ("patch", DATA, "data"), ("dot", INK, "what happened")])
    fig.savefig(path, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    specimen(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("kit-specimen.png"))
