"""The figures' style: the colours of Ashlar's README figures (examples/readme/style.py), set for print.

Teal marks the dressed model, clay the strongest baseline, slate a second one, warm greys everything else. Every
figure is 6.5 inches wide with 8 to 9 point text. Inter and JetBrains Mono are used when they are installed (looked
for in the usual font folders); otherwise matplotlib's DejaVu fonts. Figures go to out/figures/ as PDF and PNG.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c  # noqa: E402

TEAL, CLAY, SLATE = "#147a72", "#d4502a", "#6e7a86"
INK, MUTED = "#1b1a17", "#5f5a52"
GREY = "#8a8178"  # context marks
LIGHT = "#d8d1c6"  # stems, light fills
HAIR = "#e4e0da"  # gridlines and rules
TEAL_WASH = "#dfebe7"
WIDTH = 6.5  # inches

TICK, LABEL, NOTE = 8.0, 8.5, 8.0

_FOLDERS = [Path.home() / ".local" / "share" / "fonts", Path.home() / "Library" / "Fonts", Path("/Library/Fonts"),
            Path("/usr/share/fonts"), Path("/usr/local/share/fonts")]
_FOUND: dict[str, Path | None] = {}


def _find(stem: str) -> Path | None:
    if stem not in _FOUND:
        _FOUND[stem] = None
        for folder in _FOLDERS:
            if folder.is_dir():
                hits = [p for ext in (".otf", ".ttf") for p in folder.rglob(f"{stem}{ext}")]
                if hits:
                    _FOUND[stem] = hits[0]
                    break
    return _FOUND[stem]


def font(weight: str = "Regular", size: float = LABEL, mono: bool = False) -> FontProperties:
    path = _find(f"{'JetBrainsMono' if mono else 'Inter'}-{weight}")
    if path is not None:
        return FontProperties(fname=str(path), size=size)
    return FontProperties(family="DejaVu Sans Mono" if mono else "DejaVu Sans",
                          weight={"Regular": "normal", "Medium": "medium", "SemiBold": "semibold", "Bold": "bold"}[weight],
                          size=size)


def setup() -> None:
    for weight in ("Regular", "Medium", "SemiBold", "Bold"):
        for stem in (f"Inter-{weight}", f"JetBrainsMono-{weight}"):
            if _find(stem) is not None:
                font_manager.fontManager.addfont(str(_find(stem)))
    plt.rcParams.update({
        "font.family": font().get_name(),
        "font.size": LABEL,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK,
        "axes.linewidth": 0.6,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.unicode_minus": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "xtick.labelsize": TICK,
        "ytick.labelsize": TICK,
        "xtick.major.size": 2.5,
        "ytick.major.size": 0,
        "xtick.major.width": 0.6,
        "xtick.major.pad": 3,
        "ytick.major.pad": 4,
        "axes.labelsize": LABEL,
        "axes.labelpad": 4,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.fontsize": NOTE,
        "pdf.fonttype": 42,
        "savefig.dpi": 200,
    })


def clean(ax, left: bool = False, bottom: bool = True, grid: str = "") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(INK)
    if grid:
        ax.grid(True, axis=grid, color=HAIR, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def save(fig, name: str) -> str:
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(c.FIGURES / f"{name}.pdf", metadata={"Creator": "analysis/" + name, "CreationDate": None})
    fig.savefig(c.FIGURES / f"{name}.png", dpi=200)
    plt.close(fig)
    return f"wrote figures/{name}.pdf and figures/{name}.png"
