"""Draw the README logo: a joint density stacked in isometric blocks.

A 16 x 16 grid of cells, the same count as ``head="histogram"`` per axis,
each stacked to the mass of a two-mode joint density. Blocks take their color
from the mode that claims them: clay for the tall, tight mode at the back and
teal for the broad one in front. Empty cells are a pale floor grid. The dark
file is the same art on a light card. The wordmark and tagline are JetBrains
Mono, drawn as glyph outlines so the SVG does not depend on installed fonts.

    python examples/logo.py --mono JetBrainsMono[wght].ttf

Writes ``docs/images/logo-light.svg`` and ``docs/images/logo-dark.svg``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

THEMES = {
    "light": dict(ink="#1b1a17", muted="#5f5a52", a="#147a72", b="#d4502a", floor="#e9e4dc", edge=-0.28),
    # Dark mode shows the light art on its own card, so the logo looks the same everywhere.
    "dark": dict(ink="#1b1a17", muted="#5f5a52", a="#147a72", b="#d4502a", floor="#e9e4dc", edge=-0.28, card="#fbfaf7"),
}

G = 16  # cells per side
S = 11.5  # px per cell edge
HZ = 7.5  # px per course
ZMAX = 12  # courses at the tallest cell
OX, OY = 190.0, 62.0  # back corner of the floor
# (weight, mean u, mean v, sigma) for each mode
BROAD = (0.58, 0.64, 0.68, 0.15)
TIGHT = (0.42, 0.30, 0.33, 0.095)


def _component(m, u, v):
    w, mu, mv, s = m
    return w * math.exp(-0.5 * ((u - mu) ** 2 + (v - mv) ** 2) / s**2) / s**2


def _proj(i, j, z):
    return OX + (i - j) * S * 0.866, OY + (i + j) * S * 0.5 - z * HZ


def _mix(a: str, b: str, t: float) -> str:
    t = max(0.0, min(1.0, t))
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def _shade(color: str, amount: float) -> str:
    return _mix(color, "#ffffff", amount) if amount > 0 else _mix(color, "#000000", -amount)


def _face(points, fill, edge):
    d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in points) + "Z"
    return f'<path d="{d}" fill="{fill}" stroke="{edge}" stroke-width="0.7" stroke-linejoin="round"/>'


def _text_path(font: TTFont, text: str, size: float, x: float, baseline: float, tracking: float = 0.0):
    glyphs = font.getGlyphSet()
    cmap = font.getBestCmap()
    scale = size / font["head"].unitsPerEm
    pen = SVGPathPen(glyphs)
    cursor = 0.0
    for char in text:
        name = cmap[ord(char)]
        glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, x + cursor, baseline)))
        cursor += glyphs[name].width * scale + tracking
    return pen.getCommands(), cursor - tracking


def draw(theme_name: str, light: TTFont, regular: TTFont) -> str:
    theme = THEMES[theme_name]
    cells = {}
    for i in range(G):
        for j in range(G):
            u, v = (i + 0.5) / G, (j + 0.5) / G
            broad, tight = _component(BROAD, u, v), _component(TIGHT, u, v)
            cells[i, j] = (broad + tight, tight / (broad + tight))
    peak = max(f for f, _ in cells.values())
    parts = []
    # Back to front, so nearer blocks cover farther ones.
    for diagonal in range(2 * G - 1):
        for i in range(G):
            j = diagonal - i
            if not 0 <= j < G:
                continue
            mass, share = cells[i, j]
            height = round(ZMAX * mass / peak)
            if height == 0:
                floor = [_proj(i, j, 0), _proj(i + 1, j, 0), _proj(i + 1, j + 1, 0), _proj(i, j + 1, 0)]
                parts.append(_face(floor, theme["floor"], _shade(theme["floor"], theme["edge"] / 3)).replace('0.7"', '0.6"'))
                continue
            base = _mix(theme["a"], theme["b"], (share - 0.25) / 0.5)
            for z in range(height):
                color = _shade(base, 0.22 * z / ZMAX - 0.08)
                edge = _shade(color, theme["edge"])
                left = [_proj(i, j + 1, z), _proj(i + 1, j + 1, z), _proj(i + 1, j + 1, z + 1), _proj(i, j + 1, z + 1)]
                right = [_proj(i + 1, j, z), _proj(i + 1, j + 1, z), _proj(i + 1, j + 1, z + 1), _proj(i + 1, j, z + 1)]
                parts.append(_face(left, _shade(color, -0.12), edge))
                parts.append(_face(right, _shade(color, -0.3), edge))
                if z == height - 1:
                    top = [_proj(i, j, z + 1), _proj(i + 1, j, z + 1), _proj(i + 1, j + 1, z + 1), _proj(i, j + 1, z + 1)]
                    parts.append(_face(top, _shade(color, 0.18), edge))
    text_x = OX + G * S * 0.866 + 50
    word, word_width = _text_path(light, "ashlar", 100, text_x - 5, 150, tracking=-2)
    tag, tag_width = _text_path(regular, "distributional models for tabular data", 16.5, text_x, 188)
    width = math.ceil(text_x + max(word_width, tag_width) + 30)
    height = math.ceil(OY + G * S + 14)
    blocks = "\n  ".join(parts)
    if "card" in theme:
        blocks = f'<rect x="1" y="1" width="{width - 2}" height="{height - 2}" rx="14" fill="{theme["card"]}"/>\n  ' + blocks
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-labelledby="title">
  <title id="title">ashlar</title>
  {blocks}
  <path d="{word}" fill="{theme["ink"]}"/>
  <path d="{tag}" fill="{theme["muted"]}"/>
</svg>
"""


def _instance(path: str, weight: float) -> TTFont:
    font = TTFont(path)
    if "fvar" in font:
        font = instantiateVariableFont(font, {"wght": weight})
    return font


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mono", required=True, help="JetBrains Mono, variable or static")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    light, regular = _instance(args.mono, 300), _instance(args.mono, 400)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in THEMES:
        (args.out / f"logo-{name}.svg").write_text(draw(name, light, regular))


if __name__ == "__main__":
    main()
