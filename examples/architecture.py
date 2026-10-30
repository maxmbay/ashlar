"""Draw the README architecture diagram: four courses of choices, one lit path.

A model is one set of keyword arguments. The diagram lays the arguments out as
four courses of stone blocks, one course per slot, and lights the path that the
code beside it picks. Colors follow the logo: clay for the lit path, teal for
the heads that return a distribution, pale stone for the rest.

    python examples/architecture.py

Writes ``docs/images/architecture-light.svg`` and ``architecture-dark.svg``.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

THEMES = {
    "light": {"bg": "none", "card": "#fbfaf7"},
    # Dark mode shows the same light card, so the diagram looks the same everywhere.
    "dark": {"bg": "none", "card": "#fbfaf7"},
}

INK = "#1b1a17"
MUTED = "#5f5a52"
FLOOR = "#e9e4dc"
STONE = "#f3efe8"
CLAY = "#d4502a"
TEAL = "#147a72"

CH = 7.3  # px per character of the 12px mono label
PAD = 13.0  # px of block padding each side
BH = 34.0  # block height
GAP = 7.0  # px between blocks
VGAP = 52.0  # px between courses, enough for the slot label and the path's lane
LEFT = 34.0
TOP = 30.0
PANEL = 348.0  # width of the code panel on the right

# (slot, argument, [(label, lit, accent)])
COURSES = [
    (
        "encode",
        "num_embeddings, categorical_embeddings, cyclical",
        [
            ("piecewise", True, False),
            ("periodic", False, False),
            ("linear_relu", False, False),
            ("category vectors", False, False),
            ("one-hot", False, False),
            ("cyclical", False, False),
            ("set columns", False, False),
        ],
    ),
    (
        "backbone",
        "backbone, backbone_kwargs",
        [
            ("residual_mlp", True, False),
            ("tabm", False, False),
            ("transformer", False, False),
            ("set", False, False),
            ("your Equinox module", False, False),
        ],
    ),
    (
        "head",
        "task, head",
        [
            ("point", False, False),
            ("classes", False, False),
            ("ordinal", False, False),
            ("outcome tree", False, False),
            ("gaussian", False, True),
            ("mixture", True, True),
            ("flow", False, True),
            ("histogram", False, True),
            ("quantile", False, True),
            ("your head", False, False),
        ],
    ),
    (
        "train",
        "learning_rate_schedule, batch_size, patience, search",
        [
            ("cosine", True, False),
            ("constant", False, False),
            ("linear", False, False),
            ("auto batch", False, False),
            ("patience", False, False),
            ("sample weights", False, False),
            ("grid search", False, False),
        ],
    ),
]

CODE = [
    [("ASRRegressor", CLAY), ("(", MUTED)],
    [("    num_embeddings", INK), ("=", MUTED), ('"piecewise"', TEAL), (",", MUTED)],
    [("    backbone", INK), ("=", MUTED), ('"residual_mlp"', TEAL), (",", MUTED)],
    [("    task", INK), ("=", MUTED), ('"mixture"', TEAL), (",", MUTED)],
    [(")", MUTED), (".fit", INK), ("(X, y)", MUTED)],
]

CAPTION = [
    "Every slot is a keyword argument.",
    "Change one and the rest stays put:",
    "task=\"flow\" for a skewed target,",
    "backbone=\"tabm\" for the ensemble.",
]


def _width(label: str) -> float:
    return len(label) * CH + 2 * PAD


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _block(x: float, y: float, w: float, label: str, lit: bool, accent: bool) -> list[str]:
    fill = CLAY if lit else STONE
    stroke = CLAY if lit else (TEAL if accent else FLOOR)
    text = "#ffffff" if lit else (TEAL if accent else MUTED)
    weight = 500 if lit else 400
    parts = [
        (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{BH:.1f}" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        ),
        # A lit block gets a brighter top edge, the way a course of stone catches light.
        (
            f'<path d="M{x + 3:.1f} {y + 1.0:.1f}H{x + w - 3:.1f}" stroke="#ffffff" '
            f'stroke-opacity="{0.45 if lit else 0.9:.2f}" stroke-width="1.4"/>'
        ),
        (
            f'<text x="{x + w / 2:.1f}" y="{y + BH / 2 + 4.2:.1f}" text-anchor="middle" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12" '
            f'font-weight="{weight}" fill="{text}">{_escape(label)}</text>'
        ),
    ]
    return parts


def draw(theme: str) -> str:
    colors = THEMES[theme]
    rows = []
    width = 0.0
    y = TOP + 26.0
    for slot, argument, options in COURSES:
        x = LEFT
        row = {"slot": slot, "argument": argument, "y": y, "blocks": []}
        for label, lit, accent in options:
            w = _width(label)
            row["blocks"].append((x, w, label, lit, accent))
            x += w + GAP
        width = max(width, x - GAP)
        rows.append(row)
        y += BH + VGAP

    total_w = width + 36.0 + PANEL
    total_h = y - VGAP + 34.0
    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w:.0f} {total_h:.0f}" '
            f'width="{total_w:.0f}" height="{total_h:.0f}" role="img" '
            f'aria-label="Four courses of model choices with one path lit">'
        ),
        f'<rect width="{total_w:.0f}" height="{total_h:.0f}" rx="10" fill="{colors["card"]}"/>',
    ]

    # The lit path, drawn under the blocks so it reads as one line through the courses.
    lit_points = []
    for row in rows:
        for x, w, _label, lit, _accent in row["blocks"]:
            if lit:
                lit_points.append((x + w / 2, row["y"]))
    path = []
    for index, (cx, cy) in enumerate(lit_points):
        if index:
            px, py = lit_points[index - 1]
            # Orthogonal: down, across a clear lane above the next slot label, then down.
            lane = py + BH + 16.0
            turn = 6.0 if cx > px else -6.0
            path.append(
                f"M{px:.1f} {py + BH:.1f}V{lane - 6:.1f}Q{px:.1f} {lane:.1f} {px + turn:.1f} {lane:.1f}"
                f"H{cx - turn:.1f}Q{cx:.1f} {lane:.1f} {cx:.1f} {lane + 6:.1f}V{cy:.1f}"
                if abs(cx - px) > 14
                else f"M{px:.1f} {py + BH:.1f}V{cy:.1f}"
            )
    out.append(
        f'<path d="{"".join(path)}" fill="none" stroke="{CLAY}" stroke-width="2" stroke-opacity="0.55"/>'
    )

    for row in rows:
        out.append(
            f'<text x="{LEFT:.1f}" y="{row["y"] - 11:.1f}" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" '
            f'letter-spacing="1.3" fill="{INK}">{_escape(row["slot"].upper())}</text>'
        )
        out.append(
            f'<text x="{LEFT + len(row["slot"]) * 7.9 + 16:.1f}" y="{row["y"] - 11:.1f}" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="10" '
            f'fill="{MUTED}">{_escape(row["argument"])}</text>'
        )
        for x, w, label, lit, accent in row["blocks"]:
            out.extend(_block(x, row["y"], w, label, lit, accent))

    # The code panel: the arguments that light the path above.
    px = width + 36.0
    py = TOP + 26.0
    out.append(
        f'<path d="M{px - 18:.1f} {py:.1f}V{rows[-1]["y"] + BH:.1f}" stroke="{FLOOR}" stroke-width="1"/>'
    )
    for line in CODE:
        tx = px
        for piece, color in line:
            out.append(
                f'<text x="{tx:.1f}" y="{py + 12:.1f}" xml:space="preserve" '
                f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="12.5" '
                f'fill="{color}">{_escape(piece)}</text>'
            )
            tx += len(piece) * 7.6
        py += 22.0
    py += 14.0
    for line in CAPTION:
        out.append(
            f'<text x="{px:.1f}" y="{py + 12:.1f}" '
            f'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="11" '
            f'fill="{MUTED}">{_escape(line)}</text>'
        )
        py += 17.0

    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = OUT / f"architecture-{theme}.svg"
        path.write_text(draw(theme), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
