"""Draw the README's diagram of a dressed model: what ``ashlar.dressed().fit`` does, top to bottom.

Five courses of stone blocks, one per step, in the colors of ``examples/architecture.py``. Teal marks the two
parts the model adds, the label readers and the corrector; the rest is pale stone. Beside each course, what the
step does and what it picks on held-out rows.

    python examples/dressed_diagram.py

Writes ``docs/images/dressed-light.svg`` and ``dressed-dark.svg``.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "images"

# Dark mode shows the same light card, as the architecture diagram does.
THEMES = ("light", "dark")

CARD = "#fbfaf7"
INK = "#1b1a17"
MUTED = "#5f5a52"
FLOOR = "#e9e4dc"
STONE = "#f3efe8"
TEAL = "#147a72"
MONO = 'font-family="ui-monospace, SFMono-Regular, Menlo, monospace"'

WIDTH = 936.0
LEFT = 34.0
TOP = 30.0
COLUMN = 486.5  # width of the courses; five member blocks of the architecture diagram's width
BH = 34.0  # block height
GAP = 7.0  # px between blocks
LINE_GAP = 8.0  # px between the lines of one course
VGAP = 52.0  # px between courses
NOTES = LEFT + COLUMN + 30.0  # where the notes beside the courses start
PATH = LEFT + COLUMN - 46.0  # the line that joins the courses
RAIL = 9.0  # px from the members to the rail under them

# (step, fitted attribute, lines of (label, lit, share of the course's width), notes)
COURSES = [
    (
        "members",
        "members=5",
        [
            [(f"network {k}", False, 1) for k in range(1, 6)],
            [(f"reader {k}", True, 1) for k in range(1, 6)],
        ],
        [
            "Five pairs of a network and a label reader. Each",
            "pair trains on four fifths of the rows and holds",
            "out the rest. A reader predicts from the training",
            "rows' labels, not from weights on the columns.",
        ],
    ),
    (
        "pool",
        "pool_weight_",
        [[("w × networks + (1 − w) × readers", False, 1)]],
        [
            "One weight w for the table, picked by the CRPS of",
            "each row under the pair that held it out.",
        ],
    ),
    (
        "smooth",
        "bandwidth_factor_, fallback_factor_",
        [[("a normal on each reader point", False, 2.15), ("+ 1% wide normal", False, 1)]],
        [
            "A reader predicts points, so the pool has no",
            "density yet. Each point becomes a normal, with the",
            "width picked on the held-out rows.",
        ],
    ),
    (
        "correct",
        "corrector_",
        [[("corrector network, on the held-out pool's scale", True, 1)]],
        [
            "One more network learns where each label falls in",
            "its own held-out pool, and reshapes every row's",
            "pool to match.",
        ],
    ),
    (
        "gate",
        "gate_",
        [[("g × corrected + (1 − g) × smoothed", False, 1)]],
        [
            "One weight g, picked on held-out rows. Where the",
            "corrector did not help, g is 0 and the prediction",
            "is the smoothed pool.",
        ],
    ),
]

CODE = [
    [("model", INK), (" = ", MUTED), ("ashlar.dressed", INK), ("().", MUTED), ("fit", INK), ("(X, y)", MUTED)],
    [("dist", INK), (" = ", MUTED), ("model.predict_dist", INK), ("(X_new)", MUTED)],
    [("dist.log_prob", INK), ("(y_new)", MUTED), ("              # exact log density", MUTED)],
    [("dist.cdf", INK), ("(y_new)", MUTED), ("                   # exact CDF", MUTED)],
    [("dist.quantile", INK), ("([0.1, 0.5, 0.9])", MUTED)],
]

CAPTION = [
    "Teal marks what this model adds: the label readers,",
    "and a corrector that works on the scale of the",
    "pairs' held-out pool. Each choice after the members",
    "is made on rows the pairs did not train on.",
]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _block(x: float, y: float, w: float, label: str, lit: bool) -> list[str]:
    fill, stroke, text = (TEAL, TEAL, "#ffffff") if lit else (STONE, FLOOR, MUTED)
    return [
        (
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{BH:.1f}" rx="3" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
        ),
        # A brighter top edge, the way a course of stone catches light.
        (
            f'<path d="M{x + 3:.1f} {y + 1.0:.1f}H{x + w - 3:.1f}" stroke="#ffffff" '
            f'stroke-opacity="{0.45 if lit else 0.9:.2f}" stroke-width="1.4"/>'
        ),
        (
            f'<text x="{x + w / 2:.1f}" y="{y + BH / 2 + 4.2:.1f}" text-anchor="middle" {MONO} font-size="12" '
            f'font-weight="{500 if lit else 400}" fill="{text}">{_escape(label)}</text>'
        ),
    ]


def draw() -> str:
    rows = []
    y = TOP + 26.0
    for step, attribute, lines, notes in COURSES:
        blocks, line_y = [], y
        for line in lines:
            unit = (COLUMN - GAP * (len(line) - 1)) / sum(share for _, _, share in line)
            x = LEFT
            for label, lit, share in line:
                blocks.append((x, line_y, unit * share, label, lit))
                x += unit * share + GAP
            line_y += BH + LINE_GAP
        bottom = line_y - LINE_GAP
        rows.append({"step": step, "attribute": attribute, "y": y, "bottom": bottom, "blocks": blocks, "notes": notes})
        y = bottom + VGAP

    code_top = rows[-1]["bottom"] + 38.0
    height = code_top + max(len(CODE) * 22.0, len(CAPTION) * 17.0) + 24.0
    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH:.0f} {height:.0f}" width="{WIDTH:.0f}" '
            f'height="{height:.0f}" role="img" aria-label="The five steps of a dressed model, top to bottom">'
        ),
        f'<rect width="{WIDTH:.0f}" height="{height:.0f}" rx="10" fill="{CARD}"/>',
        # The line that joins the courses, drawn under the blocks so it shows only between them. It starts from a
        # rail under the members, since all ten of them go into the pool.
        (
            f'<path d="M{LEFT + 3:.1f} {rows[0]["bottom"] + RAIL:.1f}H{LEFT + COLUMN - 3:.1f}'
            f'M{PATH:.1f} {rows[0]["bottom"] + RAIL:.1f}V{rows[-1]["y"]:.1f}" fill="none" stroke="{TEAL}" '
            f'stroke-width="2" stroke-opacity="0.55" stroke-linecap="round"/>'
        ),
    ]
    for row in rows:
        label_y = row["y"] - 11.0
        out.append(
            f'<text x="{LEFT:.1f}" y="{label_y:.1f}" {MONO} font-size="11" letter-spacing="1.3" '
            f'fill="{INK}">{_escape(row["step"].upper())}</text>'
        )
        out.append(
            f'<text x="{LEFT + len(row["step"]) * 7.9 + 16:.1f}" y="{label_y:.1f}" {MONO} font-size="10" '
            f'fill="{MUTED}">{_escape(row["attribute"])}</text>'
        )
        for x, line_y, w, label, lit in row["blocks"]:
            out.extend(_block(x, line_y, w, label, lit))
        for index, note in enumerate(row["notes"]):
            out.append(
                f'<text x="{NOTES:.1f}" y="{label_y + index * 16.0:.1f}" {MONO} font-size="11" '
                f'fill="{MUTED}">{_escape(note)}</text>'
            )

    # Under the courses: the call that runs them, and the caption beside it.
    out.append(
        f'<path d="M{LEFT:.1f} {code_top - 20:.1f}H{WIDTH - LEFT:.1f}" stroke="{FLOOR}" stroke-width="1"/>'
    )
    top = code_top
    for line in CODE:
        x = LEFT
        for piece, color in line:
            out.append(
                f'<text x="{x:.1f}" y="{top + 12:.1f}" xml:space="preserve" {MONO} font-size="12.5" '
                f'fill="{color}">{_escape(piece)}</text>'
            )
            x += len(piece) * 7.6
        top += 22.0
    top = code_top
    for line in CAPTION:
        out.append(
            f'<text x="{NOTES:.1f}" y="{top + 12:.1f}" {MONO} font-size="11" fill="{MUTED}">{_escape(line)}</text>'
        )
        top += 17.0
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        path = OUT / f"dressed-{theme}.svg"
        path.write_text(draw(), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
