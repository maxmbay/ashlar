"""Point the README's relative links at GitHub, for PyPI: ``python examples/readme/absolute_urls.py``.

On GitHub the README links its images and docs by relative path, so every branch shows its own
figures. PyPI shows the README without the repository around it, so the copy that goes into a
release build needs absolute links. This rewrites the file in place:

    docs/images/readme/hero.png   ->  https://raw.githubusercontent.com/maxmbay/ashlar/<ref>/docs/images/readme/hero.png
    docs/benchmarks.md#scores     ->  https://github.com/maxmbay/ashlar/blob/<ref>/docs/benchmarks.md#scores
    examples/                     ->  https://github.com/maxmbay/ashlar/tree/<ref>/examples/

in Markdown links, images, and link definitions, and in HTML ``src``, ``srcset``, and ``href``
attributes. In-page anchors (``#...``) and links that are already absolute are left alone.

    --ref REF    the branch, tag, or commit the links point at (default: main)
    --check      rewrite nothing; fail if a relative link names a missing file or an anchor that no
                 heading makes, in the page itself or in a linked Markdown file
    --relative   the reverse: absolute links into this repository become relative paths
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO = "maxmbay/ashlar"
ROOT = Path(__file__).resolve().parents[2]
IMAGES = (".png", ".svg", ".gif", ".jpg", ".jpeg", ".webp")

MARKDOWN = re.compile(r'(\]\()([^)\s]+)((?:\s+"[^"]*")?\))')  # [text](target "title") and ![alt](target)
DEFINITION = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*)(\S+)", re.MULTILINE)  # [name]: target
ATTRIBUTE = re.compile(r'(\b(?:src|href)=")([^"]+)(")')
SRCSET = re.compile(r'(\bsrcset=")([^"]+)(")')
SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
OURS = re.compile(
    rf"https://(?:raw\.githubusercontent\.com/{REPO}/[^/]+/|github\.com/{REPO}/(?:blob|tree)/[^/]+/)(.*)"
)


def is_relative(target: str) -> bool:
    return bool(target) and not (SCHEME.match(target) or target.startswith(("#", "/")))


def absolute(target: str, ref: str) -> str:
    if not is_relative(target):
        return target
    path, hash_, anchor = target.partition("#")
    path = path.removeprefix("./")
    if path.lower().endswith(IMAGES):
        base = f"https://raw.githubusercontent.com/{REPO}/{ref}/"
    elif path.endswith("/") or (ROOT / path).is_dir():
        base = f"https://github.com/{REPO}/tree/{ref}/"
    else:
        base = f"https://github.com/{REPO}/blob/{ref}/"
    return base + path + hash_ + anchor


def relative(target: str) -> str:
    match = OURS.fullmatch(target)
    return match.group(1) if match else target


FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)


def rewrite(text: str, fn) -> str:
    """Apply ``fn`` to every link target outside fenced code blocks."""
    out, at = [], 0
    for block in FENCE.finditer(text):
        out += [_rewrite(text[at : block.start()], fn), block.group(0)]
        at = block.end()
    return "".join(out + [_rewrite(text[at:], fn)])


def _rewrite(text: str, fn) -> str:
    def one(match):
        return match.group(1) + fn(match.group(2)) + (match.group(3) if match.lastindex >= 3 else "")

    def candidates(match):
        parts = []
        for candidate in match.group(2).split(","):
            lead = candidate[: len(candidate) - len(candidate.lstrip())]
            url, *descriptor = candidate.strip().split(maxsplit=1)
            parts.append(lead + " ".join([fn(url), *descriptor]))
        return match.group(1) + ",".join(parts) + match.group(3)

    for pattern in (MARKDOWN, DEFINITION, ATTRIBUTE):
        text = pattern.sub(one, text)
    return SRCSET.sub(candidates, text)


def targets(text: str) -> list[str]:
    found = []
    rewrite(text, lambda target: found.append(target) or target)
    return found


def slugs(markdown: str) -> set[str]:
    """The anchors GitHub makes for a Markdown file's headings, plus any ``id=`` or ``name=`` it sets."""
    seen: Counter = Counter()
    out = set(re.findall(r'\b(?:id|name)="([^"]+)"', markdown))
    fenced = False
    for line in markdown.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        heading = re.match(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if fenced or not heading:
            continue
        text = re.sub(r"<[^>]+>", "", heading.group(1))
        text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
        slug = re.sub(r"[^\w\- ]", "", text.strip().lower()).replace(" ", "-")
        out.add(slug if not seen[slug] else f"{slug}-{seen[slug]}")
        seen[slug] += 1
    return out


def problems(path: Path) -> list[str]:
    text = path.read_text()
    found = []
    for target in targets(text):
        if target.startswith("#"):
            if target[1:] not in slugs(text):
                found.append(f"{target}: no heading or id makes this anchor in {path.name}")
            continue
        if not is_relative(target):
            continue
        file, _, anchor = target.partition("#")
        where = (path.parent / file).resolve()
        if not where.exists():
            found.append(f"{target}: {file} does not exist")
        elif anchor and where.suffix == ".md" and anchor not in slugs(where.read_text()):
            found.append(f"{target}: no heading or id makes #{anchor} in {file}")
    return found


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", default=str(ROOT / "README.md"))
    parser.add_argument("--ref", default="main")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--relative", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.file)
    if args.check:
        found = problems(path)
        for line in found:
            print(line)
        if found:
            raise SystemExit(f"{len(found)} broken link(s) in {path}")
        print(f"{path}: every relative link and anchor resolves")
        return
    before = path.read_text()
    after = rewrite(before, relative if args.relative else (lambda t: absolute(t, args.ref)))
    changed = sum(a != b for a, b in zip(targets(before), targets(after)))
    path.write_text(after)
    print(f"{path}: rewrote {changed} link(s)")


if __name__ == "__main__":
    main(sys.argv[1:])
