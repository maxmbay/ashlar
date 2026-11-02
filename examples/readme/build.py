"""Redraw the README figures from their committed artifacts: ``python examples/readme/build.py [name ...]``.

Each figure script in this folder has a ``draw`` stage that reads the small files under
``examples/readme/artifacts/`` and writes ``docs/images/readme/``; it never fits a model. This runs the
draw stage of every script, or of the ones named, each in its own process so one failure does not stop
the rest. Drawing needs Inter and JetBrains Mono (see ``style.py``).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOT_FIGURES = {"style.py", "build.py", "absolute_urls.py"}


def scripts(names=()) -> list[Path]:
    found = sorted(p for p in HERE.glob("*.py") if p.name not in NOT_FIGURES and "def draw" in p.read_text())
    if not names:
        return found
    wanted = {name.removesuffix(".py") for name in names}
    unknown = wanted - {p.stem for p in found}
    if unknown:
        raise SystemExit(f"no figure script named {', '.join(sorted(unknown))} in {HERE}")
    return [p for p in found if p.stem in wanted]


def main(names) -> None:
    failed = []
    for path in scripts(names):
        start = time.perf_counter()
        done = subprocess.run([sys.executable, str(path), "draw"], cwd=HERE.parents[1])
        status = "ok" if done.returncode == 0 else f"failed with exit code {done.returncode}"
        print(f"{path.name}: {status} in {time.perf_counter() - start:.1f}s", flush=True)
        if done.returncode:
            failed.append(path.name)
    if failed:
        raise SystemExit(f"{len(failed)} figure(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main(sys.argv[1:])
