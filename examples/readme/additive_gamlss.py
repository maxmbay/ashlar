"""A model you can read: Ashlar's additive negative binomial beside R's gamlss, term by term, on bike rentals.

The README's additive figure. Two stages, run from the repo root:

    JAX_PLATFORMS=cpu python examples/readme/additive_gamlss.py compute   # needs R with gamlss: the Mac
    python examples/readme/additive_gamlss.py draw                        # needs Inter: the cloud

``compute`` is the ``terms`` stage of ``examples/benchmark_additive.py``: it
refits both models on one of the benchmark's splits and writes every term of
the log mean and the dispersion, on a grid of its columns and centred on the
training rows, with the benchmark's saved scores and seconds for every split,
to ``examples/readme/artifacts/additive-gamlss/terms.json``. Option ``seed=N``
picks the split; by default it is the one whose saved gamlss pair fit was fastest.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

FIGURE = "additive-gamlss"
ARTIFACTS = HERE / "artifacts" / FIGURE


def compute(seed=None):
    import benchmark_additive

    benchmark_additive.run_terms(None if seed is None else int(seed), ARTIFACTS)


def draw():
    raise SystemExit("the draw stage arrives with the style kit, examples/readme/style.py")


def main(argv):
    stages = [arg for arg in argv if "=" not in arg] or ["draw"]
    options = dict(arg.split("=", 1) for arg in argv if "=" in arg)
    for stage in stages:
        if stage == "compute":
            compute(options.get("seed"))
        elif stage == "draw":
            draw()
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
