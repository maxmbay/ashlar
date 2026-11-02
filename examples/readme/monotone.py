"""Figure ``monotone``: a constraint on the whole distribution, on French motor insurance (freMTPL2).

Two stages, run from the repository root::

    JAX_PLATFORMS=cpu python examples/readme/monotone.py compute   # a few minutes on four cores
    python examples/readme/monotone.py draw                        # seconds; needs the style kit's fonts

``compute`` takes split 0 of ``examples/benchmark_monotone.py`` (seed 0: the
log cost per claim of every training policy with a claim, 2,000 test
policies with a claim as probes) and fits, as that benchmark does:

- ``quantile_spline``, without and with ``monotone={"BonusMalus": 1}``;
- one LightGBM quantile model per level of ``GBM_LEVELS``, without and with
  ``monotone_constraints`` rising in bonus-malus (the benchmark's ``gbm``
  stage).

For the first ``POLICIES`` probe policies it sets bonus-malus to every value
of ``BONUS_MALUS`` and writes, under ``examples/readme/artifacts/monotone/``,
``fans.json``: each model's quantiles of the cost per claim (in euros) along
that sweep, and the splines' ``P(cost > t)`` for the thresholds in
``THRESHOLDS``, with the split's scores.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_chain import CATEGORICAL, NUMERIC, policies  # noqa: E402
from benchmark_monotone import CONSTRAINT, GBM_LEVELS, GBMQuantiles, pinball, splits  # noqa: E402

NAME = "monotone"
ARTIFACTS = ROOT / "examples" / "readme" / "artifacts" / NAME
SEED = 0
POLICIES = 8
BONUS_MALUS = np.arange(50.0, 200.0001, 2.0)
LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
THRESHOLDS = (500.0, 1000.0, 2000.0, 5000.0, 10000.0)


def _round(values, digits: int = 5):
    array = np.asarray(values, dtype=float)
    return [float(f"{value:.{digits}g}") for value in array.ravel()] if array.ndim <= 1 else [_round(row, digits) for row in array]


def _swept(probe, index: int):
    frame = probe.iloc[[index] * len(BONUS_MALUS)].reset_index(drop=True)
    frame["BonusMalus"] = BONUS_MALUS
    return frame


def compute() -> None:
    from ashlar import ASRRegressor

    arguments = dict(argument.split("=", 1) for argument in sys.argv[2:] if "=" in argument)
    out = Path(arguments.get("out", ARTIFACTS))
    out.mkdir(parents=True, exist_ok=True)
    parts = splits(policies(), SEED, 200_000)
    claimed_train, claimed_test, probe = parts["claimed_train"], parts["claimed_test"], parts["probe_claimed"]
    x_train = claimed_train[NUMERIC + CATEGORICAL].reset_index(drop=True)
    y_train = np.log(claimed_train["S"].to_numpy())
    x_test = claimed_test[NUMERIC + CATEGORICAL].reset_index(drop=True)
    y_test = np.log(claimed_test["S"].to_numpy())
    print(f"{len(x_train):,} training and {len(x_test):,} test claims (split {SEED})", flush=True)

    models, scores = {}, {}
    for constrained in (False, True):
        name = f"quantile_spline{', monotone' if constrained else ''}"
        start = time.process_time()
        models[name] = ASRRegressor(
            task="quantile_spline", monotone=CONSTRAINT if constrained else None, random_state=SEED
        ).fit(x_train, y_train)
        q = models[name].predict_quantiles(x_test, list(LEVELS)).T
        scores[name] = {
            "nll": float(-np.mean(models[name].log_prob(x_test, y_test))),
            "pinball": float(np.mean([pinball(y_test, q[i], level).mean() for i, level in enumerate(LEVELS)])),
            "cpu": time.process_time() - start,
        }
        name = f"LightGBM quantiles{', monotone' if constrained else ''}"
        start = time.process_time()
        models[name] = GBMQuantiles(claimed_train, y_train, SEED, constrained, levels=LEVELS)
        q = models[name].quantiles(x_test)
        scores[name] = {
            "pinball": float(np.mean([pinball(y_test, q[i], level).mean() for i, level in enumerate(LEVELS)])),
            "test crossing": float(np.mean(np.any(np.diff(q, axis=0) < -1e-9, axis=0))),
            "cpu": time.process_time() - start,
        }
    for name, value in scores.items():
        print(name, json.dumps(value), flush=True)

    rows = []
    for index in range(POLICIES):
        frame = _swept(probe, index)
        entry = {
            "features": {key: (value.item() if hasattr(value, "item") else value)
                         for key, value in probe.iloc[index].items()},
            "quantiles": {},
            "exceedance": {},
        }
        for name, model in models.items():
            if name.startswith("LightGBM"):
                q = model.quantiles(frame)
            else:
                q = model.predict_quantiles(frame, list(LEVELS)).T
                entry["exceedance"][name] = _round(np.stack([
                    model.predict_exceedance(frame, float(np.log(threshold))) for threshold in THRESHOLDS
                ]))
            entry["quantiles"][name] = _round(np.exp(q))
        rows.append(entry)
    (out / "fans.json").write_text(json.dumps({
        "split": SEED,
        "bonus_malus": _round(BONUS_MALUS),
        "levels": list(LEVELS),
        "thresholds": list(THRESHOLDS),
        "policies": rows,
        "scores": scores,
        "training_claims": len(x_train),
        "bonus_malus_share_at_or_above": {str(int(v)): float(np.mean(claimed_train["BonusMalus"] >= v))
                                          for v in (100, 125, 150, 200)},
    }))
    print(f"{(out / 'fans.json')}: {(out / 'fans.json').stat().st_size / 1024:.0f} KB", flush=True)


def draw() -> None:
    raise SystemExit("not drawn yet")


if __name__ == "__main__":
    {"compute": compute, "draw": draw}[sys.argv[1] if len(sys.argv) > 1 else "draw"]()
