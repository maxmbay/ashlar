"""Do unconstrained heads break stochastic dominance in bonus-malus or vehicle power on freMTPL2?

The cheap test of #40. On French motor insurance (freMTPL2, see
``examples/benchmark_chain.py`` for the table), a higher bonus-malus level
means a worse claims history, so an insurer expects the whole distribution
of claims, not just its mean, to rise with it. Two outcomes:

- ``frequency``: the claim count ``N`` of every policy, with a
  ``negative_binomial`` head, on a random sample of ``rows`` policies
  (200,000 by default).
- ``severity``: the log of the average cost per claim, on every policy with
  a claim, with the ``gaussian``, ``mixture``, and ``quantile_spline`` heads.

Each head is fitted as it is, and again with ``monotone={"BonusMalus": 1}``.
On 2,000 test policies, each feature in turn is set to each of 12 values
from its 2nd to its 98th percentile, and the exceedance ``P(y > t)`` read
at a ladder of thresholds (``N`` above 0, 1, and 2; the log cost at its
5th to 95th percentiles). A violation is a step from one value to the next
at which some threshold's exceedance moves against the feature's overall
direction by more than 0.001, the direction being the sign of the mean's
average change across the grid. Reported: the share of steps with a
violation, the share of policies with at least one, the mean of each
policy's largest move against the direction, the share of steps where the
mean itself moves against it, and the test NLL.

Run with::

    JAX_PLATFORMS=cpu python examples/benchmark_monotone.py run seeds=3
    JAX_PLATFORMS=cpu python examples/benchmark_monotone.py report
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_chain import CATEGORICAL, NUMERIC, policies  # noqa: E402

from ashlar import ASRRegressor  # noqa: E402

RESULTS = Path(os.environ.get("ASHLAR_BENCH", "/tmp/ashlar-bench")) / "monotone-fremtpl2.json"
FEATURES = ("BonusMalus", "VehPower")
GRID = 12
PROBE_ROWS = 2000
TOLERANCE = 1e-3
CONSTRAINT = {"BonusMalus": 1}


def _arguments() -> dict:
    return dict(argument.split("=", 1) for argument in sys.argv[2:] if "=" in argument)


def _exceedance(model, frame, thresholds) -> np.ndarray:
    """``(rows, thresholds)`` of ``P(y > t)``, and the mean, per row."""
    dist = model.predict_dist(frame)
    cdf = np.asarray(dist.cdf(np.asarray(thresholds, dtype=np.float32)[:, None]))
    return 1.0 - cdf.T, np.asarray(dist.mean())


def violations(model, probe, train, thresholds) -> dict:
    out = {}
    for feature in FEATURES:
        values = np.unique(np.quantile(train[feature], np.linspace(0.02, 0.98, GRID)))
        curves, means = [], []
        for value in values:
            frame = probe.copy()
            frame[feature] = float(value)
            exceed, mean = _exceedance(model, frame, thresholds)
            curves.append(exceed)
            means.append(mean)
        curves, means = np.stack(curves), np.stack(means)  # (values, rows, thresholds), (values, rows)
        direction = 1.0 if np.mean(means[-1] - means[0]) >= 0 else -1.0
        against = -direction * np.diff(curves, axis=0)  # positive where a step moves against the direction
        worst = against.max(axis=2)  # (steps, rows)
        out[feature] = {
            "direction": direction,
            "steps violating": float(np.mean(worst > TOLERANCE)),
            "policies violating": float(np.mean((worst > TOLERANCE).any(axis=0))),
            "largest move against": float(np.mean(np.maximum(worst.max(axis=0), 0.0))),
            "mean against": float(np.mean(-direction * np.diff(means, axis=0) > 0)),
        }
    return out


def run(seeds="3", rows="200000"):
    frame = policies()
    results = []
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text())
    for seed in range(int(seeds)):
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(frame))
        test_size = len(frame) // 5
        test, train = frame.iloc[order[:test_size]], frame.iloc[order[test_size:]]
        probe = test.sample(PROBE_ROWS, random_state=seed)[NUMERIC + CATEGORICAL].reset_index(drop=True)
        sample = train.sample(min(int(rows), len(train)), random_state=seed)
        claimed_train, claimed_test = train[train["N"] > 0], test[test["N"] > 0]
        probe_claimed = claimed_test.sample(min(PROBE_ROWS, len(claimed_test)), random_state=seed)[NUMERIC + CATEGORICAL].reset_index(drop=True)
        log_cost = np.log(claimed_train["S"].to_numpy())
        cost_ladder = np.quantile(log_cost, np.linspace(0.05, 0.95, 9))
        jobs = [
            ("frequency", "negative_binomial", sample, test, "N", probe, np.array([0.0, 1.0, 2.0])),
            ("severity", "gaussian", claimed_train, claimed_test, "S", probe_claimed, cost_ladder),
            ("severity", "mixture", claimed_train, claimed_test, "S", probe_claimed, cost_ladder),
            ("severity", "quantile_spline", claimed_train, claimed_test, "S", probe_claimed, cost_ladder),
        ]
        for outcome, task, fit_frame, score_frame, target, probe_frame, ladder in jobs:
            for constrained in (False, True):
                if constrained and task == "mixture":
                    continue
                y = fit_frame[target].to_numpy()
                y_test = score_frame[target].to_numpy()
                if outcome == "severity":
                    y, y_test = np.log(y), np.log(y_test)
                start = time.process_time()
                model = ASRRegressor(task=task, monotone=CONSTRAINT if constrained else None, random_state=seed)
                model.fit(fit_frame[NUMERIC + CATEGORICAL].reset_index(drop=True), y)
                seconds = time.process_time() - start
                record = {
                    "seed": seed,
                    "outcome": outcome,
                    "task": task,
                    "monotone": constrained,
                    "rows": len(fit_frame),
                    "nll": float(-np.mean(model.log_prob(score_frame[NUMERIC + CATEGORICAL].reset_index(drop=True), y_test))),
                    "cpu": seconds,
                    "violations": violations(model, probe_frame, fit_frame, ladder),
                }
                results.append(record)
                print(json.dumps(record), flush=True)
                RESULTS.parent.mkdir(parents=True, exist_ok=True)
                RESULTS.write_text(json.dumps(results, indent=1))


def report():
    results = json.loads(RESULTS.read_text())
    keys = dict.fromkeys((r["outcome"], r["task"], r["monotone"]) for r in results)
    print("| Outcome | Head | Constraint | NLL ↓ | Feature | Steps violating | Policies violating | Largest move against | Mean against | CPU s |")
    print("|---|---|---|---:|---|---:|---:|---:|---:|---:|")
    for outcome, task, constrained in keys:
        rows = [r for r in results if (r["outcome"], r["task"], r["monotone"]) == (outcome, task, constrained)]
        nll = np.array([r["nll"] for r in rows])
        for feature in FEATURES:
            parts = [r["violations"][feature] for r in rows]

            def cell(name):
                values = np.array([p[name] for p in parts])
                return f"{values.mean():.4f} ± {values.std():.4f}"

            constraint = "`BonusMalus` rising" if constrained else "none"
            print(
                f"| {outcome} | `{task}` | {constraint} | {nll.mean():.4f} ± {nll.std():.4f} | {feature} | "
                f"{cell('steps violating')} | {cell('policies violating')} | {cell('largest move against')} | "
                f"{cell('mean against')} | {np.mean([r['cpu'] for r in rows]):.0f} |"
            )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "report": report}[command](**_arguments())
