"""Cure fractions, competing risks, and regimes under joint censoring, on simulated tables with known truth.

Two recipes from ``docs/observed.md``, and a test of the joint mixture:

- ``cure``: a log time that never comes for a share of the rows that changes
  with the features, censored at a random follow-up. ``gaussian`` with
  ``head_kwargs={"cure": True}`` against the plain ``gaussian`` and
  ``mixture`` heads on the same censored label. Scores: the censored
  log-likelihood of held-out rows, the predicted share that never has the
  event against the true share, and the error of ``P(T > t)`` at three
  follow-up times, averaged over rows.
- ``risks``: a cause of two and a log time whose distribution depends on the
  cause, censored at a random follow-up, so a censored row's cause is
  unknown. ``task="chain"`` with a ``categorical`` cause and then a
  ``gaussian`` time, trained on the cause as ``interval(-inf, inf)`` where
  it is censored, against cause-specific fits: one censored ``mixture`` per
  cause, each treating the other cause as censoring, the usual workaround.
  Run with every event's cause known, and with 30% of the causes
  unrecorded, which the chain reads as an exact time of unknown cause and
  the cause-specific fits can only treat as censored.
  Scores: the error of the cumulative incidence ``P(cause = c, T <= t)`` at
  three times, averaged over rows and causes, against the truth.

- ``regimes``: two columns whose correlation is +0.9 in one regime and
  -0.9 in the other, the regime's share changing with the features, each
  column censored at its own random limit. ``multivariate_mixture`` against
  ``multivariate_gaussian``, both on the censored joint label, against
  each one's oracle on the true values. Scores: the joint NLL of the true
  values of held-out rows, and the error of ``P(both columns > 1)``, from
  2,000 draws, against the truth.

Run with::

    JAX_PLATFORMS=cpu python examples/benchmark_observed_more.py run seeds=3
    JAX_PLATFORMS=cpu python examples/benchmark_observed_more.py report
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from scipy.special import expit
from scipy.stats import norm

from ashlar import ASRRegressor, observed

OUT = Path(os.environ.get("ASHLAR_BENCH", "/tmp/ashlar-bench")) / "observed-more.json"
INF = np.inf
TIMES = np.log([1.0, 3.0, 8.0])
ROWS = 6000


def _features(rng, n):
    return rng.normal(size=(n, 4)).astype(np.float32)


def cure_table(seed, n=ROWS):
    rng = np.random.default_rng(seed)
    x = _features(rng, n)
    never = expit(-0.5 + 1.0 * x[:, 0])
    loc, scale = 0.8 + 0.5 * x[:, 1], np.exp(-0.3 + 0.2 * x[:, 2])
    time_ = np.where(rng.random(n) < never, INF, loc + scale * rng.normal(size=n))
    follow = np.log(rng.uniform(0.5, 12.0, size=n))
    label = observed.censored(np.minimum(time_, follow), time_ > follow)
    survival = never[:, None] + (1 - never[:, None]) * norm.sf(TIMES[None, :], loc[:, None], scale[:, None])
    return x, label, never, survival


def risks_table(seed, n=ROWS, masked=0.0):
    rng = np.random.default_rng(seed)
    x = _features(rng, n)
    share = expit(0.3 + 1.0 * x[:, 0])  # P(cause 1)
    cause = (rng.random(n) < share).astype(float)
    loc = np.where(cause == 1, 0.4 + 0.5 * x[:, 1], 1.4 - 0.4 * x[:, 2])
    scale = np.where(cause == 1, 0.5, 0.8)
    time_ = loc + scale * rng.normal(size=n)
    follow = np.log(rng.uniform(0.5, 10.0, size=n))
    seen = time_ <= follow
    # A share of the events whose cause went unrecorded.
    known = seen & (rng.random(n) >= masked)
    label = observed.stack(
        [
            observed.interval(np.where(known, cause, -INF), np.where(known, cause, INF)),
            observed.censored(np.minimum(time_, follow), ~seen),
        ]
    )
    # The truth: P(cause = c, T <= t) for c = 0, 1.
    one = share[:, None] * norm.cdf(TIMES[None, :], (0.4 + 0.5 * x[:, 1])[:, None], 0.5)
    zero = (1 - share[:, None]) * norm.cdf(TIMES[None, :], (1.4 - 0.4 * x[:, 2])[:, None], 0.8)
    return x, label, cause, np.minimum(time_, follow), known, np.stack([zero, one], axis=1)


def run_cure(seed):
    x, label, never, survival = cure_table(seed)
    train, test = slice(0, 4500), slice(4500, None)
    rows = []
    for name, task, kwargs in [("gaussian with cure", "gaussian", {"cure": True}), ("gaussian", "gaussian", {}), ("mixture", "mixture", {})]:
        start = time.process_time()
        model = ASRRegressor(task=task, head_kwargs=kwargs or None, random_state=seed).fit(x[train], label[train])
        fitted = time.process_time() - start
        predicted = np.stack([model.predict_exceedance(x[test], t) for t in TIMES], axis=1)
        far = model.predict_exceedance(x[test], 1e6)
        rows.append(
            {
                "model": name,
                "seed": seed,
                "nll": float(-model.log_prob(x[test], label[test]).mean()),
                "never": float(far.mean()),
                "true never": float(never[test].mean()),
                "never error": float(np.abs(far - never[test]).mean()),
                "survival error": float(np.abs(predicted - survival[test]).mean()),
                "cpu": fitted,
            }
        )
    return rows


def run_risks(seed, masked=0.0):
    x, label, cause, time_, seen, incidence = risks_table(seed, masked=masked)
    train, test = slice(0, 4500), slice(4500, None)
    rows = []
    start = time.process_time()
    chain = ASRRegressor(task="chain", head_kwargs={"columns": ["categorical", "gaussian"]}, random_state=seed).fit(x[train], label[train])
    fitted = time.process_time() - start
    # P(cause = c, T <= t) exactly, as the likelihood of a label that says just that.
    rows_test = len(x[test])

    def incidence_of(c, t):
        label = observed.stack([observed.exact(np.full(rows_test, float(c))), observed.interval(np.full(rows_test, -INF), np.full(rows_test, t))])
        return np.exp(chain.log_prob(x[test], label))

    predicted = np.stack([np.stack([incidence_of(c, t) for t in TIMES], axis=1) for c in (0, 1)], axis=1)
    rows.append({"model": "chain", "masked": masked, "seed": seed, "incidence error": float(np.abs(predicted - incidence[test]).mean()), "cpu": fitted})
    # Cause-specific fits: the other cause is censoring, and so is an event of unknown cause, the usual choice.
    start = time.process_time()
    parts = []
    for c in (0, 1):
        event = seen[train] & (cause[train] == c)
        model = ASRRegressor(task="mixture", random_state=seed).fit(x[train], observed.survival(time_[train], event))
        parts.append(model)
    fitted = time.process_time() - start
    # Under independent latent times, P(cause c by t) = integral of f_c(s) S_other(s) ds, on a grid.
    grid = np.linspace(-4.0, TIMES.max(), 400)
    cdfs = [np.stack([1 - m.predict_exceedance(x[test], g) for g in grid], axis=1) for m in parts]
    predicted = np.zeros_like(incidence[test])
    for c in (0, 1):
        density = np.diff(cdfs[c], axis=1)
        other = 1 - cdfs[1 - c][:, 1:]
        cumulative = np.cumsum(density * other, axis=1)
        for k, t in enumerate(TIMES):
            predicted[:, c, k] = cumulative[:, np.searchsorted(grid[1:], t) - 1]
    rows.append({"model": "cause-specific mixtures", "masked": masked, "seed": seed, "incidence error": float(np.abs(predicted - incidence[test]).mean()), "cpu": fitted})
    return rows


def regimes_table(seed, n=ROWS):
    rng = np.random.default_rng(seed)
    x = _features(rng, n)
    regime = rng.random(n) < expit(1.5 * x[:, 0])
    rho = np.where(regime, -0.9, 0.9)
    loc = np.stack([0.5 * x[:, 1], 0.5 * x[:, 2]], axis=1)
    a, b = rng.normal(size=n), rng.normal(size=n)
    value = loc + np.stack([a, rho * a + np.sqrt(1 - rho**2) * b], axis=1)
    limit = rng.normal(1.0, 1.0, size=(n, 2))
    label = observed.stack([observed.censored(np.minimum(value[:, j], limit[:, j]), value[:, j] > limit[:, j]) for j in range(2)])
    # P(both > 1) given the row, from the two regimes' bivariate normals, by draws.
    draws = 20_000
    a, b = rng.normal(size=(draws, 1)), rng.normal(size=(draws, 1))
    share = expit(1.5 * x[:, 0])
    truth = np.zeros(n)
    for r, weight in ((-0.9, share), (0.9, 1 - share)):
        first = loc[None, :, 0] + a
        second = loc[None, :, 1] + r * a + np.sqrt(1 - r**2) * b
        truth = truth + weight * np.mean((first > 1) & (second > 1), axis=0)
    return x, label, value, truth


def run_regimes(seed):
    x, label, value, truth = regimes_table(seed)
    train, test = slice(0, 4500), slice(4500, None)
    rows = []
    for task in ("multivariate_mixture", "multivariate_gaussian"):
        for name, target in (("censored", label), ("oracle", value)):
            start = time.process_time()
            model = ASRRegressor(task=task, random_state=seed).fit(x[train], target[train])
            fitted = time.process_time() - start
            draws = model.sample(x[test], n=2000, seed=seed)
            both = np.mean((draws[..., 0] > 1) & (draws[..., 1] > 1), axis=0)
            rows.append(
                {
                    "model": f"{task}, {name}",
                    "seed": seed,
                    "nll": float(-model.log_prob(x[test], value[test]).mean()),
                    "both above error": float(np.abs(both - truth[test]).mean()),
                    "cpu": fitted,
                }
            )
    return rows


def run(seeds=3):
    results = {"cure": [], "risks": [], "regimes": []}
    for seed in range(int(seeds)):
        results["cure"] += run_cure(seed)
        results["risks"] += run_risks(seed) + run_risks(seed, masked=0.3)
        results["regimes"] += run_regimes(seed)
        print(json.dumps(results, indent=1), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))


def report():
    results = json.loads(OUT.read_text())
    for part, metrics in (("cure", ["nll", "never", "true never", "never error", "survival error", "cpu"]), ("risks", ["incidence error", "cpu"]), ("regimes", ["nll", "both above error", "cpu"])):
        print(f"\n{part}")
        print("| Model | " + " | ".join(metrics) + " |")
        print("|---|" + "---:|" * len(metrics))
        for name in dict.fromkeys((row["model"], row.get("masked")) for row in results[part]):
            rows = [row for row in results[part] if (row["model"], row.get("masked")) == name]
            cells = [f"{np.mean([r[m] for r in rows]):.4f} ± {np.std([r[m] for r in rows]):.4f}" for m in metrics]
            title = name[0] if name[1] is None else f"{name[0]}, {name[1]:.0%} of causes unknown"
            print(f"| {title} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    options = dict(arg.split("=", 1) for arg in sys.argv[2:])
    {"run": run, "report": report}[command](**options)
