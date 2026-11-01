"""A chain of a claim count and a claim cost on French motor insurance (freMTPL2), against separate heads.

freMTPL2 (from the R package CASdatasets) holds 677,991 motor policies with
their exposure in years, their number of claims, and, in a second table, the
cost of every claim. Each policy here has two outcomes: ``N``, its number of
claims (capped at 4, as is usual for this table), and ``S``, its average
cost per claim, 0 when it had none. Policies with claims but no recorded
cost are dropped (about 9,000). The features are the driver's age, the
bonus-malus level, the vehicle's power, age, brand, and fuel, the area, the
region, the log of the population density, and the exposure.

Three models of the pair, each with Ashlar's default trunk:

- ``chain``: ``task="chain"`` with a ``negative_binomial`` count and then a
  ``hurdle`` cost that reads the count, so the cost can be exactly zero
  where there was no claim and can depend on how many there were.
- ``independent heads``: a ``negative_binomial`` model of ``N`` and a
  ``hurdle`` model of ``S``, fitted apart; their joint is the product.
- ``frequency and severity``: the actuary's model by hand, a
  ``negative_binomial`` model of ``N`` and a ``gaussian`` model of ``log S``
  fitted on the policies with claims, with ``S = 0`` imposed where ``N = 0``
  and the cost taken as independent of the count given the features.

Scores on the test policies (20% of them, three random splits):

- Joint NLL: ``-log p(N, S)`` per policy, in nats; a cost's density is per
  euro. Split into the count's part and the cost's part given the count.
- The total claim cost ``T = N × S`` of each policy, read from 200 joint
  draws per policy on 50,000 test policies: the share predicted to claim,
  ``P(T > 0)``, against the share that did; the predicted share above 1,000
  and 5,000 euros against the observed; the sum of the predicted means against
  the sum of the costs; the CRPS; and how far the randomized probability
  integral transform of ``T`` is from uniform (its Kolmogorov distance, ×
  100), the PIT spread uniformly over the atom at zero.

The issue's stop rule (#38): stop if the chain gains less than about 0.01
nats per policy and its total is no better calibrated.

Run with::

    JAX_PLATFORMS=cpu python examples/benchmark_chain.py run      # downloads about 10 MB
    JAX_PLATFORMS=cpu python examples/benchmark_chain.py report

``seeds=0`` runs fewer splits and ``rows=100000`` a subsample.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

from ashlar import ASRRegressor, observed

DATA_DIR = Path("/tmp/ashlar-bench")
RESULTS = DATA_DIR / "chain-fremtpl2.json"
SOURCE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"
SEEDS = (0, 1, 2)
DRAWS = 200
DRAW_ROWS = 50_000
MODELS = ("chain", "independent heads", "frequency and severity")
THRESHOLDS = (1000.0, 5000.0)
NUMERIC = ["DrivAge", "BonusMalus", "VehPower", "VehAge", "LogDensity", "Exposure"]
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]


def _arguments() -> dict:
    return dict(argument.split("=", 1) for argument in sys.argv[2:] if "=" in argument)


def _download(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE + name, path)
    return path


def policies() -> pd.DataFrame:
    """One row per policy: the features, ``N`` (claims, at most 4), and ``S`` (average cost per claim)."""
    import pyreadr

    frequency = next(iter(pyreadr.read_r(str(_download("freMTPL2freq.rda"))).values()))
    severity = next(iter(pyreadr.read_r(str(_download("freMTPL2sev.rda"))).values()))
    frequency["IDpol"] = frequency["IDpol"].astype(str).astype(float).astype(int)
    severity["IDpol"] = severity["IDpol"].astype(str).astype(float).astype(int)
    costs = severity.groupby("IDpol")["ClaimAmount"].agg(["sum", "count"])
    frame = frequency.merge(costs, left_on="IDpol", right_index=True, how="left")
    frame["sum"] = frame["sum"].fillna(0.0)
    frame["count"] = frame["count"].fillna(0).astype(int)
    # A policy with claims and no recorded cost has no severity; one with costs counts its recorded claims.
    frame = frame[(frame["ClaimNb"] == 0) | (frame["count"] > 0)].copy()
    claims = np.where(frame["count"] > 0, frame["count"], frame["ClaimNb"]).astype(float)
    frame["N"] = np.minimum(claims, 4.0)
    frame["S"] = np.where(claims > 0, frame["sum"] / np.maximum(claims, 1.0), 0.0)
    frame["LogDensity"] = np.log(frame["Density"].astype(float))
    frame["Exposure"] = np.minimum(frame["Exposure"].astype(float), 1.0)
    for name in ("DrivAge", "BonusMalus", "VehPower", "VehAge"):
        frame[name] = frame[name].astype(float)
    for name in CATEGORICAL:
        frame[name] = frame[name].astype(str)
    return frame.reset_index(drop=True)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[NUMERIC + CATEGORICAL].reset_index(drop=True)


def _count_model(seed: int) -> ASRRegressor:
    return ASRRegressor(task="negative_binomial", random_state=seed)


def _fit_models(train: pd.DataFrame, seed: int) -> tuple[dict, dict]:
    x = _features(train)
    n, s = train["N"].to_numpy(), train["S"].to_numpy()
    models, seconds = {}, {}

    start = time.process_time()
    models["chain"] = ASRRegressor(
        task="chain",
        head_kwargs={"columns": ["negative_binomial", {"kind": "hurdle", "severity": "lognormal"}]},
        random_state=seed,
    ).fit(x, np.stack([n, s], axis=1))
    seconds["chain"] = time.process_time() - start

    start = time.process_time()
    count = _count_model(seed).fit(x, n)
    seconds["count"] = time.process_time() - start
    start = time.process_time()
    cost = ASRRegressor(task="hurdle", random_state=seed).fit(x, s)
    seconds["independent heads"] = seconds["count"] + time.process_time() - start
    models["independent heads"] = (count, cost)

    claimed = n > 0
    start = time.process_time()
    severity = ASRRegressor(task="gaussian", random_state=seed).fit(x[claimed], np.log(s[claimed]))
    seconds["frequency and severity"] = seconds["count"] + time.process_time() - start
    models["frequency and severity"] = (count, severity)
    return models, seconds


def _log_likelihood(name: str, model, x, n, s) -> tuple[np.ndarray, np.ndarray]:
    """``(log p(N), log p(S | N))`` per policy."""
    if name == "chain":
        joint = model.log_prob(x, np.stack([n, s], axis=1))
        # The count's own likelihood: the cost left open, an interval over every value.
        unknown = np.full(len(n), np.inf)
        log_n = model.log_prob(x, observed.stack([observed.exact(n), observed.interval(-unknown, unknown)]))
        return log_n, joint - log_n
    count, cost = model
    log_n = count.log_prob(x, n)
    if name == "independent heads":
        return log_n, cost.log_prob(x, s)
    claimed = n > 0
    log_s = np.zeros_like(log_n)
    if claimed.any():
        log_s[claimed] = cost.log_prob(x[claimed], np.log(s[claimed])) - np.log(s[claimed])
    return log_n, log_s


def _draws(name: str, model, x, seed: int) -> np.ndarray:
    """``(DRAWS, rows)`` draws of the total cost ``N × S``."""
    if name == "chain":
        pair = model.sample(x, n=DRAWS, seed=seed)
        return pair[..., 0] * pair[..., 1]
    count, cost = model
    n = count.sample(x, n=DRAWS, seed=seed)
    if name == "independent heads":
        return n * cost.sample(x, n=DRAWS, seed=seed + 1)
    return np.where(n > 0, n * np.exp(cost.sample(x, n=DRAWS, seed=seed + 1)), 0.0)


def _crps(draws: np.ndarray, y: np.ndarray) -> float:
    draws = np.sort(draws, axis=0)
    m = draws.shape[0]
    first = np.mean(np.abs(draws - y[None, :]), axis=0)
    weights = (2.0 * np.arange(1, m + 1) - m - 1)[:, None]
    second = np.sum(weights * draws, axis=0) / (m * m)
    return float(np.mean(first - second))


def _pit_distance(draws: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Kolmogorov distance of the randomized PIT of ``y`` among the draws from uniform."""
    rng = np.random.default_rng(seed)
    below = np.mean(draws < y[None, :], axis=0)
    at = np.mean(draws <= y[None, :], axis=0)
    pit = np.sort(below + rng.uniform(size=y.shape) * (at - below))
    grid = np.arange(1, pit.size + 1) / pit.size
    return float(np.max(np.maximum(np.abs(grid - pit), np.abs(grid - 1.0 / pit.size - pit))))


def _scores(name: str, model, test: pd.DataFrame, seed: int) -> dict:
    x = _features(test)
    n, s = test["N"].to_numpy(), test["S"].to_numpy()
    log_n, log_s = _log_likelihood(name, model, x, n, s)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(test), size=min(DRAW_ROWS, len(test)), replace=False)
    draws = _draws(name, model, x.iloc[rows].reset_index(drop=True), seed)
    total = (n * s)[rows]
    scores = {
        "nll": float(-np.mean(log_n + log_s)),
        "nll_count": float(-np.mean(log_n)),
        "nll_cost": float(-np.mean(log_s)),
        "claim_predicted": float(np.mean(draws > 0)),
        "claim_observed": float(np.mean(total > 0)),
        "mean_ratio": float(np.mean(draws) / np.mean(total)),
        "crps": _crps(draws, total),
        "pit_distance": 100.0 * _pit_distance(draws, total, seed),
    }
    for threshold in THRESHOLDS:
        scores[f"above_{int(threshold)}_predicted"] = float(np.mean(draws > threshold))
        scores[f"above_{int(threshold)}_observed"] = float(np.mean(total > threshold))
    return scores


def run() -> None:
    arguments = _arguments()
    seeds = SEEDS if "seeds" not in arguments else tuple(int(seed) for seed in arguments["seeds"].split(","))
    frame = policies()
    if "rows" in arguments:
        frame = frame.sample(n=int(arguments["rows"]), random_state=0).reset_index(drop=True)
    print(f"{len(frame):,} policies, {np.mean(frame['N'] > 0):.2%} with a claim", flush=True)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(frame))
        cut = int(0.8 * len(frame))
        train, test = frame.iloc[order[:cut]], frame.iloc[order[cut:]]
        models, seconds = _fit_models(train, seed)
        split = {}
        for name in MODELS:
            split[name] = _scores(name, models[name], test, seed)
            split[name]["cpu"] = seconds[name]
            print(seed, name, json.dumps(split[name]), flush=True)
        results[str(seed)] = split
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))


def _cell(rows, key, form, scale=1.0) -> str:
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float) * scale
    if values.size == 0:
        return ""
    if values.size == 1:
        return form.format(values[0])
    return f"{form.format(values.mean())} ± {form.format(values.std())}"


def report() -> None:
    results = json.loads(RESULTS.read_text())
    splits = list(results.values())
    print(f"\n### freMTPL2, {len(splits)} splits, mean ± standard deviation\n")
    print(
        "| Model | Joint NLL ↓ | Count NLL | Cost NLL given the count | P(T > 0), predicted | "
        "P(T > 1,000) | P(T > 5,000) | Mean T / observed | CRPS of T ↓ | PIT distance × 100 ↓ | CPU seconds |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in MODELS:
        rows = [split[name] for split in splits]
        print(
            f"| {name} | {_cell(rows, 'nll', '{:.4f}')} | {_cell(rows, 'nll_count', '{:.4f}')} | "
            f"{_cell(rows, 'nll_cost', '{:.4f}')} | {_cell(rows, 'claim_predicted', '{:.2f}', 100.0)}% | "
            f"{_cell(rows, 'above_1000_predicted', '{:.2f}', 100.0)}% | {_cell(rows, 'above_5000_predicted', '{:.2f}', 100.0)}% | "
            f"{_cell(rows, 'mean_ratio', '{:.3f}')} | {_cell(rows, 'crps', '{:.1f}')} | "
            f"{_cell(rows, 'pit_distance', '{:.2f}')} | {np.mean([row['cpu'] for row in rows]):.0f} |"
        )
    observed = splits[0]["chain"]
    print(
        f"\nObserved: P(T > 0) {100 * observed['claim_observed']:.2f}%, P(T > 1,000) "
        f"{100 * observed['above_1000_observed']:.2f}%, P(T > 5,000) {100 * observed['above_5000_observed']:.2f}% "
        "(split 0)."
    )


STAGES = {"run": run, "report": report}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
