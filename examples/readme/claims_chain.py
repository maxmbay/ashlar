"""Figure ``claims-chain``: a claim count, then a cost given the count, on French motor insurance (freMTPL2).

Two stages, run from the repository root::

    JAX_PLATFORMS=cpu python examples/readme/claims_chain.py compute   # about an hour of CPU; Max's Mac
    python examples/readme/claims_chain.py draw                        # seconds; needs the style kit's fonts

``compute`` fits the models of ``examples/benchmark_chain.py`` on its split
0 (seed 0, 542,392 training policies, 135,599 test policies), each exactly as
the benchmark does:

- ``chain``: ``task="chain"``, a ``negative_binomial`` claim count and then a
  ``hurdle`` cost per claim that reads the count;
- ``independent heads``: the same two heads fitted apart;
- ``frequency and severity``: the actuary's model by hand, a count model and
  a ``gaussian`` log cost fitted on the policies with claims;
- ``LightGBM Poisson × gamma`` and ``LightGBM Tweedie``: the benchmark's
  ``gbm`` stage, what a gradient-boosting user would build.

It writes, under ``examples/readme/artifacts/claims-chain/``:

- ``curves.json``: the portfolio's exceedance curve, the share of test
  policies whose total cost ``T = N × S`` exceeds ``t``, observed and as each
  model predicts it, for ``t`` from 1 euro to 1 million euros. Every model's
  curve is exact, not drawn: the mean over the test policies of
  ``Σ_k P(N = k) P(S > t / k | N = k)``. It also holds each model's mean
  total, and the split's benchmark scores, which reproduce the benchmark.
- ``policies.json``: for eight test policies spread over the chain's
  ``P(N ≥ 1)``, the chain's count mass over 0 to 4 claims and its density of
  the cost per claim given each count on a log-euro grid, with the LightGBM
  gamma's and the hand-built log-normal's densities beside them, and the
  observed cost per claim by count over the test policies.

``rows=60000`` runs on a subsample, ``out=DIR`` writes elsewhere (for a test).
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

from benchmark_chain import (  # noqa: E402
    MODELS,
    GBMActuary,
    GBMTweedie,
    _features,
    _fit_models,
    _gbm_scores,
    _scores,
    gamma_log_pdf,
    policies,
    split,
)

NAME = "claims-chain"
ARTIFACTS = ROOT / "examples" / "readme" / "artifacts" / NAME
SEED = 0
# The exceedance curve's thresholds, 1 euro to 1 million, 20 per decade; and the cost grid for one policy's densities.
THRESHOLDS = 10.0 ** np.round(np.arange(0.0, 6.0001, 0.05), 4)
COST_GRID = 10.0 ** np.round(np.arange(1.0, 6.0001, 1.0 / 32.0), 5)
COUNTS = 4
CANDIDATE_LEVELS = (0.5, 0.75, 0.9, 0.97, 0.99, 0.997, 0.999, 1.0)
CHUNK = 8192
LN10 = float(np.log(10.0))


def _arguments() -> dict:
    return dict(argument.split("=", 1) for argument in sys.argv[2:] if "=" in argument)


def _round(values, digits: int = 5):
    """Plain lists at ``digits`` significant digits, so the artifacts stay small."""
    array = np.asarray(values, dtype=float)
    return [float(f"{value:.{digits}g}") for value in array.ravel()] if array.ndim <= 1 else [_round(row, digits) for row in array]


def _unwrap(dist):
    from ashlar.distributions import AffineTransformed

    while isinstance(dist, AffineTransformed):
        dist = dist.base
    return dist


def chain_views(model, frame):
    """``(mass (rows, members, counts + 1), [hurdle of S given N = k for k = 1..counts])``, every member kept."""
    import jax.numpy as jnp

    chain = _unwrap(model.predict_dist(_features(frame))).components  # batch (rows, members)
    values = jnp.zeros(chain.batch_shape + (2,), dtype=jnp.float32)
    count = chain.conditional(0, values)
    mass = np.stack([np.exp(np.asarray(count.log_prob(jnp.float32(k)))) for k in range(COUNTS + 1)], axis=-1)
    hurdles = [chain.conditional(1, values.at[..., 0].set(float(k))) for k in range(1, COUNTS + 1)]
    return mass, hurdles


def chain_exceedance(model, frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rows, thresholds)`` of the chain's exact ``P(T > t)``, each row's ``P(N ≥ 1)``, and its count mass."""
    import jax.numpy as jnp

    out, claim, masses = [], [], []
    for start in range(0, len(frame), CHUNK):
        mass, hurdles = chain_views(model, frame.iloc[start : start + CHUNK])
        total = 0.0
        for k, hurdle in enumerate(hurdles, start=1):
            above = np.exp(np.asarray(hurdle.log_survival(jnp.asarray(THRESHOLDS / k, dtype=jnp.float32)[:, None, None])))
            total = total + mass[None, ..., k] * above  # (thresholds, rows, members)
        out.append(np.mean(total, axis=-1).T)
        claim.append(np.mean(1.0 - mass[..., 0], axis=-1))
        masses.append(np.mean(mass, axis=1))
    return np.concatenate(out), np.concatenate(claim), np.concatenate(masses)


def _mixture_survival(dist, value) -> np.ndarray:
    import jax.numpy as jnp

    return np.exp(np.asarray(dist.log_survival(jnp.asarray(value, dtype=jnp.float32))))


def apart_exceedance(models, name: str, frame) -> np.ndarray:
    """``(rows, thresholds)`` of ``P(T > t)`` for a count model and a cost model fitted apart."""
    import jax.numpy as jnp

    count, cost = models[name]
    out = []
    for start in range(0, len(frame), CHUNK):
        x = _features(frame.iloc[start : start + CHUNK])
        count_dist, cost_dist = count.predict_dist(x), cost.predict_dist(x)
        total = 0.0
        for k in range(1, COUNTS + 1):
            mass = np.exp(np.asarray(count_dist.log_prob(jnp.float32(k))))
            if name == "independent heads":
                above = _mixture_survival(cost_dist, (THRESHOLDS / k)[:, None])
            else:  # the log cost: P(log S > log(t / k))
                above = 1.0 - np.asarray(cost_dist.cdf(jnp.asarray(np.log(THRESHOLDS / k), dtype=jnp.float32)[:, None]))
            total = total + mass[None, :] * above
        out.append(np.asarray(total).T)
    return np.concatenate(out)


def gbm_exceedance(model, frame) -> np.ndarray:
    return np.concatenate([model.exceedance(frame.iloc[start : start + CHUNK], THRESHOLDS) for start in range(0, len(frame), CHUNK)])


def policy_view(models, gbm: GBMActuary, tweedie: GBMTweedie, row) -> dict:
    """One policy's chain: its count mass, and its cost density per log10 euro given each count, beside the others'."""
    import jax.numpy as jnp

    mass, hurdles = chain_views(models["chain"], row)
    weights = mass[0]  # (members, counts + 1)
    cost_given, zero_given, mean_given = [], [], []
    for k, hurdle in enumerate(hurdles, start=1):
        density = np.exp(np.asarray(hurdle.log_prob(jnp.asarray(COST_GRID, dtype=jnp.float32)[:, None, None])))[:, 0, :]
        # The member mixture given N = k weighs each member by its own P(N = k).
        w = weights[:, k] / weights[:, k].sum()
        cost_given.append(np.sum(density * w[None, :], axis=1) * COST_GRID * LN10)
        zero_given.append(float(np.sum(np.asarray(hurdle.zero)[0] * w)))
        mean_given.append(float(np.sum(np.asarray(hurdle.mean())[0] * w)))
    _count, severity = models["frequency and severity"]
    log_cost = severity.predict_dist(_features(row))
    hand = np.exp(np.asarray(log_cost.log_prob(jnp.asarray(np.log(COST_GRID), dtype=jnp.float32)[:, None])))[:, 0] * LN10
    count_model = models["frequency and severity"][0]
    hand_mass = np.exp(np.asarray(count_model.predict_dist(_features(row)).log_prob(jnp.arange(COUNTS + 1, dtype=jnp.float32)[:, None])))[:, 0]
    from scipy.stats import poisson

    mean = float(gbm.cost_mean(row)[0])
    rate = float(gbm.rate(row)[0])
    gbm_cost = [
        np.exp(gamma_log_pdf(COST_GRID, mean, float(gbm.cost_shape(np.array([k]))[0]))) * COST_GRID * LN10
        for k in range(1, COUNTS + 1)
    ]
    exceedance = {
        "chain": chain_exceedance(models["chain"], row)[0][0],
        "independent heads": apart_exceedance(models, "independent heads", row)[0],
        "frequency and severity": apart_exceedance(models, "frequency and severity", row)[0],
        "LightGBM Poisson × gamma": gbm.exceedance(row, THRESHOLDS)[0],
        "LightGBM Tweedie": tweedie.exceedance(row, THRESHOLDS)[0],
    }
    features = {name: (value.item() if hasattr(value, "item") else value) for name, value in row.iloc[0].items()}
    return {
        "features": {key: features[key] for key in ("DrivAge", "BonusMalus", "VehPower", "VehAge", "VehBrand", "VehGas",
                                                     "Area", "Region", "Density", "Exposure", "N", "S")},
        "chain": {
            "count_mass": _round(np.mean(weights, axis=0)),
            "cost_density_given_count": _round(np.stack(cost_given)),
            "cost_zero_given_count": _round(zero_given),
            "cost_mean_given_count": _round(mean_given),
        },
        "frequency and severity": {"count_mass": _round(hand_mass), "cost_density": _round(hand)},
        "exceedance": {name: _round(curve, 4) for name, curve in exceedance.items()},
        "LightGBM Tweedie mean": float(tweedie.mean(row)[0]),
        "LightGBM Poisson × gamma": {
            "count_mass": _round(poisson.pmf(np.arange(COUNTS + 1), rate)),
            "cost_density_given_count": _round(np.stack(gbm_cost)),
            "cost_mean": mean,
        },
    }


def compute() -> None:
    arguments = _arguments()
    out = Path(arguments.get("out", ARTIFACTS))
    out.mkdir(parents=True, exist_ok=True)
    frame = policies()
    if "rows" in arguments:
        frame = frame.sample(n=int(arguments["rows"]), random_state=0).reset_index(drop=True)
    train, test = split(frame, SEED)
    test = test.reset_index(drop=True)
    print(f"{len(train):,} training and {len(test):,} test policies (split {SEED})", flush=True)

    start = time.time()
    models, seconds = _fit_models(train, SEED)
    print(f"Ashlar fits: {json.dumps({k: round(v) for k, v in seconds.items()})} CPU seconds, {time.time() - start:.0f} s", flush=True)
    keep = Path(arguments.get("keep", "/tmp/ashlar-bench/readme-claims-chain"))
    try:
        models["chain"].save(keep / "chain", save_optimizer=False)
        print(f"saved the chain to {keep / 'chain'}", flush=True)
    except Exception as error:  # a failed save must not cost the run
        print(f"could not save the chain: {error!r}", flush=True)
    gbm, tweedie = GBMActuary(train, SEED), GBMTweedie(train, SEED)
    print(f"LightGBM choices: {gbm.choices} {tweedie.choices}", flush=True)

    scores = {name: _scores(name, models[name], test, SEED) for name in MODELS}
    scores["LightGBM Poisson × gamma"] = _gbm_scores("LightGBM Poisson × gamma", gbm, test, SEED)
    scores["LightGBM Tweedie"] = _gbm_scores("LightGBM Tweedie", tweedie, test, SEED)
    for name, value in scores.items():
        print(name, json.dumps(value), flush=True)

    total = (test["N"] * test["S"]).to_numpy()
    curves = {"observed": np.mean(total[:, None] > THRESHOLDS[None, :], axis=0)}
    chain, claim, chain_mass = chain_exceedance(models["chain"], test)
    curves["chain"] = chain.mean(axis=0)
    from scipy.stats import poisson

    count_model, x_test = models["frequency and severity"][0], _features(test)
    separate_mass = np.concatenate([
        np.exp(np.asarray(count_model.predict_dist(x_test.iloc[start : start + CHUNK]).log_prob(
            np.arange(COUNTS + 1, dtype=np.float32)[:, None]))).T
        for start in range(0, len(test), CHUNK)
    ])
    count_mass = {
        "observed": np.bincount(test["N"].astype(int), minlength=COUNTS + 1)[: COUNTS + 1] / len(test),
        "chain": chain_mass.mean(axis=0),
        "negative_binomial fitted apart": separate_mass.mean(axis=0),
        "LightGBM Poisson × gamma": poisson.pmf(np.arange(COUNTS + 1)[:, None], gbm.rate(test)[None, :]).mean(axis=1),
    }
    for name in ("independent heads", "frequency and severity"):
        curves[name] = apart_exceedance(models, name, test).mean(axis=0)
    curves["LightGBM Poisson × gamma"] = gbm_exceedance(gbm, test).mean(axis=0)
    curves["LightGBM Tweedie"] = gbm_exceedance(tweedie, test).mean(axis=0)
    zero = {
        "observed": float(np.mean(total > 0)),
        "chain": float(np.mean(claim)),
        "LightGBM Poisson × gamma": float(np.mean(1.0 - np.exp(-gbm.rate(test)))),
    }
    means = {
        "observed": float(np.mean(total)),
        "LightGBM Poisson × gamma": float(np.mean(gbm.rate(test) * gbm.cost_mean(test))),
        "LightGBM Tweedie": float(np.mean(tweedie.mean(test))),
    }
    (out / "curves.json").write_text(json.dumps({
        "split": SEED,
        "test_policies": len(test),
        "thresholds": _round(THRESHOLDS),
        "exceedance": {name: _round(curve) for name, curve in curves.items()},
        "claim_share": zero,
        "count_mass": {name: _round(mass) for name, mass in count_mass.items()},
        "mean_total": means,
        "scores": scores,
        "lightgbm_choices": {"LightGBM Poisson × gamma": gbm.choices, "LightGBM Tweedie": tweedie.choices},
        "cpu_seconds": seconds,
    }, indent=1))

    order = np.argsort(claim, kind="stable")
    picks = sorted({int(order[min(int(level * len(order)), len(order) - 1)]) for level in CANDIDATE_LEVELS})
    views = []
    for index in picks:
        row = test.iloc[[index]].reset_index(drop=True)
        view = policy_view(models, gbm, tweedie, row)
        view["test_index"] = index
        view["claim_probability"] = float(claim[index])
        view["claim_probability_rank"] = float(np.searchsorted(np.sort(claim), claim[index]) / len(claim))
        views.append(view)
    claimed = test[test["N"] > 0]
    edges = np.arange(1.0, 6.0001, 0.125)
    observed = {
        str(k): np.histogram(np.log10(claimed.loc[claimed["N"] == k, "S"]), bins=edges)[0].tolist()
        for k in range(1, COUNTS + 1)
    }
    (out / "policies.json").write_text(json.dumps({
        "cost_grid": _round(COST_GRID),
        "policies": views,
        "observed_log10_cost_by_count": {"edges": _round(edges), "counts": observed},
    }))
    for path in sorted(out.iterdir()):
        print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {path.stat().st_size / 1024:.0f} KB", flush=True)


def draw() -> None:
    raise SystemExit("the draw stage needs the style kit, examples/readme/style.py, which is not on this branch yet")


STAGES = {"compute": compute, "draw": draw}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "draw"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
