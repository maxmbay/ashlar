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


POLICY_RANK = 0.999  # the policy drawn on the left: the one nearest this rank of the chain's P(N >= 1)
ROW_NAMES = ("no claim", "1 claim", "2 claims", "3 claims", "4 claims")
# Each curve on the right: (key in curves.json, legend label, colour, line style, width), drawn in this order.
CURVES = (
    ("independent heads", "two heads fitted apart", "SLATE", (0, (5, 3)), 2.0),
    ("frequency and severity", "frequency × severity by hand", "SLATE", (0, (1, 2.2)), 2.2),
    ("LightGBM Poisson × gamma", "LightGBM Poisson × gamma", "CLAY", "-", 2.2),
    ("chain", 'task="chain"', "TEAL", "-", 3.0),
)


def _euros(value: float) -> str:
    if value >= 1e6:
        return f"€{value / 1e6:g}M"
    if value >= 1e3:
        return f"€{value / 1e3:g}k"
    return f"€{value:g}"


def draw() -> None:
    import style as k
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator
    from matplotlib.transforms import blended_transform_factory

    arguments = _arguments()
    folder = Path(arguments.get("artifacts", ARTIFACTS))
    curves = json.loads((folder / "curves.json").read_text())
    views = json.loads((folder / "policies.json").read_text())
    ranks = np.array([view["claim_probability_rank"] for view in views["policies"]])
    view = views["policies"][int(arguments.get("policy", np.argmin(np.abs(ranks - POLICY_RANK))))]
    scores = curves["scores"]
    colour = {"TEAL": k.TEAL, "CLAY": k.CLAY, "SLATE": k.SLATE}

    fig = k.card(
        "The cost reads the count, and the total ties LightGBM's.",
        f"freMTPL2, {curves['test_policies']:,} held-out French motor policies, split 0. One chained model: a negative "
        "binomial claim count, then a hurdle cost per claim that reads the count. Right: the share of policies whose "
        "total for the year, T = count × cost per claim, is above t.",
        height=8.4,
        footer=2,
    )
    box = fig.box
    left, bottom, width, height = box.cell()
    head, tail = 1.0, 0.78
    y0, h = (bottom + tail) / box.height, (height - head - tail) / box.height

    def axes(x, w, **kwargs):
        return fig.add_axes([(left + x) / box.width, y0, w / box.width, h], **kwargs)

    # ---- left: one policy, a count and then a cost given the count
    rows = len(ROW_NAMES)
    mass = np.asarray(view["chain"]["count_mass"])
    density = np.asarray(view["chain"]["cost_density_given_count"])  # (counts, grid), per log10 euro
    grid = np.log10(np.asarray(views["cost_grid"]))
    bars = axes(1.0, 1.2)
    costs = axes(2.85, 2.35, sharey=bars)
    k.style(bars, frame="")
    k.style(costs, frame="b")
    y = rows - 1 - np.arange(rows)  # no claim at the top
    bars.barh(y, mass, height=0.5, color=k.TEAL, lw=0)
    for row in range(rows):
        share = mass[row]
        text = k.pct(share, 1) if share >= 0.001 else f"{100 * share:.1g}%"
        k.value(bars, max(share, 0.0) + 0.05, y[row], text, k.TEAL, size=12.5)
        bars.text(-0.08, y[row], ROW_NAMES[row], transform=bars.get_yaxis_transform(), fontproperties=k.F.label,
                  color=k.INK, ha="right", va="center")
    bars.set_xlim(0, 1.0)
    bars.set_xticks([])
    bars.set_yticks([])
    bars.set_ylim(-0.55, rows - 0.35)

    # The cost per claim given each count, and its mean in a column of its own beside the ridges.
    peak = density.max()
    column_x = left + 5.95  # inches from the card's left edge
    column = blended_transform_factory(fig.dpi_scale_trans, costs.transData)
    fig.text(column_x, y[1] + 0.42, "mean", fontproperties=k.F.small, color=k.MUTED, ha="right", va="bottom",
             transform=column)
    for count in range(1, rows):
        base = y[count] - 0.32
        curve = density[count - 1] / peak * 0.8
        costs.fill_between(grid, base, base + curve, color=k.TEAL_PALE, lw=0, zorder=2)
        costs.plot(grid, base + curve, color=k.TEAL_DARK, lw=1.5, zorder=3)
        mean = view["chain"]["cost_mean_given_count"][count - 1]
        top = np.interp(np.log10(mean), grid, curve)
        costs.plot([np.log10(mean)] * 2, [base, base + top], color=k.TEAL_DARK, lw=1.2, zorder=4)
        fig.text(column_x, y[count], f"€{k.num(mean)}", fontproperties=k.font("SemiBold", 12.5), color=k.TEAL_DARK,
                 ha="right", va="center", transform=column)
    costs.axhline(y[0] - 0.32, color=k.HAIR, lw=0.8, zorder=1)
    k.value(costs, 1.05, y[0], "€0, for certain", k.TEAL, size=12.5)
    costs.set_xlim(1.0, 5.0)
    costs.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 5]))
    costs.set_xticklabels([_euros(10.0 ** e) for e in range(1, 6)], fontproperties=k.F.tick)
    costs.yaxis.set_major_locator(NullLocator())
    costs.set_xlabel("cost per claim, given the count", fontproperties=k.F.axis, color=k.INK, labelpad=10)
    k.rich(bars, 0.0, 1.0, [("One policy, as ", k.INK, k.F.panel), ('task="chain"', k.TEAL, k.F.mono_bold),
                            (" draws it", k.INK, k.F.panel)],
           transform=k._above(bars, 0.5), va="bottom")
    features = view["features"]
    age = int(features["VehAge"])
    car = "a new car" if age == 0 else f"a car {age} year{'s' if age != 1 else ''} old"
    bars.text(-0.75, 1.0, f"Bonus-malus {features['BonusMalus']:g}, driver aged {features['DrivAge']:g}, {car}: "
              f"chance of a claim {k.pct(view['claim_probability'], 0)}",
              transform=k._above(bars, 0.12), fontproperties=k.F.small, color=k.MUTED, ha="left", va="bottom")

    # ---- right: the portfolio's exceedance curve
    ax = axes(7.0, width - 7.0 - 0.1)
    k.style(ax, frame="lb", grid="y")
    t = np.asarray(curves["thresholds"])
    observed = np.asarray(curves["exceedance"]["observed"])
    keep = observed > 0
    ax.plot(t[keep], observed[keep], color=k.INK, lw=1.7, zorder=6, drawstyle="steps-post")
    for key, _label, name, dash, lw in CURVES:
        p = np.asarray(curves["exceedance"][key])
        ax.plot(t[p > 0], p[p > 0], color=colour[name], lw=lw, linestyle=dash, zorder=5 if key == "chain" else 4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1e6)
    ax.set_ylim(1e-5, 0.12)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1e3, 1e4, 1e5, 1e6]))
    ax.set_xticklabels([_euros(v) for v in (1, 10, 100, 1e3, 1e4, 1e5, 1e6)])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_locator(FixedLocator([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
    ax.set_yticklabels(["0.001%", "0.01%", "0.1%", "1%", "10%"])
    ax.yaxis.set_minor_locator(NullLocator())
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(k.F.tick)
    ax.set_xlabel("t, a policy's total cost for the year", fontproperties=k.F.axis, color=k.INK, labelpad=10)
    k.panel_title(ax, "Every held-out policy: the share with T above t", raise_by=0.5)

    def at(key, where):
        return float(np.interp(np.log10(where), np.log10(t), np.asarray(curves["exceedance"][key])))

    k.value(ax, 1.5, at("observed", 1.5) * 1.3, f"what happened: {k.pct(observed[0], 2)} above €1",
            k.INK, va="bottom", size=12.5)
    apart = curves["exceedance"]["independent heads"][0]
    k.value(ax, 1.5, at("independent heads", 1.5) * 0.7, f"two heads fitted apart: {k.pct(apart, 2)}", k.SLATE,
            va="top", size=12.5)
    handles = [Line2D([], [], color=k.INK, lw=1.7, label="what happened")]
    handles += [Line2D([], [], color=colour[name], lw=lw, linestyle=dash, label=label)
                for _key, label, name, dash, lw in reversed(CURVES)]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.015, 0.015), frameon=False, prop=k.F.label,
              handlelength=2.2, handletextpad=0.6, labelspacing=0.45, labelcolor=k.INK, borderaxespad=0)

    # ---- the numbers, split 0
    k.rich(fig, box.left / box.width, (0.34 + 0.32 + 0.1) / box.height, [
        ("Joint NLL per policy ", k.MUTED, k.F.label),
        (f"{scores['chain']['nll']:.4f}", k.TEAL, k.F.value),
        (" for the chain, ", k.MUTED, k.F.label),
        (f"{scores['LightGBM Poisson × gamma']['nll']:.4f}", k.CLAY, k.F.value),
        (" for LightGBM Poisson × gamma, ", k.MUTED, k.F.label),
        (f"{scores['independent heads']['nll']:.4f}", k.SLATE, k.F.value),
        (" fitted apart.  CRPS of T: ", k.MUTED, k.F.label),
        (f"{scores['chain']['crps']:.1f}", k.TEAL, k.F.value),
        (" against ", k.MUTED, k.F.label),
        (f"{scores['LightGBM Poisson × gamma']['crps']:.1f}", k.CLAY, k.F.value),
        (", a tie.", k.MUTED, k.F.label),
    ], va="bottom")
    k.footnote(fig, "Curves are exact, not drawn: the mean over policies of Σ P(N = n) P(S > t / n | N = n). "
                    "Frequency × severity by hand (dotted) lies under the chain.")
    k.save(fig, NAME, folder=Path(arguments["images"]) if "images" in arguments else None)


STAGES = {"compute": compute, "draw": draw}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "draw"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
