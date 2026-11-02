"""Forty futures for one store: the joint normal's paths on Rossmann, against its own marginals made independent.

The README's forecasting figure. Two stages, run from the repo root:

    JAX_PLATFORMS=cpu python examples/readme/forecast_paths.py compute   # where Rossmann is: the Mac
    python examples/readme/forecast_paths.py draw                        # where Inter is: the cloud

``compute`` runs the protocol of ``examples/benchmark_paths.py`` for one seed
(300 stores, seed 0): it fits ``joint``, the joint normal over the 12 days
(``multivariate_gaussian``, rank 4) with the covariates, picks test rows by a
fixed rule, and for each draws 400 paths from the joint and 400 from its
marginals with the dependence between days dropped
(``Independent(dist.marginals())``), capped as the benchmark caps them. The
rule: origins whose 28 days of history are complete and whose 12 days hold
one to three closed days, a promotion day, and a state holiday (dropped if no
row has one), in a seeded random order; the first ``candidates`` are kept and
the figure draws the first. It writes ``examples/readme/artifacts/forecast-paths/``:

- ``paths-<k>.npz``: candidate ``k``'s draws in sales (400 × 12, joint and
  independent), its realized path, and the 28 days before the origin and
  the 12 after: sales, open, promotion, state and school holiday.
- ``meta.json``: the rule, the cap, the candidates' stores and origins, and
  the benchmark's saved scores on all three seeds (``$ASHLAR_BENCH/paths``),
  copied, not refitted: the daily loss, the 12-day total, and the peak day,
  for every model, Chronos-2 and DeepAR included.

Options: ``seed=0``, ``stores=300``, ``candidates=6``; ``epochs=N`` caps
training, for a quick check of the stage only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

ARTIFACTS = HERE / "artifacts" / "forecast-paths"
SEED = 0
STORES = 300
CANDIDATES = 6
SAVED_MODELS = ("joint", "independent", "joint_flow", "no_covariates", "chronos2", "deepar")
SAVED_SEEDS = (0, 1, 2)
DAILY_KEYS = ("quantile_loss", "median_loss", "uncapped_quantile_loss", "coverage")
PATH_KEYS = ("crps", "coverage", "width", "brier", "ece", "event_rate", "mean_probability", "calibrated_coverage",
             "calibrated_width", "calibrated_brier", "calibrated_ece", "calibrated_mean_probability")


def _paths():
    import benchmark_paths

    return benchmark_paths


def choose(paths, data: pd.DataFrame, test, seed: int, count: int) -> tuple[np.ndarray, str]:
    """Test rows by the fixed rule in the docstring, in a seeded random order: ``(rows, the rule as text)``."""
    lags = test.X[[f"sales_lag{j}" for j in range(1, paths.LOOKBACK + 1)]].to_numpy(np.float64)
    opened = test.X[[f"open_step{h}" for h in range(1, paths.HORIZON + 1)]].to_numpy(np.float64)
    promo = test.X[[f"promo_step{h}" for h in range(1, paths.HORIZON + 1)]].to_numpy(np.float64)
    holiday = test.X[[f"state_holiday_step{h}" for h in range(1, paths.HORIZON + 1)]].astype(str).to_numpy()
    closed = np.sum(opened == 0.0, axis=1)
    complete = np.isfinite(lags).all(axis=1) & np.isfinite(opened).all(axis=1) & np.isfinite(promo).all(axis=1)
    base = complete & (closed >= 1) & (closed <= 3) & (np.nansum(promo, axis=1) >= 1)
    with_holiday = base & np.any(~np.isin(holiday, ["0", "0.0", "None", "nan"]), axis=1)
    if with_holiday.any():
        eligible, rule = with_holiday, "complete history; 1 to 3 closed days, a promotion day, and a state holiday ahead"
    else:
        eligible, rule = base, "complete history; 1 to 3 closed days and a promotion day ahead"
    rows = np.flatnonzero(eligible)
    order = np.random.default_rng(seed).permutation(len(rows))
    return rows[order[:count]], rule


def _window(data: pd.DataFrame, store, origin, lookback: int, horizon: int) -> pd.DataFrame:
    days = pd.date_range(pd.Timestamp(origin) - pd.Timedelta(days=lookback - 1), periods=lookback + horizon, freq="D")
    part = data[data["store"] == store].set_index("day")
    return part.reindex(days)


def _saved_scores(paths) -> dict:
    """The benchmark's saved per-seed scores, trimmed: nothing is refitted."""
    out = {}
    for name in SAVED_MODELS:
        for seed in SAVED_SEEDS:
            file = paths.ROOT / f"{name}-{STORES}-{seed}.json"
            if not file.exists():
                continue
            run = json.loads(file.read_text())
            entry = {"daily": {key: run["daily"][key] for key in DAILY_KEYS if key in run["daily"]}}
            if "paths" in run:
                entry["paths"] = {kind: {key: run["paths"][kind][key] for key in PATH_KEYS if key in run["paths"][kind]}
                                  for kind in ("total", "peak")}
            for key in ("nll", "seconds"):
                if key in run:
                    entry[key] = run[key]
            out.setdefault(name, {})[str(seed)] = entry
            if name == "joint" and "joint_marginals" in run:
                loose = run["joint_marginals"]
                out.setdefault("joint_marginals", {})[str(seed)] = {
                    "daily": {key: loose["daily"][key] for key in DAILY_KEYS if key in loose["daily"]},
                    "paths": {kind: {key: loose["paths"][kind][key] for key in PATH_KEYS if key in loose["paths"][kind]}
                              for kind in ("total", "peak")},
                }
    return out


def compute(seed: int = SEED, stores: int = STORES, candidates: int = CANDIDATES, epochs=None):
    import jax

    from ashlar import ASRRegressor
    from ashlar.distributions import Independent

    paths = _paths()
    started = time.perf_counter()
    data = paths.long_table(stores, seed)
    train, _calibration, test = paths.frames(data, covariates=True)
    cap = paths.cap_of(train)
    settings = dict(paths.MODELS["joint"], random_state=seed)
    if epochs is not None:
        settings["max_epochs"] = int(epochs)
    model = ASRRegressor(**settings).fit(train.X, train.Y)
    fit_seconds = time.perf_counter() - started
    print(f"{len(train.X):,} training rows, {len(test.X):,} test rows; fitted in {fit_seconds:.0f} s", flush=True)

    rows, rule = choose(paths, data, test, seed, candidates)
    X = test.X.iloc[rows].reset_index(drop=True)
    joint = np.asarray(model.sample(X, n=paths.N_DRAWS, seed=seed), dtype=np.float64)
    loose = Independent(model.predict_dist(X).marginals())
    loose = np.asarray(loose.sample(jax.random.PRNGKey(seed), (paths.N_DRAWS,)), dtype=np.float64)
    joint, loose = np.expm1(np.minimum(joint, cap)), np.expm1(np.minimum(loose, cap))

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    for stale in ARTIFACTS.glob("paths-*.npz"):
        stale.unlink()
    chosen = []
    for k, row in enumerate(rows):
        store = test.index["series"].iloc[row]
        origin = pd.Timestamp(test.index["origin"].iloc[row])
        window = _window(data, store, origin, paths.LOOKBACK, paths.HORIZON)
        np.savez_compressed(
            ARTIFACTS / f"paths-{k}.npz",
            joint=joint[:, k].astype(np.float32),
            independent=loose[:, k].astype(np.float32),
            realized=np.expm1(test.Y[row]).astype(np.float32),
            day=window.index.strftime("%Y-%m-%d").to_numpy().astype("U10"),
            sales=np.expm1(window["sales"].to_numpy(np.float64)).astype(np.float32),
            open=window["open"].to_numpy(np.float64).astype(np.float32),
            promo=window["promo"].to_numpy(np.float64).astype(np.float32),
            state_holiday=window["holiday"].to_numpy(np.float64).astype(np.float32),
            school_holiday=window["school_holiday"].to_numpy(np.float64).astype(np.float32),
        )
        chosen.append({"k": k, "store": str(store), "origin": origin.strftime("%Y-%m-%d"),
                       "store_type": str(test.X["store_type"].iloc[row])})
        total = joint[:, k].sum(axis=1), loose[:, k].sum(axis=1)
        print(f"candidate {k}: store {store}, origin {origin:%Y-%m-%d}; 12-day total {np.expm1(test.Y[row]).sum():,.0f}, "
              f"joint 80% [{np.quantile(total[0], 0.1):,.0f}, {np.quantile(total[0], 0.9):,.0f}], "
              f"independent [{np.quantile(total[1], 0.1):,.0f}, {np.quantile(total[1], 0.9):,.0f}]", flush=True)
    meta = {
        "seed": seed,
        "stores": stores,
        "epochs": epochs,
        "draws": int(paths.N_DRAWS),
        "cap_log1p": cap,
        "cap_sales": float(np.expm1(cap)),
        "rule": rule,
        "eligible_note": "candidates in a seeded random order (numpy default_rng(seed)); the figure draws candidate 0",
        "fit_seconds": round(fit_seconds, 1),
        "candidates": chosen,
        "saved": _saved_scores(paths),
        "seconds": round(time.perf_counter() - started, 1),
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    for path in sorted(ARTIFACTS.iterdir()):
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)", flush=True)


# ---------------------------------------------------------------- draw

FIGURE = "forecast-paths"
SHOWN = 40  # paths drawn of each kind
HISTORY_SHOWN = 11  # days of history drawn before the origin
ASHLAR_HEADS = ("joint", "independent", "joint_flow")


def seed_mean(saved: dict, model: str, *keys):
    """The mean over the saved seeds of ``saved[model][seed][keys...]``, or None."""
    values = []
    for run in saved.get(model, {}).values():
        for key in keys:
            run = run.get(key) if isinstance(run, dict) else None
        if run is not None:
            values.append(float(run))
    return float(np.mean(values)) if values else None


def headline(saved: dict) -> dict:
    """The numbers the figure prints, each a mean over the benchmark's saved seeds."""
    calibrated_peak = {name: seed_mean(saved, name, "paths", "peak", "calibrated_brier") for name in ASHLAR_HEADS}
    calibrated_peak = {name: value for name, value in calibrated_peak.items() if value is not None}
    best = min(calibrated_peak, key=calibrated_peak.get) if calibrated_peak else None
    rate = seed_mean(saved, "joint", "paths", "peak", "event_rate")
    return {
        "joint_total_coverage": seed_mean(saved, "joint", "paths", "total", "coverage"),
        "marginal_total_coverage": seed_mean(saved, "joint_marginals", "paths", "total", "coverage"),
        "joint_total_crps": seed_mean(saved, "joint", "paths", "total", "crps"),
        "deepar_total_crps": seed_mean(saved, "deepar", "paths", "total", "crps"),
        "joint_daily": seed_mean(saved, "joint", "daily", "quantile_loss"),
        "lags_daily": seed_mean(saved, "no_covariates", "daily", "quantile_loss"),
        "chronos_daily": seed_mean(saved, "chronos2", "daily", "quantile_loss"),
        "deepar_peak_brier": seed_mean(saved, "deepar", "paths", "peak", "brier"),
        "best_peak_brier": calibrated_peak.get(best),
        "best_peak_head": best,
        "peak_rate": rate,
        "base_rate_brier": None if rate is None else rate * (1 - rate),
        "seeds": len(saved.get("joint", {})),
    }


def draw():
    import matplotlib.patheffects as path_effects
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import FixedLocator, FuncFormatter

    import style as k

    meta = k.load_json(FIGURE, "meta")
    first = k.load_npz(FIGURE, "paths-0")
    numbers = headline(meta["saved"])
    joint = np.asarray(first["joint"], dtype=np.float64)
    loose = np.asarray(first["independent"], dtype=np.float64)
    realized = np.asarray(first["realized"], dtype=np.float64)
    sales = np.asarray(first["sales"], dtype=np.float64)
    days = pd.to_datetime(first["day"])
    lookback = len(days) - joint.shape[1]
    ahead = np.arange(lookback, len(days))
    start = lookback - HISTORY_SHOWN
    x_hist = np.arange(start, lookback)
    halo = [path_effects.withStroke(linewidth=4.0, foreground=k.CARD)]
    candidate = meta["candidates"][0]

    k.setup()
    fig = k.card(
        f"Days that move together: the 12-day total's interval holds {k.pct(numbers['joint_total_coverage'])}, "
        f"not {k.pct(numbers['marginal_total_coverage'])}.",
        f"Rossmann, {meta['stores']} stores: the next 12 days of log sales from the 28 before and the known days "
        "ahead (open, promotion, holidays). Top: 40 of 400 paths from a joint normal over the 12 days; bottom: 40 "
        f"from the same days' forecasts, drawn independently. Store {candidate['store']} from "
        f"{pd.Timestamp(candidate['origin']):%-d %B %Y}, by a fixed rule; the coverage is over every store and "
        f"origin, {WORDS.get(numbers['seeds'], numbers['seeds'])} seeds.",
        height=10.6, footer=3,
    )
    top = k.panel(fig, rect=(0.0, 0.60, 0.71, 1.0), head=0.45, tail=0.1, lpad=1.3, rpad=0.1, grid="y")
    bottom = k.panel(fig, rect=(0.0, 0.215, 0.71, 0.615), head=0.45, tail=0.1, lpad=1.3, rpad=0.1, grid="y")
    strip = k.panel(fig, rect=(0.0, 0.0, 0.71, 0.225), head=0.04, tail=0.55, lpad=1.3, rpad=0.1)
    totals_top = k.panel(fig, rect=(0.755, 0.60, 1.0, 1.0), head=0.45, tail=0.1, lpad=0.05, rpad=0.05)
    totals_bottom = k.panel(fig, rect=(0.755, 0.215, 1.0, 0.615), head=0.45, tail=0.1, lpad=0.05, rpad=0.05)

    reach = max(np.quantile(joint[:SHOWN], 0.995), np.quantile(loose[:SHOWN], 0.995), sales[start:].max()) * 1.08
    for ax, draws, colour, pale, title in (
        (top, joint, k.TEAL, k.TEAL_WASH, "Joint normal: each path keeps its level"),
        (bottom, loose, k.CLAY, k.CLAY_WASH, "Same days, drawn independently: the errors cancel"),
    ):
        low, high = np.quantile(draws, [0.1, 0.9], axis=0)
        ax.fill_between(ahead, low, high, color=pale, lw=0, zorder=1)
        segments = [np.c_[np.r_[lookback - 1, ahead], np.r_[sales[lookback - 1], path]] for path in draws[:SHOWN]]
        ax.add_collection(LineCollection(segments, colors=colour, linewidths=0.9, alpha=0.38, zorder=2))
        # The two shown paths with the highest and the lowest 12-day totals, drawn dark.
        sums = draws[:SHOWN].sum(axis=1)
        for pick in (int(np.argmax(sums)), int(np.argmin(sums))):
            ax.plot(np.r_[lookback - 1, ahead], np.r_[sales[lookback - 1], draws[pick]], color=k.shade(colour, -0.3),
                    lw=1.9, zorder=3.5)
        ax.plot(x_hist, sales[start:lookback], color=k.DATA_DARK, lw=1.8, zorder=3)
        ax.plot(np.r_[lookback - 1, ahead], np.r_[sales[lookback - 1], realized], color=k.INK, lw=2.2, zorder=4)
        ax.axvline(lookback - 0.5, color=k.HAIR, lw=1.0, zorder=0.5)
        ax.set_xlim(start - 0.5, len(days) - 0.5)
        ax.set_ylim(0, reach)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value / 1000:.0f}k" if value else "0"))
        ax.tick_params(labelbottom=False)
        k.style(ax, frame="lb", grid="y")
        k.panel_title(ax, title, color=k.TEAL_DARK if colour == k.TEAL else k.CLAY_DARK, raise_by=0.08)
    k.value(top, lookback - 1.0, reach * 0.95, f"the last {HISTORY_SHOWN} days", k.DATA_DARK, ha="right", va="top",
            size=12)
    k.value(top, len(days) - 0.7, reach * 0.95, "the next 12, realized", k.INK, ha="right", va="top",
            size=12).set_path_effects(halo)
    k.value(bottom, start + 0.3, reach * 0.95, "dark: the highest and lowest totals", k.MUTED,
            ha="left", va="top", size=12).set_path_effects(halo)

    # Under the paths: the days the model is told about, history and ahead alike.
    marks = (("closed", np.asarray(first["open"]) == 0), ("promotion", np.asarray(first["promo"]) == 1),
             ("state holiday", np.asarray(first["state_holiday"]) > 0),
             ("school holiday", np.asarray(first["school_holiday"]) > 0))
    for row, (label, on) in enumerate(marks):
        y = len(marks) - 1 - row
        where = np.flatnonzero(on[start:]) + start
        colours = [k.INK if day >= lookback else k.DATA_DARK for day in where]
        strip.scatter(where, np.full(len(where), y), s=34, marker="s", color=colours, lw=0, zorder=2)
        strip.text(start - 0.9, y, label, fontproperties=k.F.small, color=k.MUTED, ha="right", va="center")
    strip.axvline(lookback - 0.5, color=k.HAIR, lw=1.0, zorder=0.5)
    strip.set_xlim(start - 0.5, len(days) - 0.5)
    strip.set_ylim(-0.7, len(marks) - 0.3)
    strip.set_yticks([])
    weeks = [day for day in range(start, len(days)) if days[day].dayofweek == 0]
    strip.xaxis.set_major_locator(FixedLocator(weeks))
    strip.set_xticklabels([f"{days[day]:%-d %b}" for day in weeks])
    k.style(strip, frame="b")

    # Right: the 12-day total from each set of 400 paths, against the realized total.
    total_joint, total_loose, total = joint.sum(axis=1), loose.sum(axis=1), realized.sum()
    lo = min(np.quantile(total_joint, 0.005), np.quantile(total_loose, 0.005), total)
    hi = max(np.quantile(total_joint, 0.995), np.quantile(total_loose, 0.995), total)
    bins = np.linspace(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo), 34)
    tallest = max(np.histogram(total_joint, bins)[0].max(), np.histogram(total_loose, bins)[0].max())
    for ax, totals, colour, pale, head in ((totals_top, total_joint, k.TEAL, k.TEAL_PALE, "the 12-day total"),
                                           (totals_bottom, total_loose, k.CLAY, k.CLAY_PALE, None)):
        counts, edges = np.histogram(totals, bins)
        a, b = np.quantile(totals, [0.1, 0.9])
        centres = (edges[:-1] + edges[1:]) / 2
        inside = (centres >= a) & (centres <= b)
        ax.bar(centres, counts, width=np.diff(edges) * 0.86, color=np.where(inside, colour, pale), lw=0, zorder=2)
        ax.plot([a, b], [-0.07 * tallest] * 2, color=colour, lw=3.0, solid_capstyle="butt", zorder=3)
        ax.plot([total, total], [-0.04 * tallest, tallest * 1.14], color=k.INK, lw=2.0, zorder=4,
                solid_capstyle="butt")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_ylim(-0.3 * tallest, tallest * 1.25)
        ax.set_yticks([])
        ax.tick_params(labelbottom=False)
        k.style(ax, frame="")
        held = a <= total <= b
        k.value(ax, (a + b) / 2, -0.12 * tallest, f"80%: {k.num(a / 1000)}k to {k.num(b / 1000)}k", colour,
                ha="center", va="top", size=12)
        k.value(ax, total, tallest * 1.18, "realized" + ("" if held else ", outside"), k.INK, ha="center",
                va="bottom", size=12).set_path_effects(halo)
        if head:
            k.panel_title(ax, head, raise_by=0.08)

    rate = numbers["peak_rate"]
    k.rich(fig, fig.box.left / fig.box.width, (0.34 + 2 * 0.32 + 0.06) / fig.box.height, [
        (f"{WORDS.get(numbers['seeds'], numbers['seeds']).capitalize()} seeds, {meta['stores']} stores: "
         "the total's CRPS ", k.INK, k.F.label),
        (k.num(numbers["joint_total_crps"]), k.TEAL, k.F.value), (" against ", k.INK, k.F.label),
        (k.num(numbers["deepar_total_crps"]), k.CLAY, k.F.value), (" for DeepAR. Covariates cut the daily loss from ",
                                                                 k.INK, k.F.label),
        (k.num(numbers["lags_daily"]), k.SLATE, k.F.value), (" to ", k.INK, k.F.label),
        (k.num(numbers["joint_daily"]), k.TEAL, k.F.value), (".", k.INK, k.F.label),
    ], va="bottom")
    k.footnote(fig, f"Work in progress. Chronos-2, zero-shot, is level day by day ({numbers['chronos_daily']:.1f} "
                    f"against {numbers['joint_daily']:.1f}) and draws no paths. DeepAR calibrates the peak day best, "
                    f"Brier {numbers['deepar_peak_brier']:.3f} against {numbers['best_peak_brier']:.3f}, and none "
                    f"beats always saying {k.pct(rate)}, {numbers['base_rate_brier']:.3f}.", line=0)
    k.save(fig, FIGURE)

    def mark(ax):
        days_shown = np.arange(lookback - 10, len(days))
        segments = [np.c_[np.r_[lookback - 1, ahead], np.r_[sales[lookback - 1], path]] for path in joint[:SHOWN]]
        ax.add_collection(LineCollection(segments, colors=k.TEAL, linewidths=1.0, alpha=0.4, zorder=2))
        ax.plot(days_shown[days_shown < lookback], sales[days_shown[days_shown < lookback]], color=k.DATA_DARK, lw=2.4,
                zorder=3)
        ax.plot(np.r_[lookback - 1, ahead], np.r_[sales[lookback - 1], realized], color=k.INK, lw=2.6, zorder=4)
        ax.set_xlim(days_shown[0] - 0.3, len(days) - 0.7)
        ax.set_ylim(0, reach)

    k.thumb("forecast", "Forecasts, in progress", [
        (k.num(numbers["joint_total_crps"]), k.TEAL), (" total CRPS, against ", k.MUTED),
        (k.num(numbers["deepar_total_crps"]), k.CLAY), (" for DeepAR", k.MUTED),
    ], mark)


WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def main(argv):
    stages = [arg for arg in argv if "=" not in arg] or ["draw"]
    options = dict(arg.split("=", 1) for arg in argv if "=" in arg)
    for stage in stages:
        if stage == "compute":
            compute(int(options.get("seed", SEED)), int(options.get("stores", STORES)),
                    int(options.get("candidates", CANDIDATES)), options.get("epochs"))
        elif stage == "draw":
            draw()
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
