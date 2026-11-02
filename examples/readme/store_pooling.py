"""Small groups borrow from large ones: Rossmann's stores, pooled towards the population.

The README's pooling figure. Two stages, run from the repo root:

    JAX_PLATFORMS=cpu python examples/readme/store_pooling.py compute   # where Rossmann is: the Mac
    python examples/readme/store_pooling.py draw                        # where Inter is: the cloud

``compute`` fits one split (seed 0) of the protocol of
``examples/benchmark_effects.py`` with that benchmark's own pooled model,
``ASRRegressor(task="gaussian", random_effects={"store": ["intercept",
"embedding"]})`` on log sales, and writes ``examples/readme/artifacts/store-pooling/``:

- ``stores.json``: for every seen store, its training rows; its raw mean
  residual, the mean over those rows of log sales minus the population's
  prediction (the same model with the store read as missing), which is what
  an unpooled store effect chases; its pooled effect, the model's own mean
  minus the population's over the same rows (offset and embedding together);
  and its offset's posterior mean and sd. Also ``tau``, the learned spread
  of the offsets.
- ``unseen.npz``: every unseen store's 30 test rows with the raw 80% interval
  before and after ``with_context`` on its 5 context rows, and per store its
  offset's posterior after the update, the mean residual of its context rows
  and of its test rows from the population. And each unseen store's
  learning path: its offset's posterior after its first 1, 2, 3, 5, 10, 20
  and 35 rows, each from the prior, beside its level, its mean residual over
  all its rows in the table.
- ``buckets.json``: the benchmark's saved per-split CRPS and coverage by
  bucket for every method (``effects-rossmann.json``; nothing is refitted),
  or, if that file is gone, the three-split means of the table in
  ``docs/benchmarks.md``.
- ``check.json``: this fit's CRPS by bucket, beside the saved seed-0 run.

``draw`` reads only the artifacts.

Rossmann is the fev benchmark's ``rossmann/1D``, saved as
``$ASHLAR_BENCH/rossmann-1D.parquet`` (default ``/tmp/ashlar-bench``).
Options: ``seed=0``; ``epochs=N`` caps training, for a quick check of the stage only.
"""

from __future__ import annotations

import json
import os
import re
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

ARTIFACTS = HERE / "artifacts" / "store-pooling"
TABLE = "rossmann"
SEED = 0
METHOD = "pooled_embedding"


def _effects():
    import benchmark_effects

    if "ASHLAR_BENCH" in os.environ:
        benchmark_effects.DATA_DIR = Path(os.environ["ASHLAR_BENCH"])
    return benchmark_effects


def _fit(effects, x, y, rows, seed, broader, epochs):
    if epochs is None:
        return effects._ashlar(METHOD, x, y, rows, seed, broader)
    from ashlar import ASRRegressor

    frame = x.iloc[rows].reset_index(drop=True)
    return ASRRegressor(
        task="gaussian", random_state=seed, max_epochs=int(epochs), random_effects={"store": ["intercept", "embedding"]}
    ).fit(frame, y[rows])


def _missing_store(frame: pd.DataFrame) -> pd.DataFrame:
    """The rows with the store unknown: the model reads the population, offset N(0, tau^2) and the average embedding."""
    out = frame.copy()
    out["store"] = pd.Series([None] * len(out), dtype=object)
    return out


def _round(values, digits=6):
    return [round(float(value), digits) for value in np.asarray(values, dtype=np.float64)]


def _saved_buckets(effects) -> dict:
    """Per-split scores by bucket from the benchmark's saved run, trimmed to what the figure reads."""
    path = effects.DATA_DIR / f"effects-{TABLE}.json"
    if not path.exists():
        return _doc_buckets()
    data = json.loads(path.read_text())[TABLE]
    runs = {}
    for seed, run in sorted(data["runs"].items()):
        methods = {}
        for method, scores in run.items():
            if not isinstance(scores, dict):
                continue
            cells = {
                bucket: {key: value[key] for key in ("crps", "coverage", "width", "rows") if key in value}
                for bucket, value in scores.items()
                if isinstance(value, dict)
            }
            if "tau" in scores:
                cells["tau"] = scores["tau"]
            methods[method] = cells
        runs[seed] = methods
    return {"source": str(path), "rows": data["rows"], "stores": data["stores"], "runs": runs}


def _doc_buckets() -> dict:
    """The Rossmann table of docs/benchmarks.md, three-split means, if the saved run is not on this machine."""
    text = (REPO / "docs" / "benchmarks.md").read_text()
    section = text.split("### Rossmann, log daily sales", 1)[1].split("###", 1)[0]
    names = {
        "embedding": "embedding", "dropped": "dropped", "pooled": "pooled", "pooled + embedding": "pooled_embedding",
        "nested": "nested", "in context": "in_context", "GPBoost": "gpboost", "mixed model": "mixedlm",
    }
    columns = ["1-5", "6-20", "21-100", "101-300", "unseen", "unseen_context"]
    means = {}
    for line in section.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or cells[0] not in names:
            continue
        values = [re.sub(r"[*%,]", "", cell) for cell in cells[1:]]
        method = {}
        for column, value in zip(columns, values[:6]):
            if value:
                method[column] = {"crps": float(value)}
        if values[6]:
            method["unseen"]["coverage"] = round(float(values[6]) / 100, 4)
        if values[7]:
            method["1-5"]["coverage"] = round(float(values[7]) / 100, 4)
        if values[8]:
            method["tau"] = float(values[8])
        means[names[cells[0]]] = method
    return {"source": "docs/benchmarks.md#entity-effects (three-split means)", "splits": 3, "runs": {"mean": means}}


PATH_STEPS = (1, 2, 3, 5, 10, 20, 35)


def _learning_path(model, x, y, entity, context, unseen, names) -> dict:
    """Each unseen store's offset after reading its first k rows (its context rows, then its test rows), from the prior.

    ``path_mean`` and ``path_sd`` are ``(stores, steps)``, one ``with_context``
    from the fitted model per step, so every step starts from ``N(0, tau^2)``.
    ``level`` is the store's mean residual from the population over all its
    rows in the table (``level_rows`` of them), the level its offset would
    settle on; no row of an unseen store was fitted.
    """
    queue = {store: [] for store in names}
    for row in np.concatenate([context, unseen]):
        queue[entity[row]].append(int(row))
    means, sds = [], []
    for step in PATH_STEPS:
        rows = np.asarray([row for store in names for row in queue[store][:step]])
        adapted = model.with_context(x.iloc[rows].reset_index(drop=True), y[rows])
        updated = adapted.random_effects_["store"]
        where = {store: index for index, store in enumerate(updated["categories"])}
        means.append([np.asarray(updated["mean"])[where[store], 0] for store in names])
        sds.append([np.asarray(updated["sd"])[where[store], 0] for store in names])
    every = np.flatnonzero(np.isin(entity, np.asarray(names, dtype=entity.dtype)))
    population = np.asarray(model.predict(_missing_store(x.iloc[every].reset_index(drop=True))), dtype=np.float64)
    level = pd.Series(y[every] - population).groupby(entity[every]).agg(["mean", "size"]).loc[names]
    mean, sd = np.asarray(means, dtype=np.float64).T, np.asarray(sds, dtype=np.float64).T
    inside = np.abs(level["mean"].to_numpy()[:, None] - mean) <= 1.2815515655446004 * sd
    print("new stores whose 80% interval for the offset holds their level, by rows read: "
          + ", ".join(f"{step}: {share:.0%}" for step, share in zip(PATH_STEPS, inside.mean(axis=0))), flush=True)
    return {
        "path_steps": np.asarray(PATH_STEPS, dtype=np.int16),
        "path_mean": mean.astype(np.float32),
        "path_sd": sd.astype(np.float32),
        "level": level["mean"].to_numpy(np.float32),
        "level_rows": level["size"].to_numpy(np.int32),
    }


def compute(seed: int = SEED, epochs=None):
    effects = _effects()
    started = time.perf_counter()
    x, y, broader = effects.rossmann()
    entity = x["store"].to_numpy()
    train, _train_count, test, test_count, context, unseen = effects.split(entity, seed)
    print(f"{len(train):,} training rows, {len(np.unique(entity[train])):,} seen stores, "
          f"{len(np.unique(entity[unseen])):,} unseen", flush=True)
    model = _fit(effects, x, y, train, seed, broader, epochs)
    fit_seconds = time.perf_counter() - started
    effect = model.random_effects_["store"]
    tau = float(np.asarray(effect["tau"]).reshape(-1)[0])
    print(f"fitted in {fit_seconds:.0f} s, tau {tau:.3f}", flush=True)

    # Seen stores: raw residual from the population against the pooled effect and the offset's posterior.
    frame = x.iloc[train].reset_index(drop=True)
    own = np.asarray(model.predict(frame), dtype=np.float64)
    population = np.asarray(model.predict(_missing_store(frame)), dtype=np.float64)
    per = (
        pd.DataFrame({"store": entity[train], "residual": y[train] - population, "pooled": own - population})
        .groupby("store", sort=False)
        .agg(rows=("residual", "size"), raw=("residual", "mean"), pooled=("pooled", "mean"))
    )
    position = {store: index for index, store in enumerate(effect["categories"])}
    order = [store for store in per.index if store in position]
    index = np.asarray([position[store] for store in order])
    per = per.loc[order]
    stores = {
        "seed": seed,
        "tau": tau,
        "fit_seconds": round(fit_seconds, 1),
        "store": [str(store) for store in order],
        "rows": [int(value) for value in per["rows"]],
        "raw": _round(per["raw"]),
        "pooled": _round(per["pooled"]),
        "offset_mean": _round(np.asarray(effect["mean"])[index, 0]),
        "offset_sd": _round(np.asarray(effect["sd"])[index, 0]),
    }

    # Unseen stores: the population's interval, then the same model after five rows of each.
    held = x.iloc[unseen].reset_index(drop=True)
    shown = x.iloc[context].reset_index(drop=True)
    before = np.asarray(model.predict_interval(held, 0.8), dtype=np.float64)
    adapted = model.with_context(shown, y[context])
    after = np.asarray(adapted.predict_interval(held, 0.8), dtype=np.float64)
    held_population = np.asarray(model.predict(_missing_store(held)), dtype=np.float64)
    shown_population = np.asarray(model.predict(_missing_store(shown)), dtype=np.float64)
    updated = adapted.random_effects_["store"]
    names = sorted(set(entity[unseen].tolist()))
    where = {store: index for index, store in enumerate(updated["categories"])}
    held_store = entity[unseen]
    shown_store = entity[context]
    unseen_out = {
        "store": np.asarray(names),
        "posterior_mean": np.asarray([np.asarray(updated["mean"])[where[s], 0] for s in names], dtype=np.float32),
        "posterior_sd": np.asarray([np.asarray(updated["sd"])[where[s], 0] for s in names], dtype=np.float32),
        "context_residual": np.asarray(
            [np.mean((y[context] - shown_population)[shown_store == s]) for s in names], dtype=np.float32),
        "test_residual": np.asarray(
            [np.mean((y[unseen] - held_population)[held_store == s]) for s in names], dtype=np.float32),
        "row_store": np.asarray([names.index(s) for s in held_store], dtype=np.int16),
        "row_y": y[unseen].astype(np.float32),
        "row_population": held_population.astype(np.float32),
        "row_before": before.astype(np.float32),
        "row_after": after.astype(np.float32),
        "context_store": np.asarray([names.index(s) for s in shown_store], dtype=np.int16),
        "context_y": y[context].astype(np.float32),
        "context_population": shown_population.astype(np.float32),
    }
    inside_before = (before[:, 0] <= y[unseen]) & (y[unseen] <= before[:, 1])
    inside_after = (after[:, 0] <= y[unseen]) & (y[unseen] <= after[:, 1])
    print(f"unseen stores' raw 80% interval holds {inside_before.mean():.1%} before and {inside_after.mean():.1%} "
          "after five rows", flush=True)
    unseen_out.update(_learning_path(model, x, y, entity, context, unseen, names))

    # This fit's scores by bucket, to set beside the saved run of the same seed.
    seen = effects._ashlar_scores(model, x, y, test, seed, False)
    check = effects._bucketed(seen, test_count)
    check["unseen"] = effects._ashlar_scores(model, x, y, unseen, seed, False)(np.ones(len(unseen), dtype=bool))
    check["unseen_context"] = effects._ashlar_scores(adapted, x, y, unseen, seed, False)(np.ones(len(unseen), dtype=bool))
    buckets = _saved_buckets(effects)
    saved = buckets["runs"].get(str(seed), {}).get(METHOD, {})
    comparison = {
        bucket: {"this_fit": round(check[bucket]["crps"], 5), "saved": round(saved[bucket]["crps"], 5) if bucket in saved else None,
                 "coverage": round(check[bucket]["coverage"], 4)}
        for bucket in check
    }
    print("CRPS by bucket, this fit against the saved run:", json.dumps(comparison), flush=True)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "stores.json").write_text(json.dumps(stores))
    np.savez_compressed(ARTIFACTS / "unseen.npz", **unseen_out)
    (ARTIFACTS / "buckets.json").write_text(json.dumps(buckets, indent=1, sort_keys=True))
    (ARTIFACTS / "check.json").write_text(json.dumps(
        {"seed": seed, "epochs": epochs, "seconds": round(time.perf_counter() - started, 1), "crps": comparison,
         "unseen_coverage_before": float(inside_before.mean()), "unseen_coverage_after": float(inside_after.mean())},
        indent=1, sort_keys=True))
    for path in sorted(ARTIFACTS.iterdir()):
        print(f"wrote {path.relative_to(REPO)} ({path.stat().st_size / 1024:.0f} KB)", flush=True)


# ---------------------------------------------------------------- draw

BUCKETS = ("1-5", "6-20", "21-100", "101-300", "unseen", "unseen_context")
BUCKET_LABELS = ("1–5", "6–20", "21–100", "101–300", "new", "new,\n5 rows")
BUCKET_X = (0.0, 1.2, 2.4, 3.6, 5.0, 6.2)
Z80 = 1.2815515655446004
# Lines on the right, back to front: (method, colour, dashed, legend label).
LINES = (
    ("mixedlm", "SLATE", False, "linear mixed model"),
    ("embedding", "DATA_DARK", True, "ordinary embedding"),
    ("gpboost", "CLAY", False, "GPBoost"),
    ("pooled_embedding", "TEAL", False, "Ashlar, pooled"),
)


def bucket_stats(buckets: dict, method: str) -> dict:
    """Per bucket: the mean CRPS over the splits, its sd, and the mean raw 80% coverage where saved."""
    runs = [run[method] for run in buckets["runs"].values() if method in run]
    out = {}
    for bucket in BUCKETS:
        cells = [run[bucket] for run in runs if bucket in run]
        if not cells:
            continue
        crps = np.asarray([cell["crps"] for cell in cells])
        coverage = [cell["coverage"] for cell in cells if "coverage" in cell]
        out[bucket] = {"crps": float(crps.mean()), "sd": float(crps.std(ddof=1)) if len(crps) > 1 else 0.0,
                       "coverage": float(np.mean(coverage)) if coverage else None, "splits": len(cells)}
    return out


def highlighted(unseen: dict) -> int:
    """The new store drawn: the unseen store whose effect over its 30 test days, |mean residual from the
    population|, is nearest the 90th percentile among unseen stores. A fixed rule for a store that differs."""
    size = np.abs(unseen["test_residual"].astype(np.float64))
    return int(np.argmin(np.abs(size - np.quantile(size, 0.9))))


def draw():
    import matplotlib.patheffects as path_effects
    from matplotlib.collections import LineCollection
    from matplotlib.patches import FancyArrowPatch

    import style as k

    stores = k.load_json("store-pooling", "stores")
    unseen = k.load_npz("store-pooling", "unseen")
    buckets = k.load_json("store-pooling", "buckets")
    stats = {method: bucket_stats(buckets, method) for method, *_ in LINES}
    tau = float(stores["tau"])
    rows = np.asarray(stores["rows"], dtype=np.float64)
    raw = np.asarray(stores["raw"])
    pooled = np.asarray(stores["pooled"])
    sd = np.asarray(stores["offset_sd"])
    ours, gpb, emb = stats["pooled_embedding"], stats["gpboost"], stats["embedding"]
    splits = buckets.get("splits") or max(cell["splits"] for cell in ours.values())
    halo = [path_effects.withStroke(linewidth=4.5, foreground=k.CARD)]

    k.setup()
    fig = k.card(
        f"Stores with few rows borrow from the rest: CRPS {ours['1-5']['crps']:.3f} against GPBoost's "
        f"{gpb['1-5']['crps']:.3f}",
        f"Rossmann, {buckets.get('stores', 1115):,} stores, log daily sales given the weekday, promotions, holidays, "
        "competition, and the store's type. Left: one split, every seen store's effect against its training rows. "
        f"Right: CRPS by training rows per store, mean of {splits} splits; a fifth of the stores held out whole.",
        height=8.9, footer=2,
    )
    left = k.panel(fig, rect=(0.0, 0.0, 0.565, 1.0), head=0.62, tail=0.8, lpad=0.75, rpad=0.1, grid="y")
    right = k.panel(fig, rect=(0.62, 0.0, 1.0, 1.0), head=0.62, tail=0.8, lpad=0.72, rpad=0.05, grid="y")

    # Left: each store's own mean residual (grey), pulled to its pooled effect (teal) with its 80% interval.
    new_x, gap_x = 0.4, 0.72
    rng = np.random.default_rng(0)
    x = rows * np.exp(rng.uniform(-0.1, 0.1, len(rows)))
    limit = 0.95
    left.axhline(0.0, color=k.MUTED, lw=0.9, zorder=0.6)
    shown_raw = np.clip(raw, -limit, limit)
    left.add_collection(LineCollection(np.stack([np.c_[x, shown_raw], np.c_[x, pooled]], axis=1), colors=k.DATA,
                                       linewidths=0.8, zorder=1))
    left.scatter(x, shown_raw, s=10, color=k.DATA_DARK, lw=0, zorder=2)
    for sign, marker in ((1, "^"), (-1, "v")):
        out = sign * raw > limit
        left.scatter(x[out], np.full(int(out.sum()), sign * limit), s=24, marker=marker, color=k.DATA_DARK, lw=0, zorder=2)
    left.add_collection(LineCollection(np.stack([np.c_[x, pooled - Z80 * sd], np.c_[x, pooled + Z80 * sd]], axis=1),
                                       colors=k.TEAL, linewidths=1.0, alpha=0.38, zorder=3))
    left.scatter(x, pooled, s=12, color=k.TEAL, lw=0.4, edgecolor=k.CARD, zorder=4)

    # One new store: the population's 80% interval, then its own after five rows through with_context.
    pick = highlighted(unseen)
    truth = float(unseen["test_residual"][pick])
    shown = float(unseen["context_residual"][pick])
    after_mean, after_sd = float(unseen["posterior_mean"][pick]), float(unseen["posterior_sd"][pick])
    up = 1.0 if after_mean >= 0 else -1.0
    left.axvspan(left.get_xlim()[0] if False else 0.2, gap_x, color=k.CARD, lw=0, zorder=4.5)
    left.axvline(gap_x, color=k.HAIR, lw=1.0, zorder=4.6)
    left.plot([new_x, new_x], [-Z80 * tau, Z80 * tau], color=k.TEAL_PALE, lw=8, solid_capstyle="butt", zorder=5)
    left.scatter([new_x], [0.0], s=50, color=k.TEAL, edgecolor=k.CARD, lw=1.2, zorder=6)
    left.plot([5, 5], [after_mean - Z80 * after_sd, after_mean + Z80 * after_sd], color=k.TEAL_DARK, lw=8,
              solid_capstyle="butt", zorder=8)
    left.plot([5, 5], [shown, after_mean], color=k.MUTED, lw=1.2, zorder=8.2)
    left.scatter([5], [shown], s=44, color=k.DATA_DARK, edgecolor=k.CARD, lw=1.1, zorder=8.5)
    left.scatter([5], [after_mean], s=50, color=k.TEAL, edgecolor=k.CARD, lw=1.2, zorder=8.6)
    for x0, x1 in ((new_x * 0.8, new_x * 1.25), (5 / 1.3, 5 * 1.3)):
        left.plot([x0, x1], [truth, truth], color=k.INK, lw=2.0, zorder=9, solid_capstyle="butt")
    left.add_patch(FancyArrowPatch((new_x * 1.25, Z80 * tau * up * 0.55), (5 / 1.18, after_mean + 0.06 * up),
                                   connectionstyle=f"arc3,rad={-0.25 * up:.2f}", arrowstyle="-|>",
                                   mutation_scale=14, color=k.TEAL_DARK, lw=1.4, zorder=9))
    note = k.code(left, 1.55, Z80 * tau * up + 0.2 * up, "with_context", k.TEAL_DARK, bold=True, ha="center",
                  va="center", zorder=10)
    note.set_path_effects(halo)
    tag = k.value(left, new_x, -(Z80 * tau + 0.07) * up, "new\nstore", k.TEAL_DARK, ha="center",
                  va="top" if up > 0 else "bottom", size=12)
    tag.set_linespacing(1.1)

    left.set_xscale("log")
    left.set_xlim(0.2, 430)
    left.set_ylim(-limit - 0.07, limit + 0.07)
    left.set_xticks([1, 3, 10, 30, 100, 300])
    left.set_xticklabels(["1", "3", "10", "30", "100", "300"])
    left.minorticks_off()
    k.style(left, frame="lb", grid="y")
    k.axis_names(left, x="training rows per store", y="the store's effect on log sales")
    k.value(left, 400, 0.025, "the average store", k.MUTED, ha="right", va="bottom", size=12)

    # Right: CRPS by bucket, seen stores, then new stores.
    for method, colour, dashed, _label in LINES:
        colour = getattr(k, colour)
        cells = stats[method]
        main = method == "pooled_embedding"
        for group in (BUCKETS[:4], BUCKETS[4:]):
            keys = [key for key in group if key in cells]
            if not keys:
                continue
            xs = [BUCKET_X[BUCKETS.index(key)] for key in keys]
            ys = [cells[key]["crps"] for key in keys]
            right.plot(xs, ys, color=colour, lw=2.4 if main else 1.8, ls=(0, (5, 3)) if dashed else "-",
                       zorder=3 if main else 2)
            right.errorbar(xs, ys, yerr=[cells[key]["sd"] for key in keys], fmt="none", ecolor=colour,
                           elinewidth=1.2, capsize=0, zorder=2)
            right.scatter(xs, ys, s=36 if main else 26, color=colour, edgecolor=k.CARD, lw=1.0, zorder=4 if main else 3)
    right.axvline((BUCKET_X[3] + BUCKET_X[4]) / 2, color=k.HAIR, lw=1.0, zorder=0.5)
    right.set_xticks(BUCKET_X)
    right.set_xticklabels(BUCKET_LABELS)
    right.set_xlim(BUCKET_X[0] - 0.75, BUCKET_X[-1] + 0.45)
    top = max(cell["crps"] + cell["sd"] for method in stats for cell in stats[method].values())
    right.set_ylim(0.0, top * 1.3)
    k.style(right, frame="lb", grid="y")
    k.axis_names(right, x="training rows per store", y="CRPS of log sales, lower is better")
    first = BUCKET_X[0]
    k.value(right, first + 0.1, ours["1-5"]["crps"] - 0.004, f"{ours['1-5']['crps']:.3f}", k.TEAL, ha="left",
            va="top", size=12.5).set_path_effects(halo)
    k.value(right, first - 0.14, gpb["1-5"]["crps"], f"{gpb['1-5']['crps']:.3f}", k.CLAY, ha="right", va="center",
            size=12.5).set_path_effects(halo)
    if "unseen_context" in ours:
        k.value(right, BUCKET_X[5] - 0.12, ours["unseen_context"]["crps"] - 0.004, f"{ours['unseen_context']['crps']:.3f}",
                k.TEAL, ha="center", va="top", size=12.5).set_path_effects(halo)
        k.value(right, BUCKET_X[4] - 0.14, ours["unseen"]["crps"], f"{ours['unseen']['crps']:.3f}", k.TEAL, ha="right",
                va="center", size=12.5).set_path_effects(halo)
    coverage = {method: stats[method].get("unseen", {}).get("coverage") for method in stats}
    if None not in (coverage["pooled_embedding"], coverage["embedding"], coverage["gpboost"]):
        k.rich(right, 0.02, 0.985, [("A new store's 80% interval holds", k.MUTED, k.F.small)], ha="left", va="top")
        k.rich(right, 0.02, 0.915, [
            (k.pct(coverage["pooled_embedding"]), k.TEAL, k.F.value), (" of its days;  ", k.MUTED, k.F.small),
            (k.pct(coverage["embedding"]), k.DATA_DARK, k.F.value), (" embedding,  ", k.MUTED, k.F.small),
            (k.pct(coverage["gpboost"]), k.CLAY, k.F.value), (" GPBoost", k.MUTED, k.F.small),
        ], ha="left", va="top")

    k.legend_row(fig, [
        ("dot", k.DATA_DARK, "a store's own mean residual"),
        ("line", k.TEAL, "Ashlar, pooled: effect and 80% interval"),
        ("line", k.INK, "the new store's next 30 days"),
    ], line=1, spacing=1.8)
    k.legend_row(fig, [
        ("line", k.CLAY, "GPBoost"),
        ("line", k.SLATE, "linear mixed model"),
        ("dashed", k.DATA_DARK, "ordinary embedding"),
    ], line=0, spacing=1.8)
    k.save(fig, "store-pooling")


def main(argv):
    stages = [arg for arg in argv if "=" not in arg] or ["draw"]
    options = dict(arg.split("=", 1) for arg in argv if "=" in arg)
    for stage in stages:
        if stage == "compute":
            compute(int(options.get("seed", SEED)), options.get("epochs"))
        elif stage == "draw":
            draw()
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
