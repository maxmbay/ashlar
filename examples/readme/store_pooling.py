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

FIGURE = "store-pooling"
Z80 = 1.2815515655446004
# The right panel's rows, top to bottom, and their labels.
BUCKETS = ("1-5", "6-20", "21-100", "101-300", "unseen", "unseen_context")
BUCKET_LABELS = ("1–5 rows", "6–20", "21–100", "101–300", "no rows", "after 5 rows")
WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}
# Methods, back to front: (method, colour, legend label).
METHODS = (
    ("embedding", "DATA_DARK", "ordinary embedding"),
    ("mixedlm", "SLATE", "linear mixed model"),
    ("gpboost", "CLAY", "GPBoost"),
    ("pooled_embedding", "TEAL", "Ashlar, pooled"),
)
NONE_X, GAP_X = 0.42, 0.66  # the left panel's "no rows yet", and the break before 1 row


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
    """The new store drawn: the unseen store whose |level| is nearest the 90th percentile among unseen stores,
    a fixed rule for a store that differs from the average one."""
    size = np.abs(unseen["level"].astype(np.float64))
    return int(np.argmin(np.abs(size - np.quantile(size, 0.9))))


def raw_path(unseen: dict, pick: int) -> np.ndarray:
    """The new store's own running mean residual over its first k rows, its context rows then its test rows,
    in the order the compute stage read them."""
    first = (unseen["context_y"] - unseen["context_population"])[unseen["context_store"] == pick]
    then = (unseen["row_y"] - unseen["row_population"])[unseen["row_store"] == pick]
    residual = np.concatenate([first, then]).astype(np.float64)
    return np.cumsum(residual) / np.arange(1, len(residual) + 1)


def _legend(k, fig, items, line: int = 0):
    """The kit's legend row with one more key, ``ring``: a hollow dot, for the ordinary embedding."""
    from matplotlib.lines import Line2D

    handles = []
    for kind, colour, label in items:
        face = k.CARD if kind == "ring" else colour
        edge = colour if kind == "ring" else k.CARD
        handles.append(Line2D([], [], color=colour, marker="o", lw=0, ms=7.5, markerfacecolor=face,
                              markeredgecolor=edge, markeredgewidth=1.6 if kind == "ring" else 1.0, label=label))
    box = fig.box
    return fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(box.left / box.width, (0.3 + 0.32 * line) /
                      box.height), ncol=len(handles), frameon=False, prop=k.F.label, handlelength=1.0,
                      columnspacing=1.6, labelcolor=k.INK, borderaxespad=0, handletextpad=0.5)


def draw():
    import matplotlib.patheffects as path_effects
    from matplotlib.collections import LineCollection

    import style as k

    stores = k.load_json(FIGURE, "stores")
    unseen = k.load_npz(FIGURE, "unseen")
    buckets = k.load_json(FIGURE, "buckets")
    stats = {method: bucket_stats(buckets, method) for method, *_ in METHODS}
    ours, gpb, emb, mix = (stats[name] for name in ("pooled_embedding", "gpboost", "embedding", "mixedlm"))
    splits = buckets.get("splits") or max(cell["splits"] for cell in ours.values())
    tau = float(stores["tau"])
    rows = np.asarray(stores["rows"], dtype=np.float64)
    pooled = np.asarray(stores["pooled"], dtype=np.float64)
    sd = np.asarray(stores["offset_sd"], dtype=np.float64)
    halo = [path_effects.withStroke(linewidth=4.5, foreground=k.CARD)]
    splits_text = WORDS.get(splits, str(splits))

    k.setup()
    fig = k.card(
        f"Small stores borrow from the rest: CRPS {ours['1-5']['crps']:.3f} against GPBoost's "
        f"{gpb['1-5']['crps']:.3f}.",
        f"Rossmann, {buckets.get('stores', 1115):,} stores: log daily sales given the store's type and assortment, "
        "the date, promotions, holidays and competition, with a store effect pooled towards the average store. "
        "A fifth of the stores held out whole. Left: one split. Right: CRPS by rows per store, "
        f"mean of {splits_text} splits.",
        height=9.4, footer=3.35,
    )
    left = k.panel(fig, rect=(0.0, 0.0, 0.575, 1.0), head=0.95, tail=0.8, lpad=0.72, rpad=0.12, grid="y")
    right = k.panel(fig, rect=(0.635, 0.0, 1.0, 1.0), head=0.95, tail=0.8, lpad=1.32, rpad=0.08, grid="x")

    # Left, behind: every seen store's pooled effect with its 80% interval.
    rng = np.random.default_rng(0)
    x = rows * np.exp(rng.uniform(-0.09, 0.09, len(rows)))
    pick = highlighted(unseen)
    steps = np.asarray(unseen["path_steps"], dtype=np.float64)
    mean = np.asarray(unseen["path_mean"][pick], dtype=np.float64)
    spread = Z80 * np.asarray(unseen["path_sd"][pick], dtype=np.float64)
    level = float(unseen["level"][pick])
    level_rows = int(unseen["level_rows"][pick])
    own = raw_path(unseen, pick)
    own_steps = own[np.minimum(steps.astype(int), len(own)) - 1]
    reach = max(np.quantile(np.abs(pooled), 0.995), Z80 * tau, np.max(np.abs(mean) + spread), abs(level), 0.2)
    limit = float(np.ceil(reach * 1.08 * 10) / 10)
    left.axhline(0.0, color=k.MUTED, lw=1.0, zorder=0.6)
    left.add_collection(LineCollection(np.stack([np.c_[x, pooled - Z80 * sd], np.c_[x, pooled + Z80 * sd]], axis=1),
                                       colors=k.TEAL_PALE, linewidths=0.9, zorder=1))
    left.scatter(x, np.clip(pooled, -limit * 0.98, limit * 0.98), s=9, color=k.TEAL, alpha=0.8, lw=0, zorder=3)

    # Left, in front: one new store, from the prior (no rows) through with_context on its first 1 to 35 rows.
    left.axvspan(0.2, GAP_X, color=k.CARD, lw=0, zorder=3.5)
    left.axvline(GAP_X, color=k.HAIR, lw=1.0, zorder=3.6)
    path_x = np.r_[NONE_X, steps]
    path_mean, path_spread = np.r_[0.0, mean], np.r_[Z80 * tau, spread]
    left.plot([NONE_X * 0.72, steps[-1] * 1.5], [level, level], color=k.INK, lw=1.6, ls=(0, (4, 2.5)), zorder=4)
    bars = LineCollection(np.stack([np.c_[path_x, path_mean - path_spread], np.c_[path_x, path_mean + path_spread]],
                                   axis=1), colors=k.TEAL_DARK, linewidths=6.5, capstyle="butt", zorder=5)
    bars.set_path_effects([path_effects.withStroke(linewidth=9.5, foreground=k.CARD)])
    left.add_collection(bars)
    left.plot(path_x, path_mean, color=k.TEAL_DARK, lw=1.4, zorder=5.5)
    left.scatter(steps, own_steps, s=40, color=k.DATA_DARK, edgecolor=k.CARD, lw=1.0, zorder=6)
    left.scatter(path_x, path_mean, s=46, color=k.TEAL, edgecolor=k.CARD, lw=1.2, zorder=7)

    up = 1.0 if level >= 0 else -1.0
    k.value(left, NONE_X, -up * (Z80 * tau + 0.035 * limit), "a new\nstore", k.TEAL_DARK, ha="center",
            va="top" if up > 0 else "bottom", size=12.5, zorder=8).set_linespacing(1.05)
    edge = np.max(up * np.r_[mean + up * spread, own_steps])
    k.code(left, 3.2, up * (max(edge, Z80 * tau) + 0.09 * limit), "with_context(X, y)", k.TEAL_DARK, bold=True,
           ha="center", va="bottom" if up > 0 else "top", zorder=8).set_path_effects(halo)
    tag = k.value(left, steps[-1] * 1.6, level, f"its level over\nall {level_rows:,} days", k.INK, ha="left",
                  va="center", size=12, zorder=8)
    tag.set_linespacing(1.1)
    tag.set_path_effects(halo)

    left.set_xscale("log")
    left.set_xlim(0.3, 480)
    left.set_ylim(-limit, limit)
    step = 0.5 if limit > 0.8 else 0.25
    marks = np.arange(-np.floor(limit / step), np.floor(limit / step) + 1) * step
    left.set_yticks(marks)
    left.set_yticklabels(["0" if m == 0 else f"{m:+g}".replace("-", "\u2212") for m in marks])
    ticks = [1, 3, 10, 30, 100, 300]
    left.set_xticks([NONE_X, *ticks])
    left.set_xticklabels(["0", *map(str, ticks)])
    left.minorticks_off()
    k.style(left, frame="lb", grid="y")
    k.axis_names(left, x="rows of the store the model has read", y="the store's effect on log sales")
    k.panel_title(left, "Every seen store's effect; one new store, row by row", raise_by=0.42)

    # Right: CRPS by rows per store, a dot per method, seen stores above, new stores below.
    ys = np.r_[np.arange(4), np.arange(4, 6) + 0.55]
    ys = ys.max() - ys
    for row, bucket in enumerate(BUCKETS):
        values = [stats[method][bucket]["crps"] for method, *_ in METHODS if bucket in stats[method]]
        right.plot([min(values), max(values)], [ys[row]] * 2, color=k.HAIR, lw=2.0, zorder=1, solid_capstyle="butt")
    for method, colour, _label in METHODS:
        colour = getattr(k, colour)
        main = method == "pooled_embedding"
        for row, bucket in enumerate(BUCKETS):
            if bucket not in stats[method]:
                continue
            cell = stats[method][bucket]
            if cell["sd"] > 0:
                right.plot([cell["crps"] - cell["sd"], cell["crps"] + cell["sd"]], [ys[row]] * 2, color=colour,
                           lw=1.4, zorder=2)
            if method == "embedding":
                right.scatter([cell["crps"]], [ys[row]], s=62, facecolor=k.CARD, edgecolor=colour, lw=1.8, zorder=3)
            else:
                right.scatter([cell["crps"]], [ys[row]], s=95 if main else 70, color=colour, edgecolor=k.CARD,
                              lw=1.3, zorder=4 if main else 3)
    for bucket in ("1-5", "unseen_context"):
        row = BUCKETS.index(bucket)
        k.value(right, ours[bucket]["crps"] - 0.011, ys[row], f"{ours[bucket]['crps']:.3f}", k.TEAL,
                ha="right", va="center", size=12.5).set_path_effects(halo)
        k.value(right, gpb[bucket]["crps"], ys[row] + 0.26, f"{gpb[bucket]['crps']:.3f}", k.CLAY,
                ha="center", va="bottom", size=12.5).set_path_effects(halo)
    right.set_yticks(ys)
    right.set_yticklabels(BUCKET_LABELS)
    for label in right.get_yticklabels():
        label.set_color(k.INK)
    right.axhline((ys[3] + ys[4]) / 2, color=k.HAIR, lw=1.0, zorder=0.5)
    top = max(cell["crps"] + cell["sd"] for method in stats for cell in stats[method].values())
    right.set_xlim(0.0, top * 1.08)
    right.set_ylim(ys.min() - 0.6, ys.max() + 1.05)
    k.style(right, frame="b", grid="x")
    k.axis_names(right, x="CRPS of log sales, lower is better")
    k.panel_title(right, "Scores by rows per store", raise_by=0.42)
    for row, text, lift in ((0, "stores seen in training", 0.8), (4, "new stores, held out whole", 0.48)):
        right.text(0.002, ys[row] + lift, text, fontproperties=k.F.small, color=k.MUTED, ha="left", va="center")

    coverage = {method: stats[method].get("unseen", {}).get("coverage") for method in stats}
    later = {method: stats[method].get("unseen_context", {}).get("coverage") for method in stats}
    if None not in coverage.values():
        new = ("unseen", "unseen_context")
        worse = [stats[m][b]["crps"] / ours[b]["crps"] - 1 for m in ("gpboost", "mixedlm") for b in new]
        x0, label, number = fig.box.left / fig.box.width, k.F.label, k.F.value
        first = [
            ("A new store's 80% interval holds ", k.INK, label), (k.pct(coverage["pooled_embedding"]), k.TEAL, number),
            (" of its days, against ", k.INK, label), (k.pct(coverage["embedding"]), k.DATA_DARK, number),
            (" for an ordinary embedding", k.INK, label),
        ]
        second = [("GPBoost's and the mixed model's hold nearer 80%, ", k.MUTED, label),
                  (k.pct(coverage["gpboost"]), k.CLAY, number), (" and ", k.MUTED, label),
                  (k.pct(coverage["mixedlm"]), k.SLATE, number)]
        if None not in (later["pooled_embedding"], later["gpboost"], later["mixedlm"]):
            first += [("; after five of its rows, ", k.INK, label), (k.pct(later["pooled_embedding"]), k.TEAL, number)]
            second += [(", then ", k.MUTED, label), (k.pct(later["gpboost"]), k.CLAY, number),
                       (" and ", k.MUTED, label), (k.pct(later["mixedlm"]), k.SLATE, number)]
        first += [(".", k.INK, label)]
        second += [(f", at a CRPS {k.pct(min(worse))} to {k.pct(max(worse))} higher.", k.MUTED, label)]
        k.rich(fig, x0, (0.34 + 0.45 + 0.34) / fig.box.height, first, va="bottom")
        k.rich(fig, x0, (0.34 + 0.45) / fig.box.height, second, va="bottom")
    _legend(k, fig, [
        ("dot", k.DATA_DARK, "the new store's own mean so far"),
        ("dot", k.TEAL, "Ashlar, pooled, with its 80% interval"),
        ("dot", k.CLAY, "GPBoost"),
        ("dot", k.SLATE, "linear mixed model"),
        ("ring", k.DATA_DARK, "ordinary embedding"),
    ])
    k.save(fig, FIGURE)

    def mark(ax):
        ax.add_collection(LineCollection(np.stack([np.c_[x, pooled - Z80 * sd], np.c_[x, pooled + Z80 * sd]], axis=1),
                                         colors=k.TEAL_WASH, linewidths=0.8, zorder=1))
        ax.scatter(x, pooled, s=4, color=k.TEAL_PALE, lw=0, zorder=2)
        ax.axvspan(0.2, GAP_X, color=k.CARD, lw=0, zorder=3)
        ax.axhline(0.0, color=k.HAIR, lw=1.0, zorder=0.5)
        ax.plot([NONE_X * 0.72, steps[-1] * 1.5], [level, level], color=k.INK, lw=1.8, ls=(0, (4, 2.5)), zorder=4)
        bars = LineCollection(np.stack([np.c_[path_x, path_mean - path_spread], np.c_[path_x, path_mean + path_spread]],
                                       axis=1), colors=k.TEAL_DARK, linewidths=7, capstyle="butt", zorder=5)
        bars.set_path_effects([path_effects.withStroke(linewidth=10, foreground=k.CARD)])
        ax.add_collection(bars)
        ax.plot(path_x, path_mean, color=k.TEAL_DARK, lw=1.6, zorder=5.5)
        ax.scatter(path_x, path_mean, s=40, color=k.TEAL, edgecolor=k.CARD, lw=1.2, zorder=7)
        ax.set_xscale("log")
        ax.set_xlim(0.3, 480)
        ax.set_ylim(-limit * 0.85, limit * 0.85)
        ax.minorticks_off()
        ax.set_xticks([])
        ax.set_yticks([])

    k.thumb("pooling", "Groups borrow strength", [
        (f"{ours['1-5']['crps']:.3f}", k.TEAL), (" CRPS at 1–5 rows, against ", k.MUTED),
        (f"{gpb['1-5']['crps']:.3f}", k.CLAY),
    ], mark)


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
