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
  and of its test rows from the population.
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
    return {"source": "docs/benchmarks.md#entity-effects (three-split means)", "runs": {"mean": means}}


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


def draw():
    raise SystemExit("the draw stage arrives with the style kit, examples/readme/style.py")


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
