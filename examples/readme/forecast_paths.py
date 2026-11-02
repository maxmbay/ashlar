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


def draw():
    raise SystemExit("the draw stage arrives with the style kit, examples/readme/style.py")


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
