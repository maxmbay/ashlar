"""The sold-out figure: one FreshRetailNet-50K series whose shelf ran out, and what each fit learned of its demand.

``compute`` runs on Max's Mac, where FreshRetailNet-50K is cached in ``/tmp/ashlar-bench`` (about 15 minutes). It
follows seed 0 of ``benchmark_observed_real.py stockouts``: the days that recorded their demand, with a simulated
stock, so a day sold out when its demand reached the stock; days 0 to 59 train, 60 to 69 calibrate, 70 to 89 test.
It fits the ``mixture`` head three ways, on ``observed.censored(sales, sold_out)``, on sales as if they were
demand, and on the true demand (the oracle); LightGBM's quantile regression on sales at each level; and XGBoost's
AFT model on the censored sales, chosen as the benchmark chooses it. It writes to
``examples/readme/artifacts/sold-out/``:

- ``series.json``: candidate series chosen by a fixed rule (``candidates``), each with every one of its days (the
  split, sales, stock, whether it sold out, and the true demand) and each fit's quantiles of demand that day;
- ``densities.json``: for the first three candidates, each fit's density of demand on a grid, on every test day;
- ``check.json``: this seed's test scores of each fit, beside the benchmark's own scores for the same seed;
- ``stockouts.json``: a copy of the benchmark's results over its seeds, which the order-cost bars read.

``draw`` runs in the cloud container, which has the fonts, and writes ``docs/images/readme/sold-out.png``.

    JAX_PLATFORMS=cpu python examples/readme/sold_out.py compute
    python examples/readme/sold_out.py draw
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

ARTIFACTS = HERE / "artifacts" / "sold-out"
SEED = 0
LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)
CANDIDATES = 8
DENSITY_CANDIDATES = 3
GRID_POINTS = 81
FITS = ("censored", "sales", "oracle", "lightgbm", "xgboost aft")


def _number(value) -> float | None:
    """Four significant digits keep the artifacts small; ``None`` for a value that is not finite."""
    value = float(value)
    return float(f"{value:.4g}") if np.isfinite(value) else None


def _numbers(values) -> list:
    return [_number(value) for value in np.ravel(values)]


def _write(name: str, payload) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / name
    path.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} KB)", flush=True)


def candidates(series, day, demand, sold_out, test_from: int) -> list[str]:
    """Series that read well in one panel, by a rule that looks at the data and not at any fit.

    A candidate has at least 2 sold-out test days and a mean demand in the upper half of the sampled series, so its
    bars are not a few small values. Candidates are ordered by their test days recorded, most first, then by their
    sold-out test days, then by their days recorded of the 90, then by the series key. If fewer than eight series
    qualify, the level is dropped, then the sold-out days.
    """
    keys, index = np.unique(series, return_inverse=True)
    recorded = np.bincount(index, minlength=len(keys))
    test = day >= test_from
    test_days = np.bincount(index, weights=test, minlength=len(keys))
    sold_test = np.bincount(index, weights=test & sold_out, minlength=len(keys))
    level = np.bincount(index, weights=demand, minlength=len(keys)) / recorded
    keep = (sold_test >= 2) & (level >= np.median(level))
    if keep.sum() < CANDIDATES:
        keep = sold_test >= 1
    if keep.sum() < CANDIDATES:
        keep = np.ones(len(keys), bool)
    order = np.lexsort((keys, -recorded, -sold_test, -test_days))
    return [str(keys[position]) for position in order if keep[position]][:CANDIDATES]


def compute() -> None:
    import jax.numpy as jnp
    import lightgbm
    import pandas as pd

    import benchmark_observed_real as real
    from ashlar import ASRRegressor, observed

    cache = real.DATA_DIR / "freshretailnet-50k-train.parquet"
    if not cache.exists():
        raise SystemExit(f"{cache} is missing: FreshRetailNet-50K is cached on the Mac only, and this stage does not download it.")
    frame = real.fresh_frame()
    start = pd.to_datetime(frame["dt"]).min()
    features, day, demand, sales, sold_out, stock = real.fresh_days(frame, SEED, stock=True)
    series = (features["store_id"].astype(str) + "/" + features["product_id"].astype(str)).to_numpy()
    test_from = real.TRAIN_DAYS + real.CALIBRATION_DAYS
    train = np.flatnonzero(day < real.TRAIN_DAYS)
    calibration = np.flatnonzero((day >= real.TRAIN_DAYS) & (day < test_from))
    test = np.flatnonzero(day >= test_from)
    x_test, y_test = features.iloc[test], demand[test]
    print(
        f"seed {SEED}: {len(train)} training, {len(calibration)} calibration and {len(test)} test days, "
        f"{sold_out.mean():.1%} sold out",
        flush=True,
    )
    chosen = candidates(series, day, demand, sold_out, test_from)
    if not chosen:
        raise SystemExit("no series meets the candidate rule")
    rows = {key: np.flatnonzero(series == key) for key in chosen}
    rows = {key: index[np.argsort(day[index])] for key, index in rows.items()}
    density_rows = {key: index[day[index] >= test_from] for key, index in list(rows.items())[:DENSITY_CANDIDATES]}
    quantiles: dict[str, dict[str, np.ndarray]] = {fit: {} for fit in FITS}
    densities: dict[str, dict[str, np.ndarray]] = {fit: {} for fit in FITS if fit != "lightgbm"}
    grids = {
        key: np.linspace(0.0, 1.35 * float(max(demand[series == key].max(), stock[series == key].max())), GRID_POINTS)
        for key in density_rows
    }
    scores = {}
    seconds = {}

    for fit, label in (
        ("censored", observed.censored(sales[train], sold_out[train])),
        ("sales", sales[train]),
        ("oracle", demand[train]),
    ):
        with real._Clock() as clock:
            model = ASRRegressor(task="mixture", random_state=SEED).fit(features.iloc[train], label)
        seconds[fit] = {"cpu": clock.cpu, "wall": clock.wall}
        for key, index in rows.items():
            quantiles[fit][key] = model.predict_quantiles(features.iloc[index], list(LEVELS))
        for key, index in density_rows.items():
            dist = model.predict_dist(features.iloc[index])
            densities[fit][key] = np.stack(
                [np.exp(np.asarray(dist.log_prob(jnp.full((len(index),), value, jnp.float32)), np.float64)) for value in grids[key]],
                axis=1,
            )
        q90, q75 = model.predict_quantiles(x_test, [0.9, real.RATIO]).T
        half_bin = 0.005 * demand.mean()
        scores[fit] = {
            "nll": float(-np.mean(model.log_prob(x_test, observed.interval(y_test - half_bin, y_test + half_bin)))),
            "q90": float(np.mean(y_test <= q90)),
            "cost_value": real._newsvendor(q75, y_test),
        }
        print(f"  mixture, {fit}: {scores[fit]}, {seconds[fit]['wall']:.0f} s", flush=True)

    with real._Clock() as clock:
        boosters = {}
        for level in LEVELS:
            boosters[level] = lightgbm.LGBMRegressor(
                objective="quantile", alpha=level, n_estimators=400, learning_rate=0.05, verbose=-1, random_state=SEED
            ).fit(features.iloc[train], sales[train])
    seconds["lightgbm"] = {"cpu": clock.cpu, "wall": clock.wall}
    for key, index in rows.items():
        quantiles["lightgbm"][key] = np.stack([boosters[level].predict(features.iloc[index]) for level in LEVELS], axis=1)
    scores["lightgbm"] = {
        "q90": float(np.mean(y_test <= boosters[0.9].predict(x_test))),
        "cost_value": real._newsvendor(boosters[real.RATIO].predict(x_test), y_test),
    }
    print(f"  lightgbm quantile, sales: {scores['lightgbm']}", flush=True)

    fitted = real.stockouts_aft(features, sales, sold_out, train, calibration, SEED)
    seconds["xgboost aft"] = {"cpu": fitted["cpu"], "wall": fitted["wall"]}
    error = real.AFT_FAMILIES[fitted["family"]]
    for key, index in rows.items():
        loc = real.stockouts_aft_predict(fitted, features.iloc[index])
        quantiles["xgboost aft"][key] = real.stockouts_aft_quantiles(fitted, loc, LEVELS)
    for key, index in density_rows.items():
        loc = real.stockouts_aft_predict(fitted, features.iloc[index])
        shifted = grids[key][None, :] + fitted["shift"]
        z = (np.log(shifted) - loc[:, None]) / fitted["scale"]
        densities["xgboost aft"][key] = error.pdf(z) / (fitted["scale"] * shifted)
    record = real._stockouts_aft_record(fitted, x_test, y_test, 0.005 * demand.mean(), scores["oracle"]["cost_value"], SEED)
    scores["xgboost aft"] = {name: record[name] for name in ("nll", "q90", "cost_value", "chosen", "validation_nll")}
    print(f"  xgboost aft, censored: {scores['xgboost aft']}", flush=True)

    for fit in scores:
        scores[fit]["cost"] = float(100.0 * (scores[fit]["cost_value"] / scores["oracle"]["cost_value"] - 1.0))

    benchmark = real._load("stockouts")
    names = {
        "censored": "mixture/censored",
        "sales": "mixture/sales as demand",
        "oracle": f"mixture/{real.STOCK_FITS[0]}",
        "lightgbm": "lightgbm quantile/sales as demand",
        "xgboost aft": "xgboost aft/censored",
    }
    check = {
        fit: {"this run": scores[fit], "benchmark": benchmark.get(f"{SEED}/{name}"), "seconds": seconds[fit]}
        for fit, name in names.items()
    }
    _write("check.json", check)

    payload = {
        "seed": SEED,
        "start_date": str(start.date()),
        "train_days": real.TRAIN_DAYS,
        "calibration_days": real.CALIBRATION_DAYS,
        "levels": list(LEVELS),
        "rule": candidates.__doc__.split("\n\n", 1)[1].strip(),
        "fits": {
            "censored": "mixture on observed.censored(sales, sold_out)",
            "sales": "mixture on sales as if they were demand",
            "oracle": "mixture on the true demand",
            "lightgbm": "LightGBM quantile regression on sales, 400 trees, learning rate 0.05, one model per level",
            "xgboost aft": f"XGBoost AFT on the censored sales; grid chose {scores['xgboost aft']['chosen']}",
        },
        "series": [],
    }
    for key, index in rows.items():
        first = features.iloc[index[0]]
        payload["series"].append(
            {
                "key": key,
                "store_id": str(first["store_id"]),
                "product_id": str(first["product_id"]),
                "city_id": str(first["city_id"]),
                "categories": [str(first[name]) for name in ("first_category_id", "second_category_id", "third_category_id")],
                "day": [int(value) for value in day[index]],
                "sales": _numbers(sales[index]),
                "stock": _numbers(stock[index]),
                "sold_out": [bool(value) for value in sold_out[index]],
                "demand": _numbers(demand[index]),
                "discount": _numbers(features["discount"].to_numpy()[index]),
                "quantiles": {fit: [_numbers(row) for row in quantiles[fit][key]] for fit in FITS},
            }
        )
    _write("series.json", payload)
    _write(
        "densities.json",
        {
            key: {
                "grid": _numbers(grids[key]),
                "day": [int(value) for value in day[index]],
                "density": {fit: [_numbers(row) for row in densities[fit][key]] for fit in densities},
            }
            for key, index in density_rows.items()
        },
    )
    results = Path(str(real.RESULTS).format(stage="stockouts"))
    if results.exists():
        shutil.copy(results, ARTIFACTS / "stockouts.json")
        print(f"copied {results} to {ARTIFACTS / 'stockouts.json'}", flush=True)
    else:
        print(f"{results} is missing; run benchmark_observed_real.py stockouts for the order-cost bars", flush=True)


def draw() -> None:
    raise SystemExit("draw is written once the style kit, examples/readme/style.py, is on dev")


STAGES = {"compute": compute, "draw": draw}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "draw"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
