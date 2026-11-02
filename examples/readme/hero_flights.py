"""The hero figure and the one-flight card: every flight gets its own distribution.

Airline delay (OpenML 42728): the departure delay in minutes of a fixed
sample of 2,000,000 US flights, from the schedule alone (month, weekday,
scheduled departure and arrival, carrier, origin, destination, distance; no
weather). Every fit uses the large-table benchmark's seed-0 protocol
(``examples/benchmark_large.py``): the same 80/20 test split, the same 20%
early-stopping slice of the training rows, and the same 100,000 scored test
rows.

``compute`` fits three models as the benchmark does and saves them outside
the repository (``$ASHLAR_BENCH_CACHE/readme/hero-flights/``), so a rerun
loads them instead of refitting:

- Ashlar, ``ASRRegressor(task="mixture")`` with no other arguments.
- CatBoost ``RMSEWithUncertainty``, a normal with a mean and a variance per
  flight: the best tree distribution on this table.
- LightGBM, early-stopped, with one normal whose scale is its residual
  spread on the early-stopping rows.

It writes small artifacts under ``examples/readme/artifacts/hero-flights/``:

- ``summary.json``: each model's scores on the 100,000 rows under the
  benchmark's protocol, the calibrated interval's, P(delay > 60 min) by bins
  of the predicted probability, fit times, and the table's facts.
- ``pool.npz``: 1,500 of the scored flights drawn at random (seed 0) and
  twelve card candidates, each with its schedule, its actual delay, Ashlar's
  mixture (five weights, means, and scales, in minutes), its quantiles, mean,
  and P(delay > 60), CatBoost's mean and scale, and LightGBM's point.
  ``draw`` picks the ridges from it by a fixed rule.
- ``card.json``: the twelve card candidates, seeded picks among evening
  departures (17:00 to 21:59) whose P(delay > 60) is between the 85th and
  99th percentile, with the value of every call the quick start shows:
  ``predict``, ``predict_quantiles``, ``predict_exceedance``, ``log_prob``,
  ``sample``, ``explain`` for the 0.9 quantile, and, after ``calibrate`` on
  20,000 other held-out flights, ``predict_interval``.
- ``hours.npz``: per scheduled departure hour, the average predicted density
  of each model on a one-minute grid and the held-out delays' counts.
- ``sweep.npz``: each card candidate's schedule with only the departure hour
  moved, 5:00 to 23:00, and both models' predictions for it.

``draw`` reads them and writes ``docs/images/readme/hero-flights.png``;
``examples/readme/one_flight.py`` draws the card from the same artifacts.

    python examples/readme/hero_flights.py compute   # about 40 minutes on a laptop CPU
    python examples/readme/hero_flights.py draw
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples"))

import benchmark_large as bench  # noqa: E402

NAME = "hero-flights"
ARTIFACTS = ROOT / "examples" / "readme" / "artifacts" / NAME
TABLE = "airlines_delay"
SEED = 0
THRESHOLD = 60.0
POOL = 1500
CARDS = 12
CALIBRATION_ROWS = 20_000
EXPLAIN_BACKGROUND = 5_000
GRID = np.arange(-60.0, 301.0, 1.0)
SWEEP_HOURS = np.arange(5, 24)
P60_BINS = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
COMPONENTS = 5


# ---------------------------------------------------------------- helpers


def _column(frame, name):
    """The column called ``name``, ignoring case, or ``None``."""
    for column in frame.columns:
        if column.lower() == name.lower():
            return column
    return None


def _clock_format(values) -> str:
    """``hhmm`` (1805 is 18:05) or ``hours`` (18.08 is 18:05), read from the column's range."""
    return "hhmm" if float(np.nanpercentile(values, 99)) > 30 else "hours"


def _minutes(values, fmt):
    values = np.asarray(values, dtype=np.float64)
    if fmt == "hhmm":
        return np.floor(values / 100.0) * 60.0 + np.mod(values, 100.0)
    return values * 60.0


def _clock(minutes, fmt):
    minutes = np.mod(np.asarray(minutes, dtype=np.float64), 24 * 60)
    if fmt == "hhmm":
        return np.floor(minutes / 60.0) * 100.0 + np.mod(minutes, 60.0)
    return minutes / 60.0


def mixture_parts(model, X):
    """Each row's predictive mixture in minutes: ``(weights, means, scales)``, each ``(rows, members * 5)``."""
    import jax.numpy as jnp

    from ashlar.distributional import decode

    raw = np.asarray(model.predict_raw(X))
    rows, members, width = raw.shape
    loc, scale, weight = decode("mixture", jnp.asarray(raw.reshape(rows * members, width)), 1)
    mean = float(np.asarray(model.target_mean_, dtype=np.float64).reshape(-1)[0])
    std = float(np.asarray(model.target_std_, dtype=np.float64).reshape(-1)[0])
    loc = np.asarray(loc, dtype=np.float64).reshape(rows, members * COMPONENTS) * std + mean
    scale = np.asarray(scale, dtype=np.float64).reshape(rows, members * COMPONENTS) * std
    weight = np.asarray(weight, dtype=np.float64).reshape(rows, members * COMPONENTS) / members
    return weight, loc, scale


def mixture_density(weight, loc, scale, grid):
    """Density of each row's mixture at each grid value, ``(rows, len(grid))``."""
    z = (np.asarray(grid)[None, :, None] - loc[:, None, :]) / scale[:, None, :]
    return np.sum(weight[:, None, :] * np.exp(-0.5 * z * z) / (scale[:, None, :] * math.sqrt(2 * math.pi)), axis=2)


def mixture_cdf(weight, loc, scale, value):
    from scipy.stats import norm

    value = np.asarray(value, dtype=np.float64).reshape(-1, 1)
    return np.sum(weight * norm.cdf((value - loc) / scale), axis=1)


def normal_density(loc, scale, grid):
    z = (np.asarray(grid)[None, :] - loc[:, None]) / scale[:, None]
    return np.exp(-0.5 * z * z) / (scale[:, None] * math.sqrt(2 * math.pi))


def _p60_bins(predicted, observed):
    out = []
    edges = np.asarray(P60_BINS)
    which = np.clip(np.searchsorted(edges, predicted, side="right") - 1, 0, len(edges) - 2)
    for k in range(len(edges) - 1):
        inside = which == k
        out.append({
            "low": float(edges[k]), "high": float(edges[k + 1]), "flights": int(inside.sum()),
            "predicted": float(predicted[inside].mean()) if inside.any() else None,
            "observed": float(observed[inside].mean()) if inside.any() else None,
        })
    return out


def _scores(point, yd, *, draws=None, loc=None, scale=None, log_prob=None, p60=None, observed60=None):
    """The benchmark's distributional scores on the scored rows, and the Brier score of P(delay > 60)."""
    record = {}
    if draws is not None:
        record["crps"] = bench.crps_samples(draws, yd)
        record["nll"] = float(-np.mean(log_prob))
        record.update(bench.interval(np.quantile(draws, bench.LEVELS, axis=1).T, yd))
    else:
        record["crps"] = bench.crps_normal(loc, scale, yd)
        record["nll"] = float(np.mean(0.5 * np.log(2 * np.pi * scale**2) + 0.5 * ((yd - loc) / scale) ** 2))
        record.update(bench.interval(loc[:, None] + scale[:, None] * np.array([-bench.Z80, 0.0, bench.Z80]), yd))
    record["brier60"] = float(np.mean((p60 - observed60) ** 2))
    record["p60_bins"] = _p60_bins(p60, observed60)
    return record


def _features(frame, names):
    """The schedule columns as arrays: numbers as float32, text as fixed-width strings."""
    out = {}
    for name in names:
        values = frame[name].to_numpy()
        if np.issubdtype(np.asarray(values).dtype, np.number):
            out[f"x_{name}"] = np.asarray(values, dtype=np.float32)
        else:
            out[f"x_{name}"] = np.asarray([("" if v is None else str(v)) for v in values], dtype="U8")
    return out


# ---------------------------------------------------------------- compute


def fit_ashlar(x_train, y_train, path, epochs=None):
    from ashlar import ASRRegressor

    if (path / "metadata.json").exists():
        model = ASRRegressor.load(path)
        print(f"loaded {path}", flush=True)
        return model, None
    start = time.time()
    extra = {} if epochs is None else {"max_epochs": int(epochs)}
    model = ASRRegressor(task="mixture", random_state=SEED, **extra).fit(x_train, y_train)
    seconds = time.time() - start
    model.save(path)
    print(f"ashlar mixture fitted in {seconds:.0f} s, saved to {path}", flush=True)
    return model, seconds


def compute(out: Path, models: Path, epochs=None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    task, x_train, x_test, y_train, y_test = bench.splits(TABLE, SEED)
    print(f"{TABLE}: {len(y_train):,} training and {len(y_test):,} test flights; columns {list(x_train.columns)}", flush=True)
    names = list(x_train.columns)
    dep, arr = _column(x_train, "CRSDepTime"), _column(x_train, "CRSArrTime")
    if dep is None:
        raise SystemExit(f"no CRSDepTime column in {names}")
    fmt = _clock_format(x_train[dep].to_numpy(np.float64))
    print(f"scheduled times are {fmt}", flush=True)

    # The three fits, saved so a rerun only loads them.
    ashlar, ashlar_seconds = fit_ashlar(x_train, y_train, models / "ashlar-mixture-seed0", epochs)
    start = time.time()
    cat_path = models / "catboost-uncertainty-seed0.cbm"
    cat_fitted = not cat_path.exists()
    catboost, cat_predict = bench.fit_catboost_uncertainty(x_train, y_train, SEED, path=cat_path)
    cat_seconds = time.time() - start if cat_fitted else None
    start = time.time()
    lgbm, lgbm_frame, (val_x, val_y) = bench.fit_tree("lightgbm", task, x_train, y_train, SEED)
    lgbm_seconds = time.time() - start
    sigma = float(np.std(val_y - lgbm.predict(val_x)))
    print(f"catboost {cat_seconds} s, lightgbm {lgbm_seconds:.0f} s, lightgbm residual sd {sigma:.2f}", flush=True)

    # The benchmark's 100,000 scored test rows.
    rows = bench.dist_rows(len(y_test), SEED)
    xd, yd = x_test.iloc[rows].reset_index(drop=True), y_test[rows]
    late = (yd > THRESHOLD).astype(np.float64)
    weight, loc, scale = mixture_parts(ashlar, xd)
    probe = np.arange(min(256, len(yd)))
    for value in (-10.0, 0.0, 25.0, 90.0):
        exact = np.asarray(ashlar.predict_dist(xd.iloc[probe]).log_prob(np.full(len(probe), value, np.float32)))
        ours = np.log(mixture_density(weight[probe], loc[probe], scale[probe], [value])[:, 0])
        gap = float(np.max(np.abs(exact - ours)))
        if gap > 2e-3:
            raise SystemExit(f"the exported mixture does not reproduce predict_dist at {value}: {gap:.2e}")
    quantiles = ashlar.predict_quantiles(xd, (0.1, 0.5, 0.9), calibrated=False)
    p60 = ashlar.predict_exceedance(xd, THRESHOLD, calibrated=False)
    gap = float(np.max(np.abs(p60 - (1.0 - mixture_cdf(weight, loc, scale, np.full(len(yd), THRESHOLD))))))
    if gap > 2e-3:
        raise SystemExit(f"the exported mixture does not reproduce predict_exceedance: {gap:.2e}")
    mean = ashlar.predict(xd)
    draws = ashlar.sample(xd, n=bench.N_DRAWS, seed=SEED).T
    log_prob = ashlar.log_prob(xd, yd)
    cat_loc, cat_scale = cat_predict(xd)
    from scipy.stats import norm

    cat_p60 = norm.sf((THRESHOLD - cat_loc) / cat_scale)
    lgbm_point = bench.tree_predict(lgbm, lgbm_frame, xd)
    lgbm_p60 = norm.sf((THRESHOLD - lgbm_point) / sigma)

    scores = {
        "ashlar": _scores(mean, yd, draws=draws, log_prob=log_prob, p60=p60, observed60=late),
        "catboost_uncertainty": _scores(cat_loc, yd, loc=cat_loc, scale=cat_scale, p60=cat_p60, observed60=late),
        "lightgbm_normal": _scores(lgbm_point, yd, loc=lgbm_point, scale=np.full(len(yd), sigma), p60=lgbm_p60,
                                   observed60=late),
    }
    exact = bench.interval(quantiles, yd)
    scores["ashlar"]["exact_quantiles"] = {key: exact[key] for key in ("coverage80", "width80")}
    for key, model_point in (("ashlar", ashlar.predict(x_test)), ("catboost_uncertainty", cat_predict(x_test)[0]),
                             ("lightgbm_normal", bench.tree_predict(lgbm, lgbm_frame, x_test))):
        scores[key].update(bench.point_scores(model_point, y_test))
    for key, record in scores.items():
        print(key, {k: round(v, 4) for k, v in record.items() if isinstance(v, float)}, flush=True)

    # Per scheduled hour: average densities and the held-out delays.
    dep_minutes = _minutes(xd[dep].to_numpy(np.float64), fmt)
    hour = np.floor(dep_minutes / 60.0).astype(int) % 24
    ashlar_hour = np.zeros((24, len(GRID)))
    cat_hour = np.zeros((24, len(GRID)))
    lgbm_hour = np.zeros((24, len(GRID)))
    for start in range(0, len(yd), 4096):
        part = slice(start, start + 4096)
        density = mixture_density(weight[part], loc[part], scale[part], GRID)
        np.add.at(ashlar_hour, hour[part], density)
        np.add.at(cat_hour, hour[part], normal_density(cat_loc[part], cat_scale[part], GRID))
        np.add.at(lgbm_hour, hour[part], normal_density(lgbm_point[part], np.full(len(lgbm_point[part]), sigma), GRID))
    flights = np.bincount(hour, minlength=24)
    per = np.maximum(flights, 1)[:, None]
    edges = np.append(GRID - 0.5, GRID[-1] + 0.5)
    counts = np.stack([np.histogram(yd[hour == h], bins=edges)[0] for h in range(24)])
    np.savez_compressed(
        out / "hours.npz",
        grid=GRID.astype(np.float32),
        flights=flights,
        ashlar=(ashlar_hour / per).astype(np.float32),
        catboost_uncertainty=(cat_hour / per).astype(np.float32),
        lightgbm_normal=(lgbm_hour / per).astype(np.float32),
        counts=counts.astype(np.int32),
        late_observed=np.asarray([late[hour == h].mean() if (hour == h).any() else np.nan for h in range(24)]),
        late_ashlar=np.asarray([p60[hour == h].mean() if (hour == h).any() else np.nan for h in range(24)]),
        late_catboost=np.asarray([cat_p60[hour == h].mean() if (hour == h).any() else np.nan for h in range(24)]),
    )

    # The pool the hero's ridges are picked from, and the card candidates.
    rng = np.random.default_rng(SEED)
    pool = np.sort(rng.choice(len(yd), size=min(POOL, len(yd)), replace=False))
    evening = (dep_minutes >= 17 * 60) & (dep_minutes < 22 * 60)
    low, high = np.quantile(p60, [0.85, 0.99])
    eligible = np.flatnonzero(evening & (p60 >= low) & (p60 <= high))
    cards = np.random.default_rng(SEED + 1).choice(eligible, size=min(CARDS, len(eligible)), replace=False)
    picked = np.concatenate([pool, cards])
    np.savez_compressed(
        out / "pool.npz",
        row=rows[picked].astype(np.int64),
        is_card=np.concatenate([np.zeros(len(pool), bool), np.ones(len(cards), bool)]),
        actual=yd[picked].astype(np.float32),
        weight=weight[picked].astype(np.float32),
        loc=loc[picked].astype(np.float32),
        scale=scale[picked].astype(np.float32),
        q10=quantiles[picked, 0].astype(np.float32),
        q50=quantiles[picked, 1].astype(np.float32),
        q90=quantiles[picked, 2].astype(np.float32),
        mean=mean[picked].astype(np.float32),
        p60=p60[picked].astype(np.float32),
        cat_loc=cat_loc[picked].astype(np.float32),
        cat_scale=cat_scale[picked].astype(np.float32),
        cat_p60=cat_p60[picked].astype(np.float32),
        lgbm_point=lgbm_point[picked].astype(np.float32),
        **_features(xd.iloc[picked], names),
    )

    # Each card candidate's schedule with only the departure hour moved.
    if arr is not None and len(cards):
        base = xd.iloc[cards].reset_index(drop=True)
        frames = []
        for hour_value in SWEEP_HOURS:
            moved = base.copy()
            old = _minutes(base[dep].to_numpy(np.float64), fmt)
            new = hour_value * 60.0 + np.mod(old, 60.0)
            moved[dep] = _clock(new, fmt)
            moved[arr] = _clock(_minutes(base[arr].to_numpy(np.float64), fmt) + (new - old), fmt)
            frames.append(moved)
        import pandas as pd

        swept = pd.concat(frames, ignore_index=True)
        s_weight, s_loc, s_scale = mixture_parts(ashlar, swept)
        s_quant = ashlar.predict_quantiles(swept, (0.1, 0.5, 0.9), calibrated=False)
        s_p60 = ashlar.predict_exceedance(swept, THRESHOLD, calibrated=False)
        s_cat_loc, s_cat_scale = cat_predict(swept)
        shape = (len(SWEEP_HOURS), len(cards))
        np.savez_compressed(
            out / "sweep.npz",
            hours=SWEEP_HOURS,
            row=rows[cards].astype(np.int64),
            weight=s_weight.reshape(*shape, -1).astype(np.float32),
            loc=s_loc.reshape(*shape, -1).astype(np.float32),
            scale=s_scale.reshape(*shape, -1).astype(np.float32),
            quantiles=s_quant.reshape(*shape, 3).astype(np.float32),
            p60=s_p60.reshape(shape).astype(np.float32),
            cat_loc=s_cat_loc.reshape(shape).astype(np.float32),
            cat_scale=s_cat_scale.reshape(shape).astype(np.float32),
            cat_p60=norm.sf((THRESHOLD - s_cat_loc) / s_cat_scale).reshape(shape).astype(np.float32),
        )

    # The card: every call of the quick start, for each candidate, in the order a user makes them.
    xc, yc = xd.iloc[cards].reset_index(drop=True), yd[cards]
    card = {
        "rows": rows[cards].tolist(),
        "features": names,
        "schedule": {name: xc[name].tolist() for name in names},
        "actual": yc.tolist(),
        "predict": ashlar.predict(xc).tolist(),
        "predict_quantiles": ashlar.predict_quantiles(xc, [0.1, 0.9]).tolist(),
        "median": ashlar.predict_quantiles(xc, [0.5])[:, 0].tolist(),
        "predict_exceedance": ashlar.predict_exceedance(xc, THRESHOLD).tolist(),
        "log_prob": ashlar.log_prob(xc, yc).tolist(),
        "sample": ashlar.sample(xc, n=5, seed=SEED).T.tolist(),
        "lightgbm_point": bench.tree_predict(lgbm, lgbm_frame, xc).tolist(),
        "lightgbm_sigma": sigma,
        "catboost_loc": cat_predict(xc)[0].tolist(),
        "catboost_scale": cat_predict(xc)[1].tolist(),
    }
    explained = ashlar.explain(xc, "quantile", level=0.9)
    card["explain"] = {
        "call": 'explain(x, "quantile", level=0.9)',
        "feature_names": explained.feature_names,
        "kinds": explained.kinds,
        "values": explained.values.tolist(),
        "prediction": explained.prediction.tolist(),
        "baseline": explained.baseline.tolist(),
        "numeric_baseline": explained.numeric_baseline,
    }
    background = x_train.iloc[np.sort(np.random.default_rng(SEED).choice(len(x_train), EXPLAIN_BACKGROUND, replace=False))]
    explained = ashlar.explain(xc, "quantile", level=0.9, background=background)
    card["explain_background"] = {
        "call": f'explain(x, "quantile", level=0.9, background=<{EXPLAIN_BACKGROUND:,} training rows>)',
        "values": explained.values.tolist(),
        "baseline": explained.baseline.tolist(),
        "numeric_baseline": explained.numeric_baseline,
    }

    # Calibrate on held-out flights the scores never see, then the calibrated calls.
    rest = np.setdiff1d(np.arange(len(y_test)), rows)
    calibration = np.sort(np.random.default_rng(SEED + 2).choice(rest, size=min(CALIBRATION_ROWS, len(rest)),
                                                                 replace=False))
    ashlar.calibrate(x_test.iloc[calibration].reset_index(drop=True), y_test[calibration])
    card["calibration_rows"] = int(len(calibration))
    card["predict_interval"] = ashlar.predict_interval(xc, coverage=0.8).tolist()
    card["predict_exceedance_calibrated"] = ashlar.predict_exceedance(xc, THRESHOLD).tolist()
    card["predict_quantiles_calibrated"] = ashlar.predict_quantiles(xc, [0.1, 0.5, 0.9]).tolist()
    calibrated = ashlar.predict_interval(xd, coverage=0.8)
    scores["ashlar"]["calibrated"] = {
        "rows": int(len(calibration)),
        "coverage80": float(np.mean((yd >= calibrated[:, 0]) & (yd <= calibrated[:, 1]))),
        "width80": float(np.mean(calibrated[:, 1] - calibrated[:, 0])),
        "brier60": float(np.mean((ashlar.predict_exceedance(xd, THRESHOLD) - late) ** 2)),
    }
    (out / "card.json").write_text(json.dumps(card, indent=1))

    summary = {
        "table": TABLE,
        "seed": SEED,
        "threshold": THRESHOLD,
        "training_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "scored_rows": int(len(yd)),
        "clock_format": fmt,
        "columns": {"departure": dep, "arrival": arr, "carrier": _column(x_train, "UniqueCarrier"),
                    "origin": _column(x_train, "Origin"), "dest": _column(x_train, "Dest"),
                    "month": _column(x_train, "Month"), "weekday": _column(x_train, "DayOfWeek"),
                    "distance": _column(x_train, "Distance")},
        "late_share": float(late.mean()),
        "below_zero_share": float(np.mean(yd < 0)),
        "lightgbm_sigma": sigma,
        "seconds": {"ashlar": ashlar_seconds, "catboost_uncertainty": cat_seconds, "lightgbm": round(lgbm_seconds, 1)},
        "epochs": len(ashlar.history_["train_loss"]),
        "scores": scores,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for path in sorted(out.iterdir()):
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KB", flush=True)


# ---------------------------------------------------------------- draw


def draw() -> None:
    raise SystemExit("the draw stage needs the style kit (examples/readme/style.py) from dev")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("compute", "draw"))
    parser.add_argument("--out", default=str(ARTIFACTS), help="artifact folder (compute)")
    parser.add_argument("--models", default=str(bench.CACHE / "readme" / NAME), help="where the fits are saved")
    parser.add_argument("--smoke", action="store_true", help="a quick check of the pipeline on few epochs, never for the figure")
    args = parser.parse_args()
    if args.stage == "compute":
        if args.smoke:
            bench.DIST_ROWS = 4_000
        compute(Path(args.out), Path(args.models), epochs=2 if args.smoke else None)
    else:
        draw()


if __name__ == "__main__":
    main()
