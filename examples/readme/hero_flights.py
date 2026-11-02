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
  flight.
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

``baseline`` adds the best tree distribution on this table, LightGBMLSS's
mixture of three normals, from the parameters the benchmark saved for the
same scored flights (``examples/benchmark_large.py --models
lightgbmlss_mixture``): ``lightgbmlss.json`` with its scores and
``lightgbmlss.npz`` with its quantiles for the pool's flights.

``draw`` reads them and writes ``docs/images/readme/hero-flights.png``:
sixteen of the pool's flights at evenly spaced ranks of P(delay > 60 min),
each as its mixture's continuous part with the central 80% solid, and a stem
for the component narrower than a quarter of a minute: the mass the mixture
puts on leaving exactly on time (18% on average, where 18.5% of the pool's
flights did). Under each ridge is LightGBMLSS's 80% interval for the same
flight. ``examples/readme/one_flight.py`` draws the card from the same
artifacts.

    python examples/readme/hero_flights.py compute   # about 40 minutes on a laptop CPU
    python examples/readme/hero_flights.py baseline  # a minute, after the benchmark's LightGBMLSS run
    python examples/readme/hero_flights.py coverage  # a few minutes, after compute and baseline
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


# ---------------------------------------------------------------- the best tree distribution

BASELINE = "lightgbmlss_mixture"  # the best tree distribution on airline delay by CRPS
EARLY_CUTOFF = -15.0  # minutes: an 80% interval starting earlier than this reaches where 0.1% of flights leave


def baseline(out: Path) -> None:
    """LightGBMLSS's mixture of three normals on the same flights, from the benchmark's saved parameters.

    ``examples/benchmark_large.py --models lightgbmlss_mixture`` keeps each scored row's weights, means, and
    scales in ``$ASHLAR_BENCH_CACHE/predictions/``. This stage fits nothing: it checks them against the
    benchmark's scores and writes ``lightgbmlss.json`` (scores on the 100,000 scored flights, P(delay > 60) by
    bins, the chance of a delay that rounds to zero, and how early its intervals start) and
    ``lightgbmlss.npz`` (the mixture, quantiles, and P(delay > 60) of every flight in ``pool.npz``).
    """
    from scipy.stats import norm

    path = bench.PREDICTIONS / f"{TABLE}-{BASELINE}-seed{SEED}.npz"
    with np.load(path, allow_pickle=False) as data:
        saved = data["rows"]
        weight, loc, scale = (data[key].astype(np.float64) for key in ("weight", "loc", "scale"))
    _, _, _, _, y_test = bench.splits(TABLE, SEED)
    rows = bench.dist_rows(len(y_test), SEED)
    where = np.searchsorted(saved, rows)
    if np.any(where >= len(saved)) or not np.array_equal(saved[np.minimum(where, len(saved) - 1)], rows):
        raise SystemExit(f"{path.name} does not hold the benchmark's scored rows")
    weight, loc, scale = weight[where], loc[where], scale[where]
    yd = y_test[rows]

    def cdf(value):
        return np.sum(weight * norm.cdf((value - loc) / scale), axis=1)

    quantiles = bench.mixture_quantiles(weight, loc, scale)
    p60 = 1.0 - cdf(THRESHOLD)
    on_minute = cdf(0.5) - cdf(-0.5)
    late = (yd > THRESHOLD).astype(np.float64)
    with np.load(out / "pool.npz", allow_pickle=False) as data:
        pool_rows, is_card = data["row"], data["is_card"]
        ashlar_parts = (data["weight"].astype(np.float64), data["loc"].astype(np.float64),
                        data["scale"].astype(np.float64))
    picked = np.searchsorted(rows, pool_rows)
    if not np.array_equal(rows[np.minimum(picked, len(rows) - 1)], pool_rows):
        raise SystemExit("pool.npz holds flights outside the scored rows")
    random = picked[~is_card]
    ashlar_on_minute = (mixture_cdf(*ashlar_parts, np.full(len(pool_rows), 0.5))
                        - mixture_cdf(*ashlar_parts, np.full(len(pool_rows), -0.5)))[~is_card]
    record = {
        "model": BASELINE,
        "seed": SEED,
        "scored_rows": int(len(yd)),
        "crps": bench.crps_mixture(weight, loc, scale, yd),
        "nll": bench.mixture_nll(weight, loc, scale, yd),
        **bench.interval(quantiles, yd),
        "brier60": float(np.mean((p60 - late) ** 2)),
        "p60_bins": _p60_bins(p60, late),
        "on_minute": {
            "predicted": float(on_minute.mean()), "observed": float(np.mean(yd == 0)),
            "pool_predicted": float(on_minute[random].mean()), "pool_observed": float(np.mean(yd[random] == 0)),
            "pool_ashlar": float(ashlar_on_minute.mean()),
        },
        "early": {"cutoff": EARLY_CUTOFF, "pool_share": float(np.mean(quantiles[random, 0] < EARLY_CUTOFF)),
                  "pool_median_q10": float(np.median(quantiles[random, 0]))},
    }
    if bench.OUT.exists():
        for line in bench.OUT.read_text().splitlines():
            run = json.loads(line)
            if (run["table"], run["model"], run["seed"]) == (TABLE, BASELINE, SEED):
                record["benchmark"] = {key: run.get(key) for key in ("crps", "nll", "coverage80", "width80", "seconds",
                                                                     "trees")}
    (out / "lightgbmlss.json").write_text(json.dumps(record, indent=1))
    np.savez_compressed(
        out / "lightgbmlss.npz",
        row=pool_rows.astype(np.int64),
        weight=weight[picked].astype(np.float32),
        loc=loc[picked].astype(np.float32),
        scale=scale[picked].astype(np.float32),
        q10=quantiles[picked, 0].astype(np.float32),
        q50=quantiles[picked, 1].astype(np.float32),
        q90=quantiles[picked, 2].astype(np.float32),
        p60=p60[picked].astype(np.float32),
        on_minute=on_minute[picked].astype(np.float32),
    )
    print(json.dumps({key: value for key, value in record.items() if key != "p60_bins"}, indent=1), flush=True)
    for name in ("lightgbmlss.json", "lightgbmlss.npz"):
        print(f"{name}: {(out / name).stat().st_size / 1024:.0f} KB", flush=True)


# ---------------------------------------------------------------- coverage: the intervals at equal coverage

PERIODS = ((0, 9, "before 9:00"), (9, 14, "9:00 to 14:00"), (14, 20, "14:00 to 20:00"), (20, 24, "20:00 on"))
FIFTHS = 5


def _nominal(pit, coverage: float = 0.8) -> float:
    """``calibrate``'s step for a density head: the central level whose interval held ``coverage`` of the rows."""
    from ashlar.calibrate import conformal_quantile

    return min(conformal_quantile(np.abs(2.0 * np.asarray(pit) - 1.0), coverage), 1.0)


def _fifths(p60):
    """Each flight's fifth, 0 to 4, of a model's P(delay > 60): from the flights it calls calmest to the riskiest."""
    return np.searchsorted(np.quantile(p60, np.linspace(0, 1, FIFTHS + 1)[1:-1]), p60, side="right")


def _held(inside, width, hour, fifths) -> dict:
    """An interval's coverage and mean width, overall, by scheduled hour and period, and by fifth of each risk."""

    def group(mask):
        return {"flights": int(mask.sum()), "coverage": float(inside[mask].mean()), "width": float(width[mask].mean())}

    return {
        "coverage80": float(inside.mean()),
        "width80": float(width.mean()),
        "by_hour": [group(hour == h) if np.any(hour == h) else None for h in range(24)],
        "by_period": [group((hour >= lo) & (hour < hi)) for lo, hi, _ in PERIODS],
        "by_fifth": {name: [group(f == i) for i in range(FIFTHS)] for name, f in fifths.items()},
    }


def coverage(out: Path, models: Path) -> None:
    """The 80% intervals of Ashlar, LightGBMLSS, and CatBoost at equal coverage, and where each holds.

    On the benchmark's 100,000 scored flights, from the fits ``compute`` saved and the parameters the
    benchmark saved for LightGBMLSS; it fits nothing. Each model's raw central 80% interval, and the same
    interval after ``calibrate``'s step for a density head (distributional conformal: the central level whose
    interval held 80% of held-out flights) cross-fitted on the scored flights, a random half calibrating the
    other and back. LightGBMLSS's parameters exist only for the scored flights, so the cross-fit is the one
    calibration all three share. Also Ashlar's own ``calibrate`` on the 20,000 other held-out flights, and
    CatBoost's normal with its scale stretched by the same step on those flights. Coverage and width overall,
    by scheduled hour of departure, and by fifth of each model's P(delay > 60). Writes ``coverage.json``.
    """
    from scipy.stats import norm

    task, x_train, x_test, y_train, y_test = bench.splits(TABLE, SEED)
    rows = bench.dist_rows(len(y_test), SEED)
    xd, yd = x_test.iloc[rows].reset_index(drop=True), y_test[rows].astype(np.float64)
    rest = np.setdiff1d(np.arange(len(y_test)), rows)
    calibration = np.sort(np.random.default_rng(SEED + 2).choice(rest, size=min(CALIBRATION_ROWS, len(rest)),
                                                                 replace=False))
    xc, yc = x_test.iloc[calibration].reset_index(drop=True), y_test[calibration].astype(np.float64)
    dep = _column(x_train, "CRSDepTime")
    fmt = _clock_format(x_train[dep].to_numpy(np.float64))
    hour = (np.floor(_minutes(xd[dep].to_numpy(np.float64), fmt) / 60.0).astype(int)) % 24

    ashlar, _ = fit_ashlar(x_train, y_train, models / "ashlar-mixture-seed0")
    _, cat_predict = bench.fit_catboost_uncertainty(x_train, y_train, SEED, path=models / "catboost-uncertainty-seed0.cbm")
    path = bench.PREDICTIONS / f"{TABLE}-{BASELINE}-seed{SEED}.npz"
    with np.load(path, allow_pickle=False) as data:
        saved = data["rows"]
        lss = tuple(data[key].astype(np.float64) for key in ("weight", "loc", "scale"))
    where = np.searchsorted(saved, rows)
    if np.any(where >= len(saved)) or not np.array_equal(saved[np.minimum(where, len(saved) - 1)], rows):
        raise SystemExit(f"{path.name} does not hold the benchmark's scored rows")
    lss = tuple(part[where] for part in lss)
    ours = mixture_parts(ashlar, xd)
    cat_loc, cat_scale = cat_predict(xd)

    def mixture(parts):
        return (lambda value, i: mixture_cdf(*(part[i] for part in parts), value),
                lambda level, i: bench.mixture_quantiles(*(part[i] for part in parts), levels=(level,))[:, 0])

    def normal(loc, scale):
        return (lambda value, i: norm.cdf((value - loc[i]) / scale[i]),
                lambda level, i: loc[i] + scale[i] * norm.ppf(level))

    heads = {"ashlar": mixture(ours), "lightgbmlss": mixture(lss), "catboost_uncertainty": normal(cat_loc, cat_scale)}
    everyone = np.arange(len(yd))
    p60 = {name: 1.0 - cdf(np.full(len(yd), THRESHOLD), everyone) for name, (cdf, _) in heads.items()}
    fifths = {name: _fifths(p) for name, p in p60.items() if name != "catboost_uncertainty"}
    order = np.random.default_rng(SEED + 3).permutation(len(yd))
    halves = (np.sort(order[: len(yd) // 2]), np.sort(order[len(yd) // 2:]))
    record = {"table": TABLE, "seed": SEED, "rows": int(len(yd)), "calibration_rows": int(len(yc)),
              "periods": [label for _, _, label in PERIODS], "fifths": "fifths of each model's own P(delay > 60), "
              "calmest first; by_fifth.ashlar and by_fifth.lightgbmlss use that model's fifths for every model",
              "models": {}}
    for name, (cdf, quantile) in heads.items():
        groups = {**fifths, "own": _fifths(p60[name])}
        low, high = quantile(0.1, everyone), quantile(0.9, everyone)
        entry = {"raw": _held((yd >= low) & (yd <= high), high - low, hour, groups)}
        inside, width, nominal = np.zeros(len(yd), bool), np.zeros(len(yd)), []
        for fit_on, score_on in (halves, halves[::-1]):
            level = _nominal(cdf(yd[fit_on], fit_on))
            low, high = quantile(0.5 * (1 - level), score_on), quantile(0.5 * (1 + level), score_on)
            inside[score_on], width[score_on] = (yd[score_on] >= low) & (yd[score_on] <= high), high - low
            nominal.append(level)
        entry["crossfit"] = {"nominal": nominal, **_held(inside, width, hour, groups)}
        record["models"][name] = entry
        print(name, {key: round(value["coverage80"], 4) for key, value in entry.items()},
              {key: round(value["width80"], 2) for key, value in entry.items()}, flush=True)

    # Ashlar's own calibrate on the 20,000 other held-out flights, as the one-flight card uses it.
    ashlar.calibrate(xc, yc)
    calibrated = ashlar.predict_interval(xd, coverage=0.8)
    groups = {**fifths, "own": fifths["ashlar"]}
    record["models"]["ashlar"]["calibrate"] = _held((yd >= calibrated[:, 0]) & (yd <= calibrated[:, 1]),
                                                    calibrated[:, 1] - calibrated[:, 0], hour, groups)
    # CatBoost's normal with its scale stretched by the same step on the same 20,000 flights.
    loc_c, scale_c = cat_predict(xc)
    level = _nominal(norm.cdf((yc - loc_c) / scale_c))
    factor = float(norm.ppf(0.5 * (1 + level)) / norm.ppf(0.9))
    scaled = cat_scale * factor
    half = norm.ppf(0.9) * scaled
    late = (yd > THRESHOLD).astype(np.float64)
    p60_scaled = norm.sf((THRESHOLD - cat_loc) / scaled)
    z = (yd - cat_loc) / scaled
    crps = scaled * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / math.sqrt(math.pi))
    record["models"]["catboost_uncertainty"]["scaled"] = {
        "factor": factor, "crps": float(crps.mean()), "brier60": float(np.mean((p60_scaled - late) ** 2)),
        "p60_bins": _p60_bins(p60_scaled, late),
        **_held(np.abs(yd - cat_loc) <= half, 2 * half, hour, {**fifths, "own": _fifths(p60["catboost_uncertainty"])}),
    }
    print("catboost scaled", {key: round(value, 4) for key, value in record["models"]["catboost_uncertainty"]["scaled"].items()
                              if isinstance(value, float)}, flush=True)
    (out / "coverage.json").write_text(json.dumps(record, indent=1))
    for name, entry in record["models"].items():
        for kind, held in entry.items():
            periods = ", ".join(f"{g['coverage']:.0%}" for g in held["by_period"])
            own = ", ".join(f"{g['coverage']:.0%}" for g in held["by_fifth"]["own"])
            print(f"{name:21s} {kind:9s} {held['coverage80']:.1%} at {held['width80']:5.1f} | by period {periods} | "
                  f"by own fifth {own}", flush=True)
    print(f"coverage.json: {(out / 'coverage.json').stat().st_size / 1024:.0f} KB", flush=True)


# ---------------------------------------------------------------- draw

SOFT = 10.0  # minutes: the delay axis is asinh(delay / 10), linear near zero and logarithmic past half an hour
X_LO, X_HI = -40.0, 300.0
TICKS = (-30, -10, 0, 10, 30, 60, 120, 300)
RIDGES = 16
NEEDLE = 2.0  # the stem's length, in rows, per unit of probability at the point mass
P60 = "P(delay\u00a0>\u00a01\u00a0h)"  # unbreakable
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
DAYS = "Mon Tue Wed Thu Fri Sat Sun".split()


def warp(x):
    return np.arcsinh(np.asarray(x, dtype=np.float64) / SOFT)


def unwarp(u):
    return SOFT * np.sinh(np.asarray(u, dtype=np.float64))


def load(folder: Path):
    summary = json.loads((folder / "summary.json").read_text())
    with np.load(folder / "pool.npz", allow_pickle=False) as data:
        pool = {key: data[key] for key in data.files}
    card = json.loads((folder / "card.json").read_text())
    return summary, pool, card


def load_baseline(folder: Path):
    """LightGBMLSS's mixture on the pool's flights (``lightgbmlss.npz``, in pool order) and its scores."""
    with np.load(folder / "lightgbmlss.npz", allow_pickle=False) as data:
        flights = {key: data[key] for key in data.files}
    return flights, json.loads((folder / "lightgbmlss.json").read_text())


def clock(value, fmt) -> str:
    minutes = float(_minutes(value, fmt))
    return f"{int(minutes // 60) % 24:02d}:{int(round(minutes % 60)) % 60:02d}"


def flight_label(pool, i, summary) -> str:
    """"DL ATL→LGA · Fri 18:05 · Dec" from a pool row's schedule."""
    col = summary["columns"]
    x = lambda name: pool[f"x_{col[name]}"][i]  # noqa: E731
    return (f"{x('carrier')} {x('origin')}→{x('dest')} · {DAYS[int(x('weekday')) - 1]} "
            f"{clock(x('departure'), summary['clock_format'])} · {MONTHS[int(x('month')) - 1]}")


def pick_ridges(pool, count: int = RIDGES):
    """A fixed rule: the random pool's flights at evenly spaced ranks of Ashlar's P(delay > 60 min)."""
    candidates = np.flatnonzero(~pool["is_card"])
    order = candidates[np.argsort(pool["p60"][candidates], kind="stable")]
    ranks = ((np.arange(count) + 0.5) / count * len(order)).astype(int)
    return order[ranks]


ATOM = 0.25  # minutes: a mixture component narrower than this is a point mass (on-time departures, at zero)


def split_atom(pool, i):
    """Flight ``i``'s mixture as its continuous part and its point mass: ``(weight, loc, scale), (mass, at)``."""
    weight, loc, scale = (pool[key][i].astype(np.float64) for key in ("weight", "loc", "scale"))
    narrow = scale < ATOM
    mass = float(weight[narrow].sum())
    at = float(np.round(np.sum(weight[narrow] * loc[narrow]) / mass)) if mass > 0 else 0.0
    return (weight[~narrow], loc[~narrow], scale[~narrow]), (mass, at)


def warped_density(pool, i, u):
    """Flight ``i``'s continuous density on the warped axis, so that area is probability."""
    (weight, loc, scale), _ = split_atom(pool, i)
    density = mixture_density(weight[None], loc[None], scale[None], unwarp(u))[0]
    return density * SOFT * np.cosh(u)


def legend(fig, k, items, *, line: int):
    """``k.legend_row`` with one more kind, "needle": a short vertical stroke for a point mass."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = []
    for kind, color, label in items:
        if kind in ("patch", "pale"):
            handles.append(Patch(facecolor=color, edgecolor="none", label=label))
        elif kind == "bar":
            handles.append(Line2D([], [], color=color, lw=4.5, solid_capstyle="butt", label=label))
        elif kind == "dot":
            handles.append(Line2D([], [], color=color, marker="o", lw=0, ms=7, markeredgecolor=k.CARD, label=label))
        elif kind == "needle":
            handles.append(Line2D([], [], color=color, marker="|", lw=0, ms=16, mew=2.2, label=label))
    box = fig.box
    return fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(box.left / box.width, (0.3 + 0.32 * line) / box.height),
                      ncol=len(handles), frameon=False, prop=k.F.label, handlelength=1.8, columnspacing=2.0,
                      labelcolor=k.INK, borderaxespad=0, handletextpad=0.6)


def p60_errors(bins, least: int = 1000) -> float:
    """The largest gap, in points, between predicted and observed P(delay > 60) over bins of ``least`` flights."""
    return max(abs(b["predicted"] - b["observed"]) * 100 for b in bins if b["flights"] >= least)


def draw(folder: Path = ARTIFACTS) -> None:
    import style as k

    summary, pool, _card = load(folder)
    other, theirs = load_baseline(folder)
    if not np.array_equal(other["row"], pool["row"]):
        raise SystemExit("lightgbmlss.npz is not in pool.npz's order: rerun the baseline stage")
    ours = summary["scores"]["ashlar"]
    exact = ours["exact_quantiles"]  # the drawn 80% is predict_quantiles' 0.1 to 0.9, as LightGBMLSS's is
    picks = pick_ridges(pool)
    u = np.linspace(warp(X_LO), warp(X_HI), 1200)
    densities = np.stack([warped_density(pool, i, u) for i in picks])
    inner = [(float(warp(pool["q10"][i])), float(warp(pool["q90"][i]))) for i in picks]
    solid = k.mix(k.TEAL, k.CARD, 0.36)

    fig = k.card(
        "Every flight gets its own distribution.",
        f"Airline delay: {summary['training_rows'] + summary['test_rows']:,} US flights, schedule only, no weather. "
        "Each ridge is one held-out flight's predicted departure delay from a single ASRRegressor(task=\"mixture\") "
        "fit, its central 80% solid. Under it, the 80% interval of LightGBMLSS's mixture of three normals for the "
        "same flight, the best tree distribution on this table, and the delay that happened.",
        height=11.4,
        footer=3,
    )
    ax = k.panel(fig, (0.0, 0.0, 1.0, 1.0), head=0.1, tail=0.8, lpad=3.0, rpad=2.0)
    height = 1.5
    bases = k.ridgeline(ax, u, densities, inner=inner, gap=1.0, height=height, color=solid, pale=k.TEAL_PALE,
                        edge=k.TEAL_DARK, lw=1.2, floor=0.0015)
    left, right = u[0], u[-1]
    # The right margin's two columns of P(delay > 1 h), placed in inches from the axes' right edge.
    inch = (right - left) / (ax.get_position().width * fig.get_figwidth())
    ours_x = right + 0.55 * inch
    theirs_x = ours_x + (0.25 + k.text_width("LightGBMLSS", k.F.small)) * inch
    # The point mass at zero, flights that leave on the minute: a stem from the baseline, its length the probability.
    for row, i in enumerate(picks):
        _, (mass, at) = split_atom(pool, i)
        if mass <= 0:
            continue
        ua, base = float(warp(at)), bases[row]
        ax.plot([ua, ua], [base, base + mass * NEEDLE], color=k.TEAL_DARK, lw=2.6, solid_capstyle="butt", zorder=403)
        ax.scatter([ua], [base + mass * NEEDLE], s=34, color=k.TEAL_DARK, edgecolor=k.CARD, lw=0.8, zorder=404)
    top = bases[-1] + height + 0.25
    ax.axvline(warp(0.0), color=k.HAIR, lw=1.0, zorder=0)
    ax.axvline(warp(THRESHOLD), color=k.MUTED, lw=0.9, ls=(0, (2, 3)), zorder=0)
    for row, i in enumerate(picks):
        base = bases[row]
        y = base - 0.2
        lo, hi = float(other["q10"][i]), float(other["q90"][i])
        a, b = max(warp(lo), left), min(warp(hi), right)
        ax.plot([a, b], [y, y], color=k.CARD, lw=5.2, solid_capstyle="butt", zorder=399)
        ax.plot([a, b], [y, y], color=k.CLAY, lw=2.6, solid_capstyle="butt", zorder=400)
        actual = float(pool["actual"][i])
        marker = ">" if actual > X_HI else "<" if actual < X_LO else "o"
        ax.scatter([min(max(warp(actual), left), right)], [y], s=30, marker=marker, color=k.INK, edgecolor=k.CARD,
                   lw=1.0, zorder=401, clip_on=False)
        ax.text(left - 0.1, base - 0.08, flight_label(pool, i, summary), fontproperties=k.F.label, color=k.INK,
                ha="right", va="center")
        k.value(ax, ours_x, base - 0.08, k.pct(float(pool["p60"][i])), k.TEAL, ha="right")
        k.value(ax, theirs_x, base - 0.08, k.pct(float(other["p60"][i])), k.CLAY, ha="right")
    head = bases[-1] + 1.0
    ax.text((ours_x - k.text_width("Ashlar", k.F.small) * inch + theirs_x) / 2, head + 0.62, P60,
            fontproperties=k.F.small, color=k.MUTED, ha="center", va="bottom")
    ax.text(ours_x, head, "Ashlar", fontproperties=k.F.small, color=k.TEAL, ha="right", va="bottom")
    ax.text(theirs_x, head, "LightGBMLSS", fontproperties=k.F.small, color=k.CLAY, ha="right", va="bottom")
    k.style(ax, frame="b")
    ax.set_yticks([])
    ax.set_xticks(warp(TICKS))
    ax.set_xticklabels([k.num(t) for t in TICKS])
    for label in ax.get_xticklabels():
        label.set_fontproperties(k.F.tick)
    ax.set_xlim(left, right)
    ax.set_ylim(-0.6, top)
    k.axis_names(ax, x="departure delay, minutes (log scale beyond half an hour)")

    legend(fig, k, [("patch", solid, "Ashlar's central 80%"), ("pale", k.TEAL_PALE, "its tails"),
                    ("needle", k.TEAL_DARK, "its chance of leaving on the minute"),
                    ("bar", k.CLAY, "LightGBMLSS's 80%"), ("dot", k.INK, "what happened")], line=2)
    minutes = {key: round((value or 0) / 60) for key, value in (("ours", summary["seconds"]["ashlar"]),
                                                                ("theirs", theirs.get("benchmark", {}).get("seconds")))}
    k.footnote(fig, f"On {summary['scored_rows']:,} held-out flights, CRPS {ours['crps']:.2f} against LightGBMLSS's "
                    f"{theirs['crps']:.2f}; 80% intervals held {k.pct(exact['coverage80'], 1)} at "
                    f"{exact['width80']:.0f} minutes wide against {k.pct(theirs['coverage80'], 1)} at "
                    f"{theirs['width80']:.0f}; {P60} within {p60_errors(ours['p60_bins']):.1f} points of the observed "
                    f"rate in every bin of 1,000 flights or more, against {p60_errors(theirs['p60_bins']):.1f}. "
                    f"One fit took {minutes['ours']} minutes on a laptop CPU, LightGBMLSS's {minutes['theirs']}.",
               line=0)
    k.save(fig, NAME)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stage", choices=("compute", "baseline", "coverage", "draw"))
    parser.add_argument("--artifacts", default=str(ARTIFACTS), help="the artifact folder compute writes and draw reads")
    parser.add_argument("--models", default=str(bench.CACHE / "readme" / NAME), help="where the fits are saved")
    parser.add_argument("--smoke", action="store_true", help="a quick check of the pipeline on few epochs, never for the figure")
    args = parser.parse_args()
    if args.smoke:
        bench.DIST_ROWS = 4_000
    if args.stage == "compute":
        compute(Path(args.artifacts), Path(args.models), epochs=2 if args.smoke else None)
    elif args.stage == "baseline":
        baseline(Path(args.artifacts))
    elif args.stage == "coverage":
        coverage(Path(args.artifacts), Path(args.models))
    else:
        draw(Path(args.artifacts))


if __name__ == "__main__":
    main()
