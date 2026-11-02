"""Target shapes a tree regressor cannot return, against the distribution a tree user would build.

Each stage is one shape, one public table, and the strongest tree baseline at
hand. Every Ashlar model is ``ASRRegressor(task=...)`` with no other
arguments except the histogram's bins. The trees are early-stopped as in
``benchmark_readme.py``: learning rate 0.05, up to 4,000 trees, 100 rounds of
patience on a 20% validation slice of the training rows, native categorical
handling. Every score is on held-out rows, in the target's own units.

- ``spike``: NYC taxi fares (the seaborn ``taxis`` table, 6,383 trips with
  zones), given the zones, boroughs, hour, weekday, passengers, and cab
  color; distance, tolls, tip, and total are not features. Every trip between
  JFK and Manhattan, 110 of them, ran at the $52 flat fare. ``histogram``
  with 48 bins, ``mixture``, and ``flow``, against LightGBM's mean with a
  constant normal and CatBoost's ``RMSEWithUncertainty`` (a mean and a
  variance per row). CRPS, NLL, and the 80% interval on every test trip; on
  the JFK and Manhattan test trips, the probability each model puts within
  $0.50 and within $2 of $52, and the CRPS. Three random 80/20 splits.
- ``zero``: NYC taxi tips from the same table, payment type and total hidden,
  35.8% exactly $0, on the five splits of ``benchmark_hurdle.py``. The
  ``hurdle`` head against the tree user's hurdle: a LightGBM classifier for
  P(tip > 0) and a LightGBM regressor on the tipped trips, either a gamma
  objective with one shape fitted on the validation slice or the log tip with
  a constant normal, composed into one distribution with an exact zero.
  Beside them, ``mixture`` and LightGBM's mean with a constant normal, which
  can only put their mass below zero. Brier score of P(no tip), read as
  ``cdf(0)``; CRPS; the 80% interval.
- ``tail``: freMTPL2 claims (CASdatasets; every policy with a claim, on the
  splits of ``benchmark_monotone.py``), the log of the average cost per
  claim. Many claims sit at a few fixed amounts. ``mixture`` and
  ``gaussian`` against LightGBM with a gamma objective on the cost in euros
  (one shape fitted on the validation slice), LightGBM on the log cost with a
  constant normal (a lognormal cost), and CatBoost ``RMSEWithUncertainty`` on
  the log cost. NLL and CRPS of the log cost, the 80% interval, and the mass
  each puts within 1% of the most common amount and in the band of the four
  most common amounts (EUR 1,128 to 1,204, widened by 1%), against the share
  of claims there. Three splits.
- ``ceiling``: California median house values in $100,000, which the census
  table caps at $500,001 (4.7% of block groups sit there, 46% of those with a
  median income of $70,000 or more). ``mixture``, ``flow``, and ``histogram``
  with 48 bins, against LightGBM's mean with a constant normal and CatBoost
  ``RMSEWithUncertainty``. CRPS, NLL, the 80% interval, the Brier score of
  P(within $10k of the cap or above), and the mass more than $10k past the
  cap, which no block group has; on every test row and on the rich ones.
  Three random 80/20 splits.
- ``skew``: diamond prices in $1,000 on the 12,000-row sample of
  ``benchmark_readme.py``. ``mixture`` and ``flow`` against the same two trees:
  CRPS, NLL, the 80% interval, and the mass below $0 on the cheapest quarter
  of stones. Three random 80/20 splits.
- ``counts``: LightGBM's Poisson mean with a negative binomial around it,
  its one dispersion fitted on the validation slice: the tree user's fix for
  a Poisson that is too narrow, scored as ``benchmark_uncertainty.py
  --stage counts`` scores its count heads, on the same splits and tables.

Each stage writes ``/tmp/ashlar-bench/shapes-<stage>.json``; a finished split
is skipped on a rerun. ``report`` prints the tables of docs/benchmarks.md.

    pip install lightgbm xgboost catboost pyreadr
    JAX_PLATFORMS=cpu python examples/benchmark_shapes.py --stage all
    python examples/benchmark_shapes.py --stage report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark_readme as bench  # noqa: E402

from ashlar import ASRRegressor  # noqa: E402

DATA_DIR = bench.DATA_DIR
DRAWS = 2000
COUNT_DRAWS = 400  # as benchmark_uncertainty.py scores the count heads
Z80 = 1.2815515655446004
FLAT_FARE = 52.0
FARE_BINS = 48
SPIKE_SEEDS = (0, 1, 2)
ZERO_SEEDS = (0, 1, 2, 3, 4)  # the splits of benchmark_hurdle.py
TAIL_SEEDS = (0, 1, 2)  # the splits of benchmark_monotone.py
COUNT_SEEDS = (0, 1, 2)  # the splits of benchmark_uncertainty.py --stage counts


# ---------------------------------------------------------------- data


def fares():
    """NYC taxi fares from where and when the trip ran: the columns of ``readme_figures.fare_rows``."""
    frame = pd.read_csv(bench._path("taxis.csv"), parse_dates=["pickup"])
    frame = frame.dropna(subset=["pickup_zone", "dropoff_zone"]).reset_index(drop=True)
    pickup = frame["pickup"]
    x = pd.DataFrame(
        {
            "hour": pickup.dt.hour.astype(np.float64),
            "weekday": pickup.dt.dayofweek.astype(np.float64),
            "passengers": frame["passengers"].astype(np.float64),
            "color": frame["color"],
            "pickup_zone": frame["pickup_zone"],
            "dropoff_zone": frame["dropoff_zone"],
            "pickup_borough": frame["pickup_borough"],
            "dropoff_borough": frame["dropoff_borough"],
        }
    )
    return bench._strings(x), frame["fare"].to_numpy(np.float64)


def flat_fare_trips(x) -> np.ndarray:
    """Trips between JFK and Manhattan, in either direction: the ones the $52 flat fare applies to."""
    out = (x["pickup_zone"] == "JFK Airport") & (x["dropoff_borough"] == "Manhattan")
    back = (x["dropoff_zone"] == "JFK Airport") & (x["pickup_borough"] == "Manhattan")
    return (out | back).to_numpy()


def claims():
    """freMTPL2 policies with a claim: the features and the log of the average cost per claim."""
    from benchmark_chain import CATEGORICAL, NUMERIC, policies

    frame = policies()
    return frame, NUMERIC + CATEGORICAL


def tail_split(frame, seed):
    """The split of ``benchmark_monotone.py``: 20% of all policies held out, then the ones with a claim."""
    order = np.random.default_rng(seed).permutation(len(frame))
    test_size = len(frame) // 5
    test, train = frame.iloc[order[:test_size]], frame.iloc[order[test_size:]]
    return train[train["N"] > 0], test[test["N"] > 0]


# ---------------------------------------------------------------- the trees' distributions


def lightgbm_normal(x_train, y_train, x_test, seed):
    """LightGBM's early-stopped mean, with a normal whose scale is the residual sd on the validation slice."""
    model, frame = bench._fit_gbt("lightgbm", "regression", True, seed, x_train, y_train)
    _fit_x, val_x, _fit_y, val_y = bench._gbt_split(frame, y_train, seed)
    sigma = float(np.std(val_y - model.predict(val_x)))
    loc = model.predict(bench._align_categories(frame, x_test))
    return loc, np.full(len(loc), sigma)


def catboost_normal(x_train, y_train, x_test, seed):
    """CatBoost ``RMSEWithUncertainty``: a mean and a variance per row, early-stopped on the same slice."""
    import catboost

    frame = bench._gbt_frame(x_train)
    cats = [c for c in frame.columns if isinstance(frame[c].dtype, pd.CategoricalDtype)]

    def prep(f):
        f = f.copy()
        for c in cats:
            f[c] = f[c].astype(object).where(f[c].notna(), "__missing__").astype(str)
        return f

    fit_x, val_x, fit_y, val_y = bench._gbt_split(frame, y_train, seed)
    model = catboost.CatBoostRegressor(
        iterations=4000, learning_rate=0.05, od_type="Iter", od_wait=100, random_seed=seed, thread_count=1,
        verbose=False, cat_features=cats, loss_function="RMSEWithUncertainty",
    )
    model.fit(prep(fit_x), fit_y, eval_set=(prep(val_x), val_y), use_best_model=True)
    pred = model.predict(prep(bench._align_categories(frame, x_test)), prediction_type="RMSEWithUncertainty")
    return pred[:, 0], np.sqrt(np.maximum(pred[:, 1], 1e-12))


def _gamma_shape(y, mean) -> float:
    """The one gamma shape that best fits ``y`` around per-row means, by maximum likelihood."""
    y, mean = np.asarray(y, np.float64), np.asarray(mean, np.float64)

    def loss(log_k):
        k = np.exp(log_k)
        return -np.sum(stats.gamma.logpdf(y, a=k, scale=mean / k))

    return float(np.exp(optimize.minimize_scalar(loss, bounds=(-6.0, 6.0), method="bounded").x))


def lightgbm_gamma(x_train, y_train, x_test, seed):
    """LightGBM with a gamma objective on a positive target, and one shape fitted on the validation slice."""
    model, frame = bench._fit_gbt("lightgbm", "regression", True, seed, x_train, y_train, objective="gamma")
    _fit_x, val_x, _fit_y, val_y = bench._gbt_split(frame, y_train, seed)
    shape = _gamma_shape(val_y, model.predict(val_x))
    return model.predict(bench._align_categories(frame, x_test)), shape


def _nb_dispersion(y, mean) -> float:
    """The one negative binomial dispersion (variance = mean + d × mean²) that best fits ``y``."""
    y, mean = np.asarray(y, np.float64), np.maximum(np.asarray(mean, np.float64), 1e-9)

    def loss(log_d):
        size = np.exp(-log_d)
        return -np.sum(stats.nbinom.logpmf(y, size, size / (size + mean)))

    return float(np.exp(optimize.minimize_scalar(loss, bounds=(-12.0, 4.0), method="bounded").x))


# ---------------------------------------------------------------- scores


def crps(draws, y) -> float:
    """``draws`` is ``(rows, n)``."""
    return bench._crps(np.asarray(draws, np.float64), np.asarray(y, np.float64))


def normal_scores(loc, scale, y, seed) -> dict:
    draws = loc[:, None] + scale[:, None] * np.random.default_rng(seed).standard_normal((len(loc), DRAWS))
    low, high = loc - Z80 * scale, loc + Z80 * scale
    return {
        "crps": crps(draws, y),
        "nll": float(-np.mean(stats.norm.logpdf(y, loc, scale))),
        "coverage80": float(np.mean((y >= low) & (y <= high))),
        "width80": float(np.mean(high - low)),
    }


def ashlar_scores(dist, y, seed) -> dict:
    draws = np.asarray(dist.sample(jax.random.PRNGKey(seed), (DRAWS,)), np.float64).T
    low = np.asarray(dist.quantile(jnp.full(len(y), 0.1)), np.float64)
    high = np.asarray(dist.quantile(jnp.full(len(y), 0.9)), np.float64)
    return {
        "crps": crps(draws, y),
        "nll": float(-np.mean(np.asarray(dist.log_prob(jnp.asarray(y, jnp.float32)), np.float64))),
        "coverage80": float(np.mean((y >= low) & (y <= high))),
        "width80": float(np.mean(high - low)),
    }


def _mean_sd(values, digits=3, spread="sd"):
    values = np.asarray(values, np.float64)
    # As the other benchmark scripts: the standard deviation over splits (ddof 0), or the standard error of
    # the mean over them, as benchmark_hurdle.py reports it.
    if spread == "se":
        error = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    else:
        error = values.std()
    return f"{values.mean():.{digits}f} ± {error:.{digits}f}"


# ---------------------------------------------------------------- stage: spike


SPIKE_HEADS = {
    "histogram": {"task": "histogram", "head_kwargs": {"bins": FARE_BINS}},
    "mixture": {"task": "mixture"},
    "flow": {"task": "flow"},
}


def _window(cdf, low, high) -> np.ndarray:
    return np.asarray(cdf(high), np.float64) - np.asarray(cdf(low), np.float64)


def stage_spike(results):
    out = results.setdefault("spike", {})
    x, y = fares()
    flat = flat_fare_trips(x)
    out["table"] = {"rows": int(len(y)), "flat_trips": int(flat.sum()), "flat_share_at_52": float(np.mean(y[flat] == FLAT_FARE))}
    runs = out.setdefault("runs", {})
    for seed in SPIKE_SEEDS:
        if str(seed) in runs:
            continue
        index_train, index_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        x_train, x_test = x.iloc[index_train], x.iloc[index_test]
        y_train, y_test = y[index_train], y[index_test]
        on_flat = flat[index_test]
        run = {"flat_test_trips": int(on_flat.sum())}

        def spike_scores(cdf, draws):
            return {
                "within_050": float(np.mean(_window(cdf, FLAT_FARE - 0.5, FLAT_FARE + 0.5)[on_flat])),
                "within_2": float(np.mean(_window(cdf, FLAT_FARE - 2.0, FLAT_FARE + 2.0)[on_flat])),
                "flat_crps": crps(draws[on_flat], y_test[on_flat]),
            }

        for name, params in SPIKE_HEADS.items():
            started = time.process_time()
            model = ASRRegressor(random_state=seed, **params).fit(x_train, y_train)
            dist = model.predict_dist(x_test)
            record = ashlar_scores(dist, y_test, seed)
            draws = np.asarray(dist.sample(jax.random.PRNGKey(seed + 1), (DRAWS,)), np.float64).T
            record.update(spike_scores(lambda v: dist.cdf(jnp.full(len(y_test), v, jnp.float32)), draws))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record
        for name, fit in (("lightgbm_normal", lightgbm_normal), ("catboost_uncertainty", catboost_normal)):
            started = time.process_time()
            loc, scale = fit(x_train, y_train, x_test, seed)
            record = normal_scores(loc, scale, y_test, seed)
            draws = loc[:, None] + scale[:, None] * np.random.default_rng(seed + 1).standard_normal((len(loc), DRAWS))
            record.update(spike_scores(lambda v: stats.norm.cdf(v, loc, scale), draws))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record
        runs[str(seed)] = run
        print("spike", seed, json.dumps({k: (round(v["crps"], 3), round(v["within_050"], 3)) for k, v in run.items() if isinstance(v, dict)}), flush=True)
        _save(results)


# ---------------------------------------------------------------- stage: zero


ZERO_HEADS = {
    "hurdle_gamma": {"task": "hurdle", "head_kwargs": {"severity": "gamma"}},
    "hurdle_lognormal": {"task": "hurdle"},
    "mixture": {"task": "mixture"},
}


def _hurdle_scores(zero, amount_draws, amount_quantile, amount_logpdf, y, seed) -> dict:
    """A tree-built hurdle: ``zero`` is P(y = 0) per row; the amount is a positive distribution per row."""
    rng = np.random.default_rng(seed)
    draws = np.where(rng.random(amount_draws.shape) < zero[:, None], 0.0, amount_draws)

    def quantile(level):
        inner = np.clip((level - zero) / np.maximum(1.0 - zero, 1e-12), 1e-9, 1 - 1e-9)
        return np.where(level <= zero, 0.0, amount_quantile(inner))

    low, high = quantile(0.1), quantile(0.9)
    positive = y > 0
    log_mass = np.where(positive, np.log1p(-np.minimum(zero, 1 - 1e-7)) + amount_logpdf(np.where(positive, y, 1.0)),
                        np.log(np.clip(zero, 1e-30, 1.0)))
    return {
        "crps": crps(draws, y),
        "brier_zero": float(np.mean((zero - (y == 0)) ** 2)),
        "mixed_nll": float(-np.mean(log_mass)),
        "coverage80": float(np.mean((y >= low) & (y <= high))),
        "width80": float(np.mean(high - low)),
    }


def stage_zero(results):
    out = results.setdefault("zero", {})
    x, y = bench.taxis("tip")
    out["table"] = {"rows": int(len(y)), "zero_share": float(np.mean(y == 0))}
    runs = out.setdefault("runs", {})
    for seed in ZERO_SEEDS:
        if str(seed) in runs:
            continue
        index_train, index_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        x_train, x_test = x.iloc[index_train], x.iloc[index_test]
        y_train, y_test = y[index_train], y[index_test]
        run = {}
        for name, params in ZERO_HEADS.items():
            started = time.process_time()
            model = ASRRegressor(random_state=seed, **params).fit(x_train, y_train)
            dist = model.predict_dist(x_test)
            record = ashlar_scores(dist, y_test, seed)
            zero = np.asarray(dist.cdf(jnp.zeros(len(y_test))), np.float64)
            record["brier_zero"] = float(np.mean((zero - (y_test == 0)) ** 2))
            if name.startswith("hurdle"):
                record["mixed_nll"] = record.pop("nll")
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record

        # The tree user's hurdle: P(tip > 0) from a classifier, the amount from a regressor on the tipped trips.
        started = time.process_time()
        classifier, frame = bench._fit_gbt("lightgbm", "classification", True, seed, x_train, (y_train > 0).astype(int))
        zero = 1.0 - classifier.predict_proba(bench._align_categories(frame, x_test))[:, 1]
        classifier_seconds = time.process_time() - started
        tipped = y_train > 0
        rng = np.random.default_rng(seed)

        started = time.process_time()
        mean, shape = lightgbm_gamma(x_train[tipped], y_train[tipped], x_test, seed)
        record = _hurdle_scores(
            zero,
            rng.gamma(shape, (mean / shape)[:, None], size=(len(mean), DRAWS)),
            lambda level: stats.gamma.ppf(level, shape, scale=mean / shape),
            lambda value: stats.gamma.logpdf(value, shape, scale=mean / shape),
            y_test,
            seed,
        )
        record.update(gamma_shape=shape, cpu_seconds=classifier_seconds + time.process_time() - started)
        run["lightgbm_hurdle_gamma"] = record

        started = time.process_time()
        loc, scale = lightgbm_normal(x_train[tipped], np.log(y_train[tipped]), x_test, seed)
        record = _hurdle_scores(
            zero,
            np.exp(loc[:, None] + scale[:, None] * rng.standard_normal((len(loc), DRAWS))),
            lambda level: np.exp(loc + scale * stats.norm.ppf(level)),
            lambda value: stats.norm.logpdf(np.log(value), loc, scale) - np.log(value),
            y_test,
            seed,
        )
        record.update(log_sigma=float(scale[0]), cpu_seconds=classifier_seconds + time.process_time() - started)
        run["lightgbm_hurdle_lognormal"] = record

        started = time.process_time()
        loc, scale = lightgbm_normal(x_train, y_train, x_test, seed)
        record = normal_scores(loc, scale, y_test, seed)
        record["brier_zero"] = float(np.mean((stats.norm.cdf(0.0, loc, scale) - (y_test == 0)) ** 2))
        record["cpu_seconds"] = time.process_time() - started
        run["lightgbm_normal"] = record

        runs[str(seed)] = run
        print("zero", seed, json.dumps({k: (round(v["crps"], 3), round(v["brier_zero"], 3)) for k, v in run.items()}), flush=True)
        _save(results)


# ---------------------------------------------------------------- stage: tail


TAIL_HEADS = {"mixture": {"task": "mixture"}, "gaussian": {"task": "gaussian"}}


def stage_tail(results):
    out = results.setdefault("tail", {})
    frame, columns = claims()
    claimed = frame[frame["N"] > 0]
    log_cost = np.log(claimed["S"].to_numpy())
    values, counts = np.unique(np.round(claimed["S"].to_numpy(), 2), return_counts=True)
    lump = float(values[np.argmax(counts)])
    # The four most common amounts, all between about EUR 1,100 and 1,250, and a band 1% past them.
    common = values[np.argsort(-counts)[:4]]
    band = (float(np.log(common.min() * 0.99)), float(np.log(common.max() * 1.01)))
    out["table"] = {"claims": int(len(claimed)), "lump": lump, "lump_share": float(np.mean(np.abs(log_cost - np.log(lump)) <= 0.01)),
                    "common": [float(v) for v in np.sort(common)], "band": band,
                    "band_share": float(np.mean((log_cost >= band[0]) & (log_cost <= band[1])))}
    runs = out.setdefault("runs", {})
    for seed in TAIL_SEEDS:
        if str(seed) in runs:
            continue
        train, test = tail_split(frame, seed)
        x_train = bench._strings(train[columns].reset_index(drop=True))
        x_test = bench._strings(test[columns].reset_index(drop=True))
        cost_train, cost_test = train["S"].to_numpy(), test["S"].to_numpy()
        y_train, y_test = np.log(cost_train), np.log(cost_test)
        near = (np.log(lump) - 0.01, np.log(lump) + 0.01)
        run = {"claims_train": int(len(y_train)), "claims_test": int(len(y_test)),
               "lump_share_test": float(np.mean((y_test >= near[0]) & (y_test <= near[1]))),
               "band_share_test": float(np.mean((y_test >= band[0]) & (y_test <= band[1])))}
        for name, params in TAIL_HEADS.items():
            started = time.process_time()
            model = ASRRegressor(random_state=seed, **params).fit(x_train, y_train)
            dist = model.predict_dist(x_test)
            record = ashlar_scores(dist, y_test, seed)
            cdf = lambda v: dist.cdf(jnp.full(len(y_test), v, jnp.float32))  # noqa: E731
            record["lump_mass"] = float(np.mean(_window(cdf, *near)))
            record["band_mass"] = float(np.mean(_window(cdf, *band)))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record

        started = time.process_time()
        mean, shape = lightgbm_gamma(x_train, cost_train, x_test, seed)
        scale = mean / shape
        draws = np.log(np.maximum(np.random.default_rng(seed).gamma(shape, scale[:, None], size=(len(mean), DRAWS)), 1e-300))
        low, high = np.log(stats.gamma.ppf(0.1, shape, scale=scale)), np.log(stats.gamma.ppf(0.9, shape, scale=scale))
        run["lightgbm_gamma"] = {
            "crps": crps(draws, y_test),
            # The density of the log cost: the cost's density times the cost.
            "nll": float(-np.mean(stats.gamma.logpdf(cost_test, shape, scale=scale) + y_test)),
            "coverage80": float(np.mean((y_test >= low) & (y_test <= high))),
            "width80": float(np.mean(high - low)),
            "lump_mass": float(np.mean(stats.gamma.cdf(np.exp(near[1]), shape, scale=scale) - stats.gamma.cdf(np.exp(near[0]), shape, scale=scale))),
            "band_mass": float(np.mean(stats.gamma.cdf(np.exp(band[1]), shape, scale=scale) - stats.gamma.cdf(np.exp(band[0]), shape, scale=scale))),
            "gamma_shape": shape,
            "cpu_seconds": time.process_time() - started,
        }
        for name, fit in (("lightgbm_lognormal", lightgbm_normal), ("catboost_uncertainty", catboost_normal)):
            started = time.process_time()
            loc, scale = fit(x_train, y_train, x_test, seed)
            record = normal_scores(loc, scale, y_test, seed)
            record["lump_mass"] = float(np.mean(stats.norm.cdf(near[1], loc, scale) - stats.norm.cdf(near[0], loc, scale)))
            record["band_mass"] = float(np.mean(stats.norm.cdf(band[1], loc, scale) - stats.norm.cdf(band[0], loc, scale)))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record
        runs[str(seed)] = run
        print("tail", seed, json.dumps({k: round(v["nll"], 4) for k, v in run.items() if isinstance(v, dict)}), flush=True)
        _save(results)


# ---------------------------------------------------------------- stage: ceiling


CEILING_HEADS = {
    "mixture": {"task": "mixture"},
    "flow": {"task": "flow"},
    "histogram": {"task": "histogram", "head_kwargs": {"bins": FARE_BINS}},
}
RICH_INCOME = 7.0  # median income in $10,000s: the block groups where the cap binds most


def stage_ceiling(results):
    """California median house values, which the census table caps at $500,001."""
    out = results.setdefault("ceiling", {})
    x, y = bench.california()
    cap = float(np.max(y))
    at_cap = y >= cap - 1e-6
    rich = (x["median_income"] >= RICH_INCOME).to_numpy()
    out["table"] = {"rows": int(len(y)), "cap": cap, "cap_share": float(np.mean(at_cap)),
                    "rich_rows": int(rich.sum()), "rich_cap_share": float(np.mean(at_cap[rich]))}
    runs = out.setdefault("runs", {})
    for seed in SPIKE_SEEDS:
        if str(seed) in runs:
            continue
        index_train, index_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        x_train, x_test = x.iloc[index_train], x.iloc[index_test]
        y_train, y_test = y[index_train], y[index_test]
        top, on_rich = y_test >= cap - 1e-6, rich[index_test]
        run = {"rich_test_rows": int(on_rich.sum()), "rich_test_cap_share": float(np.mean(top[on_rich]))}

        def cap_scores(cdf, draws):
            # P(value within $10k of the cap or above), against whether it was at the cap; and the mass
            # past the cap by more than $10k, which no block group can have.
            near = 1.0 - np.asarray(cdf(cap - 0.1), np.float64)
            past = 1.0 - np.asarray(cdf(cap + 0.1), np.float64)
            return {
                "brier_cap": float(np.mean((near - top) ** 2)),
                "past_cap": float(np.mean(past)),
                "rich_past_cap": float(np.mean(past[on_rich])),
                "rich_brier_cap": float(np.mean((near[on_rich] - top[on_rich]) ** 2)),
                "rich_crps": crps(draws[on_rich], y_test[on_rich]),
            }

        for name, params in CEILING_HEADS.items():
            started = time.process_time()
            model = ASRRegressor(random_state=seed, **params).fit(x_train, y_train)
            dist = model.predict_dist(x_test)
            record = ashlar_scores(dist, y_test, seed)
            draws = np.asarray(dist.sample(jax.random.PRNGKey(seed + 1), (DRAWS,)), np.float64).T
            record.update(cap_scores(lambda v: dist.cdf(jnp.full(len(y_test), v, jnp.float32)), draws))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record
        for name, fit in (("lightgbm_normal", lightgbm_normal), ("catboost_uncertainty", catboost_normal)):
            started = time.process_time()
            loc, scale = fit(x_train, y_train, x_test, seed)
            record = normal_scores(loc, scale, y_test, seed)
            draws = loc[:, None] + scale[:, None] * np.random.default_rng(seed + 1).standard_normal((len(loc), DRAWS))
            record.update(cap_scores(lambda v: stats.norm.cdf(v, loc, scale), draws))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record
        runs[str(seed)] = run
        print("ceiling", seed, json.dumps({k: (round(v["crps"], 4), round(v["past_cap"], 4)) for k, v in run.items() if isinstance(v, dict)}), flush=True)
        _save(results)


# ---------------------------------------------------------------- stage: skew


def stage_skew(results):
    """Diamond prices in $1,000 on the 12,000-row sample of the distribution benchmark: skewed, never below 0."""
    out = results.setdefault("skew", {})
    x, y = bench._sample(*bench.diamonds(log=False), 12_000)
    out["table"] = {"rows": int(len(y))}
    runs = out.setdefault("runs", {})
    for seed in SPIKE_SEEDS:
        if str(seed) in runs:
            continue
        index_train, index_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        x_train, x_test = x.iloc[index_train], x.iloc[index_test]
        y_train, y_test = y[index_train], y[index_test]
        cheap = y_test <= np.quantile(y_train, 0.25)
        run = {}
        for name, params in (("mixture", {"task": "mixture"}), ("flow", {"task": "flow"})):
            started = time.process_time()
            model = ASRRegressor(random_state=seed, **params).fit(x_train, y_train)
            dist = model.predict_dist(x_test)
            record = ashlar_scores(dist, y_test, seed)
            record["below_zero"] = float(np.mean(np.asarray(dist.cdf(jnp.zeros(len(y_test))), np.float64)[cheap]))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record
        for name, fit in (("lightgbm_normal", lightgbm_normal), ("catboost_uncertainty", catboost_normal)):
            started = time.process_time()
            loc, scale = fit(x_train, y_train, x_test, seed)
            record = normal_scores(loc, scale, y_test, seed)
            record["below_zero"] = float(np.mean(stats.norm.cdf(0.0, loc, scale)[cheap]))
            record["cpu_seconds"] = time.process_time() - started
            run[name] = record
        runs[str(seed)] = run
        print("skew", seed, json.dumps({k: (round(v["crps"], 4), round(v["below_zero"], 4)) for k, v in run.items()}), flush=True)
        _save(results)


# ---------------------------------------------------------------- stage: counts


def stage_counts(results):
    """Ashlar's ``negative_binomial`` and LightGBM's Poisson again, and a negative binomial around LightGBM's mean."""
    from benchmark_uncertainty import COUNT_TABLES, _count_scores

    out = results.setdefault("counts", {})
    for name in ("bike_count", "taxi_pickups"):
        x, y = COUNT_TABLES[name]()
        runs = out.setdefault(name, {}).setdefault("runs", {})
        for seed in COUNT_SEEDS:
            if str(seed) in runs:
                continue
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
            run = {}
            started = time.process_time()
            model = ASRRegressor(task="negative_binomial", random_state=seed).fit(x_train, y_train)
            draws = model.sample(x_test, n=COUNT_DRAWS, seed=seed).T
            run["negative_binomial"] = _count_scores(draws, model.predict(x_test), y_test, model.log_prob(x_test, y_test))
            run["negative_binomial"]["cpu_seconds"] = time.process_time() - started

            started = time.process_time()
            gbm, frame = bench._fit_gbt("lightgbm", "regression", True, seed, x_train, y_train, objective="poisson")
            mean = np.maximum(gbm.predict(bench._align_categories(frame, x_test)), 1e-9)
            draws = np.random.default_rng(seed).poisson(mean[:, None], size=(len(mean), COUNT_DRAWS)).astype(float)
            run["lightgbm_poisson"] = _count_scores(draws, mean, y_test, stats.poisson.logpmf(y_test, mean))
            run["lightgbm_poisson"]["cpu_seconds"] = time.process_time() - started

            # The tree user's fix: one dispersion around the Poisson mean, fitted on the validation slice.
            _fit_x, val_x, _fit_y, val_y = bench._gbt_split(frame, y_train, seed)
            dispersion = _nb_dispersion(val_y, gbm.predict(val_x))
            size = 1.0 / dispersion
            p = size / (size + mean)
            draws = np.random.default_rng(seed).negative_binomial(size, p[:, None], size=(len(mean), COUNT_DRAWS)).astype(float)
            run["lightgbm_poisson_nb"] = _count_scores(draws, mean, y_test, stats.nbinom.logpmf(y_test, size, p))
            run["lightgbm_poisson_nb"]["dispersion"] = dispersion
            runs[str(seed)] = run
            print("counts", name, seed, json.dumps({k: (round(v["nll"], 3), round(v["coverage80"], 3)) for k, v in run.items()}), flush=True)
            _save(results)


# ---------------------------------------------------------------- report


SPIKE_LABELS = {
    "histogram": "Ashlar `histogram`, 48 bins",
    "mixture": "Ashlar `mixture`",
    "flow": "Ashlar `flow`",
    "lightgbm_normal": "LightGBM mean + constant normal",
    "catboost_uncertainty": "CatBoost `RMSEWithUncertainty`",
}
ZERO_LABELS = {
    "hurdle_gamma": "Ashlar `hurdle`, gamma",
    "hurdle_lognormal": "Ashlar `hurdle`, lognormal",
    "mixture": "Ashlar `mixture`",
    "lightgbm_hurdle_gamma": "LightGBM classifier + gamma regressor",
    "lightgbm_hurdle_lognormal": "LightGBM classifier + log-tip regressor",
    "lightgbm_normal": "LightGBM mean + constant normal",
}
TAIL_LABELS = {
    "mixture": "Ashlar `mixture`",
    "gaussian": "Ashlar `gaussian`",
    "lightgbm_gamma": "LightGBM, gamma objective on the cost",
    "lightgbm_lognormal": "LightGBM on the log cost + constant normal",
    "catboost_uncertainty": "CatBoost `RMSEWithUncertainty` on the log cost",
}


def _column(runs, model, key):
    return [run[model][key] for run in runs.values() if model in run and key in run[model]]


def report_spike(results):
    block = results.get("spike")
    if not block:
        return
    runs = block["runs"]
    table = block["table"]
    print(f"\n### spike: {table['rows']} trips, {table['flat_trips']} between JFK and Manhattan, "
          f"{table['flat_share_at_52']:.0%} of them at $52; {len(runs)} splits, mean ± sd")
    print("| Model | CRPS, $ ↓ | NLL ↓ | 80% interval holds | Width, $ | P(within $0.50 of $52) | P(within $2 of $52) | CRPS on JFK trips, $ ↓ |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model, label in SPIKE_LABELS.items():
        cells = [
            _mean_sd(_column(runs, model, "crps"), 3),
            _mean_sd(_column(runs, model, "nll"), 3),
            f"{np.mean(_column(runs, model, 'coverage80')):.1%}",
            f"{np.mean(_column(runs, model, 'width80')):.2f}",
            f"{np.mean(_column(runs, model, 'within_050')):.1%}",
            f"{np.mean(_column(runs, model, 'within_2')):.1%}",
            _mean_sd(_column(runs, model, "flat_crps"), 2),
        ]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("JFK test trips per split:", [run["flat_test_trips"] for run in runs.values()])


def report_zero(results):
    block = results.get("zero")
    if not block:
        return
    runs = block["runs"]
    print(f"\n### zero: {block['table']['rows']} trips, {block['table']['zero_share']:.1%} no tip; {len(runs)} splits, mean ± standard error")
    print("| Model | CRPS, $ ↓ | Brier of P(no tip) ↓ | NLL, mass at $0 and density above ↓ | 80% interval holds | Width, $ |")
    print("|---|---:|---:|---:|---:|---:|")
    for model, label in ZERO_LABELS.items():
        nll = _column(runs, model, "mixed_nll")
        cells = [
            _mean_sd(_column(runs, model, "crps"), 3, "se"),
            _mean_sd(_column(runs, model, "brier_zero"), 3, "se"),
            _mean_sd(nll, 3, "se") if nll else "—",
            f"{np.mean(_column(runs, model, 'coverage80')):.1%}",
            f"{np.mean(_column(runs, model, 'width80')):.2f}",
        ]
        print(f"| {label} | " + " | ".join(cells) + " |")


def report_tail(results):
    block = results.get("tail")
    if not block:
        return
    runs = block["runs"]
    table = block["table"]
    print(f"\n### tail: {table['claims']} claims; most common amount €{table['lump']:,.2f}, "
          f"{table['lump_share']:.1%} of claims within 1% of it; the four most common, {table.get('common')}, "
          f"{table.get('band_share', float('nan')):.1%} of claims in their band; {len(runs)} splits, mean ± sd")
    print("| Model | NLL of the log cost ↓ | CRPS of the log cost ↓ | 80% interval holds | Width | Mass within 1% of the common amount | Mass in the band of the four |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for model, label in TAIL_LABELS.items():
        cells = [
            _mean_sd(_column(runs, model, "nll"), 4),
            _mean_sd(_column(runs, model, "crps"), 4),
            f"{np.mean(_column(runs, model, 'coverage80')):.1%}",
            f"{np.mean(_column(runs, model, 'width80')):.3f}",
            f"{np.mean(_column(runs, model, 'lump_mass')):.1%}",
            f"{np.mean(_column(runs, model, 'band_mass')):.1%}" if _column(runs, model, "band_mass") else "—",
        ]
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("lump share in test:", [round(run["lump_share_test"], 4) for run in runs.values()],
          "band share in test:", [round(run.get("band_share_test", float("nan")), 4) for run in runs.values()])


CEILING_LABELS = {
    "mixture": "Ashlar `mixture`",
    "flow": "Ashlar `flow`",
    "histogram": "Ashlar `histogram`, 48 bins",
    "lightgbm_normal": "LightGBM mean + constant normal",
    "catboost_uncertainty": "CatBoost `RMSEWithUncertainty`",
}
SKEW_LABELS = {
    "mixture": "Ashlar `mixture`",
    "flow": "Ashlar `flow`",
    "lightgbm_normal": "LightGBM mean + constant normal",
    "catboost_uncertainty": "CatBoost `RMSEWithUncertainty`",
}
COUNT_LABELS = {
    "negative_binomial": "Ashlar `negative_binomial`",
    "lightgbm_poisson": "LightGBM, Poisson objective",
    "lightgbm_poisson_nb": "LightGBM Poisson mean + one negative binomial dispersion",
}


def report_ceiling(results):
    block = results.get("ceiling")
    if not block:
        return
    runs, table = block["runs"], block["table"]
    print(f"\n### ceiling: {table['rows']} block groups, {table['cap_share']:.1%} at the cap; income ≥ $70k: "
          f"{table['rich_rows']} rows, {table['rich_cap_share']:.1%} at the cap; {len(runs)} splits, mean ± sd")
    print("| Model | CRPS ↓ | NLL ↓ | 80% holds | Width | Brier of P(at the cap) ↓ | Mass past the cap | Income ≥ $70k: CRPS ↓ | Brier ↓ | Mass past the cap |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model, label in CEILING_LABELS.items():
        cells = [
            _mean_sd(_column(runs, model, "crps"), 4),
            _mean_sd(_column(runs, model, "nll"), 3),
            f"{np.mean(_column(runs, model, 'coverage80')):.1%}",
            f"{np.mean(_column(runs, model, 'width80')):.3f}",
            _mean_sd(_column(runs, model, "brier_cap"), 4),
            f"{np.mean(_column(runs, model, 'past_cap')):.2%}",
            _mean_sd(_column(runs, model, "rich_crps"), 3),
            _mean_sd(_column(runs, model, "rich_brier_cap"), 3),
            f"{np.mean(_column(runs, model, 'rich_past_cap')):.1%}",
        ]
        print(f"| {label} | " + " | ".join(cells) + " |")


def report_skew(results):
    block = results.get("skew")
    if not block:
        return
    runs = block["runs"]
    print(f"\n### skew: diamonds, {block['table']['rows']} rows, price in $1,000; {len(runs)} splits, mean ± sd")
    print("| Model | CRPS ↓ | NLL ↓ | 80% holds | Width | P(price < 0), cheapest quarter |")
    print("|---|---:|---:|---:|---:|---:|")
    for model, label in SKEW_LABELS.items():
        cells = [
            _mean_sd(_column(runs, model, "crps"), 4),
            _mean_sd(_column(runs, model, "nll"), 3),
            f"{np.mean(_column(runs, model, 'coverage80')):.1%}",
            f"{np.mean(_column(runs, model, 'width80')):.3f}",
            f"{np.mean(_column(runs, model, 'below_zero')):.1%}",
        ]
        print(f"| {label} | " + " | ".join(cells) + " |")


def report_counts(results):
    block = results.get("counts")
    if not block:
        return
    print("\n### counts, mean ± sd over splits")
    print("| Target | Model | CRPS ↓ | NLL ↓ | 80% coverage | 80% width |")
    print("|---|---|---:|---:|---:|---:|")
    for name, table in block.items():
        runs = table["runs"]
        for model, label in COUNT_LABELS.items():
            rows = [run[model] for run in runs.values() if model in run]
            if not rows:
                continue
            extra = f" (dispersion {np.mean([r['dispersion'] for r in rows]):.3g})" if "dispersion" in rows[0] else ""
            print(f"| {name} ({len(rows)}) | {label}{extra} | {_mean_sd([r['crps'] for r in rows], 2)} | "
                  f"{_mean_sd([r['nll'] for r in rows], 2)} | {np.mean([r['coverage80'] for r in rows]):.1%} | "
                  f"{np.mean([r['width80'] for r in rows]):,.0f} |")


def stage_report(results):
    report_spike(results)
    report_zero(results)
    report_tail(results)
    report_ceiling(results)
    report_skew(results)
    report_counts(results)


# ---------------------------------------------------------------- io


def _out_path(stage: str) -> Path:
    return DATA_DIR / f"shapes-{stage}.json"


def _save(results):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for stage, value in results.items():
        _out_path(stage).write_text(json.dumps({stage: value}, indent=1, sort_keys=True))


def load(stages=("spike", "zero", "tail", "ceiling", "skew", "counts")) -> dict:
    results = {}
    for stage in stages:
        path = _out_path(stage)
        if path.exists():
            results.update(json.loads(path.read_text()))
    return results


STAGES = {
    "spike": stage_spike,
    "zero": stage_zero,
    "tail": stage_tail,
    "ceiling": stage_ceiling,
    "skew": stage_skew,
    "counts": stage_counts,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=[*STAGES, "all", "report"], default="all")
    args = parser.parse_args()
    results = load()
    if args.stage == "report":
        stage_report(results)
        return
    for name, stage in STAGES.items():
        if args.stage in {name, "all"}:
            stage(results)
    stage_report(results)


if __name__ == "__main__":
    main()
