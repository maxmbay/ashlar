"""Casual and registered riders, one model: does modelling their dependence change the total's interval?

Capital Bikeshare's hourly rentals (17,379 hours, 2011 and 2012, from UCI)
record two outcomes every hour: riders without a subscription (``casual``)
and subscribers (``registered``). A dock operator asks about both and about
their total, the bikes out that hour. Weather and the calendar move both
groups, and they share the shocks the features do not record, so their
errors are correlated and do not cancel in the sum.

Every Ashlar model is ``ASRRegressor`` on ``Y = log1p([casual, registered])``
with the calendar and the weather as features, as the count benchmark has
them (``examples/benchmark_uncertainty.py``):

- ``joint``: ``task="multivariate_gaussian"``, a normal with a full 2 × 2
  covariance that changes from hour to hour.
- ``joint marginals, independent``: the same fitted model with its
  dependence dropped (``Independent(dist.marginals())``), so each group's own
  forecast is identical to ``joint``'s and only the dependence differs.
- ``independent heads``: the same task with ``head_kwargs={"rank": 0}``,
  trained with no dependence.
- ``independent + static correlation``: that rank-0 model with one
  correlation estimated from its standardized residuals on the validation
  rows, a Gaussian copula: the usual fix after the fact.
- ``joint mixture``: ``task="multivariate_mixture"``, five joint normals.

And what a gradient-boosting user would build for the total:

- ``LightGBM quantiles of the total``: LightGBM quantile regression at 0.1
  and 0.9 fitted to the total itself, the direct route to its interval. It
  answers that one question; each further question is another model.
- ``LightGBM quantiles per group, summed``: LightGBM quantile regression at
  0.1 and 0.9 for each group, and the total's interval read as the sum of
  the groups' bounds, the route from two models of the groups.

Every Ashlar model early-stops on the same random 20% of the training rows,
which also give the copula its correlation. Each LightGBM model trains at
learning rate 0.05 with up to 4,000 trees, early-stopped after 100 rounds on
a random 20% of the training rows.

Scores on the test hours (20%, three random splits, seeds 0 to 2): the joint
NLL of the two log counts; and for the total ``casual + registered``, the
coverage and mean width of its central 80% interval and its CRPS, read from
4,000 joint draws per hour, mapped back to counts.

Run with::

    JAX_PLATFORMS=cpu python examples/benchmark_joint_riders.py run       # about 30 minutes on four cores
    python examples/benchmark_joint_riders.py report
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import DATA_DIR, RAW, SOURCES, _path, _strings  # noqa: E402
from joint_demand import _crps, _static_correlation, _with_correlation  # noqa: E402

from ashlar import ASRRegressor  # noqa: E402
from ashlar.distributions import Independent  # noqa: E402

SOURCES.setdefault(
    "bike_hour.csv", f"{RAW}/udacity/deep-learning-v2-pytorch/master/project-bikesharing/Bike-Sharing-Dataset/hour.csv"
)
RESULTS = DATA_DIR / "joint-riders.json"
SEEDS = (0, 1, 2)
DRAWS = 4000
GROUPS = ("casual", "registered")
ASHLAR = {
    "joint": {"task": "multivariate_gaussian"},
    "independent heads": {"task": "multivariate_gaussian", "head_kwargs": {"rank": 0}},
    "joint mixture": {"task": "multivariate_mixture"},
}
MODELS = (
    "joint",
    "joint marginals, independent",
    "independent heads",
    "independent + static correlation",
    "joint mixture",
    "LightGBM quantiles of the total",
    "LightGBM quantiles per group, summed",
)


def riders():
    """The features, and ``(hours, 2)`` riders: casual, then registered."""
    import pandas as pd

    frame = pd.read_csv(_path("bike_hour.csv"))
    counts = frame[list(GROUPS)].to_numpy(np.float64)
    features = frame.drop(columns=["instant", "dteday", "casual", "registered", "cnt"])
    for column in ("season", "weathersit"):
        features[column] = features[column].astype(str)
    return _strings(features), counts


def total_draws(dist, rows: int, key) -> np.ndarray:
    """``(DRAWS, rows)`` draws of the total riders, from joint draws of the two log counts."""
    pieces = []
    for start in range(0, rows, 512):
        draws = np.asarray(dist[start : start + 512].sample(jax.random.fold_in(key, start), (DRAWS,)), dtype=np.float64)
        pieces.append(np.expm1(np.maximum(draws, 0.0)).sum(axis=-1))
    return np.concatenate(pieces, axis=1)


def interval_scores(low, high, observed) -> dict:
    return {
        "total_coverage_80": float(np.mean((observed >= low) & (observed <= high))),
        "total_width_80": float(np.mean(high - low)),
    }


def draw_scores(dist, y_log, observed, key) -> dict:
    total = total_draws(dist, len(observed), key)
    low, high = np.quantile(total, [0.1, 0.9], axis=0)
    record = {
        "joint_nll": float(-np.mean(np.asarray(dist.log_prob(jnp.asarray(y_log, dtype=jnp.float32))))),
        "total_crps": float(_crps(total, observed).mean()),
    }
    record.update(interval_scores(low, high, observed))
    return record


def _lightgbm_quantile(x_train, y_train, level: float, seed: int):
    import warnings

    import lightgbm
    import pandas as pd

    warnings.filterwarnings("ignore", message=".*eval_set.*deprecated", category=Warning)
    frame = x_train.copy()
    for column in ("season", "weathersit"):
        frame[column] = pd.Categorical(frame[column], categories=sorted(x_train[column].unique()))
    fit_x, val_x, fit_y, val_y = train_test_split(frame, y_train, test_size=0.2, random_state=seed)
    model = lightgbm.LGBMRegressor(objective="quantile", alpha=level, n_estimators=4000, learning_rate=0.05,
                                   random_state=seed, n_jobs=4, verbose=-1)
    model.fit(fit_x, fit_y, eval_set=[(val_x, val_y)], callbacks=[lightgbm.early_stopping(100, verbose=False)])

    def predict(x):
        x = x.copy()
        for column in ("season", "weathersit"):
            x[column] = pd.Categorical(x[column], categories=frame[column].cat.categories)
        return model.predict(x)

    return predict


def run() -> None:
    x, counts = riders()
    y = np.log1p(counts)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for seed in SEEDS:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
        observed = np.expm1(y_test).sum(axis=1)
        key = jax.random.PRNGKey(seed)
        record, seconds = {}, {}
        fitted = {}
        # Every model early-stops on the same random 20% of the training rows, which also fit the copula.
        x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
        for name, settings in ASHLAR.items():
            start = time.process_time()
            fitted[name] = ASRRegressor(random_state=seed, **settings).fit(x_fit, y_fit, validation=(x_val, y_val))
            seconds[name] = time.process_time() - start
        loose = fitted["independent heads"]
        correlation = _static_correlation(loose, x_val, y_val)
        dists = {
            "joint": fitted["joint"].predict_dist(x_test),
            "independent heads": loose.predict_dist(x_test),
            "joint mixture": fitted["joint mixture"].predict_dist(x_test),
        }
        dists["joint marginals, independent"] = Independent(dists["joint"].marginals())
        dists["independent + static correlation"] = _with_correlation(dists["independent heads"], correlation)
        for name in MODELS[:5]:
            record[name] = draw_scores(dists[name], y_test, observed, key)
            record[name]["cpu"] = seconds.get(name, seconds.get("joint" if "joint" in name else "independent heads"))
        start = time.process_time()
        total_train = np.expm1(y_train).sum(axis=1)
        low, high = (_lightgbm_quantile(x_train, total_train, level, seed)(x_test) for level in (0.1, 0.9))
        record["LightGBM quantiles of the total"] = interval_scores(low, high, observed)
        record["LightGBM quantiles of the total"]["cpu"] = time.process_time() - start
        start = time.process_time()
        bounds = np.zeros((2, len(observed)))
        for column in range(2):
            for row, level in enumerate((0.1, 0.9)):
                bounds[row] += _lightgbm_quantile(x_train, np.expm1(y_train[:, column]), level, seed)(x_test)
        record["LightGBM quantiles per group, summed"] = interval_scores(bounds[0], bounds[1], observed)
        record["LightGBM quantiles per group, summed"]["cpu"] = time.process_time() - start
        record["static correlation"] = float(correlation[0, 1])
        record["test hours"] = int(len(observed))
        for name, scores in record.items():
            print(seed, name, json.dumps(scores), flush=True)
        results[str(seed)] = record
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))


def report() -> None:
    results = json.loads(RESULTS.read_text())
    splits = [results[str(seed)] for seed in SEEDS if str(seed) in results]
    print(f"\nCapital Bikeshare riders, {len(splits)} splits, mean ± standard deviation\n")
    print("| Model | Joint NLL ↓ | Total inside its 80% interval | Width, riders | CRPS of the total ↓ |")
    print("|---|---:|---:|---:|---:|")

    def cell(rows, key, form, scale=1.0):
        values = np.array([row[key] for row in rows if key in row], dtype=float) * scale
        return "" if values.size == 0 else f"{form.format(values.mean())} ± {form.format(values.std())}"

    for name in MODELS:
        rows = [split[name] for split in splits]
        print(f"| {name} | {cell(rows, 'joint_nll', '{:.3f}')} | {cell(rows, 'total_coverage_80', '{:.1f}', 100.0)}% | "
              f"{cell(rows, 'total_width_80', '{:.0f}')} | {cell(rows, 'total_crps', '{:.2f}')} |")
    print(f"\nStatic residual correlation: {', '.join(f'{split['static correlation']:.2f}' for split in splits)}")


STAGES = {"run": run, "report": report}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
