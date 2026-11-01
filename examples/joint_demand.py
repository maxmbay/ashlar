"""Joint demand across locations: does modelling dependence change P(total > capacity)?

Eight electricity clients from the Lai et al. load matrix (321 Portuguese
clients, hourly, 2012 to 2014), the eight whose mean hourly load is between
10,000 and 30,000 kW, so no one client dominates their total. Each row is
one hour; the target is the eight loads at that hour, and the features are
the calendar and each client's load 24 hours and a week earlier, so every
forecast is a day ahead.

Four predictive distributions for the eight loads, fitted on everything
before a forecast origin and scored on the four weeks after it:

- ``joint``: ``ASRRegressor(task="multivariate_gaussian")``, a joint normal
  with a learned rank-2 covariance that changes from hour to hour.
- ``joint marginals, independent``: the same fitted model, with each
  client's marginal kept exactly and the dependence dropped
  (``Independent(dist.marginals())``).
- ``independent heads``: the same task with ``rank=0``, trained without any
  dependence.
- ``independent + static correlation``: that rank-0 model, with one
  correlation matrix estimated from its standardized residuals on the
  validation slice (a Gaussian copula). The usual fix after the fact.

Scores on the held-out weeks: the joint negative log density of the eight
loads; the CRPS of each client's own forecast (the marginals); and, for the
total load, its CRPS, the coverage of its 80% interval, and the Brier score
and log loss of P(total > capacity) at two capacities, the 80th and 95th
percentiles of the total over the 28 days before the origin, the way an
operator would set them from recent load. Rolling origins every 28 days from
day 392 (late January 2013) to the end of 2014, 25 of them, each fitted on
everything before it and scored on the 28 days after it; one fit per model
and origin.

The ``calibrated`` stage asks whether conformal calibration of the total
fixes what the joint models get wrong about it. At each origin it fits on
the training weeks but the last 15%, calibrates on those
(``model.calibrate(X_cal, Y_cal)``, which covers each client and the sum),
and scores the same model before and after on the same 28 days: the total's
80% interval, P(total > capacity), and the interval's coverage by hour of
day. ``shard=i/n`` runs every n-th origin from the i-th, so several
processes can share a machine; ``calibrated_report`` reads every shard.

    JAX_PLATFORMS=cpu python examples/joint_demand.py            # fit and score
    JAX_PLATFORMS=cpu python examples/joint_demand.py extra      # rank 4, TabM, and flow joint models
    JAX_PLATFORMS=cpu python examples/joint_demand.py report     # tables
    JAX_PLATFORMS=cpu python examples/joint_demand.py figure     # docs/images/joint-demand.png
    JAX_PLATFORMS=cpu python examples/joint_demand.py calibrated shard=0/3   # fit, calibrate, and score
    JAX_PLATFORMS=cpu python examples/joint_demand.py calibrated_report
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from ashlar import ASRRegressor
from ashlar.distributions import (
    Independent,
    LowRankMultivariateNormal,
)

DATA_DIR = Path("/tmp/ashlar-bench")
SOURCE = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz"
RESULTS = DATA_DIR / "joint-demand.json"
LOAD_RANGE = (10_000.0, 30_000.0)
TEST_DAYS = 28
ORIGIN_DAYS = tuple(range(392, 1096 - TEST_DAYS + 1, TEST_DAYS))
CAPACITY_LEVELS = (0.8, 0.95)
DRAWS = 4000
FIT = {"max_epochs": 60, "patience": 8, "split": "temporal", "validation_size": 0.15}
METHODS = ("joint", "joint marginals, independent", "independent heads", "independent + static correlation")
# Added after the first four had run, to ask whether a richer joint model closes the gap to the static correlation.
# The flow came later still, with the multivariate_flow task.
EXTRA = {
    "joint, rank 4": {"head_kwargs": {"rank": 4}},
    "joint, TabM": {"backbone": "tabm"},
    "joint flow": {"task": "multivariate_flow"},
}
# The calibrated stage: every model fitted on the training weeks but the last CALIBRATION_SHARE, calibrated on those.
CALIBRATION_SHARE = 0.15
CALIBRATED = {
    "joint": {},
    "independent heads": {"head_kwargs": {"rank": 0}},
    "joint, TabM": {"backbone": "tabm"},
    "joint flow": {"task": "multivariate_flow"},
}
CALIBRATED_RESULTS = DATA_DIR / "joint-demand-calibrated-{shard}.json"


def load_matrix() -> np.ndarray:
    cache = DATA_DIR / "electricity.npy"
    if not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        archive = DATA_DIR / "electricity.txt.gz"
        if not archive.exists():
            urllib.request.urlretrieve(SOURCE, archive)
        np.save(cache, np.loadtxt(archive, delimiter=","))
    return np.load(cache)


def clients(matrix: np.ndarray) -> np.ndarray:
    """Columns with under 1% zero hours and a mean load inside ``LOAD_RANGE``, largest first."""
    mean = matrix.mean(axis=0)
    keep = ((matrix == 0).mean(axis=0) < 0.01) & (mean >= LOAD_RANGE[0]) & (mean <= LOAD_RANGE[1])
    chosen = np.flatnonzero(keep)
    return chosen[np.argsort(-mean[chosen])]


def table(matrix: np.ndarray, columns: np.ndarray):
    """Features and the (hours, clients) target. Hour 0 is 2012-01-01 00:00."""
    loads = matrix[:, columns].astype(np.float64)
    hours = np.arange(len(loads))
    start = 168
    t = hours[start:]
    frame = {
        "hour": (t % 24).astype(np.float64),
        "weekday": ((t // 24 + 6) % 7).astype(np.float64),  # 2012-01-01 was a Sunday
        "day_of_year": ((t // 24) % 365.25).astype(np.float64),
    }
    for index in range(loads.shape[1]):
        frame[f"day_before_{index}"] = loads[t - 24, index]
        frame[f"week_before_{index}"] = loads[t - 168, index]
    return frame, loads[start:], t // 24


def _rows(frame: dict, mask: np.ndarray) -> dict:
    return {name: values[mask] for name, values in frame.items()}


def _static_correlation(model, x_val, y_val) -> np.ndarray:
    dist = model.predict_dist(x_val)
    residual = (y_val - np.asarray(dist.mean())) / np.asarray(dist.stddev())
    return np.corrcoef(residual.T)


def _with_correlation(dist, correlation: np.ndarray):
    """The independent model's marginal normals, joined by one fixed correlation matrix.

    ``S R S`` is split as ``S (R - λI) S + λ S²`` with ``λ`` half the smallest
    eigenvalue of ``R``, so the low-rank form keeps a well-conditioned diagonal.
    """
    floor = 0.5 * float(np.linalg.eigvalsh(correlation).min())
    shrunk = np.linalg.cholesky(correlation - floor * np.eye(len(correlation)))
    sd = jnp.asarray(dist.stddev())
    factor = sd[..., :, None] * jnp.asarray(shrunk, dtype=jnp.float32)
    return LowRankMultivariateNormal(dist.mean(), factor, floor * jnp.square(sd))


def _crps(draws: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS per row, ``draws`` (n, rows)."""
    first = np.mean(np.abs(draws - y[None, :]), axis=0)
    ordered = np.sort(draws, axis=0)
    n = ordered.shape[0]
    weights = (2 * np.arange(1, n + 1) - n - 1)[:, None]
    second = np.sum(weights * ordered, axis=0) / (n * n)
    return first - second


def _scores(name, dist, y, capacities, key) -> dict:
    # (draws, rows, clients), 128 rows at a time so a flow's draws stay in memory.
    draws = np.concatenate(
        [
            np.asarray(dist[start : start + 128].sample(jax.random.fold_in(key, start), (DRAWS,)), dtype=np.float64)
            for start in range(0, y.shape[0], 128)
        ],
        axis=1,
    )
    total = draws.sum(axis=-1)
    observed = y.sum(axis=-1)
    low, high = np.quantile(total, [0.1, 0.9], axis=0)
    record = {
        "joint_nll": float(-np.mean(np.asarray(dist.log_prob(jnp.asarray(y, dtype=jnp.float32))))),
        "marginal_crps": float(np.mean([_crps(draws[..., j], y[:, j]).mean() for j in range(y.shape[1])])),
        "total_crps": float(_crps(total, observed).mean()),
        "total_coverage_80": float(np.mean((observed >= low) & (observed <= high))),
        "total_width_80": float(np.mean(high - low)),
    }
    for level, capacity in capacities.items():
        p = np.mean(total > capacity, axis=0)
        event = (observed > capacity).astype(np.float64)
        clipped = np.clip(p, 1.0 / DRAWS, 1.0 - 1.0 / DRAWS)
        record[f"brier_{level}"] = float(np.mean((p - event) ** 2))
        record[f"logloss_{level}"] = float(-np.mean(event * np.log(clipped) + (1 - event) * np.log1p(-clipped)))
        record[f"rate_{level}"] = float(event.mean())
        record[f"mean_p_{level}"] = float(p.mean())
        record[f"p_{level}"] = p.round(4).tolist()
        record[f"event_{level}"] = event.astype(int).tolist()
    print(f"  {name:34s} " + " ".join(f"{k}={v:.4g}" for k, v in record.items() if not isinstance(v, list)), flush=True)
    return record


def run() -> None:
    matrix = load_matrix()
    columns = clients(matrix)
    frame, loads, day = table(matrix, columns)
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    results["clients"] = [int(c) for c in columns]
    for split, origin in enumerate(ORIGIN_DAYS):
        if str(origin) in results:
            continue
        train = day < origin
        test = (day >= origin) & (day < origin + TEST_DAYS)
        x_train, y_train = _rows(frame, train), loads[train]
        x_test, y_test = _rows(frame, test), loads[test]
        recent = y_train[-24 * 28 :].sum(axis=1)
        capacities = {str(level): float(np.quantile(recent, level)) for level in CAPACITY_LEVELS}
        cyclical = {"hour": 24.0, "weekday": 7.0, "day_of_year": 365.25}
        print(f"origin day {origin}: {train.sum()} training hours, {test.sum()} test hours", flush=True)
        started = time.perf_counter()
        joint = ASRRegressor(task="multivariate_gaussian", cyclical=cyclical, random_state=split, **FIT).fit(x_train, y_train)
        joint_seconds = time.perf_counter() - started
        loose = ASRRegressor(
            task="multivariate_gaussian", head_kwargs={"rank": 0}, cyclical=cyclical, random_state=split, **FIT
        ).fit(x_train, y_train)
        # The validation slice the fit held out at the end of the training period.
        n_val = round(FIT["validation_size"] * train.sum())
        x_val = {name: values[-n_val:] for name, values in x_train.items()}
        correlation = _static_correlation(loose, x_val, y_train[-n_val:])
        dist = joint.predict_dist(x_test)
        loose_dist = loose.predict_dist(x_test)
        key = jax.random.PRNGKey(split)
        record = {
            "train_hours": int(train.sum()),
            "test_hours": int(test.sum()),
            "capacities": capacities,
            "joint_fit_seconds": joint_seconds,
            "static_correlation_mean": float((correlation.sum() - len(correlation)) / (len(correlation) ** 2 - len(correlation))),
            "methods": {
                "joint": _scores("joint", dist, y_test, capacities, key),
                "joint marginals, independent": _scores(
                    "joint marginals, independent", Independent(dist.marginals()), y_test, capacities, key
                ),
                "independent heads": _scores("independent heads", loose_dist, y_test, capacities, key),
                "independent + static correlation": _scores(
                    "independent + static correlation", _with_correlation(loose_dist, correlation), y_test, capacities, key
                ),
            },
        }
        week = slice(0, 24 * 7)
        record["first_week"] = {
            "total": y_test[week].sum(axis=1).round(1).tolist(),
            "bands": {
                name: np.quantile(
                    np.asarray(d[week].sample(key, (DRAWS,))).sum(axis=-1), [0.1, 0.5, 0.9], axis=0
                ).round(1).tolist()
                for name, d in (("joint", dist), ("joint marginals, independent", Independent(dist.marginals())))
            },
        }
        results[str(origin)] = record
        RESULTS.write_text(json.dumps(results))


def run_extra() -> None:
    """The ``EXTRA`` joint models, scored on the same origins, draws, and capacities."""
    matrix = load_matrix()
    frame, loads, day = table(matrix, clients(matrix))
    results = json.loads(RESULTS.read_text())
    cyclical = {"hour": 24.0, "weekday": 7.0, "day_of_year": 365.25}
    for split, origin in enumerate(ORIGIN_DAYS):
        record = results[str(origin)]
        train = day < origin
        test = (day >= origin) & (day < origin + TEST_DAYS)
        for name, params in EXTRA.items():
            if name in record["methods"]:
                continue
            model = ASRRegressor(
                **{"task": "multivariate_gaussian", "cyclical": cyclical, "random_state": split, **FIT, **params}
            ).fit(_rows(frame, train), loads[train])
            print(f"origin day {origin}", flush=True)
            record["methods"][name] = _scores(
                name, model.predict_dist(_rows(frame, test)), loads[test], record["capacities"], jax.random.PRNGKey(split)
            )
            RESULTS.write_text(json.dumps(results))


def _splits(results):
    return [results[str(origin)] for origin in ORIGIN_DAYS if str(origin) in results]


def report() -> None:
    results = json.loads(RESULTS.read_text())
    splits = _splits(results)
    print(f"{len(splits)} forecast origins, clients {results['clients']}")
    metrics = [
        ("joint_nll", "Joint NLL, eight loads ↓"),
        ("marginal_crps", "CRPS per client, kW ↓"),
        ("total_crps", "CRPS of the total, kW ↓"),
        ("total_coverage_80", "Total in its 80% interval"),
        ("total_width_80", "Width of that interval, kW"),
        ("brier_0.8", "Brier, total > 80th pct ↓"),
        ("logloss_0.8", "Log loss, total > 80th pct ↓"),
        ("brier_0.95", "Brier, total > 95th pct ↓"),
        ("logloss_0.95", "Log loss, total > 95th pct ↓"),
    ]
    methods = [m for m in (*METHODS, *EXTRA) if all(m in split["methods"] for split in splits)]
    print("| Score | " + " | ".join(methods) + " |")
    print("|---|" + "---:|" * len(methods))
    for key, label in metrics:
        cells = []
        for method in methods:
            values = np.asarray([split["methods"][method][key] for split in splits])
            cells.append(f"{values.mean():.4g} ± {values.std(ddof=1) / np.sqrt(len(values)):.2g}" if len(values) > 1 else f"{values.mean():.4g}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    wins = {
        key: sum(split["methods"]["joint"][key] < split["methods"]["joint marginals, independent"][key] for split in splits)
        for key in ("joint_nll", "total_crps", "brier_0.8", "logloss_0.8", "brier_0.95", "logloss_0.95")
    }
    print("joint beats its own marginals made independent, splits won of", len(splits), wins)
    for level in ("0.8", "0.95"):
        rate = np.mean([split["methods"]["joint"][f"rate_{level}"] for split in splits])
        line = ", ".join(
            f"{method} {np.mean([split['methods'][method][f'mean_p_{level}'] for split in splits]):.4f}" for method in methods
        )
        print(f"capacity {level}: observed rate {rate:.4f}; mean predicted P: {line}")
    for method in methods:
        p = np.concatenate([split["methods"][method]["p_0.8"] for split in splits])
        event = np.concatenate([split["methods"][method]["event_0.8"] for split in splits])
        edges = np.array([0, 0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001])
        bins = np.digitize(p, edges) - 1
        rows = [f"[{edges[b]:.2f},{edges[b + 1]:.2f}) n={np.sum(bins == b)} p={p[bins == b].mean():.3f} obs={event[bins == b].mean():.3f}" for b in range(len(edges) - 1) if np.any(bins == b)]
        print(method, "reliability at 0.8:", "; ".join(rows))


def figure() -> None:
    import matplotlib.pyplot as plt

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import figures as style

    results = json.loads(RESULTS.read_text())
    splits = _splits(results)
    split = splits[0]
    week = split["first_week"]
    plt.rcParams.update({"font.size": 11})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.8, 5.4), gridspec_kw={"width_ratios": [1.7, 1]})
    hours = np.arange(72)
    capacity = split["capacities"]["0.8"]
    low, _, high = (np.asarray(v)[:72] / 1000 for v in week["bands"]["joint"])
    left.fill_between(hours, low, high, color=style.TEAL, alpha=0.25, lw=0, label="Joint, rank-2 covariance: 80% band")
    low, _, high = (np.asarray(v)[:72] / 1000 for v in week["bands"]["joint marginals, independent"])
    left.plot(hours, low, color=style.CLAY, lw=1.1, ls="--", label="Same marginals, independent: 80% band")
    left.plot(hours, high, color=style.CLAY, lw=1.1, ls="--")
    left.plot(hours, np.asarray(week["total"])[:72] / 1000, color=style.INK, lw=1.4, label="Observed total")
    left.axhline(capacity / 1000, color=style.MUTED, lw=1, ls=":")
    left.text(0, capacity / 1000 + 1, "capacity: 80th percentile of the 28 days before", color=style.MUTED, va="bottom", fontsize=9)
    left.set_xlabel("Hour of the first three held-out days")
    left.set_ylabel("Total load, MW")
    left.set_ylim(top=left.get_ylim()[1] + 18)
    left.legend(frameon=False, fontsize=9, loc="upper right")
    edges = np.array([0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0001])
    for method, color, label in (
        ("joint marginals, independent", style.CLAY, "Same marginals, independent"),
        ("independent + static correlation", style.MUTED, "Independent + static correlation"),
        ("joint", style.TEAL, "Joint, rank-2 covariance"),
    ):
        p = np.concatenate([s["methods"][method]["p_0.8"] for s in splits])
        event = np.concatenate([s["methods"][method]["event_0.8"] for s in splits])
        bins = np.digitize(p, edges) - 1
        xs = [p[bins == b].mean() for b in range(len(edges) - 1) if np.sum(bins == b) >= 20]
        ys = [event[bins == b].mean() for b in range(len(edges) - 1) if np.sum(bins == b) >= 20]
        right.plot(xs, ys, "o-", color=color, lw=1.4, ms=4, label=label)
    right.plot([0, 1], [0, 1], color=style.HAIR, lw=1)
    right.set_xlabel("Predicted P(total > capacity)")
    right.set_ylabel("Observed share of hours over capacity")
    right.set_xlim(0, 1)
    right.set_ylim(0, 1)
    right.legend(frameon=False, fontsize=9, loc="upper left")
    for ax in (left, right):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle(
        f"Eight electricity clients, a day ahead. Bands: the first three held-out days. Reliability: {sum(s['test_hours'] for s in splits):,} held-out hours, {len(splits)} origins.",
        fontsize=11,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "images" / "joint-demand.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print("wrote", out)


def _calibration_split(day: np.ndarray, origin: int):
    """Fit on the training weeks but the last ``CALIBRATION_SHARE``, calibrate on those, test on the 28 days after."""
    held = 7 * int(round(CALIBRATION_SHARE * origin / 7))
    fit = day < origin - held
    calibration = (day >= origin - held) & (day < origin)
    test = (day >= origin) & (day < origin + TEST_DAYS)
    return fit, calibration, test


def _event_scores(p: np.ndarray, event: np.ndarray) -> dict:
    clipped = np.clip(p, 1.0 / DRAWS, 1.0 - 1.0 / DRAWS)
    return {
        "brier": float(np.mean((p - event) ** 2)),
        "logloss": float(-np.mean(event * np.log(clipped) + (1 - event) * np.log1p(-clipped))),
        "mean_p": float(p.mean()),
    }


def _calibrated_scores(model, x_test, y_test, hour, capacities, calibrated: bool) -> dict:
    """The total's 80% interval, overall, by hour, and per client, and P(total > capacity), before or after calibration."""
    total = y_test.sum(axis=1)
    interval = model.predict_interval(x_test, 0.8, target="sum", n_samples=DRAWS, calibrated=calibrated)
    inside = (total >= interval[:, 0]) & (total <= interval[:, 1])
    columns = model.predict_interval(x_test, 0.8, n_samples=DRAWS, calibrated=calibrated)
    record = {
        "total_coverage_80": float(inside.mean()),
        "total_width_80": float(np.mean(interval[:, 1] - interval[:, 0])),
        "coverage_by_hour": [float(inside[hour == h].mean()) for h in range(24)],
        "client_coverage_80": np.mean((y_test >= columns[..., 0]) & (y_test <= columns[..., 1]), axis=0).round(4).tolist(),
    }
    for level, capacity in capacities.items():
        p = model.predict_exceedance(x_test, capacity, target="sum", n_samples=DRAWS, calibrated=calibrated)
        event = (total > capacity).astype(np.float64)
        record.update({f"{key}_{level}": value for key, value in _event_scores(p, event).items()})
        record[f"rate_{level}"] = float(event.mean())
        if level == "0.8":
            record["p_0.8"] = p.round(4).tolist()
            record["event_0.8"] = event.astype(int).tolist()
    return record


def _shard(arguments) -> tuple[int, int]:
    for argument in arguments:
        if argument.startswith("shard="):
            index, count = argument.split("=", 1)[1].split("/")
            return int(index), int(count)
    return 0, 1


def run_calibrated() -> None:
    """Fit on the training weeks but the last 15%, calibrate on those, and score before and after."""
    index, count = _shard(sys.argv[2:])
    matrix = load_matrix()
    columns = clients(matrix)
    frame, loads, day = table(matrix, columns)
    path = Path(str(CALIBRATED_RESULTS).format(shard=f"{index}of{count}"))
    results = json.loads(path.read_text()) if path.exists() else {}
    cyclical = {"hour": 24.0, "weekday": 7.0, "day_of_year": 365.25}
    for split, origin in enumerate(ORIGIN_DAYS):
        if split % count != index:
            continue
        record = results.setdefault(str(origin), {"methods": {}})
        fit, held, test = _calibration_split(day, origin)
        recent = loads[day < origin][-24 * 28 :].sum(axis=1)
        capacities = {str(level): float(np.quantile(recent, level)) for level in CAPACITY_LEVELS}
        record.update({"fit_hours": int(fit.sum()), "calibration_hours": int(held.sum()), "test_hours": int(test.sum()), "capacities": capacities})
        x_fit, y_fit = _rows(frame, fit), loads[fit]
        x_cal, y_cal = _rows(frame, held), loads[held]
        x_test, y_test = _rows(frame, test), loads[test]
        hour = np.asarray(x_test["hour"]).astype(int)
        for name, params in CALIBRATED.items():
            if name in record["methods"]:
                continue
            print(f"origin day {origin}: {name}, {fit.sum()} fit hours, {held.sum()} calibration hours", flush=True)
            started = time.perf_counter()
            model = ASRRegressor(
                **{"task": "multivariate_gaussian", "cyclical": cyclical, "random_state": split, **FIT, **params}
            ).fit(x_fit, y_fit)
            fitted = time.perf_counter()
            model.calibrate(x_cal, y_cal, n_samples=DRAWS)
            calibrated = time.perf_counter()
            dist = model.predict_dist(x_test)
            scores = _scores(name, dist, y_test, capacities, jax.random.PRNGKey(split))
            record["methods"][name] = {
                "fit_seconds": fitted - started,
                "calibrate_seconds": calibrated - fitted,
                "joint_nll": scores["joint_nll"],
                "total_crps": scores["total_crps"],
                "before": _calibrated_scores(model, x_test, y_test, hour, capacities, calibrated=False),
                "after": _calibrated_scores(model, x_test, y_test, hour, capacities, calibrated=True),
            }
            before, after = record["methods"][name]["before"], record["methods"][name]["after"]
            print(
                f"  {name:20s} total 80%: {before['total_coverage_80']:.3f} -> {after['total_coverage_80']:.3f}; "
                f"brier 0.8: {before['brier_0.8']:.4f} -> {after['brier_0.8']:.4f}",
                flush=True,
            )
            path.write_text(json.dumps(results))


def calibrated_report() -> None:
    results = {}
    for path in sorted(DATA_DIR.glob("joint-demand-calibrated-*.json")):
        results.update(json.loads(path.read_text()))
    splits = [results[str(origin)] for origin in ORIGIN_DAYS if str(origin) in results]
    methods = [name for name in CALIBRATED if all(name in split["methods"] for split in splits)]
    print(f"{len(splits)} forecast origins; methods {methods}")
    rows = [
        ("total_coverage_80", "Total inside its 80% interval", "{:.1%}"),
        ("total_width_80", "Width of that interval, kW", "{:,.0f}"),
        ("brier_0.8", "Brier, total > 80th percentile ↓", "{:.4f}"),
        ("logloss_0.8", "Log loss, total > 80th percentile ↓", "{:.3f}"),
        ("brier_0.95", "Brier, total > 95th percentile ↓", "{:.4f}"),
        ("logloss_0.95", "Log loss, total > 95th percentile ↓", "{:.3f}"),
    ]
    print("| Score | " + " | ".join(methods) + " |")
    print("|---|" + "---:|" * len(methods))
    for key, label, fmt in rows:
        cells = []
        for method in methods:
            before = np.mean([split["methods"][method]["before"][key] for split in splits])
            after = np.mean([split["methods"][method]["after"][key] for split in splits])
            cells.append(f"{fmt.format(before)} → {fmt.format(after)}")
        print(f"| {label} | " + " | ".join(cells) + " |")
    for key, label in (("joint_nll", "Joint NLL ↓"), ("total_crps", "CRPS of the total, kW ↓")):
        values = [np.mean([split["methods"][method][key] for split in splits]) for method in methods]
        print(f"| {label} | " + " | ".join(f"{value:.4g}" for value in values) + " |")
    for method in methods:
        for stage in ("before", "after"):
            by_hour = np.mean([split["methods"][method][stage]["coverage_by_hour"] for split in splits], axis=0)
            clients_80 = np.mean([split["methods"][method][stage]["client_coverage_80"] for split in splits], axis=0)
            print(
                f"{method} {stage}: total coverage by hour {by_hour.min():.1%} to {by_hour.max():.1%} "
                f"(hours outside 75-85%: {int(np.sum(np.abs(by_hour - 0.8) > 0.05))}); "
                f"clients {clients_80.min():.1%} to {clients_80.max():.1%}"
            )
        for key in ("brier_0.8", "logloss_0.8", "brier_0.95", "logloss_0.95"):
            won = sum(split["methods"][method]["after"][key] < split["methods"][method]["before"][key] for split in splits)
            print(f"  calibration lowers {key} at {won} of {len(splits)} origins")
        p = np.concatenate([split["methods"][method]["after"]["p_0.8"] for split in splits])
        raw = np.concatenate([split["methods"][method]["before"]["p_0.8"] for split in splits])
        event = np.concatenate([split["methods"][method]["after"]["event_0.8"] for split in splits])
        for name, values in (("before", raw), ("after", p)):
            band = (values >= 0.02) & (values < 0.10)
            print(f"  {name}: where P(total > 80th pct) is 2-10% (n={band.sum()}), mean {values[band].mean():.3f}, happened {event[band].mean():.3f}")
        seconds = np.mean([split["methods"][method]["fit_seconds"] for split in splits])
        calibrate_seconds = np.mean([split["methods"][method]["calibrate_seconds"] for split in splits])
        print(f"  fit {seconds:.0f} s, calibrate {calibrate_seconds:.1f} s per origin")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    {
        "run": run,
        "extra": run_extra,
        "report": report,
        "figure": figure,
        "calibrated": run_calibrated,
        "calibrated_report": calibrated_report,
    }[command]()
