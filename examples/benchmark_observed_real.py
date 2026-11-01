"""Labels that say what was observed, on real data: survival times, sold-out shelves, and totals.

``benchmark_observed.py`` simulates its data so the truth is known. This
script asks the same questions of real tables, with the censoring or the
grouping applied to values that were recorded exactly, so every fit can
still be scored against them.

**Survival** (``survival``). Three clinical tables as pycox ships them, from
the DeepSurv repository: SUPPORT (8,873 seriously ill hospital patients, 14
features, 68% died during follow-up), METABRIC (1,904 breast cancer
patients, 9 features, 58%), and GBSG (2,232 breast cancer patients, 7
features, 57%). Five random splits, 80% to train and 20% to test, stratified
by the event. Ashlar fits the log of the time with
``observed.survival(np.log(time), event)``: the ``gaussian`` head, a
log-normal survival model whose mean and spread are both learned per row,
and the ``mixture`` head; and the ``gaussian`` fitted on 85% of the training
rows and calibrated (``calibrate``) on the other 15%, which reads its
survival curve from ``predict_exceedance``. The baselines are XGBoost's
accelerated failure time model, whose log time is a tree ensemble's
location plus one spread for every row times a normal, logistic, or extreme
value error, the family, depth (3 or 6), and spread (0.5 to 4) chosen by
likelihood on the same 15%, each fit early stopped there; scikit-survival's
Cox model (Efron's ties, a ridge of 0.001 on standardized features) with
Breslow's baseline; and Kaplan-Meier, which ignores the features. Scores on
the test rows:

- Concordance: Harrell's C of the predicted median time (the Cox model's
  risk score), the share of comparable pairs put in the right order.
- Integrated Brier score: the Brier score of the survival curve, weighted
  by the inverse probability of censoring (Graf et al., 1999) from the
  Kaplan-Meier estimate of the test rows' censoring, as pycox does, and
  averaged over 40 times between the 10th and 90th percentiles of the test
  times.
- D-calibration (Haider et al., 2020): a calibrated model's ``S(T)`` is
  uniform, and a row censored at ``c`` spreads its weight uniformly over
  ``[0, S(c)]``. An event where a step curve (Cox, Kaplan-Meier, pycox)
  drops spreads its weight over the drop, since SUPPORT's times are whole
  days and many are tied. The chi-square statistic of ten equal bins, and in
  how many splits its p-value passes 0.05.
- NLL: the negative log-likelihood of the test labels on the log-time
  scale, the density of the log time for an event and ``P(log T >= log c)``
  for a censored row. Only the models with a density have one, and the
  calibrated fit has none of its own: ``calibrate`` remaps its survival
  curve, not its density.
- CPU and wall-clock seconds to fit, the XGBoost grid included, and those
  of XGBoost's one fit at the setting the grid chose.

**Drive failures with delayed entry** (``backblaze``). Backblaze's daily
records of every hard drive in its data centers for the first quarter of
2024, one row per drive: its model (models with under 200 drives pooled as
"other"), its size, and five SMART error counts (reallocated, reported
uncorrectable, and pending sectors, command timeouts, and offline
uncorrectable sectors) on its first day in the quarter, and its age from
the power-on hours. A drive is only in the table because it was still
running when the quarter began, or was installed during it, so its failure
age is truncated at its age on entry. Ashlar fits the log of the age with
``observed.survival(log exit_age, failed, entry=log entry_age)``, with the
``gaussian`` and ``mixture`` heads. XGBoost's AFT model and pycox's
logistic hazard (Gensheimer and Narasimhan, 2019), a neural network on 30
time intervals, cannot read a truncated age, so they fit the days each
drive was watched in the quarter, with its age on entry as a feature, the
usual way around it; Kaplan-Meier on those days is the baseline. Three
random splits of the drives, 80% to train and 20% to test, 15% of the
training drives held out for XGBoost's and pycox's early stopping. Every
model is scored on what happened to the test drives during the quarter:
the survival curve over the days watched, ``S(entry + t) / S(entry)`` for
Ashlar. The same concordance (of the risk of failing within 80 days),
integrated Brier score (from 10 to 80 days), D-calibration, NLL of the days
watched (an event's density per day), and seconds as for the clinical
tables.

**Sold-out shelves** (``stockouts``). FreshRetailNet-50K (Wang et al.,
2025): 90 days of hourly sales of 50,000 store and product pairs of fresh
food, with a flag for every hour out of stock. A day with no hour out of stock
between 6:00 and 22:00 recorded its demand. On those days only, 2,000
sampled series get a simulated stock, the way ``benchmark_observed.py``
draws it: the series' mean demand on its training days, times a lognormal
forecast error (sd 0.25) and a safety factor between 1.0 and 1.5, so a day
sold out when its demand reached the stock. Days 0 to 59 train, days 60 to
69 calibrate, and days 70 to 89 test. Each head, ``mixture`` and
``hurdle``, is fitted four ways: on the true demand (the oracle), on sales as
if they were demand, on the days that did not sell out, and on
``observed.censored(sales, sold_out)``; LightGBM's quantile regression on
sales is the tree baseline. The features are the store, the product and its
three category levels, the city, the weekday, the discount, the holiday and
activity flags, the weather, and the series' mean recorded sales on its
training days. Scored on the test days against the true demand: the NLL
of the true demand's bin, 1% of the mean demand wide, since some stocked
days sold nothing and a density cannot score that atom; CRPS from 1,000
draws; the share at or below the predicted 90% quantile; the newsvendor
cost of ordering the 75% quantile (a lost sale costs 3, a leftover unit 1)
in percent above the cost of the same head's oracle fit, the mixture's for
LightGBM; and the 80% interval after ``calibrate`` on the calibration days:
on their true demand for the oracle, their censored labels for the censored
fit, and their sales for the others.

**Daily loads from totals** (``totals``). The daily load of the eight
electricity clients of ``joint_demand.py``, in MWh, 2012 to 2014. The days
are cut into four-week blocks that start on a Monday, and every fifth block
is held out, so the test days interleave the training days. The features
are the client, the weekday, the day of the year, and the time since the
start. The ``gaussian`` head is fitted on the days themselves (the oracle),
and on two kinds of totals: each calendar week's (Monday to Sunday), and
meter readings at irregular intervals of 2 to 12 days, each with
``observed.total`` and with the total spread evenly over its days. Every
week holds each weekday once, so weekly totals say nothing about how the
load splits between weekdays; irregular readings do. Scored on the held-out
days: NLL, RMSE of the mean, CRPS, the share inside the 80% interval, and
the weekday profile error: the root mean square, over clients and weekdays,
of the error in each weekday's share of the client's mean load.

Results go to ``/tmp/ashlar-bench/observed-real-<stage>.json`` after every
fit, or every split for the survival stages, and a stage restarted with the
same command skips what is done; a Backblaze split missing a model, such as
pycox after it failed, fits only that model.

    pip install -e ".[dev]" xgboost scikit-survival h5py lightgbm pyarrow torch pycox
    JAX_PLATFORMS=cpu python examples/benchmark_observed_real.py survival     # SUPPORT, METABRIC, GBSG
    JAX_PLATFORMS=cpu python examples/benchmark_observed_real.py backblaze    # downloads about 1 GB
    JAX_PLATFORMS=cpu python examples/benchmark_observed_real.py stockouts    # downloads FreshRetailNet-50K
    JAX_PLATFORMS=cpu python examples/benchmark_observed_real.py totals
    JAX_PLATFORMS=cpu python examples/benchmark_observed_real.py report

``seeds=0,1`` runs fewer splits, and ``sets=GBSG,METABRIC`` fewer survival
tables.
"""

from __future__ import annotations

import importlib.util
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
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ashlar import ASRRegressor, observed

DATA_DIR = Path("/tmp/ashlar-bench")
RESULTS = DATA_DIR / "observed-real-{stage}.json"
SEEDS = (0, 1, 2, 3, 4)
DRAWS = 1000


def _arguments() -> dict:
    return dict(argument.split("=", 1) for argument in sys.argv[2:] if "=" in argument)


def _seeds() -> tuple[int, ...]:
    chosen = _arguments().get("seeds")
    return SEEDS if chosen is None else tuple(int(seed) for seed in chosen.split(","))


def _load(stage: str) -> dict:
    path = Path(str(RESULTS).format(stage=stage))
    return json.loads(path.read_text()) if path.exists() else {}


def _save(stage: str, results: dict) -> None:
    path = Path(str(RESULTS).format(stage=stage))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=1))
    temporary.replace(path)


class _Clock:
    """CPU seconds of every thread of the process, and wall-clock seconds."""

    def __enter__(self):
        self.cpu, self.wall = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.cpu, self.wall = time.process_time() - self.cpu, time.perf_counter() - self.wall


def _crps(draws: np.ndarray, y: np.ndarray) -> float:
    """Sample CRPS, ``E|X - y| - E|X - X'| / 2``, the second term from sorted draws."""
    draws = np.sort(draws, axis=1)
    n = draws.shape[1]
    first = np.mean(np.abs(draws - y[:, None]), axis=1)
    weights = 2.0 * np.arange(1, n + 1) - n - 1
    second = np.sum(weights * draws, axis=1) / (n * n)
    return float(np.mean(first - second))


# Survival -----------------------------------------------------------------

SURVIVAL_SOURCE = "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/master/experiments/data/"
SURVIVAL_SETS = {
    "SUPPORT": "support/support_train_test.h5",
    "METABRIC": "metabric/metabric_IHC4_clinical_train_test.h5",
    "GBSG": "gbsg/gbsg_cancer_train_test.h5",
}
SURVIVAL_MODELS = (
    "ashlar gaussian",
    "ashlar mixture",
    "ashlar gaussian, calibrated",
    "xgboost aft",
    "cox",
    "kaplan-meier",
)
GRID_POINTS = 40
D_BINS = 10
AFT_FAMILIES = {"normal": stats.norm, "logistic": stats.logistic, "extreme": stats.gumbel_l}
AFT_GRID = tuple(
    (family, depth, scale)
    for family in AFT_FAMILIES
    for depth in (3, 6)
    for scale in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
)


def survival_set(name: str):
    """Features, times, and events of one table, its train and test files together as pycox reads them."""
    import h5py

    path = DATA_DIR / Path(SURVIVAL_SETS[name]).name
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SURVIVAL_SOURCE + SURVIVAL_SETS[name], path)
    with h5py.File(path, "r") as source:
        x = np.concatenate([source[split]["x"][:] for split in ("train", "test")]).astype(np.float32)
        t = np.concatenate([source[split]["t"][:] for split in ("train", "test")]).astype(np.float64)
        e = np.concatenate([source[split]["e"][:] for split in ("train", "test")]).astype(bool)
    # One METABRIC time is zero; the log needs it positive.
    return x, np.maximum(t, 0.5 * t[t > 0].min()), e


def concordance(time_, event, risk) -> float:
    """Harrell's C: of the pairs whose earlier time is an event, the share the risk puts in that order."""
    from sksurv.metrics import concordance_index_censored

    return float(concordance_index_censored(event, time_, risk)[0])


def integrated_brier(time_, event, curves, grid) -> float:
    """The Brier score of the survival curves on ``grid``, censoring weighted as in Graf et al. (1999).

    The censoring weights come from the Kaplan-Meier estimate of the test
    rows' own censoring, as pycox computes them, so a test time past the
    last training time cannot leave the weights undefined.
    """
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv

    observed_ = Surv.from_arrays(event, time_)
    return float(integrated_brier_score(observed_, observed_, curves, grid))


def d_calibration(at_time, event, bins: int = D_BINS, before=None) -> tuple[float, float]:
    """Haider et al. (2020): the chi-square statistic and p-value of ``S(T)`` against uniform in equal bins.

    An event puts its weight in the bin of ``S(T)``. A row censored at ``c``
    has ``S(T)`` uniform on ``[0, S(c)]`` if the model is right, so it
    spreads its weight over that range. A step curve that drops at an
    event's time gives ``before``, ``S(T-)``, and the event spreads its
    weight over the drop, ``[S(T), S(T-)]``, as a count's PIT is spread over
    its jump; read only after the drop, tied times would pile into low bins.
    """
    s = np.clip(np.asarray(at_time, dtype=np.float64), 1e-12, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    top = s if before is None else np.clip(np.asarray(before, dtype=np.float64), 1e-12, 1.0)
    drop = event & (top - s > 1e-12)
    point = event & ~drop
    counts = np.bincount(np.minimum((s[point] * bins).astype(int), bins - 1), minlength=bins).astype(np.float64)
    low, high = s[drop][:, None], top[drop][:, None]
    counts += np.sum(np.clip(np.minimum(edges[1:], high) - np.maximum(edges[:-1], low), 0.0, None) / (high - low), axis=0)
    censored = s[~event][:, None]
    counts += np.sum(np.clip(np.minimum(edges[1:], censored) - edges[:-1], 0.0, None) / censored, axis=0)
    expected = len(s) / bins
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    return chi2, float(stats.chi2.sf(chi2, bins - 1))


def _step_grid(times, survival, grid) -> np.ndarray:
    """Right-continuous survival steps ``(rows, times)`` read at shared ``grid`` times, 1 before the first step."""
    index = np.searchsorted(times, grid, side="right") - 1
    return np.where(index[None, :] >= 0, survival[:, np.maximum(index, 0)], 1.0)


def _step_own(times, survival, own) -> tuple[np.ndarray, np.ndarray]:
    """The same steps, each row read at its own time: ``S(T)`` after a drop at ``T``, and ``S(T-)`` before it."""
    rows = np.arange(len(own))

    def read(index):
        return np.where(index >= 0, survival[rows, np.maximum(index, 0)], 1.0)

    return read(np.searchsorted(times, own, side="right") - 1), read(np.searchsorted(times, own, side="left") - 1)


def _ashlar_scores(model, x, log_time, event, log_grid, calibrated: bool) -> dict:
    """Survival curves on the grid, ``S(T)`` at each row's own time, the median, and the NLL."""
    if calibrated:
        curves = np.stack([model.predict_exceedance(x, float(point)) for point in log_grid], axis=1)
        own = model.predict_exceedance(x, log_time)
        median = model.predict_quantiles(x, [0.5])[:, 0]
    else:
        dist = model.predict_dist(x)
        rows = len(log_time)
        curves = np.stack(
            [np.exp(np.asarray(dist.log_survival(jnp.full((rows,), point, jnp.float32)), np.float64)) for point in log_grid],
            axis=1,
        )
        own = np.exp(np.asarray(dist.log_survival(jnp.asarray(log_time, jnp.float32)), np.float64))
        median = np.asarray(dist.median(), np.float64)
    # Calibration remaps the curve; log_prob is still the head's own density.
    nll = None if calibrated else float(-np.mean(model.log_prob(x, observed.survival(log_time, event))))
    return {"curves": curves, "own": own, "risk": -median, "nll": nll}


def _aft(x_fit, t_fit, e_fit, x_val, t_val, e_val, seed: int):
    """XGBoost's AFT model, its error family, depth, and spread chosen by validation likelihood, each early stopped.

    Returns the chosen booster, family, and spread, and the seconds of that one fit, to read beside the grid's.
    """
    import xgboost

    def matrix(x, t, e):
        data = xgboost.DMatrix(x)
        data.set_float_info("label_lower_bound", t)
        data.set_float_info("label_upper_bound", np.where(e, t, np.inf))
        return data

    fit, validation = matrix(x_fit, t_fit, e_fit), matrix(x_val, t_val, e_val)
    best = None
    for family, depth, scale in AFT_GRID:
        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": family,
            "aft_loss_distribution_scale": scale,
            "tree_method": "hist",
            "learning_rate": 0.05,
            "max_depth": depth,
            "seed": seed,
        }
        with _Clock() as clock:
            booster = xgboost.train(
                params, fit, num_boost_round=2000, evals=[(validation, "validation")], early_stopping_rounds=50,
                verbose_eval=False,
            )
        if best is None or booster.best_score < best[0]:
            best = (booster.best_score, booster, family, scale, {"cpu": clock.cpu, "wall": clock.wall})
    return best[1:]


def _aft_scores(booster, family, scale, x, log_time, event, log_grid) -> dict:
    """The log time is ``loc + scale * W`` with ``W`` from the family: normal, logistic, or minimum extreme value."""
    import xgboost

    error = AFT_FAMILIES[family]
    loc = booster.predict(xgboost.DMatrix(x), output_margin=True, iteration_range=(0, booster.best_iteration + 1))
    loc = np.asarray(loc, np.float64)
    z = (log_time - loc) / scale
    log_likelihood = np.where(event, error.logpdf(z) - np.log(scale), error.logsf(z))
    return {
        "curves": error.sf((log_grid[None, :] - loc[:, None]) / scale),
        "own": error.sf(z),
        "risk": -(loc + scale * error.median()),
        "nll": float(-np.mean(log_likelihood)),
        "family": family,
        "scale": scale,
    }


def _survival_split(x, t, e, train, test, seed: int) -> dict:
    """Every model on one split: its scores and the seconds its fit took."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sksurv.linear_model import CoxPHSurvivalAnalysis
    from sksurv.nonparametric import kaplan_meier_estimator
    from sksurv.util import Surv

    fit, held = train_test_split(train, test_size=0.15, stratify=e[train], random_state=seed)
    log_t = np.log(t)
    grid = np.unique(np.quantile(t[test], np.linspace(0.1, 0.9, GRID_POINTS)))
    # Many rows censored at one administrative time can put the 90th percentile on the last time.
    grid = grid[grid < t[test].max()]
    log_grid = np.log(grid)
    records = {}
    for name in SURVIVAL_MODELS:
        with _Clock() as clock:
            if name in ("ashlar gaussian", "ashlar mixture"):
                task = name.split()[1]
                model = ASRRegressor(task=task, random_state=seed).fit(x[train], observed.survival(log_t[train], e[train]))
            elif name == "ashlar gaussian, calibrated":
                model = ASRRegressor(task="gaussian", random_state=seed).fit(x[fit], observed.survival(log_t[fit], e[fit]))
                model.calibrate(x[held], observed.survival(log_t[held], e[held]))
            elif name == "xgboost aft":
                booster, family, scale, single = _aft(x[fit], t[fit], e[fit], x[held], t[held], e[held], seed)
            elif name == "cox":
                scaler = StandardScaler().fit(x[train])
                cox = CoxPHSurvivalAnalysis(alpha=1e-3, ties="efron").fit(
                    scaler.transform(x[train]), Surv.from_arrays(e[train], t[train])
                )
            else:
                km_times, km_survival = kaplan_meier_estimator(e[train], t[train])
        if name.startswith("ashlar"):
            scores = _ashlar_scores(model, x[test], log_t[test], e[test], log_grid, calibrated=name.endswith("calibrated"))
        elif name == "xgboost aft":
            scores = _aft_scores(booster, family, scale, x[test], log_t[test], e[test], log_grid)
        elif name == "cox":
            x_test = scaler.transform(x[test])
            survival = cox.predict_survival_function(x_test, return_array=True)
            times = cox.unique_times_
            scores = {
                "curves": _step_grid(times, survival, grid),
                "own": _step_own(times, survival, t[test]),
                "risk": cox.predict(x_test),
                "nll": None,
            }
        else:
            rows = len(test)
            curve = np.broadcast_to(km_survival, (rows, len(km_survival)))
            scores = {
                "curves": _step_grid(km_times, curve, grid),
                "own": _step_own(km_times, curve, t[test]),
                "risk": np.zeros(rows),
                "nll": None,
            }
        # A step curve gives S(T) after and before its drop at T.
        after, before = scores["own"] if isinstance(scores["own"], tuple) else (scores["own"], None)
        chi2, p = d_calibration(after, e[test], before=before)
        records[name] = {
            "c_index": concordance(t[test], e[test], scores["risk"]),
            "ibs": integrated_brier(t[test], e[test], scores["curves"], grid),
            "d_chi2": chi2,
            "d_p": p,
            "nll": scores["nll"],
            "cpu": clock.cpu,
            "wall": clock.wall,
        }
        if "family" in scores:
            records[name]["chosen"] = f"{scores['family']}, scale {scores['scale']}"
            records[name]["chosen_cpu"], records[name]["chosen_wall"] = single["cpu"], single["wall"]
        print(f"    {name:28s}", {key: value if not isinstance(value, float) else round(value, 4) for key, value in records[name].items()}, flush=True)
    return records


def run_survival() -> None:
    from sklearn.model_selection import train_test_split

    results = _load("survival")
    chosen = _arguments().get("sets")
    names = tuple(SURVIVAL_SETS) if chosen is None else tuple(chosen.split(","))
    for name in names:
        x, t, e = survival_set(name)
        print(f"{name}: {len(t)} rows, {x.shape[1]} features, {e.mean():.0%} events", flush=True)
        for seed in _seeds():
            key = f"{name}/{seed}"
            if key in results:
                continue
            train, test = train_test_split(np.arange(len(t)), test_size=0.2, stratify=e, random_state=seed)
            print(f"  split {seed}", flush=True)
            results[key] = _survival_split(x, t, e, train, test, seed)
            _save("survival", results)


def _cell(rows, key, form, scale=1.0):
    """Mean ± standard deviation over the splits of one score, skipping a split without it."""
    values = scale * np.asarray([row[key] for row in rows if row[key] is not None], dtype=np.float64)
    if not len(values):
        return ""
    return f"{form.format(values.mean())} ± {form.format(values.std(ddof=1) if len(values) > 1 else 0.0)}"


def survival_report() -> None:
    results = _load("survival")
    if not results:
        return
    names = [name for name in SURVIVAL_SETS if any(key.startswith(f"{name}/") for key in results)]
    for name in names:
        splits = [results[key] for key in results if key.startswith(f"{name}/")]
        print(f"\n### {name}, {len(splits)} splits, mean ± standard deviation\n")
        print("| Model | Concordance ↑ | Integrated Brier ↓ | D-calibration χ² ↓ | Passes D-calibration | NLL, log time ↓ | CPU seconds | Wall seconds |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|")
        for model in SURVIVAL_MODELS:
            rows = [split[model] for split in splits if model in split]
            if not rows:
                continue
            passes = sum(row["d_p"] > 0.05 for row in rows)
            cpu = np.mean([row["cpu"] for row in rows])
            wall = np.mean([row["wall"] for row in rows])
            print(
                f"| {model} | {_cell(rows, 'c_index', '{:.3f}')} | {_cell(rows, 'ibs', '{:.4f}')} | "
                f"{_cell(rows, 'd_chi2', '{:.1f}')} | {passes} of {len(rows)} | {_cell(rows, 'nll', '{:.3f}')} | "
                f"{cpu:.1f} | {wall:.1f} |"
            )
        _aft_note([split["xgboost aft"] for split in splits if "xgboost aft" in split])


def _aft_note(rows: list) -> None:
    """Which family and spread XGBoost's grid chose, and the seconds of one fit at that choice beside the grid's."""
    if not rows:
        return
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("chosen", "?")] = counts.get(row.get("chosen", "?"), 0) + 1
    chosen = ", ".join(f"{label} in {count}" for label, count in sorted(counts.items(), key=lambda item: -item[1]))
    line = f"\nXGBoost AFT chose {chosen} of {len(rows)} splits"
    single = [row["chosen_cpu"] for row in rows if "chosen_cpu" in row]
    if single:
        line += f"; one fit at its choice took {np.mean(single):.1f} CPU seconds of the grid's {np.mean([row['cpu'] for row in rows]):.1f}"
    print(line + ".")


# Sold-out shelves ---------------------------------------------------------

FRESH_API = "https://huggingface.co/api/datasets/Dingdong-Inc/FreshRetailNet-50K/parquet/default/train"
FRESH_CATEGORIES = (
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
)
FRESH_NUMBERS = (
    "discount",
    "holiday_flag",
    "activity_flag",
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
)
FRESH_COLUMNS = FRESH_CATEGORIES + FRESH_NUMBERS + ("dt", "sale_amount", "stock_hour6_22_cnt")
SERIES = 2000
TRAIN_DAYS, CALIBRATION_DAYS = 60, 10
UNDER, OVER = 3.0, 1.0
RATIO = UNDER / (UNDER + OVER)
STOCK_HEADS = ("mixture", "hurdle")
STOCK_FITS = ("oracle: true demand", "sales as demand", "days not sold out", "censored")


def fresh_frame() -> pd.DataFrame:
    """The training split of FreshRetailNet-50K, the columns this benchmark reads, cached as one parquet file."""
    cache = DATA_DIR / "freshretailnet-50k-train.parquet"
    if not cache.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # The list of parts is kept beside them, so parts already downloaded are read without a connection.
        listing = DATA_DIR / "freshretailnet-parts.json"
        urls = json.loads(listing.read_text()) if listing.exists() else None
        if not (isinstance(urls, list) and urls and all(isinstance(url, str) for url in urls)):
            with urllib.request.urlopen(FRESH_API) as response:
                urls = json.load(response)
            listing.write_text(json.dumps(urls))
        parts = []
        for index, url in enumerate(urls):
            part = DATA_DIR / f"freshretailnet-50k-train-{index}.parquet"
            if not part.exists():
                urllib.request.urlretrieve(url, part)
            parts.append(pd.read_parquet(part, columns=list(FRESH_COLUMNS)))
        pd.concat(parts, ignore_index=True).to_parquet(cache)
    return pd.read_parquet(cache)


def fresh_days(frame: pd.DataFrame, seed: int):
    """The fully stocked days of ``SERIES`` sampled series, a simulated stock for each, and the features."""
    rng = np.random.default_rng(seed)
    day = (pd.to_datetime(frame["dt"]) - pd.to_datetime(frame["dt"]).min()).dt.days.to_numpy()
    stocked = frame["stock_hour6_22_cnt"].to_numpy() == 0
    series = frame["store_id"].astype(str) + "/" + frame["product_id"].astype(str)
    usable = pd.DataFrame({"series": series, "train": stocked & (day < TRAIN_DAYS), "test": stocked & (day >= TRAIN_DAYS + CALIBRATION_DAYS),
                           "demand": np.where(stocked & (day < TRAIN_DAYS), frame["sale_amount"].to_numpy(), np.nan)})
    summary = usable.groupby("series").agg(train=("train", "sum"), test=("test", "sum"), mean=("demand", "mean"))
    eligible = summary.index[(summary["train"] >= 20) & (summary["test"] >= 5) & (summary["mean"] > 0)].to_numpy()
    chosen = rng.choice(eligible, size=min(SERIES, len(eligible)), replace=False)
    rows = np.flatnonzero(stocked & series.isin(set(chosen)).to_numpy())
    days = frame.iloc[rows].reset_index(drop=True)
    day, key = day[rows], series.to_numpy()[rows]
    demand = days["sale_amount"].to_numpy(np.float64)
    mean = summary["mean"].reindex(key).to_numpy()
    stock = mean * np.exp(rng.normal(0.0, 0.25, size=len(rows))) * rng.uniform(1.0, 1.5, size=len(rows))
    sales = np.minimum(demand, stock)
    sold_out = demand >= stock
    features = pd.DataFrame({name: pd.Categorical(days[name].astype(str)) for name in FRESH_CATEGORIES})
    for name in FRESH_NUMBERS:
        features[name] = days[name].to_numpy(np.float32)
    features["weekday"] = pd.to_datetime(days["dt"]).dt.dayofweek.to_numpy(np.float32)
    # What the shop knows of each series: its mean recorded sales on the training days.
    recorded = pd.Series(np.where(day < TRAIN_DAYS, sales, np.nan)).groupby(key).mean()
    features["recorded_level"] = recorded.reindex(key).to_numpy(np.float32)
    return features, day, demand, sales, sold_out


def _newsvendor(order, demand) -> float:
    return float(np.mean(UNDER * np.maximum(demand - order, 0) + OVER * np.maximum(order - demand, 0)))


def run_stockouts() -> None:
    results = _load("stockouts")
    frame = fresh_frame()
    for seed in _seeds():
        features, day, demand, sales, sold_out = fresh_days(frame, seed)
        train = np.flatnonzero(day < TRAIN_DAYS)
        calibration = np.flatnonzero((day >= TRAIN_DAYS) & (day < TRAIN_DAYS + CALIBRATION_DAYS))
        test = np.flatnonzero(day >= TRAIN_DAYS + CALIBRATION_DAYS)
        print(
            f"seed {seed}: {len(train)} training, {len(calibration)} calibration and {len(test)} test days; "
            f"{sold_out.mean():.1%} sold out, losing {1 - sales.sum() / demand.sum():.1%} of demand",
            flush=True,
        )
        x_test, y_test = features.iloc[test], demand[test]
        # Some stocked days sold nothing, an atom a density cannot score, so NLL is of a bin 1% of the mean demand wide.
        half_bin = 0.005 * demand.mean()
        for head in STOCK_HEADS:
            oracle_cost = None
            for fit in STOCK_FITS:
                key = f"{seed}/{head}/{fit}"
                if key in results:
                    oracle_cost = results[f"{seed}/{head}/{STOCK_FITS[0]}"]["cost_value"]
                    continue
                if fit == STOCK_FITS[0]:
                    rows, target, cal_label = train, demand[train], demand[calibration]
                elif fit == STOCK_FITS[1]:
                    rows, target, cal_label = train, sales[train], sales[calibration]
                elif fit == STOCK_FITS[2]:
                    rows = train[~sold_out[train]]
                    target, cal_label = sales[rows], sales[calibration]
                else:
                    rows, target = train, observed.censored(sales[train], sold_out[train])
                    cal_label = observed.censored(sales[calibration], sold_out[calibration])
                with _Clock() as clock:
                    model = ASRRegressor(task=head, random_state=seed).fit(features.iloc[rows], target)
                dist = model.predict_dist(x_test)
                draws = np.asarray(dist.sample(jax.random.PRNGKey(seed), (DRAWS,)), dtype=np.float64).T
                q90, q75 = model.predict_quantiles(x_test, [0.9, RATIO]).T
                cost = _newsvendor(q75, y_test)
                if fit == STOCK_FITS[0]:
                    oracle_cost = cost
                model.calibrate(features.iloc[calibration], cal_label)
                low, high = model.predict_interval(x_test, coverage=0.8).T
                results[key] = {
                    "nll": float(-np.mean(model.log_prob(x_test, observed.interval(y_test - half_bin, y_test + half_bin)))),
                    "crps": _crps(draws, y_test),
                    "q90": float(np.mean(y_test <= q90)),
                    "cost_value": cost,
                    "cost": float(100.0 * (cost / oracle_cost - 1.0)),
                    "calibrated_80": float(np.mean((y_test >= low) & (y_test <= high))),
                    "cpu": clock.cpu,
                    "wall": clock.wall,
                }
                print(f"  {head:8s} {fit:20s}", {name: round(value, 4) for name, value in results[key].items()}, flush=True)
                _save("stockouts", results)
        key = f"{seed}/lightgbm quantile/sales as demand"
        if key in results:
            continue
        try:
            import lightgbm
        except ImportError:
            print("  lightgbm is not installed; skipping its baseline", flush=True)
            continue
        orders = {}
        with _Clock() as clock:
            for level in (0.9, RATIO):
                booster = lightgbm.LGBMRegressor(
                    objective="quantile", alpha=level, n_estimators=400, learning_rate=0.05, verbose=-1, random_state=seed
                )
                booster.fit(features.iloc[train], sales[train])
                orders[level] = booster.predict(x_test)
        oracle_cost = results[f"{seed}/mixture/{STOCK_FITS[0]}"]["cost_value"]
        results[key] = {
            "q90": float(np.mean(y_test <= orders[0.9])),
            "cost_value": _newsvendor(orders[RATIO], y_test),
            "cost": float(100.0 * (_newsvendor(orders[RATIO], y_test) / oracle_cost - 1.0)),
            "cpu": clock.cpu,
            "wall": clock.wall,
        }
        print(f"  {'lightgbm':8s} {'sales as demand':20s}", {name: round(value, 4) for name, value in results[key].items()}, flush=True)
        _save("stockouts", results)


def _table(results: dict, keys: list[tuple[str, str]], columns) -> None:
    for model, fit in keys:
        rows = [results[key] for key in results if key.split("/", 1)[1] == f"{model}/{fit}"]
        if not rows:
            continue
        cells = []
        for name, form in columns:
            values = np.asarray([row[name] for row in rows if row.get(name) is not None], dtype=np.float64)
            if not len(values):
                cells.append("")
                continue
            spread = values.std(ddof=1) if len(values) > 1 else 0.0
            if form == "seconds":
                cells.append(f"{values.mean():.0f}")
            elif form == "percent":
                cells.append(f"{values.mean():.1%} ± {spread:.1%}")
            elif form == "signed":
                cells.append(f"{values.mean():+.1f}% ± {spread:.1f}%")
            else:
                cells.append(f"{form.format(values.mean())} ± {form.format(spread)}")
        print(f"| {model} | {fit} | " + " | ".join(cells) + " |")


def stockouts_report() -> None:
    results = _load("stockouts")
    if not results:
        return
    seeds = sorted({key.split("/")[0] for key in results})
    print(f"\n### Sold-out shelves, FreshRetailNet-50K, {len(seeds)} seeds, mean ± standard deviation\n")
    columns = (
        ("nll", "{:.3f}"),
        ("crps", "{:.4f}"),
        ("q90", "percent"),
        ("cost", "signed"),
        ("calibrated_80", "percent"),
        ("cpu", "seconds"),
    )
    print("| Head | Trained on | NLL ↓ | CRPS ↓ | Below the 90% quantile | Order cost above the oracle's ↓ | Calibrated 80% holds | CPU seconds |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    keys = [(head, fit) for head in STOCK_HEADS for fit in STOCK_FITS] + [("lightgbm quantile", "sales as demand")]
    _table(results, keys, columns)


# Daily loads from totals --------------------------------------------------

BLOCK_DAYS, TEST_EVERY = 28, 5
READING_GAPS = (2, 12)
TOTAL_FITS = (
    "oracle: the days",
    "weekly totals",
    "weekly total spread evenly",
    "irregular readings",
    "irregular reading spread evenly",
)


def electricity_days() -> np.ndarray:
    """Daily load in MWh of the eight clients of ``joint_demand.py``, ``(days, clients)``; day 0 is 2012-01-01, a Sunday."""
    from joint_demand import clients, load_matrix

    matrix = load_matrix()
    hourly = matrix[:, clients(matrix)]
    return hourly.reshape(-1, 24, hourly.shape[1]).sum(axis=1) / 1000.0


def _reading_periods(days: np.ndarray, rng) -> np.ndarray:
    """A period id per day of ``days`` (sorted): runs of consecutive days cut at lengths drawn from ``READING_GAPS``."""
    ids = np.empty(len(days), np.int64)
    period, left = -1, 0
    for index, day in enumerate(days):
        if left == 0 or (index and day != days[index - 1] + 1):
            period += 1
            left = int(rng.integers(READING_GAPS[0], READING_GAPS[1] + 1))
        ids[index] = period
        left -= 1
    return ids


def _spread_evenly(values: np.ndarray, group: np.ndarray) -> np.ndarray:
    return (np.bincount(group, weights=values) / np.bincount(group))[group]


def _weekday_profile_error(mean, truth, client, weekday) -> float:
    """RMS over clients and weekdays of the error in each weekday's share of the client's load, in percent."""
    errors = []
    for index in np.unique(client):
        rows = client == index
        predicted = np.array([mean[rows & (weekday == day)].mean() for day in range(7)])
        actual = np.array([truth[rows & (weekday == day)].mean() for day in range(7)])
        errors.append(predicted / predicted.mean() - actual / actual.mean())
    return float(100.0 * np.sqrt(np.mean(np.square(errors))))


def run_totals() -> None:
    results = _load("totals")
    loads = electricity_days()
    days, clients = loads.shape
    day = np.tile(np.arange(days), clients)
    client = np.repeat(np.arange(clients), days)
    y = loads.T.reshape(-1)
    weekday = (day + 6) % 7  # 0 is Monday
    frame = pd.DataFrame(
        {
            "client": pd.Categorical([f"client {index}" for index in client]),
            "weekday": weekday.astype(np.float32),
            "day_of_year": (day % 365.25).astype(np.float32),
            "years": (day / 365.25).astype(np.float32),
        }
    )
    cyclical = {"weekday": 7.0, "day_of_year": 365.25}
    # Four-week blocks that start on a Monday; every fifth is held out, so the test days interleave the training days.
    block = (day + 6) // BLOCK_DAYS
    train = np.flatnonzero(block % TEST_EVERY != TEST_EVERY - 1)
    test = np.flatnonzero(block % TEST_EVERY == TEST_EVERY - 1)
    _, week_id = np.unique(client[train] * 1000 + (day[train] + 6) // 7, return_inverse=True)
    print(f"{len(train)} training and {len(test)} test client-days, {week_id.max() + 1} weekly totals", flush=True)
    for seed in _seeds():
        rng = np.random.default_rng(seed)
        period = np.concatenate(
            [index * 10_000 + _reading_periods(day[train][client[train] == index], rng) for index in range(clients)]
        )
        _, period_id = np.unique(period, return_inverse=True)
        labels = {
            TOTAL_FITS[0]: y[train],
            TOTAL_FITS[1]: observed.total(np.bincount(week_id, weights=y[train])[week_id], week_id),
            TOTAL_FITS[2]: _spread_evenly(y[train], week_id),
            TOTAL_FITS[3]: observed.total(np.bincount(period_id, weights=y[train])[period_id], period_id),
            TOTAL_FITS[4]: _spread_evenly(y[train], period_id),
        }
        for fit, label in labels.items():
            key = f"{seed}/gaussian/{fit}"
            if key in results:
                continue
            with _Clock() as clock:
                model = ASRRegressor(task="gaussian", cyclical=cyclical, random_state=seed).fit(frame.iloc[train], label)
            dist = model.predict_dist(frame.iloc[test])
            mean = np.asarray(dist.mean(), np.float64)
            sd = np.asarray(dist.stddev(), np.float64)
            truth = y[test]
            z = (truth - mean) / sd
            crps = sd * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - 1.0 / np.sqrt(np.pi))
            low, high = model.predict_interval(frame.iloc[test], coverage=0.8).T
            results[key] = {
                "nll": float(-np.mean(model.log_prob(frame.iloc[test], truth))),
                "rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
                "crps": float(np.mean(crps)),
                "coverage_80": float(np.mean((truth >= low) & (truth <= high))),
                "profile_error": _weekday_profile_error(mean, truth, client[test], weekday[test]),
                "sd_ratio": float(np.mean(sd) / np.sqrt(np.mean((mean - truth) ** 2))),
                "groups": int(label["group"].max() + 1) if label.dtype.names else len(label),
                "cpu": clock.cpu,
                "wall": clock.wall,
            }
            print(f"  seed {seed} {fit:32s}", {name: round(value, 4) for name, value in results[key].items()}, flush=True)
            _save("totals", results)


def totals_report() -> None:
    results = _load("totals")
    if not results:
        return
    seeds = sorted({key.split("/")[0] for key in results})
    print(f"\n### Daily loads from totals, eight electricity clients, {len(seeds)} seeds, mean ± standard deviation\n")
    columns = (
        ("nll", "{:.3f}"),
        ("rmse", "{:.1f}"),
        ("crps", "{:.1f}"),
        ("coverage_80", "percent"),
        ("profile_error", "{:.1f}"),
        ("cpu", "seconds"),
    )
    print("| Head | Trained on | NLL ↓ | RMSE, MWh ↓ | CRPS, MWh ↓ | 80% interval holds | Weekday profile error, % ↓ | CPU seconds |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    _table(results, [("gaussian", fit) for fit in TOTAL_FITS], columns)


# Drive failures with delayed entry ---------------------------------------

BACKBLAZE_SOURCE = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_{quarter}.zip"
BACKBLAZE_QUARTER = "Q1_2024"
SMART = ("smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw")
BACKBLAZE_COLUMNS = ("date", "serial_number", "model", "capacity_bytes", "failure", "smart_9_raw") + SMART
MIN_MODEL_DRIVES = 200
MAX_AGE_DAYS = 15 * 365.25
WINDOW_GRID = np.arange(10.0, 90.0, 10.0)
RISK_DAYS = 80.0
BACKBLAZE_SEEDS = (0, 1, 2)
BACKBLAZE_MODELS = ("ashlar gaussian", "ashlar mixture", "xgboost aft", "pycox logistic hazard", "kaplan-meier")


def backblaze_drives() -> pd.DataFrame:
    """One row per drive seen in the quarter: its model, size, SMART counts and age on its first day, and how it left.

    ``entry_day`` and ``last_day`` count days from the quarter's first file;
    ``failed`` is set when the drive failed on its last day. A drive that is
    still running at the end, or was taken out for another reason, is
    censored there.
    """
    import zipfile

    cache = DATA_DIR / f"backblaze-{BACKBLAZE_QUARTER}-drives.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    archive = DATA_DIR / f"backblaze-{BACKBLAZE_QUARTER}.zip"
    if not archive.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(BACKBLAZE_SOURCE.format(quarter=BACKBLAZE_QUARTER), archive)
    rows_of: dict[str, int] = {}
    entries, last, failed = [], np.zeros(5_000_000, np.int64), np.zeros(5_000_000, bool)
    start = None
    with zipfile.ZipFile(archive) as source:
        names = sorted(name for name in source.namelist() if name.endswith(".csv") and "__MACOSX" not in name)
        for name in names:
            with source.open(name) as handle:
                frame = pd.read_csv(handle, usecols=lambda column: column in BACKBLAZE_COLUMNS, low_memory=False)
            frame = frame.drop_duplicates("serial_number")
            date = pd.Timestamp(frame["date"].iloc[0])
            start = date if start is None else start
            day = (date - start).days
            before = len(rows_of)
            rows = np.fromiter(
                (rows_of.setdefault(serial, len(rows_of)) for serial in frame["serial_number"].astype(str)),
                dtype=np.int64,
                count=len(frame),
            )
            fresh = frame.iloc[np.flatnonzero(rows >= before)]
            entries.append(fresh.drop(columns=["date", "serial_number", "failure"]).assign(entry_day=day))
            last[rows] = day
            failed[rows[frame["failure"].to_numpy() == 1]] = True
            print(f"  {name}: {len(frame)} drives, {len(rows_of)} seen so far", flush=True)
    drives = pd.concat(entries, ignore_index=True)
    drives["last_day"] = last[: len(drives)]
    drives["failed"] = failed[: len(drives)]
    drives.to_parquet(cache)
    return drives


def backblaze_table(drives: pd.DataFrame):
    """Features, the age at entry and exit in days, the days watched, and the failure flag, for drives with a known age."""
    age = drives["smart_9_raw"].to_numpy(np.float64) / 24.0
    keep = np.isfinite(age) & (age >= 0) & (age <= MAX_AGE_DAYS)
    drives = drives.loc[keep].reset_index(drop=True)
    entry = np.maximum(age[keep], 1.0)
    window = (drives["last_day"] - drives["entry_day"] + 1).to_numpy(np.float64)
    counts = drives["model"].value_counts()
    model = drives["model"].where(drives["model"].map(counts) >= MIN_MODEL_DRIVES, "other")
    capacity = drives["capacity_bytes"].to_numpy(np.float64) / 1e12
    features = pd.DataFrame({"model": pd.Categorical(model.astype(str)), "capacity_tb": np.where(capacity > 0, capacity, np.nan)})
    for name in SMART:
        values = drives[name].to_numpy(np.float64) if name in drives else np.full(len(drives), np.nan)
        features[name.replace("_raw", "")] = np.log1p(np.clip(values, 0.0, None)).astype(np.float32)
    return features, entry, entry + window, window, drives["failed"].to_numpy(bool)


def _numeric(features: pd.DataFrame, entry: np.ndarray, levels: list[str]) -> np.ndarray:
    """The features as one float matrix for pycox: one column per drive model, missing counts as 0 with a flag."""
    columns = [(features["model"].astype(str).to_numpy() == level).astype(np.float32) for level in levels]
    for name in features.columns.drop("model"):
        values = features[name].to_numpy(np.float64)
        columns += [np.nan_to_num(values).astype(np.float32), np.isnan(values).astype(np.float32)]
    columns.append(np.log(entry).astype(np.float32))
    return np.stack(columns, axis=1)


def _pycox_hazard(x_fit, t_fit, e_fit, x_val, t_val, e_val, seed: int):
    """pycox's logistic hazard (Gensheimer and Narasimhan, 2019) on 30 equal intervals, early stopped on validation."""
    import torch
    import torchtuples as tt
    from pycox.models import LogisticHazard

    torch.manual_seed(seed)
    np.random.seed(seed)
    labels = LogisticHazard.label_transform(30)
    y_fit = labels.fit_transform(t_fit.astype(np.float32), e_fit.astype(np.float32))
    y_val = labels.transform(t_val.astype(np.float32), e_val.astype(np.float32))
    mean, sd = x_fit.mean(axis=0), x_fit.std(axis=0) + 1e-6
    net = tt.practical.MLPVanilla(x_fit.shape[1], [64, 64], labels.out_features, batch_norm=True, dropout=0.1)
    model = LogisticHazard(net, tt.optim.Adam(0.01), duration_index=labels.cuts)
    model.fit(
        (x_fit - mean) / sd, y_fit, batch_size=1024, epochs=100, callbacks=[tt.callbacks.EarlyStopping()],
        val_data=((x_val - mean) / sd, y_val), verbose=False,
    )
    return model, mean, sd


def _backblaze_fit(name: str, features, entry, window, failed, exit_, train, fit, held, seed: int) -> dict:
    """Fit one model: Ashlar on the truncated age, the others on the days watched with the age on entry as a feature."""
    from sksurv.nonparametric import kaplan_meier_estimator

    if name.startswith("ashlar"):
        label = observed.survival(np.log(exit_[train]), failed[train], entry=np.log(entry[train]))
        return {"model": ASRRegressor(task=name.split()[1], random_state=seed).fit(features.iloc[train], label)}
    if name == "xgboost aft":
        with_entry = features.assign(log_entry=np.log(entry).astype(np.float32))
        booster, family, scale, single = _aft(
            with_entry.iloc[fit], window[fit], failed[fit], with_entry.iloc[held], window[held], failed[held], seed
        )
        return {"booster": booster, "family": family, "scale": scale, "features": with_entry, "single": single}
    if name.startswith("pycox"):
        matrix = _numeric(features, entry, sorted(features["model"].astype(str).unique()))
        hazard, mean, sd = _pycox_hazard(matrix[fit], window[fit], failed[fit], matrix[held], window[held], failed[held], seed)
        return {"hazard": hazard, "matrix": (matrix - mean) / sd}
    times, survival = kaplan_meier_estimator(failed[train], window[train])
    return {"times": times, "survival": survival}


def _backblaze_curves(name: str, fitted: dict, features, entry, exit_, window, failed, test, grid):
    """Each test drive's survival curve over the days watched on ``grid``, at its own days, its risk, and the NLL."""
    tau, event = window[test], failed[test]
    if name.startswith("ashlar"):
        model = fitted["model"]
        dist = model.predict_dist(features.iloc[test])

        def log_survival(age):
            return np.asarray(dist.log_survival(jnp.asarray(np.log(age), jnp.float32)), np.float64)

        at_entry = log_survival(entry[test])
        curves = np.stack([np.exp(log_survival(entry[test] + point) - at_entry) for point in grid], axis=1)
        own = np.exp(log_survival(exit_[test]) - at_entry)
        risk = 1.0 - np.exp(log_survival(entry[test] + RISK_DAYS) - at_entry)
        # The label's likelihood is of the log age; an event's density in days divides by the age.
        label = observed.survival(np.log(exit_[test]), event, entry=np.log(entry[test]))
        log_likelihood = np.asarray(model.log_prob(features.iloc[test], label), np.float64) - event * np.log(exit_[test])
        return curves, own, risk, float(-np.mean(log_likelihood))
    if name == "xgboost aft":
        scores = _aft_scores(
            fitted["booster"], fitted["family"], fitted["scale"], fitted["features"].iloc[test], np.log(tau), event, np.log(grid)
        )
        # Its likelihood is of the log of the days watched; an event's density in days divides by them.
        return scores["curves"], scores["own"], scores["risk"], scores["nll"] + float(np.mean(event * np.log(tau)))
    if name.startswith("pycox"):
        surv = fitted["hazard"].interpolate(10).predict_surv_df(fitted["matrix"][test])
        times, values = surv.index.to_numpy(np.float64), surv.to_numpy(np.float64).T
        risk = 1.0 - _step_grid(times, values, np.array([RISK_DAYS]))[:, 0]
        return _step_grid(times, values, grid), _step_own(times, values, tau), risk, None
    curve = np.broadcast_to(fitted["survival"], (len(test), len(fitted["survival"])))
    return _step_grid(fitted["times"], curve, grid), _step_own(fitted["times"], curve, tau), np.zeros(len(test)), None


def _backblaze_split(features, entry, exit_, window, failed, train, test, seed: int, models=BACKBLAZE_MODELS) -> dict:
    """Each of ``models`` on one split, scored on the days each test drive was watched."""
    from sklearn.model_selection import train_test_split

    fit, held = train_test_split(train, test_size=0.15, stratify=failed[train], random_state=seed)
    tau, event = window[test], failed[test]
    grid = WINDOW_GRID[(WINDOW_GRID >= tau.min()) & (WINDOW_GRID < tau.max())]
    records = {}
    for name in models:
        if name.startswith("pycox") and importlib.util.find_spec("pycox") is None:
            print("    pycox is not installed; skipping its model", flush=True)
            continue
        try:
            with _Clock() as clock:
                fitted = _backblaze_fit(name, features, entry, window, failed, exit_, train, fit, held, seed)
            curves, own, risk, nll = _backblaze_curves(name, fitted, features, entry, exit_, window, failed, test, grid)
        except Exception as error:
            # pycox and torchtuples have lagged behind numpy releases; a failure there should not stop the others.
            if not name.startswith("pycox"):
                raise
            print(f"    pycox failed, {type(error).__name__}: {error}; skipping its model", flush=True)
            continue
        after, before = own if isinstance(own, tuple) else (own, None)
        chi2, p = d_calibration(after, event, before=before)
        records[name] = {
            "c_index": concordance(tau, event, risk),
            "ibs": integrated_brier(tau, event, curves, grid),
            "d_chi2": chi2,
            "d_p": p,
            "nll": nll,
            "cpu": clock.cpu,
            "wall": clock.wall,
        }
        if name == "xgboost aft":
            records[name]["chosen"] = f"{fitted['family']}, scale {fitted['scale']}"
            records[name]["chosen_cpu"], records[name]["chosen_wall"] = fitted["single"]["cpu"], fitted["single"]["wall"]
        printable = {key: round(value, 5) if isinstance(value, float) else value for key, value in records[name].items()}
        print(f"    {name:24s}", printable, flush=True)
    return records


def run_backblaze() -> None:
    from sklearn.model_selection import train_test_split

    results = _load("backblaze")
    features, entry, exit_, window, failed = backblaze_table(backblaze_drives())
    print(
        f"{len(failed)} drives, {failed.sum()} failed; median age at entry {np.median(entry) / 365.25:.1f} years, "
        f"{features['model'].nunique()} model groups",
        flush=True,
    )
    chosen = _arguments().get("seeds")
    seeds = BACKBLAZE_SEEDS if chosen is None else tuple(int(seed) for seed in chosen.split(","))
    for seed in seeds:
        # A rerun fits only the models a split lacks, such as pycox after it failed or was not installed.
        done = results.get(str(seed), {})
        missing = [name for name in BACKBLAZE_MODELS if name not in done]
        if not missing:
            continue
        train, test = train_test_split(np.arange(len(failed)), test_size=0.2, stratify=failed, random_state=seed)
        print(f"  split {seed}: {', '.join(missing)}", flush=True)
        results[str(seed)] = {**done, **_backblaze_split(features, entry, exit_, window, failed, train, test, seed, missing)}
        _save("backblaze", results)


def backblaze_report() -> None:
    results = _load("backblaze")
    if not results:
        return
    splits = list(results.values())
    print(f"\n### Backblaze drives, {BACKBLAZE_QUARTER}, {len(splits)} splits, mean ± standard deviation\n")
    print("| Model | Concordance ↑ | Integrated Brier × 1000 ↓ | D-calibration χ² ↓ | Passes D-calibration | NLL, days ↓ | CPU seconds | Wall seconds |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for model in BACKBLAZE_MODELS:
        rows = [split[model] for split in splits if model in split]
        if not rows:
            continue
        passes = sum(row["d_p"] > 0.05 for row in rows)
        print(
            f"| {model} | {_cell(rows, 'c_index', '{:.3f}')} | {_cell(rows, 'ibs', '{:.3f}', 1000.0)} | "
            f"{_cell(rows, 'd_chi2', '{:.1f}')} | {passes} of {len(rows)} | {_cell(rows, 'nll', '{:.4f}')} | "
            f"{np.mean([row['cpu'] for row in rows]):.0f} | {np.mean([row['wall'] for row in rows]):.0f} |"
        )
    _aft_note([split["xgboost aft"] for split in splits if "xgboost aft" in split])


def report() -> None:
    survival_report()
    backblaze_report()
    stockouts_report()
    totals_report()


STAGES = {
    "survival": run_survival,
    "backblaze": run_backblaze,
    "stockouts": run_stockouts,
    "totals": run_totals,
    "report": report,
}


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
