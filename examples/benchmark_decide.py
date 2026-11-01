"""Decisions from the distribution: is training through the newsvendor cost worth it?

Issue #32's kill test and its real-data sweep. A newsvendor orders before
demand is known; a unit short costs ``under`` and a unit left over costs
``over``, so the cost-optimal order is the ``under / (under + over)``
quantile. A model trained on its likelihood orders that quantile of its
predictive distribution (the plug-in order). A model trained on its
likelihood plus the realised cost of that order (``newsvendor_objective``,
the "hybrid") can move the quantile it will be judged on. The issue's rule:
drop decision training if the hybrid's gain on the mixture head is under 1%.

**kill** (synthetic, seeds 0 to 4). Four features; demand is
``exp(0.5 x0 + s(x1) G)`` with ``G ~ Gamma(2)`` and ``s(x1) = 0.4 + 0.3
tanh(x1)``: right-skewed, its spread set by one feature. 6,000 rows to
train, 1,500 to calibrate on, 20,000 to test. Costs 9:1, so the 0.9
quantile. Each of ``gaussian`` (misspecified: symmetric) and ``mixture``
(flexible) is trained on the likelihood (``nll``), the hybrid (``hybrid``,
weights 1 and 1), and the cost alone (``cost``). LightGBM's quantile
regression at 0.9 and the true quantile (``oracle``) are the baselines.
Scored on the test rows: cost in percent above the oracle's, the share of
demand at or below the order (the service level, 90% is right), and the NLL
of the demand. ``calibrated`` columns: the same after ``calibrate`` on the
calibration rows, which turns the order into a conformal quantile whose
service level holds.

**bike** (hourly Capital Bikeshare rentals, seeds 0 to 2, 80/20 split).
Cost ratios 1:1, 3:1, 9:1, and 19:1. The mixture trained on its likelihood
orders each ratio's quantile from one fit; the hybrid is refitted per
ratio; LightGBM's quantile regression is fitted per ratio. Cost in percent
against the likelihood fit's plug-in order (negative is better), and NLL.

    python examples/benchmark_decide.py kill bike [--seeds 0 1 2 3 4] [--out decide.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

import jax.numpy as jnp
import lightgbm
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ashlar import ASRRegressor
from ashlar.decide import critical_ratio, newsvendor_cost
from ashlar.errors import AshlarError

DECISION_TASKS = ("gaussian", "mixture", "flow", "histogram", "student_t")

KILL_COSTS = (9.0, 1.0)
BIKE_COSTS = ((1.0, 1.0), (3.0, 1.0), (9.0, 1.0), (19.0, 1.0))
EPOCHS = 60


def _cost(y, order, under, over) -> float:
    return float(np.mean(np.asarray(newsvendor_cost(y, order, under, over))))


def _lgbm_quantile(level, seed, X, y, X_test):
    model = lightgbm.LGBMRegressor(
        objective="quantile", alpha=level, n_estimators=400, learning_rate=0.05, verbose=-1, random_state=seed
    )
    model.fit(X, y)
    return model.predict(X_test)


def _objective(task, kind, under, over):
    if kind == "nll":
        return None
    if kind == "hybrid":
        return newsvendor_objective(task, under, over)
    return newsvendor_objective(task, under, over, nll=0.0)


# ---------------------------------------------------------------- the objective tried


def newsvendor_objective(task: str, under: float, over: float, *, nll: float = 1.0, cost: float = 1.0):
    """A training objective: ``nll`` times the head's loss plus ``cost`` times the newsvendor cost.

    Pass it as ``objective=`` to a univariate continuous head (``gaussian``,
    ``mixture``, ``flow``, ``histogram``, ``student_t``)::

        model = ASRRegressor(task="mixture", objective=newsvendor_objective("mixture", under=9, over=1))

    The cost is that of ordering the critical-ratio quantile of the members'
    equal-weight mixture, the distribution ``predict_dist`` returns, on the
    standardized target, so its gradient reaches the parameters through the
    quantile's implicit gradient. The model stays a full distribution; with
    ``nll=0`` it is trained for the decision alone. The cost is divided by
    ``under + over``, so it is the pinball loss at the critical ratio and its
    size does not depend on the scale of the costs. 
    """
    return NewsvendorObjective(task, under, over, nll=nll, cost=cost)


class NewsvendorObjective:
    """``objective(raw, y, sample_weight)`` for ``newsvendor_objective``; see there."""

    def __init__(self, task: str, under: float, over: float, *, nll: float = 1.0, cost: float = 1.0):
        from ashlar.objectives import make_distribution_objective

        if task not in DECISION_TASKS:
            raise AshlarError(f"decision training needs a univariate continuous head: {', '.join(DECISION_TASKS)}.")
        self.ratio = critical_ratio(under, over)
        nll, cost = float(nll), float(cost)
        if nll < 0.0 or cost < 0.0 or not (math.isfinite(nll) and math.isfinite(cost)) or nll + cost == 0.0:
            raise AshlarError("nll= and cost= must be finite and non-negative, and one must be positive.")
        self.task, self.under, self.over, self.nll, self.cost = task, float(under), float(over), nll, cost
        self._likelihood = make_distribution_objective(task)

    def __repr__(self) -> str:
        return f"newsvendor_objective({self.task!r}, under={self.under}, over={self.over}, nll={self.nll}, cost={self.cost})"

    def __call__(self, raw, y, sample_weight):
        from ashlar.distributions import Mixture, head_distribution
        from ashlar.weights import weighted_mean

        target = jnp.asarray(y, dtype=raw.dtype).reshape(-1)
        order = Mixture(head_distribution(self.task, raw)).quantile(self.ratio)
        ratio = self.ratio
        pinball = ratio * jnp.maximum(target - order, 0.0) + (1.0 - ratio) * jnp.maximum(order - target, 0.0)
        loss = self.cost * weighted_mean(pinball, sample_weight)
        if self.nll > 0.0:
            loss = loss + self.nll * self._likelihood(raw, y, sample_weight)
        return loss


# ---------------------------------------------------------------- kill


def _kill_data(n, rng):
    X = rng.normal(size=(n, 4)).astype(np.float32)
    scale = 0.4 + 0.3 * np.tanh(X[:, 1])
    y = np.exp(0.5 * X[:, 0] + scale * rng.standard_gamma(2.0, size=n))
    return X, y, scale


def _kill_oracle(X, scale, level):
    return np.exp(0.5 * X[:, 0] + scale * stats.gamma.ppf(level, 2.0))


def kill(seeds, out):
    under, over = KILL_COSTS
    level = critical_ratio(under, over)
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        X, y, _ = _kill_data(7500, rng)
        X_test, y_test, scale_test = _kill_data(20000, np.random.default_rng(1000 + seed))
        X_fit, y_fit, X_cal, y_cal = X[:6000], y[:6000], X[6000:], y[6000:]
        oracle = _cost(y_test, _kill_oracle(X_test, scale_test, level), under, over)
        lgbm = _lgbm_quantile(level, seed, X_fit, y_fit, X_test)
        rows.append(_kill_row(seed, "lightgbm", "quantile", lgbm, None, y_test, oracle, under, over, 0.0))
        for task in ("gaussian", "mixture"):
            for kind in ("nll", "hybrid", "cost"):
                start = time.time()
                model = ASRRegressor(task=task, objective=_objective(task, kind, under, over), max_epochs=EPOCHS,
                                     random_state=seed).fit(X_fit, y_fit)
                elapsed = time.time() - start
                order = model.predict_order(X_test, under, over)
                nll = -float(np.mean(model.log_prob(X_test, y_test)))
                model.calibrate(X_cal, y_cal)
                calibrated = model.predict_order(X_test, under, over)
                row = _kill_row(seed, task, kind, order, calibrated, y_test, oracle, under, over, elapsed)
                row["nll"] = nll
                rows.append(row)
                print(json.dumps(row), flush=True)
    _save(out, "kill", rows)
    _kill_table(rows)


def _kill_row(seed, model, kind, order, calibrated, y, oracle, under, over, seconds):
    row = {
        "seed": seed,
        "model": model,
        "objective": kind,
        "cost_pct": 100.0 * (_cost(y, order, under, over) / oracle - 1.0),
        "service": float(np.mean(y <= order)),
        "seconds": seconds,
    }
    if calibrated is not None:
        row["cal_cost_pct"] = 100.0 * (_cost(y, calibrated, under, over) / oracle - 1.0)
        row["cal_service"] = float(np.mean(y <= calibrated))
    return row


def _kill_table(rows):
    print("\n| model | objective | cost above oracle | service level | NLL | calibrated cost | calibrated service |")
    print("|---|---|---:|---:|---:|---:|---:|")
    keys = sorted({(row["model"], row["objective"]) for row in rows}, key=lambda k: (k[0] != "lightgbm", k))
    for key in keys:
        group = [row for row in rows if (row["model"], row["objective"]) == key]
        cells = [_ms(group, "cost_pct", "%"), _ms(group, "service", "", 100, "%"), _ms(group, "nll"),
                 _ms(group, "cal_cost_pct", "%"), _ms(group, "cal_service", "", 100, "%")]
        print(f"| {key[0]} | {key[1]} | " + " | ".join(cells) + " |")
    for task in ("gaussian", "mixture"):
        base = {row["seed"]: row["cost_pct"] for row in rows if row["model"] == task and row["objective"] == "nll"}
        for kind in ("hybrid", "cost"):
            gain = [
                (100.0 + base[row["seed"]]) / (100.0 + row["cost_pct"]) * 100.0 - 100.0
                for row in rows
                if row["model"] == task and row["objective"] == kind
            ]
            print(f"{task} {kind}: cost {np.mean(gain):+.2f}% ± {np.std(gain):.2f} lower than nll (positive is a gain)")


def _ms(group, key, unit="", factor=1.0, suffix=None):
    values = [factor * row[key] for row in group if key in row]
    if not values:
        return "–"
    suffix = unit if suffix is None else suffix
    return f"{np.mean(values):.2f}{suffix} ± {np.std(values):.2f}"


# ---------------------------------------------------------------- bike


def bike(seeds, out):
    from benchmark_uncertainty import bike as load_bike

    X, y = load_bike("cnt")
    rows = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        base = ASRRegressor(task="mixture", max_epochs=EPOCHS, cyclical={"hr": 24}, random_state=seed).fit(X_train, y_train)
        base_nll = -float(np.mean(base.log_prob(X_test, y_test)))
        numeric = X_train.copy()
        numeric_test = X_test.copy()
        for column in ("season", "weathersit"):
            numeric[column] = numeric[column].astype(float)
            numeric_test[column] = numeric_test[column].astype(float)
        for under, over in BIKE_COSTS:
            level = critical_ratio(under, over)
            plug = _cost(y_test, base.predict_order(X_test, under, over), under, over)
            start = time.time()
            hybrid = ASRRegressor(task="mixture", objective=newsvendor_objective("mixture", under, over),
                                  max_epochs=EPOCHS, cyclical={"hr": 24}, random_state=seed).fit(X_train, y_train)
            elapsed = time.time() - start
            order = hybrid.predict_order(X_test, under, over)
            lgbm = _lgbm_quantile(level, seed, numeric, y_train, numeric_test)
            row = {
                "seed": seed,
                "ratio": f"{under:g}:{over:g}",
                "nll_cost": plug,
                "hybrid_pct": 100.0 * (_cost(y_test, order, under, over) / plug - 1.0),
                "lightgbm_pct": 100.0 * (_cost(y_test, lgbm, under, over) / plug - 1.0),
                "nll_service": float(np.mean(y_test <= base.predict_order(X_test, under, over))),
                "hybrid_service": float(np.mean(y_test <= order)),
                "nll_nll": base_nll,
                "hybrid_nll": -float(np.mean(hybrid.log_prob(X_test, y_test))),
                "seconds": elapsed,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    _save(out, "bike", rows)
    print("\n| costs (short:over) | level | hybrid cost vs nll | LightGBM cost vs nll | NLL: nll fit | NLL: hybrid |")
    print("|---|---:|---:|---:|---:|---:|")
    for under, over in BIKE_COSTS:
        group = [row for row in rows if row["ratio"] == f"{under:g}:{over:g}"]
        print(f"| {under:g}:{over:g} | {critical_ratio(under, over):.2f} | {_ms(group, 'hybrid_pct', '%')} | "
              f"{_ms(group, 'lightgbm_pct', '%')} | {_ms(group, 'nll_nll')} | {_ms(group, 'hybrid_nll')} |")


def _save(out, stage, rows):
    path = Path(out)
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload[stage] = rows
    path.write_text(json.dumps(payload, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stages", nargs="*", default=["kill", "bike"])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--out", default="decide-results.json")
    args = parser.parse_args()
    for stage in args.stages:
        if stage == "kill":
            kill(args.seeds or [0, 1, 2, 3, 4], args.out)
        elif stage == "bike":
            bike(args.seeds or [0, 1, 2], args.out)
        else:
            raise SystemExit(f"unknown stage {stage!r}: kill or bike")


if __name__ == "__main__":
    main()
