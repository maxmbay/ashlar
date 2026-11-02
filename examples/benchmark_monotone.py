"""Do unconstrained heads break stochastic dominance in bonus-malus or vehicle power on freMTPL2?

The cheap test of #40. On French motor insurance (freMTPL2, see
``examples/benchmark_chain.py`` for the table), a higher bonus-malus level
means a worse claims history, so an insurer expects the whole distribution
of claims, not just its mean, to rise with it. Two outcomes:

- ``frequency``: the claim count ``N`` of every policy, with a
  ``negative_binomial`` head, on a random sample of ``rows`` policies
  (200,000 by default).
- ``severity``: the log of the average cost per claim, on every policy with
  a claim, with the ``gaussian``, ``mixture``, and ``quantile_spline`` heads.

Each head is fitted as it is, and again with ``monotone={"BonusMalus": 1}``.
On 2,000 test policies, each feature in turn is set to each of 12 values
from its 2nd to its 98th percentile, and the exceedance ``P(y > t)`` read
at a ladder of thresholds (``N`` above 0, 1, and 2; the log cost at its
5th to 95th percentiles). A violation is a step from one value to the next
at which some threshold's exceedance moves against the feature's overall
direction by more than 0.001, the direction being the sign of the mean's
average change across the grid. Reported: the share of steps with a
violation, the share of policies with at least one, the mean of each
policy's largest move against the direction, the share of steps where the
mean itself moves against it, and the test NLL.

The ``gbm`` stage adds what a gradient-boosting user would build, on the
same splits, samples, probes, and ladders:

- the claim count: LightGBM with the Poisson objective, with and without
  ``monotone_constraints`` rising in bonus-malus. A Poisson's every
  exceedance rises with its mean, so the constrained one is dominance-safe
  by construction; the question is its likelihood against the constrained
  ``negative_binomial``.
- the log cost per claim: quantile regression at each level of
  ``GBM_LEVELS``, the boosted way to a distribution. LightGBM refuses
  ``monotone_constraints`` with its quantile objective, so it runs without
  them, one model per level. XGBoost takes them: one multi-quantile model
  (``objective="reg:quantileerror"``, a tree per level per round), with and
  without the constraint. Reported: the mean pinball loss over the levels on
  the test claims, against ``quantile_spline`` with and without ``monotone``
  (refitted here, so their quantiles can be scored), the share of test
  claims and of probe points (a policy at one level of the sweep) where some
  level's quantile is below the level under it, and the dominance test above
  on the CDF read from the levels, sorted where they cross and linear
  between them.

Each LightGBM model trains at learning rate 0.05 with up to 4,000 trees,
early-stopped after 100 rounds on a random 20% of its training rows, and
takes the best of a small grid of leaf counts and leaf sizes by validation
loss; XGBoost the same, with a grid of depths and smallest leaf weights.

Run with::

    JAX_PLATFORMS=cpu python examples/benchmark_monotone.py run seeds=3
    JAX_PLATFORMS=cpu python examples/benchmark_monotone.py gbm seeds=3
    JAX_PLATFORMS=cpu python examples/benchmark_monotone.py report
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_chain import CATEGORICAL, GBM_GRID, NUMERIC, _lightgbm, gbm_frame, poisson_log_pmf, policies  # noqa: E402

from ashlar import ASRRegressor  # noqa: E402

RESULTS = Path(os.environ.get("ASHLAR_BENCH", "/tmp/ashlar-bench")) / "monotone-fremtpl2.json"
GBM_RESULTS = RESULTS.with_name("monotone-fremtpl2-gbm.json")
GBM_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
# (depth, smallest leaf weight) tried for XGBoost's quantile models.
XGB_GRID = ((2, 1000), (3, 100), (5, 20), (7, 400))
FEATURES = ("BonusMalus", "VehPower")
GRID = 12
PROBE_ROWS = 2000
TOLERANCE = 1e-3
CONSTRAINT = {"BonusMalus": 1}


def _arguments() -> dict:
    return dict(argument.split("=", 1) for argument in sys.argv[2:] if "=" in argument)


def _exceedance(model, frame, thresholds) -> np.ndarray:
    """``(rows, thresholds)`` of ``P(y > t)``, and the mean, per row."""
    dist = model.predict_dist(frame)
    cdf = np.asarray(dist.cdf(np.asarray(thresholds, dtype=np.float32)[:, None]))
    return 1.0 - cdf.T, np.asarray(dist.mean())


def violations(model, probe, train, thresholds) -> dict:
    out = {}
    for feature in FEATURES:
        values = sweep(train, feature)
        curves, means = [], []
        for value in values:
            frame = probe.copy()
            frame[feature] = float(value)
            exceed, mean = _exceedance(model, frame, thresholds)
            curves.append(exceed)
            means.append(mean)
        curves, means = np.stack(curves), np.stack(means)  # (values, rows, thresholds), (values, rows)
        direction = 1.0 if np.mean(means[-1] - means[0]) >= 0 else -1.0
        against = -direction * np.diff(curves, axis=0)  # positive where a step moves against the direction
        worst = against.max(axis=2)  # (steps, rows)
        out[feature] = {
            "direction": direction,
            "steps violating": float(np.mean(worst > TOLERANCE)),
            "policies violating": float(np.mean((worst > TOLERANCE).any(axis=0))),
            "largest move against": float(np.mean(np.maximum(worst.max(axis=0), 0.0))),
            "mean against": float(np.mean(-direction * np.diff(means, axis=0) > 0)),
        }
    return out


def sweep(train, feature: str) -> np.ndarray:
    """The values a feature is set to: its training quantiles from the 2nd to the 98th percentile, distinct."""
    return np.unique(np.quantile(train[feature], np.linspace(0.02, 0.98, GRID)))


def splits(frame, seed: int, rows: int) -> dict:
    """The split, samples, probes, and cost ladder of one seed, as ``run`` draws them."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    test_size = len(frame) // 5
    test, train = frame.iloc[order[:test_size]], frame.iloc[order[test_size:]]
    probe = test.sample(PROBE_ROWS, random_state=seed)[NUMERIC + CATEGORICAL].reset_index(drop=True)
    sample = train.sample(min(int(rows), len(train)), random_state=seed)
    claimed_train, claimed_test = train[train["N"] > 0], test[test["N"] > 0]
    probe_claimed = claimed_test.sample(min(PROBE_ROWS, len(claimed_test)), random_state=seed)[NUMERIC + CATEGORICAL].reset_index(drop=True)
    log_cost = np.log(claimed_train["S"].to_numpy())
    return {
        "train": train, "test": test, "probe": probe, "sample": sample, "claimed_train": claimed_train,
        "claimed_test": claimed_test, "probe_claimed": probe_claimed,
        "cost_ladder": np.quantile(log_cost, np.linspace(0.05, 0.95, 9)),
    }


def run(seeds="3", rows="200000"):
    frame = policies()
    results = []
    if RESULTS.exists():
        results = json.loads(RESULTS.read_text())
    for seed in range(int(seeds)):
        parts = splits(frame, seed, int(rows))
        test, probe, sample = parts["test"], parts["probe"], parts["sample"]
        claimed_train, claimed_test = parts["claimed_train"], parts["claimed_test"]
        probe_claimed, cost_ladder = parts["probe_claimed"], parts["cost_ladder"]
        jobs = [
            ("frequency", "negative_binomial", sample, test, "N", probe, np.array([0.0, 1.0, 2.0])),
            ("severity", "gaussian", claimed_train, claimed_test, "S", probe_claimed, cost_ladder),
            ("severity", "mixture", claimed_train, claimed_test, "S", probe_claimed, cost_ladder),
            ("severity", "quantile_spline", claimed_train, claimed_test, "S", probe_claimed, cost_ladder),
        ]
        for outcome, task, fit_frame, score_frame, target, probe_frame, ladder in jobs:
            for constrained in (False, True):
                if constrained and task == "mixture":
                    continue
                y = fit_frame[target].to_numpy()
                y_test = score_frame[target].to_numpy()
                if outcome == "severity":
                    y, y_test = np.log(y), np.log(y_test)
                start = time.process_time()
                model = ASRRegressor(task=task, monotone=CONSTRAINT if constrained else None, random_state=seed)
                model.fit(fit_frame[NUMERIC + CATEGORICAL].reset_index(drop=True), y)
                seconds = time.process_time() - start
                record = {
                    "seed": seed,
                    "outcome": outcome,
                    "task": task,
                    "monotone": constrained,
                    "rows": len(fit_frame),
                    "nll": float(-np.mean(model.log_prob(score_frame[NUMERIC + CATEGORICAL].reset_index(drop=True), y_test))),
                    "cpu": seconds,
                    "violations": violations(model, probe_frame, fit_frame, ladder),
                }
                results.append(record)
                print(json.dumps(record), flush=True)
                RESULTS.parent.mkdir(parents=True, exist_ok=True)
                RESULTS.write_text(json.dumps(results, indent=1))


# ---------------------------------------------------------------- boosted trees, what a GBM user would build


def _constraints(constrained: bool) -> list[int]:
    """``monotone_constraints`` for LightGBM or XGBoost, one per column of ``NUMERIC + CATEGORICAL``."""
    return [CONSTRAINT.get(name, 0) if constrained else 0 for name in NUMERIC + CATEGORICAL]


class _PoissonView:
    """A Poisson per row, with the two methods ``violations`` reads."""

    def __init__(self, rate):
        self.rate = np.asarray(rate, dtype=float)

    def cdf(self, thresholds):
        from scipy.stats import poisson

        return poisson.cdf(np.floor(np.asarray(thresholds, dtype=float)), self.rate[None, :])

    def mean(self):
        return self.rate


class GBMPoisson:
    """LightGBM Poisson on the claim count, with or without the constraint."""

    def __init__(self, frame, y, seed: int, constrained: bool):
        self.categories = {name: sorted(frame[name].unique()) for name in CATEGORICAL}
        x = gbm_frame(frame, self.categories)
        inner = np.random.default_rng(seed).random(len(frame)) < 0.2
        best = None
        for leaves, smallest in GBM_GRID:
            model = _lightgbm(
                "poisson", (x[~inner], y[~inner]), (x[inner], y[inner]), seed, leaves, smallest,
                monotone_constraints=_constraints(constrained), monotone_constraints_method="advanced",
            )
            score = float(np.mean(poisson_log_pmf(y[inner], model.predict(x[inner]))))
            if best is None or score > best[0]:
                best = (score, model, (leaves, smallest))
        _score, self.model, self.choice = best

    def predict_dist(self, frame):
        return _PoissonView(self.model.predict(gbm_frame(frame, self.categories)))

    def log_prob(self, frame, y):
        return poisson_log_pmf(np.asarray(y, dtype=float), self.predict_dist(frame).rate)


class _QuantileView:
    """The distribution read from quantiles at fixed levels: sorted where they cross, linear between them."""

    def __init__(self, quantiles, levels):
        self.values = np.sort(np.asarray(quantiles, dtype=float), axis=0)  # (levels, rows)
        self.levels = np.asarray(levels, dtype=float)

    def cdf(self, thresholds):
        thresholds = np.asarray(thresholds, dtype=float).reshape(-1)
        return np.stack([
            np.array([np.interp(t, self.values[:, row], self.levels, left=0.0, right=1.0) for row in range(self.values.shape[1])])
            for t in thresholds
        ])

    def mean(self):
        return self.values.mean(axis=0)


class GBMQuantiles:
    """Boosted quantile regression at each of ``levels``: LightGBM (one model per level, no constraint, which
    LightGBM refuses for quantiles) or XGBoost (one multi-quantile model, with or without the constraint)."""

    def __init__(self, frame, y, seed: int, constrained: bool, levels=GBM_LEVELS, library: str = "xgboost"):
        if library == "lightgbm" and constrained:
            raise ValueError("LightGBM refuses monotone_constraints with the quantile objective.")
        self.categories = {name: sorted(frame[name].unique()) for name in CATEGORICAL}
        self.levels = tuple(levels)
        self.library = library
        x = gbm_frame(frame, self.categories)
        inner = np.random.default_rng(seed).random(len(frame)) < 0.2
        fit, val = (x[~inner], y[~inner]), (x[inner], y[inner])
        if library == "lightgbm":
            self.models, self.choices = [], []
            for level in self.levels:
                best = None
                for leaves, smallest in GBM_GRID:
                    model = _lightgbm("quantile", fit, val, seed, leaves, smallest, alpha=level)
                    score = float(np.mean(pinball(val[1], model.predict(val[0]), level)))
                    if best is None or score < best[0]:
                        best = (score, model, (leaves, smallest))
                self.models.append(best[1])
                self.choices.append(best[2])
            return
        import xgboost

        best = None
        for depth, weight in XGB_GRID:
            model = xgboost.XGBRegressor(
                objective="reg:quantileerror", quantile_alpha=np.asarray(self.levels), n_estimators=4000,
                learning_rate=0.05, max_depth=depth, min_child_weight=weight, early_stopping_rounds=100,
                enable_categorical=True, tree_method="hist", max_cat_to_onehot=1, random_state=seed,
                n_jobs=int(os.environ.get("GBM_THREADS", "4")),
                monotone_constraints=tuple(_constraints(constrained)),
            )
            model.fit(*fit, eval_set=[val], verbose=False)
            q = model.predict(val[0]).T
            score = float(np.mean([pinball(val[1], q[i], level).mean() for i, level in enumerate(self.levels)]))
            if best is None or score < best[0]:
                best = (score, model, (depth, weight))
        _score, self.model, self.choices = best

    def quantiles(self, frame) -> np.ndarray:
        """``(levels, rows)``, as the models give them, crossings and all."""
        x = gbm_frame(frame, self.categories)
        if self.library == "lightgbm":
            return np.stack([model.predict(x) for model in self.models])
        return np.asarray(self.model.predict(x)).reshape(len(x), -1).T

    def predict_dist(self, frame):
        return _QuantileView(self.quantiles(frame), self.levels)


def pinball(y, q, level) -> np.ndarray:
    diff = np.asarray(y, dtype=float) - np.asarray(q, dtype=float)
    return np.maximum(level * diff, (level - 1.0) * diff)


def crossings(quantiles_of, probe, train, feature: str = "BonusMalus") -> dict:
    """Over the sweep of ``feature``: the share of probe points, and of policies, where a level's quantile falls below the one under it."""
    values = sweep(train, feature)
    crossed = []
    for value in values:
        frame = probe.copy()
        frame[feature] = float(value)
        q = quantiles_of(frame)  # (levels, rows)
        crossed.append(np.any(np.diff(q, axis=0) < -1e-9, axis=0))
    crossed = np.stack(crossed)  # (values, rows)
    return {"points crossing": float(np.mean(crossed)), "policies crossing": float(np.mean(crossed.any(axis=0)))}


def gbm(seeds="3", rows="200000"):
    """The boosted baselines and ``quantile_spline``, on ``run``'s splits; writes ``monotone-fremtpl2-gbm.json``."""
    frame = policies()
    results = json.loads(GBM_RESULTS.read_text()) if GBM_RESULTS.exists() else []
    for seed in range(int(seeds)):
        parts = splits(frame, seed, int(rows))
        sample, test, probe = parts["sample"], parts["test"], parts["probe"]
        claimed_train, claimed_test = parts["claimed_train"], parts["claimed_test"]
        x_test = test[NUMERIC + CATEGORICAL].reset_index(drop=True)
        x_claimed = claimed_test[NUMERIC + CATEGORICAL].reset_index(drop=True)
        y_claimed = np.log(claimed_test["S"].to_numpy())
        for constrained in (False, True):
            start = time.process_time()
            model = GBMPoisson(sample, sample["N"].to_numpy(), seed, constrained)
            record = {
                "seed": seed, "outcome": "frequency", "model": "LightGBM Poisson", "monotone": constrained,
                "nll": float(-np.mean(model.log_prob(x_test, test["N"].to_numpy()))),
                "cpu": time.process_time() - start, "choices": model.choice,
                "violations": violations(model, probe, sample, np.array([0.0, 1.0, 2.0])),
            }
            results.append(record)
            print(json.dumps(record), flush=True)
        log_cost = np.log(claimed_train["S"].to_numpy())
        for library, constrained in (("lightgbm", False), ("xgboost", False), ("xgboost", True)):
            start = time.process_time()
            model = GBMQuantiles(claimed_train, log_cost, seed, constrained, library=library)
            seconds = time.process_time() - start
            q = model.quantiles(x_claimed)
            record = {
                "seed": seed, "outcome": "severity", "model": f"{'LightGBM' if library == 'lightgbm' else 'XGBoost'} quantiles",
                "monotone": constrained,
                "pinball": float(np.mean([pinball(y_claimed, q[i], level).mean() for i, level in enumerate(GBM_LEVELS)])),
                "pinball by level": [float(pinball(y_claimed, q[i], level).mean()) for i, level in enumerate(GBM_LEVELS)],
                "test crossing": float(np.mean(np.any(np.diff(q, axis=0) < -1e-9, axis=0))),
                "cpu": seconds, "choices": model.choices,
                "crossings": crossings(model.quantiles, parts["probe_claimed"], claimed_train),
                "violations": violations(model, parts["probe_claimed"], claimed_train, parts["cost_ladder"]),
            }
            results.append(record)
            print(json.dumps(record), flush=True)
        for constrained in (False, True):
            start = time.process_time()
            spline = ASRRegressor(task="quantile_spline", monotone=CONSTRAINT if constrained else None, random_state=seed)
            spline.fit(claimed_train[NUMERIC + CATEGORICAL].reset_index(drop=True), log_cost)
            seconds = time.process_time() - start
            q = spline.predict_quantiles(x_claimed, list(GBM_LEVELS)).T

            def spline_quantiles(frame, spline=spline):
                return spline.predict_quantiles(frame, list(GBM_LEVELS)).T

            record = {
                "seed": seed, "outcome": "severity", "model": "quantile_spline", "monotone": constrained,
                "nll": float(-np.mean(spline.log_prob(x_claimed, y_claimed))),
                "pinball": float(np.mean([pinball(y_claimed, q[i], level).mean() for i, level in enumerate(GBM_LEVELS)])),
                "pinball by level": [float(pinball(y_claimed, q[i], level).mean()) for i, level in enumerate(GBM_LEVELS)],
                "test crossing": float(np.mean(np.any(np.diff(q, axis=0) < -1e-9, axis=0))),
                "cpu": seconds,
                "crossings": crossings(spline_quantiles, parts["probe_claimed"], claimed_train),
            }
            results.append(record)
            print(json.dumps(record), flush=True)
        GBM_RESULTS.parent.mkdir(parents=True, exist_ok=True)
        GBM_RESULTS.write_text(json.dumps(results, indent=1))


def gbm_report():
    results = json.loads(GBM_RESULTS.read_text())
    keys = dict.fromkeys((r["outcome"], r["model"], r["monotone"]) for r in results)

    def cell(rows, pick, form="{:.4f}"):
        values = np.array([pick(r) for r in rows], dtype=float)
        return f"{form.format(values.mean())} ± {form.format(values.std())}"

    print("| Outcome | Model | `BonusMalus` rising | NLL ↓ | Pinball ↓ | Test claims crossing | Probe points crossing | "
          "Policies crossing | Steps violating | Policies violating | CPU s |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for outcome, model, constrained in keys:
        rows = [r for r in results if (r["outcome"], r["model"], r["monotone"]) == (outcome, model, constrained)]
        nll = cell(rows, lambda r: r["nll"]) if all("nll" in r for r in rows) else ""
        loss = cell(rows, lambda r: r["pinball"]) if all("pinball" in r for r in rows) else ""
        test_cross = cell(rows, lambda r: 100 * r["test crossing"], "{:.1f}") + "%" if all("test crossing" in r for r in rows) else ""
        points = cell(rows, lambda r: 100 * r["crossings"]["points crossing"], "{:.1f}") + "%" if all("crossings" in r for r in rows) else ""
        policies_ = cell(rows, lambda r: 100 * r["crossings"]["policies crossing"], "{:.1f}") + "%" if all("crossings" in r for r in rows) else ""
        steps = cell(rows, lambda r: 100 * r["violations"]["BonusMalus"]["steps violating"], "{:.1f}") + "%" if all("violations" in r for r in rows) else ""
        held = cell(rows, lambda r: 100 * r["violations"]["BonusMalus"]["policies violating"], "{:.1f}") + "%" if all("violations" in r for r in rows) else ""
        print(f"| {outcome} | {model} | {'yes' if constrained else 'no'} | {nll} | {loss} | {test_cross} | {points} | "
              f"{policies_} | {steps} | {held} | {np.mean([r['cpu'] for r in rows]):.0f} |")


def report():
    results = json.loads(RESULTS.read_text())
    keys = dict.fromkeys((r["outcome"], r["task"], r["monotone"]) for r in results)
    print("| Outcome | Head | Constraint | NLL ↓ | Feature | Steps violating | Policies violating | Largest move against | Mean against | CPU s |")
    print("|---|---|---|---:|---|---:|---:|---:|---:|---:|")
    for outcome, task, constrained in keys:
        rows = [r for r in results if (r["outcome"], r["task"], r["monotone"]) == (outcome, task, constrained)]
        nll = np.array([r["nll"] for r in rows])
        for feature in FEATURES:
            parts = [r["violations"][feature] for r in rows]

            def cell(name):
                values = np.array([p[name] for p in parts])
                return f"{values.mean():.4f} ± {values.std():.4f}"

            constraint = "`BonusMalus` rising" if constrained else "none"
            print(
                f"| {outcome} | `{task}` | {constraint} | {nll.mean():.4f} ± {nll.std():.4f} | {feature} | "
                f"{cell('steps violating')} | {cell('policies violating')} | {cell('largest move against')} | "
                f"{cell('mean against')} | {np.mean([r['cpu'] for r in rows]):.0f} |"
            )


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "gbm": gbm, "report": report, "gbm_report": gbm_report}[command](**_arguments())
