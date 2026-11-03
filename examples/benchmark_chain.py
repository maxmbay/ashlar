"""A chain of a claim count and a claim cost on French motor insurance (freMTPL2), against separate heads.

freMTPL2 (from the R package CASdatasets) holds 677,991 motor policies with
their exposure in years, their number of claims, and, in a second table, the
cost of every claim. Each policy here has two outcomes: ``N``, its number of
claims (capped at 4, as is usual for this table), and ``S``, its average
cost per claim, 0 when it had none; 3.7% of policies had a claim. A
policy's count is its number of recorded costs, and a policy with claims but
no recorded cost would be dropped, though the current CASdatasets table has
none. The features are the driver's age, the
bonus-malus level, the vehicle's power, age, brand, and fuel, the area, the
region, the log of the population density, and the exposure.

Three models of the pair, each with Ashlar's default trunk:

- ``chain``: ``task="chain"`` with a ``negative_binomial`` count and then a
  ``hurdle`` cost that reads the count, so the cost can be exactly zero
  where there was no claim and can depend on how many there were.
- ``independent heads``: a ``negative_binomial`` model of ``N`` and a
  ``hurdle`` model of ``S``, fitted apart; their joint is the product.
- ``frequency and severity``: the actuary's model by hand, a
  ``negative_binomial`` model of ``N`` and a ``gaussian`` model of ``log S``
  fitted on the policies with claims, with ``S = 0`` imposed where ``N = 0``
  and the cost taken as independent of the count given the features.

Scores on the test policies (20% of them, three random splits):

- Joint NLL: ``-log p(N, S)`` per policy, in nats; a cost's density is per
  euro. Split into the count's part and the cost's part given the count.
- The total claim cost ``T = N × S`` of each policy, read from 200 joint
  draws per policy on 50,000 test policies: the share predicted to claim,
  ``P(T > 0)``, against the share that did; the predicted share above 1,000
  and 5,000 euros against the observed; the sum of the predicted means against
  the sum of the costs; the CRPS; and how far the randomized probability
  integral transform of ``T`` is from uniform (its Kolmogorov distance, ×
  100), the PIT spread uniformly over the atom at zero.

The issue's stop rule (#38): stop if the chain gains less than about 0.01
nats per policy and its total is no better calibrated.

The ``gbm`` stage adds what a gradient-boosting user would build, on the
same splits and scored the same way:

- ``LightGBM Poisson × gamma``: the actuary's model with LightGBM, a
  Poisson objective on ``N`` and a gamma objective on ``S`` of the policies
  with claims, composed by hand. The cost is a gamma with the fitted mean and
  one shape estimated by maximum likelihood on the validation claims; with
  ``N`` claims the average is a gamma of shape ``N`` times that (the usual
  claim-count weights), or of the one shape, whichever the validation claims
  prefer. Joint NLL and draws are exact under that composition.
- ``LightGBM Tweedie``: one model of the total ``T`` with the Tweedie
  objective, the usual gradient-boosted pure premium. It gives a mean; its
  Tweedie distribution, a compound Poisson-gamma with the dispersion fitted
  on the validation rows, gives the rest. It models ``T`` alone, so it has
  no joint NLL.

Each LightGBM model trains at learning rate 0.05 with up to 4,000 trees,
early-stopped after 100 rounds on a random 20% of the training split, and
takes the best of a small grid of leaf counts and leaf sizes (and Tweedie
powers) by validation log-likelihood.

Run with::

    JAX_PLATFORMS=cpu python examples/benchmark_chain.py run      # downloads about 10 MB
    JAX_PLATFORMS=cpu python examples/benchmark_chain.py gbm      # minutes, adds the LightGBM rows
    JAX_PLATFORMS=cpu python examples/benchmark_chain.py report

``seeds=0`` runs fewer splits and ``rows=100000`` a subsample.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

from ashlar import ASRRegressor, observed

DATA_DIR = Path("/tmp/ashlar-bench")
RESULTS = DATA_DIR / "chain-fremtpl2.json"
SOURCE = "https://raw.githubusercontent.com/dutangc/CASdatasets/master/data/"
SEEDS = (0, 1, 2)
DRAWS = 200
DRAW_ROWS = 50_000
MODELS = ("chain", "independent heads", "frequency and severity")
GBM_MODELS = ("LightGBM Poisson × gamma", "LightGBM Tweedie")
GBM_THREADS = int(os.environ.get("GBM_THREADS", "4"))
# (leaves, smallest leaf) tried for every LightGBM model; the Tweedie also tries each power.
GBM_GRID = ((7, 1000), (15, 100), (31, 20), (63, 400))
TWEEDIE_POWERS = (1.3, 1.5, 1.7)
THRESHOLDS = (1000.0, 5000.0)
NUMERIC = ["DrivAge", "BonusMalus", "VehPower", "VehAge", "LogDensity", "Exposure"]
CATEGORICAL = ["VehBrand", "VehGas", "Area", "Region"]


def _arguments() -> dict:
    return dict(argument.split("=", 1) for argument in sys.argv[2:] if "=" in argument)


def _download(name: str) -> Path:
    path = DATA_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE + name, path)
    return path


def policies() -> pd.DataFrame:
    """One row per policy: the features, ``N`` (claims, at most 4), and ``S`` (average cost per claim)."""
    import pyreadr

    frequency = next(iter(pyreadr.read_r(str(_download("freMTPL2freq.rda"))).values()))
    severity = next(iter(pyreadr.read_r(str(_download("freMTPL2sev.rda"))).values()))
    frequency["IDpol"] = frequency["IDpol"].astype(str).astype(float).astype(int)
    severity["IDpol"] = severity["IDpol"].astype(str).astype(float).astype(int)
    costs = severity.groupby("IDpol")["ClaimAmount"].agg(["sum", "count"])
    frame = frequency.merge(costs, left_on="IDpol", right_index=True, how="left")
    frame["sum"] = frame["sum"].fillna(0.0)
    frame["count"] = frame["count"].fillna(0).astype(int)
    # A policy with claims and no recorded cost has no severity; one with costs counts its recorded claims.
    frame = frame[(frame["ClaimNb"] == 0) | (frame["count"] > 0)].copy()
    claims = np.where(frame["count"] > 0, frame["count"], frame["ClaimNb"]).astype(float)
    frame["N"] = np.minimum(claims, 4.0)
    frame["S"] = np.where(claims > 0, frame["sum"] / np.maximum(claims, 1.0), 0.0)
    frame["LogDensity"] = np.log(frame["Density"].astype(float))
    frame["Exposure"] = np.minimum(frame["Exposure"].astype(float), 1.0)
    for name in ("DrivAge", "BonusMalus", "VehPower", "VehAge"):
        frame[name] = frame[name].astype(float)
    for name in CATEGORICAL:
        frame[name] = frame[name].astype(str)
    return frame.reset_index(drop=True)


def _features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[NUMERIC + CATEGORICAL].reset_index(drop=True)


def _count_model(seed: int) -> ASRRegressor:
    return ASRRegressor(task="negative_binomial", random_state=seed)


def _fit_models(train: pd.DataFrame, seed: int) -> tuple[dict, dict]:
    x = _features(train)
    n, s = train["N"].to_numpy(), train["S"].to_numpy()
    models, seconds = {}, {}

    start = time.process_time()
    models["chain"] = ASRRegressor(
        task="chain",
        head_kwargs={"columns": ["negative_binomial", {"kind": "hurdle", "severity": "lognormal"}]},
        random_state=seed,
    ).fit(x, np.stack([n, s], axis=1))
    seconds["chain"] = time.process_time() - start

    start = time.process_time()
    count = _count_model(seed).fit(x, n)
    seconds["count"] = time.process_time() - start
    start = time.process_time()
    cost = ASRRegressor(task="hurdle", random_state=seed).fit(x, s)
    seconds["independent heads"] = seconds["count"] + time.process_time() - start
    models["independent heads"] = (count, cost)

    claimed = n > 0
    start = time.process_time()
    severity = ASRRegressor(task="gaussian", random_state=seed).fit(x[claimed], np.log(s[claimed]))
    seconds["frequency and severity"] = seconds["count"] + time.process_time() - start
    models["frequency and severity"] = (count, severity)
    return models, seconds


def _log_likelihood(name: str, model, x, n, s) -> tuple[np.ndarray, np.ndarray]:
    """``(log p(N), log p(S | N))`` per policy."""
    if name == "chain":
        joint = model.log_prob(x, np.stack([n, s], axis=1))
        # The count's own likelihood: the cost left open, an interval over every value.
        unknown = np.full(len(n), np.inf)
        log_n = model.log_prob(x, observed.stack([observed.exact(n), observed.interval(-unknown, unknown)]))
        return log_n, joint - log_n
    count, cost = model
    log_n = count.log_prob(x, n)
    if name == "independent heads":
        return log_n, cost.log_prob(x, s)
    claimed = n > 0
    log_s = np.zeros_like(log_n)
    if claimed.any():
        log_s[claimed] = cost.log_prob(x[claimed], np.log(s[claimed])) - np.log(s[claimed])
    return log_n, log_s


def _draws(name: str, model, x, seed: int) -> np.ndarray:
    """``(DRAWS, rows)`` draws of the total cost ``N × S``."""
    if name == "chain":
        pair = model.sample(x, n=DRAWS, seed=seed)
        return pair[..., 0] * pair[..., 1]
    count, cost = model
    n = count.sample(x, n=DRAWS, seed=seed)
    if name == "independent heads":
        return n * cost.sample(x, n=DRAWS, seed=seed + 1)
    return np.where(n > 0, n * np.exp(cost.sample(x, n=DRAWS, seed=seed + 1)), 0.0)


def _crps(draws: np.ndarray, y: np.ndarray) -> float:
    draws = np.sort(draws, axis=0)
    m = draws.shape[0]
    first = np.mean(np.abs(draws - y[None, :]), axis=0)
    weights = (2.0 * np.arange(1, m + 1) - m - 1)[:, None]
    second = np.sum(weights * draws, axis=0) / (m * m)
    return float(np.mean(first - second))


def _pit_distance(draws: np.ndarray, y: np.ndarray, seed: int) -> float:
    """Kolmogorov distance of the randomized PIT of ``y`` among the draws from uniform."""
    rng = np.random.default_rng(seed)
    below = np.mean(draws < y[None, :], axis=0)
    at = np.mean(draws <= y[None, :], axis=0)
    pit = np.sort(below + rng.uniform(size=y.shape) * (at - below))
    grid = np.arange(1, pit.size + 1) / pit.size
    return float(np.max(np.maximum(np.abs(grid - pit), np.abs(grid - 1.0 / pit.size - pit))))


def _scores(name: str, model, test: pd.DataFrame, seed: int) -> dict:
    x = _features(test)
    n, s = test["N"].to_numpy(), test["S"].to_numpy()
    log_n, log_s = _log_likelihood(name, model, x, n, s)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(test), size=min(DRAW_ROWS, len(test)), replace=False)
    draws = _draws(name, model, x.iloc[rows].reset_index(drop=True), seed)
    total = (n * s)[rows]
    scores = {
        "nll": float(-np.mean(log_n + log_s)),
        "nll_count": float(-np.mean(log_n)),
        "nll_cost": float(-np.mean(log_s)),
        "claim_predicted": float(np.mean(draws > 0)),
        "claim_observed": float(np.mean(total > 0)),
        "mean_ratio": float(np.mean(draws) / np.mean(total)),
        "crps": _crps(draws, total),
        "pit_distance": 100.0 * _pit_distance(draws, total, seed),
    }
    for threshold in THRESHOLDS:
        scores[f"above_{int(threshold)}_predicted"] = float(np.mean(draws > threshold))
        scores[f"above_{int(threshold)}_observed"] = float(np.mean(total > threshold))
    return scores


# ---------------------------------------------------------------- LightGBM, what a GBM user would build


def gbm_frame(frame: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """LightGBM's view of the features: the numbers as they are, the categorical columns as pandas categories."""
    x = frame[NUMERIC + CATEGORICAL].reset_index(drop=True).copy()
    for name in CATEGORICAL:
        x[name] = pd.Categorical(x[name], categories=categories[name])
    return x


def _lightgbm(objective, fit, val, seed, leaves, smallest, weights=(None, None), **extra):
    """Early-stopped LightGBM at learning rate 0.05, up to 4,000 trees, 100 rounds of patience."""
    import warnings

    import lightgbm

    warnings.filterwarnings("ignore", message=".*eval_set.*deprecated", category=Warning)

    model = lightgbm.LGBMRegressor(
        objective=objective, n_estimators=4000, learning_rate=0.05, num_leaves=leaves, min_child_samples=smallest,
        random_state=seed, n_jobs=GBM_THREADS, verbose=-1, **extra,
    )
    model.fit(
        fit[0], fit[1], sample_weight=weights[0], eval_set=[val],
        eval_sample_weight=None if weights[1] is None else [weights[1]],
        callbacks=[lightgbm.early_stopping(100, verbose=False)],
    )
    return model


def poisson_log_pmf(n, rate):
    from scipy.special import gammaln

    return n * np.log(rate) - rate - gammaln(n + 1.0)


def gamma_log_pdf(s, mean, shape):
    from scipy.special import gammaln

    return shape * np.log(shape / mean) + (shape - 1.0) * np.log(s) - shape * s / mean - gammaln(shape)


def _best_scalar(loss) -> float:
    from scipy.optimize import minimize_scalar

    return float(np.exp(minimize_scalar(loss, bounds=(-12.0, 12.0), method="bounded").x))


def tweedie_parts(mean, power, dispersion):
    """A Tweedie of mean ``mean`` is a Poisson number of gammas: ``(rate, shape, scale)``."""
    rate = np.power(mean, 2.0 - power) / (dispersion * (2.0 - power))
    shape = (2.0 - power) / (power - 1.0)
    scale = dispersion * (power - 1.0) * np.power(mean, power - 1.0)
    return rate, shape, scale


def tweedie_log_pdf(t, mean, power, dispersion, terms: int = 60):
    """Log probability of ``T = 0``, and log density per euro of ``T > 0``, under the compound Poisson-gamma."""
    from scipy.special import gammaln, logsumexp

    rate, shape, scale = tweedie_parts(np.asarray(mean, dtype=float), power, dispersion)
    out = -rate.copy()
    positive = np.asarray(t) > 0
    if positive.any():
        n = np.arange(1, terms + 1, dtype=float)[:, None]
        value, r, theta = np.asarray(t)[positive], rate[positive], scale[positive]
        log_terms = (
            poisson_log_pmf(n, r) + (n * shape - 1.0) * np.log(value) - value / theta
            - n * shape * np.log(theta) - gammaln(n * shape)
        )
        out[positive] = logsumexp(log_terms, axis=0)
    return out


class GBMActuary:
    """LightGBM Poisson on the claim count and LightGBM gamma on the cost per claim, composed by hand."""

    def __init__(self, train: pd.DataFrame, seed: int):
        self.categories = {name: sorted(train[name].unique()) for name in CATEGORICAL}
        x = gbm_frame(train, self.categories)
        n, s = train["N"].to_numpy(), train["S"].to_numpy()
        inner = np.random.default_rng(seed).random(len(train)) < 0.2
        self.choices = {}
        best = None
        for leaves, smallest in GBM_GRID:
            model = _lightgbm("poisson", (x[~inner], n[~inner]), (x[inner], n[inner]), seed, leaves, smallest)
            score = float(np.mean(poisson_log_pmf(n[inner], model.predict(x[inner]))))
            if best is None or score > best[0]:
                best = (score, model, (leaves, smallest))
        self.count, self.choices["count"] = best[1], best[2]
        claimed = n > 0
        fit, val = claimed & ~inner, claimed & inner
        best = None
        for weighted in (True, False):
            w = n if weighted else np.ones_like(n)
            for leaves, smallest in GBM_GRID:
                model = _lightgbm(
                    "gamma", (x[fit], s[fit]), (x[val], s[val]), seed, leaves, smallest, weights=(w[fit], w[val])
                )
                mean = model.predict(x[val])
                shape = _best_scalar(lambda log_k: -np.sum(gamma_log_pdf(s[val], mean, w[val] * np.exp(log_k))))
                score = float(np.mean(gamma_log_pdf(s[val], mean, w[val] * shape)))
                if best is None or score > best[0]:
                    best = (score, model, shape, weighted, (leaves, smallest))
        _score, self.cost, self.shape, self.weighted, self.choices["cost"] = best
        self.choices["cost weighted by the count"] = self.weighted
        self.choices["gamma shape"] = self.shape

    def rate(self, frame):
        return self.count.predict(gbm_frame(frame, self.categories))

    def cost_mean(self, frame):
        return self.cost.predict(gbm_frame(frame, self.categories))

    def cost_shape(self, n):
        """The gamma shape of the average cost of ``n`` claims."""
        return self.shape * (np.maximum(n, 1.0) if self.weighted else np.ones_like(n, dtype=float))

    def log_likelihood(self, frame, n, s):
        log_n = poisson_log_pmf(n, self.rate(frame))
        log_s = np.zeros_like(log_n)
        claimed = n > 0
        if claimed.any():
            mean = self.cost_mean(frame)[claimed]
            log_s[claimed] = gamma_log_pdf(s[claimed], mean, self.cost_shape(n[claimed]))
        return log_n, log_s

    def draws(self, frame, draws: int, rng) -> np.ndarray:
        rate, mean = self.rate(frame), self.cost_mean(frame)
        n = rng.poisson(rate[None, :], size=(draws, len(rate))).astype(float)
        shape = self.cost_shape(n)
        s = rng.gamma(shape, mean[None, :] / shape)
        return np.where(n > 0, n * s, 0.0)

    def exceedance(self, frame, thresholds, counts: int = 8) -> np.ndarray:
        """``(rows, thresholds)``: exact ``P(T > t)``, summed over 1 to ``counts`` claims."""
        from scipy.special import gammaincc
        from scipy.stats import poisson

        rate, mean = self.rate(frame), self.cost_mean(frame)
        out = np.zeros((len(rate), len(thresholds)))
        for k in range(1, counts + 1):
            shape = float(self.cost_shape(np.array([k]))[0])
            mass = poisson.pmf(k, rate)[:, None]
            out += mass * gammaincc(shape, np.asarray(thresholds)[None, :] / k / (mean[:, None] / shape))
        return out


class GBMTweedie:
    """LightGBM Tweedie on the total ``T``, and its compound Poisson-gamma with the dispersion fitted on validation rows."""

    def __init__(self, train: pd.DataFrame, seed: int):
        self.categories = {name: sorted(train[name].unique()) for name in CATEGORICAL}
        x = gbm_frame(train, self.categories)
        t = (train["N"] * train["S"]).to_numpy()
        inner = np.random.default_rng(seed).random(len(train)) < 0.2
        best = None
        for power in TWEEDIE_POWERS:
            for leaves, smallest in GBM_GRID:
                model = _lightgbm(
                    "tweedie", (x[~inner], t[~inner]), (x[inner], t[inner]), seed, leaves, smallest,
                    tweedie_variance_power=power,
                )
                mean = model.predict(x[inner])
                dispersion = _best_scalar(lambda log_phi: -np.sum(tweedie_log_pdf(t[inner], mean, power, np.exp(log_phi))))
                score = float(np.mean(tweedie_log_pdf(t[inner], mean, power, dispersion)))
                if best is None or score > best[0]:
                    best = (score, model, power, dispersion, (leaves, smallest))
        _score, self.model, self.power, self.dispersion, grid = best
        self.choices = {"total": grid, "power": self.power, "dispersion": self.dispersion}

    def mean(self, frame):
        return self.model.predict(gbm_frame(frame, self.categories))

    def draws(self, frame, draws: int, rng) -> np.ndarray:
        rate, shape, scale = tweedie_parts(self.mean(frame), self.power, self.dispersion)
        n = rng.poisson(rate[None, :], size=(draws, len(rate))).astype(float)
        total = rng.gamma(np.maximum(n, 1.0) * shape, np.broadcast_to(scale[None, :], n.shape))
        return np.where(n > 0, total, 0.0)

    def exceedance(self, frame, thresholds, counts: int = 60) -> np.ndarray:
        """``(rows, thresholds)``: exact ``P(T > t)``, summed over the claim counts that carry any mass."""
        from scipy.special import gammaincc
        from scipy.stats import poisson

        rate, shape, scale = tweedie_parts(self.mean(frame), self.power, self.dispersion)
        counts = int(min(counts, max(1, poisson.isf(1e-12, rate.max()) + 1)))
        out = np.zeros((len(rate), len(thresholds)))
        for k in range(1, counts + 1):
            out += poisson.pmf(k, rate)[:, None] * gammaincc(k * shape, np.asarray(thresholds)[None, :] / scale[:, None])
        return out


def _gbm_scores(name: str, model, test: pd.DataFrame, seed: int) -> dict:
    """The scores of ``_scores``, on the same test policies and draw rows."""
    n, s = test["N"].to_numpy(), test["S"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(test), size=min(DRAW_ROWS, len(test)), replace=False)
    picked = test.iloc[rows].reset_index(drop=True)
    total = (n * s)[rows]
    draws = model.draws(picked, DRAWS, np.random.default_rng([seed, 1]))
    if isinstance(model, GBMActuary):
        log_n, log_s = model.log_likelihood(test, n, s)
        expected = model.rate(picked) * model.cost_mean(picked)
    else:
        log_n = log_s = None
        expected = model.mean(picked)
    scores = {
        "nll": None if log_n is None else float(-np.mean(log_n + log_s)),
        "nll_count": None if log_n is None else float(-np.mean(log_n)),
        "nll_cost": None if log_s is None else float(-np.mean(log_s)),
        "claim_predicted": float(np.mean(draws > 0)),
        "claim_observed": float(np.mean(total > 0)),
        "mean_ratio": float(np.mean(draws) / np.mean(total)),
        "mean_ratio_exact": float(np.mean(expected) / np.mean(total)),
        "crps": _crps(draws, total),
        "pit_distance": 100.0 * _pit_distance(draws, total, seed),
        "choices": model.choices,
    }
    for threshold in THRESHOLDS:
        scores[f"above_{int(threshold)}_predicted"] = float(np.mean(draws > threshold))
        scores[f"above_{int(threshold)}_observed"] = float(np.mean(total > threshold))
    return scores


def split(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The benchmark's 80/20 split for ``seed``."""
    order = np.random.default_rng(seed).permutation(len(frame))
    cut = int(0.8 * len(frame))
    return frame.iloc[order[:cut]], frame.iloc[order[cut:]]


def gbm() -> None:
    """Fit and score the LightGBM models on the benchmark's splits; adds them to the results file."""
    arguments = _arguments()
    seeds = SEEDS if "seeds" not in arguments else tuple(int(seed) for seed in arguments["seeds"].split(","))
    frame = policies()
    if "rows" in arguments:
        frame = frame.sample(n=int(arguments["rows"]), random_state=0).reset_index(drop=True)
    for seed in seeds:
        train, test = split(frame, seed)
        for name, build in zip(GBM_MODELS, (GBMActuary, GBMTweedie)):
            start = time.process_time()
            model = build(train, seed)
            seconds = time.process_time() - start
            scores = _gbm_scores(name, model, test, seed)
            scores["cpu"] = seconds
            print(seed, name, json.dumps(scores), flush=True)
            results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
            results.setdefault(str(seed), {})[name] = scores
            RESULTS.parent.mkdir(parents=True, exist_ok=True)
            RESULTS.write_text(json.dumps(results, indent=1))


def run() -> None:
    arguments = _arguments()
    seeds = SEEDS if "seeds" not in arguments else tuple(int(seed) for seed in arguments["seeds"].split(","))
    frame = policies()
    if "rows" in arguments:
        frame = frame.sample(n=int(arguments["rows"]), random_state=0).reset_index(drop=True)
    print(f"{len(frame):,} policies, {np.mean(frame['N'] > 0):.2%} with a claim", flush=True)
    for seed in seeds:
        train, test = split(frame, seed)
        models, seconds = _fit_models(train, seed)
        scores = {}
        for name in MODELS:
            scores[name] = _scores(name, models[name], test, seed)
            scores[name]["cpu"] = seconds[name]
            print(seed, name, json.dumps(scores[name]), flush=True)
        results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        results.setdefault(str(seed), {}).update(scores)
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(results, indent=1))


def _cell(rows, key, form, scale=1.0) -> str:
    values = np.asarray([row[key] for row in rows if row.get(key) is not None], dtype=float) * scale
    if values.size == 0:
        return ""
    if values.size == 1:
        return form.format(values[0])
    return f"{form.format(values.mean())} ± {form.format(values.std())}"


def report() -> None:
    results = json.loads(RESULTS.read_text())
    splits = list(results.values())
    print(f"\n### freMTPL2, {len(splits)} splits, mean ± standard deviation\n")
    print(
        "| Model | Joint NLL ↓ | Count NLL | Cost NLL given the count | P(T > 0), predicted | "
        "P(T > 1,000) | P(T > 5,000) | Mean T / observed | CRPS of T ↓ | PIT distance × 100 ↓ | CPU seconds |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in MODELS + GBM_MODELS:
        rows = [split[name] for split in splits if name in split]
        if not rows:
            continue
        print(
            f"| {name} | {_cell(rows, 'nll', '{:.4f}')} | {_cell(rows, 'nll_count', '{:.4f}')} | "
            f"{_cell(rows, 'nll_cost', '{:.4f}')} | {_cell(rows, 'claim_predicted', '{:.2f}', 100.0)}% | "
            f"{_cell(rows, 'above_1000_predicted', '{:.2f}', 100.0)}% | {_cell(rows, 'above_5000_predicted', '{:.2f}', 100.0)}% | "
            f"{_cell(rows, 'mean_ratio', '{:.3f}')} | {_cell(rows, 'crps', '{:.1f}')} | "
            f"{_cell(rows, 'pit_distance', '{:.2f}')} | {np.mean([row['cpu'] for row in rows]):.0f} |"
        )
    observed = next(iter(splits[0].values()))
    print(
        f"\nObserved: P(T > 0) {100 * observed['claim_observed']:.2f}%, P(T > 1,000) "
        f"{100 * observed['above_1000_observed']:.2f}%, P(T > 5,000) {100 * observed['above_5000_observed']:.2f}% "
        "(split 0)."
    )
    for name in GBM_MODELS:
        for seed, scores in results.items():
            if name in scores:
                print(f"{name}, split {seed}: exact mean T / observed {scores[name]['mean_ratio_exact']:.3f}; "
                      f"choices {scores[name]['choices']}")


STAGES = {"run": run, "gbm": gbm, "report": report}

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "report"
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; choose from {', '.join(STAGES)}")
    STAGES[stage]()
