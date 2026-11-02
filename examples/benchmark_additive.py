"""Neural GAMLSS: additive Ashlar models against R's gamlss, and against black-box Ashlar.

The cheap test of #42. Every distribution parameter of an additive model is a
sum of one function per column. R's ``gamlss`` fits that with penalized
B-splines (``pb``) and backfitting; ``backbone="additive"`` fits it with one
small network per column, trained by gradient descent with the rest of the
head. Two tables, three random 80/20 splits each (seeds 0, 1, 2):

- ``bike``: Capital Bikeshare rentals per hour, 17,379 rows, a count. Every
  model is a negative binomial: Ashlar's ``negative_binomial`` head and
  gamlss's ``NBI``, mean and dispersion each additive in every column. Hour,
  month, and weekday are numeric for Ashlar, which learns their shapes, and
  factors for gamlss. Rentals depend on the hour differently on working days,
  so each additive model is also fitted with that one pair:
  ``pairs=[["hr", "workingday"]]`` for Ashlar, ``hr:workingday`` for gamlss.
  gamlss leaves ``workingday`` out of its main effects, since weekday and
  holiday determine it.
- ``abalone``: rings of 4,177 abalone, treated as continuous. Ashlar's
  ``gaussian`` against gamlss's ``NO``, each with an additive mean and log
  scale, and Ashlar's ``mixture`` against gamlss's ``BCCGo`` (Box-Cox
  Cole-Green, an additive median and scale with a constant skewness).

The black-box Ashlar rows use the default residual network with the same head.
Scored on the test rows: the negative log-likelihood (log mass for the counts,
log density for the rings), the CRPS, and the uncalibrated central 80%
interval's coverage and width. The CRPS sums ``(F(z) - 1{y <= z})^2`` over
the counts 0 to 4,000, and over a grid of rings from 0.01 to 40 in steps of
0.01 times the step; R computes it the same way for gamlss, from its own CDFs.

Stages, each skipping what is already saved:

    JAX_PLATFORMS=cpu python examples/benchmark_additive.py ashlar
    python examples/benchmark_additive.py gamlss      # needs R with gamlss
    python examples/benchmark_additive.py rescore     # scores the saved gamlss fits again, without refitting
    python examples/benchmark_additive.py report
    JAX_PLATFORMS=cpu python examples/benchmark_additive.py figure
    JAX_PLATFORMS=cpu python examples/benchmark_additive.py terms   # needs R with gamlss; the README figure's terms

``terms`` refits Ashlar's additive negative binomial and gamlss's ``NBI``,
without and with the pair, on one bike split (by default the one whose saved
gamlss pair fit was fastest, which favours gamlss), and writes every term of
the log mean and the dispersion on a grid of its columns, centred on the
training rows, with the saved scores of every split, to
``examples/readme/artifacts/additive-gamlss/terms.json``.

``gamlss`` runs ``Rscript`` with the user library first on its path. Results
go to ``$ASHLAR_BENCH/additive`` (default ``/tmp/ashlar-bench/additive``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import abalone  # noqa: E402
from benchmark_uncertainty import bike  # noqa: E402

from ashlar import ASRRegressor  # noqa: E402

ROOT = Path(os.environ.get("ASHLAR_BENCH", "/tmp/ashlar-bench")) / "additive"
SEEDS = (0, 1, 2)
COUNT_GRID = np.arange(0, 4001, dtype=np.float64)
RING_STEP = 0.01
RING_GRID = np.round(np.arange(1, 4001) * RING_STEP, 2)
PAIRS = {"bike": [["hr", "workingday"]]}

# Ashlar models per table: (name, task, backbone, pairs).
ASHLAR = {
    "bike": [
        ("ashlar_additive", "negative_binomial", "additive", None),
        ("ashlar_additive_pair", "negative_binomial", "additive", PAIRS["bike"]),
        ("ashlar_blackbox", "negative_binomial", "residual_mlp", None),
    ],
    "abalone": [
        ("ashlar_additive_gaussian", "gaussian", "additive", None),
        ("ashlar_blackbox_gaussian", "gaussian", "residual_mlp", None),
        ("ashlar_additive_mixture", "mixture", "additive", None),
        ("ashlar_blackbox_mixture", "mixture", "residual_mlp", None),
    ],
}
GAMLSS = {"bike": ["gamlss_nbi", "gamlss_nbi_pair"], "abalone": ["gamlss_no", "gamlss_bccgo"]}
# Columns gamlss reads as factors.
FACTORS = {
    "bike": ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit"],
    "abalone": ["sex"],
}
LABELS = {
    "ashlar_additive": "Ashlar, additive",
    "ashlar_additive_pair": "Ashlar, additive + hour × working day",
    "ashlar_blackbox": "Ashlar, residual network",
    "gamlss_nbi": "gamlss `NBI`, additive",
    "gamlss_nbi_pair": "gamlss `NBI`, additive + hour × working day",
    "ashlar_additive_gaussian": "Ashlar `gaussian`, additive",
    "ashlar_blackbox_gaussian": "Ashlar `gaussian`, residual network",
    "gamlss_no": "gamlss `NO`, additive",
    "ashlar_additive_mixture": "Ashlar `mixture`, additive",
    "ashlar_blackbox_mixture": "Ashlar `mixture`, residual network",
    "gamlss_bccgo": "gamlss `BCCGo`, additive, constant skewness",
}


def table(name: str):
    if name == "bike":
        return bike("cnt")
    if name == "abalone":
        return abalone()
    raise ValueError(name)


def split(name: str, seed: int):
    X, y = table(name)
    train, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
    return X.iloc[train].reset_index(drop=True), y[train], X.iloc[test].reset_index(drop=True), y[test]


def _path(name: str, seed: int, model: str) -> Path:
    return ROOT / f"{name}-{seed}-{model}.json"


def _grid(name: str) -> tuple[np.ndarray, float]:
    return (COUNT_GRID, 1.0) if name == "bike" else (RING_GRID, RING_STEP)


def crps_from_cdf(cdf: np.ndarray, y: np.ndarray, grid: np.ndarray, step: float) -> np.ndarray:
    """``sum_z (F(z) - 1{y <= z})^2 * step`` per row, ``cdf`` of shape ``(rows, grid)``."""
    below = (y[:, None] <= grid[None, :]).astype(np.float64)
    return np.sum(np.square(cdf - below), axis=1) * step


def _cdf_on_grid(dist, grid: np.ndarray, block: int = 250) -> np.ndarray:
    pieces = []
    for start in range(0, len(grid), block):
        values = grid[start : start + block][:, None]
        pieces.append(np.asarray(dist.cdf(values), dtype=np.float64).T)
    return np.concatenate(pieces, axis=1)


def _summary(nll, crps, low, high, y, seconds) -> dict:
    return {
        "nll": float(np.mean(nll)),
        "crps": float(np.mean(crps)),
        "coverage": float(np.mean((low <= y) & (y <= high))),
        "width": float(np.mean(high - low)),
        "seconds": float(seconds),
        "rows": int(len(y)),
    }


# ---------------------------------------------------------------- Ashlar


def _model(task: str, backbone: str, pairs, seed: int) -> ASRRegressor:
    kwargs = {} if pairs is None else {"backbone_kwargs": {"pairs": pairs}}
    return ASRRegressor(task=task, backbone=backbone, random_state=seed, **kwargs)


def run_ashlar(tables, seeds):
    ROOT.mkdir(parents=True, exist_ok=True)
    for name in tables:
        for seed in seeds:
            X_train, y_train, X_test, y_test = split(name, seed)
            grid, step = _grid(name)
            for model_name, task, backbone, pairs in ASHLAR[name]:
                path = _path(name, seed, model_name)
                if path.exists():
                    continue
                started = time.perf_counter()
                model = _model(task, backbone, pairs, seed).fit(X_train, y_train)
                seconds = time.perf_counter() - started
                nll = -np.asarray(model.log_prob(X_test, y_test), dtype=np.float64)
                cdf = _cdf_on_grid(model.predict_dist(X_test), grid)
                crps = crps_from_cdf(cdf, y_test, grid, step)
                interval = np.asarray(model.predict_interval(X_test, 0.8), dtype=np.float64)
                result = _summary(nll, crps, interval[:, 0], interval[:, 1], y_test, seconds)
                path.write_text(json.dumps(result))
                print(name, seed, model_name, json.dumps({key: round(value, 4) for key, value in result.items()}), flush=True)


# ---------------------------------------------------------------- gamlss

# Shared by the fit and the rescoring scripts: the grid, the CRPS, and every score of a fit's per-row parameters.
R_SCORING = r"""
args <- commandArgs(trailingOnly = TRUE)
name <- args[1]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(gamlss))
if (name == "bike") { grid <- 0:4000; step <- 1 } else { grid <- (1:4000) * 0.01; step <- 0.01 }
crps <- function(cdf, y) rowSums((cdf - outer(y, grid, "<="))^2) * step
# Each row's CDF at every grid point. gamlss.dist's p functions do not recycle q against parameters of another
# length (pNBI keeps sigma's length, so every grid point got the CDF at 0), so every argument is given in full.
on_grid <- function(pfun, ...) {
  parameters <- list(...)
  n <- length(parameters[[1]])
  full <- lapply(parameters, rep, times = length(grid))
  matrix(do.call(pfun, c(list(rep(grid, each = n)), full)), nrow = n)
}
score <- function(model, y, mu, sigma, nu = NULL) {
  if (model %in% c("gamlss_nbi", "gamlss_nbi_pair")) {
    nll <- -dNBI(y, mu = mu, sigma = sigma, log = TRUE)
    cdf <- on_grid(pNBI, mu = mu, sigma = sigma)
    low <- qNBI(0.1, mu = mu, sigma = sigma); high <- qNBI(0.9, mu = mu, sigma = sigma)
  } else if (model == "gamlss_no") {
    nll <- -dNO(y, mu = mu, sigma = sigma, log = TRUE)
    cdf <- on_grid(pNO, mu = mu, sigma = sigma)
    low <- qNO(0.1, mu = mu, sigma = sigma); high <- qNO(0.9, mu = mu, sigma = sigma)
  } else {
    nll <- -dBCCGo(y, mu = mu, sigma = sigma, nu = nu, log = TRUE)
    cdf <- on_grid(pBCCGo, mu = mu, sigma = sigma, nu = nu)
    low <- qBCCGo(0.1, mu = mu, sigma = sigma, nu = nu); high <- qBCCGo(0.9, mu = mu, sigma = sigma, nu = nu)
  }
  data.frame(nll = nll, crps = crps(cdf, y), low = low, high = high, y = y,
             mu = mu, sigma = sigma, nu = if (is.null(nu)) NA else nu)
}
"""

R_FIT = r"""
train_path <- args[2]; test_path <- args[3]; out_prefix <- args[4]
models <- strsplit(args[5], ",")[[1]]
train <- read.csv(train_path, stringsAsFactors = FALSE)
test <- read.csv(test_path, stringsAsFactors = FALSE)
factors <- strsplit(args[6], ",")[[1]]
for (f in factors) {
  levels <- sort(unique(as.character(train[[f]])))
  train[[f]] <- factor(as.character(train[[f]]), levels = levels)
  test[[f]] <- factor(as.character(test[[f]]), levels = levels)
}
if (name == "bike") {
  main <- "season + yr + mnth + hr + holiday + weekday + weathersit + pb(temp) + pb(atemp) + pb(hum) + pb(windspeed)"
} else {
  main <- "sex + pb(length) + pb(diameter) + pb(height) + pb(whole) + pb(shucked) + pb(viscera) + pb(shell)"
}
response <- if (name == "bike") "cnt" else "rings"
y <- test[[response]]
for (model in models) {
  terms <- if (model == "gamlss_nbi_pair") paste(main, "+ hr:workingday") else main
  mu <- as.formula(paste(response, "~", terms))
  sigma <- as.formula(paste("~", terms))
  family <- switch(model, gamlss_nbi = NBI(), gamlss_nbi_pair = NBI(), gamlss_no = NO(), gamlss_bccgo = BCCGo())
  started <- Sys.time()
  fit <- tryCatch(
    gamlss(mu, sigma.formula = sigma, nu.formula = ~1, family = family, data = train,
           control = gamlss.control(n.cyc = 200, trace = FALSE)),
    error = function(e) { message(model, " failed: ", conditionMessage(e)); NULL })
  if (is.null(fit)) next
  seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  p <- predictAll(fit, newdata = test, data = train, type = "response")
  write.csv(score(model, y, p$mu, p$sigma, p$nu), paste0(out_prefix, "-", model, ".csv"), row.names = FALSE)
  cat(model, "seconds", seconds, "df", fit$df.fit, "deviance", deviance(fit), "\n")
  writeLines(as.character(seconds), paste0(out_prefix, "-", model, ".seconds"))
}
"""

# The README figure's fits: each NBI fit's terms of log mu and log sigma on its own training rows.
R_TERMS = r"""
train_path <- args[2]; out_prefix <- args[3]
models <- strsplit(args[4], ",")[[1]]
factors <- strsplit(args[5], ",")[[1]]
train <- read.csv(train_path, stringsAsFactors = FALSE)
for (f in factors) {
  train[[f]] <- factor(as.character(train[[f]]), levels = sort(unique(as.character(train[[f]]))))
}
main <- "season + yr + mnth + hr + holiday + weekday + weathersit + pb(temp) + pb(atemp) + pb(hum) + pb(windspeed)"
for (model in models) {
  terms <- if (model == "gamlss_nbi_pair") paste(main, "+ hr:workingday") else main
  started <- Sys.time()
  fit <- tryCatch(
    gamlss(as.formula(paste("cnt ~", terms)), sigma.formula = as.formula(paste("~", terms)), nu.formula = ~1,
           family = NBI(), data = train, control = gamlss.control(n.cyc = 200, trace = FALSE)),
    error = function(e) { message(model, " failed: ", conditionMessage(e)); NULL })
  if (is.null(fit)) next
  seconds <- as.numeric(difftime(Sys.time(), started, units = "secs"))
  for (what in c("mu", "sigma")) {
    # Without newdata, predict(type = "terms") is lpred's: each term's part of the linear predictor on the
    # training rows, the design centred on its column means, plus the term's smoother.
    values <- predict(fit, what = what, type = "terms")
    out <- data.frame(values, check.names = FALSE)
    out[[".constant"]] <- attr(values, "constant")
    out[[".lp"]] <- fit[[paste0(what, ".lp")]]
    write.csv(out, paste0(out_prefix, "-", model, "-", what, "-terms.csv"), row.names = FALSE)
  }
  writeLines(c(as.character(seconds), as.character(isTRUE(fit$converged)), as.character(fit$iter),
               as.character(deviance(fit))), paste0(out_prefix, "-", model, "-fit.txt"))
  cat(model, "seconds", seconds, "converged", isTRUE(fit$converged), "cycles", fit$iter, "\n")
}
"""

R_RESCORE = r"""
rows_path <- args[2]; model <- args[3]
rows <- read.csv(rows_path)
nu <- if (all(is.na(rows$nu))) NULL else rows$nu
write.csv(score(model, rows$y, rows$mu, rows$sigma, nu), rows_path, row.names = FALSE)
"""


def _gamlss_frame(name: str, X_train, y_train, X_test, y_test):
    """The split as gamlss reads it: a test category unseen in training takes the training mode."""
    response = "cnt" if name == "bike" else "rings"
    train = X_train.copy()
    test = X_test.copy()
    for column in FACTORS[name]:
        train[column] = train[column].astype(str)
        test[column] = test[column].astype(str)
        known = set(train[column])
        mode = train[column].mode().iloc[0]
        test[column] = [value if value in known else mode for value in test[column]]
    train[response] = y_train
    test[response] = y_test
    return train, test


def run_gamlss(tables, seeds):
    ROOT.mkdir(parents=True, exist_ok=True)
    script = ROOT / "fit_gamlss.R"
    script.write_text(R_SCORING + R_FIT)
    for name in tables:
        for seed in seeds:
            missing = [model for model in GAMLSS[name] if not _path(name, seed, model).exists()]
            if not missing:
                continue
            X_train, y_train, X_test, y_test = split(name, seed)
            train, test = _gamlss_frame(name, X_train, y_train, X_test, y_test)
            prefix = ROOT / f"{name}-{seed}"
            train.to_csv(f"{prefix}-train.csv", index=False)
            test.to_csv(f"{prefix}-test.csv", index=False)
            command = [
                "Rscript", str(script), name, f"{prefix}-train.csv", f"{prefix}-test.csv", str(prefix),
                ",".join(missing), ",".join(FACTORS[name]),
            ]
            completed = subprocess.run(command, capture_output=True, text=True)
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr[-4000:])
            for model in missing:
                _gamlss_result(name, seed, model, prefix)


def rescore_gamlss(tables, seeds):
    """Score every saved gamlss fit again from its per-row parameters, without refitting."""
    ROOT.mkdir(parents=True, exist_ok=True)
    script = ROOT / "score_gamlss.R"
    script.write_text(R_SCORING + R_RESCORE)
    for name in tables:
        for seed in seeds:
            prefix = ROOT / f"{name}-{seed}"
            for model in GAMLSS[name]:
                rows_path = Path(f"{prefix}-{model}.csv")
                if not rows_path.exists():
                    continue
                completed = subprocess.run(["Rscript", str(script), name, str(rows_path), model], capture_output=True, text=True)
                sys.stderr.write(completed.stderr[-4000:])
                _gamlss_result(name, seed, model, prefix)


def _gamlss_result(name: str, seed: int, model: str, prefix: Path):
    """Summarize one gamlss fit's per-row scores into its result file."""
    rows_path = Path(f"{prefix}-{model}.csv")
    if not rows_path.exists():
        print(name, seed, model, "no result", flush=True)
        return
    rows = pd.read_csv(rows_path)
    seconds = float(Path(f"{prefix}-{model}.seconds").read_text())
    result = _summary(
        rows["nll"].to_numpy(), rows["crps"].to_numpy(), rows["low"].to_numpy(), rows["high"].to_numpy(),
        rows["y"].to_numpy(), seconds,
    )
    result["check"] = _python_check(name, model, rows)
    _path(name, seed, model).write_text(json.dumps(result))
    print(name, seed, model, json.dumps({key: round(value, 4) for key, value in result.items()}), flush=True)


def _python_check(name: str, model: str, rows: pd.DataFrame) -> float:
    """The largest gap between R's per-row CRPS and Python's from R's parameters, for the families SciPy has."""
    from scipy import stats

    grid, step = _grid(name)
    y = rows["y"].to_numpy(np.float64)
    mu = rows["mu"].to_numpy(np.float64)[:, None]
    sigma = rows["sigma"].to_numpy(np.float64)[:, None]
    if model.startswith("gamlss_nbi"):
        # NBI: variance mu + sigma mu^2, so SciPy's n = 1 / sigma and p = 1 / (1 + sigma mu). Like R's pNBI, a
        # sigma below 1e-4 is read as a Poisson: at n = 1 / sigma that large, SciPy's nbinom loses its precision.
        poisson = sigma < 1e-4
        safe = np.where(poisson, 1.0, sigma)
        cdf = np.where(
            poisson,
            stats.poisson.cdf(grid[None, :], mu),
            stats.nbinom.cdf(grid[None, :], 1.0 / safe, 1.0 / (1.0 + safe * mu)),
        )
    elif model == "gamlss_no":
        cdf = stats.norm.cdf(grid[None, :], mu, sigma)
    else:
        return float("nan")
    return float(np.max(np.abs(crps_from_cdf(cdf, y, grid, step) - rows["crps"].to_numpy())))


# ---------------------------------------------------------------- terms, for the README figure

BIKE_MODELS = ("ashlar_additive", "gamlss_nbi", "ashlar_additive_pair", "gamlss_nbi_pair", "ashlar_blackbox")


def _saved_runs(name: str, models) -> dict:
    """Every saved split's summary of each model: the benchmark's numbers, read and not refitted."""
    out = {}
    for model in models:
        for seed in SEEDS:
            path = _path(name, seed, model)
            if path.exists():
                out.setdefault(model, {})[str(seed)] = json.loads(path.read_text())
    return out


def _terms_seed(saved: dict) -> tuple[int, str]:
    """The split whose saved gamlss pair fit was fastest: gamlss's best-behaved split, so the choice favours it."""
    pair = saved.get("gamlss_nbi_pair", {})
    if not pair:
        return 0, "seed 0: no saved gamlss pair fit to choose by"
    seed = min(pair, key=lambda key: pair[key]["seconds"])
    return int(seed), f"the split whose saved gamlss pair fit was fastest ({pair[seed]['seconds']:.0f} s)"


def _term_grid(values: np.ndarray, frame: pd.DataFrame, columns) -> dict:
    """One term's value at each distinct value of its columns, and the rows there."""
    columns = list(columns)
    table = frame[columns].copy()
    for column in columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].astype(str)
    table["value"] = np.asarray(values, dtype=np.float64)
    grouped = table.groupby(columns, sort=True)["value"]
    keys = list(grouped.groups)
    to_python = (lambda key: [_plain(part) for part in key]) if len(columns) > 1 else _plain
    spread = grouped.std(ddof=0).fillna(0.0)
    return {
        "columns": columns,
        "x": [to_python(key) for key in keys],
        "y": [round(float(value), 6) for value in grouped.mean().to_numpy()],
        "rows": [int(value) for value in grouped.size().to_numpy()],
        "spread": float(spread.max()),
    }


def _plain(value):
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return str(value)


def _gamlss_columns(label: str) -> tuple[str, ...]:
    """The columns of X behind a gamlss term label: ``pb(temp)`` is ``temp``, ``hr:workingday`` two columns."""
    if label.startswith("pb(") and label.endswith(")"):
        return (label[3:-1],)
    return tuple(label.split(":"))


# Terms read together, as one shape against the first column: the temperature and the felt temperature move together,
# so each model splits their effect between them in its own way.
COMBINED = {"temp+atemp": ("temp", "atemp")}


def _term_tables(values: dict, columns: dict, outputs, frame: pd.DataFrame) -> tuple[dict, dict]:
    """Each term's grid on every output, the combined terms, and each term's importance (mean |value| over rows).

    ``values[name]`` is ``(rows, outputs)``, centred on the rows; ``columns[name]`` its columns of ``frame``.
    """
    terms, importance = {}, {}
    pieces = dict(values)
    for combined, parts in COMBINED.items():
        if all(part in values for part in parts):
            pieces[combined] = sum(values[part] for part in parts)
            columns = {**columns, combined: (parts[0],)}
    for name, value in pieces.items():
        entry = None
        for position, output_name in enumerate(outputs):
            cell = _term_grid(value[:, position], frame, columns[name])
            if entry is None:
                entry = {key: cell[key] for key in ("columns", "x", "rows")}
                entry["spread"] = 0.0
            entry[output_name] = cell["y"]
            if name not in COMBINED:
                entry["spread"] = max(entry["spread"], cell["spread"])
        terms[name] = entry
        if name not in COMBINED:
            importance[name] = [round(float(np.mean(np.abs(value[:, position]))), 6) for position in range(len(outputs))]
    return terms, importance


def run_terms(seed: int | None, output: Path):
    """Ashlar's additive negative binomial and gamlss's NBI on one bike split, each term on a grid of its columns.

    Both are fitted as the benchmark fits them, without and with the hour ×
    working day pair: Ashlar's terms come from ``shape_functions`` on the
    training rows, gamlss's from ``predict(..., what=, type="terms")`` on the
    same rows. Every term is centred on the training rows, so the curves
    compare directly. Ashlar's dispersion terms are on its raw scale, before
    the softplus that makes the dispersion; gamlss's on log sigma (NBI's
    sigma is Ashlar's dispersion, the variance being mean + dispersion ×
    mean²). The benchmark's saved scores and seconds for every split go in too.
    """
    import shutil

    ROOT.mkdir(parents=True, exist_ok=True)
    rscript = os.environ.get("RSCRIPT") or shutil.which("Rscript")
    if rscript is None:
        raise SystemExit("Rscript is not on PATH; set RSCRIPT to its path")
    saved = _saved_runs("bike", BIKE_MODELS)
    if seed is None:
        seed, rule = _terms_seed(saved)
    else:
        rule = f"seed {seed}, as asked"
    print(f"terms on bike split {seed}: {rule}", flush=True)
    X_train, y_train, X_test, y_test = split("bike", seed)
    grid, step = _grid("bike")
    out = {"seed": seed, "seed_rule": rule, "train_rows": len(y_train), "ashlar": {}, "gamlss": {}, "benchmark": saved}

    for model_name, pairs in (("ashlar_additive", None), ("ashlar_additive_pair", PAIRS["bike"])):
        started = time.perf_counter()
        model = _model("negative_binomial", "additive", pairs, seed).fit(X_train, y_train)
        seconds = time.perf_counter() - started
        shapes = model.shape_functions(X_train)
        crps = float(np.mean(crps_from_cdf(_cdf_on_grid(model.predict_dist(X_test), grid), y_test, grid, step)))
        values = {name: shapes.values[:, index, :] for index, name in enumerate(shapes.names)}
        terms, importance = _term_tables(values, dict(zip(shapes.names, shapes.columns)), shapes.outputs, X_train)
        saved_crps = saved.get(model_name, {}).get(str(seed), {}).get("crps")
        out["ashlar"][model_name] = {
            "seconds": round(seconds, 2),
            "outputs": list(shapes.outputs),
            "intercept": [round(float(value), 6) for value in shapes.intercept],
            "terms": terms,
            "importance": importance,
            "test_crps": round(crps, 4),
            "saved_test_crps": saved_crps,
        }
        print(model_name, f"{seconds:.1f} s, test CRPS {crps:.3f} (saved {saved_crps})", flush=True)

    script = ROOT / "terms_gamlss.R"
    script.write_text(R_SCORING + R_TERMS)
    train, _test = _gamlss_frame("bike", X_train, y_train, X_test, y_test)
    prefix = ROOT / f"terms-bike-{seed}"
    train.to_csv(f"{prefix}-train.csv", index=False)
    models = ["gamlss_nbi", "gamlss_nbi_pair"]
    for model_name in models:
        for stale in ROOT.glob(f"{prefix.name}-{model_name}-*"):
            stale.unlink()
    command = [rscript, str(script), "bike", f"{prefix}-train.csv", str(prefix), ",".join(models), ",".join(FACTORS["bike"])]
    completed = subprocess.run(command, capture_output=True, text=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr[-4000:])
    outputs = ("log_mean", "log_dispersion")
    for model_name in models:
        fit_path = Path(f"{prefix}-{model_name}-fit.txt")
        if not fit_path.exists():
            print(model_name, "no fit", flush=True)
            continue
        seconds, converged, cycles, deviance = fit_path.read_text().split()
        tables = [pd.read_csv(f"{prefix}-{model_name}-{what}-terms.csv") for what in ("mu", "sigma")]
        labels = [label for label in tables[0].columns if not label.startswith(".")]
        if [label for label in tables[1].columns if not label.startswith(".")] != labels:
            raise SystemExit(f"{model_name}: the mu and sigma formulas gave different terms")
        raw = np.stack([table[labels].to_numpy(np.float64) for table in tables], axis=-1)  # (rows, terms, outputs)
        means = raw.mean(axis=0)
        gaps = [float(np.abs(table[".constant"].to_numpy() + raw[:, :, k].sum(axis=1) - table[".lp"].to_numpy()).max())
                for k, table in enumerate(tables)]
        names = [":".join(_gamlss_columns(label)) for label in labels]
        values = {name: raw[:, index, :] - means[index] for index, name in enumerate(names)}
        columns = {name: _gamlss_columns(label) for name, label in zip(names, labels)}
        terms, importance = _term_tables(values, columns, outputs, X_train)
        out["gamlss"][model_name] = {
            "seconds": round(float(seconds), 1),
            "converged": converged == "TRUE",
            "cycles": int(cycles),
            "deviance": float(deviance),
            "outputs": list(outputs),
            "labels": dict(zip(names, labels)),
            "intercept": [round(float(table[".constant"].iloc[0] + means[:, k].sum()), 6) for k, table in enumerate(tables)],
            "lp_check": gaps,
            "terms": terms,
            "importance": importance,
        }
        print(model_name, f"{float(seconds):.0f} s, converged {converged}, {cycles} cycles, lp check {max(gaps):.1e}",
              flush=True)

    output.mkdir(parents=True, exist_ok=True)
    path = output / "terms.json"
    path.write_text(json.dumps(out, indent=None, separators=(",", ":")))
    print("wrote", path, f"({path.stat().st_size / 1024:.0f} KB)", flush=True)


# ---------------------------------------------------------------- report


def report():
    for name in ASHLAR:
        models = [model for model, *_ in ASHLAR[name]] + GAMLSS[name]
        print(f"\n{name}\n")
        print("| Model | NLL | CRPS | 80% coverage | 80% width | Seconds per fit |")
        print("|---|---:|---:|---:|---:|---:|")
        for model in models:
            runs = [json.loads(_path(name, seed, model).read_text()) for seed in SEEDS if _path(name, seed, model).exists()]
            if not runs:
                continue

            def cell(key, digits):
                values = np.asarray([run[key] for run in runs])
                spread = f" ± {values.std(ddof=1) / np.sqrt(len(values)):.{digits}f}" if len(values) > 1 else ""
                return f"{values.mean():.{digits}f}{spread}"

            coverage = np.mean([run["coverage"] for run in runs])
            seconds = np.mean([run["seconds"] for run in runs])
            note = "" if len(runs) == len(SEEDS) else f" ({len(runs)} splits)"
            print(
                f"| {LABELS[model]}{note} | {cell('nll', 3)} | {cell('crps', 3 if name == 'abalone' else 2)} | "
                f"{coverage:.1%} | {cell('width', 2)} | {seconds:.0f} |"
            )
            checks = [run["check"] for run in runs if "check" in run and np.isfinite(run["check"])]
            if checks:
                print(f"<!-- {model}: R and Python CRPS agree to {max(checks):.2e} -->")


# ---------------------------------------------------------------- figure


def figure(output: Path):
    """Shape functions of the additive negative binomial on bike rentals, and its calibrated quantiles by temperature."""
    import matplotlib.pyplot as plt

    X, y = table("bike")
    rest, test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0)
    train, cal = train_test_split(rest, test_size=0.2, random_state=0)
    model = _model("negative_binomial", "additive", PAIRS["bike"], 0).fit(X.iloc[train], y[train])
    model.calibrate(X.iloc[cal], y[cal])
    frame = X.iloc[train].reset_index(drop=True)
    shapes = model.shape_functions(frame)
    interval = model.predict_interval(X.iloc[test], 0.8)
    held = float(np.mean((interval[:, 0] <= y[test]) & (y[test] <= interval[:, 1])))
    print(f"calibrated 80% interval holds {held:.1%} of {len(test)} test hours")
    print("importance on the log mean", {key: round(value, 3) for key, value in shapes.importance("log_mean").items()})
    print("importance on the dispersion", {key: round(value, 3) for key, value in shapes.importance("dispersion").items()})

    ink, muted, hair, teal, clay = "#1a1916", "#5e5a54", "#e4e0da", "#0f6e68", "#d4471c"
    plt.rcParams.update({"font.family": ["Inter", "DejaVu Sans"], "font.size": 9.5, "text.color": ink, "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.6), constrained_layout=True)

    def sorted_line(ax, column, values, color, label=None):
        order = np.argsort(frame[column].to_numpy(), kind="stable")
        ax.plot(frame[column].to_numpy()[order], values[order], color=color, lw=1.9, label=label)

    ax = axes[0, 0]
    hours = np.arange(24)
    for workingday, color, label in ((1, teal, "working day"), (0, clay, "weekend or holiday")):
        rows = frame[frame["workingday"] == workingday]
        part = model.shape_functions(rows, center=False)
        combined = part.term("hr", "log_mean") + part.term("hr:workingday", "log_mean") + part.term("workingday", "log_mean")
        means = np.asarray([combined[rows["hr"].to_numpy() == hour].mean() for hour in hours])
        ax.plot(hours, means - means.mean(), color=color, lw=1.9, label=label)
    ax.set_title("Hour, with its pair, on the log mean", loc="left")
    ax.set_xlabel("hour of the day", color=muted)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[0, 1]
    sorted_line(ax, "temp", shapes.term("temp", "log_mean"), teal, "temperature")
    sorted_line(ax, "atemp", shapes.term("atemp", "log_mean"), clay, "felt temperature")
    ax.set_title("Temperature on the log mean", loc="left")
    ax.set_xlabel("temperature, normalized", color=muted)
    ax.legend(frameon=False, fontsize=8.5, loc="lower center")

    ax = axes[1, 0]
    sorted_line(ax, "hum", shapes.term("hum", "log_mean"), teal, "log mean")
    sorted_line(ax, "hum", shapes.term("hum", "dispersion"), clay, "dispersion, before its softplus")
    ax.set_title("Humidity on the log mean and the dispersion", loc="left")
    ax.set_xlabel("humidity, normalized", color=muted)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")

    ax = axes[1, 1]
    base = frame[(frame["workingday"] == 1) & (frame["hr"] == 17)].iloc[[0]]
    temps = np.linspace(frame["temp"].quantile(0.02), frame["temp"].quantile(0.98), 40)
    # The felt temperature moves with the temperature, on the training rows' line.
    slope, intercept = np.polyfit(frame["temp"].to_numpy(), frame["atemp"].to_numpy(), 1)
    rows = pd.concat([base] * len(temps), ignore_index=True).assign(temp=temps, atemp=intercept + slope * temps)
    quantiles = model.predict_quantiles(rows, [0.1, 0.5, 0.9])
    ax.fill_between(temps, quantiles[:, 0], quantiles[:, 2], color=teal, alpha=0.16, lw=0, label="calibrated 80% interval")
    ax.plot(temps, quantiles[:, 1], color=teal, lw=1.9, label="median")
    ax.set_title("Rentals at 5 pm on a working day", loc="left")
    ax.set_xlabel("temperature, normalized", color=muted)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax in axes.flat:
        ax.tick_params(colors=muted, labelsize=8.5)
        ax.grid(color=hair, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(hair)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="#ffffff")
    print("wrote", output)


def main(argv):
    stages = [arg for arg in argv if "=" not in arg] or ["ashlar", "report"]
    options = dict(arg.split("=", 1) for arg in argv if "=" in arg)
    tables = options.get("tables", "bike,abalone").split(",")
    seeds = tuple(int(seed) for seed in options.get("seeds", "0,1,2").split(","))
    for stage in stages:
        if stage == "ashlar":
            run_ashlar(tables, seeds)
        elif stage == "gamlss":
            run_gamlss(tables, seeds)
        elif stage == "rescore":
            rescore_gamlss(tables, seeds)
        elif stage == "report":
            report()
        elif stage == "figure":
            figure(Path(options.get("output", "docs/images/additive-bike.png")))
        elif stage == "terms":
            seed = int(options["seed"]) if "seed" in options else None
            run_terms(seed, Path(options.get("output", "examples/readme/artifacts/additive-gamlss")))
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
