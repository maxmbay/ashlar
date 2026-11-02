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
    python examples/benchmark_additive.py report
    JAX_PLATFORMS=cpu python examples/benchmark_additive.py figure

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

R_SCRIPT = r"""
args <- commandArgs(trailingOnly = TRUE)
name <- args[1]; train_path <- args[2]; test_path <- args[3]; out_prefix <- args[4]
models <- strsplit(args[5], ",")[[1]]
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(gamlss))
train <- read.csv(train_path, stringsAsFactors = FALSE)
test <- read.csv(test_path, stringsAsFactors = FALSE)
factors <- strsplit(args[6], ",")[[1]]
for (f in factors) {
  levels <- sort(unique(as.character(train[[f]])))
  train[[f]] <- factor(as.character(train[[f]]), levels = levels)
  test[[f]] <- factor(as.character(test[[f]]), levels = levels)
}
crps <- function(cdf, y, grid, step) rowSums((cdf - outer(y, grid, "<="))^2) * step
on_grid <- function(pfun, grid, ...) {
  n <- length(list(...)[[1]])
  matrix(pfun(rep(grid, each = n), ...), nrow = n)
}
if (name == "bike") {
  main <- "season + yr + mnth + hr + holiday + weekday + weathersit + pb(temp) + pb(atemp) + pb(hum) + pb(windspeed)"
  grid <- 0:4000; step <- 1
} else {
  main <- "sex + pb(length) + pb(diameter) + pb(height) + pb(whole) + pb(shucked) + pb(viscera) + pb(shell)"
  grid <- (1:4000) * 0.01; step <- 0.01
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
  if (model %in% c("gamlss_nbi", "gamlss_nbi_pair")) {
    nll <- -dNBI(y, mu = p$mu, sigma = p$sigma, log = TRUE)
    cdf <- on_grid(pNBI, grid, mu = p$mu, sigma = p$sigma)
    low <- qNBI(0.1, mu = p$mu, sigma = p$sigma); high <- qNBI(0.9, mu = p$mu, sigma = p$sigma)
  } else if (model == "gamlss_no") {
    nll <- -dNO(y, mu = p$mu, sigma = p$sigma, log = TRUE)
    cdf <- on_grid(pNO, grid, mu = p$mu, sigma = p$sigma)
    low <- qNO(0.1, mu = p$mu, sigma = p$sigma); high <- qNO(0.9, mu = p$mu, sigma = p$sigma)
  } else {
    nll <- -dBCCGo(y, mu = p$mu, sigma = p$sigma, nu = p$nu, log = TRUE)
    cdf <- on_grid(pBCCGo, grid, mu = p$mu, sigma = p$sigma, nu = p$nu)
    low <- qBCCGo(0.1, mu = p$mu, sigma = p$sigma, nu = p$nu); high <- qBCCGo(0.9, mu = p$mu, sigma = p$sigma, nu = p$nu)
  }
  rows <- data.frame(nll = nll, crps = crps(cdf, y, grid, step), low = low, high = high, y = y,
                     mu = p$mu, sigma = p$sigma, nu = if (is.null(p$nu)) NA else p$nu)
  write.csv(rows, paste0(out_prefix, "-", model, ".csv"), row.names = FALSE)
  cat(model, "seconds", seconds, "df", fit$df.fit, "deviance", deviance(fit), "\n")
  writeLines(as.character(seconds), paste0(out_prefix, "-", model, ".seconds"))
}
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
    script.write_text(R_SCRIPT)
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
                rows_path = Path(f"{prefix}-{model}.csv")
                if not rows_path.exists():
                    print(name, seed, model, "no result", flush=True)
                    continue
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
        # NBI: variance mu + sigma mu^2, so SciPy's n = 1 / sigma and p = 1 / (1 + sigma mu).
        cdf = stats.nbinom.cdf(grid[None, :], 1.0 / sigma, 1.0 / (1.0 + sigma * mu))
    elif model == "gamlss_no":
        cdf = stats.norm.cdf(grid[None, :], mu, sigma)
    else:
        return float("nan")
    return float(np.max(np.abs(crps_from_cdf(cdf, y, grid, step) - rows["crps"].to_numpy())))


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
        elif stage == "report":
            report()
        elif stage == "figure":
            figure(Path(options.get("output", "docs/images/additive-bike.png")))
        else:
            raise SystemExit(f"unknown stage {stage!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
