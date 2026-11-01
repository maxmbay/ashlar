"""Uncertain inputs: do missing and noisy features widen the interval enough?

Issue #35's cheap test. A fitted model fills a missing numeric cell with the
training median, and its interval is as narrow as if the value were known.
``with_uncertain_inputs(draws=32)`` draws each missing cell from a Gaussian
copula of the numeric features given the row's observed cells, and mixes
the predictive distributions of the completions.

**mask** (California housing and hourly bike rentals, seeds 0 to 2). A
``mixture`` head with default settings is fitted on 64% of the rows,
calibrated on 11% (complete rows, as a user would), and tested on 25%,
where 0, 10, 30, and 50% of the numeric cells are masked at random.
Scored: coverage of the calibrated 80% interval, its mean width, and the
NLL of the test target, with median imputation and with 32 draws.

**noise** (same tables and splits). Gaussian noise with a standard
deviation of 0.25, 0.5, and 1 times the column's own is added at test time to
the feature the model leans on most (``median_income``; ``temp`` for bikes).
Scored the same, ignoring the noise and declaring it with
``noise={column: sd}``.

**shift** (same). ``input_log_density`` of the test rows against rows
from outside the training range (California rows with ``median_income``
above the training 95th percentile held out of training), as a ranking
score: the area under the ROC curve of telling the two apart.

    python examples/benchmark_inputs.py mask noise shift [--seeds 0 1 2] [--out inputs.json]
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

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import california
from benchmark_uncertainty import bike

from ashlar import ASRRegressor

RATES = (0.0, 0.1, 0.3, 0.5)
NOISE = (0.25, 0.5, 1.0)
DRAWS = 32
EPOCHS = 60
TABLES = {
    "california": (california, "median_income", {}),
    "bike": (lambda: bike("cnt"), "temp", {"cyclical": {"hr": 24}}),
}


def _split(X, y, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    X_fit, X_cal, y_fit, y_cal = train_test_split(X_train, y_train, test_size=0.15, random_state=seed)
    return X_fit, X_cal, X_test, y_fit, y_cal, y_test


def _fit(table, seed):
    loader, _, extra = TABLES[table]
    X, y = loader()
    X_fit, X_cal, X_test, y_fit, y_cal, y_test = _split(X, y, seed)
    start = time.time()
    model = ASRRegressor(task="mixture", max_epochs=EPOCHS, random_state=seed, **extra).fit(X_fit, y_fit)
    model.calibrate(X_cal, y_cal)
    return model, X_fit, X_test, y_test, time.time() - start


def _numeric(model):
    return [column.name for column in model.encoder_.columns if column.role == "numeric"]


def _score(model, X, y):
    lower, upper = model.predict_interval(X, 0.8).T
    return {
        "coverage": float(np.mean((y >= lower) & (y <= upper))),
        "width": float(np.mean(upper - lower)),
        "nll": -float(np.mean(model.log_prob(X, y))),
    }


def mask(seeds, out):
    rows = []
    for table in TABLES:
        for seed in seeds:
            model, _, X_test, y_test, seconds = _fit(table, seed)
            rng = np.random.default_rng(seed)
            numeric = [name for name in _numeric(model) if name in X_test.columns]
            uniform = rng.random((len(X_test), len(numeric)))
            for rate in RATES:
                masked = X_test.copy()
                for k, name in enumerate(numeric):
                    masked.loc[uniform[:, k] < rate, name] = np.nan
                for method, view in (("median", model), ("draws", model.with_uncertain_inputs(DRAWS, seed=seed))):
                    start = time.time()
                    row = {"table": table, "seed": seed, "rate": rate, "method": method, **_score(view, masked, y_test)}
                    row["seconds"] = time.time() - start
                    row["fit_seconds"] = seconds
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    _save(out, "mask", rows)
    _table(rows, "rate", "masked cells", lambda v: f"{100 * v:.0f}%")


def noise(seeds, out):
    rows = []
    for table in TABLES:
        column = TABLES[table][1]
        for seed in seeds:
            model, X_fit, X_test, y_test, _ = _fit(table, seed)
            spread = float(np.std(np.asarray(X_fit[column], dtype=np.float64)))
            rng = np.random.default_rng(100 + seed)
            for factor in NOISE:
                sd = factor * spread
                noisy = X_test.copy()
                noisy[column] = np.asarray(noisy[column], dtype=np.float64) + sd * rng.standard_normal(len(noisy))
                views = (("ignored", model), ("declared", model.with_uncertain_inputs(DRAWS, noise={column: sd}, seed=seed)))
                for method, view in views:
                    row = {"table": table, "seed": seed, "noise": factor, "method": method, **_score(view, noisy, y_test)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    _save(out, "noise", rows)
    _table(rows, "noise", "noise sd / column sd", lambda v: f"{v:g}")


def shift(seeds, out):
    rows = []
    X, y = california()
    for seed in seeds:
        cut = np.quantile(X["median_income"], 0.95)
        inside = X["median_income"] <= cut
        X_in, y_in = X[inside].reset_index(drop=True), y[np.asarray(inside)]
        X_out = X[~inside].reset_index(drop=True)
        X_train, X_test, y_train, _ = train_test_split(X_in, y_in, test_size=0.25, random_state=seed)
        model = ASRRegressor(task="mixture", max_epochs=EPOCHS, random_state=seed).fit(X_train, y_train)
        scores = np.concatenate([model.input_log_density(X_test), model.input_log_density(X_out)])
        labels = np.concatenate([np.zeros(len(X_test)), np.ones(len(X_out))])
        row = {"seed": seed, "auc_log_density": float(roc_auc_score(labels, -scores))}
        rows.append(row)
        print(json.dumps(row), flush=True)
    _save(out, "shift", rows)
    values = [row["auc_log_density"] for row in rows]
    print(f"auc_log_density: {np.mean(values):.3f} ± {np.std(values):.3f}")


def _table(rows, key, label, fmt):
    methods = sorted({row["method"] for row in rows}, key=lambda m: m not in ("median", "ignored"))
    print(f"\n| table | {label} | " + " | ".join(f"{m}: 80% coverage | {m}: width | {m}: NLL" for m in methods) + " |")
    print("|---|---:|" + "---:|" * (3 * len(methods)))
    for table in TABLES:
        for value in sorted({row[key] for row in rows}):
            cells = []
            for method in methods:
                group = [r for r in rows if r["table"] == table and r[key] == value and r["method"] == method]
                if not group:
                    cells += ["–"] * 3
                    continue
                cells.append(f"{100 * np.mean([r['coverage'] for r in group]):.1f}%")
                cells.append(f"{np.mean([r['width'] for r in group]):.3g}")
                cells.append(f"{np.mean([r['nll'] for r in group]):.3f}")
            print(f"| {table} | {fmt(value)} | " + " | ".join(cells) + " |")


def _save(out, stage, rows):
    path = Path(out)
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload[stage] = rows
    path.write_text(json.dumps(payload, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stages", nargs="*", default=["mask", "noise", "shift"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--out", default="inputs-results.json")
    args = parser.parse_args()
    for stage in args.stages:
        {"mask": mask, "noise": noise, "shift": shift}[stage](args.seeds, args.out)


if __name__ == "__main__":
    main()
