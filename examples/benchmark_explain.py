"""Explanations of the distribution, beside SHAP on LightGBM.

On hourly Capital Bikeshare rentals (seeds 0 to 2, an 80/20 split), a
``mixture`` head is fitted as in ``examples/explain_bike.py`` and
``model.explain`` attributes four numbers per test row to the features: the
mean, the median, the width of the 80% interval, and P(rentals > 500).

Beside it, the gradient-boosting way, with SHAP's ``TreeExplainer``, on
early-stopped LightGBM models (learning rate 0.05, up to 4,000 trees, 100
rounds of patience on a 20% validation slice of the training rows):

- the mean: SHAP of an L2 regressor, the usual explanation;
- the width: SHAP of a quantile model at 0.9 minus SHAP of one at 0.1, the
  GBM user's way to explain an 80% interval's width (SHAP values add, so
  this is the attribution of the difference of the two models);
- the tail: SHAP of a classifier of rentals > 500, in probability
  (interventional, against a background of training rows).

``season`` and ``weathersit`` are integer codes to the trees. Reported per
target: each feature's mean absolute attribution and its rank, averaged over
the seeds, and how often the two quantile models cross. The question is what
each method can attribute, not a score: there is no true attribution to
score against.

    python examples/benchmark_explain.py [--seeds 0 1 2] [--rows 2000] [--out explain.json]
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
from scipy import stats
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import _fit_gbt
from benchmark_uncertainty import bike

from ashlar import ASRRegressor

EPOCHS = 40
THRESHOLD = 500.0
BACKGROUND = 2000
TREE_BACKGROUND = 200
ASHLAR_TARGETS = {
    "mean": {"target": "mean"},
    "median": {"target": "median"},
    "width": {"target": "interval_width", "coverage": 0.8},
    "tail": {"target": "exceedance", "threshold": THRESHOLD},
}
COLUMNS = ("lightgbm_mean", "ashlar_mean", "ashlar_median", "lightgbm_width", "ashlar_width", "lightgbm_tail", "ashlar_tail")


def _codes(frame):
    """The trees read ``season`` and ``weathersit`` as the integer codes they are in the source."""
    frame = frame.copy()
    for column in ("season", "weathersit"):
        frame[column] = frame[column].astype(float)
    return frame


def _importance(values, names):
    scores = np.mean(np.abs(np.asarray(values, dtype=np.float64)), axis=0)
    return {name: float(score) for name, score in zip(names, scores, strict=True)}


def _tree_shap(model, X):
    import shap

    return np.asarray(shap.TreeExplainer(model).shap_values(X), dtype=np.float64)


def run(seeds, n_rows, out):
    import shap

    X, y = bike("cnt")
    names = list(X.columns)
    results = []
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        rows = X_test.iloc[:n_rows]
        background = X_train.sample(BACKGROUND, random_state=seed)
        result = {"seed": seed, "rows": int(len(rows)), "importance": {}, "prediction": {}, "seconds": {}}

        start = time.time()
        model = ASRRegressor(task="mixture", max_epochs=EPOCHS, cyclical={"hr": 24}, random_state=seed)
        model.fit(X_train, y_train)
        result["seconds"]["ashlar_fit"] = time.time() - start
        for label, kwargs in ASHLAR_TARGETS.items():
            start = time.time()
            explanation = model.explain(rows, background=background, **kwargs)
            result["seconds"][f"ashlar_{label}"] = time.time() - start
            importance = explanation.importance()
            result["importance"][f"ashlar_{label}"] = {name: importance[name] for name in names}
            result["prediction"][f"ashlar_{label}"] = float(np.mean(explanation.prediction))

        trees_X, trees_rows = _codes(X_train), _codes(rows)
        start = time.time()
        mean, _ = _fit_gbt("lightgbm", "regression", True, seed, trees_X, y_train)
        low, _ = _fit_gbt("lightgbm", "regression", True, seed, trees_X, y_train, objective="quantile", alpha=0.1)
        high, _ = _fit_gbt("lightgbm", "regression", True, seed, trees_X, y_train, objective="quantile", alpha=0.9)
        tail, _ = _fit_gbt("lightgbm", "classification", True, seed, trees_X, (y_train > THRESHOLD).astype(int))
        result["seconds"]["lightgbm_fit"] = time.time() - start

        start = time.time()
        result["importance"]["lightgbm_mean"] = _importance(_tree_shap(mean, trees_rows), names)
        width = _tree_shap(high, trees_rows) - _tree_shap(low, trees_rows)
        result["importance"]["lightgbm_width"] = _importance(width, names)
        explainer = shap.TreeExplainer(
            tail,
            data=_codes(background).sample(TREE_BACKGROUND, random_state=seed),
            model_output="probability",
            feature_perturbation="interventional",
        )
        values = np.asarray(explainer.shap_values(trees_rows), dtype=np.float64)
        result["importance"]["lightgbm_tail"] = _importance(values[..., 1] if values.ndim == 3 else values, names)
        result["seconds"]["shap"] = time.time() - start

        result["prediction"]["lightgbm_mean"] = float(np.mean(mean.predict(trees_rows)))
        spread = high.predict(trees_rows) - low.predict(trees_rows)
        result["prediction"]["lightgbm_width"] = float(np.mean(spread))
        result["prediction"]["lightgbm_tail"] = float(np.mean(tail.predict_proba(trees_rows)[:, 1]))
        test_spread = high.predict(_codes(X_test)) - low.predict(_codes(X_test))
        result["crossed"] = {"explained_rows": float(np.mean(spread < 0)), "test_rows": float(np.mean(test_spread < 0))}
        result["trees"] = {name: int(m.best_iteration_ or m.n_estimators) for name, m in
                           (("mean", mean), ("low", low), ("high", high), ("tail", tail))}
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "importance"}), flush=True)
        _save(out, results)
    report(results)


def report(results):
    names = list(results[0]["importance"]["ashlar_mean"])
    columns = [c for c in COLUMNS if all(c in r["importance"] for r in results)]
    mean_abs = {c: {n: np.mean([r["importance"][c][n] for r in results]) for n in names} for c in columns}
    ranks = {}
    for c in columns:
        per_seed = [stats.rankdata([-r["importance"][c][n] for n in names]) for r in results]
        ranks[c] = dict(zip(names, np.mean(per_seed, axis=0), strict=True))
    order = sorted(names, key=lambda n: -mean_abs["ashlar_median"][n])
    print(f"\n{len(results)} seeds; mean absolute attribution (rank in brackets, mean over seeds)\n")
    print("| feature | " + " | ".join(columns) + " |")
    print("|---|" + "---:|" * len(columns))
    for name in order:
        cells = [f"{_fmt(mean_abs[c][name])} ({ranks[c][name]:.3g})" for c in columns]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("\nSpearman correlation of the mean absolute attributions, mean over seeds:")
    pairs = [("lightgbm_mean", "ashlar_mean"), ("ashlar_mean", "ashlar_median"), ("ashlar_median", "ashlar_width"),
             ("lightgbm_mean", "lightgbm_width"), ("lightgbm_width", "ashlar_width"), ("lightgbm_tail", "ashlar_tail"),
             ("ashlar_median", "ashlar_tail")]
    for a, b in pairs:
        if a in columns and b in columns:
            rho = [stats.spearmanr([r["importance"][a][n] for n in names], [r["importance"][b][n] for n in names]).statistic
                   for r in results]
            print(f"  {a} ~ {b}: {np.mean(rho):.2f}")
    crossed = [r["crossed"]["test_rows"] for r in results if "crossed" in r]
    if crossed:
        print(f"\nLightGBM's 0.9 model below its 0.1 model on {100 * np.mean(crossed):.2f}% of test rows "
              f"({', '.join(f'{100 * c:.2f}%' for c in crossed)})")


def _fmt(value):
    return f"{value:.3g}" if value >= 0.01 else f"{value:.2g}"


def _save(out, results):
    path = Path(out)
    payload = json.loads(path.read_text()) if path.exists() else {}
    payload["shap"] = results
    path.write_text(json.dumps(payload, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rows", type=int, default=2000, help="test rows to explain per seed")
    parser.add_argument("--out", default="explain-results.json")
    parser.add_argument("--report", action="store_true", help="print the table from --out without fitting")
    args = parser.parse_args()
    if args.report:
        report(json.loads(Path(args.out).read_text())["shap"])
        return
    run(args.seeds, args.rows, args.out)


if __name__ == "__main__":
    main()
