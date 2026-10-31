"""The hurdle head against smooth heads on a target that is often exactly zero.

NYC taxi tips (the seaborn ``taxis`` table, 6,341 trips with zones; about a
third tip nothing, mostly cash fares). Payment type and the total are
dropped, since they give the tip away. Five random 80/20 splits, seeds 0 to 4.

Every head is scored the same way, from ``predict_dist``:

- CRPS from 2,000 draws, in dollars; proper for a distribution with an atom.
- Brier score of ``P(tip = 0)``, read as ``cdf(0)``. A smooth head can only
  put the mass below zero there; the hurdle puts an exact atom at zero.
- The share of held-out tips inside the uncalibrated 80% interval.

    JAX_PLATFORMS=cpu python examples/benchmark_hurdle.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_readme import _crps, taxis

from ashlar import ASRRegressor

SEEDS = (0, 1, 2, 3, 4)
DRAWS = 2000
MODELS = {
    "gaussian": {"task": "gaussian"},
    "mixture": {"task": "mixture"},
    "hurdle, lognormal": {"task": "hurdle"},
    "hurdle, gamma": {"task": "hurdle", "head_kwargs": {"severity": "gamma"}},
}


def main() -> None:
    x, y = taxis("tip")
    print(f"{len(y)} trips, {np.mean(y == 0):.1%} with no tip")
    scores = {name: [] for name in MODELS}
    for seed in SEEDS:
        index_train, index_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed)
        x_train, x_test = x.iloc[index_train], x.iloc[index_test]
        y_train, y_test = y[index_train], y[index_test]
        for name, params in MODELS.items():
            model = ASRRegressor(random_state=seed, **params).fit(x_train, y_train)
            dist = model.predict_dist(x_test)
            draws = np.asarray(dist.sample(jax.random.PRNGKey(seed), (DRAWS,)), dtype=np.float64).T
            zero = np.asarray(dist.cdf(jnp.zeros(len(y_test))), dtype=np.float64)
            low, high = model.predict_interval(x_test, coverage=0.8).T
            record = {
                "crps": _crps(draws, y_test),
                "brier_zero": float(np.mean((zero - (y_test == 0)) ** 2)),
                "coverage_80": float(np.mean((y_test >= low) & (y_test <= high))),
                "width_80": float(np.mean(high - low)),
            }
            scores[name].append(record)
            print(seed, name, {key: round(value, 4) for key, value in record.items()}, flush=True)
    print(f"\n{len(SEEDS)} splits, mean ± standard error")
    print("| Head | CRPS, $ ↓ | Brier of P(no tip) ↓ | 80% interval holds | Width, $ |")
    print("|---|---:|---:|---:|---:|")
    for name, rows in scores.items():
        cells = []
        for key in ("crps", "brier_zero", "coverage_80", "width_80"):
            values = np.asarray([row[key] for row in rows])
            cells.append(f"{values.mean():.3f} ± {values.std(ddof=1) / np.sqrt(len(values)):.3f}")
        print(f"| `{name}` | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
