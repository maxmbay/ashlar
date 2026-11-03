"""Explain bike rentals' predictive distribution: the median, the interval's width, and a tail.

Fits a mixture head to hourly Capital Bikeshare rentals and attributes three
numbers per test row to the features with ``model.explain``: the median, the
width of the 80% interval, and the probability of more than 500 rentals in
the hour. The median and the tail are driven by the hour and the season;
the width says which features make an hour hard to call.

    python examples/explain_bike.py [--rows 2000] [--epochs 40]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_uncertainty import bike

from ashlar import ASRRegressor


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=2000, help="test rows to explain")
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()

    X, y = bike("cnt")
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=0)
    model = ASRRegressor(task="mixture", max_epochs=args.epochs, cyclical={"hr": 24}, random_state=0)
    start = time.time()
    model.fit(X_train, y_train)
    print(f"fit in {time.time() - start:.0f} s")
    rows = X_test.iloc[: args.rows]
    targets = {
        "median": dict(target="median"),
        "80% width": dict(target="interval_width", coverage=0.8),
        "P(>500)": dict(target="exceedance", threshold=500.0),
    }
    table = {}
    for label, kwargs in targets.items():
        start = time.time()
        result = model.explain(rows, background=X_train.sample(2000, random_state=0), **kwargs)
        table[label] = result.importance()
        print(f"{label}: {time.time() - start:.0f} s, mean prediction {np.mean(result.prediction):.3f}")
    names = list(table["median"])
    print("\n| feature | " + " | ".join(table) + " |")
    print("|---|" + "---:|" * len(table))
    for name in names:
        print(f"| {name} | " + " | ".join(f"{table[label][name]:.3g}" for label in table) + " |")


if __name__ == "__main__":
    main()
