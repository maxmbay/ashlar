"""Turn on the default hyperparameter search with one argument.

``search=True`` compares two weight-decay values for two epochs each,
then trains with the winner using the estimator's own ``max_epochs``.
Learning rate, feature dropout, and patience stay on the constructor
unless the space names them. Omitting ``search`` does not search.

Widen the grid by passing a dict, for example:

    model.fit(X, y, search={
        "n_trials": 4,
        "max_epochs": 3,
        "space": {
            "learning_rate": [1e-3, 3e-3],
            "weight_decay": [0.0, 1e-4],
        },
    })

The product is truncated to ``n_trials`` in list order. Trial checkpoints
are discarded. The final ``fit`` starts again with the chosen settings.

Run from the package root:

    python examples/search.py
"""

from __future__ import annotations

import time

import numpy as np

from ashlar import __version__
from ashlar.estimator import TabularEstimator
from ashlar.search import DEFAULT_SEARCH


def main():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(32, 2)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 1]).astype(np.float32)
    model = TabularEstimator(max_epochs=3, patience=2, batch_size=16, learning_rate=1e-3, random_state=0)
    started = time.perf_counter()
    model.fit(x, y, search=True)
    fit_seconds = time.perf_counter() - started
    chosen = model.search_["chosen"]["weight_decay"]
    losses = [trial["val_loss"] for trial in model.search_["trials"]]
    prediction = np.asarray(model.predict(x[:2]))
    print(f"ashlar {__version__} search")
    print(f"fit_seconds {fit_seconds:.4f}")
    print(f"switch search=True")
    print(f"default_space {DEFAULT_SEARCH['space']}")
    print(f"chosen_weight_decay {chosen}")
    print(f"constructor_weight_decay {model.weight_decay}")
    print(f"trial_val_losses {losses}")
    print(f"constructor_learning_rate {model.learning_rate}")
    print(f"prediction {prediction.tolist()}")
    if chosen not in DEFAULT_SEARCH["space"]["weight_decay"] or not np.isfinite(prediction).all():
        raise SystemExit("search example failed")


if __name__ == "__main__":
    main()
