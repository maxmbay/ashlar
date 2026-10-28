"""Ordinal classifier: ordered labels, probabilities, and an expected score.

``ashlar.ordinal()`` uses a cumulative logit. ``predict`` returns the most
likely label. ``predict_score`` returns ``probabilities @ classes_``.

Run from the package root:

    python examples/ordinal.py
"""

from __future__ import annotations

import numpy as np

from ashlar import __version__, ordinal


def main():
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(size=(n, 1)).astype(np.float32)
    y = np.where(x[:, 0] < -0.4, 3, np.where(x[:, 0] < 0.4, 6, 8)).astype(np.int32)
    model = ordinal(
        max_epochs=8,
        patience=4,
        batch_size=16,
        learning_rate=2e-2,
        backbone_kwargs={"n_blocks": 1, "d_block": 16, "dropout": 0.0},
        random_state=0,
    )
    model.fit(x, y)
    labels = np.asarray(model.predict(x[:4]))
    probabilities = np.asarray(model.predict_proba(x[:4]))
    score = np.asarray(model.predict_score(x[:4]))
    print("ashlar", __version__)
    print("classes", model.classes_.tolist())
    print("labels", labels.tolist())
    print("probabilities", probabilities.round(3).tolist())
    print("expected", score.round(3).tolist())


if __name__ == "__main__":
    main()
