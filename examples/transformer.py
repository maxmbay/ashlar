"""Transformer backbone over a feature set, including a variable-length category set.

``city`` is a learned embedding. ``tags`` is a list of labels, mean-pooled
inside the core. The transformer then attends over the field tokens
(rooms, city, tags). ``backbone='set'`` is the same tokens with a mean pool
instead of attention.

Run from the package root:

    python examples/transformer.py
"""

from __future__ import annotations

import time

import numpy as np

from ashlar import __version__
from ashlar.estimator import ASRModel


def main():
    rng = np.random.default_rng(5)
    n = 36
    rooms = rng.normal(size=n).astype(np.float32)
    city = rng.choice(["ny", "sf"], size=n).astype(object)
    catalog = [["red"], ["red", "blue"], ["blue"], ["red", "blue", "green"]]
    tags = np.empty(n, dtype=object)
    for index in range(n):
        tags[index] = catalog[index % len(catalog)]
    y = rooms + np.where(city == "sf", 0.3, 0.0) + np.array([0.1 * len(item) for item in tags], dtype=np.float32)
    model = ASRModel(
        backbone="transformer",
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "n_heads": 2, "dropout": 0.0},
        categorical_embeddings=["city"],
        set_features=["tags"],
        categorical_embedding_dim=4,
        set_max_length=4,
        max_epochs=4,
        patience=2,
        batch_size=12,
        learning_rate=1e-2,
        random_state=0,
    )
    started = time.perf_counter()
    model.fit({"rooms": rooms, "city": city, "tags": tags}, y)
    fit_seconds = time.perf_counter() - started
    query_tags = np.empty(2, dtype=object)
    query_tags[0] = ["blue", "missing"]
    query_tags[1] = []
    prediction = np.asarray(
        model.predict(
            {
                "rooms": np.array([0.0, 0.4], dtype=np.float32),
                "city": np.array(["sf", "boston"], dtype=object),
                "tags": query_tags,
            }
        )
    )
    print(f"ashlar {__version__} transformer")
    print(f"fit_seconds {fit_seconds:.4f}")
    print(f"member_count {model.core_.backbone.k}")
    print(f"set_length {model.core_.set_length}")
    print(f"prediction {prediction.tolist()}")
    if model.core_.backbone.k != 1 or not np.isfinite(prediction).all():
        raise SystemExit("transformer example failed")


if __name__ == "__main__":
    main()
