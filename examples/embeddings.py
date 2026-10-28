"""Choose which categoricals are learned embeddings.

The no-argument fit embeds every categorical column. Naming a column in
``categorical_embeddings`` embeds that column and one-hots the others.
``categorical_embeddings=[]`` one-hots every categorical. An unseen category
uses index 0, the unknown row.

Run from the package root:

    python examples/embeddings.py
"""

from __future__ import annotations

import time

import numpy as np

from ashlar import __version__
from ashlar.estimator import TabularEstimator


def main():
    rng = np.random.default_rng(1)
    n = 48
    rooms = rng.normal(size=n).astype(np.float32)
    city = rng.choice(["ny", "sf", "la"], size=n).astype(object)
    y = rooms + np.where(city == "sf", 0.4, 0.0)
    model = TabularEstimator(
        max_epochs=6,
        patience=3,
        batch_size=16,
        learning_rate=1e-2,
        random_state=0,
        categorical_embeddings=["city"],
        categorical_embedding_dim=4,
    )
    started = time.perf_counter()
    model.fit({"rooms": rooms, "city": city}, y.astype(np.float32))
    fit_seconds = time.perf_counter() - started
    prediction = np.asarray(
        model.predict({"rooms": np.array([0.2, np.nan], dtype=np.float32), "city": np.array(["boston", "sf"], dtype=object)})
    )
    table = model.core_.categorical_embeddings.tables[0]
    print(f"ashlar {__version__} embeddings")
    print(f"fit_seconds {fit_seconds:.4f}")
    print(f"onehot_widths {model.encoder_.categorical_widths}")
    print(f"embedding_table {tuple(table.shape)}")
    print(f"prediction {prediction.tolist()}")
    print(f"search {model.search_}")
    if model.encoder_.categorical_widths or table.shape[0] < 2 or not np.isfinite(prediction).all():
        raise SystemExit("embedding example failed")


if __name__ == "__main__":
    main()
