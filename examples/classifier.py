"""Default classifier: labels, probabilities, search, and a learned category.

``ashlar.classifier()`` is the classification recipe. ``TabularEstimator(task="classification")``
is the same task flag. ``predict`` returns labels. ``predict_proba`` returns one
column per class. ``fit(..., search=True)`` ranks trials by weighted accuracy,
not by squared error. Omitting ``search`` does not score that metric.

Run from the package root:

    python examples/classifier.py
"""

from __future__ import annotations

import time

import numpy as np

from ashlar import __version__, classifier


def main():
    rng = np.random.default_rng(7)
    n = 48
    rooms = rng.normal(size=n).astype(np.float32)
    rooms[:2] = np.nan
    city = rng.choice(["ny", "sf", "la"], size=n).astype(object)
    score = np.nan_to_num(rooms, nan=0.0) + np.where(city == "sf", 0.6, 0.0)
    labels = np.where(score > 0.4, "high", np.where(score < -0.4, "low", "mid"))
    model = classifier(max_epochs=6, patience=3, batch_size=16, learning_rate=1e-2, random_state=0)
    started = time.perf_counter()
    model.fit({"rooms": rooms, "city": city}, labels)
    fit_seconds = time.perf_counter() - started
    query = {"rooms": np.array([0.2, np.nan], dtype=np.float32), "city": np.array(["boston", "sf"], dtype=object)}
    predicted = np.asarray(model.predict(query))
    probabilities = np.asarray(model.predict_proba(query))

    binary_x = rng.normal(size=(36, 2)).astype(np.float32)
    binary_y = (binary_x[:, 0] > 0).astype(int)
    searched = classifier(max_epochs=3, patience=2, batch_size=12, learning_rate=1e-3, random_state=0)
    search_started = time.perf_counter()
    searched.fit(binary_x, binary_y, search=True, sample_weight=np.linspace(0.5, 1.5, len(binary_y)))
    search_seconds = time.perf_counter() - search_started
    trial_metrics = [trial["metric"] for trial in searched.search_["trials"]]
    trial_scores = [trial["val_score"] for trial in searched.search_["trials"]]

    embedded = classifier(
        backbone="transformer",
        backbone_kwargs={"n_blocks": 1, "d_block": 8, "n_heads": 2, "dropout": 0.0},
        categorical_embeddings=["city"],
        categorical_embedding_dim=4,
        max_epochs=3,
        patience=2,
        batch_size=12,
        learning_rate=1e-2,
        random_state=0,
    )
    embedded.fit({"rooms": np.nan_to_num(rooms, nan=0.0), "city": city}, labels)
    embedded_probabilities = np.asarray(embedded.predict_proba(query))

    print(f"ashlar {__version__} classifier")
    print(f"fit_seconds {fit_seconds:.4f}")
    print(f"classes {model.classes_.tolist()}")
    print(f"labels {predicted.tolist()}")
    print(f"probabilities {probabilities.tolist()}")
    print(f"search_seconds {search_seconds:.4f}")
    print(f"search_metric {trial_metrics[0]}")
    print(f"search_scores {trial_scores}")
    print(f"chosen_learning_rate {searched.search_['chosen']['learning_rate']}")
    print(f"constructor_learning_rate {searched.learning_rate}")
    print(f"embedding_onehot_widths {embedded.encoder_.categorical_widths}")
    print(f"embedding_probabilities {embedded_probabilities.round(6).tolist()}")
    if model.search_ is not None or probabilities.shape != (2, 3):
        raise SystemExit("default classifier path failed")
    if trial_metrics != ["weighted_accuracy", "weighted_accuracy"]:
        raise SystemExit("classification search did not score weighted accuracy")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise SystemExit("probabilities do not sum to one")
    if embedded.encoder_.categorical_widths or not np.isfinite(embedded_probabilities).all():
        raise SystemExit("classification embeddings failed")


if __name__ == "__main__":
    main()
