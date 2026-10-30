"""Ordinary workflow: mixed columns, missing values, unseen categories, save/reload.

Run from the repository root:

    python examples/ordinary.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ashlar import __version__
from ashlar.estimator import ASRModel


def main():
    rng = np.random.default_rng(0)
    n = 80
    frame = pd.DataFrame(
        {
            "rooms": rng.normal(size=n).astype(np.float32),
            "age": rng.uniform(0, 40, size=n).astype(np.float32),
            "city": rng.choice(["ny", "sf", "la"], size=n),
        }
    )
    frame.loc[frame.sample(3, random_state=0).index, "age"] = np.nan
    frame.loc[frame.sample(2, random_state=1).index, "city"] = None
    target = frame["rooms"].to_numpy() + 0.1 * frame["age"].fillna(frame["age"].median()).to_numpy()
    target = target + np.where(frame["city"].fillna("ny").eq("sf"), 0.5, 0.0)
    # Default recipe: residual MLP, robust scale, categorical embeddings, no search.
    # The overrides only shorten the toy run and raise the learning rate.
    model = ASRModel(max_epochs=8, patience=4, batch_size=16, learning_rate=1e-2, random_state=0)
    started = time.perf_counter()
    model.fit(frame, target.astype(np.float32))
    fit_seconds = time.perf_counter() - started
    test_pd = pd.DataFrame({"rooms": [0.2, np.nan], "age": [np.nan, 10.0], "city": ["boston", "sf"]})
    test_pl = pl.DataFrame({"rooms": [0.2, None], "age": [None, 10.0], "city": ["boston", "sf"]})
    test_np = {
        "rooms": np.array([0.2, np.nan], dtype=np.float32),
        "age": np.array([np.nan, 10.0], dtype=np.float32),
        "city": np.array(["boston", "sf"], dtype=object),
    }
    started = time.perf_counter()
    pandas_pred = np.asarray(model.predict(test_pd))
    pandas_pred = np.asarray(pandas_pred)
    first_predict = time.perf_counter() - started
    polars_pred = np.asarray(model.predict(test_pl))
    numpy_pred = np.asarray(model.predict(test_np))
    with tempfile.TemporaryDirectory() as directory:
        path = model.save(Path(directory) / "ordinary")
        restored = ASRModel.load(path)
        reloaded = np.asarray(restored.predict(test_pd))
    print(f"ashlar {__version__} ordinary")
    print(f"backbone {model.backbone}")
    print(f"search {model.search_}")
    print(f"fit_seconds {fit_seconds:.4f}")
    print(f"first_predict_seconds {first_predict:.4f}")
    print(f"val_loss_tail {model.history_['val_loss'][-1]:.6f}")
    print(f"pandas {pandas_pred.tolist()}")
    print(f"polars_matches {np.allclose(pandas_pred, polars_pred)}")
    print(f"numpy_matches {np.allclose(pandas_pred, numpy_pred)}")
    print(f"reload_matches {np.allclose(pandas_pred, reloaded)}")
    if model.search_ is not None or model.backbone != "residual_mlp":
        raise SystemExit("ordinary example left the default recipe")
    if not (np.allclose(pandas_pred, polars_pred) and np.allclose(pandas_pred, numpy_pred) and np.allclose(pandas_pred, reloaded)):
        raise SystemExit("ordinary predictions did not agree")


if __name__ == "__main__":
    main()
