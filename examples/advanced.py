"""Advanced workflow: TabM trunk, linked heads, masked outcome, direct weights.

Run from the repository root:

    python examples/advanced.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ashlar import __version__
from ashlar.demos.linked import LinkedHead, linked_dist, linked_objective, linked_point
from ashlar.estimator import ASRModel
from ashlar.predict import MeanParameterNormal, NormalMixture
from ashlar.recipes import TABM_PROBE


def main():
    rng = np.random.default_rng(2)
    n = 64
    rooms = rng.normal(size=n).astype(np.float32)
    city = rng.choice(["ny", "sf"], size=n).astype(object)
    continuous = rooms + np.where(city == "sf", 0.3, 0.0) + rng.normal(scale=0.05, size=n)
    event = (continuous > 0).astype(np.float32)
    event[rng.choice(n, size=10, replace=False)] = np.nan
    y = np.column_stack([continuous, event]).astype(np.float32)
    weights = np.linspace(0.2, 1.5, n).astype(np.float64)
    weights[:3] = 0.0
    x = {"rooms": rooms, "city": city}
    params = dict(TABM_PROBE)
    params.update(
        task="custom",
        head=LinkedHead,
        objective=linked_objective,
        point_fn=linked_point,
        dist_fn=linked_dist,
        max_epochs=8,
        categorical=["city"],
    )
    model = ASRModel(**params)
    started = time.perf_counter()
    model.fit(x, y, sample_weight=weights)
    fit_seconds = time.perf_counter() - started
    raw = model.predict_raw({"rooms": np.array([0.0], dtype=np.float32), "city": np.array(["sf"], dtype=object)})
    member_shape = tuple(raw["normal"].shape)
    point = np.asarray(model.predict({"rooms": np.array([0.0, 1.0], dtype=np.float32), "city": np.array(["sf", "ny"], dtype=object)}))
    mixture = model.predict_dist({"rooms": np.array([0.0], dtype=np.float32), "city": np.array(["sf"], dtype=object)})
    collapsed = model.predict_dist(
        {"rooms": np.array([0.0], dtype=np.float32), "city": np.array(["sf"], dtype=object)},
        policy="mean_parameters",
    )
    # The semantic gap does not depend on this fit. Two unit normals at -5 and +5.
    loc = jnp.array([[[-5.0], [5.0]]], dtype=jnp.float32)
    scale = jnp.ones_like(loc)
    reference_mixture = float(NormalMixture(loc=loc, scale=scale).variance()[0, 0])
    reference_average = float(MeanParameterNormal(loc=jnp.mean(loc, axis=1), scale=jnp.mean(scale, axis=1)).variance()[0, 0])
    with tempfile.TemporaryDirectory() as directory:
        restored = ASRModel.load(model.save(Path(directory) / "linked"))
        reloaded = np.asarray(restored.predict({"rooms": np.array([0.0], dtype=np.float32), "city": np.array(["sf"], dtype=object)}))
    print(f"ashlar {__version__} advanced")
    print(f"fit_seconds {fit_seconds:.4f}")
    print(f"member_normal_shape {member_shape}")
    print(f"train_loss_tail {model.history_['train_loss'][-1]:.6f}")
    print(f"point {point.tolist()}")
    print(f"fitted_mixture_variance {float(mixture.variance()[0]):.6f}")
    print(f"fitted_mean_parameter_variance {float(collapsed.variance()[0]):.6f}")
    print(f"reference_mixture_variance {reference_mixture:.1f}")
    print(f"reference_mean_parameter_variance {reference_average:.1f}")
    print(f"reload_matches {np.allclose(point[:1], reloaded)}")
    if member_shape[1] != 4 or reference_mixture != 26.0 or reference_average != 1.0:
        raise SystemExit("advanced checks failed")
    if not np.allclose(point[:1], reloaded):
        raise SystemExit("advanced reload failed")


if __name__ == "__main__":
    main()
