"""Use a fitted core inside a bounded continuous optimization.

This example passes ``categorical_embeddings=[]`` so the category stays a
constant one-hot vector. The numeric value is a JAX argument, and the
gradient passes through median imputation and the fitted numeric map.

Run from the repository root:

    python examples/jax_composition.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ashlar import __version__
from ashlar.estimator import ASRModel


def main():
    rng = np.random.default_rng(3)
    n = 48
    rooms = rng.normal(size=n).astype(np.float32)
    city = rng.choice(["ny", "sf"], size=n).astype(object)
    y = rooms + np.where(city == "sf", 0.4, -0.2)
    model = ASRModel(
        task="regression",
        backbone="tabm",
        backbone_kwargs={"n_blocks": 2, "d_block": 16, "dropout": 0.0, "k": 4, "start_scaling_init": "random-signs"},
        max_epochs=8,
        patience=4,
        batch_size=16,
        validation_size=0.25,
        learning_rate=2e-3,
        weight_decay=0.0,
        random_state=0,
        categorical=["city"],
        categorical_embeddings=[],
    )
    started = time.perf_counter()
    model.fit({"rooms": rooms, "city": city}, y.astype(np.float32))
    fit_seconds = time.perf_counter() - started
    _numeric, categorical = model.prepare({"rooms": np.array([0.0], dtype=np.float32), "city": np.array(["sf"], dtype=object)})
    fixed_category = categorical

    def predicted_mean(raw_rooms):
        point = model.forward_transformed(raw_rooms.reshape(1, 1), fixed_category)
        return point[0]

    predicted_mean(jnp.array(0.2, dtype=jnp.float32)).block_until_ready()
    started = time.perf_counter()
    gradient = jax.grad(predicted_mean)(jnp.array(0.2, dtype=jnp.float32))
    gradient.block_until_ready()
    grad_seconds = time.perf_counter() - started
    step = 1e-3
    base = jnp.array(0.2, dtype=jnp.float32)
    numeric = (predicted_mean(base + step) - predicted_mean(base - step)) / (2 * step)
    value = base
    optimizer = optax.sgd(0.2)
    state = optimizer.init(value)
    for _ in range(15):
        grad_value = jax.grad(predicted_mean)(value)
        updates, state = optimizer.update(grad_value, state, value)
        value = optax.apply_updates(value, updates)
        value = jnp.clip(value, -2.0, 2.0)
    before = float(predicted_mean(base))
    after = float(predicted_mean(value))
    print(f"ashlar {__version__} composition")
    print(f"fit_seconds {fit_seconds:.4f}")
    print(f"grad_seconds {grad_seconds:.4f}")
    print(f"jax_grad {float(gradient):.6f}")
    print(f"finite_difference {float(numeric):.6f}")
    print(f"prediction_before {before:.6f}")
    print(f"prediction_after {after:.6f}")
    print(f"optimized_rooms {float(value):.6f}")
    print(f"category_width {int(fixed_category.shape[1])}")
    if abs(float(gradient) - float(numeric)) > 1e-2:
        raise SystemExit("gradient does not match the finite difference")
    if not np.isfinite(after):
        raise SystemExit("optimized prediction is not finite")


if __name__ == "__main__":
    main()
