"""Small behavior checks for the distributional forecast heads.

CPU only. A synthetic sine, a two-mode series, and Melbourne daily minimum
temperatures. This is not a benchmark and it does not fit XGBoost.

    JAX_PLATFORMS=cpu python examples/distributional_probe.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from ashlar.distributional import HISTOGRAM_BINS, histogram_bin
from ashlar.forecast import ASRForecaster, _forecast_windows

LOOKBACK = 24
HORIZON = 1
EPOCHS = 8
PATIENCE = 4
TEMPERATURES = Path("/tmp/ashlar-dist/daily-min-temperatures.csv")
# Central 80% of a normal is mean ± this many standard deviations.
_NORMAL_80 = 1.2815515655446004


def _fit(series: np.ndarray, head: str) -> ASRForecaster:
    cut = int(len(series) * 0.8)
    return ASRForecaster(
        lookback=LOOKBACK,
        horizon=HORIZON,
        head=head,
        max_epochs=EPOCHS,
        patience=PATIENCE,
        batch_size=64,
        random_state=0,
    ).fit(series[:cut])


def _test_batch(series: np.ndarray):
    cut = int(len(series) * 0.8)
    origins, inputs, targets = _forecast_windows(series, LOOKBACK, HORIZON)
    test = origins >= cut
    return inputs[test], targets[test]


def _sine(n: int = 640) -> np.ndarray:
    time = np.arange(n, dtype=np.float64)
    noise = np.random.default_rng(0).normal(0.0, 0.08, size=n)
    return np.sin(2.0 * np.pi * time / 24.0) + noise


def _bimodal(n: int = 640) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.choice(np.array([-2.0, 2.0]), size=n) + rng.normal(0.0, 0.3, size=n)


def _temperatures() -> np.ndarray:
    values = np.loadtxt(TEMPERATURES, delimiter=",", skiprows=1, usecols=1, dtype=np.float64)
    if values.ndim != 1 or len(values) < 1000 or not np.isfinite(values).all():
        raise RuntimeError(f"temperature series has shape {values.shape}.")
    return np.ascontiguousarray(values)


def _quantile(series: np.ndarray) -> None:
    model = _fit(series, "quantile")
    inputs, targets = _test_batch(series)
    quantiles = model.quantiles(inputs)
    covered = (targets >= quantiles[..., 0]) & (targets <= quantiles[..., 2])
    width = float(np.mean(quantiles[..., 2] - quantiles[..., 0]))
    print(
        f"quantile temperatures coverage {float(np.mean(covered)):.6f} "
        f"median_width {width:.6f} n {int(covered.size)} epoch {model.best_epoch_}",
        flush=True,
    )


def _gaussian(series: np.ndarray) -> None:
    model = _fit(series, "gaussian")
    inputs, targets = _test_batch(series)
    loc, scale = model.gaussian(inputs)
    low = loc - _NORMAL_80 * scale
    high = loc + _NORMAL_80 * scale
    covered = (targets >= low) & (targets <= high)
    print(
        f"gaussian sine coverage80 {float(np.mean(covered)):.6f} "
        f"mean_scale {float(np.mean(scale)):.6f} epoch {model.best_epoch_}",
        flush=True,
    )


def _mixture(series: np.ndarray) -> None:
    model = _fit(series, "mixture")
    inputs, _targets = _test_batch(series)
    rows = len(inputs)

    def density(level: float) -> float:
        value = np.full((rows, HORIZON), level, dtype=np.float64)
        return float(np.mean(model.log_prob(inputs, value)))

    low = density(-2.0)
    mid = density(0.0)
    high = density(2.0)
    gap = min(low, high) - mid
    print(
        f"mixture bimodal logp_minus {low:.6f} logp_zero {mid:.6f} "
        f"logp_plus {high:.6f} gap {gap:.6f} epoch {model.best_epoch_}",
        flush=True,
    )


def _flow(series: np.ndarray) -> None:
    model = _fit(series, "flow")
    inputs, targets = _test_batch(series)
    near = float(np.mean(model.log_prob(inputs, targets)))
    far = float(np.mean(model.log_prob(inputs, np.full_like(targets, 6.0))))
    print(
        f"flow sine logp_target {near:.6f} logp_far {far:.6f} "
        f"gap {near - far:.6f} epoch {model.best_epoch_}",
        flush=True,
    )


def _histogram(series: np.ndarray) -> None:
    model = _fit(series, "histogram")
    inputs, targets = _test_batch(series)
    probabilities, _centers = model.bin_probabilities(inputs)
    standardized = (targets - model.mean_) / model.std_
    index = np.asarray(histogram_bin(jnp.asarray(standardized, dtype=jnp.float32)))
    mass = np.take_along_axis(probabilities, index[..., None], axis=-1)[..., 0]
    print(
        f"histogram sine true_bin {float(np.mean(mass)):.6f} "
        f"uniform {1.0 / HISTOGRAM_BINS:.6f} epoch {model.best_epoch_}",
        flush=True,
    )


def main() -> None:
    sine = _sine()
    print("gaussian", flush=True)
    _gaussian(sine)
    print("flow", flush=True)
    _flow(sine)
    print("histogram", flush=True)
    _histogram(sine)
    print("mixture", flush=True)
    _mixture(_bimodal())
    print("quantile", flush=True)
    _quantile(_temperatures())


if __name__ == "__main__":
    main()
