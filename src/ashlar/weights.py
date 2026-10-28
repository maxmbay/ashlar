"""Direct per-example loss weights.

Reduction name: ``batch_weight_sum``.

    sum_i (w_i * loss_i) / sum_i w_i

computed inside the minibatch that was actually drawn. Rows are drawn uniformly
(or in the order of a permutation). Weights are not sampling probabilities.

This is the weighted empirical risk of the rows inside the batch, normalized by
that batch's own weight mass. It is not the gradient of
``sum_all (w_i * loss_i) / sum_all w_i`` when weight mass differs across
batches: each batch is scaled as if its weights already sum to one. A single
batch that contains every training row matches the full-sample objective.
Padding rows use weight zero and drop out of both sums.

Zero-weight rows stay in the training partition, so they still affect fitted
preprocessing. Drop the rows themselves to exclude them from the schema.
Negative or non-finite weights are rejected before any fitted attribute changes.

Class weights are a separate multiplier. For classification the effective
weight is ``sample_weight * class_weight[label]``, applied once. There is no
weighted sampler, so the two mechanisms cannot be stacked by accident.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from ashlar.errors import WeightError

REDUCTION = "batch_weight_sum"


def validate_sample_weight(sample_weight, n_rows: int, *, name: str = "sample_weight") -> np.ndarray:
    """Return a finite, one-dimensional, non-negative weight vector of length ``n_rows``."""
    if sample_weight is None:
        return np.ones(n_rows, dtype=np.float64)
    try:
        weights = np.asarray(sample_weight, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise WeightError(f"{name} must contain numeric values.") from exc
    if weights.shape != (n_rows,):
        raise WeightError(
            f"{name} must have shape ({n_rows},), one weight per row; received shape {weights.shape}."
        )
    if not np.isfinite(weights).all():
        raise WeightError(f"{name} must be finite.")
    if (weights < 0).any():
        raise WeightError(f"{name} must be non-negative. Direct loss weights are not sampling probabilities.")
    return weights


def class_weight_multiplier(y: np.ndarray, n_classes: int, class_weight) -> np.ndarray:
    """Map a class-weight specification to a per-row multiplier.

    ``None`` leaves every row at 1. ``"balanced"`` uses
    ``n_rows / (n_classes * count[label])``. A dict maps labels to multipliers.
    An array of shape ``(n_classes,)`` follows sorted class index order.
    """
    y = np.asarray(y)
    if class_weight is None:
        return np.ones(len(y), dtype=np.float64)
    counts = np.bincount(y.astype(int), minlength=n_classes)
    if isinstance(class_weight, str):
        if class_weight != "balanced":
            raise WeightError("class_weight must be None, 'balanced', a dict, or an array of length n_classes.")
        if np.any(counts == 0):
            raise WeightError("class_weight='balanced' requires every class to appear in the training partition.")
        per_class = len(y) / (n_classes * counts.astype(np.float64))
        return per_class[y.astype(int)]
    if isinstance(class_weight, dict):
        missing = [label for label in np.unique(y).tolist() if label not in class_weight]
        if missing:
            raise WeightError(f"class_weight is missing label(s) {missing}.")
        return np.asarray([float(class_weight[label]) for label in y.tolist()], dtype=np.float64)
    array = np.asarray(class_weight, dtype=np.float64)
    if array.shape != (n_classes,):
        raise WeightError(f"class_weight array must have shape ({n_classes},); received {array.shape}.")
    if not np.isfinite(array).all() or (array < 0).any():
        raise WeightError("class_weight values must be finite and non-negative.")
    return array[y.astype(int)]


def weighted_mean(per_example, sample_weight):
    """``sum(w * loss) / sum(w)`` over a batch. Zero weights contribute to neither sum."""
    per_example = jnp.asarray(per_example)
    sample_weight = jnp.asarray(sample_weight, dtype=per_example.dtype)
    if per_example.shape != sample_weight.shape:
        raise WeightError(
            "per-example loss and sample_weight must share a shape; "
            f"received {per_example.shape} and {sample_weight.shape}."
        )
    total = jnp.sum(sample_weight)
    return jnp.sum(sample_weight * per_example) / jnp.maximum(total, jnp.asarray(1e-12, dtype=per_example.dtype))
