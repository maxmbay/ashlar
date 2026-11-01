"""Explain a function of the predictive distribution, not just its mean.

``explain(model, X, target=...)`` attributes one number per row, read off
``predict_dist``, to the features: a quantile, the standard deviation, the
width of a central interval, the probability of exceeding a threshold, the
mean, or any ``fn(dist) -> (rows,)``. It answers "what drives this row's 95%
quantile" and "which features widen this row's interval".

Numeric features get integrated gradients (Sundararajan et al., 2017) along
the straight line from a baseline row to the row, through imputation, the
fitted scaling, the core, and the distribution object (``forward_dist``).
Their attributions sum to ``f(row) - f(row with every numeric feature at its
baseline)``. A quantile is differentiated by the implicit-function rule, so
its gradient is exact rather than a finite difference.

Categorical features are looked up on the host, so they have no input
gradient. Each gets an occlusion attribution instead: ``f(row)`` minus the
mean of ``f`` with that one column replaced by each reference category
(the background's values, or the training categories with equal weight).
Categorical attributions do not enter the numeric sum.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from ashlar.errors import AshlarError
from ashlar.preprocess import _is_missing, frame_columns

TARGETS = ("quantile", "median", "mean", "std", "interval_width", "exceedance")
METHODS = ("integrated_gradients", "gradient_x_input")
MAX_REFERENCE_CATEGORIES = 32


@dataclass
class Explanation:
    """Per-row attributions of one function of the predictive distribution.

    - ``values``: ``(rows, features)``, one column per feature of ``X`` in
      ``feature_names`` order, in the units of the target (the units of
      ``y`` for a quantile, spread, or width; probability for exceedance).
    - ``prediction``: the target at each row, ``(rows,)``.
    - ``baseline``: the target with every numeric feature at its baseline and
      the categories as given, ``(rows,)``. For numeric features,
      ``values[:, numeric].sum(axis=1)`` is ``prediction - baseline`` up to
      the integration error of ``steps``.
    - ``kinds``: ``"numeric"`` or ``"categorical"`` per feature.
    """

    values: np.ndarray
    feature_names: list[str]
    prediction: np.ndarray
    baseline: np.ndarray
    kinds: list[str]
    target: str
    method: str
    numeric_baseline: dict = field(default_factory=dict)

    def importance(self) -> dict:
        """Mean absolute attribution per feature, largest first."""
        scores = np.mean(np.abs(self.values), axis=0)
        order = np.argsort(-scores)
        return {self.feature_names[i]: float(scores[i]) for i in order}

    def to_frame(self):
        """The attributions as a pandas ``DataFrame``, one column per feature."""
        import pandas as pd

        return pd.DataFrame(self.values, columns=self.feature_names)

    def numeric_gap(self) -> np.ndarray:
        """``prediction - baseline - sum of numeric attributions`` per row: the integration error."""
        numeric = [i for i, kind in enumerate(self.kinds) if kind == "numeric"]
        return self.prediction - self.baseline - self.values[:, numeric].sum(axis=1)


def target_function(
    target: str | Callable = "quantile",
    *,
    level: float = 0.5,
    coverage: float = 0.8,
    threshold: float | None = None,
) -> Callable:
    """``fn(dist) -> (rows,)`` for a named target, as ``explain`` reads it.

    - ``"quantile"``: the ``level`` quantile; ``"median"``: the 0.5 quantile.
    - ``"mean"``; ``"std"``: the predictive standard deviation.
    - ``"interval_width"``: the width of the central ``coverage`` interval.
    - ``"exceedance"``: ``P(y > threshold)``.
    - a callable is returned as it is.
    """
    if callable(target):
        return target
    if target not in TARGETS:
        raise AshlarError(f"target must be one of {', '.join(TARGETS)}, or a callable fn(dist) -> (rows,).")
    if target == "quantile":
        if not 0.0 < float(level) < 1.0:
            raise AshlarError("level must be in (0, 1).")
        return lambda dist: dist.quantile(float(level))
    if target == "median":
        return lambda dist: dist.quantile(0.5)
    if target == "mean":
        return lambda dist: dist.mean()
    if target == "std":
        return lambda dist: jnp.sqrt(dist.variance())
    if target == "interval_width":
        if not 0.0 < float(coverage) < 1.0:
            raise AshlarError("coverage must be in (0, 1).")
        tail = 0.5 * (1.0 - float(coverage))
        return lambda dist: dist.quantile(1.0 - tail) - dist.quantile(tail)
    if threshold is None:
        raise AshlarError("target='exceedance' needs threshold=.")
    return lambda dist: 1.0 - dist.cdf(jnp.asarray(threshold, dtype=jnp.float32))


def explain(
    model,
    X,
    target: str | Callable = "quantile",
    *,
    level: float = 0.5,
    coverage: float = 0.8,
    threshold: float | None = None,
    method: str = "integrated_gradients",
    steps: int = 32,
    background=None,
    batch_rows: int = 512,
) -> Explanation:
    """Attribute a function of each row's predictive distribution to its features.

    ``target`` is one of ``"quantile"`` (at ``level``), ``"median"``,
    ``"mean"``, ``"std"``, ``"interval_width"`` (at ``coverage``),
    ``"exceedance"`` (``P(y > threshold)``), or a callable
    ``fn(dist) -> (rows,)`` on the ``Distribution`` that ``predict_dist``
    returns, for example ``lambda d: d.quantile(0.95) - d.quantile(0.5)``.

    ``method="integrated_gradients"`` (default) integrates the gradient from
    the baseline over ``steps`` midpoints; ``"gradient_x_input"`` takes one
    gradient at the row times ``row - baseline``, which is exact only for a
    target linear in the features.

    The numeric baseline is the training median of each column, or the
    median of ``background`` when given. A missing numeric cell is set to
    its baseline and gets attribution 0. ``background`` also supplies the
    reference categories for categorical columns (at most 32 distinct
    values per column, the most frequent).

    Returns an ``Explanation``; ``.importance()`` ranks features by mean
    absolute attribution and ``.to_frame()`` gives a ``DataFrame``.
    """
    model._require_fitted()
    if method not in METHODS:
        raise AshlarError(f"method must be one of {', '.join(METHODS)}.")
    if int(steps) < 1:
        raise AshlarError("steps must be at least 1.")
    fn = target_function(target, level=level, coverage=coverage, threshold=threshold)
    frame = model._with_cyclical(X)
    names, _ = frame_columns(frame)
    raw_names, _ = frame_columns(X)
    specs = model.encoder_.columns
    numeric_names = [spec.name for spec in specs if spec.role == "numeric"]
    batch = model.encoder_.transform(frame)
    numeric = np.asarray(batch.numeric_raw, dtype=np.float32)
    baseline = _numeric_baseline(model, background, numeric_names)
    numeric = np.where(np.isnan(numeric), baseline[None, :], numeric)

    def value(num, onehot, index, tokens, mask):
        out = fn(model.forward_dist(num, onehot, index, tokens, mask))
        if out.ndim != 1:
            raise AshlarError(
                "the explained target must be one number per row; for a joint head pass a callable such as "
                "lambda d: d.project(weights).quantile(0.9)."
            )
        return out

    value_jit = jax.jit(value)
    grad_jit = jax.jit(jax.grad(lambda *args: value(*args).sum()))
    alphas = (np.arange(int(steps), dtype=np.float32) + 0.5) / int(steps)
    if method == "gradient_x_input":
        alphas = np.ones(1, dtype=np.float32)

    n_rows = numeric.shape[0]
    side = (batch.categorical_onehot, batch.category_index, batch.set_tokens, batch.set_mask)
    numeric_attr = np.zeros_like(numeric)
    prediction = np.zeros(n_rows)
    base_value = np.zeros(n_rows)
    for start in range(0, n_rows, int(batch_rows)):
        rows = slice(start, start + int(batch_rows))
        x = jnp.asarray(numeric[rows])
        x0 = jnp.broadcast_to(jnp.asarray(baseline), x.shape)
        cat = tuple(jnp.asarray(array[rows]) for array in side)
        prediction[rows] = np.asarray(value_jit(x, *cat))
        base_value[rows] = np.asarray(value_jit(x0, *cat))
        if x.shape[1] == 0:
            continue
        total = jnp.zeros_like(x)
        for alpha in alphas:
            total = total + grad_jit(x0 + float(alpha) * (x - x0), *cat)
        numeric_attr[rows] = np.asarray(total / len(alphas) * (x - x0))

    values = np.zeros((n_rows, len(names)))
    kinds = []
    numeric_position = {name: i for i, name in enumerate(numeric_names)}
    for j, spec in enumerate(specs):
        if spec.role == "numeric":
            values[:, j] = numeric_attr[:, numeric_position[spec.name]]
            kinds.append("numeric")
        else:
            values[:, j] = prediction - _occluded(model, frame, names, j, spec, background, baseline, value_jit, batch_rows)
            kinds.append("categorical")
    feature_names = list(names)
    if names != raw_names:
        feature_names, values, kinds = _fold_cyclical(model, raw_names, names, values, kinds)
    return Explanation(
        values=values,
        feature_names=feature_names,
        prediction=prediction,
        baseline=base_value,
        kinds=kinds,
        target=target if isinstance(target, str) else getattr(target, "__name__", "callable"),
        method=method,
        numeric_baseline={name: float(baseline[i]) for i, name in enumerate(numeric_names)},
    )


def _numeric_baseline(model, background, numeric_names) -> np.ndarray:
    if background is None:
        return np.asarray(model.numeric_transform_.median, dtype=np.float32)
    encoded = model.encoder_.transform(model._with_cyclical(background))
    values = np.asarray(encoded.numeric_raw, dtype=np.float64)
    fallback = np.asarray(model.numeric_transform_.median, dtype=np.float64)
    medians = np.array(
        [
            np.nanmedian(values[:, i]) if np.isfinite(values[:, i]).any() else fallback[i]
            for i in range(len(numeric_names))
        ]
    )
    return medians.astype(np.float32)


def _references(spec, background, j, names) -> list:
    if background is not None:
        _, columns = frame_columns(background)
        if j >= len(columns):
            return list(spec.categories[:MAX_REFERENCE_CATEGORIES])
        observed = [value for value in columns[j].tolist() if not _is_missing(value)]
        if observed:
            unique, counts = np.unique(np.asarray([str(value) for value in observed], dtype=object), return_counts=True)
            order = np.argsort(-counts)[:MAX_REFERENCE_CATEGORIES]
            lookup = {str(value): value for value in observed}
            return [lookup[unique[i]] for i in order]
    return list(spec.categories[:MAX_REFERENCE_CATEGORIES])


def _occluded(model, frame, names, j, spec, background, baseline, value_jit, batch_rows) -> np.ndarray:
    """Mean of the target with column ``j`` set to each reference category in turn."""
    _, columns = frame_columns(frame)
    references = _references(spec, None if background is None else model._with_cyclical(background), j, names)
    n_rows = len(columns[0])
    total = np.zeros(n_rows)
    for category in references:
        replaced = dict(zip(names, columns, strict=True))
        replaced[names[j]] = np.full(n_rows, category, dtype=object)
        encoded = model.encoder_.transform(replaced)
        numeric = np.asarray(encoded.numeric_raw, dtype=np.float32)
        numeric = np.where(np.isnan(numeric), baseline[None, :], numeric)
        arrays = (numeric, encoded.categorical_onehot, encoded.category_index, encoded.set_tokens, encoded.set_mask)
        for start in range(0, n_rows, int(batch_rows)):
            rows = slice(start, start + int(batch_rows))
            total[rows] += np.asarray(value_jit(*(jnp.asarray(array[rows]) for array in arrays)))
    return total / len(references)


def _fold_cyclical(model, raw_names, names, values, kinds):
    """Sum a cyclical column's sine and cosine attributions back onto the column."""
    out = np.zeros((values.shape[0], len(raw_names)))
    out_kinds = []
    index = {name: i for i, name in enumerate(names)}
    for k, name in enumerate(raw_names):
        if name in index:
            out[:, k] = values[:, index[name]]
            out_kinds.append(kinds[index[name]])
        else:
            out[:, k] = values[:, index[f"{name}_sin"]] + values[:, index[f"{name}_cos"]]
            out_kinds.append("numeric")
    return list(raw_names), out, out_kinds
