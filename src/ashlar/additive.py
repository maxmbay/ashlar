"""Differentiable additive models for the five-score bar.

Main effects are piecewise-linear shapes on the numeric columns. A categorical
column contributes a width-8 embedding and a linear map. At most ten pairwise
products of numeric shapes are added, and only when the internal-validation
metric strictly rises. Each product is represented by the leading subspace of
its standardized columns that carries 99.9% of the energy on the fitting rows,
which keeps the same interaction and makes the ridge problem identifiable.
The ridge grid, the eight quantile knots, and the pair cap are fixed. Gradients
flow through the linear map of those bases.

The residual regression default does not use this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from ashlar._compat import renamed
from ashlar.errors import AshlarError
from ashlar.estimator import _stratified_indices

RIDGE_GRID = (1e-4, 1e-2, 1.0, 100.0)
PAIR_CAP = 10
PAIR_ENERGY = 0.999
EMBED_DIM = 8
BASIS_WIDTH = 9
KNOT_FRACTIONS = np.linspace(0.1, 0.8, 8)
EMBED_STEPS = 8
EMBED_STEP_SIZE = 0.5
GATE_TEMPERATURE_SCALE = 0.05
MIN_SIDE_ROWS = 20
NEIGHBOR_SCALE_GRID = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
NEIGHBOR_BANDWIDTH_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
NEIGHBOR_PASSES = 2


class AdditiveBody(Module):
    """Linear map of a fixed basis, plus optional categorical embeddings.

    ``coefficients @ basis`` is the numeric shape. Each embedding is a learned
    vector of width 8, and ``projection`` maps it into the outputs. Both are
    parameters, so a loss of ``logits`` has a gradient in either one.
    """

    coefficients: jax.Array
    intercept: jax.Array
    embeddings: tuple[jax.Array, ...]
    projections: tuple[jax.Array, ...]

    def logits(self, basis: jax.Array, level_ids: tuple[jax.Array, ...] | None = None) -> jax.Array:
        out = basis @ self.coefficients + self.intercept
        if level_ids is None:
            return out
        for index, table, projection in zip(level_ids, self.embeddings, self.projections, strict=True):
            out = out + table[index] @ projection
        return out


class NeighborhoodBody(Module):
    """Distance-weighted prediction with one scale per column and one bandwidth."""

    scales: jax.Array
    bandwidth: jax.Array
    reference: jax.Array
    values: jax.Array

    def predict(self, query: jax.Array) -> jax.Array:
        scaled_query = query * self.scales
        scaled_reference = self.reference * self.scales
        distance2 = (
            jnp.sum(scaled_query * scaled_query, axis=1, keepdims=True)
            + jnp.sum(scaled_reference * scaled_reference, axis=1)[None, :]
            - 2.0 * scaled_query @ scaled_reference.T
        )
        log_weight = -0.5 * distance2 / jnp.square(self.bandwidth)
        log_weight = log_weight - jnp.max(log_weight, axis=1, keepdims=True)
        weight = jnp.exp(log_weight)
        weight = weight / jnp.sum(weight, axis=1, keepdims=True)
        return weight @ self.values


def _as_float_matrix(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise AshlarError(f"numeric predictors must be a 2-d array; received shape {array.shape}.")
    if not np.isfinite(array).all():
        raise AshlarError("numeric predictors must be finite.")
    return array


def _labels(y) -> np.ndarray:
    array = np.asarray(y)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise AshlarError(f"labels must be a vector; received shape {array.shape}.")
    return array


def _frame_parts(X):
    """Split a matrix or a dataframe into numeric columns and categorical columns."""

    if hasattr(X, "columns") and hasattr(X, "to_numpy"):
        import pandas as pd

        numeric_names = []
        categorical_names = []
        for name in X.columns:
            series = X[name]
            if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
                categorical_names.append(name)
            else:
                numeric_names.append(name)
        numeric = np.zeros((len(X), len(numeric_names)), dtype=np.float64) if numeric_names else np.zeros((len(X), 0))
        if numeric_names:
            numeric = X.loc[:, numeric_names].to_numpy(dtype=np.float64)
        categoricals = [X.loc[:, name].to_numpy() for name in categorical_names]
        return numeric, categoricals
    return _as_float_matrix(X), []


def _split_indices(y, task: str, random_state: int):
    y = _labels(y)
    if task == "regression":
        fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=random_state)
        return np.asarray(fit_idx), np.asarray(val_idx)
    if task in ("classification", "ordinal"):
        return _stratified_indices(y, 0.2, random_state)
    raise AshlarError(f"task must be regression, classification, or ordinal; received {task!r}.")


def _hinges(column: np.ndarray, knots: np.ndarray) -> np.ndarray:
    centered = column[:, None] - knots[None, :]
    return np.concatenate([column[:, None], np.maximum(centered, 0.0)], axis=1)


@dataclass
class FeatureMap:
    knots: list[np.ndarray]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_rows(cls, rows: np.ndarray) -> "FeatureMap":
        knots = []
        blocks = []
        for index in range(rows.shape[1]):
            column = rows[:, index]
            knot = np.quantile(column, KNOT_FRACTIONS).astype(np.float64)
            knots.append(knot)
            blocks.append(_hinges(column, knot))
        raw = np.concatenate(blocks, axis=1) if blocks else np.zeros((len(rows), 0))
        mean = raw.mean(axis=0) if raw.size else np.zeros(0)
        std = raw.std(axis=0) if raw.size else np.zeros(0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(knots=knots, mean=mean, std=std)

    def transform(self, rows: np.ndarray) -> np.ndarray:
        if rows.shape[1] == 0:
            return np.zeros((len(rows), 0))
        blocks = [_hinges(rows[:, index], knot) for index, knot in enumerate(self.knots)]
        raw = np.concatenate(blocks, axis=1)
        return (raw - self.mean) / self.std


def _pair_block(standardized: np.ndarray, pair: tuple[int, int], n_features: int) -> np.ndarray:
    width = BASIS_WIDTH
    left, right = pair
    if standardized.shape[1] != n_features * width:
        raise AshlarError("the standardized basis does not match the feature count.")
    left_block = standardized[:, left * width : (left + 1) * width]
    right_block = standardized[:, right * width : (right + 1) * width]
    product = left_block[:, :, None] * right_block[:, None, :]
    return product.reshape(len(standardized), width * width)


def _fit_pair_transform(block: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leading subspace of one pairwise product, fit on the rows being scored.

    The product has ``BASIS_WIDTH ** 2`` columns. Directions whose singular
    values carry the last tenth of a percent of energy are dropped, and so are
    numerical null directions. The retained columns are centered and scaled
    from these same rows.
    """

    mean = block.mean(axis=0)
    centered = block - mean
    width = centered.shape[1]
    if width == 0 or not np.isfinite(centered).all() or float(np.linalg.norm(centered)) == 0.0:
        directions = np.zeros((width, 1))
        if width:
            directions[0, 0] = 1.0
        return mean, directions, np.ones(1)
    _left, singular, vt = np.linalg.svd(centered, full_matrices=False)
    singular = np.asarray(singular, dtype=np.float64)
    energy = np.cumsum(singular ** 2)
    total = float(energy[-1]) if len(energy) else 0.0
    if total <= 0.0 or singular[0] <= 0.0:
        directions = np.zeros((width, 1))
        directions[0, 0] = 1.0
        return mean, directions, np.ones(1)
    rank = int(np.searchsorted(energy / total, PAIR_ENERGY) + 1)
    numeric_rank = int(np.sum(singular > 1e-8 * singular[0]))
    rank = max(1, min(rank, numeric_rank if numeric_rank else 1, len(singular)))
    directions = np.asarray(vt[:rank], dtype=np.float64).T
    scores = centered @ directions
    scale = scores.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return mean, directions, scale


def _apply_pair(block: np.ndarray, transform: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mean, directions, scale = transform
    return ((block - mean) @ directions) / scale


def _design(standardized: np.ndarray, pairs: list[tuple[int, int]], pair_stats: list[tuple[np.ndarray, np.ndarray, np.ndarray]], n_features: int) -> np.ndarray:
    if not pairs:
        return standardized
    blocks = [_apply_pair(_pair_block(standardized, pair, n_features), transform) for pair, transform in zip(pairs, pair_stats, strict=True)]
    return np.concatenate([standardized, *blocks], axis=1)


def _candidate_designs(standardized_fit: np.ndarray, standardized_val: np.ndarray, pairs: list[tuple[int, int]], pair_stats, extra: tuple[int, int], n_features: int):
    """Designs for the current pairs plus one candidate. The new subspace is fit on the fit rows."""

    base_fit = _design(standardized_fit, pairs, pair_stats, n_features)
    base_val = _design(standardized_val, pairs, pair_stats, n_features)
    block_fit = _pair_block(standardized_fit, extra, n_features)
    transform = _fit_pair_transform(block_fit)
    extra_fit = _apply_pair(block_fit, transform)
    extra_val = _apply_pair(_pair_block(standardized_val, extra, n_features), transform)
    return np.concatenate([base_fit, extra_fit], axis=1), np.concatenate([base_val, extra_val], axis=1), transform


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _fit_ridge(design: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    """Minimizer of mean squared error plus ``ridge * mean(slope ** 2)``.

    The intercept is unpenalized. ``design`` does not include a column of ones.
    """

    target = np.asarray(y, dtype=np.float64).reshape(-1)
    n_rows, n_columns = design.shape
    if n_columns == 0:
        return np.zeros(0), float(target.mean())
    ones = np.ones((n_rows, 1))
    gram = np.empty((n_columns + 1, n_columns + 1))
    gram[0, 0] = n_rows
    gram[0, 1:] = design.sum(axis=0)
    gram[1:, 0] = gram[0, 1:]
    gram[1:, 1:] = design.T @ design
    penalty = n_rows * ridge / n_columns
    gram[1:, 1:][np.diag_indices(n_columns)] += penalty
    right = np.empty(n_columns + 1)
    right[0] = target.sum()
    right[1:] = design.T @ target
    try:
        weight = np.linalg.solve(gram, right)
    except np.linalg.LinAlgError:
        weight = np.linalg.lstsq(gram, right, rcond=None)[0]
    return weight[1:], float(weight[0])


def _fit_logistic(design: np.ndarray, y: np.ndarray, ridge: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multinomial or binary logistic regression. Returns coefficients aligned to ``classes``.

    The objective is scikit-learn's L2 logistic loss at ``C = 1 / ridge``.
    Newton-Cholesky converges on the piecewise-linear design; L-BFGS was
    stopping at the iteration cap with the same penalty.
    """

    import warnings

    from scipy.linalg import LinAlgWarning
    from sklearn.exceptions import ConvergenceWarning

    labels = _labels(y)
    classes = np.unique(labels)
    n_columns = design.shape[1]
    if len(classes) == 1:
        coef = np.zeros((n_columns, 1))
        return coef, np.zeros(1), classes
    penalty = 1.0 / float(ridge)
    model = LogisticRegression(C=penalty, solver="newton-cholesky", max_iter=50, tol=1e-4, fit_intercept=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", LinAlgWarning)
        model.fit(design, labels)
    if len(classes) == 2:
        slope = np.zeros((n_columns, 2))
        intercept = np.zeros(2)
        slope[:, 1] = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        intercept[1] = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
        return slope, intercept, np.asarray(model.classes_)
    slope = np.asarray(model.coef_, dtype=np.float64).T
    intercept = np.asarray(model.intercept_, dtype=np.float64).reshape(-1)
    return slope, intercept, np.asarray(model.classes_)


def _predict_logits(design: np.ndarray, coef: np.ndarray, intercept: np.ndarray) -> np.ndarray:
    if design.shape[1] == 0:
        return np.broadcast_to(intercept, (len(design), intercept.shape[0])).copy()
    return design @ coef + intercept


def _task_score(y_true, prediction, task: str) -> float:
    if len(y_true) < 2:
        return float("-inf")
    if task == "classification":
        return float(accuracy_score(y_true, prediction))
    return float(r2_score(np.asarray(y_true, dtype=np.float64), np.asarray(prediction, dtype=np.float64)))


def _regression_prediction(design, coef, intercept) -> np.ndarray:
    if design.shape[1] == 0:
        return np.full(len(design), intercept)
    return design @ coef + intercept


def _class_prediction(design, coef, intercept, classes) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _softmax(_predict_logits(design, coef, intercept))
    labels = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    return labels, probabilities


def _fit_outputs(design, y, ridge: float, task: str):
    if task in ("regression", "ordinal_mean"):
        coef, intercept = _fit_ridge(design, y, ridge)
        return coef.reshape(-1, 1), np.asarray([intercept]), None
    coef, intercept, classes = _fit_logistic(design, y, ridge)
    return coef, intercept, classes


def _score_design(design, y, coef, intercept, classes, task: str) -> float:
    if task in ("regression", "ordinal_mean"):
        prediction = _regression_prediction(design, coef.reshape(-1), float(intercept.reshape(-1)[0]))
        return _task_score(y, prediction, "regression")
    _labels_out, probabilities = _class_prediction(design, coef, intercept, classes)
    if task == "ordinal":
        expected = probabilities @ np.asarray(classes, dtype=np.float64)
        return _task_score(y, expected, "regression")
    return _task_score(y, _labels_out, "classification")


@dataclass
class _Solution:
    pairs: list[tuple[int, int]]
    pair_stats: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    ridge: float
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray | None
    selection_metric: float


def _choose_ridge(design_fit, y_fit, design_val, y_val, task: str) -> tuple[float, np.ndarray, np.ndarray, np.ndarray | None, float]:
    best = None
    for ridge in RIDGE_GRID:
        coef, intercept, classes = _fit_outputs(design_fit, y_fit, ridge, task)
        score = _score_design(design_val, y_val, coef, intercept, classes, task)
        if best is None or score > best[0]:
            best = (score, ridge, coef, intercept, classes)
    assert best is not None
    score, ridge, coef, intercept, classes = best
    return ridge, coef, intercept, classes, score


def _score_one_candidate(extra, z_fit, z_val, y_fit, y_val, pairs, pair_stats, n_features, ridge, task):
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        design_fit, design_val, stats = _candidate_designs(z_fit, z_val, pairs, pair_stats, extra, n_features)
        coef, intercept, classes = _fit_outputs(design_fit, y_fit, ridge, task)
        score = _score_design(design_val, y_val, coef, intercept, classes, task)
    return extra, score, stats, coef, intercept, classes


def _score_candidates(candidates, z_fit, z_val, y_fit, y_val, pairs, pair_stats, n_features, ridge, task):
    """Score pair candidates in list order. Large tables use one process per core."""

    if len(candidates) < 12 or len(z_fit) < 800:
        return [
            _score_one_candidate(extra, z_fit, z_val, y_fit, y_val, pairs, pair_stats, n_features, ridge, task)
            for extra in candidates
        ]
    from joblib import Parallel, delayed

    workers = min(4, len(candidates))
    return Parallel(n_jobs=workers, prefer="processes")(
        delayed(_score_one_candidate)(extra, z_fit, z_val, y_fit, y_val, pairs, pair_stats, n_features, ridge, task)
        for extra in candidates
    )


def _select_pairs(standardized_fit, y_fit, standardized_val, y_val, task: str, n_features: int, pair_cap: int, row_fit: np.ndarray | None = None, row_val: np.ndarray | None = None, verbose: bool = False) -> _Solution:
    """Forward-select pairs on one internal split. Moments come from the rows being fit."""

    if row_fit is None:
        row_fit = np.ones(len(standardized_fit), dtype=bool)
    if row_val is None:
        row_val = np.ones(len(standardized_val), dtype=bool)
    z_fit = standardized_fit[row_fit]
    z_val = standardized_val[row_val]
    y_side_fit = np.asarray(y_fit)[row_fit]
    y_side_val = np.asarray(y_val)[row_val]

    ridge, coef, intercept, classes, current = _choose_ridge(z_fit, y_side_fit, z_val, y_side_val, task)
    pairs: list[tuple[int, int]] = []
    pair_stats: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    candidates = [(left, right) for left in range(n_features) for right in range(left + 1, n_features)]
    for round_index in range(pair_cap):
        if verbose:
            print(f"  select round {round_index + 1} candidates {len(candidates)}", flush=True)
        winner = None
        winner_score = current
        winner_stats = None
        winner_coef = None
        winner_intercept = None
        winner_classes = None
        scored = _score_candidates(candidates, z_fit, z_val, y_side_fit, y_side_val, pairs, pair_stats, n_features, ridge, task)
        for extra, score, stats, trial_coef, trial_intercept, trial_classes in scored:
            if score > winner_score:
                winner = extra
                winner_score = score
                winner_stats = stats
                winner_coef = trial_coef
                winner_intercept = trial_intercept
                winner_classes = trial_classes
        if winner is None:
            break
        pairs.append(winner)
        pair_stats.append(winner_stats)
        candidates.remove(winner)
        current = winner_score
        coef, intercept, classes = winner_coef, winner_intercept, winner_classes
        if verbose:
            print(f"  pair {len(pairs)} {winner} rank {winner_stats[1].shape[1]} selection {current:.6f}", flush=True)
    design_fit = _design(z_fit, pairs, pair_stats, n_features)
    design_val = _design(z_val, pairs, pair_stats, n_features)
    ridge, coef, intercept, classes, current = _choose_ridge(design_fit, y_side_fit, design_val, y_side_val, task)
    return _Solution(pairs, pair_stats, ridge, coef, intercept, classes, current)


def _refit(design, y, ridge: float, task: str, classes_hint):
    coef, intercept, classes = _fit_outputs(design, y, ridge, task)
    if classes is None:
        classes = classes_hint
    return coef, intercept, classes


def _level_maps(columns: list[np.ndarray], fit_idx: np.ndarray):
    maps = []
    for column in columns:
        values = column[fit_idx]
        levels = pd_unique(values)
        mapping = {level: index for index, level in enumerate(levels)}
        maps.append((levels, mapping))
    return maps


def pd_unique(values: np.ndarray) -> np.ndarray:
    _classes, index = np.unique(values, return_index=True)
    # np.unique sorts. Preserve first-seen order so the embedding rows are stable.
    order = np.sort(index)
    return values[order]


def _encode_levels(column: np.ndarray, mapping: dict, n_known: int) -> np.ndarray:
    encoded = np.full(len(column), n_known, dtype=np.int32)
    for index, value in enumerate(column):
        key = value.item() if isinstance(value, np.generic) else value
        found = mapping.get(key)
        if found is not None:
            encoded[index] = found
    return encoded


def _embedding_block(ids: np.ndarray, table: np.ndarray) -> np.ndarray:
    return table[ids]


def _class_index(classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mapping = {value.item() if isinstance(value, np.generic) else value: index for index, value in enumerate(classes)}
    return np.asarray([mapping[value.item() if isinstance(value, np.generic) else value] for value in labels], dtype=np.int32)


def _update_embedding(ids: np.ndarray, table: np.ndarray, design_numeric: np.ndarray, coef: np.ndarray, intercept: np.ndarray, y: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """One gradient step on the embedding table. The linear map is the trailing block of ``coef``."""

    extra = _embedding_block(ids, table)
    numeric_width = design_numeric.shape[1]
    logits = _predict_logits(np.concatenate([design_numeric, extra], axis=1), coef, intercept)
    probabilities = _softmax(logits)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y)), _class_index(classes, y)] = 1.0
    residual = (probabilities - one_hot) / max(len(y), 1)
    projection = coef[numeric_width:, :]
    row_grad = residual @ projection.T
    gradient = np.zeros_like(table)
    np.add.at(gradient, ids, row_grad)
    return table - EMBED_STEP_SIZE * gradient


class ASRAdditive:
    """Main effects, a width-8 embedding per categorical, and at most ten pairs."""

    def __init__(self, random_state: int = 0, verbose: bool = False):
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

    def fit(self, X, y, task: str):
        numeric, categoricals = _frame_parts(X)
        labels = _labels(y)
        if len(labels) != len(numeric):
            raise AshlarError("X and y have different lengths.")
        if task == "ordinal":
            labels = np.asarray(labels, dtype=np.float64)
            if not np.isfinite(labels).all():
                raise AshlarError("ordinal labels must be finite.")
        fit_idx, val_idx = _split_indices(labels, task, self.random_state)
        feature_map = FeatureMap.from_rows(numeric[fit_idx])
        standardized_fit = feature_map.transform(numeric[fit_idx])
        standardized_val = feature_map.transform(numeric[val_idx])
        # Ordinal fits a class distribution and selects on expected-label R².
        fit_task = task
        solution = _select_pairs(
            standardized_fit,
            labels[fit_idx],
            standardized_val,
            labels[val_idx],
            fit_task,
            numeric.shape[1],
            PAIR_CAP,
            verbose=self.verbose,
        )
        level_state = _fit_embeddings(
            categoricals,
            fit_idx,
            val_idx,
            numeric,
            labels,
            feature_map,
            solution,
            fit_task,
            self.random_state,
        )
        standardized_all = feature_map.transform(numeric)
        design_all = _design(standardized_all, solution.pairs, solution.pair_stats, numeric.shape[1])
        design_all, embedding_tables = _attach_embeddings(design_all, categoricals, level_state, np.arange(len(labels)))
        coef, intercept, classes = _refit(design_all, labels, solution.ridge, fit_task, solution.classes)
        if level_state is not None:
            coef, intercept, classes, embedding_tables = _refine_embeddings(
                design_all[:, : _numeric_width(solution, numeric.shape[1])],
                embedding_tables,
                level_state,
                categoricals,
                labels,
                solution.ridge,
                fit_task,
                classes,
            )
            design_width = _numeric_width(solution, numeric.shape[1])
            coef = coef
            _ = design_width
        self.task_ = task
        self.feature_map_ = feature_map
        self.n_features_ = int(numeric.shape[1])
        self.pairs_ = list(solution.pairs)
        self.pair_stats_ = solution.pair_stats
        self.ridge_ = float(solution.ridge)
        self.coef_ = coef
        self.intercept_ = intercept
        self.classes_ = classes
        self.class_values_ = np.asarray(classes, dtype=np.float64) if task == "ordinal" else None
        self.selection_metric_ = float(solution.selection_metric)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        self.level_state_ = level_state
        self.embedding_tables_ = embedding_tables
        self.body_ = _make_body(coef, intercept, embedding_tables)
        return self

    def _design(self, X) -> np.ndarray:
        numeric, categoricals = _frame_parts(X)
        standardized = self.feature_map_.transform(numeric)
        design = _design(standardized, self.pairs_, self.pair_stats_, self.n_features_)
        design, _tables = _attach_embeddings(design, categoricals, self.level_state_, None, self.embedding_tables_)
        return design

    def predict(self, X) -> np.ndarray:
        design = self._design(X)
        if self.task_ in ("regression", "ordinal"):
            values = _regression_prediction(design, self.coef_.reshape(-1), float(self.intercept_.reshape(-1)[0])) if self.task_ == "regression" else None
            if self.task_ == "regression":
                return values
        labels, _probabilities = _class_prediction(design, self.coef_, self.intercept_, self.classes_)
        if self.task_ == "ordinal":
            return np.asarray(self.class_values_)[np.argmax(_softmax(_predict_logits(design, self.coef_, self.intercept_)), axis=1)]
        return labels

    def predict_score(self, X) -> np.ndarray:
        design = self._design(X)
        if self.task_ == "regression":
            return _regression_prediction(design, self.coef_.reshape(-1), float(self.intercept_.reshape(-1)[0]))
        _labels_out, probabilities = _class_prediction(design, self.coef_, self.intercept_, self.classes_)
        if self.class_values_ is None:
            raise AshlarError("predict_score on a classification model needs numeric class values.")
        return probabilities @ self.class_values_


def _numeric_width(solution: _Solution, n_features: int) -> int:
    pair_width = sum(int(transform[1].shape[1]) for transform in solution.pair_stats)
    return n_features * BASIS_WIDTH + pair_width


def _make_body(coef, intercept, embedding_tables) -> AdditiveBody:
    embeddings = tuple(jnp.asarray(table) for table in embedding_tables) if embedding_tables else ()
    projections: tuple[jax.Array, ...] = ()
    if embedding_tables:
        offset = coef.shape[0] - sum(table.shape[1] for table in embedding_tables)
        numeric = coef[:offset]
        slices = []
        for table in embedding_tables:
            slices.append(jnp.asarray(coef[offset : offset + table.shape[1]]))
            offset += table.shape[1]
        projections = tuple(slices)
        coefficients = jnp.asarray(numeric)
    else:
        coefficients = jnp.asarray(coef)
    return AdditiveBody(
        coefficients=coefficients,
        intercept=jnp.asarray(intercept),
        embeddings=embeddings,
        projections=projections,
    )


def _fit_embeddings(categoricals, fit_idx, val_idx, numeric, labels, feature_map, solution, fit_task, random_state):
    if not categoricals:
        return None
    return _level_maps(categoricals, fit_idx)


def _attach_embeddings(design, categoricals, level_state, rows, tables=None):
    if level_state is None:
        return design, [] if tables is None else tables
    generator = np.random.default_rng(0)
    built = []
    blocks = []
    for index, (levels, mapping) in enumerate(level_state):
        if tables is None:
            table = generator.normal(scale=0.01, size=(len(levels) + 1, EMBED_DIM))
        else:
            table = tables[index]
        column = categoricals[index] if rows is None else categoricals[index][rows]
        ids = _encode_levels(column, mapping, len(levels))
        blocks.append(_embedding_block(ids, table))
        built.append(table)
    if not blocks:
        return design, built
    return np.concatenate([design, *blocks], axis=1), built


def _refine_embeddings(numeric_design, tables, level_state, categoricals, labels, ridge, fit_task, classes):
    """Alternate a linear refit with a gradient step on each width-8 embedding."""

    current_tables = [table.copy() for table in tables]
    coef = intercept = found = None
    for _step in range(EMBED_STEPS):
        id_list = []
        blocks = []
        for index, (levels, mapping) in enumerate(level_state):
            ids = _encode_levels(categoricals[index], mapping, len(levels))
            id_list.append(ids)
            blocks.append(_embedding_block(ids, current_tables[index]))
        design = np.concatenate([numeric_design, *blocks], axis=1)
        coef, intercept, found = _refit(design, labels, ridge, fit_task, classes)
        offset = numeric_design.shape[1]
        rebuilt = []
        for index, ids in enumerate(id_list):
            slice_coef = np.concatenate([coef[: numeric_design.shape[1]], coef[offset : offset + EMBED_DIM]], axis=0)
            rebuilt.append(_update_embedding(ids, current_tables[index], numeric_design, slice_coef, intercept, labels, found))
            offset += EMBED_DIM
        current_tables = rebuilt
    blocks = []
    for index, (levels, mapping) in enumerate(level_state):
        ids = _encode_levels(categoricals[index], mapping, len(levels))
        blocks.append(_embedding_block(ids, current_tables[index]))
    design = np.concatenate([numeric_design, *blocks], axis=1) if blocks else numeric_design
    coef, intercept, found = _refit(design, labels, ridge, fit_task, classes)
    return coef, intercept, found, current_tables


def _equally_spaced(labels: np.ndarray):
    values = np.unique(labels)
    if len(values) < 3:
        return False, values
    number = np.asarray(values, dtype=np.float64)
    if not np.isfinite(number).all():
        return False, values
    gaps = np.diff(number)
    spaced = bool(np.allclose(gaps, gaps[0], rtol=1e-5, atol=1e-8) and gaps[0] > 0)
    return spaced, values


def _mode(labels: np.ndarray):
    values, counts = np.unique(labels, return_counts=True)
    return values[int(np.argmax(counts))]


class ASRGatedAdditive:
    """Two additive specialists behind a gate learned from the training labels or one feature."""

    def __init__(self, random_state: int = 0, verbose: bool = False):
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

    def fit(self, X, y, task: str):
        numeric, categoricals = _frame_parts(X)
        if categoricals:
            raise AshlarError("the gate is fit on numeric columns. Categorical embeddings stay on the additive model.")
        labels = _labels(y)
        if task == "ordinal":
            labels = np.asarray(labels, dtype=np.float64)
        fit_idx, val_idx = _split_indices(labels, task, self.random_state)
        feature_map = FeatureMap.from_rows(numeric[fit_idx])
        z_fit = feature_map.transform(numeric[fit_idx])
        z_val = feature_map.transform(numeric[val_idx])
        y_fit = labels[fit_idx]
        y_val = labels[val_idx]
        raw_fit = numeric[fit_idx]
        spaced, support = _equally_spaced(labels)
        specialist_cap = PAIR_CAP // 2
        if spaced and task != "classification":
            peak, kind = _grow_peak(z_fit, y_fit, z_val, y_val, support, numeric.shape[1], verbose=self.verbose)
            if kind == "feature":
                feature, threshold, temperature = _search_threshold(raw_fit, z_fit, y_fit, raw_val=numeric[val_idx], z_val=z_val, y_val=y_val, task=task, n_features=numeric.shape[1], verbose=self.verbose)
                self._fit_feature_gate(feature_map, numeric, labels, fit_idx, val_idx, feature, threshold, temperature, task, specialist_cap)
            else:
                self._fit_label_gate(feature_map, numeric, labels, fit_idx, val_idx, peak, task, specialist_cap)
        else:
            feature, threshold, temperature = _search_threshold(raw_fit, z_fit, y_fit, numeric[val_idx], z_val, y_val, task, numeric.shape[1], self.verbose)
            self._fit_feature_gate(feature_map, numeric, labels, fit_idx, val_idx, feature, threshold, temperature, task, specialist_cap)
        self.task_ = task
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        return self

    def _fit_label_gate(self, feature_map, numeric, labels, fit_idx, val_idx, peak, task, specialist_cap):
        z_all_fit = feature_map.transform(numeric[fit_idx])
        z_all_val = feature_map.transform(numeric[val_idx])
        y_fit = labels[fit_idx]
        y_val = labels[val_idx]
        inside_fit = np.isin(y_fit, peak)
        inside_val = np.isin(y_val, peak)
        outside_fit = ~inside_fit
        outside_val = ~inside_val
        mean_task = "ordinal_mean"
        inside = _select_pairs(z_all_fit, y_fit, z_all_val, y_val, mean_task, numeric.shape[1], specialist_cap, inside_fit, inside_val, self.verbose)
        outside = _select_pairs(z_all_fit, y_fit, z_all_val, y_val, mean_task, numeric.shape[1], specialist_cap, outside_fit, outside_val, self.verbose)
        membership_fit = inside_fit.astype(np.int32)
        membership_val = inside_val.astype(np.int32)
        gate = _select_pairs(z_all_fit, membership_fit, z_all_val, membership_val, "classification", numeric.shape[1], 0, verbose=self.verbose)
        z_all = feature_map.transform(numeric)
        inside_all = np.isin(labels, peak)
        self.kind_ = "label"
        self.peak_ = np.asarray(peak)
        self.feature_map_ = feature_map
        self.n_features_ = int(numeric.shape[1])
        self.inside_ = _refit_solution(z_all[inside_all], labels[inside_all], inside, mean_task, numeric.shape[1])
        self.outside_ = _refit_solution(z_all[~inside_all], labels[~inside_all], outside, mean_task, numeric.shape[1])
        self.gate_ = _refit_solution(z_all, inside_all.astype(np.int32), gate, "classification", numeric.shape[1])
        self.feature_index_ = None
        self.threshold_ = None
        self.temperature_ = None

    def _fit_feature_gate(self, feature_map, numeric, labels, fit_idx, val_idx, feature, threshold, temperature, task, specialist_cap):
        raw_fit = numeric[fit_idx, feature]
        raw_val = numeric[val_idx, feature]
        z_fit = feature_map.transform(numeric[fit_idx])
        z_val = feature_map.transform(numeric[val_idx])
        y_fit = labels[fit_idx]
        y_val = labels[val_idx]
        left_fit = raw_fit <= threshold
        right_fit = ~left_fit
        left_val = raw_val <= threshold
        right_val = ~left_val
        fit_task = "classification" if task == "classification" else "ordinal_mean"
        left = _select_pairs(z_fit, y_fit, z_val, y_val, fit_task, numeric.shape[1], specialist_cap, left_fit, left_val, self.verbose)
        right = _select_pairs(z_fit, y_fit, z_val, y_val, fit_task, numeric.shape[1], specialist_cap, right_fit, right_val, self.verbose)
        z_all = feature_map.transform(numeric)
        left_all = numeric[:, feature] <= threshold
        self.kind_ = "feature"
        self.peak_ = None
        self.feature_map_ = feature_map
        self.n_features_ = int(numeric.shape[1])
        self.inside_ = _refit_solution(z_all[left_all], labels[left_all], left, fit_task, numeric.shape[1])
        self.outside_ = _refit_solution(z_all[~left_all], labels[~left_all], right, fit_task, numeric.shape[1])
        self.gate_ = None
        self.feature_index_ = int(feature)
        self.threshold_ = float(threshold)
        self.temperature_ = float(temperature)

    def predict_score(self, X) -> np.ndarray:
        numeric, _categoricals = _frame_parts(X)
        standardized = self.feature_map_.transform(numeric)
        if self.kind_ == "label":
            gate_design = _design(standardized, self.gate_.pairs, self.gate_.pair_stats, self.n_features_)
            _labels_out, probabilities = _class_prediction(gate_design, self.gate_.coef, self.gate_.intercept, self.gate_.classes)
            class_list = list(self.gate_.classes)
            positive = class_list.index(1) if 1 in class_list else int(np.argmax(self.gate_.classes))
            membership = probabilities[:, positive]
            inside = _mean_prediction(standardized, self.inside_, self.n_features_)
            outside = _mean_prediction(standardized, self.outside_, self.n_features_)
            return membership * inside + (1.0 - membership) * outside
        side = _sigmoid((numeric[:, self.feature_index_] - self.threshold_) / self.temperature_)
        if self.task_ == "classification":
            classes = self._feature_classes()
            left = _reindex_proba(
                _aligned_probabilities(standardized, self.inside_, self.n_features_, None),
                self.inside_.classes,
                classes,
            )
            right = _reindex_proba(
                _aligned_probabilities(standardized, self.outside_, self.n_features_, None),
                self.outside_.classes,
                classes,
            )
            return (1.0 - side)[:, None] * left + side[:, None] * right
        left = _mean_prediction(standardized, self.inside_, self.n_features_)
        right = _mean_prediction(standardized, self.outside_, self.n_features_)
        return (1.0 - side) * left + side * right

    def predict(self, X) -> np.ndarray:
        if self.task_ == "classification" and self.kind_ == "feature":
            probabilities = self.predict_score(X)
            return self._feature_classes()[np.argmax(probabilities, axis=1)]
        if self.task_ == "classification":
            raise AshlarError("label gates are for equally spaced regression and ordinal targets.")
        return self.predict_score(X)

    def _feature_classes(self):
        left = [] if self.inside_.classes is None else list(self.inside_.classes)
        right = [] if self.outside_.classes is None else list(self.outside_.classes)
        return np.unique(np.asarray(left + right))


def _mean_prediction(standardized, solution: _Solution, n_features: int) -> np.ndarray:
    design = _design(standardized, solution.pairs, solution.pair_stats, n_features)
    return _regression_prediction(design, solution.coef.reshape(-1), float(solution.intercept.reshape(-1)[0]))


def _prediction_any(standardized, solution: _Solution, n_features: int, task: str):
    design = _design(standardized, solution.pairs, solution.pair_stats, n_features)
    if task == "classification":
        _labels_out, probabilities = _class_prediction(design, solution.coef, solution.intercept, solution.classes)
        return probabilities
    return _mean_prediction(standardized, solution, n_features)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _refit_solution(standardized_rows, y_rows, solution: _Solution, task: str, n_features: int) -> _Solution:
    if len(y_rows) == 0:
        raise AshlarError("a specialist has no rows to refit.")
    design = _design(standardized_rows, solution.pairs, solution.pair_stats, n_features)
    coef, intercept, classes = _refit(design, y_rows, solution.ridge, task, solution.classes)
    return _Solution(solution.pairs, solution.pair_stats, solution.ridge, coef, intercept, classes, solution.selection_metric)


def _grow_peak(z_fit, y_fit, z_val, y_val, support, n_features: int, verbose: bool):
    mode = _mode(y_fit)
    positions = {value: index for index, value in enumerate(support)}
    if mode not in positions:
        mode = support[int(np.argmin(np.abs(np.asarray(support, dtype=np.float64) - float(mode))))]
    left = right = positions[mode]
    peak = [support[left]]

    def legal(lo, hi):
        members = support[lo : hi + 1]
        inside = np.isin(y_fit, members)
        outside = ~inside
        return int(inside.sum()) >= MIN_SIDE_ROWS and int(outside.sum()) >= MIN_SIDE_ROWS

    def score(lo, hi):
        members = support[lo : hi + 1]
        inside = np.isin(y_fit, members)
        outside = ~inside
        inside_val = np.isin(y_val, members)
        outside_val = ~inside_val
        if inside_val.sum() < 2 or outside_val.sum() < 2:
            return float("-inf")
        inside_model = _select_pairs(z_fit, y_fit, z_val, y_val, "ordinal_mean", n_features, 0, inside, inside_val, False)
        outside_model = _select_pairs(z_fit, y_fit, z_val, y_val, "ordinal_mean", n_features, 0, outside, outside_val, False)
        membership = inside.astype(np.int32)
        gate = _select_pairs(z_fit, membership, z_val, inside_val.astype(np.int32), "classification", n_features, 0, verbose=False)
        prediction = _mixture_mean(z_val, inside_model, outside_model, gate, n_features)
        return _task_score(y_val, prediction, "regression")

    if not legal(left, right):
        return peak, "feature"
    current = score(left, right)
    while True:
        options = []
        if left > 0 and legal(left - 1, right):
            options.append((left - 1, right))
        if right + 1 < len(support) and legal(left, right + 1):
            options.append((left, right + 1))
        best_option = None
        best_score = current
        for option in options:
            trial = score(*option)
            if trial > best_score:
                best_score = trial
                best_option = option
        if best_option is None:
            break
        left, right = best_option
        current = best_score
        if verbose:
            print(f"  peak {list(support[left : right + 1])} selection {current:.6f}", flush=True)
    return list(support[left : right + 1]), "label"


def _mixture_mean(standardized, inside: _Solution, outside: _Solution, gate: _Solution, n_features: int) -> np.ndarray:
    gate_design = _design(standardized, gate.pairs, gate.pair_stats, n_features)
    _labels_out, probabilities = _class_prediction(gate_design, gate.coef, gate.intercept, gate.classes)
    class_list = [int(value) if not isinstance(value, str) else value for value in gate.classes]
    positive = class_list.index(1) if 1 in class_list else 0
    membership = probabilities[:, positive]
    left = _mean_prediction(standardized, inside, n_features)
    right = _mean_prediction(standardized, outside, n_features)
    return membership * left + (1.0 - membership) * right


def _search_threshold(raw_fit, z_fit, y_fit, raw_val, z_val, y_val, task, n_features, verbose):
    fit_task = "classification" if task == "classification" else "ordinal_mean"
    best = None
    for feature in range(raw_fit.shape[1]):
        column = raw_fit[:, feature]
        std = float(column.std())
        if std < 1e-8:
            continue
        thresholds = np.unique(np.quantile(column, KNOT_FRACTIONS))
        temperature = GATE_TEMPERATURE_SCALE * std
        for threshold in thresholds:
            left_fit = column <= threshold
            right_fit = ~left_fit
            if int(left_fit.sum()) < MIN_SIDE_ROWS or int(right_fit.sum()) < MIN_SIDE_ROWS:
                continue
            left_val = raw_val[:, feature] <= threshold
            right_val = ~left_val
            if int(left_val.sum()) < 2 or int(right_val.sum()) < 2:
                continue
            left = _select_pairs(z_fit, y_fit, z_val, y_val, fit_task, n_features, 0, left_fit, left_val, False)
            right = _select_pairs(z_fit, y_fit, z_val, y_val, fit_task, n_features, 0, right_fit, right_val, False)
            score = _routed_score(z_val, raw_val[:, feature], threshold, temperature, left, right, y_val, task, n_features)
            if best is None or score > best[0]:
                best = (score, feature, float(threshold), temperature)
    if best is None:
        raise AshlarError("no feature threshold left at least twenty fitting rows on both sides.")
    if verbose:
        print(f"  threshold feature {best[1]} value {best[2]:.6g} selection {best[0]:.6f}", flush=True)
    return best[1], best[2], best[3]


def _routed_score(z_val, raw_val, threshold, temperature, left, right, y_val, task, n_features) -> float:
    side = _sigmoid((raw_val - threshold) / temperature)
    if task == "classification":
        left_p = _aligned_probabilities(z_val, left, n_features, y_val)
        right_p = _aligned_probabilities(z_val, right, n_features, y_val)
        # Align to the union of classes.
        classes = np.unique(np.concatenate([np.asarray(left.classes), np.asarray(right.classes)]))
        left_full = _reindex_proba(left_p, left.classes, classes)
        right_full = _reindex_proba(right_p, right.classes, classes)
        mixed = (1.0 - side)[:, None] * left_full + side[:, None] * right_full
        labels = classes[np.argmax(mixed, axis=1)]
        return _task_score(y_val, labels, "classification")
    left_mean = _mean_prediction(z_val, left, n_features)
    right_mean = _mean_prediction(z_val, right, n_features)
    prediction = (1.0 - side) * left_mean + side * right_mean
    return _task_score(y_val, prediction, "regression")


def _aligned_probabilities(standardized, solution: _Solution, n_features: int, y_unused) -> np.ndarray:
    design = _design(standardized, solution.pairs, solution.pair_stats, n_features)
    _labels_out, probabilities = _class_prediction(design, solution.coef, solution.intercept, solution.classes)
    return probabilities


def _reindex_proba(probabilities, source_classes, target_classes) -> np.ndarray:
    out = np.zeros((len(probabilities), len(target_classes)))
    source = list(source_classes)
    target = list(target_classes)
    for index, label in enumerate(source):
        out[:, target.index(label)] = probabilities[:, index]
    return out


class ASRDiagonalNeighborhood:
    """Nadaraya–Watson or a class vote in a learned diagonal metric."""

    def __init__(self, random_state: int = 0, verbose: bool = False):
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

    def fit(self, X, y, task: str):
        numeric, categoricals = _frame_parts(X)
        if categoricals:
            raise AshlarError("the neighborhood on this bar uses numeric columns. Embeddings stay on the additive model.")
        labels = _labels(y)
        if task != "classification":
            labels = np.asarray(labels, dtype=np.float64)
        fit_idx, val_idx = _split_indices(labels, task, self.random_state)
        mean = numeric[fit_idx].mean(axis=0)
        std = numeric[fit_idx].std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        scaled = (numeric - mean) / std
        scales = np.ones(numeric.shape[1], dtype=np.float64)
        bandwidth = _choose_bandwidth(scaled[fit_idx], labels[fit_idx], scaled[val_idx], labels[val_idx], scales, task)
        for _pass in range(NEIGHBOR_PASSES):
            for feature in range(numeric.shape[1]):
                current = _neighbor_score(scaled[fit_idx], labels[fit_idx], scaled[val_idx], labels[val_idx], scales, bandwidth, task)
                best_scale = scales[feature]
                best_score = current
                for scale in NEIGHBOR_SCALE_GRID:
                    trial = scales.copy()
                    trial[feature] = scale
                    score = _neighbor_score(scaled[fit_idx], labels[fit_idx], scaled[val_idx], labels[val_idx], trial, bandwidth, task)
                    if score > best_score:
                        best_score = score
                        best_scale = scale
                scales[feature] = best_scale
            bandwidth = _choose_bandwidth(scaled[fit_idx], labels[fit_idx], scaled[val_idx], labels[val_idx], scales, task)
        self.task_ = task
        self.mean_ = mean
        self.std_ = std
        self.scales_ = scales
        self.bandwidth_ = float(bandwidth)
        self.reference_ = scaled
        self.reference_y_ = labels
        self.classes_ = np.unique(labels) if task == "classification" else None
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        self.selection_metric_ = float(
            _neighbor_score(scaled[fit_idx], labels[fit_idx], scaled[val_idx], labels[val_idx], scales, bandwidth, task)
        )
        self.body_ = NeighborhoodBody(
            scales=jnp.asarray(scales),
            bandwidth=jnp.asarray(bandwidth),
            reference=jnp.asarray(scaled[fit_idx]),
            values=jnp.asarray(np.zeros(len(fit_idx)) if task == "classification" else labels[fit_idx]),
        )
        if self.verbose:
            print(f"  neighborhood bandwidth {self.bandwidth_:.6g} selection {self.selection_metric_:.6f}", flush=True)
        return self

    def predict_score(self, X) -> np.ndarray:
        if self.task_ == "classification":
            return self._probabilities(X)
        return self._mean(X)

    def predict(self, X) -> np.ndarray:
        if self.task_ == "classification":
            probabilities = self._probabilities(X)
            return self.classes_[np.argmax(probabilities, axis=1)]
        return self._mean(X)

    def _scaled(self, X) -> np.ndarray:
        numeric, _categoricals = _frame_parts(X)
        return (numeric - self.mean_) / self.std_

    def _mean(self, X) -> np.ndarray:
        return _nw_mean(self.reference_, self.reference_y_.astype(np.float64), self._scaled(X), self.scales_, self.bandwidth_)

    def _probabilities(self, X) -> np.ndarray:
        return _nw_vote(self.reference_, self.reference_y_, self._scaled(X), self.scales_, self.bandwidth_, self.classes_)


def _neighbor_score(reference, y_reference, query, y_query, scales, bandwidth, task: str) -> float:
    if task == "classification":
        probabilities = _nw_vote(reference, y_reference, query, scales, bandwidth, np.unique(y_reference))
        labels = np.unique(y_reference)[np.argmax(probabilities, axis=1)]
        return _task_score(y_query, labels, "classification")
    prediction = _nw_mean(reference, np.asarray(y_reference, dtype=np.float64), query, scales, bandwidth)
    return _task_score(y_query, prediction, "regression")


def _choose_bandwidth(reference, y_reference, query, y_query, scales, task: str) -> float:
    median = _median_distance(reference, scales)
    best_bandwidth = median
    best_score = float("-inf")
    for factor in NEIGHBOR_BANDWIDTH_FACTORS:
        bandwidth = max(median * factor, 1e-3)
        score = _neighbor_score(reference, y_reference, query, y_query, scales, bandwidth, task)
        if score > best_score:
            best_score = score
            best_bandwidth = bandwidth
    return float(best_bandwidth)


def _median_distance(reference, scales) -> float:
    sample = reference[: min(300, len(reference))]
    scaled = sample * scales
    gram = scaled @ scaled.T
    squared = np.clip(np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2.0 * gram, 0.0, None)
    iu = np.triu_indices(len(sample), k=1)
    distances = np.sqrt(squared[iu])
    if len(distances) == 0:
        return 1.0
    median = float(np.median(distances))
    return median if median > 1e-8 else 1.0


def _nw_mean(reference, values, query, scales, bandwidth) -> np.ndarray:
    output = np.empty(len(query))
    scaled_reference = reference * scales
    ref_norm = np.sum(scaled_reference * scaled_reference, axis=1)
    for start in range(0, len(query), 512):
        block = query[start : start + 512] * scales
        distance2 = ref_norm[None, :] + np.sum(block * block, axis=1, keepdims=True) - 2.0 * block @ scaled_reference.T
        output[start : start + 512] = _weighted_sum(distance2, values, bandwidth)
    return output


def _nw_vote(reference, labels, query, scales, bandwidth, classes) -> np.ndarray:
    class_list = list(classes)
    index = np.empty(len(labels), dtype=np.int32)
    mapping = {label: position for position, label in enumerate(class_list)}
    for row, label in enumerate(labels):
        key = label.item() if isinstance(label, np.generic) else label
        index[row] = mapping[key]
    one_hot = np.eye(len(class_list))[index]
    votes = np.empty((len(query), len(class_list)))
    scaled_reference = reference * scales
    ref_norm = np.sum(scaled_reference * scaled_reference, axis=1)
    for start in range(0, len(query), 512):
        block = query[start : start + 512] * scales
        distance2 = ref_norm[None, :] + np.sum(block * block, axis=1, keepdims=True) - 2.0 * block @ scaled_reference.T
        weights = _weights(distance2, bandwidth)
        votes[start : start + 512] = weights @ one_hot
    return votes


def _weights(distance2: np.ndarray, bandwidth: float) -> np.ndarray:
    log_weight = -0.5 * np.maximum(distance2, 0.0) / (bandwidth * bandwidth)
    log_weight -= log_weight.max(axis=1, keepdims=True)
    weight = np.exp(log_weight)
    total = weight.sum(axis=1, keepdims=True)
    total = np.where(total <= 0, 1.0, total)
    return weight / total


def _weighted_sum(distance2: np.ndarray, values: np.ndarray, bandwidth: float) -> np.ndarray:
    return _weights(distance2, bandwidth) @ values


__getattr__ = renamed(
    __name__,
    globals(),
    {
        "AdditiveModel": "ASRAdditive",
        "GatedAdditive": "ASRGatedAdditive",
        "DiagonalNeighborhood": "ASRDiagonalNeighborhood",
    },
)
