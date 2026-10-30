"""One layer of differentiable oblivious trees.

The layer is the default NODE architecture from Popov, Morozov, and Babenko
(arXiv:1909.06312): depth 6, 1.5-entmax feature choices, 1.5-entmax soft
thresholds, trained end to end. Tree count is reduced from that paper's 2048
so the leaf tensor fits this CPU. The residual regression default does not
use this module.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module
from sklearn.model_selection import train_test_split

from ashlar._compat import renamed
from ashlar.errors import AshlarError
from ashlar.estimator import _stratified_indices

TREE_COUNT = 512
DEPTH = 6
LEARNING_RATE = 0.1
BATCH_SIZE = 512
MAX_EPOCHS = 80
PATIENCE = 16
CHECKPOINT_AVERAGE = 5
SCALE_INIT = 2.0


class ObliviousLayer(Module):
    """Averaged ensemble of oblivious trees. Every split and every leaf is a parameter."""

    feature_logits: jax.Array
    thresholds: jax.Array
    raw_scales: jax.Array
    leaves: jax.Array

    def __call__(self, features: jax.Array) -> jax.Array:
        selectors = entmax15(self.feature_logits)
        chosen = jnp.einsum("bf,tdf->btd", features, selectors)
        scale = jax.nn.softplus(self.raw_scales) + 1e-3
        score = (chosen - self.thresholds[None, :, :]) / scale[None, :, :]
        gate = entmax15(jnp.stack([score, jnp.zeros_like(score)], axis=-1))[..., 0]
        weights = jnp.ones((features.shape[0], gate.shape[1], 1), dtype=features.dtype)
        for level in range(gate.shape[2]):
            decision = gate[:, :, level]
            stay = weights * (1.0 - decision[:, :, None])
            go = weights * decision[:, :, None]
            weights = jnp.concatenate([stay, go], axis=-1)
        return jnp.einsum("btl,tlo->bo", weights, self.leaves) / self.leaves.shape[0]


def _entmax15_forward(scores: jax.Array) -> jax.Array:
    """1.5-entmax along the last axis. Matches the deep-spin forward pass."""

    width = scores.shape[-1]
    centered = scores - jnp.max(scores, axis=-1, keepdims=True)
    scaled = centered / 2.0
    ordered = jnp.flip(jnp.sort(scaled, axis=-1), axis=-1)
    ranks = jnp.arange(1, width + 1, dtype=scaled.dtype).reshape((1,) * (scaled.ndim - 1) + (width,))
    cumulative = jnp.cumsum(ordered, axis=-1)
    cumulative_sq = jnp.cumsum(jnp.square(ordered), axis=-1)
    mean = cumulative / ranks
    mean_sq = cumulative_sq / ranks
    second = ranks * (mean_sq - jnp.square(mean))
    delta = (1.0 - second) / ranks
    tau = mean - jnp.sqrt(jnp.clip(delta, 0.0))
    support = jnp.clip(jnp.sum(tau <= ordered, axis=-1, keepdims=True), 1, width)
    tau_star = jnp.take_along_axis(tau, support.astype(jnp.int32) - 1, axis=-1)
    return jnp.square(jnp.clip(scaled - tau_star, 0.0))


def _entmax15_backward(probs: jax.Array, gradient: jax.Array) -> jax.Array:
    slope = jnp.sqrt(jnp.clip(probs, 0.0))
    propagated = gradient * slope
    total = jnp.sum(slope, axis=-1, keepdims=True)
    correction = jnp.sum(propagated, axis=-1, keepdims=True) / jnp.clip(total, 1e-8)
    return propagated - correction * slope


@jax.custom_vjp
def entmax15(scores: jax.Array) -> jax.Array:
    return _entmax15_forward(scores)


def _entmax15_fwd(scores: jax.Array):
    probs = _entmax15_forward(scores)
    return probs, probs


def _entmax15_bwd(probs: jax.Array, gradient: jax.Array):
    return (_entmax15_backward(probs, gradient),)


entmax15.defvjp(_entmax15_fwd, _entmax15_bwd)


def _labels(y) -> np.ndarray:
    array = np.asarray(y)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise AshlarError(f"labels must be a vector; received shape {array.shape}.")
    return array


def _matrix(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise AshlarError(f"predictors must be a 2-d array; received shape {array.shape}.")
    if not np.isfinite(array).all():
        raise AshlarError("predictors must be finite.")
    return array


def _split_indices(y, task: str, random_state: int):
    y = _labels(y)
    if task == "regression":
        fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=random_state)
        return np.asarray(fit_idx), np.asarray(val_idx)
    if task in ("classification", "ordinal"):
        return _stratified_indices(y, 0.2, random_state)
    raise AshlarError(f"task must be regression, classification, or ordinal; received {task!r}.")


def _scale_init(value: float) -> float:
    safe = max(float(value) - 1e-3, 1e-3)
    return float(np.log(np.expm1(safe)))


def _init_layer(key, features: np.ndarray, n_outputs: int, n_trees: int, depth: int) -> ObliviousLayer:
    logit_key, leaf_key = jax.random.split(key)
    n_rows, n_features = features.shape
    generator = np.random.default_rng(int(np.asarray(jax.random.bits(key))))
    columns = generator.integers(0, n_features, size=(n_trees, depth))
    rows = generator.integers(0, n_rows, size=(n_trees, depth))
    thresholds = features[rows, columns].astype(np.float32)
    return ObliviousLayer(
        feature_logits=jax.random.uniform(logit_key, (n_trees, depth, n_features), dtype=jnp.float32),
        thresholds=jnp.asarray(thresholds),
        raw_scales=jnp.full((n_trees, depth), _scale_init(SCALE_INIT), dtype=jnp.float32),
        leaves=jax.random.normal(leaf_key, (n_trees, 2**depth, n_outputs), dtype=jnp.float32),
    )


def _moments(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = rows.mean(axis=0).astype(np.float32)
    std = rows.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize(rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((rows - mean) / std).astype(np.float32)


def _task_kind(task: str) -> str:
    if task == "regression":
        return "regression"
    return "classes"


def _loss(model: ObliviousLayer, features: jax.Array, target: jax.Array, regression: bool) -> jax.Array:
    prediction = model(features)
    if regression:
        return jnp.mean(jnp.square(prediction[:, 0] - target))
    index = jnp.arange(features.shape[0])
    return -jnp.mean(jax.nn.log_softmax(prediction)[index, target])


def _make_step(learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def step(model, opt_state, features, target, regression: bool):
        loss, grads = eqx.filter_value_and_grad(_loss)(model, features, target, regression)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return optimizer, step


def _predict_raw(model: ObliviousLayer, rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    @eqx.filter_jit
    def apply(model, batch):
        return model(batch)

    standardized = _standardize(rows, mean, std)
    pieces = []
    for start in range(0, len(standardized), 1024):
        batch = jnp.asarray(standardized[start : start + 1024])
        pieces.append(np.asarray(apply(model, batch)))
    return np.concatenate(pieces, axis=0)


def _metric(task: str, y_true, prediction, classes: np.ndarray | None) -> float:
    if task == "classification":
        labels = classes[np.argmax(prediction, axis=1)]
        return float(np.mean(labels == y_true))
    if task == "ordinal":
        expected = prediction_softmax(prediction) @ np.asarray(classes, dtype=np.float64)
        return _r2(np.asarray(y_true, dtype=np.float64), expected)
    return _r2(np.asarray(y_true, dtype=np.float64), prediction)


def prediction_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("-inf")
    total = float(np.sum(np.square(y_true - y_true.mean())))
    if total <= 0.0:
        return float("-inf")
    residual = float(np.sum(np.square(y_true - y_pred)))
    return 1.0 - residual / total


def _snapshot(model: ObliviousLayer) -> ObliviousLayer:
    return jax.tree.map(lambda array: jnp.array(array), model)


def _average(models: list[ObliviousLayer]) -> ObliviousLayer:
    leaves, treedef = jax.tree.flatten(models[0])
    collected = [jax.tree.leaves(model) for model in models]
    averaged = [sum(column) / len(column) for column in zip(*collected, strict=True)]
    return jax.tree.unflatten(treedef, averaged)


class ASRObliviousEnsemble:
    """End-to-end differentiable depth-6 oblivious ensemble for one tabular task."""

    def __init__(
        self,
        random_state: int = 0,
        verbose: bool = False,
        n_trees: int = TREE_COUNT,
        depth: int = DEPTH,
        max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE,
        learning_rate: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
    ):
        if n_trees < 1 or depth < 1:
            raise AshlarError("n_trees and depth must be positive.")
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.n_trees = int(n_trees)
        self.depth = int(depth)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)

    def fit(self, X, y, task: str):
        features = _matrix(X)
        labels = _labels(y)
        if len(labels) != len(features):
            raise AshlarError("X and y have different lengths.")
        if task == "ordinal":
            labels = np.asarray(labels, dtype=np.float64)
            if not np.isfinite(labels).all():
                raise AshlarError("ordinal labels must be finite.")
        if task == "regression" and not np.isfinite(np.asarray(labels, dtype=np.float64)).all():
            raise AshlarError("regression labels must be finite.")
        fit_idx, val_idx = _split_indices(labels, task, self.random_state)
        classes = None if task == "regression" else np.unique(labels)
        outputs = 1 if task == "regression" else len(classes)
        coded = None if classes is None else _class_index(classes, labels)
        best_epoch, selection = self._select_epochs(features, labels, coded, classes, fit_idx, val_idx, outputs, task)
        mean, std = _moments(features)
        target_mean, target_std = (0.0, 1.0)
        train_target = coded
        if task == "regression":
            numeric = np.asarray(labels, dtype=np.float64)
            target_mean = float(numeric.mean())
            target_std = float(numeric.std())
            if target_std < 1e-6:
                target_std = 1.0
            train_target = ((numeric - target_mean) / target_std).astype(np.float32)
        model = self._run(
            features,
            train_target,
            mean,
            std,
            outputs,
            task == "regression",
            best_epoch,
            np.arange(len(features)),
        )
        self.task_ = task
        self.model_ = model
        self.mean_ = mean
        self.std_ = std
        self.target_mean_ = float(target_mean)
        self.target_std_ = float(target_std)
        self.classes_ = classes
        self.best_epoch_ = int(best_epoch)
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        self.n_trees_ = self.n_trees
        self.depth_ = self.depth
        return self

    def _select_epochs(self, features, labels, coded, classes, fit_idx, val_idx, outputs, task: str):
        mean, std = _moments(features[fit_idx])
        if task == "regression":
            numeric = np.asarray(labels, dtype=np.float64)
            target_mean = float(numeric[fit_idx].mean())
            target_std = float(numeric[fit_idx].std())
            if target_std < 1e-6:
                target_std = 1.0
            target = ((numeric - target_mean) / target_std).astype(np.float32)
        else:
            target = coded
            target_mean = 0.0
            target_std = 1.0
        history: list[ObliviousLayer] = []
        model = _init_layer(jax.random.PRNGKey(self.random_state), _standardize(features[fit_idx], mean, std), outputs, self.n_trees, self.depth)
        optimizer, step = _make_step(self.learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        self._step = step
        best_epoch = 1
        best_score = float("-inf")
        wait = 0
        for epoch in range(1, self.max_epochs + 1):
            model, opt_state = self._epoch(model, opt_state, optimizer, features, target, mean, std, fit_idx, epoch, task == "regression")
            history.append(_snapshot(model))
            history = history[-CHECKPOINT_AVERAGE:]
            averaged = _average(history)
            score = self._score(averaged, features[val_idx], labels[val_idx], mean, std, target_mean, target_std, classes, task)
            if self.verbose:
                print(f"  epoch {epoch} selection {score:.6f}", flush=True)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return best_epoch, best_score

    def _run(self, features, target, mean, std, outputs, regression: bool, epochs: int, rows: np.ndarray) -> ObliviousLayer:
        standardized = _standardize(features, mean, std)
        model = _init_layer(jax.random.PRNGKey(self.random_state), standardized[rows], outputs, self.n_trees, self.depth)
        optimizer, step = _make_step(self.learning_rate)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        self._step = step
        history: list[ObliviousLayer] = []
        for epoch in range(1, epochs + 1):
            model, opt_state = self._epoch(model, opt_state, optimizer, features, target, mean, std, rows, epoch, regression)
            history.append(_snapshot(model))
            history = history[-CHECKPOINT_AVERAGE:]
        return _average(history)

    def _epoch(self, model, opt_state, optimizer, features, target, mean, std, rows, epoch: int, regression: bool):
        standardized = _standardize(features, mean, std)
        order = np.asarray(jax.random.permutation(jax.random.PRNGKey(self.random_state + epoch), len(rows)))
        chosen = np.asarray(rows)[order]
        for start in range(0, len(chosen), self.batch_size):
            batch = chosen[start : start + self.batch_size]
            model, opt_state, _loss_value = self._step(
                model,
                opt_state,
                jnp.asarray(standardized[batch]),
                jnp.asarray(target[batch]),
                regression,
            )
        return model, opt_state

    def _score(self, model, rows, labels, mean, std, target_mean, target_std, classes, task: str) -> float:
        raw = _predict_raw(model, rows, mean, std)
        if task == "regression":
            prediction = raw[:, 0] * target_std + target_mean
            return _metric(task, labels, prediction, None)
        return _metric(task, labels, raw, classes)

    def predict(self, X) -> np.ndarray:
        raw = _predict_raw(self.model_, _matrix(X), self.mean_, self.std_)
        if self.task_ == "regression":
            return raw[:, 0] * self.target_std_ + self.target_mean_
        labels = self.classes_[np.argmax(raw, axis=1)]
        if self.task_ == "ordinal":
            return np.asarray(self.classes_, dtype=np.float64)[np.argmax(raw, axis=1)]
        return labels

    def predict_score(self, X) -> np.ndarray:
        raw = _predict_raw(self.model_, _matrix(X), self.mean_, self.std_)
        if self.task_ == "regression":
            return raw[:, 0] * self.target_std_ + self.target_mean_
        probabilities = prediction_softmax(raw)
        if self.classes_ is None:
            raise AshlarError("predict_score on a classification model needs numeric class values.")
        return probabilities @ np.asarray(self.classes_, dtype=np.float64)


def _class_index(classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    mapping = {value.item() if isinstance(value, np.generic) else value: index for index, value in enumerate(classes)}
    coded = []
    for value in labels:
        key = value.item() if isinstance(value, np.generic) else value
        coded.append(mapping[key])
    return np.asarray(coded, dtype=np.int32)


__getattr__ = renamed(
    __name__,
    globals(),
    {
        "ObliviousEnsemble": "ASRObliviousEnsemble",
    },
)
