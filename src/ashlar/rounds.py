"""Three nonlinear tabular fitters that do not replace the residual default.

Histogram boosting grows leaf-wise constant-leaf trees on binned features.
Linear-leaf boosting grows one split at a time and fits a ridge line in each
side. Learned neighborhood trains an embedding and predicts from labels of
nearby rows in that embedding. None of the three is the residual MLP.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from ashlar.errors import AshlarError
from ashlar.estimator import _stratified_indices

HISTOGRAM_LEARNING_RATE = 0.1
HISTOGRAM_MAX_ITER = 300
HISTOGRAM_PATIENCE = 20
HISTOGRAM_MAX_LEAF_NODES = 31
HISTOGRAM_MIN_SAMPLES_LEAF = 20
HISTOGRAM_L2 = 0.0
HISTOGRAM_MAX_BINS = 255

LEAF_SHRINKAGE = 0.1
LEAF_MAX_STAGES = 200
LEAF_PATIENCE = 20
LEAF_MIN_ROWS = 32
LEAF_THRESHOLDS = 16
LEAF_RIDGE = 1.0

NEIGHBOR_WIDTH = 64
NEIGHBOR_EMBED_DIM = 32
NEIGHBOR_DROPOUT = 0.1
NEIGHBOR_LEARNING_RATE = 1e-3
NEIGHBOR_WEIGHT_DECAY = 1e-4
NEIGHBOR_BATCH_SIZE = 256
NEIGHBOR_MAX_EPOCHS = 80
NEIGHBOR_PATIENCE = 16
NEIGHBOR_CHECKPOINT_AVERAGE = 5


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
    return np.ascontiguousarray(array)


def _split_indices(y, task: str, random_state: int):
    y = _labels(y)
    if task == "regression":
        fit_idx, val_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=random_state)
        return np.asarray(fit_idx), np.asarray(val_idx)
    if task in ("classification", "ordinal"):
        return _stratified_indices(y, 0.2, random_state)
    raise AshlarError(f"task must be regression, classification, or ordinal; received {task!r}.")


def _check_task(task: str, features: np.ndarray, labels: np.ndarray) -> None:
    if len(labels) != len(features):
        raise AshlarError("X and y have different lengths.")
    if task == "ordinal":
        numeric = np.asarray(labels, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise AshlarError("ordinal labels must be finite.")
    elif task == "regression":
        numeric = np.asarray(labels, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise AshlarError("regression labels must be finite.")
    elif task != "classification":
        raise AshlarError(f"task must be regression, classification, or ordinal; received {task!r}.")


def _moments(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = rows.mean(axis=0)
    std = rows.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize(rows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (rows - mean) / std


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if len(y_true) < 2:
        return float("-inf")
    total = float(np.sum(np.square(y_true - y_true.mean())))
    if total <= 0.0:
        return float("-inf")
    residual = float(np.sum(np.square(y_true - y_pred)))
    return 1.0 - residual / total


def _codes(classes: np.ndarray, labels: np.ndarray) -> np.ndarray:
    index = {value: position for position, value in enumerate(classes.tolist())}
    try:
        return np.asarray([index[value] for value in labels.tolist()], dtype=np.int32)
    except KeyError as error:
        raise AshlarError(f"label {error} is outside the fitted classes.") from error


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _accuracy(y_true, y_pred) -> float:
    true = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    if len(true) != len(pred):
        raise AshlarError("accuracy received different lengths.")
    return float(np.sum(true == pred)) / len(true)


def _score_outputs(task: str, labels: np.ndarray, outputs: np.ndarray, classes: np.ndarray | None) -> float:
    if task == "regression":
        return _r2(labels, outputs[:, 0] if outputs.ndim == 2 else outputs)
    if classes is None:
        raise AshlarError("class scores need the fitted class values.")
    if task == "classification":
        return _accuracy(labels, classes[np.argmax(outputs, axis=1)])
    expected = _softmax(outputs) @ np.asarray(classes, dtype=np.float64)
    return _r2(labels, expected)


def _histogram_estimator(task: str, rounds: int, random_state: int):
    kwargs = dict(
        learning_rate=HISTOGRAM_LEARNING_RATE,
        max_iter=int(rounds),
        max_leaf_nodes=HISTOGRAM_MAX_LEAF_NODES,
        max_depth=None,
        min_samples_leaf=HISTOGRAM_MIN_SAMPLES_LEAF,
        l2_regularization=HISTOGRAM_L2,
        max_bins=HISTOGRAM_MAX_BINS,
        early_stopping=False,
        random_state=int(random_state),
    )
    if task == "regression":
        return HistGradientBoostingRegressor(loss="squared_error", **kwargs)
    return HistGradientBoostingClassifier(loss="log_loss", **kwargs)


class HistogramBooster:
    """Leaf-wise histogram trees. Ordinal readout is the expected label."""

    def __init__(self, random_state: int = 0, verbose: bool = False, max_iter: int = HISTOGRAM_MAX_ITER, patience: int = HISTOGRAM_PATIENCE):
        if max_iter < 1 or patience < 1:
            raise AshlarError("max_iter and patience must be positive.")
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_iter = int(max_iter)
        self.patience = int(patience)

    def fit(self, X, y, task: str):
        features = _matrix(X)
        labels = _labels(y)
        _check_task(task, features, labels)
        fit_idx, val_idx = _split_indices(labels, task, self.random_state)
        best_iter, selection = self._select(features[fit_idx], labels[fit_idx], features[val_idx], labels[val_idx], task)
        model = _histogram_estimator(task, best_iter, self.random_state)
        model.fit(features, labels if task != "regression" else np.asarray(labels, dtype=np.float64))
        self.task_ = task
        self.model_ = model
        self.best_iter_ = int(best_iter)
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        self.classes_ = None if task == "regression" else np.asarray(model.classes_)
        return self

    def predict(self, X) -> np.ndarray:
        features = _matrix(X)
        if self.task_ == "regression":
            return np.asarray(self.model_.predict(features), dtype=np.float64)
        return np.asarray(self.model_.predict(features))

    def predict_score(self, X) -> np.ndarray:
        features = _matrix(X)
        if self.task_ == "regression":
            return np.asarray(self.model_.predict(features), dtype=np.float64)
        probabilities = np.asarray(self.model_.predict_proba(features), dtype=np.float64)
        return probabilities @ np.asarray(self.model_.classes_, dtype=np.float64)

    def _select(self, x_fit, y_fit, x_val, y_val, task: str) -> tuple[int, float]:
        model = _histogram_estimator(task, self.max_iter, self.random_state)
        y_fit_use = np.asarray(y_fit, dtype=np.float64) if task == "regression" else y_fit
        model.fit(x_fit, y_fit_use)
        best_iter = 1
        best_score = float("-inf")
        wait = 0
        staged = model.staged_predict(x_val) if task == "regression" else model.staged_predict_proba(x_val)
        for iteration, prediction in enumerate(staged, start=1):
            score = self._staged_score(task, y_val, prediction, None if task == "regression" else model.classes_)
            if score > best_score:
                best_score = score
                best_iter = iteration
                wait = 0
                if self.verbose:
                    print(f"  histogram iter {iteration} selection {score:.6f}", flush=True)
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if self.verbose:
            print(f"  histogram selected {best_iter} selection {best_score:.6f}", flush=True)
        return best_iter, best_score

    @staticmethod
    def _staged_score(task: str, labels, prediction, classes) -> float:
        if task == "regression":
            return _r2(labels, prediction)
        if task == "classification":
            return _accuracy(labels, np.asarray(classes)[np.argmax(prediction, axis=1)])
        expected = np.asarray(prediction, dtype=np.float64) @ np.asarray(classes, dtype=np.float64)
        return _r2(labels, expected)


def _min_leaf(n_rows: int, n_features: int) -> int:
    return int(max(n_features + 2, min(LEAF_MIN_ROWS, n_rows // 5)))


def _cut_positions(column: np.ndarray, min_leaf: int) -> list[int]:
    n_rows = len(column)
    change = np.flatnonzero(column[1:] > column[:-1]) + 1
    change = change[(change >= min_leaf) & (change <= n_rows - min_leaf)]
    if len(change) == 0:
        return []
    fractions = np.linspace(0.05, 0.95, LEAF_THRESHOLDS)
    chosen = []
    for fraction in fractions:
        guess = int(np.clip(round(float(fraction) * n_rows), min_leaf, n_rows - min_leaf))
        chosen.append(int(change[np.argmin(np.abs(change - guess))]))
    return sorted(set(chosen))


def _ridge_beta(xtx: np.ndarray, xty: np.ndarray, n_rows: int) -> np.ndarray:
    """Slopes take a mean-square penalty. The intercept stays free."""

    width = xtx.shape[0]
    n_slopes = width - 1
    system = np.array(xtx, copy=True)
    if n_slopes > 0 and n_rows > 0:
        diag = np.arange(1, width)
        system[diag, diag] += n_rows * LEAF_RIDGE / n_slopes
    system[0, 0] += 1e-8
    try:
        return np.linalg.solve(system, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(system, xty, rcond=None)[0]


def _side_sse(xtx: np.ndarray, xty: np.ndarray, yty: float, n_rows: int) -> tuple[float, np.ndarray]:
    beta = _ridge_beta(xtx, xty, n_rows)
    quad = xtx @ beta
    sse = float(yty - 2.0 * np.sum(beta * xty) + np.sum(beta * quad))
    if not np.isfinite(sse):
        return float("inf"), beta
    return max(sse, 0.0), beta


def _best_leaf(scaled: np.ndarray, target: np.ndarray):
    n_rows, n_features = scaled.shape
    min_leaf = _min_leaf(n_rows, n_features)
    if n_rows < 2 * min_leaf:
        return None
    center = target.mean(axis=0, keepdims=True)
    base = float(np.sum(np.square(target - center)))
    tolerance = 1e-8 * max(1.0, base)
    aug = np.concatenate([np.ones((n_rows, 1), dtype=np.float64), np.asarray(scaled, dtype=np.float64)], axis=1)
    best_sse = base
    best = None
    response = np.asarray(target, dtype=np.float64)
    for feature in range(n_features):
        order = np.argsort(scaled[:, feature], kind="mergesort")
        column = np.asarray(scaled[order, feature], dtype=np.float64)
        cuts = _cut_positions(column, min_leaf)
        if not cuts:
            continue
        design = aug[order]
        outcome = response[order]
        outer = design[:, :, None] * design[:, None, :]
        xtx_prefix = np.cumsum(outer, axis=0)
        xty_prefix = np.cumsum(design[:, :, None] * outcome[:, None, :], axis=0)
        yty_prefix = np.cumsum(np.sum(outcome * outcome, axis=1), axis=0)
        total_xtx = xtx_prefix[-1]
        total_xty = xty_prefix[-1]
        total_yty = float(yty_prefix[-1])
        for cut in cuts:
            left_rows = int(cut)
            right_rows = n_rows - left_rows
            left_xtx = xtx_prefix[cut - 1]
            left_xty = xty_prefix[cut - 1]
            left_yty = float(yty_prefix[cut - 1])
            left_sse, left_beta = _side_sse(left_xtx, left_xty, left_yty, left_rows)
            right_sse, right_beta = _side_sse(total_xtx - left_xtx, total_xty - left_xty, total_yty - left_yty, right_rows)
            sse = left_sse + right_sse
            if sse + tolerance < best_sse:
                threshold = float(0.5 * (column[cut - 1] + column[cut]))
                best_sse = sse
                best = (feature, threshold, left_beta, right_beta)
    return best


def _leaf_predict(scaled: np.ndarray, feature: int, threshold: float, beta_left: np.ndarray, beta_right: np.ndarray) -> np.ndarray:
    aug = np.concatenate([np.ones((len(scaled), 1), dtype=np.float64), np.asarray(scaled, dtype=np.float64)], axis=1)
    left = scaled[:, feature] <= threshold
    prediction = aug @ beta_right
    if np.any(left):
        prediction = np.array(prediction, copy=True)
        prediction[left] = aug[left] @ beta_left
    return prediction


class LinearLeafBooster:
    """Boosted depth-1 trees whose leaves are ridge linear models."""

    def __init__(self, random_state: int = 0, verbose: bool = False, max_stages: int = LEAF_MAX_STAGES, patience: int = LEAF_PATIENCE):
        if max_stages < 1 or patience < 1:
            raise AshlarError("max_stages and patience must be positive.")
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_stages = int(max_stages)
        self.patience = int(patience)

    def fit(self, X, y, task: str):
        features = _matrix(X)
        labels = _labels(y)
        _check_task(task, features, labels)
        fit_idx, val_idx = _split_indices(labels, task, self.random_state)
        classes = None if task == "regression" else np.unique(labels)
        best_stages, selection = self._select(features, labels, classes, fit_idx, val_idx, task)
        mean, std, init, stages = self._grow(features, labels, classes, task, best_stages)
        self.task_ = task
        self.mean_ = mean
        self.std_ = std
        self.init_ = init
        self.stages_ = stages
        self.classes_ = classes
        self.best_stages_ = int(best_stages)
        self.n_stages_ = int(len(stages))
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        return self

    def predict(self, X) -> np.ndarray:
        outputs = self._replay(_matrix(X))
        if self.task_ == "regression":
            return outputs[:, 0]
        return self.classes_[np.argmax(outputs, axis=1)]

    def predict_score(self, X) -> np.ndarray:
        outputs = self._replay(_matrix(X))
        if self.task_ == "regression":
            return outputs[:, 0]
        if self.classes_ is None:
            raise AshlarError("predict_score needs class values.")
        return _softmax(outputs) @ np.asarray(self.classes_, dtype=np.float64)

    def _select(self, features, labels, classes, fit_idx, val_idx, task: str) -> tuple[int, float]:
        mean, std = _moments(features[fit_idx])
        fit_scaled = _standardize(features[fit_idx], mean, std)
        val_scaled = _standardize(features[val_idx], mean, std)
        fit_state, val_state, init = self._initial_state(labels[fit_idx], classes, task, len(fit_idx), len(val_idx))
        best_stages = 0
        best_score = _score_outputs(task, labels[val_idx], val_state, classes)
        wait = 0
        target_source = self._target_source(labels[fit_idx], classes, task)
        if self.verbose:
            print(f"  linear-leaf stage 0 selection {best_score:.6f}", flush=True)
        kept = 0
        for stage in range(1, self.max_stages + 1):
            target = self._residual(task, fit_state, target_source)
            found = _best_leaf(fit_scaled, target)
            if found is None:
                break
            feature, threshold, beta_left, beta_right = found
            fit_state = fit_state + LEAF_SHRINKAGE * _leaf_predict(fit_scaled, feature, threshold, beta_left, beta_right)
            val_state = val_state + LEAF_SHRINKAGE * _leaf_predict(val_scaled, feature, threshold, beta_left, beta_right)
            score = _score_outputs(task, labels[val_idx], val_state, classes)
            kept = stage
            if score > best_score:
                best_score = score
                best_stages = stage
                wait = 0
                if self.verbose:
                    print(f"  linear-leaf stage {stage} selection {score:.6f}", flush=True)
            else:
                wait += 1
                if wait >= self.patience:
                    break
        del kept, init
        if self.verbose:
            print(f"  linear-leaf selected {best_stages} selection {best_score:.6f}", flush=True)
        return best_stages, best_score

    def _grow(self, features, labels, classes, task: str, stages: int):
        mean, std = _moments(features)
        scaled = _standardize(features, mean, std)
        state, _unused, init = self._initial_state(labels, classes, task, len(features), 0)
        target_source = self._target_source(labels, classes, task)
        built = []
        for _stage in range(stages):
            found = _best_leaf(scaled, self._residual(task, state, target_source))
            if found is None:
                break
            feature, threshold, beta_left, beta_right = found
            state = state + LEAF_SHRINKAGE * _leaf_predict(scaled, feature, threshold, beta_left, beta_right)
            built.append((int(feature), float(threshold), np.asarray(beta_left, dtype=np.float64), np.asarray(beta_right, dtype=np.float64)))
        return mean, std, init, built

    def _replay(self, features: np.ndarray) -> np.ndarray:
        scaled = _standardize(features, self.mean_, self.std_)
        state = np.repeat(self.init_.reshape(1, -1), len(features), axis=0)
        for feature, threshold, beta_left, beta_right in self.stages_:
            state = state + LEAF_SHRINKAGE * _leaf_predict(scaled, feature, threshold, beta_left, beta_right)
        return state

    @staticmethod
    def _initial_state(labels, classes, task: str, n_fit: int, n_val: int):
        if task == "regression":
            numeric = np.asarray(labels, dtype=np.float64)
            init = np.asarray([float(numeric.mean())], dtype=np.float64)
            fit_state = np.full((n_fit, 1), init[0], dtype=np.float64)
            val_state = np.full((n_val, 1), init[0], dtype=np.float64)
            return fit_state, val_state, init
        codes = _codes(classes, labels)
        counts = np.bincount(codes, minlength=len(classes)).astype(np.float64)
        priors = np.clip(counts / counts.sum(), 1e-6, None)
        priors = priors / priors.sum()
        init = np.log(priors)
        fit_state = np.repeat(init.reshape(1, -1), n_fit, axis=0)
        val_state = np.repeat(init.reshape(1, -1), n_val, axis=0)
        return fit_state, val_state, init

    @staticmethod
    def _target_source(labels, classes, task: str) -> np.ndarray:
        if task == "regression":
            return np.asarray(labels, dtype=np.float64).reshape(-1, 1)
        return np.eye(len(classes), dtype=np.float64)[_codes(classes, labels)]

    @staticmethod
    def _residual(task: str, state: np.ndarray, target_source: np.ndarray) -> np.ndarray:
        if task == "regression":
            return target_source - state
        return target_source - _softmax(state)


class Encoder(Module):
    """Feature map and positive temperature for the learned neighborhood."""

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array
    raw_temperature: jax.Array


def _dropout(values: jax.Array, key, rate: float) -> jax.Array:
    keep = jax.random.bernoulli(key, 1.0 - rate, values.shape)
    return jnp.where(keep, values / (1.0 - rate), 0.0)


def _embed_train(model: Encoder, rows: jax.Array, key) -> tuple[jax.Array, jax.Array]:
    first, second = jax.random.split(key)
    hidden = jax.nn.relu(rows @ model.w1 + model.b1)
    hidden = _dropout(hidden, first, NEIGHBOR_DROPOUT)
    hidden = jax.nn.relu(hidden @ model.w2 + model.b2)
    hidden = _dropout(hidden, second, NEIGHBOR_DROPOUT)
    embedding = hidden @ model.w3 + model.b3
    temperature = jax.nn.softplus(model.raw_temperature) + 1e-4
    return embedding, temperature


def _embed_eval(model: Encoder, rows: jax.Array) -> jax.Array:
    hidden = jax.nn.relu(rows @ model.w1 + model.b1)
    hidden = jax.nn.relu(hidden @ model.w2 + model.b2)
    return hidden @ model.w3 + model.b3


def _temperature(model: Encoder) -> jax.Array:
    return jax.nn.softplus(model.raw_temperature) + 1e-4


def _class_nll(embedding: jax.Array, temperature: jax.Array, targets: jax.Array) -> jax.Array:
    distance = jnp.sum(jnp.square(embedding[:, None, :] - embedding[None, :, :]), axis=-1)
    logits = -distance / temperature
    width = logits.shape[0]
    eye = jnp.eye(width, dtype=bool)
    logits = jnp.where(eye, -1e9, logits)
    same = (targets[:, None] == targets[None, :]) & ~eye
    positive = jnp.where(same, logits, -1e9)
    log_num = jax.nn.logsumexp(positive, axis=1)
    log_den = jax.nn.logsumexp(logits, axis=1)
    has_positive = jnp.any(same, axis=1)
    nll = jnp.where(has_positive, log_den - log_num, 0.0)
    return jnp.sum(nll) / jnp.maximum(jnp.sum(has_positive), 1.0)


def _regression_mse(embedding: jax.Array, temperature: jax.Array, targets: jax.Array) -> jax.Array:
    distance = jnp.sum(jnp.square(embedding[:, None, :] - embedding[None, :, :]), axis=-1)
    logits = -distance / temperature
    width = logits.shape[0]
    logits = logits.at[jnp.arange(width), jnp.arange(width)].set(-1e9)
    weights = jax.nn.softmax(logits, axis=1)
    prediction = weights @ targets
    return jnp.mean(jnp.square(prediction - targets))


def _init_encoder(key, n_features: int) -> Encoder:
    first, second, third = jax.random.split(key, 3)

    def weight(subkey, fan_in: int, fan_out: int, scale: float) -> jax.Array:
        return jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32) * np.float32(scale)

    raw = np.log(np.expm1(np.float32(1.0 - 1e-4)))
    return Encoder(
        w1=weight(first, n_features, NEIGHBOR_WIDTH, float(np.sqrt(2.0 / n_features))),
        b1=jnp.zeros((NEIGHBOR_WIDTH,), dtype=jnp.float32),
        w2=weight(second, NEIGHBOR_WIDTH, NEIGHBOR_WIDTH, float(np.sqrt(2.0 / NEIGHBOR_WIDTH))),
        b2=jnp.zeros((NEIGHBOR_WIDTH,), dtype=jnp.float32),
        w3=weight(third, NEIGHBOR_WIDTH, NEIGHBOR_EMBED_DIM, float(np.sqrt(1.0 / NEIGHBOR_WIDTH))),
        b3=jnp.zeros((NEIGHBOR_EMBED_DIM,), dtype=jnp.float32),
        raw_temperature=jnp.asarray(raw, dtype=jnp.float32),
    )


def _make_neighbor_step(learning_rate: float, regression: bool):
    optimizer = optax.adamw(learning_rate, weight_decay=NEIGHBOR_WEIGHT_DECAY)

    def loss(model: Encoder, rows: jax.Array, target: jax.Array, key) -> jax.Array:
        embedding, temperature = _embed_train(model, rows, key)
        if regression:
            return _regression_mse(embedding, temperature, target)
        return _class_nll(embedding, temperature, target)

    @eqx.filter_jit
    def step(model, opt_state, rows, target, key):
        value, grads = eqx.filter_value_and_grad(loss)(model, rows, target, key)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, value

    return optimizer, step


@eqx.filter_jit
def _encode_rows(model: Encoder, rows: jax.Array) -> jax.Array:
    return _embed_eval(model, rows)


def _snapshot(model: Encoder) -> Encoder:
    return jax.tree.map(lambda array: jnp.array(array), model)


def _average(models: list[Encoder]) -> Encoder:
    _leaves, treedef = jax.tree.flatten(models[0])
    collected = [jax.tree.leaves(model) for model in models]
    averaged = [sum(column) / len(column) for column in zip(*collected, strict=True)]
    return jax.tree.unflatten(treedef, averaged)


class LearnedNeighborhood:
    """Embed the row, then predict from a softmax over training neighbors."""

    def __init__(
        self,
        random_state: int = 0,
        verbose: bool = False,
        max_epochs: int = NEIGHBOR_MAX_EPOCHS,
        patience: int = NEIGHBOR_PATIENCE,
        batch_size: int = NEIGHBOR_BATCH_SIZE,
    ):
        if max_epochs < 1 or patience < 1 or batch_size < 2:
            raise AshlarError("max_epochs, patience, and batch_size must allow a neighbor pair.")
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)

    def fit(self, X, y, task: str):
        features = _matrix(X)
        labels = _labels(y)
        _check_task(task, features, labels)
        fit_idx, val_idx = _split_indices(labels, task, self.random_state)
        classes = None if task == "regression" else np.unique(labels)
        best_epoch, selection = self._select_epochs(features, labels, classes, fit_idx, val_idx, task)
        mean, std = _moments(features)
        scaled = _standardize(features, mean, std).astype(np.float32)
        target_mean, target_std, train_target = self._train_target(labels, classes, task, np.arange(len(features)))
        model = self._run(scaled, train_target, task == "regression", best_epoch, features.shape[1])
        self.task_ = task
        self.model_ = model
        self.mean_ = mean
        self.std_ = std
        self.reference_ = scaled
        self.reference_target_ = train_target
        self.target_mean_ = float(target_mean)
        self.target_std_ = float(target_std)
        self.classes_ = classes
        self.best_epoch_ = int(best_epoch)
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        return self

    def predict(self, X) -> np.ndarray:
        outputs = self._outputs(_matrix(X))
        if self.task_ == "regression":
            return outputs
        return self.classes_[np.argmax(outputs, axis=1)]

    def predict_score(self, X) -> np.ndarray:
        outputs = self._outputs(_matrix(X))
        if self.task_ == "regression":
            return outputs
        if self.classes_ is None:
            raise AshlarError("predict_score needs class values.")
        return outputs @ np.asarray(self.classes_, dtype=np.float64)

    def _select_epochs(self, features, labels, classes, fit_idx, val_idx, task: str) -> tuple[int, float]:
        mean, std = _moments(features[fit_idx])
        scaled = _standardize(features, mean, std).astype(np.float32)
        _mean, _std, target = self._train_target(labels, classes, task, fit_idx)
        regression = task == "regression"
        model = _init_encoder(jax.random.PRNGKey(self.random_state), features.shape[1])
        optimizer, step = _make_neighbor_step(NEIGHBOR_LEARNING_RATE, regression)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        history: list[Encoder] = []
        best_epoch = 1
        best_score = float("-inf")
        wait = 0
        for epoch in range(1, self.max_epochs + 1):
            model, opt_state = self._epoch(model, opt_state, step, scaled, target, fit_idx, epoch, regression)
            history.append(_snapshot(model))
            history = history[-NEIGHBOR_CHECKPOINT_AVERAGE:]
            averaged = _average(history)
            score = self._score(averaged, scaled, labels, classes, task, fit_idx, val_idx, _mean, _std)
            if self.verbose:
                print(f"  neighbor epoch {epoch} selection {score:.6f}", flush=True)
            if score > best_score:
                best_score = score
                best_epoch = epoch
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        return best_epoch, best_score

    def _run(self, scaled: np.ndarray, target: np.ndarray, regression: bool, epochs: int, n_features: int) -> Encoder:
        model = _init_encoder(jax.random.PRNGKey(self.random_state), n_features)
        optimizer, step = _make_neighbor_step(NEIGHBOR_LEARNING_RATE, regression)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        rows = np.arange(len(scaled))
        history: list[Encoder] = []
        for epoch in range(1, epochs + 1):
            model, opt_state = self._epoch(model, opt_state, step, scaled, target, rows, epoch, regression)
            history.append(_snapshot(model))
            history = history[-NEIGHBOR_CHECKPOINT_AVERAGE:]
        return _average(history)

    def _epoch(self, model, opt_state, step, scaled, target, rows, epoch: int, regression: bool):
        generator = np.random.default_rng(self.random_state + epoch)
        order = generator.permutation(len(rows))
        chosen = np.asarray(rows)[order]
        for start in range(0, len(chosen), self.batch_size):
            batch = chosen[start : start + self.batch_size]
            if len(batch) < 2:
                continue
            key = jax.random.PRNGKey(self.random_state + epoch * 100003 + start)
            batch_target = target[batch]
            if not regression:
                batch_target = np.asarray(batch_target, dtype=np.int32)
            model, opt_state, _value = step(
                model,
                opt_state,
                jnp.asarray(scaled[batch]),
                jnp.asarray(batch_target),
                key,
            )
        return model, opt_state

    def _score(self, model, scaled, labels, classes, task: str, fit_idx, val_idx, target_mean: float, target_std: float) -> float:
        outputs = self._from_reference(
            model,
            scaled[val_idx],
            scaled[fit_idx],
            self._reference_values(labels, classes, task, fit_idx, target_mean, target_std),
            task,
            None if classes is None else len(classes),
        )
        if task == "regression":
            return _r2(labels[val_idx], outputs * target_std + target_mean)
        if task == "classification":
            return _accuracy(labels[val_idx], classes[np.argmax(outputs, axis=1)])
        expected = outputs @ np.asarray(classes, dtype=np.float64)
        return _r2(labels[val_idx], expected)

    def _outputs(self, features: np.ndarray) -> np.ndarray:
        scaled = _standardize(features, self.mean_, self.std_).astype(np.float32)
        n_classes = None if self.classes_ is None else len(self.classes_)
        outputs = self._from_reference(self.model_, scaled, self.reference_, self.reference_target_, self.task_, n_classes)
        if self.task_ == "regression":
            return outputs * self.target_std_ + self.target_mean_
        return outputs

    def _from_reference(
        self,
        model,
        queries: np.ndarray,
        references: np.ndarray,
        reference_target: np.ndarray,
        task: str,
        n_classes: int | None,
    ) -> np.ndarray:
        query_emb = np.asarray(_encode_rows(model, jnp.asarray(queries)))
        reference_emb = np.asarray(_encode_rows(model, jnp.asarray(references)))
        temperature = float(np.asarray(_temperature(model)))
        if task == "regression":
            values = np.asarray(reference_target, dtype=np.float64).reshape(-1, 1)
        else:
            if n_classes is None:
                raise AshlarError("neighbor class aggregation needs the class count.")
            codes = np.asarray(reference_target, dtype=np.int32)
            values = np.eye(n_classes, dtype=np.float64)[codes]
        pieces = []
        for start in range(0, len(query_emb), 512):
            query = query_emb[start : start + 512]
            distance = np.sum(query * query, axis=1, keepdims=True) + np.sum(reference_emb * reference_emb, axis=1)
            distance = distance - 2.0 * (query @ reference_emb.T)
            logits = -np.maximum(distance, 0.0) / temperature
            logits -= logits.max(axis=1, keepdims=True)
            weights = np.exp(logits)
            weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)
            pieces.append(weights @ values)
        stacked = np.concatenate(pieces, axis=0)
        if task == "regression":
            return stacked[:, 0]
        return stacked

    @staticmethod
    def _reference_values(labels, classes, task: str, rows, target_mean: float, target_std: float) -> np.ndarray:
        chosen = labels[rows]
        if task == "regression":
            numeric = np.asarray(chosen, dtype=np.float64)
            return ((numeric - target_mean) / target_std).astype(np.float32)
        return _codes(classes, chosen)

    @staticmethod
    def _train_target(labels, classes, task: str, rows) -> tuple[float, float, np.ndarray]:
        if task == "regression":
            numeric = np.asarray(labels, dtype=np.float64)
            target_mean = float(numeric[rows].mean())
            target_std = float(numeric[rows].std())
            if target_std < 1e-6:
                target_std = 1.0
            target = ((numeric - target_mean) / target_std).astype(np.float32)
            return target_mean, target_std, target
        return 0.0, 1.0, _codes(classes, labels)
