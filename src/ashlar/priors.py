"""Differentiable classifiers with a prior or a data-dependent penalty.

Three bodies, none of them a hard tree booster. ``ASRBartPriorEnsemble`` is a sum
of soft depth-2 trees with a BART-shaped penalty on depth, feature choice,
gate decisiveness, and leaf size. ``ASRVariationalPairwise`` is a variational
logistic head on soft quantile bins and every pairwise bin interaction.
``ASRGradientSparseNet`` is a small network whose input gradient is L1 penalized,
which is the smooth version of a path that touches few columns. Two classes
keep one logit and sigmoid cross-entropy. More classes use one logit per class,
softmax cross-entropy, and the same penalty on the input Jacobian. Categorical
columns use a learned table: index 0 is unknown, and known levels occupy
indices 1..V in first-seen order. The network reads z-scored numerics
concatenated with those embedding vectors, and the penalty is the Jacobian
with respect to that vector. The residual regression default does not use
this module.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import Module

from ashlar._compat import renamed
from ashlar.errors import AshlarError
from ashlar.preprocess import _is_missing, _numeric_column, _stringify, frame_columns
from ashlar.rounds import _accuracy, _codes, _labels, _matrix, _moments, _split_indices, _standardize

BART_TREES = 40
BART_ALPHA = 0.95
BART_BETA = 2.0
BART_LEAF_SCALE = 3.0
BART_LEARNING_RATE = 0.05
BART_FEATURE_COEF = 1e-3
BART_GATE_COEF = 1e-3
BART_DEPTH_COEF = 0.1
BART_BATCH_SIZE = 256
BART_MAX_EPOCHS = 80
BART_PATIENCE = 16

VARIATIONAL_BINS = 4
VARIATIONAL_LEARNING_RATE = 1e-2
VARIATIONAL_BATCH_SIZE = 256
VARIATIONAL_MAX_EPOCHS = 80
VARIATIONAL_PATIENCE = 16
VARIATIONAL_TEMP_FLOOR = 0.05
VARIATIONAL_TEMP_INIT = 0.5
VARIATIONAL_BIAS_PRIOR = 4.0
VARIATIONAL_QUANTILES = (0.2, 0.4, 0.6, 0.8)

GRADIENT_WIDTH = 128
GRADIENT_LAMBDA = 1e-2
GRADIENT_LEARNING_RATE = 1e-3
GRADIENT_BATCH_SIZE = 256
GRADIENT_MAX_EPOCHS = 80
GRADIENT_PATIENCE = 16
GRADIENT_EMBED_DIM = 8
GRADIENT_EMBED_SCALE = 0.02

CHECKPOINT_AVERAGE = 5


class BartPrior(Module):
    """Parameters of a sum of soft depth-2 trees."""

    feature_logits: jax.Array
    thresholds: jax.Array
    raw_scales: jax.Array
    rhos: jax.Array
    leaves: jax.Array


class VariationalHead(Module):
    """Mean-field logistic weights and a shared bin temperature."""

    raw_temperature: jax.Array
    weight_mean: jax.Array
    weight_raw_scale: jax.Array
    bias_mean: jax.Array
    bias_raw_scale: jax.Array


class SparseNet(Module):
    """Two hidden layers and one logit per class. Two classes keep a single logit.

    ``embeddings`` is empty for a numeric matrix. With categorical columns it
    is one table per column, shaped ``(1 + n_levels, embed_dim)``. Index 0 is
    the unknown row.
    """

    w1: jax.Array
    b1: jax.Array
    w2: jax.Array
    b2: jax.Array
    w3: jax.Array
    b3: jax.Array
    embeddings: tuple[jax.Array, ...] = eqx.field(default_factory=tuple)


def _snapshot(model):
    return jax.tree.map(lambda array: jnp.array(array), model)


def _average(models):
    _leaves, treedef = jax.tree.flatten(models[0])
    collected = [jax.tree.leaves(model) for model in models]
    averaged = [sum(column) / len(column) for column in zip(*collected, strict=True)]
    return jax.tree.unflatten(treedef, averaged)


def _binary_targets(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    if len(classes) != 2:
        raise AshlarError(f"these priors expect two classes; received {len(classes)}.")
    return classes, _codes(classes, labels).astype(np.float32)


def _scale_raw(value: float) -> float:
    safe = max(float(value) - 1e-3, 1e-3)
    return float(np.log(np.expm1(safe)))


def _depth_targets() -> np.ndarray:
    root = BART_ALPHA / (1.0 + 0.0) ** BART_BETA
    child = BART_ALPHA / (1.0 + 1.0) ** BART_BETA
    return np.asarray([root, child, child], dtype=np.float32)


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-4), 1.0 - 1e-4)
    return float(np.log(clipped / (1.0 - clipped)))


def _init_bart(key, features: np.ndarray, n_trees: int) -> BartPrior:
    generator = np.random.default_rng(int(np.asarray(jax.random.bits(key))))
    n_rows, n_features = features.shape
    columns = generator.integers(0, n_features, size=(n_trees, 3))
    rows = generator.integers(0, n_rows, size=(n_trees, 3))
    thresholds = features[rows, columns].astype(np.float32)
    targets = _depth_targets()
    rhos = np.asarray([_logit(float(value)) for value in targets], dtype=np.float32)
    return BartPrior(
        feature_logits=jnp.zeros((n_trees, 3, n_features), dtype=jnp.float32),
        thresholds=jnp.asarray(thresholds),
        raw_scales=jnp.full((n_trees, 3), _scale_raw(1.0), dtype=jnp.float32),
        rhos=jnp.tile(jnp.asarray(rhos), (n_trees, 1)),
        leaves=jnp.zeros((n_trees, 7), dtype=jnp.float32),
    )


def _bart_paths(model: BartPrior, rows: jax.Array):
    selectors = jax.nn.softmax(model.feature_logits, axis=-1)
    chosen = jnp.einsum("bf,tnf->btn", rows, selectors)
    scale = jax.nn.softplus(model.raw_scales) + 1e-3
    gate = jax.nn.sigmoid((chosen - model.thresholds[None, :, :]) / scale[None, :, :])
    split = jax.nn.sigmoid(model.rhos)
    go_root = gate[:, :, 0]
    go_left = gate[:, :, 1]
    go_right = gate[:, :, 2]
    split_root = split[:, 0]
    split_left = split[:, 1]
    split_right = split[:, 2]
    stay_root = 1.0 - go_root
    root_stop = jnp.broadcast_to(1.0 - split_root, go_root.shape)
    paths = jnp.stack(
        [
            root_stop,
            split_root * stay_root * (1.0 - split_left),
            split_root * stay_root * split_left * (1.0 - go_left),
            split_root * stay_root * split_left * go_left,
            split_root * go_root * (1.0 - split_right),
            split_root * go_root * split_right * (1.0 - go_right),
            split_root * go_root * split_right * go_right,
        ],
        axis=-1,
    )
    return jnp.einsum("btk,tk->b", paths, model.leaves), selectors, gate, split


def _bernoulli_kl(probability: jax.Array, target: jax.Array) -> jax.Array:
    p = jnp.clip(probability, 1e-5, 1.0 - 1e-5)
    q = jnp.clip(target, 1e-5, 1.0 - 1e-5)
    return p * jnp.log(p / q) + (1.0 - p) * jnp.log((1.0 - p) / (1.0 - q))


def _bart_loss(model: BartPrior, rows: jax.Array, target: jax.Array, n_train: int) -> jax.Array:
    logit, selectors, gate, split = _bart_paths(model, rows)
    likelihood = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, target))
    targets = jnp.asarray(_depth_targets())
    depth = jnp.mean(_bernoulli_kl(split, targets[None, :]))
    feature = jnp.mean(jnp.sum(jnp.log(jnp.clip(selectors, 1e-6, 1.0)), axis=-1))
    safe_gate = jnp.clip(gate, 1e-4, 1.0 - 1e-4)
    decisive = jnp.mean(jnp.log(safe_gate * (1.0 - safe_gate)))
    sigma = BART_LEAF_SCALE / jnp.sqrt(model.leaves.shape[0])
    leaf = jnp.sum(jnp.square(model.leaves)) / (2.0 * n_train * jnp.square(sigma))
    return likelihood + BART_DEPTH_COEF * depth + BART_FEATURE_COEF * feature + BART_GATE_COEF * decisive + leaf


def _make_bart_step(learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def step(model, opt_state, rows, target, n_train: int):
        loss, grads = eqx.filter_value_and_grad(_bart_loss)(model, rows, target, n_train)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return optimizer, step


def _pair_index(n_features: int) -> tuple[np.ndarray, np.ndarray]:
    left = []
    right = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            left.append(i)
            right.append(j)
    return np.asarray(left, dtype=np.int32), np.asarray(right, dtype=np.int32)


def _quantile_centers(rows: np.ndarray) -> np.ndarray:
    return np.quantile(rows, VARIATIONAL_QUANTILES, axis=0).T.astype(np.float32)


def _design(rows: jax.Array, centers: jax.Array, temperature: jax.Array, pair_left: jax.Array, pair_right: jax.Array) -> jax.Array:
    distance = jnp.abs(rows[:, :, None] - centers[None, :, :])
    bins = jax.nn.softmax(-distance / temperature, axis=-1)
    mains = bins.reshape(bins.shape[0], -1)
    if pair_left.shape[0] == 0:
        return mains
    left = bins[:, pair_left, :]
    right = bins[:, pair_right, :]
    pairs = (left[:, :, :, None] * right[:, :, None, :]).reshape(bins.shape[0], -1)
    return jnp.concatenate([mains, pairs], axis=-1)


def _init_variational(n_columns: int) -> VariationalHead:
    raw_temp = np.log(np.expm1(np.float32(VARIATIONAL_TEMP_INIT - VARIATIONAL_TEMP_FLOOR)))
    raw_scale = np.log(np.expm1(np.float32(0.1)))
    return VariationalHead(
        raw_temperature=jnp.asarray(raw_temp, dtype=jnp.float32),
        weight_mean=jnp.zeros((n_columns,), dtype=jnp.float32),
        weight_raw_scale=jnp.full((n_columns,), raw_scale, dtype=jnp.float32),
        bias_mean=jnp.zeros((), dtype=jnp.float32),
        bias_raw_scale=jnp.asarray(raw_scale, dtype=jnp.float32),
    )


def _temperature(model: VariationalHead) -> jax.Array:
    return jax.nn.softplus(model.raw_temperature) + VARIATIONAL_TEMP_FLOOR


def _variational_logit(model: VariationalHead, design: jax.Array, noise: jax.Array | None) -> jax.Array:
    weight_scale = jax.nn.softplus(model.weight_raw_scale) + 1e-4
    bias_scale = jax.nn.softplus(model.bias_raw_scale) + 1e-4
    mean = design @ model.weight_mean + model.bias_mean
    if noise is None:
        return mean
    variance = (jnp.square(design) @ jnp.square(weight_scale)) + jnp.square(bias_scale)
    return mean + jnp.sqrt(variance + 1e-8) * noise


def _variational_loss(model, rows, target, centers, pair_left, pair_right, prior_var, noise, n_train: int):
    design = _design(rows, centers, _temperature(model), pair_left, pair_right)
    logit = _variational_logit(model, design, noise)
    likelihood = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, target))
    weight_scale = jax.nn.softplus(model.weight_raw_scale) + 1e-4
    bias_scale = jax.nn.softplus(model.bias_raw_scale) + 1e-4
    scale2 = jnp.square(weight_scale)
    kl = 0.5 * jnp.sum(scale2 / prior_var + jnp.square(model.weight_mean) / prior_var - 1.0 - jnp.log(scale2) + jnp.log(prior_var))
    bias_var = jnp.square(bias_scale)
    kl = kl + 0.5 * (bias_var / VARIATIONAL_BIAS_PRIOR + jnp.square(model.bias_mean) / VARIATIONAL_BIAS_PRIOR - 1.0 - jnp.log(bias_var) + jnp.log(VARIATIONAL_BIAS_PRIOR))
    return likelihood + kl / n_train


def _make_variational_step(learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def step(model, opt_state, rows, target, centers, pair_left, pair_right, prior_var, noise, n_train: int):
        loss, grads = eqx.filter_value_and_grad(_variational_loss)(
            model, rows, target, centers, pair_left, pair_right, prior_var, noise, n_train
        )
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return optimizer, step


@eqx.filter_jit
def _variational_moments(model, rows, centers, pair_left, pair_right):
    design = _design(rows, centers, _temperature(model), pair_left, pair_right)
    return jnp.mean(jnp.square(design), axis=0)


@eqx.filter_jit
def _variational_mean_logit(model, rows, centers, pair_left, pair_right):
    design = _design(rows, centers, _temperature(model), pair_left, pair_right)
    return _variational_logit(model, design, None)


def _init_sparse(key, n_features: int) -> SparseNet:
    k1, k2, k3 = jax.random.split(key, 3)

    def weight(subkey, fan_in: int, fan_out: int, scale: float) -> jax.Array:
        return jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32) * np.float32(scale)

    return SparseNet(
        w1=weight(k1, n_features, GRADIENT_WIDTH, float(np.sqrt(2.0 / n_features))),
        b1=jnp.zeros((GRADIENT_WIDTH,), dtype=jnp.float32),
        w2=weight(k2, GRADIENT_WIDTH, GRADIENT_WIDTH, float(np.sqrt(2.0 / GRADIENT_WIDTH))),
        b2=jnp.zeros((GRADIENT_WIDTH,), dtype=jnp.float32),
        w3=weight(k3, GRADIENT_WIDTH, 1, float(np.sqrt(1.0 / GRADIENT_WIDTH))),
        b3=jnp.zeros((1,), dtype=jnp.float32),
    )


def _sparse_logits(model: SparseNet, rows: jax.Array) -> jax.Array:
    hidden = jax.nn.silu(rows @ model.w1 + model.b1)
    hidden = jax.nn.silu(hidden @ model.w2 + model.b2)
    return jnp.squeeze(hidden @ model.w3 + model.b3, axis=-1)


def _sparse_loss(model: SparseNet, rows: jax.Array, target: jax.Array) -> jax.Array:
    logit = _sparse_logits(model, rows)
    likelihood = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, target))

    def one_logit(row):
        return _sparse_logits(model, row[None, :])[0]

    gradient = jax.vmap(jax.grad(one_logit))(rows)
    return likelihood + GRADIENT_LAMBDA * jnp.mean(jnp.abs(gradient))


def _make_sparse_step(learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def step(model, opt_state, rows, target):
        loss, grads = eqx.filter_value_and_grad(_sparse_loss)(model, rows, target)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return optimizer, step


def _init_sparse_multi(key, n_features: int, n_classes: int) -> SparseNet:
    k1, k2, k3 = jax.random.split(key, 3)

    def weight(subkey, fan_in: int, fan_out: int, scale: float) -> jax.Array:
        return jax.random.normal(subkey, (fan_in, fan_out), dtype=jnp.float32) * np.float32(scale)

    return SparseNet(
        w1=weight(k1, n_features, GRADIENT_WIDTH, float(np.sqrt(2.0 / n_features))),
        b1=jnp.zeros((GRADIENT_WIDTH,), dtype=jnp.float32),
        w2=weight(k2, GRADIENT_WIDTH, GRADIENT_WIDTH, float(np.sqrt(2.0 / GRADIENT_WIDTH))),
        b2=jnp.zeros((GRADIENT_WIDTH,), dtype=jnp.float32),
        w3=weight(k3, GRADIENT_WIDTH, n_classes, float(np.sqrt(1.0 / GRADIENT_WIDTH))),
        b3=jnp.zeros((n_classes,), dtype=jnp.float32),
    )


def _multi_logits(model: SparseNet, rows: jax.Array) -> jax.Array:
    hidden = jax.nn.silu(rows @ model.w1 + model.b1)
    hidden = jax.nn.silu(hidden @ model.w2 + model.b2)
    return hidden @ model.w3 + model.b3


def _multi_loss(model: SparseNet, rows: jax.Array, target: jax.Array) -> jax.Array:
    logits = _multi_logits(model, rows)
    likelihood = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target.astype(jnp.int32)))

    def one_logits(row):
        return _multi_logits(model, row[None, :])[0]

    jacobian = jax.vmap(jax.jacobian(one_logits))(rows)
    return likelihood + GRADIENT_LAMBDA * jnp.mean(jnp.abs(jacobian))


def _make_multi_step(learning_rate: float):
    optimizer = optax.adam(learning_rate)

    @eqx.filter_jit
    def step(model, opt_state, rows, target):
        loss, grads = eqx.filter_value_and_grad(_multi_loss)(model, rows, target)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    return optimizer, step


@eqx.filter_jit
def _multi_apply(model: SparseNet, rows: jax.Array) -> jax.Array:
    return _multi_logits(model, rows)


def _embedded_width(n_numeric: int, vocabs: tuple[tuple[str, ...], ...]) -> int:
    return int(n_numeric) + len(vocabs) * GRADIENT_EMBED_DIM


def _init_embedded(key, n_features: int, vocabs: tuple[tuple[str, ...], ...], n_classes: int, multi: bool) -> SparseNet:
    if n_features < 1 or not vocabs:
        raise AshlarError("embedded gradient penalty needs a numeric column or a categorical column.")
    mlp_key, emb_key = jax.random.split(key)
    if multi:
        model = _init_sparse_multi(mlp_key, n_features, n_classes)
    else:
        model = _init_sparse(mlp_key, n_features)
    pieces = jax.random.split(emb_key, len(vocabs))
    scale = jnp.asarray(GRADIENT_EMBED_SCALE, dtype=jnp.float32)
    tables = []
    for piece, vocab in zip(pieces, vocabs, strict=True):
        table = jax.random.normal(piece, (1 + len(vocab), GRADIENT_EMBED_DIM), dtype=jnp.float32) * scale
        tables.append(table)
    return SparseNet(
        w1=model.w1,
        b1=model.b1,
        w2=model.w2,
        b2=model.b2,
        w3=model.w3,
        b3=model.b3,
        embeddings=tuple(tables),
    )


def _embedded_features(model: SparseNet, numeric: jax.Array, indices: jax.Array) -> jax.Array:
    pieces = []
    for column, table in enumerate(model.embeddings):
        pieces.append(table[indices[:, column]])
    embedded = jnp.concatenate(pieces, axis=-1)
    if numeric.shape[-1] == 0:
        return embedded
    return jnp.concatenate([numeric.astype(embedded.dtype), embedded], axis=-1)


def _embedded_loss(model: SparseNet, numeric: jax.Array, indices: jax.Array, target: jax.Array) -> jax.Array:
    features = _embedded_features(model, numeric, indices)
    logit = _sparse_logits(model, features)
    likelihood = jnp.mean(optax.sigmoid_binary_cross_entropy(logit, target))

    def one_logit(row):
        return _sparse_logits(model, row[None, :])[0]

    gradient = jax.vmap(jax.grad(one_logit))(features)
    return likelihood + GRADIENT_LAMBDA * jnp.mean(jnp.abs(gradient))


def _embedded_multi_loss(model: SparseNet, numeric: jax.Array, indices: jax.Array, target: jax.Array) -> jax.Array:
    features = _embedded_features(model, numeric, indices)
    logits = _multi_logits(model, features)
    likelihood = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, target.astype(jnp.int32)))

    def one_logits(row):
        return _multi_logits(model, row[None, :])[0]

    jacobian = jax.vmap(jax.jacobian(one_logits))(features)
    return likelihood + GRADIENT_LAMBDA * jnp.mean(jnp.abs(jacobian))


def _make_embedded_step(learning_rate: float, multi: bool):
    optimizer = optax.adam(learning_rate)
    loss = _embedded_multi_loss if multi else _embedded_loss

    @eqx.filter_jit
    def step(model, opt_state, numeric, indices, target):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, numeric, indices, target)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    return optimizer, step


@eqx.filter_jit
def _embedded_apply(model: SparseNet, numeric: jax.Array, indices: jax.Array) -> jax.Array:
    return _sparse_logits(model, _embedded_features(model, numeric, indices))


@eqx.filter_jit
def _embedded_apply_multi(model: SparseNet, numeric: jax.Array, indices: jax.Array) -> jax.Array:
    return _multi_logits(model, _embedded_features(model, numeric, indices))


def _missing_category(value) -> bool:
    if _is_missing(value):
        return True
    return type(value).__name__ in {"NAType", "NaTType"}


def _column_role(values: np.ndarray) -> str:
    if np.issubdtype(values.dtype, np.bool_):
        return "numeric"
    if _numeric_column(values):
        return "numeric"
    return "categorical"


def _gradient_columns(X):
    """Return names, columns, and roles when X has a real categorical column."""
    if isinstance(X, np.ndarray) and np.issubdtype(X.dtype, np.number):
        return None
    try:
        names, columns = frame_columns(X)
    except AshlarError:
        return None
    if not columns:
        return None
    roles = tuple(_column_role(values) for values in columns)
    if "categorical" not in roles:
        return None
    return names, columns, roles


def _numeric_matrix(columns, roles, names) -> np.ndarray:
    n_rows = len(columns[0]) if columns else 0
    pieces = []
    for values, role, name in zip(columns, roles, names, strict=True):
        if role != "numeric":
            continue
        if np.issubdtype(values.dtype, np.bool_):
            number = np.asarray(values, dtype=np.float64)
        else:
            try:
                number = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise AshlarError(f"numeric column {name!r} is not numeric.") from error
        if number.shape != (n_rows,):
            number = np.reshape(number, (n_rows,))
        if not np.isfinite(number).all():
            raise AshlarError(f"numeric column {name!r} must be finite.")
        pieces.append(number)
    if not pieces:
        return np.zeros((n_rows, 0), dtype=np.float64)
    return np.ascontiguousarray(np.column_stack(pieces))


def _category_vocabs(columns, names, roles, row_idx) -> tuple[tuple[str, ...], ...]:
    order = np.sort(np.asarray(row_idx, dtype=int))
    vocabs = []
    for values, role, name in zip(columns, roles, names, strict=True):
        if role != "categorical":
            continue
        seen: list[str] = []
        seen_set: set[str] = set()
        for value in values[order].tolist():
            if _missing_category(value):
                continue
            text = _stringify(value)
            if text not in seen_set:
                seen_set.add(text)
                seen.append(text)
        if not seen:
            raise AshlarError(f"categorical column {name!r} has no observed categories in the training partition.")
        vocabs.append(tuple(seen))
    if not vocabs:
        raise AshlarError("gradient penalty embeddings need a categorical column.")
    return tuple(vocabs)


def _category_indices(columns, roles, vocabs) -> np.ndarray:
    n_rows = len(columns[0]) if columns else 0
    lookups = [{category: index + 1 for index, category in enumerate(vocab)} for vocab in vocabs]
    mapped = []
    cursor = 0
    for values, role in zip(columns, roles, strict=True):
        if role != "categorical":
            continue
        if cursor >= len(vocabs):
            raise AshlarError("categorical columns do not match the fitted vocabulary.")
        lookup = lookups[cursor]
        cursor += 1
        indices = np.zeros(n_rows, dtype=np.int32)
        for row, value in enumerate(values.tolist()):
            if _missing_category(value):
                continue
            indices[row] = lookup.get(_stringify(value), 0)
        mapped.append(indices)
    if cursor != len(vocabs):
        raise AshlarError("categorical columns do not match the fitted vocabulary.")
    return np.ascontiguousarray(np.column_stack(mapped).astype(np.int32))


def _batches_embedded(numeric, indices, target, batch_size: int, epoch: int, random_state: int):
    generator = np.random.default_rng(random_state + epoch)
    order = generator.permutation(len(target))
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        if len(chosen) < 2:
            continue
        yield numeric[chosen], indices[chosen], target[chosen]


@eqx.filter_jit
def _bart_logit(model: BartPrior, rows: jax.Array) -> jax.Array:
    logit, _selectors, _gate, _split = _bart_paths(model, rows)
    return logit


def _train_until(init, step_epoch, score, max_epochs: int, patience: int, verbose: bool, name: str):
    state = init()
    history = []
    best_epoch = 1
    best_score = float("-inf")
    wait = 0
    for epoch in range(1, max_epochs + 1):
        state = step_epoch(state, epoch)
        history.append(_snapshot(state))
        history = history[-CHECKPOINT_AVERAGE:]
        value = float(score(_average(history)))
        if verbose:
            print(f"  {name} epoch {epoch} selection {value:.6f}", flush=True)
        if value > best_score:
            best_score = value
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best_epoch, best_score


def _train_for(init, step_epoch, epochs: int):
    state = init()
    history = []
    for epoch in range(1, epochs + 1):
        state = step_epoch(state, epoch)
        history.append(_snapshot(state))
        history = history[-CHECKPOINT_AVERAGE:]
    return _average(history)


def _batches(rows: np.ndarray, target: np.ndarray, batch_size: int, epoch: int, random_state: int):
    generator = np.random.default_rng(random_state + epoch)
    order = generator.permutation(len(rows))
    for start in range(0, len(order), batch_size):
        chosen = order[start : start + batch_size]
        if len(chosen) < 2:
            continue
        yield rows[chosen], target[chosen], start


class ASRBartPriorEnsemble:
    """Soft depth-2 trees with a BART-shaped penalty. Summed, not grown greedily."""

    def __init__(
        self,
        random_state: int = 0,
        verbose: bool = False,
        max_epochs: int = BART_MAX_EPOCHS,
        patience: int = BART_PATIENCE,
        batch_size: int = BART_BATCH_SIZE,
        n_trees: int = BART_TREES,
    ):
        if max_epochs < 1 or patience < 1 or batch_size < 2 or n_trees < 1:
            raise AshlarError("epochs, patience, batch size, and tree count must be positive.")
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.n_trees = int(n_trees)

    def fit(self, X, y, task: str = "classification"):
        features, target, classes, fit_idx, val_idx = _prepare(X, y, task, self.random_state)
        best_epoch, selection = self._select(features, target, fit_idx, val_idx)
        self.best_epoch_ = int(best_epoch)
        self._opt_state = None
        mean, std = _moments(features)
        scaled = _standardize(features, mean, std).astype(np.float32)
        model = self._run(scaled, target, len(features))
        self._store(model, mean, std, classes, best_epoch, selection, len(fit_idx), len(val_idx), scaled)
        return self

    def predict(self, X) -> np.ndarray:
        logit = self._logit(_matrix(X))
        bit = (logit > 0.0).astype(np.int32)
        return self.classes_[bit]

    def _select(self, features, target, fit_idx, val_idx):
        mean, std = _moments(features[fit_idx])
        scaled = _standardize(features, mean, std).astype(np.float32)
        self._opt_state = None

        def init():
            return _init_bart(jax.random.PRNGKey(self.random_state), scaled[fit_idx], self.n_trees)

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, target, fit_idx, epoch, len(fit_idx))

        def score(state):
            return _accuracy(self.classes_from_target(target[val_idx]), self._bits(state, scaled[val_idx]))

        self._bind_classes(target)
        return _train_until(init, step_epoch, score, self.max_epochs, self.patience, self.verbose, "bart-prior")

    def _run(self, scaled, target, n_rows: int):
        def init():
            return _init_bart(jax.random.PRNGKey(self.random_state), scaled, self.n_trees)

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, target, np.arange(n_rows), epoch, n_rows)

        return _train_for(init, step_epoch, self.best_epoch_)

    def _epoch(self, model, scaled, target, rows, epoch: int, n_train: int):
        if self._opt_state is None:
            optimizer, step = _make_bart_step(BART_LEARNING_RATE)
            self._opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            self._bart_step = step
        for batch_rows, batch_target, _start in _batches(scaled[rows], target[rows], self.batch_size, epoch, self.random_state):
            model, self._opt_state, _loss = self._bart_step(
                model,
                self._opt_state,
                jnp.asarray(batch_rows),
                jnp.asarray(batch_target),
                int(n_train),
            )
        return model

    def _bits(self, model, rows: np.ndarray) -> np.ndarray:
        pieces = []
        for start in range(0, len(rows), 1024):
            logit = np.asarray(_bart_logit(model, jnp.asarray(rows[start : start + 1024])))
            pieces.append(logit > 0.0)
        return np.concatenate(pieces)

    def _logit(self, features: np.ndarray) -> np.ndarray:
        scaled = _standardize(features, self.mean_, self.std_).astype(np.float32)
        pieces = []
        for start in range(0, len(scaled), 1024):
            pieces.append(np.asarray(_bart_logit(self.model_, jnp.asarray(scaled[start : start + 1024]))))
        return np.concatenate(pieces)

    def _bind_classes(self, target: np.ndarray) -> None:
        # Target is 0/1 in class-sorted order. Accuracy during selection compares bits to that coding.
        self._positive = target

    def classes_from_target(self, target: np.ndarray) -> np.ndarray:
        return np.asarray(target).reshape(-1)

    def _store(self, model, mean, std, classes, best_epoch, selection, n_fit, n_val, scaled):
        self.task_ = "classification"
        self.model_ = model
        self.mean_ = mean
        self.std_ = std
        self.classes_ = classes
        self.best_epoch_ = int(best_epoch)
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(n_fit)
        self.n_val_ = int(n_val)
        self.n_trees_ = int(self.n_trees)
        _logit, selectors, gate, split = _bart_paths(model, jnp.asarray(scaled[: min(len(scaled), 1024)]))
        self.feature_peak_ = float(np.mean(np.max(np.asarray(selectors), axis=-1)))
        safe = np.asarray(gate)
        self.gate_margin_ = float(np.mean(np.minimum(safe, 1.0 - safe)))
        split_rate = np.asarray(jax.nn.sigmoid(model.rhos))
        self.root_split_ = float(np.mean(split_rate[:, 0]))
        self.child_split_ = float(np.mean(split_rate[:, 1:]))


class ASRVariationalPairwise:
    """Variational logistic head on soft bins and every pairwise bin product."""

    def __init__(
        self,
        random_state: int = 0,
        verbose: bool = False,
        max_epochs: int = VARIATIONAL_MAX_EPOCHS,
        patience: int = VARIATIONAL_PATIENCE,
        batch_size: int = VARIATIONAL_BATCH_SIZE,
    ):
        if max_epochs < 1 or patience < 1 or batch_size < 2:
            raise AshlarError("epochs, patience, and batch size must be positive.")
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)

    def fit(self, X, y, task: str = "classification"):
        features, target, classes, fit_idx, val_idx = _prepare(X, y, task, self.random_state)
        best_epoch, selection = self._select(features, target, fit_idx, val_idx)
        self.best_epoch_ = int(best_epoch)
        mean, std = _moments(features)
        scaled = _standardize(features, mean, std).astype(np.float32)
        centers = _quantile_centers(scaled)
        model = self._run(scaled, target, centers)
        self.task_ = "classification"
        self.model_ = model
        self.mean_ = mean
        self.std_ = std
        self.centers_ = centers
        self.classes_ = classes
        self.best_epoch_ = int(best_epoch)
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        return self

    def predict(self, X) -> np.ndarray:
        logit = self._logit(_matrix(X))
        return self.classes_[(logit > 0.0).astype(np.int32)]

    def _select(self, features, target, fit_idx, val_idx):
        mean, std = _moments(features[fit_idx])
        scaled = _standardize(features, mean, std).astype(np.float32)
        centers = _quantile_centers(scaled[fit_idx])
        self._reset_optimizer()

        def init():
            return _init_variational(self._width(features.shape[1]))

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, target, fit_idx, centers, epoch, len(fit_idx))

        def score(state):
            logit = self._logit_rows(state, scaled[val_idx], centers)
            return _accuracy(target[val_idx], (logit > 0.0).astype(np.float32))

        return _train_until(init, step_epoch, score, self.max_epochs, self.patience, self.verbose, "variational")

    def _run(self, scaled, target, centers):
        self._reset_optimizer()

        def init():
            return _init_variational(self._width(scaled.shape[1]))

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, target, np.arange(len(scaled)), centers, epoch, len(scaled))

        return _train_for(init, step_epoch, self.best_epoch_)

    def _epoch(self, model, scaled, target, rows, centers, epoch: int, n_train: int):
        pair_left, pair_right = self._pairs(scaled.shape[1])
        centers_j = jnp.asarray(centers)
        moment = np.asarray(_variational_moments(model, jnp.asarray(scaled[rows]), centers_j, pair_left, pair_right))
        prior_var = (1.0 / np.clip(moment, 1e-4, None)).astype(np.float32)
        if self._opt_state is None:
            optimizer, step = _make_variational_step(VARIATIONAL_LEARNING_RATE)
            self._opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            self._var_step = step
        for batch_rows, batch_target, start in _batches(scaled[rows], target[rows], self.batch_size, epoch, self.random_state):
            noise = jax.random.normal(jax.random.PRNGKey(self.random_state + epoch * 100003 + start), (len(batch_rows),), dtype=jnp.float32)
            model, self._opt_state, _loss = self._var_step(
                model,
                self._opt_state,
                jnp.asarray(batch_rows),
                jnp.asarray(batch_target),
                centers_j,
                pair_left,
                pair_right,
                jnp.asarray(prior_var),
                noise,
                int(n_train),
            )
        return model

    def _logit(self, features: np.ndarray) -> np.ndarray:
        scaled = _standardize(features, self.mean_, self.std_).astype(np.float32)
        return self._logit_rows(self.model_, scaled, self.centers_)

    def _logit_rows(self, model, rows: np.ndarray, centers: np.ndarray) -> np.ndarray:
        pair_left, pair_right = self._pairs(rows.shape[1])
        pieces = []
        centers_j = jnp.asarray(centers)
        for start in range(0, len(rows), 1024):
            logit = np.asarray(
                _variational_mean_logit(model, jnp.asarray(rows[start : start + 1024]), centers_j, pair_left, pair_right)
            )
            pieces.append(logit)
        return np.concatenate(pieces)

    def _width(self, n_features: int) -> int:
        n_pairs = n_features * (n_features - 1) // 2
        return n_features * VARIATIONAL_BINS + n_pairs * VARIATIONAL_BINS * VARIATIONAL_BINS

    def _pairs(self, n_features: int):
        key = n_features
        if getattr(self, "_pair_key", None) != key:
            left, right = _pair_index(n_features)
            self._pair_left = jnp.asarray(left)
            self._pair_right = jnp.asarray(right)
            self._pair_key = key
        return self._pair_left, self._pair_right

    def _reset_optimizer(self) -> None:
        self._opt_state = None
        self._var_step = None


class ASRGradientSparseNet:
    """Network classifier with an L1 penalty on the input gradient.

    Two classes keep the original one-logit sigmoid loss. More classes use a
    softmax over one logit per class. The penalty is the mean absolute input
    Jacobian of those logits, with the same coefficient. A numeric matrix stays
    on that path. Categorical columns add a learned table of width 8, initial
    scale 0.02. Index 0 is the unknown row for missing values and unseen
    levels. Known levels are 1..V in first-seen order on that phase's fit
    rows. The penalty then applies to the concatenation of z-scored numerics
    and the embedding vectors.
    """

    def __init__(
        self,
        random_state: int = 0,
        verbose: bool = False,
        max_epochs: int = GRADIENT_MAX_EPOCHS,
        patience: int = GRADIENT_PATIENCE,
        batch_size: int = GRADIENT_BATCH_SIZE,
    ):
        if max_epochs < 1 or patience < 1 or batch_size < 2:
            raise AshlarError("epochs, patience, and batch size must be positive.")
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)

    def fit(self, X, y, task: str = "classification"):
        if task != "classification":
            raise AshlarError("these priors are classification models.")
        parsed = _gradient_columns(X)
        if parsed is not None:
            return self._fit_categorical(parsed, y)
        self.categorical_ = False
        n_classes = len(np.unique(_labels(y)))
        if n_classes < 2:
            raise AshlarError(f"gradient penalty expects at least two classes; received {n_classes}.")
        self._multi = n_classes > 2
        if self._multi:
            features, target, classes, fit_idx, val_idx = _prepare_multiclass(X, y, task, self.random_state)
        else:
            features, target, classes, fit_idx, val_idx = _prepare(X, y, task, self.random_state)
        self.n_classes_ = int(len(classes))
        best_epoch, selection = self._select(features, target, fit_idx, val_idx)
        self.best_epoch_ = int(best_epoch)
        mean, std = _moments(features)
        scaled = _standardize(features, mean, std).astype(np.float32)
        model = self._run(scaled, target)
        self.task_ = "classification"
        self.model_ = model
        self.mean_ = mean
        self.std_ = std
        self.classes_ = classes
        self.best_epoch_ = int(best_epoch)
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        return self

    def predict(self, X) -> np.ndarray:
        if getattr(self, "categorical_", False):
            values = self._categorical_logits(X)
        else:
            values = self._logit(_matrix(X))
        if values.ndim == 1:
            index = (values > 0.0).astype(np.int32)
        else:
            index = np.argmax(values, axis=1).astype(np.int32)
        return self.classes_[index]

    def _fit_categorical(self, parsed, y):
        names, columns, roles = parsed
        n_rows = len(columns[0])
        for values in columns:
            if len(values) != n_rows:
                raise AshlarError("feature columns have different lengths.")
        labels = _labels(y)
        if len(labels) != n_rows:
            raise AshlarError("X and y have different lengths.")
        n_classes = len(np.unique(labels))
        if n_classes < 2:
            raise AshlarError(f"gradient penalty expects at least two classes; received {n_classes}.")
        self.categorical_ = True
        self._multi = n_classes > 2
        if self._multi:
            classes = np.unique(labels)
            target = _codes(classes, labels).astype(np.int32)
        else:
            classes, target = _binary_targets(labels)
        fit_idx, val_idx = _split_indices(labels, "classification", self.random_state)
        self.n_classes_ = int(len(classes))
        self.feature_names_ = tuple(names)
        self.column_roles_ = roles
        numeric = _numeric_matrix(columns, roles, names)
        best_epoch, selection = self._select_categorical(numeric, columns, names, roles, target, fit_idx, val_idx)
        self.best_epoch_ = int(best_epoch)
        vocabs = _category_vocabs(columns, names, roles, np.arange(n_rows))
        indices = _category_indices(columns, roles, vocabs)
        mean, std = _moments(numeric)
        scaled = _standardize(numeric, mean, std).astype(np.float32)
        self.categories_ = vocabs
        model = self._run_categorical(scaled, indices, target, vocabs)
        self.task_ = "classification"
        self.model_ = model
        self.mean_ = mean
        self.std_ = std
        self.classes_ = classes
        self.best_epoch_ = int(best_epoch)
        self.selection_metric_ = float(selection)
        self.n_fit_ = int(len(fit_idx))
        self.n_val_ = int(len(val_idx))
        self.embed_dim_ = int(GRADIENT_EMBED_DIM)
        return self

    def _select_categorical(self, numeric, columns, names, roles, target, fit_idx, val_idx):
        vocabs = _category_vocabs(columns, names, roles, fit_idx)
        indices = _category_indices(columns, roles, vocabs)
        mean, std = _moments(numeric[fit_idx])
        scaled = _standardize(numeric, mean, std).astype(np.float32)
        self._reset_optimizer()

        def init():
            return _init_embedded(
                jax.random.PRNGKey(self.random_state),
                _embedded_width(scaled.shape[1], vocabs),
                vocabs,
                self.n_classes_,
                self._multi,
            )

        def step_epoch(state, epoch):
            return self._epoch_embedded(state, scaled, indices, target, fit_idx, epoch)

        def score(state):
            return _accuracy(target[val_idx], self._predict_embedded(state, scaled[val_idx], indices[val_idx]))

        return _train_until(init, step_epoch, score, self.max_epochs, self.patience, self.verbose, "gradient")

    def _run_categorical(self, scaled, indices, target, vocabs):
        self._reset_optimizer()

        def init():
            return _init_embedded(
                jax.random.PRNGKey(self.random_state),
                _embedded_width(scaled.shape[1], vocabs),
                vocabs,
                self.n_classes_,
                self._multi,
            )

        def step_epoch(state, epoch):
            return self._epoch_embedded(state, scaled, indices, target, np.arange(len(target)), epoch)

        return _train_for(init, step_epoch, self.best_epoch_)

    def _epoch_embedded(self, model, scaled, indices, target, rows, epoch: int):
        if self._opt_state is None:
            optimizer, step = _make_embedded_step(GRADIENT_LEARNING_RATE, self._multi)
            self._opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            self._sparse_step = step
        chosen = np.asarray(rows)
        for batch_numeric, batch_index, batch_target in _batches_embedded(
            scaled[chosen], indices[chosen], target[chosen], self.batch_size, epoch, self.random_state
        ):
            model, self._opt_state, _loss = self._sparse_step(
                model,
                self._opt_state,
                jnp.asarray(batch_numeric),
                jnp.asarray(batch_index),
                jnp.asarray(batch_target),
            )
        return model

    def _categorical_logits(self, X) -> np.ndarray:
        names, columns = frame_columns(X)
        if tuple(names) != self.feature_names_:
            raise AshlarError(
                "feature columns do not match the fitted schema. "
                f"Expected {self.feature_names_}; received {tuple(names)}."
            )
        n_rows = len(columns[0]) if columns else 0
        for values in columns:
            if len(values) != n_rows:
                raise AshlarError("feature columns have different lengths.")
        numeric = _numeric_matrix(columns, self.column_roles_, names)
        indices = _category_indices(columns, self.column_roles_, self.categories_)
        scaled = _standardize(numeric, self.mean_, self.std_).astype(np.float32)
        return self._embedded_rows(self.model_, scaled, indices)

    def _embedded_rows(self, model, numeric, indices) -> np.ndarray:
        apply = _embedded_apply_multi if self._multi else _embedded_apply
        pieces = []
        for start in range(0, len(indices), 1024):
            stop = start + 1024
            pieces.append(np.asarray(apply(model, jnp.asarray(numeric[start:stop]), jnp.asarray(indices[start:stop]))))
        return np.concatenate(pieces)

    def _predict_embedded(self, model, numeric, indices) -> np.ndarray:
        values = self._embedded_rows(model, numeric, indices)
        if values.ndim == 1:
            return (values > 0.0).astype(np.float32)
        return np.argmax(values, axis=1).astype(np.int32)

    def _select(self, features, target, fit_idx, val_idx):
        mean, std = _moments(features[fit_idx])
        scaled = _standardize(features, mean, std).astype(np.float32)
        self._reset_optimizer()

        def init():
            return self._initial(features.shape[1])

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, target, fit_idx, epoch)

        def score(state):
            return _accuracy(target[val_idx], self._predict_codes(state, scaled[val_idx]))

        return _train_until(init, step_epoch, score, self.max_epochs, self.patience, self.verbose, "gradient")

    def _run(self, scaled, target):
        self._reset_optimizer()

        def init():
            return self._initial(scaled.shape[1])

        def step_epoch(state, epoch):
            return self._epoch(state, scaled, target, np.arange(len(scaled)), epoch)

        return _train_for(init, step_epoch, self.best_epoch_)

    def _epoch(self, model, scaled, target, rows, epoch: int):
        if self._opt_state is None:
            maker = _make_multi_step if self._multi else _make_sparse_step
            optimizer, step = maker(GRADIENT_LEARNING_RATE)
            self._opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
            self._sparse_step = step
        for batch_rows, batch_target, _start in _batches(scaled[rows], target[rows], self.batch_size, epoch, self.random_state):
            model, self._opt_state, _loss = self._sparse_step(model, self._opt_state, jnp.asarray(batch_rows), jnp.asarray(batch_target))
        return model

    def _logit(self, features: np.ndarray) -> np.ndarray:
        scaled = _standardize(features, self.mean_, self.std_).astype(np.float32)
        return self._logit_rows(self.model_, scaled)

    def _initial(self, n_features: int) -> SparseNet:
        key = jax.random.PRNGKey(self.random_state)
        if self._multi:
            return _init_sparse_multi(key, n_features, self.n_classes_)
        return _init_sparse(key, n_features)

    def _predict_codes(self, model, rows: np.ndarray) -> np.ndarray:
        values = self._logit_rows(model, rows)
        if values.ndim == 1:
            return (values > 0.0).astype(np.float32)
        return np.argmax(values, axis=1).astype(np.int32)

    def _logit_rows(self, model, rows: np.ndarray) -> np.ndarray:
        apply = _multi_apply if self._multi else _sparse_apply
        pieces = []
        for start in range(0, len(rows), 1024):
            pieces.append(np.asarray(apply(model, jnp.asarray(rows[start : start + 1024]))))
        return np.concatenate(pieces)

    def _reset_optimizer(self) -> None:
        self._opt_state = None
        self._sparse_step = None


@eqx.filter_jit
def _sparse_apply(model: SparseNet, rows: jax.Array) -> jax.Array:
    return _sparse_logits(model, rows)


def _prepare(X, y, task: str, random_state: int):
    if task != "classification":
        raise AshlarError("these priors are classification models.")
    features = _matrix(X)
    labels = _labels(y)
    if len(labels) != len(features):
        raise AshlarError("X and y have different lengths.")
    classes, coded = _binary_targets(labels)
    fit_idx, val_idx = _split_indices(labels, "classification", random_state)
    return features, coded, classes, fit_idx, val_idx


def _prepare_multiclass(X, y, task: str, random_state: int):
    if task != "classification":
        raise AshlarError("these priors are classification models.")
    features = _matrix(X)
    labels = _labels(y)
    if len(labels) != len(features):
        raise AshlarError("X and y have different lengths.")
    classes = np.unique(labels)
    if len(classes) < 3:
        raise AshlarError(f"multiclass gradient penalty expects at least three classes; received {len(classes)}.")
    fit_idx, val_idx = _split_indices(labels, "classification", random_state)
    return features, _codes(classes, labels).astype(np.int32), classes, fit_idx, val_idx


__getattr__ = renamed(
    __name__,
    globals(),
    {
        "BartPriorEnsemble": "ASRBartPriorEnsemble",
        "VariationalPairwise": "ASRVariationalPairwise",
        "GradientSparseNet": "ASRGradientSparseNet",
    },
)
