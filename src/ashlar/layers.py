"""TabM batch-ensemble layers, a residual MLP, the field backbones, and the additive backbone.

The TabM path follows the equations in ``tabm`` 0.0.3 (commit
``28e47ae301c92ec37787dde1ce923a0793f405b4``):

* ``EnsembleView`` expands ``(batch, features)`` to ``(batch, member, features)``
  with identical copies. A tensor that is already ``(batch, member, features)``
  passes through only when the member count matches.
* ``LinearBatchEnsemble`` is equation (5) of BatchEnsemble:
  ``((x * r) @ W.T) * s + bias``.
* TabM initialization: the first ``r`` is random (signs or normal) and may be
  constant inside feature chunks; every later ``r`` and every ``s`` is ones.
  The ``k`` biases of a batch-ensemble layer share one initialization draw.
* Each backbone block is linear, ReLU, then dropout.
* The output ``LinearEnsemble`` is a head, not part of the backbone, so a
  residual MLP and TabM can use the same head.

Deviations from a full paper run, recorded on purpose: no packed or mini
architecture, no piecewise-linear or periodic embeddings, and no claim that a
forward pass reproduces a published benchmark score.
"""

from __future__ import annotations

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module, field, nn

from ashlar.errors import ShapeError

_ACTIVATIONS = {"relu": jax.nn.relu, "gelu": jax.nn.gelu, "silu": jax.nn.silu}

# Width used when an estimator sets ``bottleneck=True``. The residual default
# leaves the bottleneck off; this constant is the narrow size, not a switch.
DEFAULT_BOTTLENECK_WIDTH = 32

# Fixed per-stage multiplier for ``SoftStageHistogram``. Matches the shrinkage
# of the default XGBoost classifier on the wine split. It is not a learned rate.
SOFT_STAGE_SHRINKAGE = 0.3


def _rsqrt_uniform(key, shape, d: int, dtype):
    limit = d**-0.5
    return jax.random.uniform(key, shape, dtype, minval=-limit, maxval=limit)


def _scaling(key, shape, distribution: str, chunks: list[int] | None, dtype):
    if distribution == "ones":
        if chunks is not None:
            raise ShapeError("scaling chunks are only used with a random scaling initialization.")
        return jnp.ones(shape, dtype)
    if chunks is None:
        if distribution == "random-signs":
            return jax.random.bernoulli(key, 0.5, shape).astype(dtype) * 2 - 1
        if distribution == "normal":
            return jax.random.normal(key, shape, dtype)
        raise ShapeError("scaling initialization must be 'ones', 'random-signs', or 'normal'.")
    if sum(chunks) != shape[-1]:
        raise ShapeError(f"scaling chunks sum to {sum(chunks)} but the feature axis has length {shape[-1]}.")
    pieces = []
    keys = jax.random.split(key, len(chunks))
    for piece_key, width in zip(keys, chunks, strict=True):
        if distribution == "random-signs":
            draw = jax.random.bernoulli(piece_key, 0.5, shape[:-1] + (1,)).astype(dtype) * 2 - 1
        elif distribution == "normal":
            draw = jax.random.normal(piece_key, shape[:-1] + (1,), dtype)
        else:
            raise ShapeError("chunked scaling requires 'random-signs' or 'normal'.")
        pieces.append(jnp.broadcast_to(draw, shape[:-1] + (width,)))
    return jnp.concatenate(pieces, axis=-1)


def ensemble_view(x, k: int):
    """Expand ``(batch, features)`` to ``k`` identical member views."""
    if x.ndim == 2:
        return jnp.broadcast_to(x[:, None, :], (x.shape[0], k, x.shape[-1]))
    if x.ndim == 3:
        if x.shape[1] != k:
            raise ShapeError(
                f"ensemble view expected member axis {k}; received shape {x.shape} "
                "(batch, member, features)."
            )
        return x
    raise ShapeError(f"ensemble view expected 2 or 3 dimensions; received shape {x.shape}.")


class LinearBatchEnsemble(Module):
    """Shared linear map with per-member rank-one scalings. Input ``(batch, member, in)``."""

    weight: jax.Array
    r: jax.Array
    s: jax.Array
    bias: jax.Array

    def __call__(self, x):
        if x.ndim != 3 or x.shape[1] != self.r.shape[0] or x.shape[-1] != self.weight.shape[1]:
            raise ShapeError(
                "LinearBatchEnsemble expected (batch, member, in_features) with "
                f"member={self.r.shape[0]} and in_features={self.weight.shape[1]}; received {x.shape}."
            )
        y = x * self.r
        y = y @ self.weight.T
        y = y * self.s
        return y + self.bias


class LinearEnsemble(Module):
    """Independent linear map per member. Weight shape ``(member, in, out)``."""

    weight: jax.Array
    bias: jax.Array

    def __call__(self, x):
        if x.ndim != 3 or x.shape[1] != self.weight.shape[0] or x.shape[-1] != self.weight.shape[1]:
            raise ShapeError(
                "LinearEnsemble expected (batch, member, in_features) with "
                f"member={self.weight.shape[0]} and in_features={self.weight.shape[1]}; received {x.shape}."
            )
        return jnp.einsum("bki,kio->bko", x, self.weight) + self.bias


class LateBottleneck(Module):
    """Per-member linear map and ReLU, narrower than the backbone.

    Sits after the backbone blocks and before the head. Each member keeps its
    own map. There is no dropout on this projection.
    """

    projection: LinearEnsemble
    width: int = field(static=True)

    def __init__(self, d_in: int, k: int, width: int, *, key):
        if width < 1 or width >= d_in:
            raise ShapeError(
                f"bottleneck width {width} must be positive and narrower than the backbone width {d_in}."
            )
        weight_key, bias_key = jax.random.split(key)
        self.projection = LinearEnsemble(
            weight=_rsqrt_uniform(weight_key, (k, d_in, width), d_in, jnp.float32),
            bias=_rsqrt_uniform(bias_key, (k, width), d_in, jnp.float32),
        )
        self.width = int(width)

    def __call__(self, hidden):
        return jax.nn.relu(self.projection(hidden))


class SoftStageHistogram(Module):
    """Additive soft stumps. Each side of a stump is a class-logit histogram.

    A stage mixes the numeric columns with a softmax, passes that coordinate
    through a sigmoid threshold, and mixes the two leaf vectors. The stages
    are summed with a fixed shrinkage and trained jointly from the class loss.
    Leaf logits start at zero, so the sum starts at zero. Splits stay soft.
    """

    feature_logits: jax.Array
    threshold: jax.Array
    log_temperature: jax.Array
    left_logits: jax.Array
    right_logits: jax.Array
    shrinkage: float = field(static=True)
    n_stages: int = field(static=True)
    n_features: int = field(static=True)
    n_classes: int = field(static=True)

    def __init__(self, n_features: int, n_classes: int, n_stages: int, *, shrinkage: float, key):
        if n_features < 1 or n_classes < 2 or n_stages < 1:
            raise ShapeError("a soft stage histogram needs features, at least 2 classes, and at least 1 stage.")
        if not math.isfinite(float(shrinkage)) or float(shrinkage) <= 0.0:
            raise ShapeError("soft-stage shrinkage must be positive and finite.")
        feature_key, threshold_key = jax.random.split(key)
        self.feature_logits = jax.random.normal(feature_key, (n_stages, n_features), dtype=jnp.float32) * jnp.asarray(
            0.1, dtype=jnp.float32
        )
        self.threshold = jax.random.normal(threshold_key, (n_stages,), dtype=jnp.float32) * jnp.asarray(
            0.5, dtype=jnp.float32
        )
        self.log_temperature = jnp.zeros((n_stages,), dtype=jnp.float32)
        self.left_logits = jnp.zeros((n_stages, n_classes), dtype=jnp.float32)
        self.right_logits = jnp.zeros((n_stages, n_classes), dtype=jnp.float32)
        self.shrinkage = float(shrinkage)
        self.n_stages = int(n_stages)
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)

    def __call__(self, numeric_z):
        if numeric_z.ndim != 2 or numeric_z.shape[-1] != self.n_features:
            raise ShapeError(
                f"soft stage histogram expected (batch, {self.n_features}); received {numeric_z.shape}."
            )
        weights = jax.nn.softmax(self.feature_logits, axis=-1)
        coordinate = numeric_z @ weights.T
        temperature = jnp.exp(self.log_temperature)
        gate = jax.nn.sigmoid((coordinate - self.threshold) / temperature)
        mixed = (1.0 - gate[..., None]) * self.left_logits + gate[..., None] * self.right_logits
        return jnp.asarray(self.shrinkage, dtype=mixed.dtype) * jnp.sum(mixed, axis=1)


class BatchedLinear(Module):
    """Dense map on ``(batch, features)`` or ``(batch, member, features)``."""

    weight: jax.Array
    bias: jax.Array

    def __call__(self, x):
        if x.shape[-1] != self.weight.shape[1]:
            raise ShapeError(
                f"BatchedLinear expected a feature axis of length {self.weight.shape[1]}; received {x.shape}."
            )
        return x @ self.weight.T + self.bias


def _dropout(drop: nn.Dropout | None, x, key):
    if drop is None:
        return x
    if key is None:
        return drop(x, inference=True)
    return drop(x, key=key, inference=False)


class TabMBackbone(Module):
    """Batch-ensemble MLP trunk. Output shape ``(batch, member, d_block)``."""

    layers: tuple[LinearBatchEnsemble, ...]
    dropout: nn.Dropout | None
    k: int = field(static=True)
    d_out: int = field(static=True)
    dropout_rate: float = field(static=True)
    share_training_batches: bool = field(static=True)

    def __init__(
        self,
        d_in: int,
        *,
        key,
        n_blocks: int = 3,
        d_block: int = 512,
        dropout: float = 0.1,
        k: int = 32,
        start_scaling_init: str = "random-signs",
        scaling_chunks: list[int] | None = None,
        share_training_batches: bool = True,
    ):
        if n_blocks < 1 or d_block < 1 or k < 1:
            raise ShapeError("TabM n_blocks, d_block, and k must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ShapeError("dropout must be in [0, 1).")
        keys = jax.random.split(key, n_blocks * 4)
        layers = []
        for index in range(n_blocks):
            in_features = d_in if index == 0 else d_block
            base = keys[index]
            weight_key, r_key, bias_key = jax.random.split(base, 3)
            if index == 0:
                r_dist = start_scaling_init
                r_chunks = scaling_chunks
            else:
                r_dist = "ones"
                r_chunks = None
            bias = _rsqrt_uniform(bias_key, (d_block,), in_features, jnp.float32)
            layers.append(
                LinearBatchEnsemble(
                    weight=_rsqrt_uniform(weight_key, (d_block, in_features), in_features, jnp.float32),
                    r=_scaling(r_key, (k, in_features), r_dist, r_chunks, jnp.float32),
                    s=jnp.ones((k, d_block), dtype=jnp.float32),
                    bias=jnp.repeat(bias[None, :], k, axis=0),
                )
            )
        self.layers = tuple(layers)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.k = k
        self.d_out = d_block
        self.dropout_rate = dropout
        self.share_training_batches = bool(share_training_batches)

    def __call__(self, x, *, key):
        if x.ndim == 2 and not self.share_training_batches and key is not None:
            if x.shape[0] % self.k != 0:
                raise ShapeError(
                    f"unshared TabM batches must contain a multiple of k={self.k} rows; received {x.shape[0]}."
                )
            x = x.reshape((x.shape[0] // self.k, self.k, x.shape[-1]))
        else:
            x = ensemble_view(x, self.k)
        keys = None if key is None or self.dropout is None else jax.random.split(key, len(self.layers))
        for index, layer in enumerate(self.layers):
            x = jax.nn.relu(layer(x))
            drop_key = None if keys is None else keys[index]
            x = _dropout(self.dropout, x, drop_key)
        return x


class ResidualMLP(Module):
    """Residual MLP with an explicit member axis of length 1.

    The member axis is present so a head written for TabM does not broadcast a
    missing axis. This is not a TabM model and it does not share TabM's init.
    The default activation is SiLU. ``relu`` and ``gelu`` are optional.
    """

    input_layer: BatchedLinear
    blocks: tuple[tuple[BatchedLinear, BatchedLinear], ...]
    dropout: nn.Dropout | None
    k: int = field(static=True)
    d_out: int = field(static=True)
    dropout_rate: float = field(static=True)
    activation: str = field(static=True)

    def __init__(
        self,
        d_in: int,
        *,
        key,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.1,
        activation: str = "silu",
        scaling_chunks: list[int] | None = None,
    ):
        del scaling_chunks
        if n_blocks < 1 or d_block < 1:
            raise ShapeError("ResidualMLP n_blocks and d_block must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ShapeError("dropout must be in [0, 1).")
        if activation not in _ACTIVATIONS:
            raise ShapeError(f"activation must be one of {tuple(_ACTIVATIONS)}; received {activation!r}.")
        keys = jax.random.split(key, 1 + n_blocks * 2)
        self.input_layer = BatchedLinear(
            weight=_rsqrt_uniform(keys[0], (d_block, d_in), d_in, jnp.float32),
            bias=jnp.zeros((d_block,), dtype=jnp.float32),
        )
        blocks = []
        for index in range(n_blocks):
            k1, k2 = keys[1 + 2 * index], keys[2 + 2 * index]
            blocks.append(
                (
                    BatchedLinear(
                        weight=_rsqrt_uniform(k1, (d_block, d_block), d_block, jnp.float32),
                        bias=jnp.zeros((d_block,), dtype=jnp.float32),
                    ),
                    BatchedLinear(
                        weight=_rsqrt_uniform(k2, (d_block, d_block), d_block, jnp.float32),
                        bias=jnp.zeros((d_block,), dtype=jnp.float32),
                    ),
                )
            )
        self.blocks = tuple(blocks)
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.k = 1
        self.d_out = d_block
        self.dropout_rate = float(dropout)
        self.activation = activation

    def __call__(self, x, *, key):
        if x.ndim != 2:
            raise ShapeError(f"ResidualMLP expected (batch, features); received {x.shape}.")
        activate = _ACTIVATIONS[self.activation]
        hidden = activate(self.input_layer(x))
        n_drops = len(self.blocks)
        keys = None if key is None or self.dropout is None else jax.random.split(key, n_drops)
        for index, (first, second) in enumerate(self.blocks):
            residual = activate(first(hidden))
            residual = _dropout(self.dropout, residual, None if keys is None else keys[index])
            hidden = hidden + second(residual)
        return hidden[:, None, :]


class LinearReLUEmbeddings(Module):
    """Per-feature linear map and ReLU, matching ``rtdl_num_embeddings.LinearReLUEmbeddings``.

    Input ``(batch, n_features)``. Output ``(batch, n_features * d_embedding)``.
    The unflattened embedding is ``bias + weight * x[..., None]``, then ReLU.
    Initialization is uniform on ``[-d_embedding**-0.5, d_embedding**-0.5]``.
    Piecewise-linear and periodic embeddings are not implemented.
    """

    weight: jax.Array
    bias: jax.Array
    d_embedding: int = field(static=True)

    def __init__(self, n_features: int, d_embedding: int, *, key):
        if n_features < 1 or d_embedding < 1:
            raise ShapeError("LinearReLUEmbeddings requires positive n_features and d_embedding.")
        weight_key, bias_key = jax.random.split(key)
        limit = d_embedding**-0.5
        self.weight = jax.random.uniform(weight_key, (n_features, d_embedding), minval=-limit, maxval=limit)
        self.bias = jax.random.uniform(bias_key, (n_features, d_embedding), minval=-limit, maxval=limit)
        self.d_embedding = d_embedding

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.weight.shape[0]:
            raise ShapeError(
                f"LinearReLUEmbeddings expected (batch, {self.weight.shape[0]}); received {x.shape}."
            )
        hidden = self.bias + self.weight * x[..., None]
        return jax.nn.relu(hidden).reshape(x.shape[0], -1)


class PeriodicEmbeddings(Module):
    """Periodic numeric embeddings, the PLR form of Gorishniy et al. (2022).

    Each feature ``x`` becomes ``cos(2π c x)`` and ``sin(2π c x)`` at
    ``n_frequencies`` learned frequencies ``c``, drawn at init from a normal
    with standard deviation ``frequency_scale``. A per-feature linear map to
    ``d_embedding`` and a ReLU follow. Output ``(batch, n_features * d_embedding)``.
    """

    frequencies: jax.Array
    weight: jax.Array
    bias: jax.Array
    d_embedding: int = field(static=True)

    def __init__(
        self,
        n_features: int,
        d_embedding: int,
        *,
        key,
        n_frequencies: int = 16,
        frequency_scale: float = 0.1,
    ):
        if n_features < 1 or d_embedding < 1 or n_frequencies < 1:
            raise ShapeError("PeriodicEmbeddings requires positive n_features, d_embedding, and n_frequencies.")
        freq_key, weight_key, bias_key = jax.random.split(key, 3)
        width = 2 * n_frequencies
        self.frequencies = frequency_scale * jax.random.normal(freq_key, (n_features, n_frequencies))
        self.weight = _rsqrt_uniform(weight_key, (n_features, width, d_embedding), width, jnp.float32)
        self.bias = _rsqrt_uniform(bias_key, (n_features, d_embedding), width, jnp.float32)
        self.d_embedding = d_embedding

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.frequencies.shape[0]:
            raise ShapeError(f"PeriodicEmbeddings expected (batch, {self.frequencies.shape[0]}); received {x.shape}.")
        angle = 2 * math.pi * self.frequencies * x[..., None]
        periodic = jnp.concatenate([jnp.cos(angle), jnp.sin(angle)], axis=-1)
        hidden = jnp.einsum("bfp,fpd->bfd", periodic, self.weight) + self.bias
        return jax.nn.relu(hidden).reshape(x.shape[0], -1)


def quantile_edges(numeric_z, n_bins: int, max_rows: int = 200_000, seed: int = 0):
    """Bin edges for ``PiecewiseLinearEmbeddings``, shape ``(n_features, n_bins + 1)``.

    Edges are training-row quantiles of each scaled feature. A feature with
    fewer distinct quantiles than ``n_bins + 1`` repeats its top edge; a
    repeated edge gives a bin that is always zero. Up to ``max_rows`` rows,
    drawn without replacement, set the quantiles.
    """
    import numpy as np

    values = np.asarray(numeric_z, dtype=np.float64)
    if values.shape[0] > max_rows:
        rows = np.random.default_rng(seed).choice(values.shape[0], size=max_rows, replace=False)
        values = values[rows]
    levels = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.empty((values.shape[1], n_bins + 1), dtype=np.float32)
    for column in range(values.shape[1]):
        unique = np.unique(np.quantile(values[:, column], levels))
        if unique.size < 2:
            unique = np.array([unique[0] - 0.5, unique[0] + 0.5])
        edges[column, : unique.size] = unique
        edges[column, unique.size :] = unique[-1]
    return jnp.asarray(edges)


class PiecewiseLinearEmbeddings(Module):
    """Piecewise-linear numeric embeddings, version B of ``rtdl_num_embeddings``.

    ``edges`` has shape ``(n_features, n_bins + 1)`` and is fit on training
    quantiles. It is frozen: the forward pass stops its gradient and the
    optimizer does not decay it. A value's encoding is its fill fraction of
    each bin, ``clip((x - left) / width, 0, 1)``, with the first bin open
    below and the last open above. The embedding is a linear map of ``x``
    plus a per-feature linear map of that encoding, which starts at zero, so
    training begins from the plain linear embedding. ``activation=True`` adds
    a ReLU. Output ``(batch, n_features * d_embedding)``.
    """

    edges: jax.Array
    linear_weight: jax.Array
    linear_bias: jax.Array
    weight: jax.Array
    bias: jax.Array
    d_embedding: int = field(static=True)
    activation: bool = field(static=True)

    def __init__(self, edges, d_embedding: int, *, key, activation: bool = False):
        edges = jnp.asarray(edges, dtype=jnp.float32)
        if edges.ndim != 2 or edges.shape[1] < 2 or d_embedding < 1:
            raise ShapeError("PiecewiseLinearEmbeddings needs edges (n_features, n_bins + 1) and a positive d_embedding.")
        n_features, n_bins = edges.shape[0], edges.shape[1] - 1
        weight_key, bias_key = jax.random.split(key)
        self.edges = edges
        self.linear_weight = _rsqrt_uniform(weight_key, (n_features, d_embedding), 1, jnp.float32)
        self.linear_bias = _rsqrt_uniform(bias_key, (n_features, d_embedding), 1, jnp.float32)
        self.weight = jnp.zeros((n_features, n_bins, d_embedding), dtype=jnp.float32)
        self.bias = jnp.zeros((n_features, d_embedding), dtype=jnp.float32)
        self.d_embedding = d_embedding
        self.activation = bool(activation)

    def encode(self, x):
        edges = jax.lax.stop_gradient(self.edges)
        left = edges[:, :-1]
        width = edges[:, 1:] - left
        empty = width <= 0
        fraction = (x[..., None] - left) / jnp.where(empty, 1.0, width)
        n_bins = left.shape[1]
        lower = jnp.where(jnp.arange(n_bins) == 0, -jnp.inf, 0.0)
        upper = jnp.where(jnp.arange(n_bins) == n_bins - 1, jnp.inf, 1.0)
        fraction = jnp.clip(fraction, lower, upper)
        return jnp.where(empty, 0.0, fraction)

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.edges.shape[0]:
            raise ShapeError(
                f"PiecewiseLinearEmbeddings expected (batch, {self.edges.shape[0]}); received {x.shape}."
            )
        linear = self.linear_bias + self.linear_weight * x[..., None]
        piecewise = jnp.einsum("bfn,fnd->bfd", self.encode(x), self.weight) + self.bias
        hidden = linear + piecewise
        if self.activation:
            hidden = jax.nn.relu(hidden)
        return hidden.reshape(x.shape[0], -1)


# Leaves the optimizer must not move: fitted bin edges.
FROZEN_FIELDS = frozenset({"edges"})


class CategoricalEmbeddings(Module):
    """One learned table per named categorical column, including the unknown row.

    ``indices`` has shape ``(batch, n_columns)``. Output is
    ``(batch, n_columns * d_embedding)`` and is part of the fitted core.
    """

    tables: tuple[jax.Array, ...]
    hidden: tuple[tuple[jax.Array, jax.Array], ...] | None = None

    def __call__(self, indices):
        if indices.ndim != 2 or indices.shape[1] != len(self.tables):
            raise ShapeError(
                f"category indices must have shape (batch, {len(self.tables)}); received {indices.shape}."
            )
        pieces = []
        for column, table in enumerate(self.tables):
            vector = table[indices[:, column]]
            if self.hidden is not None:
                # A per-column hidden layer: the category vector passes through
                # its own dense map and ReLU, so the column's contribution is not
                # a fixed linear read of the one-hot. The residual keeps the lookup.
                weight, bias = self.hidden[column]
                vector = vector + jax.nn.relu(vector @ weight + bias)
            pieces.append(vector)
        return jnp.concatenate(pieces, axis=-1)


class SetEmbeddings(Module):
    """Masked mean of learned token embeddings. One table per set column.

    ``tokens`` and ``mask`` have shape ``(batch, n_columns, length)``.
    An empty set (mask all zero) becomes a zero vector, not the unknown token.
    """

    tables: tuple[jax.Array, ...]

    def __call__(self, tokens, mask):
        if tokens.ndim != 3 or mask.shape != tokens.shape or tokens.shape[1] != len(self.tables):
            raise ShapeError(
                "set tokens and mask must have shape "
                f"(batch, {len(self.tables)}, length); received {tokens.shape} and {mask.shape}."
            )
        pieces = []
        for column, table in enumerate(self.tables):
            gathered = table[tokens[:, column, :]]
            weight = mask[:, column, :, None]
            summed = jnp.sum(gathered * weight, axis=1)
            count = jnp.maximum(jnp.sum(mask[:, column], axis=1, keepdims=True), jnp.asarray(1.0, dtype=mask.dtype))
            pieces.append(summed / count)
        return jnp.concatenate(pieces, axis=-1)


class _FieldProjection(Module):
    """Project each field chunk to ``d_model``. Chunks keep column identity."""

    weights: tuple[jax.Array, ...]
    biases: tuple[jax.Array, ...]

    def __call__(self, x):
        if x.ndim != 2:
            raise ShapeError(f"field projection expected (batch, features); received {x.shape}.")
        pieces = []
        cursor = 0
        for weight, bias in zip(self.weights, self.biases, strict=True):
            width = weight.shape[1]
            chunk = x[:, cursor : cursor + width]
            cursor += width
            pieces.append(chunk @ weight.T + bias)
        if cursor != x.shape[1]:
            raise ShapeError(
                f"feature chunks cover {cursor} columns; the backbone received {x.shape[1]}."
            )
        return jnp.stack(pieces, axis=1)


def _field_projection(key, chunks: list[int], d_model: int) -> _FieldProjection:
    keys = jax.random.split(key, len(chunks))
    weights = []
    biases = []
    for piece, width in zip(keys, chunks, strict=True):
        if width < 1:
            raise ShapeError("feature chunks must be positive widths.")
        weight_key, bias_key = jax.random.split(piece)
        weights.append(_rsqrt_uniform(weight_key, (d_model, width), width, jnp.float32))
        biases.append(jnp.zeros((d_model,), dtype=jnp.float32))
    return _FieldProjection(weights=tuple(weights), biases=tuple(biases))


def _layer_norm(x, scale, bias):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    normed = (x - mean) * jax.lax.rsqrt(variance + jnp.asarray(1e-5, dtype=x.dtype))
    return normed * scale + bias


class _AttentionBlock(Module):
    attention: nn.MultiheadAttention
    norm1_scale: jax.Array
    norm1_bias: jax.Array
    mlp_up: jax.Array
    mlp_up_bias: jax.Array
    mlp_down: jax.Array
    mlp_down_bias: jax.Array
    norm2_scale: jax.Array
    norm2_bias: jax.Array
    dropout_rate: float = field(static=True)

    def __call__(self, tokens, *, key):
        def one(row):
            normed = _layer_norm(row, self.norm1_scale, self.norm1_bias)
            return self.attention(normed, normed, normed, key=None)

        attended = eqx.filter_vmap(one)(tokens)
        drop_key = None if key is None else jax.random.split(key, 2)
        attended = _apply_dropout(attended, self.dropout_rate, None if drop_key is None else drop_key[0])
        hidden = tokens + attended
        normed = _layer_norm(hidden, self.norm2_scale, self.norm2_bias)
        mid = jax.nn.relu(normed @ self.mlp_up.T + self.mlp_up_bias)
        mid = _apply_dropout(mid, self.dropout_rate, None if drop_key is None else drop_key[1])
        return hidden + mid @ self.mlp_down.T + self.mlp_down_bias


def _apply_dropout(x, rate: float, key):
    if rate == 0.0 or key is None:
        return x
    keep = jax.random.bernoulli(key, 1.0 - rate, x.shape)
    return jnp.where(keep, x / (1.0 - rate), 0.0)


class FeatureTransformer(Module):
    """Self-attention over field tokens, then a mean pool.

    Each fitted column is one token. There is no positional encoding: column
    identity is the per-field projection. The member axis is explicit and has
    length 1, same as the residual MLP. Use this for a feature set, including
    a set-valued categorical that has already been pooled into one field.
    """

    projection: _FieldProjection
    blocks: tuple[_AttentionBlock, ...]
    k: int = field(static=True)
    d_out: int = field(static=True)
    dropout_rate: float = field(static=True)

    def __init__(
        self,
        d_in: int,
        *,
        key,
        n_blocks: int = 1,
        d_block: int = 32,
        n_heads: int = 4,
        dropout: float = 0.0,
        feature_chunks: list[int] | None = None,
    ):
        if n_blocks < 1 or d_block < 1 or n_heads < 1:
            raise ShapeError("transformer n_blocks, d_block, and n_heads must be positive.")
        if d_block % n_heads != 0:
            raise ShapeError(f"d_block ({d_block}) must be divisible by n_heads ({n_heads}).")
        if not 0.0 <= dropout < 1.0:
            raise ShapeError("dropout must be in [0, 1).")
        chunks = [1] * d_in if feature_chunks is None else list(feature_chunks)
        if sum(chunks) != d_in:
            raise ShapeError(f"feature chunks sum to {sum(chunks)}; the encoded width is {d_in}.")
        keys = jax.random.split(key, 1 + n_blocks * 4)
        self.projection = _field_projection(keys[0], chunks, d_block)
        blocks = []
        for index in range(n_blocks):
            base = index * 4
            attn_key, up_key, down_key, _rest = jax.random.split(keys[1 + index], 4)
            blocks.append(
                _AttentionBlock(
                    attention=nn.MultiheadAttention(
                        n_heads, d_block, dropout_p=0.0, inference=True, key=attn_key
                    ),
                    norm1_scale=jnp.ones((d_block,), dtype=jnp.float32),
                    norm1_bias=jnp.zeros((d_block,), dtype=jnp.float32),
                    mlp_up=_rsqrt_uniform(up_key, (d_block * 2, d_block), d_block, jnp.float32),
                    mlp_up_bias=jnp.zeros((d_block * 2,), dtype=jnp.float32),
                    mlp_down=_rsqrt_uniform(down_key, (d_block, d_block * 2), d_block * 2, jnp.float32),
                    mlp_down_bias=jnp.zeros((d_block,), dtype=jnp.float32),
                    norm2_scale=jnp.ones((d_block,), dtype=jnp.float32),
                    norm2_bias=jnp.zeros((d_block,), dtype=jnp.float32),
                    dropout_rate=float(dropout),
                )
            )
            del base, _rest
        self.blocks = tuple(blocks)
        self.k = 1
        self.d_out = d_block
        self.dropout_rate = float(dropout)

    def __call__(self, x, *, key):
        tokens = self.projection(x)
        keys = None if key is None or self.dropout_rate == 0.0 else jax.random.split(key, len(self.blocks))
        for index, block in enumerate(self.blocks):
            tokens = block(tokens, key=None if keys is None else keys[index])
        return jnp.mean(tokens, axis=1)[:, None, :]


class SetEncoder(Module):
    """DeepSets-style pool over the same field tokens the transformer attends over.

    Per-field projections keep column identity. The mean over fields does not
    use order. A variable-length category list is already one field by the time
    it reaches this module: ``SetEmbeddings`` mean-pools that list.
    """

    projection: _FieldProjection
    up: jax.Array
    up_bias: jax.Array
    down: jax.Array
    down_bias: jax.Array
    k: int = field(static=True)
    d_out: int = field(static=True)
    dropout_rate: float = field(static=True)

    def __init__(
        self,
        d_in: int,
        *,
        key,
        n_blocks: int = 1,
        d_block: int = 32,
        dropout: float = 0.0,
        feature_chunks: list[int] | None = None,
    ):
        if n_blocks != 1:
            raise ShapeError("the set encoder pools field tokens once. n_blocks must be 1.")
        if d_block < 1:
            raise ShapeError("set encoder d_block must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ShapeError("dropout must be in [0, 1).")
        chunks = [1] * d_in if feature_chunks is None else list(feature_chunks)
        if sum(chunks) != d_in:
            raise ShapeError(f"feature chunks sum to {sum(chunks)}; the encoded width is {d_in}.")
        proj_key, up_key, down_key = jax.random.split(key, 3)
        self.projection = _field_projection(proj_key, chunks, d_block)
        self.up = _rsqrt_uniform(up_key, (d_block, d_block), d_block, jnp.float32)
        self.up_bias = jnp.zeros((d_block,), dtype=jnp.float32)
        self.down = _rsqrt_uniform(down_key, (d_block, d_block), d_block, jnp.float32)
        self.down_bias = jnp.zeros((d_block,), dtype=jnp.float32)
        self.k = 1
        self.d_out = d_block
        self.dropout_rate = float(dropout)

    def __call__(self, x, *, key):
        del key
        tokens = self.projection(x)
        pooled = jnp.mean(tokens, axis=1)
        hidden = jax.nn.relu(pooled @ self.up.T + self.up_bias)
        return (pooled + hidden @ self.down.T + self.down_bias)[:, None, :]


@functools.lru_cache(maxsize=64)
def _term_gather(chunks: tuple[int, ...], terms: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """``(terms, width)``: the input column behind each slot of each term, ``sum(chunks)`` for an empty slot."""
    starts = np.concatenate([[0], np.cumsum(chunks)[:-1]]).astype(int)
    d_in = int(sum(chunks))
    columns = [
        np.concatenate([np.arange(starts[index], starts[index] + chunks[index]) for index in term]) for term in terms
    ]
    gather = np.full((len(terms), max(len(column) for column in columns)), d_in, dtype=np.int32)
    for row, column in enumerate(columns):
        gather[row, : len(column)] = column
    return gather


class AdditiveBackbone(Module):
    """One small network per term, side by side. Output ``(batch, member, terms * d_term)``.

    A term reads its own input fields, one column or the two of a pair, and
    nothing else, and the terms' outputs are concatenated. A head that is
    linear in them, as every built-in distribution head is, therefore gives
    each of its outputs (a location, a raw scale, a zero logit) a sum of one
    function per term: a generalized additive model for location, scale, and
    shape, each function a network. ``feature_chunks`` are the widths of the
    input fields in the order the core concatenates them, and ``terms`` lists
    the fields each term reads, one term per field by default; a field no term
    reads is ignored. Each term is ``n_blocks`` layers of width ``d_block``
    and an output of width ``d_term``, and each of the ``k`` members has its
    own networks.
    """

    first: jax.Array
    first_bias: jax.Array
    blocks: tuple[tuple[jax.Array, jax.Array], ...]
    last: jax.Array
    last_bias: jax.Array
    chunks: tuple[int, ...] = field(static=True)
    terms: tuple[tuple[int, ...], ...] = field(static=True)
    k: int = field(static=True)
    d_out: int = field(static=True)
    d_term: int = field(static=True)
    dropout_rate: float = field(static=True)
    activation: str = field(static=True)

    def __init__(
        self,
        d_in: int,
        *,
        key,
        feature_chunks: list[int] | None = None,
        terms=None,
        n_blocks: int = 2,
        d_block: int = 64,
        d_term: int = 8,
        k: int = 1,
        dropout: float = 0.0,
        activation: str = "silu",
    ):
        chunks = tuple([1] * int(d_in) if feature_chunks is None else (int(width) for width in feature_chunks))
        if sum(chunks) != int(d_in) or any(width < 1 for width in chunks):
            raise ShapeError(f"feature chunks must be positive widths summing to the encoded width {d_in}; received {chunks}.")
        if terms is None:
            terms = tuple((index,) for index in range(len(chunks)))
        else:
            terms = tuple(tuple(int(index) for index in term) for term in terms)
        if not terms:
            raise ShapeError("the additive backbone needs at least one term.")
        for term in terms:
            if not term or len(set(term)) != len(term) or any(not 0 <= index < len(chunks) for index in term):
                raise ShapeError(f"each term lists distinct fields between 0 and {len(chunks) - 1}; received {term}.")
        if len({tuple(sorted(term)) for term in terms}) != len(terms):
            raise ShapeError("the additive backbone's terms must be distinct.")
        if n_blocks < 1 or d_block < 1 or d_term < 1 or k < 1:
            raise ShapeError("additive n_blocks, d_block, d_term, and k must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ShapeError("dropout must be in [0, 1).")
        if activation not in _ACTIVATIONS:
            raise ShapeError(f"activation must be one of {tuple(_ACTIVATIONS)}; received {activation!r}.")
        gather = _term_gather(chunks, terms)
        count, width = gather.shape
        filled = gather < int(d_in)
        keys = jax.random.split(key, n_blocks + 1)
        # Each term's first layer is scaled by its own fan-in; empty slots start, and stay, at zero.
        limit = jnp.asarray(filled.sum(axis=1), dtype=jnp.float32) ** -0.5
        first = jax.random.uniform(keys[0], (k, count, width, d_block), jnp.float32, minval=-1.0, maxval=1.0)
        self.first = first * limit[None, :, None, None] * jnp.asarray(filled, dtype=jnp.float32)[None, :, :, None]
        self.first_bias = jnp.zeros((k, count, d_block), dtype=jnp.float32)
        self.blocks = tuple(
            (
                _rsqrt_uniform(keys[index], (k, count, d_block, d_block), d_block, jnp.float32),
                jnp.zeros((k, count, d_block), dtype=jnp.float32),
            )
            for index in range(1, n_blocks)
        )
        self.last = _rsqrt_uniform(keys[n_blocks], (k, count, d_block, d_term), d_block, jnp.float32)
        self.last_bias = jnp.zeros((k, count, d_term), dtype=jnp.float32)
        self.chunks = chunks
        self.terms = terms
        self.k = int(k)
        self.d_term = int(d_term)
        self.d_out = count * int(d_term)
        self.dropout_rate = float(dropout)
        self.activation = activation

    def term_outputs(self, x, *, key=None):
        """``(batch, member, terms, d_term)``: each term's network on its own fields."""
        d_in = sum(self.chunks)
        if x.ndim not in (2, 3) or x.shape[-1] != d_in or (x.ndim == 3 and x.shape[1] != self.k):
            raise ShapeError(
                f"the additive backbone expected (batch, {d_in}) or (batch, {self.k}, {d_in}); received {x.shape}."
            )
        gather = jnp.asarray(_term_gather(self.chunks, self.terms))
        padded = jnp.concatenate([x, jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype)], axis=-1)
        inputs = padded[..., gather]
        pattern = "btw,ktwh->bkth" if x.ndim == 2 else "bktw,ktwh->bkth"
        act = _ACTIVATIONS[self.activation]
        keys = None if key is None or self.dropout_rate == 0.0 else jax.random.split(key, len(self.blocks) + 1)
        hidden = act(jnp.einsum(pattern, inputs, self.first) + self.first_bias)
        hidden = _apply_dropout(hidden, self.dropout_rate, None if keys is None else keys[0])
        for index, (weight, bias) in enumerate(self.blocks):
            hidden = act(jnp.einsum("bkth,ktho->bkto", hidden, weight) + bias)
            hidden = _apply_dropout(hidden, self.dropout_rate, None if keys is None else keys[index + 1])
        return jnp.einsum("bkth,ktho->bkto", hidden, self.last) + self.last_bias

    def __call__(self, x, *, key):
        outputs = self.term_outputs(x, key=key)
        return outputs.reshape(outputs.shape[0], self.k, self.d_out)
