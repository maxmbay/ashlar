"""The fitted numerical core is an ordinary Equinox module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from equinox import Module, field

from ashlar.effects import embed_key
from ashlar.errors import ShapeError
from ashlar.layers import CategoricalEmbeddings, LinearReLUEmbeddings, SetEmbeddings


class PredictiveCore(Module):
    """Backbone and head applied to standardized numbers and encoded categories.

    ``numeric_z`` has shape ``(batch, n_numeric)`` and is already scaled.
    ``categorical_onehot`` has shape ``(batch, n_onehot)``. Learned category
    indices and set tokens are separate arguments so they stay inside this
    pytree. Any of those widths may be zero. The head output keeps the member axis.
    """

    embeddings: Module | None
    categorical_embeddings: CategoricalEmbeddings | None
    set_embeddings: SetEmbeddings | None
    backbone: Module
    bottleneck: Module | None
    head: Module
    soft_histogram: Module | None
    n_numeric: int = field(static=True)
    n_categorical: int = field(static=True)
    n_embedding_columns: int = field(static=True)
    n_set_columns: int = field(static=True)
    set_length: int = field(static=True)
    feature_dropout: float = field(static=True)
    feature_chunks: tuple[int, ...] = field(static=True)
    # ``ashlar.monotone.MonotoneTerm``: features kept out of the trunk and added to the head's outputs, one way.
    monotone: Module | None = None
    # ``ashlar.effects.RandomEffects``: grouping columns kept out of the trunk, as pooled offsets on the location.
    random_effects: Module | None = None
    # The columns of ``category_index`` the learned embeddings read; ``None`` is every column.
    embedding_positions: tuple[int, ...] | None = field(static=True, default=None)

    def __call__(self, numeric_z, categorical_onehot, category_index, set_tokens, set_mask, *, key, offsets: bool = True):
        """The head's raw output, ``(batch, members, outputs)``.

        ``offsets=False`` leaves out the random effects' offsets on the location (``ashlar.effects``), so a
        caller can add its own, as ``with_context`` does.
        """
        if numeric_z.ndim != 2 or categorical_onehot.ndim != 2:
            raise ShapeError(
                "core inputs must be (batch, n_numeric) and (batch, n_onehot); "
                f"received {numeric_z.shape} and {categorical_onehot.shape}."
            )
        if numeric_z.shape[0] != categorical_onehot.shape[0]:
            raise ShapeError("numeric and categorical batches have different lengths.")
        if numeric_z.shape[1] != self.n_numeric or categorical_onehot.shape[1] != self.n_categorical:
            raise ShapeError(
                "encoded widths do not match the fitted core: "
                f"expected numeric {self.n_numeric} and one-hot {self.n_categorical}; "
                f"received {numeric_z.shape[1]} and {categorical_onehot.shape[1]}."
            )
        if category_index.ndim != 2 or category_index.shape != (numeric_z.shape[0], self.n_embedding_columns):
            raise ShapeError(
                "category_index must have shape "
                f"(batch, {self.n_embedding_columns}); received {category_index.shape}."
            )
        expected_set = (numeric_z.shape[0], self.n_set_columns, self.set_length)
        if set_tokens.shape != expected_set or set_mask.shape != expected_set:
            raise ShapeError(
                f"set tokens and mask must have shape {expected_set}; "
                f"received {set_tokens.shape} and {set_mask.shape}."
            )
        pieces = []
        if key is None:
            feature_key, backbone_key = None, None
        else:
            feature_key, backbone_key = jax.random.split(key)
        trunk_z = numeric_z if self.monotone is None else self.monotone.without(numeric_z)
        if self.n_numeric:
            pieces.append(trunk_z if self.embeddings is None else self.embeddings(trunk_z))
        if self.n_categorical:
            pieces.append(categorical_onehot)
        if self.categorical_embeddings is not None:
            indices = category_index if self.embedding_positions is None else category_index[:, list(self.embedding_positions)]
            pieces.append(self.categorical_embeddings(indices))
        if self.set_embeddings is not None:
            pieces.append(self.set_embeddings(set_tokens, set_mask))
        if self.random_effects is not None and self.random_effects.embedded:
            # Pooled group embeddings: a posterior draw on a training step, the posterior mean at prediction.
            pieces.append(self.random_effects.embed(category_index, embed_key(key)))
        if not pieces:
            raise ShapeError("the core has no features.")
        encoded = pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces, axis=1)
        encoded = drop_features(encoded, self.feature_chunks, self.feature_dropout, feature_key)
        hidden = self.backbone(encoded, key=backbone_key)
        if self.bottleneck is not None:
            hidden = self.bottleneck(hidden)
        log_probs = self.head(hidden, key=None)
        if self.monotone is not None:
            log_probs = log_probs + self.monotone(numeric_z)
        if self.random_effects is not None:
            if not offsets:
                return log_probs
            # A training step (it has a key) reads the offsets on nodes; prediction integrates them.
            return self.random_effects(log_probs, category_index, numeric_z, training=key is not None)
        if self.monotone is not None or self.soft_histogram is None:
            return log_probs
        residual = self.soft_histogram(numeric_z)
        if residual.ndim != 2 or residual.shape[0] != log_probs.shape[0] or residual.shape[-1] != log_probs.shape[-1]:
            raise ShapeError(
                "soft stage histogram residual must be (batch, class) matching the head; "
                f"received {residual.shape} for head output {log_probs.shape}."
            )
        return jax.nn.log_softmax(log_probs + residual[:, None, :], axis=-1)


    def penalty(self):
        """Terms of the model's own beside the objective, per training row: the random effects' KL divergence."""
        if self.random_effects is None:
            return jnp.asarray(0.0, dtype=jnp.float32)
        return self.random_effects.penalty()


def drop_features(encoded, chunks: tuple[int, ...], rate: float, key):
    """Drop whole input fields during training, then rescale the keepers.

    ``chunks`` are the widths of the concatenated fields. A field is one numeric
    column, one one-hot column, one learned embedding, or one pooled set.
    ``key=None`` and ``rate=0`` leave the input unchanged, which is the
    prediction path and the way to turn this regularizer off.
    """
    if key is None or rate == 0.0:
        return encoded
    keep = jax.random.bernoulli(key, 1.0 - rate, (len(chunks),)).astype(encoded.dtype)
    scales = keep / jnp.asarray(1.0 - rate, dtype=encoded.dtype)
    pieces = [
        jnp.broadcast_to(scales[index], (encoded.shape[0], int(width)))
        for index, width in enumerate(chunks)
    ]
    return encoded * jnp.concatenate(pieces, axis=1)
