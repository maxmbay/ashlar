"""The forecast heads on a table, through the custom-head API.

``TabularEstimator`` ships a point head and a normal head (``task="normal"``).
The other distributional heads live in ``ashlar.distributional`` and are wired
into the two forecasters. This file attaches any of them to a table with
``task="custom"``: a head that emits the flat parameters per member, the head's
own loss as the objective, and its point summary as ``point_fn``.

Standardize the target first. The histogram bins are fixed on [-4, 4] of that
axis, and the mixture starts with its means spread across it.

    from table_heads import table_head

    mean, std = y.mean(), y.std()
    model = table_head("histogram").fit(X, (y - mean) / std)
    raw = model.predict_raw(X_new)   # (rows, members, 16) logits
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ashlar import TabularEstimator
from ashlar.distributional import (
    decode,
    distributional_loss,
    initial_bias,
    initial_weight_scale,
    output_width,
    point_values,
)
from ashlar.layers import LinearEnsemble


class DistributionHead(eqx.Module):
    """A forecast head on a table: one flat projection per member, decoded by ``ashlar.distributional``."""

    output: LinearEnsemble
    k: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, kind="gaussian"):
        del n_events, n_classes
        width = output_width(kind, 1)
        scale = initial_weight_scale(kind, d_in)
        weight = scale * jax.random.normal(key, (k, d_in, width), dtype=jnp.float32)
        bias = jnp.broadcast_to(initial_bias(kind, 1), (k, width))
        self.output = LinearEnsemble(weight=weight, bias=jnp.asarray(bias))
        self.k = k
        self.kind = kind

    def __call__(self, hidden, *, key):
        del key
        return self.output(hidden)


def flatten_members(raw):
    """``(batch, member, width)`` to ``(batch * member, width)``, with the two sizes."""
    batch, members, width = raw.shape
    return raw.reshape(batch * members, width), batch, members


def _objective(kind):
    def objective(raw, y, sample_weight):
        del sample_weight
        flat, _batch, members = flatten_members(raw)
        target = jnp.repeat(jnp.asarray(y, dtype=flat.dtype).reshape(-1), members)[:, None]
        return distributional_loss(kind, decode(kind, flat, 1), target)

    return objective


def _point_fn(kind):
    def point(raw):
        flat, batch, members = flatten_members(raw)
        values = point_values(kind, decode(kind, flat, 1))
        return jnp.mean(values.reshape(batch, members), axis=1)

    return point


def table_head(kind: str, **overrides) -> TabularEstimator:
    """A default ``TabularEstimator`` whose head is ``kind``: gaussian, mixture, flow, histogram, or quantile."""
    params = dict(
        task="custom",
        head=DistributionHead,
        head_kwargs={"kind": kind},
        objective=_objective(kind),
        point_fn=_point_fn(kind),
    )
    params.update(overrides)
    return TabularEstimator(**params)
