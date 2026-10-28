"""Shared TabM trunk with a normal outcome and a sometimes-missing binary outcome.

The head is an ordinary Equinox module. Training uses one weighted objective:
member-mean normal NLL, plus member-mean Bernoulli NLL only where the binary
label is present. Sample weights multiply that sum once. A missing binary
label does not drop the row and does not remove its weight from the normal term.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from equinox import Module, field

from ashlar.heads import LogitsHead, NormalHead
from ashlar.predict import distribution_from_normal_raw, positive_scale
from ashlar.weights import weighted_mean


class LinkedHead(Module):
    """One hidden state, two heads. Output is a dict pytree, not a vector."""

    normal: NormalHead
    event: LogitsHead
    k: int = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, **kwargs):
        del n_events, n_classes, kwargs
        normal_key, event_key = jax.random.split(key)
        self.normal = NormalHead(d_in, k, n_events=1, n_classes=2, key=normal_key)
        self.event = LogitsHead(d_in, k, n_events=1, n_classes=2, key=event_key)
        self.k = k

    def __call__(self, hidden, *, key):
        del key
        return {
            "normal": self.normal(hidden, key=None),
            "event": self.event(hidden, key=None),
        }


def linked_objective(raw, y, sample_weight):
    y = jnp.asarray(y)
    loc = raw["normal"][..., 0, 0]
    scale = positive_scale(raw["normal"][..., 0, 1])
    residual = (y[:, 0][:, None] - loc) / scale
    normal_nll = 0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(scale) + jnp.square(residual))
    observed = jnp.isfinite(y[:, 1])
    labels = jnp.where(observed, y[:, 1], 0).astype(jnp.int32)
    log_prob = jax.nn.log_softmax(raw["event"], axis=-1)
    chosen = jnp.take_along_axis(log_prob, labels[:, None, None], axis=-1)[..., 0]
    event_nll = jnp.where(observed, -jnp.mean(chosen, axis=1), 0.0)
    return weighted_mean(jnp.mean(normal_nll, axis=1) + event_nll, sample_weight)


def linked_point(raw):
    return jnp.mean(raw["normal"][..., 0, 0], axis=1)


def linked_dist(raw):
    return distribution_from_normal_raw(raw["normal"], "mixture")
