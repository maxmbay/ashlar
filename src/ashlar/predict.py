"""Predictive distributions and explicit ensemble reductions.

Member axes are ensemble members, not posterior draws. A mixture of member
predictive distributions is one aggregation policy. Averaging location and
scale, then evaluating one distribution, is a different policy. Neither
policy is a calibrated split of aleatoric and epistemic uncertainty.

For independent events, member variance and between-member variance of the
means combine by the law of total variance, separately on each event axis.
The event axis is never the member axis.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from equinox import Module

from ashlar.errors import ShapeError


def _as_events(y, n_events: int):
    y = jnp.asarray(y)
    if y.ndim == 1:
        if n_events != 1:
            raise ShapeError(f"y must have shape (batch, {n_events}); received {y.shape}.")
        y = y[:, None]
    if y.ndim != 2 or y.shape[1] != n_events:
        raise ShapeError(f"y must have shape (batch, {n_events}); received {y.shape}.")
    return y


def normal_log_prob(y, loc, scale):
    """Elementwise normal log-density. ``scale`` is standard deviation."""
    z = (y - loc) / scale
    return -0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(scale) + z**2)


class NormalMixture(Module):
    """Equal-weight mixture of diagonal normals, one component per member.

    ``loc`` and ``scale`` have shape ``(batch, member, event)``.
    """

    loc: jax.Array
    scale: jax.Array

    def __check_init__(self):
        if self.loc.shape != self.scale.shape or self.loc.ndim != 3:
            raise ShapeError(
                "NormalMixture loc and scale must have shape (batch, member, event); "
                f"received {self.loc.shape} and {self.scale.shape}."
            )

    def mean(self):
        return jnp.mean(self.loc, axis=1)

    def variance(self):
        member_var = jnp.mean(jnp.square(self.scale), axis=1)
        centered = self.loc - self.mean()[:, None, :]
        between = jnp.mean(jnp.square(centered), axis=1)
        return member_var + between

    def log_prob(self, y):
        y = _as_events(y, self.loc.shape[-1])
        log_comp = normal_log_prob(y[:, None, :], self.loc, self.scale).sum(axis=-1)
        log_mix = log_comp + jnp.log(jnp.asarray(1.0 / self.loc.shape[1], dtype=self.loc.dtype))
        return jax.nn.logsumexp(log_mix, axis=1)

    def sample(self, key, sample_shape=()):
        member_key, noise_key = jax.random.split(key)
        batch = self.loc.shape[0]
        choices = jax.random.randint(member_key, sample_shape + (batch,), 0, self.loc.shape[1])
        loc_by_member = jnp.swapaxes(self.loc, 0, 1)
        scale_by_member = jnp.swapaxes(self.scale, 0, 1)
        rows = jnp.arange(batch)
        loc = loc_by_member[choices, rows]
        scale = scale_by_member[choices, rows]
        noise = jax.random.normal(noise_key, loc.shape, dtype=loc.dtype)
        return loc + scale * noise


class MeanParameterNormal(Module):
    """One diagonal normal whose parameters are the arithmetic means of members.

    ``loc`` and ``scale`` have shape ``(batch, event)``. A two-component
    mixture of N(-5, 1) and N(5, 1) has variance 26. This class, built from
    the averaged parameters, has variance 1.
    """

    loc: jax.Array
    scale: jax.Array

    def __check_init__(self):
        if self.loc.shape != self.scale.shape or self.loc.ndim != 2:
            raise ShapeError(
                "MeanParameterNormal loc and scale must have shape (batch, event); "
                f"received {self.loc.shape} and {self.scale.shape}."
            )

    def mean(self):
        return self.loc

    def variance(self):
        return jnp.square(self.scale)

    def log_prob(self, y):
        y = _as_events(y, self.loc.shape[-1])
        return normal_log_prob(y, self.loc, self.scale).sum(axis=-1)

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, sample_shape + self.loc.shape, dtype=self.loc.dtype)
        return self.loc + self.scale * noise


def positive_scale(raw_scale):
    """Map an unconstrained head output to a standard deviation.

    ``softplus(raw_scale) + 1e-4``. The floor keeps the density finite.
    """
    return jax.nn.softplus(raw_scale) + jnp.asarray(1e-4, dtype=raw_scale.dtype)


def split_normal_parameters(raw):
    """Split a normal head output ``(batch, member, event, 2)`` into loc and scale."""
    if raw.ndim != 4 or raw.shape[-1] != 2:
        raise ShapeError(
            "normal head output must have shape (batch, member, event, 2); "
            f"received {getattr(raw, 'shape', None)}."
        )
    return raw[..., 0], positive_scale(raw[..., 1])


def distribution_from_normal_raw(raw, policy: str):
    loc, scale = split_normal_parameters(raw)
    if policy == "mixture":
        return NormalMixture(loc=loc, scale=scale)
    if policy == "mean_parameters":
        return MeanParameterNormal(loc=jnp.mean(loc, axis=1), scale=jnp.mean(scale, axis=1))
    raise ShapeError("normal prediction policy must be 'mixture' or 'mean_parameters'.")


def mean_probabilities(logits):
    """Average member probabilities. ``logits`` is ``(batch, member, class)``."""
    if logits.ndim != 3:
        raise ShapeError(f"logits must have shape (batch, member, class); received {logits.shape}.")
    return jax.nn.softmax(logits, axis=-1).mean(axis=1)


def mean_logits(logits):
    """Softmax of the averaged member logits. Different from ``mean_probabilities``."""
    if logits.ndim != 3:
        raise ShapeError(f"logits must have shape (batch, member, class); received {logits.shape}.")
    return jax.nn.softmax(jnp.mean(logits, axis=1), axis=-1)
