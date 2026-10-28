"""Output heads. Every head reads ``(batch, member, hidden)`` and keeps the member axis.

Constructor contract, shared by built-in and user heads:

    Head(d_in, k, *, n_events, n_classes, key, **user_kwargs)

``d_in`` is the backbone width. ``k`` must equal ``backbone.k``. Unused task
arguments may be ignored. A user head is an ordinary Equinox module; it does
not need a package-specific estimator subclass.
"""

from __future__ import annotations

import jax
from equinox import Module, field

import jax.numpy as jnp

from ashlar.errors import ShapeError
from ashlar.layers import LinearEnsemble, _rsqrt_uniform
from ashlar.objectives import ordinal_log_probabilities


class _EnsembleHead(Module):
    output: LinearEnsemble
    k: int = field(static=True)

    def _check(self, hidden):
        if hidden.ndim != 3 or hidden.shape[1] != self.k or hidden.shape[-1] != self.output.weight.shape[1]:
            raise ShapeError(
                "head expected (batch, member, hidden) with "
                f"member={self.k} and hidden={self.output.weight.shape[1]}; received {hidden.shape}."
            )


class PointHead(_EnsembleHead):
    """Point regression. Output ``(batch, member, event)``."""

    n_events: int = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, **kwargs):
        del n_classes, kwargs
        if n_events < 1:
            raise ShapeError("n_events must be positive.")
        weight_key, bias_key = jax.random.split(key)
        self.output = LinearEnsemble(
            weight=_rsqrt_uniform(weight_key, (k, d_in, n_events), d_in, jax.numpy.float32),
            bias=_rsqrt_uniform(bias_key, (k, n_events), d_in, jax.numpy.float32),
        )
        self.k = k
        self.n_events = n_events

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        return self.output(hidden)


class LogitsHead(_EnsembleHead):
    """Classification logits. Output ``(batch, member, class)``."""

    n_classes: int = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, **kwargs):
        del n_events, kwargs
        if n_classes < 2:
            raise ShapeError("classification requires at least 2 classes.")
        weight_key, bias_key = jax.random.split(key)
        self.output = LinearEnsemble(
            weight=_rsqrt_uniform(weight_key, (k, d_in, n_classes), d_in, jax.numpy.float32),
            bias=_rsqrt_uniform(bias_key, (k, n_classes), d_in, jax.numpy.float32),
        )
        self.k = k
        self.n_classes = n_classes

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        return self.output(hidden)


class SoftmaxHead(LogitsHead):
    """Independent class scores. Output ``(batch, member, class)`` log-probabilities.

    Each class has its own linear map from the hidden state. A higher score
    for one quality value does not shift the other values along one latent.
    """

    def __call__(self, hidden, *, key):
        logits = super().__call__(hidden, key=key)
        return jax.nn.log_softmax(logits, axis=-1)


def _initial_cut_raw(k: int, n_classes: int):
    n_cuts = n_classes - 1
    if n_cuts == 1:
        return jnp.zeros((k, 1), dtype=jnp.float32)
    grid = jnp.linspace(-2.0, 2.0, n_cuts, dtype=jnp.float32)
    gaps = grid[1:] - grid[:-1]
    raw_gaps = jnp.log(jnp.expm1(gaps))
    raw = jnp.concatenate([grid[:1], raw_gaps], axis=0)
    return jnp.repeat(raw[None, :], k, axis=0)


def cut_raw_from_class_prior(class_prior, k: int):
    """Cut parameters whose zero latent reproduces ``class_prior``.

    ``class_prior`` is a probability vector over the ordered classes. The
    returned array has shape ``(k, n_classes - 1)`` and matches
    ``ordered_thresholds``: the first column is the lowest logit cut, and
    each later column is the softplus-inverse of the gap after the ``1e-4``
    separation used by that function.
    """
    prior = jnp.asarray(class_prior, dtype=jnp.float32).reshape(-1)
    if prior.shape[0] < 2:
        raise ShapeError("an ordinal class prior needs at least 2 classes.")
    if not bool(jnp.isfinite(prior).all()) or bool(jnp.any(prior <= 0)):
        raise ShapeError("an ordinal class prior must be finite and positive.")
    prior = prior / jnp.sum(prior)
    cumulative = jnp.cumsum(prior)[:-1]
    cumulative = jnp.clip(cumulative, 1e-4, 1.0 - 1e-4)
    theta = jnp.log(cumulative) - jnp.log1p(-cumulative)
    if theta.shape[0] == 1:
        raw = theta
    else:
        gaps = jnp.maximum(theta[1:] - theta[:-1], jnp.asarray(1e-3, dtype=theta.dtype))
        soft = gaps - jnp.asarray(1e-4, dtype=theta.dtype)
        raw = jnp.concatenate([theta[:1], jnp.log(jnp.expm1(soft))], axis=0)
    return jnp.repeat(raw[None, :], int(k), axis=0)


class OrdinalHead(_EnsembleHead):
    """Cumulative logit. Output ``(batch, member, class)`` log-probabilities.

    Each member has one latent score and strictly increasing cutpoints. The
    cuts do not depend on the row. Class order is the numeric order of the
    training labels, not an unordered softmax.
    """

    cut_raw: jax.Array
    n_classes: int = field(static=True)
    freeze_class_prior: bool = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, class_prior=None, **kwargs):
        del n_events
        if kwargs:
            raise ShapeError(f"ordinal head received unknown arguments {sorted(kwargs)}.")
        if n_classes < 2:
            raise ShapeError("ordinal classification requires at least 2 classes.")
        weight_key, bias_key = jax.random.split(key)
        if class_prior is None:
            bias = _rsqrt_uniform(bias_key, (k, 1), d_in, jax.numpy.float32)
            self.cut_raw = _initial_cut_raw(k, n_classes)
            self.freeze_class_prior = False
        else:
            bias = jnp.zeros((k, 1), dtype=jnp.float32)
            self.cut_raw = cut_raw_from_class_prior(class_prior, k)
            if int(self.cut_raw.shape[-1]) != n_classes - 1:
                raise ShapeError(
                    f"class prior has length {int(self.cut_raw.shape[-1]) + 1}; the head has {n_classes} classes."
                )
            self.freeze_class_prior = True
        self.output = LinearEnsemble(
            weight=_rsqrt_uniform(weight_key, (k, d_in, 1), d_in, jax.numpy.float32),
            bias=bias,
        )
        self.k = k
        self.n_classes = n_classes

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        latent = self.output(hidden)[..., 0]
        cuts = jax.lax.stop_gradient(self.cut_raw) if self.freeze_class_prior else self.cut_raw
        return ordinal_log_probabilities(latent, cuts)


class NormalHead(_EnsembleHead):
    """Diagonal normal parameters. Output ``(batch, member, event, 2)``.

    The last axis is ``(location, raw_scale)``. ``raw_scale`` is unconstrained;
    the predictive object maps it through ``softplus(raw_scale) + 1e-4``.
    """

    n_events: int = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, **kwargs):
        del n_classes, kwargs
        if n_events < 1:
            raise ShapeError("n_events must be positive.")
        weight_key, bias_key = jax.random.split(key)
        self.output = LinearEnsemble(
            weight=_rsqrt_uniform(weight_key, (k, d_in, n_events * 2), d_in, jax.numpy.float32),
            bias=_rsqrt_uniform(bias_key, (k, n_events * 2), d_in, jax.numpy.float32),
        )
        self.k = k
        self.n_events = n_events

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        raw = self.output(hidden)
        batch = raw.shape[0]
        return raw.reshape(batch, self.k, self.n_events, 2)
