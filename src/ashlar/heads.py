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


class DistributionHead(_EnsembleHead):
    """Flat parameters of one ``ashlar.distributional`` head per member. Output ``(batch, member, width)``.

    ``kind`` is ``gaussian``, ``mixture``, ``flow``, ``histogram``,
    ``quantile``, ``student_t``, ``poisson``, or ``negative_binomial``. One
    event only. The initialization matches the forecasters: a small projection
    for the mixture, the flow, and the count heads, mixture means on a grid.
    ``zero_inflated`` adds a structural zero to a count head, and
    ``log_mean`` starts a count head's mean at ``exp(log_mean)``; the
    estimator sets it from the training target.
    """

    kind: str = field(static=True)

    def __init__(
        self,
        d_in,
        k,
        *,
        n_events,
        n_classes,
        key,
        kind="gaussian",
        bins=None,
        zero_inflated=False,
        log_mean=0.0,
        **kwargs,
    ):
        from ashlar.distributional import (
            HISTOGRAM_BINS,
            initial_bias,
            initial_weight_scale,
            output_width,
        )

        del n_classes
        if kwargs:
            raise ShapeError(f"distribution head received unknown arguments {sorted(kwargs)}.")
        if n_events != 1:
            raise ShapeError("a distributional table head models one target column.")
        if bins is not None and kind != "histogram":
            raise ShapeError(f"bins is the histogram head's bin count; the {kind} head does not take it.")
        count = HISTOGRAM_BINS if bins is None else bins
        width = output_width(kind, 1, count, bool(zero_inflated))
        scale = initial_weight_scale(kind, d_in)
        weight = scale * jax.random.normal(key, (k, d_in, width), dtype=jnp.float32)
        start = initial_bias(kind, 1, count, zero_inflated=bool(zero_inflated), log_mean=float(log_mean))
        bias = jnp.broadcast_to(start, (k, width))
        self.output = LinearEnsemble(weight=weight, bias=jnp.asarray(bias))
        self.k = k
        self.kind = kind

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        return self.output(hidden)


class MultivariateNormalHead(_EnsembleHead):
    """Joint normal over ``n_events`` target columns. Output ``(batch, member, event * (2 + rank))``.

    Per row and member: ``event`` locations, ``event`` raw scales, and an
    ``event × rank`` covariance factor, so the covariance is ``W Wᵀ + diag(D)``
    with ``D = (softplus(raw) + 1e-4)²``. ``rank`` (2 by default, at most the
    number of columns) sets how many shared directions the columns can move
    along together; ``rank=0`` makes them independent. The factor starts
    small and nonzero, so a fit starts close to independent columns and
    learns the dependence the data supports.
    """

    n_events: int = field(static=True)
    rank: int = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, rank=None, **kwargs):
        del n_classes
        if kwargs:
            raise ShapeError(f"multivariate normal head received unknown arguments {sorted(kwargs)}.")
        if n_events < 1:
            raise ShapeError("n_events must be positive.")
        rank = min(2, int(n_events)) if rank is None else int(rank)
        if not 0 <= rank <= int(n_events):
            raise ShapeError(f"rank must be between 0 and the number of target columns, {n_events}; received {rank}.")
        loc_key, factor_key, bias_key = jax.random.split(key, 3)
        scale = (1.0 / d_in) ** 0.5
        moments = scale * jax.random.normal(loc_key, (k, d_in, 2 * n_events), dtype=jnp.float32)
        factor = 0.01 * jax.random.normal(factor_key, (k, d_in, n_events * rank), dtype=jnp.float32)
        factor_bias = 0.1 * jax.random.normal(bias_key, (k, n_events * rank), dtype=jnp.float32)
        bias = jnp.concatenate([jnp.zeros((k, 2 * n_events), dtype=jnp.float32), factor_bias], axis=-1)
        self.output = LinearEnsemble(weight=jnp.concatenate([moments, factor], axis=-1), bias=bias)
        self.k = k
        self.n_events = int(n_events)
        self.rank = rank

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        return self.output(hidden)


class MultivariateFlowHead(_EnsembleHead):
    """Conditional masked autoregressive flow over ``n_events`` target columns.

    Output ``(context, w_in, w_out)``, read by
    ``ashlar.distributions.AutoregressiveFlow``: a per-row context
    ``(batch, member, depth * width)`` from the trunk, and masked network
    weights shared by every row. ``hidden`` (64 by default) is the masked
    network's width and ``depth`` (1 by default) the number of autoregressive
    blocks, the column order reversed in every other one. The context starts
    near zero, which is the standard normal on the standardized target.
    """

    w_in: jax.Array
    w_out: jax.Array
    n_events: int = field(static=True)
    depth: int = field(static=True)
    hidden: int = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, depth=1, hidden=64, **kwargs):
        from ashlar.distributions import autoregressive_context_width

        del n_classes
        if kwargs:
            raise ShapeError(f"multivariate flow head received unknown arguments {sorted(kwargs)}.")
        if n_events < 1:
            raise ShapeError("n_events must be positive.")
        if int(depth) < 1 or int(hidden) < 1:
            raise ShapeError(f"depth and hidden must be at least 1; received {depth} and {hidden}.")
        d, depth, hidden = int(n_events), int(depth), int(hidden)
        width = autoregressive_context_width(d, hidden)
        out_key, in_key, made_key = jax.random.split(key, 3)
        self.output = LinearEnsemble(
            weight=0.01 * jax.random.normal(out_key, (k, d_in, depth * width), dtype=jnp.float32),
            bias=jnp.zeros((k, depth * width), dtype=jnp.float32),
        )
        self.w_in = jax.random.normal(in_key, (k, depth, hidden, d), dtype=jnp.float32) / max(d, 1) ** 0.5
        self.w_out = 0.01 * jax.random.normal(made_key, (k, depth, width - hidden, hidden), dtype=jnp.float32)
        self.k = k
        self.n_events, self.depth, self.hidden = d, depth, hidden

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        return self.output(hidden), self.w_in, self.w_out


class HurdleHead(_EnsembleHead):
    """Probability of an exact zero plus a positive severity. Output ``(batch, member, 3)``.

    The three outputs are the zero logit and the severity's two parameters,
    read by ``ashlar.distributions.hurdle_distribution``. ``severity`` is
    ``lognormal`` (default) or ``gamma``. ``start`` is the initial bias; the
    estimator sets it from the training target, so the fit starts at the
    observed share of zeros and the spread of the positive values.
    """

    severity: str = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, severity="lognormal", start=(0.0, 0.0, 0.0), **kwargs):
        from ashlar.distributions import HURDLE_SEVERITIES

        del n_classes
        if kwargs:
            raise ShapeError(f"hurdle head received unknown arguments {sorted(kwargs)}.")
        if n_events != 1:
            raise ShapeError("a hurdle head models one target column.")
        if severity not in HURDLE_SEVERITIES:
            raise ShapeError(f"hurdle severity must be one of {HURDLE_SEVERITIES}; received {severity!r}.")
        weight = 0.01 * jax.random.normal(key, (k, d_in, 3), dtype=jnp.float32)
        bias = jnp.broadcast_to(jnp.asarray(start, dtype=jnp.float32), (k, 3))
        self.output = LinearEnsemble(weight=weight, bias=jnp.asarray(bias))
        self.k = k
        self.severity = severity

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        return self.output(hidden)
