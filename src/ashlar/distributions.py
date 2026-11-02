"""Predictive distributions as JAX pytrees, one interface for every head.

Every class here is an Equinox module, so a distribution is an ordinary JAX
pytree: it passes through ``jax.jit``, ``jax.vmap``, and ``jax.grad`` like any
other argument, and its parameters are the leaves.

Shapes follow the usual convention. ``batch_shape`` indexes independent rows;
``event_shape`` is the shape of one outcome, ``()`` for one number and
``(d,)`` for a joint outcome over ``d`` columns. Every array leaf starts with
the full batch shape, so ``dist[2:5]`` slices rows the way it would slice an
array.

Methods, with ``value`` broadcasting against ``batch_shape + event_shape``:

- ``log_prob(value)``: log density, or log mass for a count. Shape: the
  broadcast batch shape. A joint outcome sums over its event axis.
- ``sample(key, sample_shape=())``: shape ``sample_shape + batch_shape +
  event_shape``. ``reparameterized`` says whether gradients flow from a draw
  back to the parameters.
- ``mean()``, ``variance()``, ``stddev()``: shape ``batch_shape + event_shape``.
- ``cdf(value)`` and ``quantile(q)`` for one number. ``q`` broadcasts against
  ``batch_shape``, so ``dist.quantile(jnp.array([0.1, 0.9])[:, None])`` is
  ``(2, rows)``. A quantile without a closed form is found by bisection on the
  CDF and then given the implicit-function gradient, so it is exact to float
  precision and still differentiable.
- ``marginals()``: the univariate marginal of every coordinate, with the event
  axis moved into the batch. A univariate distribution returns itself.
- ``project(weights)``: for a joint outcome, the univariate distribution of
  the weighted sum ``weights · Y``, such as the total, where it has a closed
  form: the joint normal, a mixture of them, and an affine map of either.
- ``log_cdf(value)`` and ``log_survival(value)``: ``log P(Y <= value)`` and
  ``log P(Y > value)``, computed in log space so a far tail stays finite.
  ``log_cdf_left`` and ``log_survival_left`` are ``log P(Y < value)`` and
  ``log P(Y >= value)``, which differ only at an atom.
- ``log_interval(lower, upper)``: ``log P(lower <= Y <= upper)`` for the
  closed interval, with ``-inf`` or ``inf`` for an open side. It subtracts on
  the side of the median the interval lies on, in log space, so a censored or
  interval label far in a tail keeps a finite likelihood and gradient. This
  is what ``ashlar.observed`` trains on.
- ``convolve_normal(scale)``: the distribution of ``Y`` plus independent
  normal noise of standard deviation ``scale``, for the families where that
  has a closed form: ``Normal``, and a ``Mixture`` or ``AffineTransformed`` of
  normals.

A method a family cannot support raises ``NotImplementedError``: the quantile
head has three levels and no density, and a multivariate outcome has no
single CDF.

``Mixture`` is the ensemble: an estimator's members form an equal-weight
mixture over the last batch axis of its components. ``AffineTransformed``
puts a head trained on a standardized target back in the units of ``y``.
"""

from __future__ import annotations

import math
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module, field
from jax.scipy.special import betainc, betaln, gammainc, gammaincc, log_ndtr, ndtr, ndtri

from ashlar.errors import AshlarError, ShapeError

QUADRATURE_NODES = 512
_BISECTION_STEPS = 80


class Distribution(Module):
    """Base class. Subclasses set the parameters and the methods they support."""

    reparameterized: ClassVar[bool] = False
    support: ClassVar[str] = "real"
    # True for a family with a density and no atom, where ``P(Y < v) = P(Y <= v)``.
    atomless: ClassVar[bool] = False
    # True where some mass sits beyond every finite value, as for a time that never comes (``Cured``).
    defective: ClassVar[bool] = False

    @property
    def batch_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def event_shape(self) -> tuple[int, ...]:
        return ()

    def log_prob(self, value):
        raise NotImplementedError(f"{type(self).__name__} has no log density.")

    def prob(self, value):
        return jnp.exp(self.log_prob(value))

    def sample(self, key, sample_shape=()):
        raise NotImplementedError(f"{type(self).__name__} has no sampler.")

    def mean(self):
        raise NotImplementedError(f"{type(self).__name__} has no closed-form mean.")

    def variance(self):
        raise NotImplementedError(f"{type(self).__name__} has no closed-form variance.")

    def stddev(self):
        return jnp.sqrt(self.variance())

    def covariance(self):
        """``batch_shape + event_shape + event_shape`` for a joint outcome."""
        raise NotImplementedError(f"{type(self).__name__} has no covariance.")

    def cdf(self, value):
        raise NotImplementedError(f"{type(self).__name__} has no CDF.")

    def cdf_left(self, value):
        """``P(Y < value)``. The CDF itself, except where the distribution has an atom."""
        if self.support == "nonnegative_integer":
            return self.cdf(jnp.ceil(jnp.asarray(value)) - 1.0)
        return self.cdf(value)

    def log_cdf(self, value):
        """``log P(Y <= value)``. Families with a closed form override this to stay finite in the far tail."""
        return _safe_log(self.cdf(value))

    def log_survival(self, value):
        """``log P(Y > value)``."""
        return _safe_log(1.0 - self.cdf(value))

    def log_cdf_left(self, value):
        """``log P(Y < value)``: ``log_cdf`` except at an atom."""
        if self.atomless:
            return self.log_cdf(value)
        if self.support == "nonnegative_integer":
            return self.log_cdf(jnp.ceil(jnp.asarray(value)) - 1.0)
        return _safe_log(self.cdf_left(value))

    def log_survival_left(self, value):
        """``log P(Y >= value)``: ``log_survival`` except at an atom."""
        if self.atomless:
            return self.log_survival(value)
        if self.support == "nonnegative_integer":
            return self.log_survival(jnp.ceil(jnp.asarray(value)) - 1.0)
        return _safe_log(1.0 - self.cdf_left(value))

    def log_interval(self, lower, upper):
        """``log P(lower <= Y <= upper)`` for the closed interval. ``-inf`` or ``inf`` leaves a side open.

        Both bounds broadcast against ``batch_shape``. The difference is taken
        on the side of the median the interval lies on, in log space, so it
        stays accurate and finite far in either tail. An empty interval,
        ``lower > upper``, has probability 0.
        """
        if self.event_shape:
            raise NotImplementedError("a joint outcome has no single interval; use marginals().log_interval(...).")
        lower, upper = _bounds(lower, upper)
        has_lower = lower > -jnp.inf
        has_upper = upper < jnp.inf
        low = jnp.where(jnp.isfinite(lower), lower, 0.0)
        high = jnp.where(jnp.isfinite(upper), upper, 0.0)
        below_upper = jnp.where(has_upper, self.log_cdf(high), 0.0)
        above_upper = jnp.where(has_upper, self.log_survival(high), -jnp.inf)
        below_lower = jnp.where(has_lower, self.log_cdf_left(low), -jnp.inf)
        above_lower = jnp.where(has_lower, self.log_survival_left(low), 0.0)
        result = _interval_from_tails(below_upper, above_upper, below_lower, above_lower)
        return jnp.where(_empty(lower, upper), -jnp.inf, result)

    def cdf_interval(self, lower, upper):
        """``(P(Y < lower), P(Y <= upper))``: where the closed interval starts and ends on the CDF axis.

        A coarse label's probability integral transform is uniform between the two.
        """
        lower, upper = _bounds(lower, upper)
        low = jnp.where(lower > -jnp.inf, self.cdf_left(jnp.where(jnp.isfinite(lower), lower, 0.0)), 0.0)
        high = jnp.where(upper < jnp.inf, self.cdf(jnp.where(jnp.isfinite(upper), upper, 0.0)), 1.0)
        return low, high

    def convolve_normal(self, scale):
        """The distribution of ``Y + e`` for independent ``e ~ Normal(0, scale)``."""
        raise NotImplementedError(f"{type(self).__name__} has no closed form with added normal noise.")

    def log_box(self, lower, upper, points=None):
        """For a joint outcome, the log-likelihood of every column lying in its closed interval ``[lower, upper]``.

        ``lower`` and ``upper`` broadcast against ``batch_shape + event_shape``.
        A column with ``lower == upper`` is an exact value and contributes its
        density; any other column contributes the probability of its interval,
        with ``-inf`` or ``inf`` for an open side. So a row whose columns are
        each exact, censored, or in an interval scores ``p(exact columns) *
        P(coarse columns in their intervals | exact columns)``. A family that
        has to integrate over its coarse columns does so at the quasi-random
        ``points``, shaped ``(count, columns)`` in ``(0, 1)``;
        ``box_points(BOX_POINTS, d)`` by default.
        """
        raise NotImplementedError(f"{type(self).__name__} has no box likelihood.")

    def quantile(self, q):
        if self.event_shape:
            raise NotImplementedError("a joint outcome has no single quantile; use marginals().quantile(q).")
        if self.support == "nonnegative_integer":
            return _integer_quantile(self, q)
        return _bisection_quantile(self, q)

    def median(self):
        return self.quantile(0.5)

    def marginals(self) -> "Distribution":
        if self.event_shape:
            raise NotImplementedError(f"{type(self).__name__} does not expose its marginals.")
        return self

    def project(self, weights) -> "Distribution":
        """For a joint outcome, the distribution of the weighted sum ``weights · Y``, batch ``batch_shape``.

        ``weights`` is one number per column, or broadcasts against
        ``batch_shape + event_shape``. A weighted sum of jointly normal columns
        is normal, so the joint normal, a mixture of them, and an affine map
        of either have it in closed form; a flow does not, and its sum is read
        from draws.
        """
        raise NotImplementedError(f"{type(self).__name__} has no closed form for a weighted sum of its columns.")

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        if any(item is Ellipsis or item is None for item in index):
            raise ShapeError("index a distribution's batch axes with integers and slices only.")
        if len(index) > len(self.batch_shape):
            raise ShapeError(f"index has {len(index)} axes; the batch shape is {self.batch_shape}.")
        return self._index(index)

    def _index(self, index):
        """This distribution at ``index`` over its leading batch axes. Every leaf starts with the batch."""
        return jax.tree.map(lambda leaf: leaf[index], self)

    def __repr__(self):
        return f"{type(self).__name__}(batch_shape={self.batch_shape}, event_shape={self.event_shape})"


def _broadcast(*arrays):
    arrays = [jnp.asarray(array) for array in arrays]
    dtype = jnp.result_type(*arrays, jnp.float32)
    shape = jnp.broadcast_shapes(*(array.shape for array in arrays))
    return [jnp.broadcast_to(array.astype(dtype), shape) for array in arrays]


def _normal_log_prob(value, loc, scale):
    z = (value - loc) / scale
    return -0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(scale) + z**2)


# The smallest probability a log is taken of; float32's smallest normal number is about 1.2e-38.
_TINY = 1e-37
_LOG_HALF = math.log(0.5)


def _safe_log(probability):
    """``log`` of a probability, floored at ``log(1e-37)`` so neither the value nor its gradient is infinite."""
    return jnp.log(jnp.maximum(probability, _TINY))


def _log_or(probability, fallback):
    """``log(probability)``, or ``fallback`` where the probability underflows, and never below ``fallback``.

    ``fallback`` is a lower bound of the log probability that stays finite,
    such as the log mass of one count inside the interval.
    """
    usable = probability > _TINY
    direct = jnp.log(jnp.where(usable, probability, 1.0))
    return jnp.where(usable, jnp.maximum(direct, fallback), fallback)


def _log1mexp(t):
    """``log(1 - exp(t))`` for ``t <= 0``, accurate on both sides of ``-log 2`` (Maechler, 2012).

    ``t`` is held below ``-1e-7``, so an interval of zero width gets a very
    small probability with a finite gradient instead of ``-inf``.
    """
    t = jnp.minimum(t, -1e-7)
    return jnp.where(t > _LOG_HALF, jnp.log(-jnp.expm1(t)), jnp.log1p(-jnp.exp(t)))


def _log_diff_exp(big, small):
    """``log(exp(big) - exp(small))`` for ``big >= small``. ``small`` may be ``-inf``."""
    finite = jnp.isfinite(big)
    safe_big = jnp.where(finite, big, 0.0)
    gap = jnp.where(finite, small - safe_big, -jnp.inf)
    return jnp.where(finite, safe_big + _log1mexp(gap), big)


def _interval_from_tails(below_upper, above_upper, below_lower, above_lower):
    """``log P(a <= Y <= b)`` from ``log P(Y <= b)``, ``log P(Y > b)``, ``log P(Y < a)``, ``log P(Y >= a)``.

    Below the median the probability is ``F(b) - F(a-)``; above it,
    ``S(a-) - S(b)``; across it, ``1 - F(a-) - S(b)``. Each is the form whose
    terms are not both close to 1.
    """
    lower_side = _log_diff_exp(below_upper, below_lower)
    upper_side = _log_diff_exp(above_lower, above_upper)
    outside = jnp.minimum(jnp.exp(below_lower) + jnp.exp(above_upper), 1.0 - 1e-7)
    across = jnp.log1p(-outside)
    return jnp.where(below_upper <= _LOG_HALF, lower_side, jnp.where(above_lower <= _LOG_HALF, upper_side, across))


def _bounds(lower, upper):
    lower = jnp.asarray(lower, dtype=jnp.float32)
    upper = jnp.asarray(upper, dtype=jnp.float32)
    shape = jnp.broadcast_shapes(lower.shape, upper.shape)
    return jnp.broadcast_to(lower, shape), jnp.broadcast_to(upper, shape)


def _empty(lower, upper):
    return (lower > upper) | (lower == jnp.inf) | (upper == -jnp.inf)


def _finite_standardized(value, loc, scale):
    """``(value - loc) / scale``, with an infinite ``value`` passed through so no infinity meets ``loc`` or ``scale``."""
    value = jnp.asarray(value, dtype=jnp.float32)
    finite = jnp.isfinite(value)
    inner = (jnp.where(finite, value, 0.0) - loc) / scale
    return jnp.where(finite, inner, value)


@jax.custom_jvp
def _betainc(a, b, x):
    """The regularized incomplete beta function with a gradient in ``a`` and ``b`` as well as ``x``.

    JAX's ``betainc`` differentiates only in ``x``. The Student-t's degrees of
    freedom and the negative binomial's dispersion sit in ``a`` or ``b``, so
    their CDFs need the other two: central differences at a relative step of
    1%, accurate to about 1e-4 in float32.
    """
    return betainc(a, b, x)


def _broadcast_betainc(a, b, x):
    a, b, x = jnp.broadcast_arrays(*(jnp.asarray(part, dtype=jnp.float32) for part in (a, b, x)))
    return _betainc(a, b, x)


def _shape_step(value):
    return 1e-2 * jnp.maximum(jnp.abs(value), 1e-3)


@_betainc.defjvp
def _betainc_jvp(primals, tangents):
    a, b, x = primals
    da, db, dx = tangents
    value = betainc(a, b, x)
    step_a, step_b = _shape_step(a), _shape_step(b)
    slope_a = (betainc(a + step_a, b, x) - betainc(a - step_a, b, x)) / (2.0 * step_a)
    slope_b = (betainc(a, b + step_b, x) - betainc(a, b - step_b, x)) / (2.0 * step_b)
    inside = jnp.clip(x, 1e-30, 1.0 - 1e-7)
    log_density = (a - 1.0) * jnp.log(inside) + (b - 1.0) * jnp.log1p(-inside) - betaln(a, b)
    slope_x = jnp.exp(jnp.minimum(log_density, 80.0))
    return value, slope_a * da + slope_b * db + slope_x * dx


class Normal(Distribution):
    """Normal with location ``loc`` and standard deviation ``scale``."""

    loc: jax.Array
    scale: jax.Array
    reparameterized: ClassVar[bool] = True
    atomless: ClassVar[bool] = True

    def __init__(self, loc, scale):
        self.loc, self.scale = _broadcast(loc, scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    def log_prob(self, value):
        return _normal_log_prob(value, self.loc, self.scale)

    def log_cdf(self, value):
        return log_ndtr((value - self.loc) / self.scale)

    def log_survival(self, value):
        return log_ndtr((self.loc - value) / self.scale)

    def convolve_normal(self, scale):
        return Normal(self.loc, jnp.sqrt(jnp.square(self.scale) + jnp.square(jnp.asarray(scale, dtype=self.scale.dtype))))

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, tuple(sample_shape) + self.loc.shape, dtype=self.loc.dtype)
        return self.loc + self.scale * noise

    def mean(self):
        return self.loc

    def variance(self):
        return jnp.square(self.scale)

    def cdf(self, value):
        return ndtr((value - self.loc) / self.scale)

    def quantile(self, q):
        return self.loc + self.scale * ndtri(jnp.asarray(q, dtype=self.loc.dtype))


class StudentT(Distribution):
    """Student-t with location, scale, and degrees of freedom ``df``."""

    loc: jax.Array
    scale: jax.Array
    df: jax.Array
    reparameterized: ClassVar[bool] = True
    atomless: ClassVar[bool] = True

    def __init__(self, loc, scale, df):
        self.loc, self.scale, self.df = _broadcast(loc, scale, df)

    @property
    def batch_shape(self):
        return self.loc.shape

    def log_prob(self, value):
        from ashlar.distributional import student_t_log_prob

        return student_t_log_prob(value, self.loc, self.scale, self.df)

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.loc.shape
        noise = jax.random.t(key, jnp.broadcast_to(self.df, shape), shape, dtype=self.loc.dtype)
        return self.loc + self.scale * noise

    def mean(self):
        return jnp.where(self.df > 1, self.loc, jnp.nan)

    def variance(self):
        return jnp.where(self.df > 2, jnp.square(self.scale) * self.df / (self.df - 2), jnp.inf)

    def _tail(self, value):
        """The standardized value and ``P(T > |t|)``, which is accurate even when it is tiny."""
        t = (value - self.loc) / self.scale
        return t, 0.5 * _betainc(0.5 * self.df, jnp.full_like(self.df, 0.5), self.df / (self.df + t * t))

    def cdf(self, value):
        t, tail = self._tail(value)
        return jnp.where(t > 0, 1.0 - tail, tail)

    def log_cdf(self, value):
        t, tail = self._tail(value)
        return jnp.where(t < 0, _safe_log(tail), jnp.log1p(-jnp.minimum(tail, 0.5)))

    def log_survival(self, value):
        t, tail = self._tail(value)
        return jnp.where(t > 0, _safe_log(tail), jnp.log1p(-jnp.minimum(tail, 0.5)))

    def convolve_normal(self, scale):
        return NormalNoise(self, scale)


class Mixture(Distribution):
    """Finite mixture over the last batch axis of ``components``.

    ``logits`` has shape ``batch_shape + (component,)`` and the components have
    batch shape ``batch_shape + (component,)``. An estimator's ensemble is this
    class with equal logits, one component per member.
    """

    components: Distribution
    logits: jax.Array

    def __init__(self, components: Distribution, logits=None):
        shape = components.batch_shape
        if not shape:
            raise ShapeError("mixture components need a component axis at the end of their batch shape.")
        if logits is None:
            logits = jnp.zeros(shape, dtype=jnp.float32)
        logits = jnp.asarray(logits)
        if logits.shape != shape:
            logits = jnp.broadcast_to(logits, shape)
        self.components = components
        self.logits = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)

    @property
    def batch_shape(self):
        return self.logits.shape[:-1]

    @property
    def event_shape(self):
        return self.components.event_shape

    @property
    def support(self):
        return self.components.support

    @property
    def defective(self):
        return self.components.defective

    def _expand(self, value):
        """Add the component axis to a value shaped ``... + batch_shape + event_shape``."""
        value = jnp.asarray(value)
        return jnp.expand_dims(value, value.ndim - len(self.event_shape))

    def _weights_for_event(self):
        weight = jnp.exp(self.logits)
        return weight.reshape(weight.shape + (1,) * len(self.event_shape))

    def log_prob(self, value):
        return jax.nn.logsumexp(self.components.log_prob(self._expand(value)) + self.logits, axis=-1)

    def cdf(self, value):
        return jnp.sum(jnp.exp(self.logits) * self.components.cdf(self._expand(value)), axis=-1)

    def cdf_left(self, value):
        return jnp.sum(jnp.exp(self.logits) * self.components.cdf_left(self._expand(value)), axis=-1)

    def _log_mix(self, values):
        return jax.nn.logsumexp(values + self.logits, axis=-1)

    def log_cdf(self, value):
        return self._log_mix(self.components.log_cdf(self._expand(value)))

    def log_survival(self, value):
        return self._log_mix(self.components.log_survival(self._expand(value)))

    def log_cdf_left(self, value):
        return self._log_mix(self.components.log_cdf_left(self._expand(value)))

    def log_survival_left(self, value):
        return self._log_mix(self.components.log_survival_left(self._expand(value)))

    def log_interval(self, lower, upper):
        """Each component's interval probability, mixed: accurate even where the mixture's own CDF is flat."""
        lower, upper = _bounds(lower, upper)
        return self._log_mix(self.components.log_interval(self._expand(lower), self._expand(upper)))

    def cdf_interval(self, lower, upper):
        lower, upper = _bounds(lower, upper)
        low, high = self.components.cdf_interval(self._expand(lower), self._expand(upper))
        weight = jnp.exp(self.logits)
        return jnp.sum(weight * low, axis=-1), jnp.sum(weight * high, axis=-1)

    def log_box(self, lower, upper, points=None):
        """Each component's box likelihood, mixed by its weight."""
        lower, upper = _bounds(lower, upper)
        return self._log_mix(self.components.log_box(self._expand(lower), self._expand(upper), points))

    def project(self, weights):
        """Each component's weighted sum, mixed with the same weights."""
        if len(self.event_shape) != 1:
            raise NotImplementedError("a weighted sum is for a joint outcome over one event axis.")
        return Mixture(self.components.project(self._expand(jnp.asarray(weights, dtype=jnp.float32))), self.logits)

    def convolve_normal(self, scale):
        return Mixture(self.components.convolve_normal(self._expand(scale)), self.logits)

    def mean(self):
        axis = len(self.batch_shape)
        return jnp.sum(self._weights_for_event() * self.components.mean(), axis=axis)

    def variance(self):
        axis = len(self.batch_shape)
        weight = self._weights_for_event()
        means = self.components.mean()
        mean = jnp.sum(weight * means, axis=axis, keepdims=True)
        within = jnp.sum(weight * self.components.variance(), axis=axis)
        between = jnp.sum(weight * jnp.square(means - mean), axis=axis)
        return within + between

    def covariance(self):
        if len(self.event_shape) != 1:
            raise NotImplementedError("covariance is for a joint outcome over one event axis.")
        axis = len(self.batch_shape)
        weight = jnp.exp(self.logits)[..., None, None]
        means = self.components.mean()
        centered = means - jnp.sum(jnp.exp(self.logits)[..., None] * means, axis=axis, keepdims=True)
        outer = centered[..., :, None] * centered[..., None, :]
        return jnp.sum(weight * (self.components.covariance() + outer), axis=axis)

    def _index(self, index):
        return Mixture(self.components._index(index), self.logits[index])

    def sample(self, key, sample_shape=()):
        sample_shape = tuple(sample_shape)
        choice_key, draw_key = jax.random.split(key)
        choice = jax.random.categorical(choice_key, self.logits, axis=-1, shape=sample_shape + self.batch_shape)
        if getattr(self.components, "shared_parameters", False):
            # Weights shared across rows cannot be gathered per draw: draw every component, then keep the chosen one.
            draws = self.components.sample(draw_key, sample_shape)
            index = choice.reshape(choice.shape + (1,) * (1 + len(self.event_shape)))
            return jnp.take_along_axis(draws, index, axis=len(sample_shape) + len(self.batch_shape))[
                (slice(None),) * (len(sample_shape) + len(self.batch_shape)) + (0,)
            ]
        return self.select(choice, len(sample_shape)).sample(draw_key)

    def select(self, choice, sample_ndim: int = 0) -> Distribution:
        """The component named by ``choice`` (shape ``sample + batch_shape``) for every row."""
        outer = self.batch_shape
        grids = tuple(
            jnp.arange(size).reshape((1,) * (sample_ndim + axis) + (size,) + (1,) * (len(outer) - axis - 1))
            for axis, size in enumerate(outer)
        )
        return jax.tree.map(lambda leaf: leaf[grids + (choice,)], self.components)

    def marginals(self):
        if not self.event_shape:
            return self
        parts = self.components.marginals()
        axis = len(self.batch_shape)
        moved = jax.tree.map(lambda leaf: jnp.moveaxis(leaf, axis, axis + len(self.event_shape)), parts)
        logits = jnp.broadcast_to(
            jnp.expand_dims(self.logits, tuple(range(axis, axis + len(self.event_shape)))),
            moved.batch_shape,
        )
        return Mixture(moved, logits)


def member_mixture(components: Distribution) -> Mixture:
    """The equal-weight mixture of an estimator's members, the last batch axis of ``components``."""
    return Mixture(components)


class AffineTransformed(Distribution):
    """``loc + scale * x`` for ``x`` from ``base``, with ``scale > 0`` per coordinate.

    The estimators use it to report a head trained on a standardized target in
    the units of ``y``. ``loc`` and ``scale`` broadcast to ``batch_shape +
    event_shape``.
    """

    base: Distribution
    loc: jax.Array
    scale: jax.Array

    def __init__(self, base: Distribution, loc, scale):
        shape = tuple(base.batch_shape) + tuple(base.event_shape)
        self.base = base
        self.loc = jnp.broadcast_to(jnp.asarray(loc, dtype=jnp.float32), shape)
        self.scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), shape)

    @property
    def batch_shape(self):
        return self.base.batch_shape

    @property
    def event_shape(self):
        return self.base.event_shape

    @property
    def reparameterized(self):
        return self.base.reparameterized

    @property
    def support(self):
        return self.base.support

    @property
    def defective(self):
        return self.base.defective

    def _log_scale(self):
        log_scale = jnp.log(self.scale)
        for _ in self.event_shape:
            log_scale = jnp.sum(log_scale, axis=-1)
        return log_scale

    def log_prob(self, value):
        return self.base.log_prob((value - self.loc) / self.scale) - self._log_scale()

    def sample(self, key, sample_shape=()):
        return self.loc + self.scale * self.base.sample(key, sample_shape)

    def mean(self):
        return self.loc + self.scale * self.base.mean()

    def variance(self):
        return jnp.square(self.scale) * self.base.variance()

    def cdf(self, value):
        return self.base.cdf((value - self.loc) / self.scale)

    def cdf_left(self, value):
        return self.base.cdf_left((value - self.loc) / self.scale)

    def _inner(self, value):
        return _finite_standardized(value, self.loc, self.scale)

    def log_cdf(self, value):
        return self.base.log_cdf(self._inner(value))

    def log_survival(self, value):
        return self.base.log_survival(self._inner(value))

    def log_cdf_left(self, value):
        return self.base.log_cdf_left(self._inner(value))

    def log_survival_left(self, value):
        return self.base.log_survival_left(self._inner(value))

    def log_interval(self, lower, upper):
        lower, upper = _bounds(lower, upper)
        return self.base.log_interval(self._inner(lower), self._inner(upper))

    def cdf_interval(self, lower, upper):
        lower, upper = _bounds(lower, upper)
        return self.base.cdf_interval(self._inner(lower), self._inner(upper))

    def log_box(self, lower, upper, points=None):
        """The base's box likelihood at the standardized bounds, less the log scale of every exact column."""
        if len(self.event_shape) != 1:
            raise NotImplementedError("a box likelihood is for a joint outcome over one event axis.")
        lower, upper = _bounds(lower, upper)
        exact = (lower == upper) & jnp.isfinite(lower)
        inner = self.base.log_box(self._inner(lower), self._inner(upper), points)
        return inner - jnp.sum(jnp.where(exact, jnp.log(self.scale), 0.0), axis=-1)

    def convolve_normal(self, scale):
        return AffineTransformed(self.base.convolve_normal(jnp.asarray(scale, dtype=jnp.float32) / self.scale), self.loc, self.scale)

    def project(self, weights):
        """``weights · (loc + scale x)`` is ``weights · loc`` plus the base's sum with weights ``weights * scale``."""
        if len(self.event_shape) != 1:
            raise NotImplementedError("a weighted sum is for a joint outcome over one event axis.")
        weights = jnp.asarray(weights, dtype=jnp.float32)
        return AffineTransformed(self.base.project(weights * self.scale), jnp.sum(weights * self.loc, axis=-1), 1.0)

    def quantile(self, q):
        return self.loc + self.scale * self.base.quantile(q)

    def covariance(self):
        cov = self.base.covariance()
        return cov * self.scale[..., :, None] * self.scale[..., None, :]

    def marginals(self):
        if not self.event_shape:
            return self
        return AffineTransformed(self.base.marginals(), self.loc, self.scale)

    def _index(self, index):
        return AffineTransformed(self.base._index(index), self.loc[index], self.scale[index])


class Independent(Distribution):
    """Reinterpret the last batch axis of ``base`` as one joint outcome of independent coordinates.

    ``Independent(joint.marginals())`` keeps every marginal of ``joint`` and
    drops the dependence between them.
    """

    base: Distribution

    def __init__(self, base: Distribution):
        if not base.batch_shape or base.event_shape:
            raise ShapeError("Independent takes a univariate distribution with at least one batch axis.")
        self.base = base

    @property
    def batch_shape(self):
        return self.base.batch_shape[:-1]

    @property
    def event_shape(self):
        return self.base.batch_shape[-1:]

    @property
    def reparameterized(self):
        return self.base.reparameterized

    @property
    def support(self):
        return self.base.support

    def log_prob(self, value):
        return jnp.sum(self.base.log_prob(value), axis=-1)

    def log_box(self, lower, upper, points=None):
        """Each column's own likelihood, summed: independent columns need no integral."""
        del points
        lower, upper = _bounds(lower, upper)
        exact = lower == upper
        point = self.base.log_prob(jnp.where(exact, lower, 0.0))
        coarse = self.base.log_interval(jnp.where(exact, -jnp.inf, lower), jnp.where(exact, jnp.inf, upper))
        return jnp.sum(jnp.where(exact, point, coarse), axis=-1)

    def sample(self, key, sample_shape=()):
        return self.base.sample(key, sample_shape)

    def mean(self):
        return self.base.mean()

    def variance(self):
        return self.base.variance()

    def covariance(self):
        variance = self.variance()
        return variance[..., :, None] * jnp.eye(variance.shape[-1], dtype=variance.dtype)

    def project(self, weights):
        """A weighted sum of independent normal columns is normal; other columns have no closed form."""
        if not isinstance(self.base, Normal):
            raise NotImplementedError(f"a weighted sum of independent {type(self.base).__name__} columns has no closed form.")
        weights = jnp.asarray(weights, dtype=jnp.float32)
        spread = jnp.sqrt(jnp.sum(jnp.square(weights * self.base.scale), axis=-1))
        return Normal(jnp.sum(weights * self.base.loc, axis=-1), spread)

    def marginals(self):
        return self.base

    def _index(self, index):
        return Independent(self.base._index(index))


# Quasi-random points per row for a box likelihood that integrates: scoring and ``log_box`` by default.
BOX_POINTS = 256
# Past this many standard deviations a truncated draw's CDF underflows in float32, so it sits at the near end.
_DRAW_LIMIT = 10.0


def _primes(count: int) -> list[int]:
    found: list[int] = []
    candidate = 2
    while len(found) < count:
        if all(candidate % prime for prime in found if prime * prime <= candidate):
            found.append(candidate)
        candidate += 1
    return found


def box_points(count: int, dim: int) -> jax.Array:
    """``count`` quasi-random points in ``(0, 1)^dim``: the Halton sequence, centred.

    Coordinate ``j`` is the radical inverse in the ``j``-th prime base ``b``,
    moved up by half its finest complete cell, ``1 / (2 b^m)`` with ``b^m <=
    count``, modulo 1, so that the first coordinate of ``2^m`` points is the
    midpoint rule. A box likelihood that integrates over a row's coarse columns
    averages over these, one coordinate per coarse column in the order it
    visits them. The same points serve every row, which keeps a training
    objective deterministic.
    """
    count, dim = int(count), int(dim)
    if count < 1 or dim < 1:
        raise ShapeError(f"box points need a positive count and dimension; received {count} and {dim}.")
    import numpy as np

    points = np.empty((count, dim), dtype=np.float64)
    for axis, base in enumerate(_primes(dim)):
        cell = 1.0
        while cell * base <= count:
            cell *= base
        for row in range(count):
            index, fraction, value = row, 1.0 / base, 0.0
            while index:
                index, digit = divmod(index, base)
                value += digit * fraction
                fraction /= base
            points[row, axis] = (value + 0.5 / cell) % 1.0
    return jnp.asarray(points, dtype=jnp.float32)


def _check_box_points(points, d: int):
    points = box_points(BOX_POINTS, d) if points is None else jnp.asarray(points, dtype=jnp.float32)
    if points.ndim != 2 or points.shape[1] < d:
        raise ShapeError(f"box points must be (count, {d}) or wider; received {points.shape}.")
    return points


def _standard_log_interval(alpha, beta):
    """``log P(alpha <= Z <= beta)`` for a standard normal ``Z``, accurate in either tail."""
    return Normal(0.0, 1.0).log_interval(alpha, beta)


def _truncated_standard_normal(alpha, beta, log_mass, u):
    """The ``u`` quantile of a standard normal truncated to ``[alpha, beta]``, with its implicit gradient.

    ``log_mass`` is ``log P(alpha <= Z <= beta)``. An interval above zero is
    reflected below it first, so the CDF at its near end is not close to 1,
    and the inverse CDF works from ``log P(Z < near end)``. The value comes
    from ``ndtri``; the gradient is that of ``Φ(draw) = Φ(alpha) + u (Φ(beta)
    - Φ(alpha))``, which stays finite where ``ndtri``'s own would not. An
    interval more than ten standard deviations out puts the draw at its
    near end.
    """
    flip = alpha > 0
    low = jnp.where(flip, -beta, alpha)
    high = jnp.where(flip, -alpha, beta)
    has_low = jnp.isfinite(low)
    log_below = jnp.where(has_low, log_ndtr(jnp.where(has_low, low, 0.0)), -jnp.inf)
    log_target = jnp.logaddexp(log_below, jnp.log(u) + log_mass)
    far = high < -_DRAW_LIMIT
    target = jnp.clip(jnp.exp(jnp.where(far, _LOG_HALF, log_target)), _TINY, 1.0 - 1e-7)
    value = jax.lax.stop_gradient(jnp.clip(ndtri(target), low, high))
    log_density = -0.5 * (math.log(2 * math.pi) + jnp.square(value))
    slope = jax.lax.stop_gradient(jnp.exp(jnp.minimum(log_target - log_density, 30.0)))
    moved = value + slope * (log_target - jax.lax.stop_gradient(log_target))
    draw = jnp.where(far, jnp.where(jnp.isfinite(high), high, 0.0), moved)
    return jnp.where(flip, -draw, draw)


def _box_order(exact, marginal_log_mass):
    """Exact columns first, then the coarse ones from the least likely: the order a box likelihood visits them.

    Visiting the exact columns first makes their part exact. Among coarse
    columns, the least likely first is Genz's (1992) ordering, which puts
    the most of the probability in the terms computed without draws.
    """
    key = jax.lax.stop_gradient(jnp.where(exact, -jnp.inf, marginal_log_mass))
    return jnp.argsort(key, axis=-1, stable=True)


def _coarse_points(points, exact):
    """Each column's uniform at every point, ``(count,) + exact.shape``: coarse column ``j`` reads coordinate ``j``."""
    index = jnp.clip(jnp.cumsum(~exact, axis=-1) - 1, 0, points.shape[1] - 1)
    return points[:, index]


class LowRankMultivariateNormal(Distribution):
    """Joint normal over ``d`` coordinates with covariance ``W Wᵀ + diag(D)``.

    ``loc`` is ``(..., d)``, ``cov_factor`` (``W``) is ``(..., d, rank)``, and
    ``cov_diag`` (``D``, variances) is ``(..., d)``. The density uses the
    Woodbury identity and the matrix determinant lemma, so it costs ``O(d r²)``
    per row instead of ``O(d³)``. A rank of 0 is independent coordinates.
    """

    loc: jax.Array
    cov_factor: jax.Array
    cov_diag: jax.Array
    reparameterized: ClassVar[bool] = True

    def __init__(self, loc, cov_factor, cov_diag):
        loc = jnp.asarray(loc, dtype=jnp.float32)
        cov_factor = jnp.asarray(cov_factor, dtype=jnp.float32)
        cov_diag = jnp.asarray(cov_diag, dtype=jnp.float32)
        if loc.ndim < 1 or cov_factor.ndim < 2 or cov_factor.shape[-2] != loc.shape[-1]:
            raise ShapeError(
                f"loc must be (..., d) and cov_factor (..., d, rank); received {loc.shape} and {cov_factor.shape}."
            )
        batch = jnp.broadcast_shapes(loc.shape[:-1], cov_factor.shape[:-2], cov_diag.shape[:-1])
        d, rank = loc.shape[-1], cov_factor.shape[-1]
        self.loc = jnp.broadcast_to(loc, batch + (d,))
        self.cov_factor = jnp.broadcast_to(cov_factor, batch + (d, rank))
        self.cov_diag = jnp.broadcast_to(cov_diag, batch + (d,))

    @property
    def batch_shape(self):
        return self.loc.shape[:-1]

    @property
    def event_shape(self):
        return self.loc.shape[-1:]

    @property
    def rank(self) -> int:
        return int(self.cov_factor.shape[-1])

    def log_prob(self, value):
        diff = jnp.asarray(value) - self.loc
        d = self.loc.shape[-1]
        log_det = jnp.sum(jnp.log(self.cov_diag), axis=-1)
        mahalanobis = jnp.sum(jnp.square(diff) / self.cov_diag, axis=-1)
        if self.rank:
            scaled = self.cov_factor / self.cov_diag[..., :, None]
            capacitance = jnp.eye(self.rank, dtype=diff.dtype) + jnp.einsum("...dr,...ds->...rs", self.cov_factor, scaled)
            chol = jnp.linalg.cholesky(capacitance)
            log_det = log_det + 2.0 * jnp.sum(jnp.log(jnp.diagonal(chol, axis1=-2, axis2=-1)), axis=-1)
            projected = jnp.einsum("...dr,...d->...r", jnp.broadcast_to(scaled, diff.shape + (self.rank,)), diff)
            chol = jnp.broadcast_to(chol, projected.shape + (self.rank,))
            solved = jax.scipy.linalg.solve_triangular(chol, projected[..., None], lower=True)[..., 0]
            mahalanobis = mahalanobis - jnp.sum(jnp.square(solved), axis=-1)
        return -0.5 * (d * math.log(2 * math.pi) + log_det + mahalanobis)

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.loc.shape
        factor_key, diag_key = jax.random.split(key)
        noise = jax.random.normal(diag_key, shape, dtype=self.loc.dtype)
        draw = self.loc + jnp.sqrt(self.cov_diag) * noise
        if self.rank:
            common = jax.random.normal(factor_key, tuple(sample_shape) + self.batch_shape + (self.rank,), dtype=self.loc.dtype)
            draw = draw + jnp.einsum("...dr,...r->...d", self.cov_factor, common)
        return draw

    def mean(self):
        return self.loc

    def variance(self):
        return self.cov_diag + jnp.sum(jnp.square(self.cov_factor), axis=-1)

    def covariance(self):
        diag = self.cov_diag[..., :, None] * jnp.eye(self.loc.shape[-1], dtype=self.loc.dtype)
        return diag + jnp.einsum("...dr,...er->...de", self.cov_factor, self.cov_factor)

    def correlation(self):
        cov = self.covariance()
        sd = jnp.sqrt(jnp.diagonal(cov, axis1=-2, axis2=-1))
        return cov / (sd[..., :, None] * sd[..., None, :])

    def marginals(self):
        return Normal(self.loc, jnp.sqrt(self.variance()))

    def project(self, weights):
        """``weights · Y`` is normal, with variance ``wᵀ D w + |Wᵀ w|²``."""
        weights = jnp.asarray(weights, dtype=self.loc.dtype)
        weights = jnp.broadcast_to(weights, jnp.broadcast_shapes(weights.shape, self.loc.shape))
        variance = jnp.sum(jnp.square(weights) * self.cov_diag, axis=-1)
        if self.rank:
            factor = jnp.broadcast_to(self.cov_factor, weights.shape + (self.rank,))
            variance = variance + jnp.sum(jnp.square(jnp.einsum("...d,...dr->...r", weights, factor)), axis=-1)
        return Normal(jnp.sum(weights * self.loc, axis=-1), jnp.sqrt(variance))

    def log_box(self, lower, upper, points=None):
        """The GHK simulator (Geweke, Hajivassiliou, Keane), run on the ``rank`` shared factors.

        Given the factors the columns are independent, so each column is one
        Kalman update of the factors' normal posterior, ``O(rank²)``, with a
        variance that never falls below the column's own ``D``. Columns are
        visited exact first, which is the exact marginal density of the exact
        columns, then the coarse ones from the least likely. The first coarse
        column's probability is exact given the exact ones; each later one is
        averaged over truncated draws of the columns before it, one per point.
        So a row with at most one coarse column is closed form, and a row with
        several is a quasi-Monte Carlo estimate whose log sits slightly low,
        by about half its relative variance.
        """
        d, rank = self.loc.shape[-1], self.rank
        lower, upper = _bounds(lower, upper)
        batch = jnp.broadcast_shapes(lower.shape[:-1], upper.shape[:-1], self.batch_shape)
        lower, upper, loc, diag = (jnp.broadcast_to(part, batch + (d,)) for part in (lower, upper, self.loc, self.cov_diag))
        factor = jnp.broadcast_to(self.cov_factor, batch + (d, rank))
        points = _check_box_points(points, d)
        exact = (lower == upper) & jnp.isfinite(lower)
        spread = jnp.sqrt(diag + jnp.sum(jnp.square(factor), axis=-1))
        marginal = Normal(loc, spread).log_interval(jnp.where(exact, -jnp.inf, lower), jnp.where(exact, jnp.inf, upper))
        order = _box_order(exact, marginal)
        lower, upper, loc, diag, exact = (jnp.take_along_axis(part, order, axis=-1) for part in (lower, upper, loc, diag, exact))
        factor = jnp.take_along_axis(factor, order[..., None], axis=-2)
        uniform = _coarse_points(points, exact)
        count = points.shape[0]
        mean = jnp.zeros((count,) + batch + (rank,), dtype=jnp.float32)
        cov = jnp.broadcast_to(jnp.eye(rank, dtype=jnp.float32), batch + (rank, rank))
        total = jnp.zeros((count,) + batch, dtype=jnp.float32)
        for step in range(d):
            weight = factor[..., step, :]
            projected = jnp.einsum("...rs,...s->...r", cov, weight)
            variance = diag[..., step] + jnp.maximum(jnp.sum(weight * projected, axis=-1), 0.0)
            sd = jnp.sqrt(variance)
            centre = loc[..., step] + jnp.einsum("n...r,...r->n...", mean, weight)
            is_exact = exact[..., step]
            low, high = lower[..., step], upper[..., step]
            # An exact column: its density given the columns before it.
            z = (jnp.where(is_exact, low, 0.0) - centre) / sd
            point = -0.5 * (math.log(2 * math.pi) + jnp.square(z)) - jnp.log(sd)
            # A coarse one: the probability of its interval, then a draw inside it for the columns after it.
            alpha = _finite_standardized(jnp.where(is_exact, -jnp.inf, low), centre, sd)
            beta = _finite_standardized(jnp.where(is_exact, jnp.inf, high), centre, sd)
            log_mass = _standard_log_interval(alpha, beta)
            draw = _truncated_standard_normal(alpha, beta, log_mass, uniform[..., step])
            total = total + jnp.where(is_exact, point, log_mass)
            gain = projected / variance[..., None]
            mean = mean + (sd * jnp.where(is_exact, z, draw))[..., None] * gain
            cov = cov - variance[..., None, None] * gain[..., :, None] * gain[..., None, :]
        return jax.nn.logsumexp(total, axis=0) - math.log(count)


class Flow(Distribution):
    """The ``flow`` head: a standard normal pushed through a rational-quadratic spline, then shifted and scaled.

    ``y = shift + scale * spline(z)``. ``params`` is ``batch_shape +
    (parameters,)``: the unconstrained spline of ``ashlar.distributional``,
    then the shift and the raw scale, one of each per row, so every row has
    its own spread and its own tails. A bare spline (``flow_parameter_count()``
    values, the layout saved before the shift and scale) has shift 0 and
    scale 1. The map is monotone, so the quantile at ``q`` is the map at the
    normal quantile of ``q``. Mean and variance are exact normal integrals
    on the linear tails plus a 512-point midpoint rule inside the spline's
    range.
    """

    params: jax.Array
    reparameterized: ClassVar[bool] = True
    atomless: ClassVar[bool] = True

    def __init__(self, params):
        self.params = jnp.asarray(params, dtype=jnp.float32)

    @property
    def batch_shape(self):
        return self.params.shape[:-1]

    def _params_for(self, value):
        shape = jnp.broadcast_shapes(jnp.shape(value), self.batch_shape)
        return jnp.broadcast_to(self.params, shape + self.params.shape[-1:]), jnp.broadcast_to(value, shape)

    def log_prob(self, value):
        from ashlar.distributional import flow_log_density

        params, value = self._params_for(jnp.asarray(value, dtype=self.params.dtype))
        return flow_log_density(params, value)

    def _forward(self, base):
        from ashlar.distributional import flow_forward

        params, base = self._params_for(base)
        return flow_forward(params, base)

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, tuple(sample_shape) + self.batch_shape, dtype=self.params.dtype)
        return self._forward(noise)

    def _base(self, value):
        """The standard-normal value that the flow maps to ``value``."""
        from ashlar.distributional import flow_inverse

        params, value = self._params_for(jnp.asarray(value, dtype=self.params.dtype))
        base, _logdet = flow_inverse(params, value)
        return base

    def cdf(self, value):
        return ndtr(self._base(value))

    def log_cdf(self, value):
        return log_ndtr(self._base(value))

    def log_survival(self, value):
        return log_ndtr(-self._base(value))

    def quantile(self, q):
        return self._forward(ndtri(jnp.asarray(q, dtype=self.params.dtype)))

    def _moments(self):
        """First two moments. The spline is the identity outside ``[-bound, bound]``,
        so the tails are closed-form normal integrals; the inside is a midpoint rule.
        The shift and scale then move them."""
        from ashlar.distributional import FLOW_BOUND, flow_affine, flow_forward

        spline, shift, scale = flow_affine(self.params)
        bound = float(FLOW_BOUND)
        width = 2.0 * bound / QUADRATURE_NODES
        nodes = -bound + width * (jnp.arange(QUADRATURE_NODES, dtype=self.params.dtype) + 0.5)
        weights = width * jnp.exp(-0.5 * jnp.square(nodes)) / math.sqrt(2 * math.pi)
        grid = jnp.broadcast_to(
            nodes.reshape((-1,) + (1,) * len(self.batch_shape)), (QUADRATURE_NODES,) + tuple(self.batch_shape)
        )
        values = flow_forward(jnp.broadcast_to(spline, grid.shape + spline.shape[-1:]), grid)
        weights = weights.reshape((-1,) + (1,) * len(self.batch_shape))
        density = math.exp(-0.5 * bound * bound) / math.sqrt(2 * math.pi)
        tail_mass = 0.5 * math.erfc(bound / math.sqrt(2.0))
        first = jnp.sum(weights * values, axis=0)
        second = jnp.sum(weights * jnp.square(values), axis=0) + 2.0 * (bound * density + tail_mass)
        spread = jnp.maximum(second - jnp.square(first), 0.0)
        return shift + scale * first, jnp.square(scale) * spread

    def mean(self):
        return self._moments()[0]

    def variance(self):
        return self._moments()[1]

    def convolve_normal(self, scale):
        return NormalNoise(self, scale)


class Histogram(Distribution):
    """The ``histogram`` head: bin probabilities over fixed edges on the standardized axis.

    The density is flat inside each bin. ``log_prob`` reads a value outside the
    edges in the nearest end bin, as the training loss does, while draws, the
    CDF, and the quantiles stay inside the edges.
    """

    logits: jax.Array
    atomless: ClassVar[bool] = True

    def __init__(self, logits):
        self.logits = jnp.asarray(logits, dtype=jnp.float32)

    @property
    def batch_shape(self):
        return self.logits.shape[:-1]

    def _edges(self):
        from ashlar.distributional import histogram_edges

        return histogram_edges(self.logits.dtype, self.logits.shape[-1])

    def _probabilities(self):
        return jax.nn.softmax(self.logits, axis=-1)

    def log_prob(self, value):
        from ashlar.distributional import histogram_bin

        value = jnp.asarray(value, dtype=self.logits.dtype)
        shape = jnp.broadcast_shapes(value.shape, self.batch_shape)
        logits = jnp.broadcast_to(self.logits, shape + self.logits.shape[-1:])
        index = histogram_bin(jnp.broadcast_to(value, shape), self.logits.shape[-1])
        edges = self._edges()
        chosen = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1)[..., 0]
        return chosen - jnp.log(edges[1] - edges[0])

    def sample(self, key, sample_shape=()):
        from ashlar.distributional import sample_values

        draws = sample_values("histogram", self.logits.reshape((-1, 1, self.logits.shape[-1])), key, _count(sample_shape))
        return draws.reshape(tuple(sample_shape) + self.batch_shape)

    def mean(self):
        edges = self._edges()
        return self._probabilities() @ (0.5 * (edges[:-1] + edges[1:]))

    def variance(self):
        edges = self._edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]
        second = self._probabilities() @ (jnp.square(centers) + jnp.square(width) / 12.0)
        return second - jnp.square(self.mean())

    def cdf(self, value):
        edges = self._edges()
        value = jnp.asarray(value, dtype=self.logits.dtype)
        cumulative = jnp.concatenate(
            [jnp.zeros(self.batch_shape + (1,), dtype=self.logits.dtype), jnp.cumsum(self._probabilities(), axis=-1)],
            axis=-1,
        )
        fraction = jnp.clip((value[..., None] - edges[:-1]) / (edges[1:] - edges[:-1]), 0.0, 1.0)
        return jnp.sum(fraction * (cumulative[..., 1:] - cumulative[..., :-1]), axis=-1)

    def _inside(self, lower, upper):
        """Interval bounds held inside the edges. The head has no mass past them, so a bound beyond
        an end bin's center is read at that center, and an interval wholly out there reads the
        outer half of the end bin, as ``log_prob`` reads a value out there in the end bin."""
        edges = self._edges()
        half = 0.5 * (edges[1] - edges[0])
        lower, upper = _bounds(lower, upper)
        return jnp.clip(lower, edges[0], edges[-1] - half), jnp.clip(upper, edges[0] + half, edges[-1])

    def log_interval(self, lower, upper):
        empty = _empty(*_bounds(lower, upper))
        low, high = self._inside(lower, upper)
        edges = self._edges()
        overlap = jnp.minimum(high[..., None], edges[1:]) - jnp.maximum(low[..., None], edges[:-1])
        fraction = jnp.clip(overlap / (edges[1:] - edges[:-1]), 0.0, 1.0)
        probability = jnp.sum(fraction * self._probabilities(), axis=-1)
        return jnp.where(empty, -jnp.inf, _safe_log(probability))

    def log_survival(self, value):
        edges = self._edges()
        value = jnp.asarray(value, dtype=self.logits.dtype)
        above = jnp.clip((edges[1:] - value[..., None]) / (edges[1:] - edges[:-1]), 0.0, 1.0)
        return _safe_log(jnp.sum(above * self._probabilities(), axis=-1))

    def cdf_interval(self, lower, upper):
        low, high = self._inside(lower, upper)
        return self.cdf(low), self.cdf(high)

    def quantile(self, q):
        edges = self._edges()
        q = jnp.asarray(q, dtype=self.logits.dtype)
        shape = jnp.broadcast_shapes(q.shape, self.batch_shape)
        probabilities = jnp.broadcast_to(self._probabilities(), shape + self.logits.shape[-1:])
        upper = jnp.cumsum(probabilities, axis=-1)
        q = jnp.broadcast_to(q, shape)
        index = jnp.clip(jnp.sum(upper < q[..., None], axis=-1), 0, self.logits.shape[-1] - 1)
        mass = jnp.take_along_axis(probabilities, index[..., None], axis=-1)[..., 0]
        below = jnp.take_along_axis(upper, index[..., None], axis=-1)[..., 0] - mass
        fraction = jnp.clip((q - below) / jnp.maximum(mass, 1e-30), 0.0, 1.0)
        left = edges[index]
        return left + fraction * (edges[1] - edges[0])

    def convolve_normal(self, scale):
        return NormalNoise(self, scale)


def _count(sample_shape) -> int:
    count = 1
    for size in tuple(sample_shape):
        count *= int(size)
    return count


class _Count(Distribution):
    """Shared code of the count families, each with an optional structural zero."""

    support: ClassVar[str] = "nonnegative_integer"

    def _log_count(self, value):
        raise NotImplementedError

    def _count_cdf(self, k):
        raise NotImplementedError

    def _count_variance(self):
        raise NotImplementedError

    def log_prob(self, value):
        from ashlar.distributional import zero_inflated_log_prob

        value = jnp.asarray(value, dtype=self.mean_count.dtype)
        return zero_inflated_log_prob(value, self._log_count(value), self.zero)

    def cdf(self, value):
        k = jnp.floor(jnp.asarray(value, dtype=self.mean_count.dtype))
        inside = self.zero + (1.0 - self.zero) * self._count_cdf(jnp.maximum(k, 0.0))
        return jnp.where(k < 0, 0.0, jnp.clip(inside, 0.0, 1.0))

    def _log_count_cdf(self, k):
        raise NotImplementedError

    def _log_count_survival(self, k):
        raise NotImplementedError

    def log_cdf(self, value):
        k = jnp.floor(jnp.asarray(value, dtype=self.mean_count.dtype))
        count = jnp.log1p(-jnp.minimum(self.zero, 1.0 - 1e-7)) + self._log_count_cdf(jnp.maximum(k, 0.0))
        return jnp.where(k < 0, -jnp.inf, jnp.logaddexp(_safe_log(self.zero), count))

    def log_survival(self, value):
        k = jnp.floor(jnp.asarray(value, dtype=self.mean_count.dtype))
        count = jnp.log1p(-jnp.minimum(self.zero, 1.0 - 1e-7)) + self._log_count_survival(jnp.maximum(k, 0.0))
        return jnp.where(k < 0, 0.0, count)

    def mean(self):
        return (1.0 - self.zero) * self.mean_count

    def variance(self):
        keep = 1.0 - self.zero
        return keep * (self._count_variance() + jnp.square(self.mean_count)) - jnp.square(keep * self.mean_count)


class Poisson(_Count):
    """Poisson with mean ``mean_count`` and structural-zero probability ``zero``."""

    mean_count: jax.Array
    zero: jax.Array

    def __init__(self, mean_count, zero=0.0):
        self.mean_count, self.zero = _broadcast(mean_count, zero)

    @property
    def batch_shape(self):
        return self.mean_count.shape

    def _log_count(self, value):
        from ashlar.distributional import poisson_log_prob

        return poisson_log_prob(value, self.mean_count)

    def _count_cdf(self, k):
        return gammaincc(k + 1.0, self.mean_count)

    def _log_count_cdf(self, k):
        # The mass at k is a lower bound that stays finite where the CDF underflows.
        return _log_or(gammaincc(k + 1.0, self.mean_count), self._log_count(k))

    def _log_count_survival(self, k):
        return _log_or(gammainc(k + 1.0, self.mean_count), self._log_count(k + 1.0))

    def _count_variance(self):
        return self.mean_count

    def sample(self, key, sample_shape=()):
        from ashlar.distributional import sample_values

        draws = sample_values("poisson", (self.mean_count, self.zero), key, _count(sample_shape))
        return draws.reshape(tuple(sample_shape) + self.batch_shape)


class NegativeBinomial(_Count):
    """Negative binomial with mean ``mean_count``, variance ``mean + dispersion * mean²``, and structural zero ``zero``."""

    mean_count: jax.Array
    dispersion: jax.Array
    zero: jax.Array

    def __init__(self, mean_count, dispersion, zero=0.0):
        self.mean_count, self.dispersion, self.zero = _broadcast(mean_count, dispersion, zero)

    @property
    def batch_shape(self):
        return self.mean_count.shape

    def _log_count(self, value):
        from ashlar.distributional import negative_binomial_log_prob

        return negative_binomial_log_prob(value, self.mean_count, self.dispersion)

    def _count_cdf(self, k):
        size = 1.0 / self.dispersion
        return _broadcast_betainc(size, k + 1.0, size / (size + self.mean_count))

    def _log_count_cdf(self, k):
        size = 1.0 / self.dispersion
        return _log_or(_broadcast_betainc(size, k + 1.0, size / (size + self.mean_count)), self._log_count(k))

    def _log_count_survival(self, k):
        size = 1.0 / self.dispersion
        below = _broadcast_betainc(k + 1.0, size, self.mean_count / (size + self.mean_count))
        return _log_or(below, self._log_count(k + 1.0))

    def _count_variance(self):
        return self.mean_count + self.dispersion * jnp.square(self.mean_count)

    def sample(self, key, sample_shape=()):
        from ashlar.distributional import sample_values

        decoded = (self.mean_count, self.dispersion, self.zero)
        draws = sample_values("negative_binomial", decoded, key, _count(sample_shape))
        return draws.reshape(tuple(sample_shape) + self.batch_shape)


class QuantileLevels(Distribution):
    """The ``quantile`` head: three trained levels per row and nothing between them.

    ``values`` is ``batch_shape + (level,)``. ``quantile(q)`` answers only at
    the trained ``levels``; there is no density, CDF, or sampler.
    """

    values: jax.Array
    levels: tuple[float, ...] = field(static=True)

    @property
    def batch_shape(self):
        return self.values.shape[:-1]

    def quantile(self, q):
        try:
            requested = [float(level) for level in jnp.ravel(jnp.asarray(q)).tolist()]
        except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError) as exc:
            raise NotImplementedError("the quantile head answers at its trained levels, given as Python numbers.") from exc
        missing = [level for level in requested if not any(abs(level - known) < 1e-6 for known in self.levels)]
        if missing:
            raise NotImplementedError(f"the quantile head was trained at {self.levels} only; received {missing}.")
        q = jnp.asarray(q)
        index = jnp.argmin(jnp.abs(q[..., None] - jnp.asarray(self.levels)), axis=-1)
        shape = jnp.broadcast_shapes(q.shape, self.batch_shape)
        values = jnp.broadcast_to(self.values, shape + self.values.shape[-1:])
        return jnp.take_along_axis(values, jnp.broadcast_to(index, shape)[..., None], axis=-1)[..., 0]

    def median(self):
        return self.quantile(0.5)


class Categorical(Distribution):
    """One class out of ``logits.shape[-1]``. Values are class indices, as in ``classes_`` order."""

    logits: jax.Array
    support: ClassVar[str] = "category"

    def __init__(self, logits=None, probs=None):
        if (logits is None) == (probs is None):
            raise AshlarError("pass exactly one of logits and probs.")
        if probs is not None:
            logits = jnp.log(jnp.clip(jnp.asarray(probs, dtype=jnp.float32), 1e-30))
        self.logits = jax.nn.log_softmax(jnp.asarray(logits, dtype=jnp.float32), axis=-1)

    @property
    def batch_shape(self):
        return self.logits.shape[:-1]

    @property
    def probs(self):
        return jnp.exp(self.logits)

    def log_prob(self, value):
        value = jnp.asarray(value, dtype=jnp.int32)
        shape = jnp.broadcast_shapes(value.shape, self.batch_shape)
        logits = jnp.broadcast_to(self.logits, shape + self.logits.shape[-1:])
        return jnp.take_along_axis(logits, jnp.broadcast_to(value, shape)[..., None], axis=-1)[..., 0]

    def sample(self, key, sample_shape=()):
        return jax.random.categorical(key, self.logits, axis=-1, shape=tuple(sample_shape) + self.batch_shape)

    def entropy(self):
        return -jnp.sum(self.probs * self.logits, axis=-1)

    def quantile(self, q):
        raise NotImplementedError("classes have no order, so a categorical outcome has no quantile.")


def _bisection_quantile(dist: Distribution, q):
    """Quantile of a continuous univariate distribution by bisection on its CDF.

    The bracket is Cantelli's bound around the mean, so it holds for any
    distribution with finite variance. The result then takes one Newton step
    whose value is already the bisection answer, so gradients with respect to
    the parameters and to ``q`` are the implicit-function ones.
    """
    q = jnp.asarray(q, dtype=jnp.float32)
    frozen = jax.lax.stop_gradient(dist)
    shape = jnp.broadcast_shapes(q.shape, frozen.batch_shape)
    level = jnp.broadcast_to(jax.lax.stop_gradient(q), shape)
    inner = jnp.clip(level, 1e-7, 1.0 - 1e-7)
    mean = jnp.broadcast_to(frozen.mean(), shape)
    sd = jnp.broadcast_to(frozen.stddev(), shape)
    low = mean - sd * jnp.sqrt((1.0 - inner) / inner) - 1e-6
    high = mean + sd * jnp.sqrt(inner / (1.0 - inner)) + 1e-6

    def step(_, bounds):
        low, high = bounds
        middle = 0.5 * (low + high)
        above = frozen.cdf(middle) >= level
        return jnp.where(above, low, middle), jnp.where(above, middle, high)

    low, high = jax.lax.fori_loop(0, _BISECTION_STEPS, step, (low, high))
    x0 = jax.lax.stop_gradient(0.5 * (low + high))
    try:
        density = jnp.exp(dist.log_prob(x0))
    except NotImplementedError:
        # No density, as for draws: the bisection answer, with no gradient.
        refined = x0
    else:
        gap = dist.cdf(x0) - q
        newton = jnp.where(density > 1e-12, gap / jnp.maximum(density, 1e-12), 0.0)
        refined = x0 - newton + jax.lax.stop_gradient(newton)
    if dist.support in ("nonnegative", "positive"):
        # An atom at zero, or a lower edge the bracket straddles, is answered exactly.
        refined = jnp.where(frozen.cdf(jnp.zeros_like(x0)) >= level, 0.0, jnp.maximum(refined, 0.0))
    else:
        refined = jnp.where(level <= 0.0, -jnp.inf, refined)
    if dist.defective:
        # A level above the mass that ever happens is never reached.
        refined = jnp.where(frozen.cdf(jnp.full_like(x0, jnp.inf)) < level, jnp.inf, refined)
    return jnp.where(level >= 1.0, jnp.inf, refined)


def _integer_quantile(dist: Distribution, q):
    """Smallest whole number ``k`` with ``cdf(k) >= q``, by bisection over the integers."""
    q = jnp.asarray(q, dtype=jnp.float32)
    shape = jnp.broadcast_shapes(q.shape, dist.batch_shape)
    level = jnp.broadcast_to(q, shape)
    inner = jnp.clip(level, 1e-7, 1.0 - 1e-7)
    mean = jnp.broadcast_to(dist.mean(), shape)
    sd = jnp.broadcast_to(dist.stddev(), shape)
    low = jnp.full(shape, -1.0, dtype=jnp.float32)
    high = jnp.ceil(mean + sd * jnp.sqrt(inner / (1.0 - inner))) + 1.0

    def step(_, bounds):
        low, high = bounds
        middle = jnp.floor(0.5 * (low + high))
        above = dist.cdf(middle) >= level
        return jnp.where(above, low, middle), jnp.where(above, middle, high)

    _low, high = jax.lax.fori_loop(0, 64, step, (low, high))
    return jnp.where(level <= 0.0, 0.0, high)


def randomized_pit(dist: Distribution, value, key):
    """Probability integral transform of ``value``, uniform on ``(0, 1)`` when ``value`` is a draw from ``dist``.

    Where the CDF jumps (a count, or the zero of a hurdle), the PIT is spread
    uniformly over the jump at the observed value, so the score is still
    uniform.
    """
    upper = dist.cdf(value)
    lower = dist.cdf_left(value)
    return lower + jax.random.uniform(key, jnp.shape(upper), dtype=upper.dtype) * (upper - lower)


# Where the quantile spline's slope may change, in logistic scores on each side of the median: the levels
# 0.5, 0.75, 0.91, and 0.977, a normal's median and 0.67, 1.34, and 2 standard deviations.
SPLINE_BREAKS = (0.0, 1.1, 2.3, 3.75)
# The slope that gives a logistic score unit variance: sqrt(3) / pi.
SPLINE_UNIT_SLOPE = math.sqrt(3.0) / math.pi
SPLINE_VARIANCE_POINTS = 2048


def _spline_widths(dtype=jnp.float32):
    breaks = jnp.asarray(SPLINE_BREAKS, dtype=dtype)
    return breaks, jnp.concatenate([jnp.diff(breaks), jnp.asarray([jnp.inf], dtype=dtype)])


class QuantileSpline(Distribution):
    """A quantile function piecewise linear in the logistic score: ``Y = Q(Z)``, ``Z`` standard logistic.

    ``median`` is ``Q(0)``; ``upper[..., k]`` is the slope of ``Q`` on the
    ``k``-th piece above the median and ``lower[..., k]`` on the ``k``-th
    below, the pieces breaking at the logistic scores ``SPLINE_BREAKS`` and
    the last running to infinity. Every slope is positive, so ``Q`` rises
    and its inverse is the sum of each piece's clipped share, in closed
    form: ``cdf``, ``log_prob``, and ``quantile`` need no bisection. Equal
    slopes give a logistic distribution, close to a normal in the middle;
    unequal slopes give skew and either tail heavier or lighter. Beyond the
    last break each tail is exponential, so a far value costs a likelihood
    linear in its distance, not quadratic: an outlier cannot swamp the fit.

    Because each quantile is the median plus or minus a sum of slopes, a
    quantile rises with the median, with every upper slope, and as every
    lower slope falls. The ``quantile_spline`` head with ``monotone=``
    builds those three directions in, which is first-order stochastic
    dominance in the constrained features.
    """

    median: jax.Array
    upper: jax.Array
    lower: jax.Array
    reparameterized: ClassVar[bool] = True
    atomless: ClassVar[bool] = True

    def __init__(self, median, upper, lower):
        self.median = jnp.asarray(median, dtype=jnp.float32)
        self.upper = jnp.broadcast_to(jnp.asarray(upper, dtype=jnp.float32), self.median.shape + (len(SPLINE_BREAKS),))
        self.lower = jnp.broadcast_to(jnp.asarray(lower, dtype=jnp.float32), self.median.shape + (len(SPLINE_BREAKS),))

    @property
    def batch_shape(self):
        return self.median.shape

    def _knots(self):
        """Values of ``Q`` at the breaks above and below the median, ``batch + (pieces,)``."""
        _breaks, widths = _spline_widths(self.median.dtype)
        finite = jnp.where(jnp.isfinite(widths), widths, 0.0)
        above = self.median[..., None] + jnp.cumsum(self.upper * finite, axis=-1) - self.upper * finite
        below = self.median[..., None] - jnp.cumsum(self.lower * finite, axis=-1) + self.lower * finite
        return above, below, widths

    def score(self, value):
        """The logistic score ``Q^{-1}(value)`` and the slope of ``Q`` there."""
        value = jnp.asarray(value, dtype=self.median.dtype)[..., None]
        above, below, widths = self._knots()
        up = jnp.clip((value - above) / self.upper, 0.0, widths)
        down = jnp.clip((below - value) / self.lower, 0.0, widths)
        z = jnp.sum(up, axis=-1) - jnp.sum(down, axis=-1)
        # The piece the score falls in gives the slope: read from z, not from the knots, so rounding at a
        # knot cannot leave a value in no piece (and a zero slope).
        breaks, _widths = _spline_widths(self.median.dtype)
        piece = jnp.sum(jnp.abs(z)[..., None] >= breaks, axis=-1) - 1
        shape = z.shape + self.upper.shape[-1:]
        upper = jnp.take_along_axis(jnp.broadcast_to(self.upper, shape), piece[..., None], axis=-1)[..., 0]
        lower = jnp.take_along_axis(jnp.broadcast_to(self.lower, shape), piece[..., None], axis=-1)[..., 0]
        slope = jnp.where(z >= 0, upper, lower)
        return z, slope

    def quantile_of_score(self, z):
        z = jnp.asarray(z, dtype=self.median.dtype)
        breaks, widths = _spline_widths(self.median.dtype)
        up = jnp.clip(jnp.maximum(z, 0.0)[..., None] - breaks, 0.0, widths)
        down = jnp.clip(jnp.maximum(-z, 0.0)[..., None] - breaks, 0.0, widths)
        return self.median + jnp.sum(self.upper * up, axis=-1) - jnp.sum(self.lower * down, axis=-1)

    def log_prob(self, value):
        z, slope = self.score(value)
        return -z - 2.0 * jax.nn.softplus(-z) - jnp.log(slope)

    def cdf(self, value):
        return jax.nn.sigmoid(self.score(value)[0])

    def log_cdf(self, value):
        return -jax.nn.softplus(-self.score(value)[0])

    def log_survival(self, value):
        return -jax.nn.softplus(self.score(value)[0])

    def quantile(self, q):
        q = jnp.asarray(q, dtype=self.median.dtype)
        return self.quantile_of_score(jnp.log(q) - jnp.log1p(-q))

    def sample(self, key, sample_shape=()):
        z = jax.random.logistic(key, tuple(sample_shape) + self.batch_shape, dtype=self.median.dtype)
        return self.quantile_of_score(z)

    def mean(self):
        # E[clip(Z - b, 0, w)] = the integral of P(Z > t) from b to b + w; the lower side mirrors it.
        breaks, widths = _spline_widths(self.median.dtype)
        end = jnp.where(jnp.isfinite(widths), breaks + jnp.where(jnp.isfinite(widths), widths, 0.0), 0.0)
        piece = jax.nn.softplus(-breaks) - jnp.where(jnp.isfinite(widths), jax.nn.softplus(-end), 0.0)
        return self.median + jnp.sum((self.upper - self.lower) * piece, axis=-1)

    def variance(self):
        # The midpoints of SPLINE_VARIANCE_POINTS equal slices of probability.
        levels = (jnp.arange(SPLINE_VARIANCE_POINTS, dtype=self.median.dtype) + 0.5) / SPLINE_VARIANCE_POINTS
        z = (jnp.log(levels) - jnp.log1p(-levels)).reshape((-1,) + (1,) * len(self.batch_shape))
        values = self.quantile_of_score(z)
        mean = self.mean()
        return jnp.mean(jnp.square(values - mean), axis=0)

    def convolve_normal(self, scale):
        return NormalNoise(self, scale)


CURE_HEADS = ("gaussian", "mixture", "flow", "student_t")


def cure_width(kind: str, flat) -> bool:
    """Whether ``flat`` carries one more output than ``kind``'s head: the logit of never happening."""
    from ashlar.distributional import output_width

    return kind in CURE_HEADS and int(jnp.shape(flat)[-1]) == output_width(kind, 1) + 1


def head_distribution(kind: str, flat, horizon: int = 1) -> Distribution:
    """The distribution of one ``ashlar.distributional`` head from its flat projection.

    ``flat`` is ``batch + (width,)`` on the standardized axis (the raw count
    for a count head). With ``horizon > 1`` the steps of a forecast become the
    last batch axis; each step is its own marginal, and nothing is claimed
    about the dependence between steps. A head built with ``cure=True`` has
    one more output, the logit of never happening, and gives a ``Cured``.
    """
    from ashlar.distributional import QUANTILE_LEVELS, decode

    flat = jnp.asarray(flat)
    if int(horizon) == 1 and cure_width(kind, flat):
        return Cured(head_distribution(kind, flat[..., :-1]), jax.nn.sigmoid(flat[..., -1]))
    batch = flat.shape[:-1]
    decoded = decode(kind, flat.reshape((-1, flat.shape[-1])), int(horizon))
    steps = (int(horizon),) if int(horizon) > 1 else ()

    def shaped(array):
        return array.reshape(batch + steps + array.shape[2:])

    if kind == "gaussian":
        loc, scale = decoded
        return Normal(shaped(loc), shaped(scale))
    if kind == "mixture":
        loc, scale, weight = decoded
        log_weight = jnp.log(jnp.maximum(weight, 1e-30))
        return Mixture(Normal(shaped(loc), shaped(scale)), shaped(log_weight))
    if kind == "flow":
        return Flow(shaped(decoded))
    if kind == "histogram":
        return Histogram(shaped(decoded))
    if kind == "student_t":
        return StudentT(*(shaped(part) for part in decoded))
    if kind == "quantile_spline":
        return QuantileSpline(*(shaped(part) for part in decoded))
    if kind == "poisson":
        return Poisson(*(shaped(part) for part in decoded))
    if kind == "negative_binomial":
        return NegativeBinomial(*(shaped(part) for part in decoded))
    if kind == "quantile":
        return QuantileLevels(values=shaped(decoded), levels=tuple(QUANTILE_LEVELS))
    raise AshlarError(f"the {kind} head has no distribution.")


def joint_normal_distribution(flat, n_events: int) -> LowRankMultivariateNormal:
    """The joint normal of the ``multivariate_gaussian`` head from its flat projection ``batch + (width,)``.

    The width is ``n_events * (2 + rank)``: locations, raw scales, then the
    covariance factor row by row.
    """
    from ashlar.predict import positive_scale

    flat = jnp.asarray(flat)
    d = int(n_events)
    width = flat.shape[-1]
    if width < 2 * d or (width - 2 * d) % d:
        raise ShapeError(f"a joint normal over {d} columns needs a width of {d} * (2 + rank); received {width}.")
    rank = (width - 2 * d) // d
    loc = flat[..., :d]
    std = positive_scale(flat[..., d : 2 * d])
    factor = flat[..., 2 * d :].reshape(flat.shape[:-1] + (d, rank))
    return LowRankMultivariateNormal(loc, factor, jnp.square(std))


def joint_mixture_distribution(flat, n_events: int, components: int) -> Mixture:
    """The ``multivariate_mixture`` head's mixture of joint normals from its flat projection ``batch + (width,)``.

    The width is ``components * (1 + n_events * (2 + rank))``: the component
    logits, then each component's locations, raw scales, and covariance
    factor, laid out as in ``joint_normal_distribution``. The result is a
    ``Mixture`` whose components are ``LowRankMultivariateNormal`` with batch
    ``batch + (components,)``.
    """
    flat = jnp.asarray(flat)
    count, d = int(components), int(n_events)
    width = int(flat.shape[-1])
    per = (width - count) // count if count > 0 else 0
    if count < 1 or (width - count) % count or per < 2 * d or (per - 2 * d) % d:
        raise ShapeError(
            f"a mixture of {count} joint normals over {d} columns needs a width of {count} * (1 + {d} * (2 + rank)); "
            f"received {width}."
        )
    blocks = flat[..., count:].reshape(flat.shape[:-1] + (count, per))
    return Mixture(joint_normal_distribution(blocks, d), flat[..., :count])



MOMENT_DRAWS = 1024
_MOMENT_CHUNK = 64


class Empirical(Distribution):
    """Equal-weight draws, ``values`` shaped ``batch_shape + (draws,)``.

    The marginals of a ``MultivariateFlow`` are this: its coordinates have no
    closed-form CDF, so they are read from fixed draws. ``cdf`` is the share of
    draws at or below a value and ``quantile`` the matching order statistic.
    It has no density.
    """

    values: jax.Array

    def __init__(self, values):
        self.values = jnp.asarray(values, dtype=jnp.float32)

    @property
    def batch_shape(self):
        return self.values.shape[:-1]

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.batch_shape
        index = jax.random.randint(key, shape, 0, self.values.shape[-1])
        values = jnp.broadcast_to(self.values, shape + self.values.shape[-1:])
        return jnp.take_along_axis(values, index[..., None], axis=-1)[..., 0]

    def mean(self):
        return jnp.mean(self.values, axis=-1)

    def variance(self):
        return jnp.var(self.values, axis=-1)

    def cdf(self, value):
        return jnp.mean(self.values <= jnp.asarray(value, dtype=self.values.dtype)[..., None], axis=-1)

    def cdf_left(self, value):
        return jnp.mean(self.values < jnp.asarray(value, dtype=self.values.dtype)[..., None], axis=-1)

    def quantile(self, q):
        q = jnp.asarray(q, dtype=self.values.dtype)
        n = self.values.shape[-1]
        shape = jnp.broadcast_shapes(q.shape, self.batch_shape)
        ordered = jnp.broadcast_to(jnp.sort(self.values, axis=-1), shape + (n,))
        index = jnp.clip(jnp.ceil(jnp.broadcast_to(q, shape) * n) - 1, 0, n - 1).astype(jnp.int32)
        return jnp.take_along_axis(ordered, index[..., None], axis=-1)[..., 0]


def autoregressive_masks(n_events: int, hidden: int):
    """MADE masks: column ``j`` of the output sees only columns before ``j`` of the input.

    Hidden unit ``h`` has degree ``h mod (d - 1) + 1`` and sees inputs of
    degree at most its own; output column ``j`` (degree ``j + 1``) sees hidden
    units of lower degree. With one column there is nothing to condition on,
    and both masks are zero.
    """
    d, hidden = int(n_events), int(hidden)
    inputs = jnp.arange(1, d + 1)
    units = jnp.arange(hidden) % max(d - 1, 1) + 1
    mask_in = (units[:, None] >= inputs[None, :]) & (d > 1)
    mask_out = inputs[:, None] > units[None, :]
    return mask_in.astype(jnp.float32), mask_out.astype(jnp.float32)


def autoregressive_context_width(n_events: int, hidden: int) -> int:
    """Per-row context per block: a shift for every hidden unit and for every column's parameters."""
    from ashlar.distributional import flow_parameter_count

    return int(hidden) + int(n_events) * (flow_parameter_count() + 2)


class AutoregressiveFlow(Distribution):
    """A conditional masked autoregressive flow over ``d`` coordinates.

    ``p(y | x) = Π_j p(y_j | y_<j, x)``. Each conditional pushes a standard
    normal through ``y_j = μ_j + σ_j s_j(z_j)``, where ``s_j`` is a monotone
    rational-quadratic spline (the ``flow`` head's spline) and ``μ_j``,
    ``σ_j``, and the spline's knots come from a masked network (MADE) that
    reads the columns before ``j`` and the row's context. So each column's
    whole conditional shape, not only its location, can change with the
    columns before it: curved, skewed, heteroscedastic, or tail dependence.
    ``depth`` blocks stack, with the column order reversed in every other one.

    ``context`` is ``batch_shape + (depth * autoregressive_context_width,)``,
    one per row and member, from the trunk. The masked weights ``w_in``
    ``(members, depth, hidden, d)`` and ``w_out`` ``(members, depth, d (S + 2),
    hidden)`` are shared by every row and line up with the last batch axis,
    the member axis of an estimator's mixture. Indexing a batch slices the
    context only.

    The density is exact and evaluated in one pass; a draw takes ``d`` passes,
    one column at a time, and is reparameterized. Mean, variance, and
    covariance are Monte Carlo estimates from ``MOMENT_DRAWS`` fixed draws, and
    ``marginals()`` is an ``Empirical`` of those draws. With all parameters at
    zero it is the standard normal.
    """

    context: jax.Array
    w_in: jax.Array
    w_out: jax.Array
    n_events: int = field(static=True)
    reparameterized: ClassVar[bool] = True
    shared_parameters: ClassVar[bool] = True

    def __init__(self, context, w_in, w_out, n_events: int):
        context = jnp.asarray(context, dtype=jnp.float32)
        w_in = jnp.asarray(w_in, dtype=jnp.float32)
        w_out = jnp.asarray(w_out, dtype=jnp.float32)
        d = int(n_events)
        if w_in.ndim != 4 or w_in.shape[-1] != d:
            raise ShapeError(f"w_in must be (members, depth, hidden, {d}); received {w_in.shape}.")
        members, depth, hidden = w_in.shape[:3]
        width = autoregressive_context_width(d, hidden)
        if context.ndim < 1 or context.shape[-1] != depth * width or (context.ndim > 1 and context.shape[-2] != members):
            raise ShapeError(
                f"context must be (..., {members}, {depth * width}) for {members} members and {depth} blocks; "
                f"received {context.shape}."
            )
        if w_out.shape != (members, depth, width - hidden, hidden):
            raise ShapeError(f"w_out must be {(members, depth, width - hidden, hidden)}; received {w_out.shape}.")
        self.context, self.w_in, self.w_out, self.n_events = context, w_in, w_out, d

    @property
    def batch_shape(self):
        return self.context.shape[:-1]

    @property
    def event_shape(self):
        return (self.n_events,)

    @property
    def depth(self) -> int:
        return int(self.w_in.shape[1])

    @property
    def hidden(self) -> int:
        return int(self.w_in.shape[2])

    def _index(self, index):
        return AutoregressiveFlow(self.context[index], self.w_in, self.w_out, self.n_events)

    def _conditioner(self, block: int, context, inputs):
        """Spline parameters, shift, and scale of every column in block order, from the columns before it."""
        from ashlar.distributional import flow_parameter_count

        d, hidden, spline = self.n_events, self.hidden, flow_parameter_count()
        mask_in, mask_out = autoregressive_masks(d, hidden)
        width = autoregressive_context_width(d, hidden)
        part = context[..., block * width : (block + 1) * width]
        w_in = self.w_in[:, block] * mask_in
        w_out = self.w_out[:, block] * jnp.repeat(mask_out, spline + 2, axis=0)
        units = jnp.tanh(jnp.einsum("...kd,khd->...kh", inputs, w_in) + part[..., :hidden])
        out = jnp.einsum("...kh,koh->...ko", units, w_out) + part[..., hidden:]
        out = out.reshape(out.shape[:-1] + (d, spline + 2))
        scale = jax.nn.softplus(out[..., spline + 1] + _SOFTPLUS_ONE) + 1e-3
        return out[..., :spline], out[..., spline], scale

    def _aligned(self, value):
        value = jnp.asarray(value, dtype=self.context.dtype)
        batch = jnp.broadcast_shapes(value.shape[:-1], self.batch_shape)
        context = jnp.broadcast_to(self.context, batch + self.context.shape[-1:])
        return context, jnp.broadcast_to(value, batch + (self.n_events,))

    def inverse(self, value):
        """Map ``value`` to the base noise, with ``log|det dz/dy|``. One network pass per block."""
        from ashlar.distributional import flow_inverse

        context, value = self._aligned(value)
        logdet = jnp.zeros(value.shape[:-1], dtype=value.dtype)
        for block in reversed(range(self.depth)):
            flip = block % 2 == 1
            ordered = jnp.flip(value, axis=-1) if flip else value
            splines, shift, scale = self._conditioner(block, context, ordered)
            base, spline_logdet = flow_inverse(splines, (ordered - shift) / scale)
            logdet = logdet + jnp.sum(spline_logdet - jnp.log(scale), axis=-1)
            value = jnp.flip(base, axis=-1) if flip else base
        return value, logdet

    def forward(self, noise):
        """Push standard-normal ``noise`` through the flow, one column at a time."""
        from ashlar.distributional import flow_forward

        context, value = self._aligned(noise)
        for block in range(self.depth):
            flip = block % 2 == 1
            base = jnp.flip(value, axis=-1) if flip else value
            ordered = jnp.zeros_like(base)
            for column in range(self.n_events):
                splines, shift, scale = self._conditioner(block, context, ordered)
                step = shift[..., column] + scale[..., column] * flow_forward(splines[..., column, :], base[..., column])
                ordered = ordered.at[..., column].set(step)
            value = jnp.flip(ordered, axis=-1) if flip else ordered
        return value

    def log_prob(self, value):
        base, logdet = self.inverse(value)
        return jnp.sum(-0.5 * (math.log(2 * math.pi) + jnp.square(base)), axis=-1) + logdet

    def _column_conditioner(self, column: int, inputs):
        """One column's spline parameters, shift, and scale in the first block, from the inputs before it.

        The same numbers as ``_conditioner(0, ...)[..., column]``, with only
        that column's slice of the output layer computed.
        """
        from ashlar.distributional import flow_parameter_count

        d, hidden, spline = self.n_events, self.hidden, flow_parameter_count()
        mask_in, mask_out = autoregressive_masks(d, hidden)
        part = self.context[..., : autoregressive_context_width(d, hidden)]
        units = jnp.tanh(jnp.einsum("...kd,khd->...kh", inputs, self.w_in[:, 0] * mask_in) + part[..., :hidden])
        rows = slice(column * (spline + 2), (column + 1) * (spline + 2))
        w_out = self.w_out[:, 0, rows] * mask_out[column]
        out = jnp.einsum("...kh,koh->...ko", units, w_out) + part[..., hidden:][..., rows]
        scale = jax.nn.softplus(out[..., spline + 1] + _SOFTPLUS_ONE) + 1e-3
        return out[..., :spline], out[..., spline], scale

    def log_box(self, lower, upper, points=None):
        """The GHK simulator in the flow's own column order. One block only (``depth=1``).

        An exact column scores its density given the columns before it. A
        coarse column scores the probability of its interval under its
        conditional, the interval pulled back through the column's spline to
        the base normal, and then draws a value inside it (a truncated normal
        pushed through the spline) for the columns after it, one draw per
        point. So a row whose only coarse column is the last is closed form;
        otherwise it is a quasi-Monte Carlo estimate. With more than one block
        a column depends on columns after it, so there is no such order.
        """
        from ashlar.distributional import flow_forward, flow_inverse

        if self.depth != 1:
            raise NotImplementedError(
                "a flow's box likelihood needs one autoregressive block (depth=1); "
                "with more, a column depends on the columns after it."
            )
        d = self.n_events
        lower, upper = _bounds(lower, upper)
        batch = jnp.broadcast_shapes(lower.shape[:-1], upper.shape[:-1], self.batch_shape)
        lower, upper = jnp.broadcast_to(lower, batch + (d,)), jnp.broadcast_to(upper, batch + (d,))
        points = _check_box_points(points, d)
        exact = (lower == upper) & jnp.isfinite(lower)
        uniform = _coarse_points(points, exact)
        count = points.shape[0]

        def pulled_back(splines, shift, scale, bound):
            finite = jnp.isfinite(bound)
            base, _logdet = flow_inverse(splines, (jnp.where(finite, bound, 0.0) - shift) / scale)
            return jnp.where(finite, base, bound)

        values = jnp.zeros((count,) + batch + (d,), dtype=jnp.float32)
        total = jnp.zeros((count,) + batch, dtype=jnp.float32)
        for column in range(d):
            splines, shift, scale = self._column_conditioner(column, values)
            is_exact = exact[..., column]
            low, high = lower[..., column], upper[..., column]
            seen = jnp.where(is_exact, low, 0.0)
            base, logdet = flow_inverse(splines, (seen - shift) / scale)
            point = -0.5 * (math.log(2 * math.pi) + jnp.square(base)) + logdet - jnp.log(scale)
            alpha = pulled_back(splines, shift, scale, jnp.where(is_exact, -jnp.inf, low))
            beta = pulled_back(splines, shift, scale, jnp.where(is_exact, jnp.inf, high))
            log_mass = _standard_log_interval(alpha, beta)
            draw = _truncated_standard_normal(alpha, beta, log_mass, uniform[..., column])
            drawn = shift + scale * flow_forward(splines, draw)
            total = total + jnp.where(is_exact, point, log_mass)
            values = values.at[..., column].set(jnp.where(is_exact, seen, drawn))
        return jax.nn.logsumexp(total, axis=0) - math.log(count)

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.batch_shape + self.event_shape
        return self.forward(jax.random.normal(key, shape, dtype=self.context.dtype))

    def _moments(self):
        """Mean and covariance from ``MOMENT_DRAWS`` fixed draws, in chunks so memory stays flat."""
        keys = jax.random.split(jax.random.PRNGKey(0), MOMENT_DRAWS // _MOMENT_CHUNK)
        d = self.n_events
        zero = (
            jnp.zeros(self.batch_shape + (d,), dtype=self.context.dtype),
            jnp.zeros(self.batch_shape + (d, d), dtype=self.context.dtype),
        )

        def step(totals, key):
            draws = self.sample(key, (_MOMENT_CHUNK,))
            first = totals[0] + jnp.sum(draws, axis=0)
            second = totals[1] + jnp.einsum("n...i,n...j->...ij", draws, draws)
            return (first, second), None

        (first, second), _ = jax.lax.scan(step, zero, keys)
        mean = first / MOMENT_DRAWS
        covariance = second / MOMENT_DRAWS - mean[..., :, None] * mean[..., None, :]
        return mean, covariance

    def mean(self):
        return self._moments()[0]

    def covariance(self):
        return self._moments()[1]

    def variance(self):
        return jnp.maximum(jnp.diagonal(self.covariance(), axis1=-2, axis2=-1), 0.0)

    def marginals(self):
        draws = self.sample(jax.random.PRNGKey(0), (MOMENT_DRAWS,))
        return Empirical(jnp.moveaxis(draws, 0, -1))


def joint_flow_distribution(raw, n_events: int) -> AutoregressiveFlow:
    """The ``multivariate_flow`` head's distribution from its output ``(context, w_in, w_out)``."""
    context, w_in, w_out = raw
    return AutoregressiveFlow(context, w_in, w_out, n_events)


_SOFTPLUS_ONE = math.log(math.expm1(1.0 - 1e-3))  # softplus(raw + this) + 1e-3 is 1 at raw = 0


class LogNormal(Distribution):
    """``exp`` of a normal with location ``loc`` and scale ``scale`` on the log axis. Positive values."""

    loc: jax.Array
    scale: jax.Array
    reparameterized: ClassVar[bool] = True
    support: ClassVar[str] = "positive"
    atomless: ClassVar[bool] = True

    def __init__(self, loc, scale):
        self.loc, self.scale = _broadcast(loc, scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    def log_prob(self, value):
        value = jnp.asarray(value, dtype=self.loc.dtype)
        safe = jnp.maximum(value, 1e-30)
        density = _normal_log_prob(jnp.log(safe), self.loc, self.scale) - jnp.log(safe)
        return jnp.where(value > 0, density, -jnp.inf)

    def sample(self, key, sample_shape=()):
        noise = jax.random.normal(key, tuple(sample_shape) + self.loc.shape, dtype=self.loc.dtype)
        return jnp.exp(self.loc + self.scale * noise)

    def mean(self):
        return jnp.exp(self.loc + 0.5 * jnp.square(self.scale))

    def variance(self):
        return jnp.expm1(jnp.square(self.scale)) * jnp.exp(2 * self.loc + jnp.square(self.scale))

    def cdf(self, value):
        value = jnp.asarray(value, dtype=self.loc.dtype)
        return jnp.where(value > 0, ndtr((jnp.log(jnp.maximum(value, 1e-30)) - self.loc) / self.scale), 0.0)

    def _log_axis(self, value):
        value = jnp.asarray(value, dtype=self.loc.dtype)
        positive = value > 0
        return positive, (jnp.log(jnp.where(positive, value, 1.0)) - self.loc) / self.scale

    def log_cdf(self, value):
        positive, z = self._log_axis(value)
        return jnp.where(positive, log_ndtr(z), -jnp.inf)

    def log_survival(self, value):
        positive, z = self._log_axis(value)
        return jnp.where(positive, log_ndtr(-z), 0.0)

    def quantile(self, q):
        return jnp.exp(self.loc + self.scale * ndtri(jnp.asarray(q, dtype=self.loc.dtype)))


class Gamma(Distribution):
    """Gamma with mean ``mean_value`` and shape ``concentration``; variance ``mean² / concentration``."""

    mean_value: jax.Array
    concentration: jax.Array
    reparameterized: ClassVar[bool] = True
    support: ClassVar[str] = "positive"
    atomless: ClassVar[bool] = True

    def __init__(self, mean_value, concentration):
        self.mean_value, self.concentration = _broadcast(mean_value, concentration)

    @property
    def batch_shape(self):
        return self.mean_value.shape

    def _rate(self):
        return self.concentration / self.mean_value

    def log_prob(self, value):
        value = jnp.asarray(value, dtype=self.mean_value.dtype)
        safe = jnp.maximum(value, 1e-30)
        k, rate = self.concentration, self._rate()
        density = k * jnp.log(rate) + (k - 1) * jnp.log(safe) - rate * safe - jax.scipy.special.gammaln(k)
        return jnp.where(value > 0, density, -jnp.inf)

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.batch_shape
        draw = jax.random.gamma(key, jnp.broadcast_to(self.concentration, shape), shape, dtype=self.mean_value.dtype)
        return draw / self._rate()

    def mean(self):
        return self.mean_value

    def variance(self):
        return jnp.square(self.mean_value) / self.concentration

    def cdf(self, value):
        value = jnp.asarray(value, dtype=self.mean_value.dtype)
        return jnp.where(value > 0, gammainc(self.concentration, self._rate() * jnp.maximum(value, 0.0)), 0.0)

    def _scaled(self, value):
        value = jnp.asarray(value, dtype=self.mean_value.dtype)
        positive = value > 0
        return positive, self._rate() * jnp.where(positive, value, 1.0)

    def log_cdf(self, value):
        positive, x = self._scaled(value)
        return jnp.where(positive, _safe_log(gammainc(self.concentration, x)), -jnp.inf)

    def log_survival(self, value):
        positive, x = self._scaled(value)
        return jnp.where(positive, _safe_log(gammaincc(self.concentration, x)), 0.0)


class Hurdle(Distribution):
    """An exact zero with probability ``zero``, otherwise a draw from the positive ``severity``.

    The outcome of an insurance claim, a tip, a day's rain, or an order: most
    rows are exactly zero, and the rest follow their own positive
    distribution. ``log_prob`` is the log probability of the zero at 0 and
    ``log(1 - zero)`` plus the severity's log density elsewhere, the usual
    likelihood of a mixed outcome.
    """

    zero: jax.Array
    severity: Distribution
    support: ClassVar[str] = "nonnegative"

    def __init__(self, zero, severity: Distribution):
        self.zero = jnp.broadcast_to(jnp.asarray(zero, dtype=jnp.float32), severity.batch_shape)
        self.severity = severity

    @property
    def batch_shape(self):
        return self.zero.shape

    def _log_zero(self):
        return jnp.log(jnp.clip(self.zero, 1e-30, 1.0))

    def _log_keep(self):
        return jnp.log1p(-jnp.minimum(self.zero, 1.0 - 1e-7))

    def log_prob(self, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        positive = self._log_keep() + self.severity.log_prob(jnp.where(value > 0, value, 1.0))
        return jnp.where(value > 0, positive, jnp.where(value == 0, self._log_zero(), -jnp.inf))

    def sample(self, key, sample_shape=()):
        zero_key, severity_key = jax.random.split(key)
        shape = tuple(sample_shape) + self.batch_shape
        draw = self.severity.sample(severity_key, sample_shape)
        return jnp.where(jax.random.bernoulli(zero_key, jnp.broadcast_to(self.zero, shape)), 0.0, draw)

    def mean(self):
        return (1.0 - self.zero) * self.severity.mean()

    def convolve_normal(self, scale):
        return HurdleNoise(self, scale)

    def variance(self):
        keep = 1.0 - self.zero
        severity_mean = self.severity.mean()
        return keep * (self.severity.variance() + jnp.square(severity_mean)) - jnp.square(keep * severity_mean)

    def cdf(self, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        inside = self.zero + (1.0 - self.zero) * self.severity.cdf(jnp.maximum(value, 0.0))
        return jnp.where(value < 0, 0.0, inside)

    def cdf_left(self, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        inside = self.zero + (1.0 - self.zero) * self.severity.cdf(value)
        return jnp.where(value <= 0, 0.0, inside)

    def _severity_tails(self, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        positive = value > 0
        inner = jnp.where(positive, value, 1.0)
        return value, positive, self.severity.log_cdf(inner), self.severity.log_survival(inner)

    def log_cdf(self, value):
        value, positive, below, _above = self._severity_tails(value)
        inside = jnp.logaddexp(self._log_zero(), self._log_keep() + jnp.where(positive, below, -jnp.inf))
        return jnp.where(value >= 0, inside, -jnp.inf)

    def log_cdf_left(self, value):
        value, positive, below, _above = self._severity_tails(value)
        return jnp.where(positive, jnp.logaddexp(self._log_zero(), self._log_keep() + below), -jnp.inf)

    def log_survival(self, value):
        value, positive, _below, above = self._severity_tails(value)
        return jnp.where(value >= 0, self._log_keep() + jnp.where(positive, above, 0.0), 0.0)

    def log_survival_left(self, value):
        value, positive, _below, above = self._severity_tails(value)
        return jnp.where(positive, self._log_keep() + above, 0.0)

    def quantile(self, q):
        q = jnp.asarray(q, dtype=jnp.float32)
        level = jnp.clip((q - self.zero) / jnp.maximum(1.0 - self.zero, 1e-30), 1e-7, 1.0 - 1e-7)
        return jnp.where(q <= self.zero, 0.0, self.severity.quantile(level))


HURDLE_SEVERITIES = ("lognormal", "gamma")

# Gauss-Hermite nodes for a value plus normal noise with no closed form.
NOISE_NODES = 32


def _hermite(count: int = NOISE_NODES):
    nodes, weights = np.polynomial.hermite.hermgauss(int(count))
    return nodes.astype(np.float32), np.log(weights / math.sqrt(math.pi)).astype(np.float32)


class NormalNoise(Distribution):
    """``Y + e`` for ``Y`` from ``base`` and independent ``e ~ Normal(0, scale)``, by Gauss-Hermite quadrature.

    The density, CDF, and interval probabilities are means over
    ``NOISE_NODES`` points of the noise, ``E_e[p(y - e)]``, taken in log
    space so the tails stay finite. Exact where the closed forms exist, and
    accurate when the noise is no wider than the spread of ``base``; where
    the noise is far wider than a narrow peak, the points can step over it.
    ``scale`` may be 0 for a row with no noise.
    """

    base: Distribution
    scale: jax.Array
    atomless: ClassVar[bool] = True

    def __init__(self, base: Distribution, scale):
        self.base = base
        self.scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), base.batch_shape)

    @property
    def batch_shape(self):
        return self.scale.shape

    def _offsets(self, value):
        nodes, log_weights = _hermite()
        shape = (-1,) + (1,) * jnp.ndim(jnp.broadcast_to(jnp.asarray(value, jnp.float32), jnp.broadcast_shapes(jnp.shape(value), self.batch_shape)))
        offsets = math.sqrt(2.0) * self.scale * jnp.asarray(nodes).reshape(shape)
        return offsets, jnp.asarray(log_weights).reshape(shape)

    def _mean_log(self, fn, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        offsets, log_weights = self._offsets(value)
        return jax.nn.logsumexp(fn(value - offsets) + log_weights, axis=0)

    def log_prob(self, value):
        return self._mean_log(self.base.log_prob, value)

    def cdf(self, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        offsets, log_weights = self._offsets(value)
        return jnp.sum(jnp.exp(log_weights) * self.base.cdf(value - offsets), axis=0)

    def log_cdf(self, value):
        return self._mean_log(self.base.log_cdf, value)

    def log_survival(self, value):
        return self._mean_log(self.base.log_survival, value)

    def log_interval(self, lower, upper):
        lower, upper = _bounds(lower, upper)
        offsets, log_weights = self._offsets(jnp.broadcast_to(lower, jnp.broadcast_shapes(lower.shape, upper.shape)))
        inner = self.base.log_interval(lower - offsets, upper - offsets)
        return jax.nn.logsumexp(inner + log_weights, axis=0)

    def sample(self, key, sample_shape=()):
        base_key, noise_key = jax.random.split(key)
        shape = tuple(sample_shape) + self.batch_shape
        return self.base.sample(base_key, sample_shape) + self.scale * jax.random.normal(noise_key, shape)

    def mean(self):
        return self.base.mean()

    def variance(self):
        return self.base.variance() + jnp.square(self.scale)


class HurdleNoise(Distribution):
    """A hurdle outcome plus normal noise: the zero becomes a normal at 0, and the amount a ``NormalNoise``."""

    zero: jax.Array
    noise: Normal
    amount: NormalNoise
    hurdle: Distribution
    atomless: ClassVar[bool] = True

    def __init__(self, hurdle, scale):
        self.hurdle = hurdle
        self.zero = hurdle.zero
        scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float32), hurdle.batch_shape)
        self.noise = Normal(jnp.zeros_like(scale), jnp.maximum(scale, 1e-6))
        self.amount = NormalNoise(hurdle.severity, scale)

    @property
    def batch_shape(self):
        return self.zero.shape

    def _mix(self, at_zero, amount):
        log_zero = jnp.log(jnp.clip(self.zero, 1e-30, 1.0))
        log_keep = jnp.log1p(-jnp.minimum(self.zero, 1.0 - 1e-7))
        return jnp.logaddexp(log_zero + at_zero, log_keep + amount)

    def log_prob(self, value):
        # A row recorded without noise keeps the hurdle's own likelihood: the mass of the zero, not a density.
        noisy = self._mix(self.noise.log_prob(value), self.amount.log_prob(value))
        return jnp.where(self.amount.scale > 0, noisy, self.hurdle.log_prob(value))

    def cdf(self, value):
        return self.zero * self.noise.cdf(value) + (1.0 - self.zero) * self.amount.cdf(value)

    def log_cdf(self, value):
        return self._mix(self.noise.log_cdf(value), self.amount.log_cdf(value))

    def log_survival(self, value):
        return self._mix(self.noise.log_survival(value), self.amount.log_survival(value))

    def log_interval(self, lower, upper):
        return self._mix(self.noise.log_interval(lower, upper), self.amount.log_interval(lower, upper))

    def sample(self, key, sample_shape=()):
        zero_key, amount_key, noise_key = jax.random.split(key, 3)
        shape = tuple(sample_shape) + self.batch_shape
        draw = self.amount.base.sample(amount_key, sample_shape)
        draw = jnp.where(jax.random.bernoulli(zero_key, jnp.broadcast_to(self.zero, shape)), 0.0, draw)
        return draw + self.amount.scale * jax.random.normal(noise_key, shape)

    def mean(self):
        return (1.0 - self.zero) * self.amount.mean()


class Cured(Distribution):
    """A time that never comes with probability ``never``, and otherwise follows ``base``.

    The cure model of survival analysis (Boag, 1949; Farewell, 1982), and a
    conversion that may never happen: some patients are cured, some
    customers never buy, some machines never fail in the units watched. It is
    a defective distribution: ``cdf(t) = (1 - never) F(t)`` never reaches 1
    at any finite ``t``, and ``cdf(inf)`` is ``1 - never``, so ``P(Y > inf)``,
    ``1 - cdf(inf)``, is the probability of never. A censored row at ``t``
    scores ``never + (1 - never) S(t)``, an event its density times ``1 -
    never``. A draw is ``inf`` where it never happens, and a quantile above
    ``1 - never`` is ``inf``. ``mean`` and ``variance`` are those of the time
    given that it happens; the time itself has no finite mean.
    """

    base: Distribution
    never: jax.Array
    atomless: ClassVar[bool] = True
    defective: ClassVar[bool] = True

    def __init__(self, base: Distribution, never):
        self.base = base
        self.never = jnp.broadcast_to(jnp.asarray(never, dtype=jnp.float32), base.batch_shape)

    @property
    def batch_shape(self):
        return self.never.shape

    def _log_never(self):
        return jnp.log(jnp.clip(self.never, 1e-30, 1.0))

    def _log_happens(self):
        return jnp.log1p(-jnp.minimum(self.never, 1.0 - 1e-7))

    def _finite(self, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        return value, jnp.isposinf(value), jnp.where(jnp.isposinf(value), 0.0, value)

    def log_prob(self, value):
        return self._log_happens() + self.base.log_prob(value)

    def cdf(self, value):
        value, top, inner = self._finite(value)
        return (1.0 - self.never) * jnp.where(top, 1.0, self.base.cdf(inner))

    def log_cdf(self, value):
        value, top, inner = self._finite(value)
        return self._log_happens() + jnp.where(top, 0.0, self.base.log_cdf(inner))

    def log_cdf_left(self, value):
        return self.log_cdf(value)

    def log_survival(self, value):
        value, top, inner = self._finite(value)
        tail = jnp.where(top, -jnp.inf, self.base.log_survival(inner))
        return jnp.logaddexp(self._log_never(), self._log_happens() + tail)

    def log_survival_left(self, value):
        return self.log_survival(value)

    def sample(self, key, sample_shape=()):
        never_key, base_key = jax.random.split(key)
        shape = tuple(sample_shape) + self.batch_shape
        draw = self.base.sample(base_key, sample_shape)
        return jnp.where(jax.random.bernoulli(never_key, jnp.broadcast_to(self.never, shape)), jnp.inf, draw)

    def quantile(self, q):
        q = jnp.asarray(q, dtype=jnp.float32)
        happens = 1.0 - self.never
        level = jnp.clip(q / jnp.maximum(happens, 1e-30), 1e-7, 1.0 - 1e-7)
        return jnp.where(q < happens, self.base.quantile(level), jnp.inf)

    def mean(self):
        return self.base.mean()

    def variance(self):
        return self.base.variance()

    def _index(self, index):
        return Cured(self.base._index(index), self.never[index])


def hurdle_distribution(flat, severity: str = "lognormal") -> Hurdle:
    """The ``hurdle`` head from its flat projection ``batch + (3,)``: the zero logit, then two severity parameters.

    ``lognormal``: the location and raw scale of ``log y``. ``gamma``: the log
    mean and raw shape. Raw values go through ``softplus(raw) + 1e-4``.
    """
    from ashlar.predict import positive_scale

    flat = jnp.asarray(flat)
    if flat.shape[-1] != 3:
        raise ShapeError(f"a hurdle projection has width 3; received {flat.shape[-1]}.")
    zero = jax.nn.sigmoid(flat[..., 0])
    if severity == "lognormal":
        return Hurdle(zero, LogNormal(flat[..., 1], positive_scale(flat[..., 2])))
    if severity == "gamma":
        mean = jnp.exp(jnp.clip(flat[..., 1], -30.0, 30.0))
        return Hurdle(zero, Gamma(mean, positive_scale(flat[..., 2])))
    raise AshlarError(f"hurdle severity must be one of {HURDLE_SEVERITIES}; received {severity!r}.")
