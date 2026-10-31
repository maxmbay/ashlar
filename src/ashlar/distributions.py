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
from equinox import Module, field
from jax.scipy.special import betainc, gammaincc, ndtr, ndtri

from ashlar.errors import AshlarError, ShapeError

QUADRATURE_NODES = 512
_BISECTION_STEPS = 80


class Distribution(Module):
    """Base class. Subclasses set the parameters and the methods they support."""

    reparameterized: ClassVar[bool] = False
    support: ClassVar[str] = "real"

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

    def __getitem__(self, index):
        index = index if isinstance(index, tuple) else (index,)
        if any(item is Ellipsis or item is None for item in index):
            raise ShapeError("index a distribution's batch axes with integers and slices only.")
        if len(index) > len(self.batch_shape):
            raise ShapeError(f"index has {len(index)} axes; the batch shape is {self.batch_shape}.")
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


class Normal(Distribution):
    """Normal with location ``loc`` and standard deviation ``scale``."""

    loc: jax.Array
    scale: jax.Array
    reparameterized: ClassVar[bool] = True

    def __init__(self, loc, scale):
        self.loc, self.scale = _broadcast(loc, scale)

    @property
    def batch_shape(self):
        return self.loc.shape

    def log_prob(self, value):
        return _normal_log_prob(value, self.loc, self.scale)

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

    def cdf(self, value):
        t = (value - self.loc) / self.scale
        tail = 0.5 * betainc(0.5 * self.df, 0.5, self.df / (self.df + t * t))
        return jnp.where(t > 0, 1.0 - tail, tail)


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

    def sample(self, key, sample_shape=()):
        sample_shape = tuple(sample_shape)
        choice_key, draw_key = jax.random.split(key)
        choice = jax.random.categorical(choice_key, self.logits, axis=-1, shape=sample_shape + self.batch_shape)
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

    def quantile(self, q):
        return self.loc + self.scale * self.base.quantile(q)

    def covariance(self):
        cov = self.base.covariance()
        return cov * self.scale[..., :, None] * self.scale[..., None, :]

    def marginals(self):
        if not self.event_shape:
            return self
        return AffineTransformed(self.base.marginals(), self.loc, self.scale)


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

    def sample(self, key, sample_shape=()):
        return self.base.sample(key, sample_shape)

    def mean(self):
        return self.base.mean()

    def variance(self):
        return self.base.variance()

    def covariance(self):
        variance = self.variance()
        return variance[..., :, None] * jnp.eye(variance.shape[-1], dtype=variance.dtype)

    def marginals(self):
        return self.base


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


class Flow(Distribution):
    """The ``flow`` head: a rational-quadratic spline pushed forward from a standard normal.

    ``params`` is ``batch_shape + (parameters,)``, the unconstrained spline of
    ``ashlar.distributional``. The spline is monotone, so the quantile at
    ``q`` is the spline at the normal quantile of ``q``. Mean and variance
    are exact normal integrals on the identity tails plus a
    512-point midpoint rule inside the spline's range.
    """

    params: jax.Array
    reparameterized: ClassVar[bool] = True

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

    def cdf(self, value):
        from ashlar.distributional import flow_inverse

        params, value = self._params_for(jnp.asarray(value, dtype=self.params.dtype))
        base, _logdet = flow_inverse(params, value)
        return ndtr(base)

    def quantile(self, q):
        return self._forward(ndtri(jnp.asarray(q, dtype=self.params.dtype)))

    def _moments(self):
        """First two moments. The spline is the identity outside ``[-bound, bound]``,
        so the tails are closed-form normal integrals; the inside is a midpoint rule."""
        from ashlar.distributional import FLOW_BOUND

        bound = float(FLOW_BOUND)
        width = 2.0 * bound / QUADRATURE_NODES
        nodes = -bound + width * (jnp.arange(QUADRATURE_NODES, dtype=self.params.dtype) + 0.5)
        weights = width * jnp.exp(-0.5 * jnp.square(nodes)) / math.sqrt(2 * math.pi)
        values = self._forward(nodes.reshape((-1,) + (1,) * len(self.batch_shape)))
        weights = weights.reshape((-1,) + (1,) * len(self.batch_shape))
        density = math.exp(-0.5 * bound * bound) / math.sqrt(2 * math.pi)
        tail_mass = 0.5 * math.erfc(bound / math.sqrt(2.0))
        first = jnp.sum(weights * values, axis=0)
        second = jnp.sum(weights * jnp.square(values), axis=0) + 2.0 * (bound * density + tail_mass)
        return first, second

    def mean(self):
        return self._moments()[0]

    def variance(self):
        first, second = self._moments()
        return jnp.maximum(second - jnp.square(first), 0.0)


class Histogram(Distribution):
    """The ``histogram`` head: bin probabilities over fixed edges on the standardized axis.

    The density is flat inside each bin. ``log_prob`` reads a value outside the
    edges in the nearest end bin, as the training loss does, while draws, the
    CDF, and the quantiles stay inside the edges.
    """

    logits: jax.Array

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
        return betainc(size, k + 1.0, size / (size + self.mean_count))

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
    density = jnp.exp(dist.log_prob(x0))
    gap = dist.cdf(x0) - q
    newton = jnp.where(density > 1e-12, gap / jnp.maximum(density, 1e-12), 0.0)
    refined = x0 - newton + jax.lax.stop_gradient(newton)
    refined = jnp.where(level <= 0.0, -jnp.inf, refined)
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

    For a count the PIT is spread uniformly over the jump of the CDF at the
    observed value, so a discrete head still gives a uniform score.
    """
    upper = dist.cdf(value)
    if dist.support != "nonnegative_integer":
        return upper
    lower = dist.cdf(jnp.asarray(value) - 1.0)
    return lower + jax.random.uniform(key, jnp.shape(upper), dtype=upper.dtype) * (upper - lower)


def head_distribution(kind: str, flat, horizon: int = 1) -> Distribution:
    """The distribution of one ``ashlar.distributional`` head from its flat projection.

    ``flat`` is ``batch + (width,)`` on the standardized axis (the raw count
    for a count head). With ``horizon > 1`` the steps of a forecast become the
    last batch axis; each step is its own marginal, and nothing is claimed
    about the dependence between steps.
    """
    from ashlar.distributional import QUANTILE_LEVELS, decode

    flat = jnp.asarray(flat)
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
    if kind == "poisson":
        return Poisson(*(shaped(part) for part in decoded))
    if kind == "negative_binomial":
        return NegativeBinomial(*(shaped(part) for part in decoded))
    if kind == "quantile":
        return QuantileLevels(values=shaped(decoded), levels=tuple(QUANTILE_LEVELS))
    raise AshlarError(f"the {kind} head has no distribution.")
