"""Monotone constraints: features that can only push the outcome one way.

``ASRRegressor(monotone={"price": -1, "bonus_malus": 1})`` keeps the named
numeric features out of the trunk and adds them back through a term of
their own, a sum over the features of non-decreasing piecewise-linear
shapes with non-negative weights. The term is added to the head's
outputs whose rise moves every quantile up: a location, a log mean, the
median and upper slopes of the ``quantile_spline`` head, and, with its
sign flipped, the spline's lower slopes. The other outputs read only the
other features. So each constrained feature moves the whole distribution
one way, first-order stochastic dominance: ``P(y > t | x)`` does not
fall as a feature with ``+1`` rises, for every ``t``, and every quantile
with it. The price is that a constrained feature does not interact with
the others: its effect is the same shape on every row, on the scale of
the outputs it reaches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module, field

from ashlar.distributional import SPLINE_PIECES
from ashlar.errors import AshlarError, ShapeError

# The tasks whose outputs move every quantile one way, so a monotone term gives stochastic dominance.
MONOTONE_TASKS = ("regression", "gaussian", "student_t", "quantile_spline", "poisson", "negative_binomial")
MONOTONE_BINS = 16
# A shape's rise across a bin is softplus(MONOTONE_GAIN * weight). Every shape starts nearly flat, rising
# MONOTONE_START in all across its bins, so a feature on a wide scale does not start far from the median; the
# gain lets Adam's steps of about the learning rate grow it within a few hundred steps.
MONOTONE_GAIN = 4.0
MONOTONE_START = 0.1


def output_signs(task: str, width: int) -> tuple[int, ...]:
    """For each head output, +1 where a rise moves every quantile up, -1 where a fall does, 0 where it is left alone."""
    if task == "regression":
        return (1,) * int(width)
    if task in ("gaussian", "student_t", "poisson", "negative_binomial"):
        # The location, or the log mean, comes first; scales, degrees of freedom, and the structural zero stay put.
        return (1,) + (0,) * (int(width) - 1)
    if task == "quantile_spline":
        if int(width) != 1 + 2 * SPLINE_PIECES:
            raise ShapeError(f"the quantile_spline head has {1 + 2 * SPLINE_PIECES} outputs; received {width}.")
        return (1,) + (1,) * SPLINE_PIECES + (-1,) * SPLINE_PIECES
    raise AshlarError(f"monotone constraints need one of {', '.join(MONOTONE_TASKS)}; this model is {task}.")


def check_monotone(monotone) -> dict:
    """``{feature: +1 or -1}``, checked."""
    if monotone is None:
        return {}
    if not isinstance(monotone, dict) or not monotone:
        raise AshlarError("monotone must be a dict of feature names to +1 (rising) or -1 (falling).")
    checked = {}
    for name, direction in monotone.items():
        if isinstance(direction, bool) or direction not in (1, -1):
            raise AshlarError(f"monotone[{name!r}] must be +1 or -1; received {direction!r}.")
        checked[str(name)] = int(direction)
    return checked


def monotone_edges(numeric_z, columns, bins: int = MONOTONE_BINS) -> np.ndarray:
    """Bin edges per constrained column, training quantiles of its scaled values, shape ``(features, bins + 1)``."""
    from ashlar.layers import quantile_edges

    if numeric_z is None:
        return np.tile(np.linspace(-1.0, 1.0, int(bins) + 1, dtype=np.float32), (len(columns), 1))
    return quantile_edges(np.asarray(numeric_z)[:, list(columns)], int(bins))


class MonotoneTerm(Module):
    """``(batch, members, outputs)``: each output's sum of non-decreasing shapes of the constrained features.

    A feature's shape is ``sum_b rise_b * ramp_b(direction * z)``, with
    ``rise_b = softplus(MONOTONE_GAIN * weight_b)`` and ``ramp_b`` the fill
    fraction of bin ``b`` (open below in the first bin and above in the
    last), so it is piecewise linear, rises without bound at the ends, and
    never falls. The bins are training quantiles, so the shape spends its
    resolution where the rows are. ``signs`` flips it for the outputs that must fall.
    """

    edges: jax.Array
    weight: jax.Array
    columns: tuple[int, ...] = field(static=True)
    directions: tuple[int, ...] = field(static=True)
    signs: tuple[int, ...] = field(static=True)

    def __init__(self, edges, members: int, columns, directions, signs):
        edges = jnp.asarray(edges, dtype=jnp.float32)
        if edges.ndim != 2 or edges.shape[0] != len(columns):
            raise ShapeError(f"monotone edges must be (features, bins + 1) for {len(columns)} features; received {edges.shape}.")
        self.columns = tuple(int(column) for column in columns)
        self.directions = tuple(int(direction) for direction in directions)
        self.signs = tuple(int(sign) for sign in signs)
        # Edges are fitted on z; a falling feature reads -z, whose edges are the reversed negatives.
        flip = jnp.asarray(self.directions, dtype=jnp.float32)[:, None] < 0
        self.edges = jnp.where(flip, -edges[:, ::-1], edges)
        bins = edges.shape[1] - 1
        start = float(np.log(np.expm1(MONOTONE_START / bins))) / MONOTONE_GAIN
        self.weight = jnp.full((int(members), len(columns), len(signs), bins), start, dtype=jnp.float32)

    def encode(self, numeric_z):
        x = numeric_z[:, list(self.columns)] * jnp.asarray(self.directions, dtype=numeric_z.dtype)
        edges = jax.lax.stop_gradient(self.edges)
        left = edges[:, :-1]
        width = edges[:, 1:] - left
        empty = width <= 0
        fraction = (x[..., None] - left) / jnp.where(empty, 1.0, width)
        bins = left.shape[1]
        lower = jnp.where(jnp.arange(bins) == 0, -jnp.inf, 0.0)
        upper = jnp.where(jnp.arange(bins) == bins - 1, jnp.inf, 1.0)
        return jnp.where(empty, 0.0, jnp.clip(fraction, lower, upper))

    def __call__(self, numeric_z):
        ramps = self.encode(numeric_z)
        shapes = jnp.einsum("bjn,kjpn->bkp", ramps, jax.nn.softplus(MONOTONE_GAIN * self.weight))
        return shapes * jnp.asarray(self.signs, dtype=shapes.dtype)

    def parts(self, numeric_z):
        """``(batch, members, features, outputs)``: each constrained feature's shape; they sum to the term."""
        ramps = self.encode(numeric_z)
        shapes = jnp.einsum("bjn,kjpn->bkjp", ramps, jax.nn.softplus(MONOTONE_GAIN * self.weight))
        return shapes * jnp.asarray(self.signs, dtype=shapes.dtype)

    def without(self, numeric_z):
        """``numeric_z`` with the constrained columns set to 0, the trunk's input."""
        return numeric_z.at[:, list(self.columns)].set(0.0)
