"""Decisions from predictive distributions, and training through their cost.

Most users of a predictive distribution act on one number: an order, a
capacity, a reserve, a threshold. This module turns a ``Distribution`` (from
``predict_dist`` or ``forward_dist``) into that number, and gives an
objective that trains a head on its likelihood plus the realised cost of the
decision it will be used for.

Everything here is traceable JAX on the distribution object, so it composes
with ``jax.jit`` and ``jax.grad``.

- ``critical_ratio(under, over)`` and ``newsvendor_order(dist, under, over)``:
  the cost-optimal order when a unit short costs ``under`` and a unit left
  over costs ``over`` is the ``under / (under + over)`` quantile.
- ``newsvendor_cost(y, order, under, over)``: the realised cost per row.
- ``exceedance(dist, threshold)``: ``P(y > threshold)``; with ``weights``,
  of a weighted sum of a joint outcome's columns.
- ``cvar(dist, level)``: the mean of the outcome above its ``level`` quantile.
- ``expected_utility(dist, utility, key)`` and ``best_action(dist, actions,
  cost, key)``: any utility or cost, by Monte Carlo over draws.
- ``newsvendor_objective(task, under, over)``: the head's negative
  log-likelihood plus ``cost`` times the newsvendor cost of the mixture's
  order, for ``objective=`` on a univariate continuous head.

``ASRModel.predict_order(X, under, over)`` is the order from a fitted model;
after ``calibrate`` it is the calibrated quantile, so the share of rows whose
demand it covers (the service level) holds at the critical ratio on new rows.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp

from ashlar.errors import AshlarError

DECISION_TASKS = ("gaussian", "mixture", "flow", "histogram", "student_t")
CVAR_POINTS = 64


def critical_ratio(under: float, over: float) -> float:
    """``under / (under + over)``: the service level that minimises the expected newsvendor cost."""
    under, over = float(under), float(over)
    if not (math.isfinite(under) and math.isfinite(over)) or under <= 0.0 or over <= 0.0:
        raise AshlarError("the newsvendor costs under= and over= must be positive and finite.")
    return under / (under + over)


def newsvendor_order(dist, under: float, over: float):
    """The cost-optimal order per row: the critical-ratio quantile of ``dist``."""
    return dist.quantile(critical_ratio(under, over))


def newsvendor_cost(y, order, under: float, over: float):
    """Realised cost per row: ``under`` per unit of ``y`` above ``order``, ``over`` per unit below."""
    y = jnp.asarray(y)
    order = jnp.asarray(order)
    return float(under) * jnp.maximum(y - order, 0.0) + float(over) * jnp.maximum(order - y, 0.0)


def _univariate(dist, weights):
    if weights is not None:
        return dist.project(jnp.asarray(weights))
    if dist.event_shape:
        raise AshlarError("a joint outcome needs weights= to name the weighted sum of its columns to decide on.")
    return dist


def exceedance(dist, threshold, weights=None):
    """``P(y > threshold)`` per row; with ``weights``, of ``y @ weights`` on a joint head."""
    return 1.0 - _univariate(dist, weights).cdf(jnp.asarray(threshold, dtype=jnp.float32))


def cvar(dist, level: float = 0.95, weights=None, points: int = CVAR_POINTS):
    """Conditional value at risk: the mean outcome above the ``level`` quantile, per row.

    Exact up to quadrature: the mean of the quantile function at ``points``
    midpoints of ``(level, 1)``, so it differentiates through the quantile's
    implicit gradient. With ``weights``, of a weighted sum of a joint
    outcome's columns (exact for the normal-based joint heads; use draws and
    ``expected_utility`` for the flow).
    """
    level = float(level)
    if not 0.0 < level < 1.0:
        raise AshlarError("the CVaR level must be in (0, 1).")
    base = _univariate(dist, weights)
    grid = level + (1.0 - level) * (jnp.arange(int(points), dtype=jnp.float32) + 0.5) / int(points)
    return jnp.mean(jnp.stack([base.quantile(float(u)) for u in grid.tolist()], axis=0), axis=0)


def expected_utility(dist, utility: Callable, key, n: int = 1000):
    """``E[utility(y)]`` per row by ``n`` draws. ``utility`` maps draws ``(n, rows, ...)`` to ``(n, rows)``."""
    draws = dist.sample(key, (int(n),))
    return jnp.mean(utility(draws), axis=0)


def best_action(dist, actions, cost: Callable, key, n: int = 1000):
    """The action of least expected cost per row, from a grid of candidates.

    ``actions`` is ``(k,)`` or ``(k, rows)``; ``cost(action, draws)`` maps an
    action broadcast against draws ``(n, rows, ...)`` to costs ``(n, rows)``.
    The same draws score every candidate. Returns ``(best, expected_costs)``
    with ``best`` of shape ``(rows,)`` and ``expected_costs`` ``(k, rows)``.
    """
    draws = dist.sample(key, (int(n),))
    actions = jnp.asarray(actions)
    rows = draws.shape[1]
    grid = jnp.broadcast_to(actions.reshape(actions.shape[0], -1), (actions.shape[0], rows))
    costs = jax.vmap(lambda action: jnp.mean(cost(action[None, :], draws), axis=0))(grid)
    choice = jnp.argmin(costs, axis=0)
    return jnp.take_along_axis(grid, choice[None, :], axis=0)[0], costs


def newsvendor_objective(task: str, under: float, over: float, *, nll: float = 1.0, cost: float = 1.0):
    """A training objective: ``nll`` times the head's loss plus ``cost`` times the newsvendor cost.

    Pass it as ``objective=`` to a univariate continuous head (``gaussian``,
    ``mixture``, ``flow``, ``histogram``, ``student_t``)::

        model = ASRRegressor(task="mixture", objective=newsvendor_objective("mixture", under=9, over=1))

    The cost is that of ordering the critical-ratio quantile of the members'
    equal-weight mixture, the distribution ``predict_dist`` returns, on the
    standardized target, so its gradient reaches the parameters through the
    quantile's implicit gradient. The model stays a full distribution; with
    ``nll=0`` it is trained for the decision alone. The cost is divided by
    ``under + over``, so it is the pinball loss at the critical ratio and its
    size does not depend on the scale of the costs. A model trained on it
    saves and loads like any other.
    """
    return NewsvendorObjective(task, under, over, nll=nll, cost=cost)


class NewsvendorObjective:
    """``objective(raw, y, sample_weight)`` for ``newsvendor_objective``; see there."""

    def __init__(self, task: str, under: float, over: float, *, nll: float = 1.0, cost: float = 1.0):
        from ashlar.objectives import make_distribution_objective

        if task not in DECISION_TASKS:
            raise AshlarError(f"decision training needs a univariate continuous head: {', '.join(DECISION_TASKS)}.")
        self.ratio = critical_ratio(under, over)
        nll, cost = float(nll), float(cost)
        if nll < 0.0 or cost < 0.0 or not (math.isfinite(nll) and math.isfinite(cost)) or nll + cost == 0.0:
            raise AshlarError("nll= and cost= must be finite and non-negative, and one must be positive.")
        self.task, self.under, self.over, self.nll, self.cost = task, float(under), float(over), nll, cost
        self._likelihood = make_distribution_objective(task)

    def ashlar_config(self) -> dict:
        return {"task": self.task, "under": self.under, "over": self.over, "nll": self.nll, "cost": self.cost}

    def __repr__(self) -> str:
        return f"newsvendor_objective({self.task!r}, under={self.under}, over={self.over}, nll={self.nll}, cost={self.cost})"

    def __call__(self, raw, y, sample_weight):
        from ashlar.distributions import Mixture, head_distribution
        from ashlar.weights import weighted_mean

        target = jnp.asarray(y, dtype=raw.dtype).reshape(-1)
        order = Mixture(head_distribution(self.task, raw)).quantile(self.ratio)
        ratio = self.ratio
        pinball = ratio * jnp.maximum(target - order, 0.0) + (1.0 - ratio) * jnp.maximum(order - target, 0.0)
        loss = self.cost * weighted_mean(pinball, sample_weight)
        if self.nll > 0.0:
            loss = loss + self.nll * self._likelihood(raw, y, sample_weight)
        return loss
