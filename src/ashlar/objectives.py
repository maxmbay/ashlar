"""Member-wise objectives with one documented weight reduction.

TabM trains the members independently: the optimized value is the mean of the
member losses, not the loss of the averaged prediction. Sample weights are
applied once, after that member mean, with reduction ``batch_weight_sum``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from ashlar.errors import AshlarError, ShapeError
from ashlar.predict import mean_logits, mean_probabilities, positive_scale
from ashlar.weights import weighted_mean


def _target_events(y, n_events: int):
    y = jnp.asarray(y)
    if n_events == 1 and y.ndim == 1:
        return y[:, None]
    if y.ndim == 2 and y.shape[1] == n_events:
        return y
    raise ShapeError(f"y must have shape (batch,) or (batch, {n_events}); received {y.shape}.")


def regression_objective(raw, y, sample_weight):
    """Mean squared error, averaged over members and events, then weighted over the batch.

    A target shaped ``(batch, member)`` or ``(batch, member, event)`` is the
    unshared-batch case: each member has its own rows. A target without a
    member axis is broadcast across members.
    """
    if raw.ndim != 3:
        raise ShapeError(f"point head output must have shape (batch, member, event); received {raw.shape}.")
    y = jnp.asarray(y)
    sample_weight = jnp.asarray(sample_weight)
    if y.ndim >= 2 and y.shape[0] == raw.shape[0] and y.shape[1] == raw.shape[1]:
        target = y if y.ndim == 3 else y[..., None]
        if target.shape[-1] != raw.shape[-1]:
            raise ShapeError(
                f"unshared regression target event axis must be {raw.shape[-1]}; received {target.shape}."
            )
        per_member = jnp.mean(jnp.square(raw - target), axis=-1)
        if sample_weight.ndim == 1:
            return weighted_mean(jnp.mean(per_member, axis=1), sample_weight)
        return weighted_mean(per_member.reshape(-1), sample_weight.reshape(-1))
    target = _target_events(y, raw.shape[-1])
    per_member = jnp.mean(jnp.square(raw - target[:, None, :]), axis=-1)
    return weighted_mean(jnp.mean(per_member, axis=1), sample_weight)


def classification_objective(raw, y, sample_weight):
    """Mean member cross-entropy. ``raw`` is ``(batch, member, class)`` and ``y`` is class indices."""
    if raw.ndim != 3:
        raise ShapeError(f"logits must have shape (batch, member, class); received {raw.shape}.")
    if y.ndim != 1:
        raise ShapeError(f"classification y must have shape (batch,); received {y.shape}.")
    log_prob = jax_log_softmax(raw)
    index = y.astype(jnp.int32)[:, None, None]
    chosen = jnp.take_along_axis(log_prob, index, axis=-1)[..., 0]
    return weighted_mean(-jnp.mean(chosen, axis=1), sample_weight)


def make_candidate_objective(task: str):
    """Negative log-probability of each row's set of classes: ``y`` is a ``(rows, classes)`` 0/1 mask.

    ``raw`` is logits for ``classification`` and log-probabilities from the
    cumulative logit for ``ordinal``. A row whose set holds every class adds
    nothing.
    """
    import jax

    def objective(raw, y, sample_weight):
        if raw.ndim != 3:
            raise ShapeError(f"class scores must have shape (batch, member, class); received {raw.shape}.")
        log_prob = jax_log_softmax(raw) if task == "classification" else raw
        allowed = jnp.where(jnp.asarray(y)[:, None, :] > 0, 0.0, -jnp.inf)
        chosen = jax.nn.logsumexp(log_prob + allowed, axis=-1)
        return weighted_mean(-jnp.mean(chosen, axis=1), sample_weight)

    return objective


def classification_accuracy(raw, y, sample_weight, *, policy: str = "mean_probabilities"):
    """Weighted accuracy. Higher is better.

    ``sample_weight`` is the effective row weight already used by the training
    objective. This function does not multiply by a class weight again.
    ``policy`` matches ``predict_proba``: ``mean_probabilities`` or ``mean_logits``.
    """
    if policy == "mean_probabilities":
        probabilities = mean_probabilities(raw)
    elif policy == "mean_logits":
        probabilities = mean_logits(raw)
    else:
        raise ShapeError("classification accuracy policy must be 'mean_probabilities' or 'mean_logits'.")
    predicted = jnp.argmax(probabilities, axis=-1)
    correct = (predicted == jnp.asarray(y).astype(jnp.int32)).astype(probabilities.dtype)
    return weighted_mean(correct, sample_weight)


def jax_log_softmax(raw):
    import jax.nn as jnn

    return jnn.log_softmax(raw, axis=-1)


def ordered_thresholds(cut_raw):
    """Strictly increasing cutpoints. ``cut_raw`` is ``(member, class - 1)``.

    The first column is the lowest cut. Each later column is a softplus gap,
    so the cuts cannot cross.
    """
    if cut_raw.ndim != 2 or cut_raw.shape[-1] < 1:
        raise ShapeError(f"ordinal cutpoints must have shape (member, class - 1); received {cut_raw.shape}.")
    if cut_raw.shape[-1] == 1:
        return cut_raw
    gaps = jax_softplus(cut_raw[..., 1:]) + jnp.asarray(1e-4, dtype=cut_raw.dtype)
    higher = cut_raw[..., :1] + jnp.cumsum(gaps, axis=-1)
    return jnp.concatenate([cut_raw[..., :1], higher], axis=-1)


def jax_softplus(values):
    import jax.nn as jnn

    return jnn.softplus(values)


def _log_sigmoid_diff(hi, lo):
    """log(sigmoid(hi) - sigmoid(lo)) for hi > lo."""
    import jax.nn as jnn

    log_hi = jnn.log_sigmoid(hi)
    log_lo = jnn.log_sigmoid(lo)
    delta = jnp.minimum(log_lo - log_hi, jnp.asarray(-1e-6, dtype=log_hi.dtype))
    return log_hi + jnp.log1p(-jnp.exp(delta))


def ordinal_log_probabilities(latent, cut_raw):
    """Class log-probabilities from a cumulative logit.

    ``latent`` is ``(batch, member)``. One shared latent per member, plus that
    member's ordered cuts, gives P(Y = class). A larger latent moves mass
    toward the higher class. This is not a softmax over independent logits.
    """
    import jax.nn as jnn

    latent = jnp.asarray(latent)
    cut_raw = jnp.asarray(cut_raw)
    if latent.ndim != 2:
        raise ShapeError(f"ordinal latent must have shape (batch, member); received {latent.shape}.")
    theta = ordered_thresholds(cut_raw)
    if theta.shape[0] != latent.shape[1]:
        raise ShapeError(
            f"ordinal cuts have {theta.shape[0]} members and the latent has {latent.shape[1]}."
        )
    hi = theta[None, :, :] - latent[:, :, None]
    first = jnn.log_sigmoid(hi[..., :1])
    last = jnn.log_sigmoid(-hi[..., -1:])
    if hi.shape[-1] == 1:
        log_probs = jnp.concatenate([first, last], axis=-1)
    else:
        middle = _log_sigmoid_diff(hi[..., 1:], hi[..., :-1])
        log_probs = jnp.concatenate([first, middle, last], axis=-1)
    return log_probs - jnn.logsumexp(log_probs, axis=-1, keepdims=True)


def ordinal_objective(log_probs, y, sample_weight):
    """Mean member negative log-probability of the true ordered class.

    ``log_probs`` is ``(batch, member, class)`` from the cumulative logit.
    ``y`` is the class index, not the original label value.
    """
    if log_probs.ndim != 3:
        raise ShapeError(f"ordinal log-probabilities must have shape (batch, member, class); received {log_probs.shape}.")
    if y.ndim != 1:
        raise ShapeError(f"ordinal y must have shape (batch,); received {y.shape}.")
    index = y.astype(jnp.int32)[:, None, None]
    chosen = jnp.take_along_axis(log_probs, index, axis=-1)[..., 0]
    return weighted_mean(-jnp.mean(chosen, axis=1), sample_weight)


def ordinal_expectation_squared_error(log_probs, y, sample_weight, class_values):
    """Squared error of the expected label, the same readout as ``predict_score``.

    Probabilities are the mean of the member probabilities. The expected label
    is those probabilities times ``class_values``. ``y`` is the class index
    into that same vector.
    """
    if log_probs.ndim != 3:
        raise ShapeError(f"ordinal log-probabilities must have shape (batch, member, class); received {log_probs.shape}.")
    if y.ndim != 1 or y.shape[0] != log_probs.shape[0]:
        raise ShapeError(
            f"ordinal y must have shape (batch,) matching {log_probs.shape[0]} rows; received {y.shape}."
        )
    values = jnp.asarray(class_values, dtype=log_probs.dtype)
    if values.ndim != 1 or values.shape[0] != log_probs.shape[-1]:
        raise ShapeError(
            f"class values must have one entry per class; received {values.shape} for {log_probs.shape[-1]} classes."
        )
    expected = mean_probabilities(log_probs) @ values
    target = values[jnp.asarray(y, dtype=jnp.int32)]
    return weighted_mean(jnp.square(expected - target), sample_weight)


def make_ordinal_expectation_objective(class_values, weight: float):
    """Class negative log-likelihood plus ``weight`` times the expected-label squared error."""
    coef = float(weight)
    values = jnp.asarray(class_values, dtype=jnp.float32)

    def objective(log_probs, y, sample_weight):
        likelihood = ordinal_objective(log_probs, y, sample_weight)
        penalty = ordinal_expectation_squared_error(log_probs, y, sample_weight, values)
        return likelihood + jnp.asarray(coef, dtype=likelihood.dtype) * penalty

    return objective


def gaussian_bin_probabilities(class_index, n_classes: int, variance: float):
    """Truncated-Gaussian mass on equal-width class bins.

    Bin ``i`` covers ``[i - 1/2, i + 1/2]`` in class-index units. The Gaussian
    is centered on the observed class and truncated to the outer edges. This is
    the HL-Gaussian target of Imani and White for bins that already sit on the
    integer labels.
    """
    import jax.scipy.stats

    index = jnp.asarray(class_index, dtype=jnp.float32).reshape(-1)
    n_bins = int(n_classes)
    bound = float(variance)
    if n_bins < 2:
        raise ShapeError("a histogram target needs at least 2 classes.")
    if not math.isfinite(bound) or bound <= 0.0:
        raise ShapeError("histogram variance must be positive and finite.")
    edges = jnp.arange(n_bins + 1, dtype=jnp.float32) - jnp.asarray(0.5, dtype=jnp.float32)
    scale = jnp.sqrt(jnp.asarray(bound, dtype=jnp.float32))
    cdf = jax.scipy.stats.norm.cdf((edges[None, :] - index[:, None]) / scale)
    bins = cdf[:, 1:] - cdf[:, :-1]
    total = jnp.clip(cdf[:, -1:] - cdf[:, :1], jnp.asarray(1e-12, dtype=cdf.dtype))
    return bins / total


def ordinal_histogram_objective(log_probs, y, sample_weight, variance: float):
    """Cross-entropy from a truncated Gaussian on the class bins to ``log_probs``.

    ``log_probs`` is ``(batch, member, class)``. ``y`` is the class index.
    Members are averaged, then the batch is reduced with ``sample_weight``.
    """
    if log_probs.ndim != 3:
        raise ShapeError(f"ordinal log-probabilities must have shape (batch, member, class); received {log_probs.shape}.")
    if y.ndim != 1 or y.shape[0] != log_probs.shape[0]:
        raise ShapeError(
            f"ordinal y must have shape (batch,) matching {log_probs.shape[0]} rows; received {y.shape}."
        )
    targets = gaussian_bin_probabilities(y, int(log_probs.shape[-1]), variance)
    negative = -jnp.sum(targets[:, None, :] * log_probs, axis=-1)
    return weighted_mean(jnp.mean(negative, axis=1), sample_weight)


def make_ordinal_histogram_objective(variance: float):
    """Bind one HL-Gaussian variance. The returned function is the training objective."""
    bound = float(variance)

    def objective(log_probs, y, sample_weight):
        return ordinal_histogram_objective(log_probs, y, sample_weight, bound)

    return objective


def normal_objective(raw, y, sample_weight):
    """Mean member negative log-density of a diagonal normal, summed across events.

    Summing events is the joint density of independent coordinates. Members are
    averaged, not mixed, during training.
    """
    if raw.ndim != 4 or raw.shape[-1] != 2:
        raise ShapeError(f"normal raw output must have shape (batch, member, event, 2); received {raw.shape}.")
    target = _target_events(y, raw.shape[2])
    loc = raw[..., 0]
    scale = positive_scale(raw[..., 1])
    z = (target[:, None, :] - loc) / scale
    nll = 0.5 * (jnp.log(2 * jnp.pi) + 2 * jnp.log(scale) + jnp.square(z))
    per_example = jnp.sum(jnp.mean(nll, axis=1), axis=-1)
    return weighted_mean(per_example, sample_weight)


def make_distribution_objective(kind: str):
    """Mean member loss of a distributional head on a standardized target, weighted over rows.

    Gaussian, mixture, and flow use the negative log-density; histogram adds
    its small location term; quantile is the pinball loss at 0.1, 0.5, 0.9.
    Members are averaged, not mixed, during training, as in TabM.
    """
    from ashlar.distributional import decode, row_loss

    def objective(raw, y, sample_weight):
        if raw.ndim != 3:
            raise ShapeError(f"distribution head output must be (batch, member, width); received {raw.shape}.")
        batch, members, width = raw.shape
        decoded = decode(kind, raw.reshape(batch * members, width), 1)
        target = jnp.repeat(jnp.asarray(y, dtype=raw.dtype).reshape(-1), members)[:, None]
        losses = row_loss(kind, decoded, target).reshape(batch, members)
        return weighted_mean(jnp.mean(losses, axis=1), sample_weight)

    return objective


def make_joint_objective(n_events: int, family: str = "normal", *, components: int | None = None, columns=None):
    """Mean member negative log-density of a joint head, weighted over rows.

    ``family`` is ``normal`` for ``multivariate_gaussian``, ``mixture`` for
    ``multivariate_mixture`` (with its number of ``components``), or
    ``flow`` for ``multivariate_flow``, or ``chain`` with its ``columns``.

    ``y`` is the standardized ``(batch, event)`` target. The density is the
    full joint one, so the loss rewards getting the dependence between
    columns right, not only each column.
    """

    def objective(raw, y, sample_weight):
        context = raw[0] if family in ("flow", "chain") else raw
        if context.ndim != 3:
            raise ShapeError(f"joint head output must be (batch, member, width); received {context.shape}.")
        target = _target_events(y, int(n_events))
        dist = joint_member_distribution(family, raw, int(n_events), components, columns=columns)
        losses = -dist.log_prob(target[:, None, :])
        return weighted_mean(jnp.mean(losses, axis=1), sample_weight)

    return objective


def joint_member_distribution(family: str, raw, n_events: int, components: int | None = None, columns=None):
    """Each member's joint distribution for every row, batch ``(rows, members)``, on the standardized target.

    ``family`` is ``normal`` (``multivariate_gaussian``), ``mixture``
    (``multivariate_mixture``, with its number of ``components``), ``flow``
    (``multivariate_flow``), or ``chain`` (``chain``, with its ``columns``
    from ``ashlar.chain.chain_columns``).
    """
    from ashlar.distributional import MIXTURE_COMPONENTS
    from ashlar.distributions import (
        joint_flow_distribution,
        joint_mixture_distribution,
        joint_normal_distribution,
    )

    if family == "chain":
        from ashlar.chain import chain_distribution

        return chain_distribution(raw, columns)
    if family == "flow":
        return joint_flow_distribution(raw, n_events)
    if family == "mixture":
        return joint_mixture_distribution(raw, n_events, MIXTURE_COMPONENTS if components is None else int(components))
    return joint_normal_distribution(raw, n_events)


# The proper scores a joint head trains on, alone or blended: its log density, and two read from draws.
SCORES = ("nll", "energy", "variogram")
# Reparameterized draws per row and member when a score is estimated from draws; a mixture splits them over its components.
SCORE_DRAWS = 16
# The variogram score's order. Scheuerer and Hamill (2015) found 0.5 the most sensitive to a wrong correlation.
VARIOGRAM_ORDER = 0.5
# Under the root of a squared distance, so a gradient stays finite where two values meet.
_DISTANCE_FLOOR = 1e-12
# Under the variogram's |a|^order, the same for its difference of two columns, on the standardized scale.
_VARIOGRAM_FLOOR = 1e-6


def score_weights(objective) -> dict | None:
    """``{score: weight}`` of an objective given by name or as a blend; ``None`` for a callable.

    ``None`` is the log density alone, ``{"nll": 1.0}``. A name from
    ``SCORES`` is that score alone, and a dict blends them with non-negative
    weights. The variogram score compares only differences between columns,
    so it is proper but not strictly proper, and it trains only in a blend
    with ``nll`` or ``energy``.
    """
    if objective is None:
        return {"nll": 1.0}
    if callable(objective):
        return None
    if isinstance(objective, str):
        weights = {objective: 1.0}
    elif isinstance(objective, dict):
        weights = dict(objective)
    else:
        raise AshlarError(
            f"objective is a callable, a score name ({', '.join(SCORES)}), or a blend {{name: weight}}; received {objective!r}."
        )
    unknown = sorted(str(name) for name in weights if name not in SCORES)
    if unknown:
        raise AshlarError(f"objective scores are {', '.join(SCORES)}; received {', '.join(unknown)}.")
    try:
        weights = {name: float(value) for name, value in weights.items()}
    except (TypeError, ValueError) as exc:
        raise AshlarError(f"each score's weight is a number; received {objective!r}.") from exc
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()) or not any(weights.values()):
        raise AshlarError(f"score weights must be finite and non-negative, and one must be positive; received {objective!r}.")
    weights = {name: value for name, value in weights.items() if value > 0.0}
    if "variogram" in weights and not ({"nll", "energy"} & set(weights)):
        raise AshlarError(
            "the variogram score compares only differences between columns, so it cannot place them on its own: "
            "a shift of every column leaves it unchanged. Blend it with 'nll' or 'energy', "
            "such as {'energy': 1.0, 'variogram': 1.0}."
        )
    return weights


def _distance(a, b):
    """Euclidean distance over the last axis, with a floor under the root so its gradient stays finite."""
    return jnp.sqrt(jnp.sum(jnp.square(a - b), axis=-1) + _DISTANCE_FLOOR)


def score_draws(dist, key, draws: int = SCORE_DRAWS):
    """Reparameterized draws for a score, grouped by mixture component, and each component's probability.

    Returns ``points`` of shape ``(components, n) + batch_shape + event_shape``
    and ``probabilities`` of shape ``batch_shape + (components,)``. A
    ``Mixture`` is drawn from every component, ``n = max(2, ceil(draws /
    components))`` each, rather than through its discrete choice, so a score's
    gradient reaches the component probabilities. Any other distribution is
    one component with ``n = draws``.
    """
    from ashlar.distributions import Mixture

    if isinstance(dist, Mixture):
        count = int(dist.logits.shape[-1])
        each = max(2, -(-int(draws) // count))
        points = dist.components.sample(key, (each,))
        axis = 1 + len(dist.batch_shape)
        return jnp.moveaxis(points, axis, 0), jnp.exp(dist.logits)
    points = dist.sample(key, (max(2, int(draws)),))
    return points[None], jnp.ones(tuple(dist.batch_shape) + (1,), dtype=points.dtype)


def energy_score(points, probabilities, y):
    """Unbiased estimate of the energy score ``E‖X − y‖ − ½ E‖X − X′‖``, shape ``batch``.

    ``points`` and ``probabilities`` come from ``score_draws``; ``y``
    broadcasts against ``batch + (d,)``. The first term averages each
    component's distance to ``y`` and weights the components. The second
    averages every pair of draws from two components, and only pairs of
    distinct draws within one, so ``E‖X − X′‖`` is estimated without bias
    (Gneiting and Raftery, 2007). Strictly proper; with one column it is
    the CRPS.
    """
    count, n = int(points.shape[0]), int(points.shape[1])
    if n < 2:
        raise ShapeError("an unbiased energy score needs at least two draws per component.")
    weight = jnp.moveaxis(probabilities, -1, 0)
    first = jnp.sum(weight * jnp.mean(_distance(points, y), axis=1), axis=0)
    flat = points.reshape((count * n,) + tuple(points.shape[2:]))
    left, right = np.triu_indices(count * n, k=1)
    same = (left // n) == (right // n)
    scale = np.where(same, 2.0 / (n * (n - 1)), 2.0 / (n * n)).astype(np.float32)
    pair = weight[left // n] * weight[right // n] * scale.reshape((-1,) + (1,) * (weight.ndim - 1))
    second = jnp.sum(pair * _distance(flat[left], flat[right]), axis=0)
    return first - 0.5 * second


def variogram_score(points, probabilities, y, order: float = VARIOGRAM_ORDER):
    """Unbiased estimate of the variogram score of ``order``, averaged over pairs of columns, shape ``batch``.

    ``mean over i < j of (γ(y_i − y_j) − E γ(X_i − X_j))²``, with
    ``γ(a) = (a² + ε)^(order / 2)``, ``|a|^order`` with a small ``ε`` so the
    gradient stays finite (Scheuerer and Hamill, 2015). The expectation is
    the component-weighted mean of the draws; the variance that mean's
    error adds to the square is subtracted, so the estimate has no bias.
    It is proper for any ``γ``, but blind to a shift of every column, so
    ``score_weights`` blends it with a strictly proper score.
    """
    d = int(points.shape[-1])
    left, right = np.triu_indices(d, k=1)
    batch = jnp.broadcast_shapes(tuple(points.shape[2:-1]), tuple(jnp.shape(y))[:-1])
    if left.size == 0:
        return jnp.zeros(batch, dtype=points.dtype)

    def gamma(values):
        return jnp.power(jnp.square(values[..., left] - values[..., right]) + _VARIOGRAM_FLOOR, 0.5 * float(order))

    n = int(points.shape[1])
    drawn = gamma(points)
    weight = jnp.moveaxis(probabilities, -1, 0)[..., None]
    expected = jnp.sum(weight * jnp.mean(drawn, axis=1), axis=0)
    noise = jnp.sum(jnp.square(weight) * jnp.var(drawn, axis=1, ddof=1), axis=0) / n
    return jnp.mean(jnp.square(gamma(jnp.asarray(y, dtype=points.dtype)) - expected) - noise, axis=-1)


def make_joint_score_objective(
    n_events: int,
    family: str = "normal",
    weights: dict | None = None,
    *,
    components: int | None = None,
    draws: int = SCORE_DRAWS,
):
    """A blend of proper scores of a joint head: mean over members, weighted over rows.

    ``weights`` maps ``nll``, ``energy``, and ``variogram`` (see
    ``score_weights``) to their weights, on the standardized ``(batch,
    event)`` target, so each column counts the same whatever its units. The
    energy and variogram scores are estimated without bias from ``draws``
    fresh reparameterized draws of every member's distribution on every
    call, so the gradient is unbiased too. Such an objective takes ``key``
    and is marked ``stochastic``, which tells the training loop to pass one.
    """
    weights = dict(score_weights(None) if weights is None else weights)
    sampled = bool(weights.get("energy") or weights.get("variogram"))

    def objective(raw, y, sample_weight, key=None):
        context = raw[0] if family == "flow" else raw
        if context.ndim != 3:
            raise ShapeError(f"joint head output must be (batch, member, width); received {context.shape}.")
        target = _target_events(y, int(n_events))[:, None, :]
        dist = joint_member_distribution(family, raw, int(n_events), components)
        losses = 0.0
        if weights.get("nll"):
            losses = losses - weights["nll"] * dist.log_prob(target)
        if sampled:
            if key is None:
                raise AshlarError("the energy and variogram scores are read from draws and need a key.")
            points, probabilities = score_draws(dist, key, draws)
            if weights.get("energy"):
                losses = losses + weights["energy"] * energy_score(points, probabilities, target)
            if weights.get("variogram"):
                losses = losses + weights["variogram"] * variogram_score(points, probabilities, target)
        return weighted_mean(jnp.mean(losses, axis=1), sample_weight)

    objective.stochastic = sampled
    return objective


# Quasi-random points per row when a joint head trains on coarse cells.
BOX_TRAINING_POINTS = 32


def make_joint_observed_objective(
    n_events: int,
    family: str = "normal",
    *,
    components: int | None = None,
    points: int = BOX_TRAINING_POINTS,
    columns=None,
):
    """Mean member negative log-likelihood of a joint label, weighted over rows.

    ``y`` is the ``(batch, event, 6)`` array of ``ashlar.observed.fields`` on
    the standardized target. Each row scores ``log_box``: the joint density
    of its exact cells times the probability of its coarse cells given them,
    integrated at ``points`` fixed quasi-random points where a row has more
    than one coarse cell (fewer than ``ashlar.distributions.BOX_POINTS``,
    which scoring uses, to keep a training step cheap).
    """
    from ashlar.distributions import box_points

    d = int(n_events)
    grid = box_points(int(points), d)

    def objective(raw, y, sample_weight):
        context = raw[0] if family in ("flow", "chain") else raw
        if context.ndim != 3:
            raise ShapeError(f"joint head output must be (batch, member, width); received {context.shape}.")
        y = jnp.asarray(y, dtype=jnp.float32)
        if y.ndim != 3 or y.shape[1] != d:
            raise ShapeError(f"a joint label's fields are (batch, {d}, 6); received {y.shape}.")
        dist = joint_member_distribution(family, raw, d, components, columns=columns)
        losses = -dist.log_box(y[:, None, :, 0], y[:, None, :, 1], grid)
        return weighted_mean(jnp.mean(losses, axis=1), sample_weight)

    return objective


def make_hurdle_objective(severity: str):
    """Mean member negative log-likelihood of the hurdle head on the raw target, weighted over rows."""
    from ashlar.distributions import hurdle_distribution

    def objective(raw, y, sample_weight):
        if raw.ndim != 3:
            raise ShapeError(f"hurdle head output must be (batch, member, 3); received {raw.shape}.")
        value = jnp.asarray(y, dtype=raw.dtype).reshape(-1)
        losses = -hurdle_distribution(raw, severity).log_prob(value[:, None])
        return weighted_mean(jnp.mean(losses, axis=1), sample_weight)

    return objective


def member_distribution(task: str, raw, severity: str = "lognormal"):
    """Each member's distribution for every row, batch ``(rows, members)``, on the head's own scale.

    The distribution heads and ``hurdle`` read their flat projection; ``normal``
    reads its one event's location and scale.
    """
    from ashlar.distributions import Normal, head_distribution, hurdle_distribution
    from ashlar.predict import split_normal_parameters

    if task == "normal":
        loc, scale = split_normal_parameters(raw)
        if loc.shape[-1] != 1:
            raise ShapeError(f"an observed label is one target column; the normal head has {loc.shape[-1]}.")
        return Normal(loc[..., 0], scale[..., 0])
    if raw.ndim != 3:
        raise ShapeError(f"{task} head output must be (batch, member, width); received {raw.shape}.")
    if task == "hurdle":
        return hurdle_distribution(raw, severity)
    return head_distribution(task, raw)


def make_observed_objective(task: str, *, severity: str = "lognormal", truncated: bool = False, noisy: bool = False):
    """Mean member negative log-likelihood of what was observed, weighted over rows.

    ``y`` is the ``(batch, 6)`` array of ``ashlar.observed.fields`` on the
    head's own scale: standardized for a scaled task, raw for a count, the
    hurdle, and ``normal``. An exact row scores its density, a coarse row the
    probability of its interval; ``truncated`` divides by the probability of
    each row's truncation range and ``noisy`` widens each row by its noise.
    Both are fixed when the objective is built, so labels without them pay
    nothing. Members are averaged, not mixed, as in the other objectives.
    """
    from ashlar.observed import log_likelihood

    def objective(raw, y, sample_weight):
        dist = member_distribution(task, raw, severity)
        y = jnp.asarray(y, dtype=jnp.float32)
        column = [y[:, index][:, None] for index in range(5)]
        losses = -log_likelihood(
            dist,
            column[0],
            column[1],
            column[2] if truncated else None,
            column[3] if truncated else None,
            column[4] if noisy else None,
        )
        return weighted_mean(jnp.mean(losses, axis=1), sample_weight)

    return objective


TOTAL_TASKS = ("gaussian", "normal", "poisson")
# Heads whose sum over rows has no closed form: their totals are scored on a grid (``grid_total_log_likelihood``).
GRID_TOTAL_TASKS = ("mixture", "flow", "histogram", "student_t", "quantile_spline", "negative_binomial", "poisson")
# Points of the grid a total is convolved on, and how many of the sum's standard deviations it spans.
GRID_POINTS = 512
# A count's grid is longer, so its bins stay whole numbers for sums with a standard deviation up to 128.
GRID_COUNT_POINTS = 2048
GRID_SPAN = 16.0


def grid_total_log_likelihood(dist, segment, num_segments: int, max_rows: int, observation, discrete: bool):
    """Log-likelihood of each segment's observed sum of independent rows, by convolving the rows on a grid.

    ``dist`` has batch ``(rows, members)`` and its rows are grouped into
    adjacent segments by ``segment``; ``observation`` is ``(segments, 4)``:
    each sum's ``lower``, ``upper``, ``truncate_lower``, and
    ``truncate_upper`` on the head's own scale. Each row's distribution is
    put on a grid of ``GRID_POINTS`` bins of one width per segment, its mass
    in each bin from its CDF, centred on its own mean, so any head with a
    CDF works. The width spreads the grid over ``GRID_SPAN`` standard
    deviations of the sum, so each row gets about ``GRID_POINTS /
    (GRID_SPAN √rows)`` bins per standard deviation of its own; a count's
    grid has ``GRID_COUNT_POINTS`` bins, each a whole number wide. The rows of a
    segment are convolved by multiplying their discrete Fourier transforms;
    with ``discrete`` (a count), the bins are whole numbers and the mass of
    an exact total is exact up to the grid's span. An exact continuous total
    scores the sum's mass in its bin over the bin's width; an interval scores
    the CDF between its ends, read linearly within a bin; an exact continuous
    total reads the sum's density linearly between bin centres. ``max_rows`` is the
    largest segment, a static size. Returns ``(segments, members)``.
    """
    import jax

    rows = int(segment.shape[0])
    members = int(dist.batch_shape[-1])
    points = GRID_COUNT_POINTS if discrete else GRID_POINTS
    half = points // 2
    frozen = jax.lax.stop_gradient(dist)
    mean = frozen.mean()
    variance = jnp.maximum(frozen.variance(), 1e-12)
    spread = jnp.sqrt(jax.ops.segment_sum(variance, segment, num_segments=num_segments))
    width = GRID_SPAN * spread / points
    if discrete:
        width = jnp.maximum(jnp.ceil(width), 1.0)
    width = jnp.maximum(width, 1e-6)
    row_width = width[segment]
    centre = row_width * jnp.round(mean / row_width)
    edges = centre[..., None] + row_width[..., None] * (jnp.arange(points + 1, dtype=jnp.float32) - half - 0.5)
    if discrete:
        edges = jnp.floor(edges + 0.5) - 0.5 + 1e-3
    cdf = dist.cdf(jnp.moveaxis(edges, -1, 0))
    mass = jnp.moveaxis(jnp.clip(cdf[1:] - cdf[:-1], 0.0, 1.0), 0, -1)
    # Each segment's rows side by side, a point mass at the centre for the slots past its end.
    index = jnp.arange(rows)
    first = jax.ops.segment_min(index, segment, num_segments=num_segments)
    size = jax.ops.segment_sum(jnp.ones_like(index), segment, num_segments=num_segments)
    slot = jnp.clip(first, 0, rows - 1)[:, None] + jnp.arange(int(max_rows))[None, :]
    live = jnp.arange(int(max_rows))[None, :] < size[:, None]
    gathered = mass[jnp.clip(slot, 0, rows - 1)]
    delta = jax.nn.one_hot(half, points, dtype=mass.dtype)
    gathered = jnp.where(live[..., None, None], gathered, delta)
    spectrum = jnp.prod(jnp.fft.rfft(gathered, axis=-1), axis=1)
    circular = jnp.fft.irfft(spectrum, n=points, axis=-1)
    # Bin i of the sum is the offset i - half from the sum of the centres.
    shift = (jnp.arange(points) - half + int(max_rows) * half) % points
    pmf = jnp.clip(circular[..., shift], 0.0, None)
    total_centre = jax.ops.segment_sum(centre, segment, num_segments=num_segments)
    cumulative = jnp.concatenate([jnp.zeros(pmf.shape[:-1] + (1,), pmf.dtype), jnp.cumsum(pmf, axis=-1)], axis=-1)
    cumulative = cumulative / jnp.maximum(cumulative[..., -1:], 1e-30)
    seg_width = width

    def position(value):
        return (value[:, None] - total_centre) / seg_width + half + 0.5

    def cdf_at(value):
        finite = jnp.isfinite(value)
        t = position(jnp.where(finite, value, 0.0))
        low = jnp.clip(jnp.floor(t), 0, points - 1).astype(jnp.int32)
        frac = jnp.clip(t - low, 0.0, 1.0)
        below = jnp.take_along_axis(cumulative, low[..., None], axis=-1)[..., 0]
        above = jnp.take_along_axis(cumulative, (low + 1)[..., None], axis=-1)[..., 0]
        inside = jnp.where(t <= 0, 0.0, jnp.where(t >= points, 1.0, below + frac * (above - below)))
        return jnp.where(finite[:, None], inside, jnp.where(value[:, None] > 0, 1.0, 0.0))

    lower, upper, truncate_lower, truncate_upper = (observation[:, index] for index in range(4))
    exact = lower == upper
    if discrete:
        lower = jnp.where(jnp.isfinite(lower), jnp.round(lower) - 0.5, lower)
        upper = jnp.where(jnp.isfinite(upper), jnp.round(upper) + 0.5, upper)
        truncate_lower = jnp.where(jnp.isfinite(truncate_lower), jnp.round(truncate_lower) - 0.5, truncate_lower)
        truncate_upper = jnp.where(jnp.isfinite(truncate_upper), jnp.round(truncate_upper) + 0.5, truncate_upper)
        coarse = jnp.log(jnp.maximum(cdf_at(upper) - cdf_at(lower), 1e-30))
        result = coarse
    else:
        # The density between bin centres, read linearly: bin i's centre sits at position i + 0.5.
        t = position(jnp.where(exact, lower, 0.0)) - 0.5
        left = jnp.clip(jnp.floor(t), 0, points - 2).astype(jnp.int32)
        frac = jnp.clip(t - left, 0.0, 1.0)
        inside = (t >= 0) & (t <= points - 1)
        near = jnp.take_along_axis(pmf, left[..., None], axis=-1)[..., 0]
        far = jnp.take_along_axis(pmf, (left + 1)[..., None], axis=-1)[..., 0]
        density = (near + frac * (far - near)) / seg_width
        point = jnp.log(jnp.maximum(jnp.where(inside, density, 0.0), 1e-30))
        span = cdf_at(jnp.where(exact, jnp.inf, upper)) - cdf_at(jnp.where(exact, -jnp.inf, lower))
        coarse = jnp.log(jnp.maximum(span, 1e-30))
        result = jnp.where(exact[:, None], point, coarse)
    truncation = cdf_at(truncate_upper) - cdf_at(truncate_lower)
    return result - jnp.log(jnp.maximum(truncation, 1e-30))


def sum_distribution(dist, segment, num_segments: int, live):
    """The distribution of each segment's sum of independent rows, for normal and Poisson rows.

    ``dist`` has batch ``(rows, members)``; ``segment`` maps each row to its
    observation. A sum of normals is normal, with the variances added, and a
    sum of Poisson counts is Poisson. Where ``live`` is false (padding, or an
    observation with no weight) the result is a placeholder that stays finite.
    """
    import jax

    from ashlar.distributions import Normal, Poisson

    def total(values):
        return jax.ops.segment_sum(values, segment, num_segments=num_segments)

    if isinstance(dist, Normal):
        variance = jnp.where(live, total(jnp.square(dist.scale)), 1.0)
        return Normal(total(dist.loc), jnp.sqrt(variance))
    if isinstance(dist, Poisson):
        return Poisson(jnp.where(live, total(dist.mean_count), 1.0))
    raise ShapeError(f"a total over rows needs normal or Poisson rows; received {type(dist).__name__}.")


def make_total_objective(task: str, *, truncated: bool = False, noisy: bool = False, max_rows: int | None = None):
    """Mean member negative log-likelihood when some observations are totals over groups of rows.

    ``y`` is ``(batch, 7)``: the six ``ashlar.observed.fields`` on the head's
    own scale and a flag on the first row of each observation. The rows of an
    observation are adjacent in the batch (``ashlar.train.pack_units``), and
    its weight sits on its first row. Each observation scores the
    distribution of the sum of its rows, so a row of its own scores itself.
    Each member sums its own rows, and members are averaged as elsewhere.

    With ``max_rows``, the largest group, the sum is convolved on a grid
    (``grid_total_log_likelihood``), for a head whose sum has no closed form;
    a row of its own still scores its own likelihood exactly.
    """
    import jax

    from ashlar.observed import log_likelihood

    if max_rows is not None:
        return _grid_total_objective(task, truncated=truncated, max_rows=int(max_rows))

    def objective(raw, y, sample_weight):
        y = jnp.asarray(y, dtype=jnp.float32)
        rows = y.shape[0]
        start = y[:, 6] > 0
        segment = jnp.clip(jnp.cumsum(start.astype(jnp.int32)) - 1, 0, rows - 1)
        weight = jax.ops.segment_sum(
            jnp.where(start, jnp.asarray(sample_weight, dtype=jnp.float32), 0.0), segment, num_segments=rows
        )
        live = (weight > 0)[:, None]
        dist = sum_distribution(member_distribution(task, raw), segment, rows, live)
        observation = jax.ops.segment_sum(jnp.where(start[:, None], y[:, :5], 0.0), segment, num_segments=rows)
        column = [observation[:, index][:, None] for index in range(5)]
        losses = -log_likelihood(
            dist,
            column[0],
            column[1],
            column[2] if truncated else None,
            column[3] if truncated else None,
            column[4] if noisy else None,
        )
        losses = jnp.where(live, losses, 0.0)
        return weighted_mean(jnp.mean(losses, axis=1), weight)

    return objective


def segment_layout(start, sample_weight):
    """Each row's observation, each observation's weight, and its first row, from the flags on first rows."""
    import jax

    rows = start.shape[0]
    segment = jnp.clip(jnp.cumsum(start.astype(jnp.int32)) - 1, 0, rows - 1)
    weight = jax.ops.segment_sum(jnp.where(start, jnp.asarray(sample_weight, dtype=jnp.float32), 0.0), segment, num_segments=rows)
    size = jax.ops.segment_sum(jnp.ones((rows,), jnp.int32), segment, num_segments=rows)
    return segment, weight, size


def grid_totals(dist, y, segment, size, max_rows: int, truncated: bool):
    """``(observations, members)`` log-likelihoods of totals on a grid, and of a row of its own exactly.

    ``y`` holds the six ``ashlar.observed.fields`` per row and a flag on the
    first row of each observation, as ``make_total_objective`` reads them.
    """
    import jax

    from ashlar.observed import log_likelihood

    rows = y.shape[0]
    start = y[:, 6] > 0
    fields = jnp.where(start[:, None], y[:, :4], 0.0)
    observation = jax.ops.segment_sum(fields, segment, num_segments=rows)
    if not truncated:
        observation = observation.at[:, 2].set(-jnp.inf).at[:, 3].set(jnp.inf)
    discrete = dist.support == "nonnegative_integer"
    summed = grid_total_log_likelihood(dist, segment, rows, max_rows, observation, discrete)
    own = log_likelihood(
        dist,
        y[:, 0:1],
        y[:, 1:2],
        y[:, 2:3] if truncated else None,
        y[:, 3:4] if truncated else None,
    )
    own = jax.ops.segment_sum(jnp.where(start[:, None], own, 0.0), segment, num_segments=rows)
    return jnp.where((size == 1)[:, None], own, summed)


def _grid_total_objective(task: str, *, truncated: bool, max_rows: int):
    def objective(raw, y, sample_weight):
        y = jnp.asarray(y, dtype=jnp.float32)
        segment, weight, size = segment_layout(y[:, 6] > 0, sample_weight)
        live = (weight > 0)[:, None]
        values = grid_totals(member_distribution(task, raw), y, segment, size, max_rows, truncated)
        losses = jnp.where(live, -values, 0.0)
        return weighted_mean(jnp.mean(losses, axis=1), weight)

    return objective
