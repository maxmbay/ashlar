"""Member-wise objectives with one documented weight reduction.

TabM trains the members independently: the optimized value is the mean of the
member losses, not the loss of the averaged prediction. Sample weights are
applied once, after that member mean, with reduction ``batch_weight_sum``.
"""

from __future__ import annotations

import math

import jax.numpy as jnp

from ashlar.errors import ShapeError
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


def make_joint_objective(n_events: int, family: str = "normal"):
    """Mean member negative log-density of a joint head, weighted over rows.

    ``family`` is ``normal`` for ``multivariate_gaussian`` or ``flow`` for
    ``multivariate_flow``.

    ``y`` is the standardized ``(batch, event)`` target. The density is the
    full joint one, so the loss rewards getting the dependence between
    columns right, not only each column.
    """
    from ashlar.distributions import joint_flow_distribution, joint_normal_distribution

    build = joint_flow_distribution if family == "flow" else joint_normal_distribution

    def objective(raw, y, sample_weight):
        context = raw[0] if family == "flow" else raw
        if context.ndim != 3:
            raise ShapeError(f"joint head output must be (batch, member, width); received {context.shape}.")
        target = _target_events(y, int(n_events))
        dist = build(raw, n_events)
        losses = -dist.log_prob(target[:, None, :])
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


def make_total_objective(task: str, *, truncated: bool = False, noisy: bool = False):
    """Mean member negative log-likelihood when some observations are totals over groups of rows.

    ``y`` is ``(batch, 7)``: the six ``ashlar.observed.fields`` on the head's
    own scale and a flag on the first row of each observation. The rows of an
    observation are adjacent in the batch (``ashlar.train.pack_units``), and
    its weight sits on its first row. Each observation scores the
    distribution of the sum of its rows, so a row of its own scores itself.
    Each member sums its own rows, and members are averaged as elsewhere.
    """
    import jax

    from ashlar.observed import log_likelihood

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
