"""Several outcomes of different kinds, chained by the rule of probability.

``p(y_1, ..., y_d | x) = p(y_1 | x) · p(y_2 | y_1, x) · ... · p(y_d | y_<d, x)``

Each column has a head of its own kind: a count (``poisson``,
``negative_binomial``, zero-inflated or not), an exact zero or a positive
amount (``hurdle``), classes (``categorical``), or a continuous value
(``gaussian``, ``mixture``, ``flow``, ``histogram``, ``student_t``). The
parameters of column ``j`` come from the row's hidden state and from the
columns before it, so a claim's cost can depend on how many claims there
were, and a category on both. The joint log-likelihood is the sum of the
columns' conditional log-likelihoods, exact, and a joint draw takes one
column at a time. ``ASRRegressor(task="chain", head_kwargs={"columns":
[...]})`` fits it; see ``ChainHead`` and ``Chain``.

The earlier columns enter each later column's head through a small
conditioner: ``params_j = direct_j(h) + skip_j · e(y_<j) + out_j ·
tanh(pre_j(h) + prev_j · e(y_<j))``, where ``h`` is the trunk's hidden state and ``e`` a fixed
encoding of each earlier value (its standardized value, ``log1p`` of a count
or amount with a flag for zero, or a one-hot of a class). In training the
earlier columns are the observed ones (teacher forcing); in a draw they are
the draws before.
"""

from __future__ import annotations

import math
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module, field

from ashlar.distributions import Distribution
from ashlar.errors import AshlarError, ShapeError
from ashlar.layers import LinearEnsemble

# Kinds of column a chain takes. The first five work on a standardized column.
SCALED_KINDS = ("gaussian", "mixture", "flow", "histogram", "student_t")
COUNT_KINDS = ("poisson", "negative_binomial")
CHAIN_KINDS = (*SCALED_KINDS, *COUNT_KINDS, "hurdle", "categorical")
# Width of each column's conditioner, the default.
CHAIN_HIDDEN = 32
# A standardized earlier value is clipped here before it conditions a later column.
_VALUE_CLIP = 6.0
MOMENT_DRAWS = 1024
_MOMENT_CHUNK = 64


class ChainColumn(NamedTuple):
    """One column of a chain: its kind and the options that kind reads.

    ``zero_inflated`` is a count's structural zero, ``severity`` a hurdle's
    positive part (``lognormal`` or ``gamma``), ``bins`` a histogram's bin
    count, ``classes`` a categorical column's class count (values are the
    codes ``0 .. classes - 1``), and ``scale`` the spread of ``log1p`` of a
    count or amount, which divides it when it conditions a later column.
    """

    kind: str
    zero_inflated: bool = False
    severity: str = "lognormal"
    bins: int | None = None
    classes: int | None = None
    scale: float = 1.0

    @property
    def scaled(self) -> bool:
        return self.kind in SCALED_KINDS

    @property
    def width(self) -> int:
        """Parameters of this column's head."""
        from ashlar.distributional import HISTOGRAM_BINS, output_width

        if self.kind == "hurdle":
            return 3
        if self.kind == "categorical":
            return int(self.classes)
        bins = HISTOGRAM_BINS if self.bins is None else int(self.bins)
        return output_width(self.kind, 1, bins, bool(self.zero_inflated))

    @property
    def features(self) -> int:
        """Width of this column's encoding when it conditions a later column."""
        if self.kind == "categorical":
            return int(self.classes)
        return 1 if self.scaled else 2

    def to_json(self) -> dict:
        payload = {"kind": self.kind}
        if self.kind in COUNT_KINDS and self.zero_inflated:
            payload["zero_inflated"] = True
        if self.kind == "hurdle":
            payload["severity"] = self.severity
        if self.kind == "histogram" and self.bins is not None:
            payload["bins"] = int(self.bins)
        if self.kind == "categorical" and self.classes is not None:
            payload["classes"] = int(self.classes)
        if self.kind in (*COUNT_KINDS, "hurdle"):
            payload["scale"] = float(self.scale)
        return payload


def chain_columns(spec) -> tuple[ChainColumn, ...]:
    """Read ``head_kwargs["columns"]``: a list of kind names or ``{"kind": ..., options}`` dicts."""
    from ashlar.distributions import HURDLE_SEVERITIES

    if spec is None:
        raise AshlarError(
            "a chain needs its columns, head_kwargs={'columns': [...]}: one kind per target column, "
            f"from {', '.join(CHAIN_KINDS)}, such as ['negative_binomial', 'hurdle']."
        )
    if isinstance(spec, (str, dict)) or not hasattr(spec, "__iter__"):
        raise AshlarError("head_kwargs['columns'] is a list with one kind per target column.")
    columns = []
    for item in spec:
        if isinstance(item, ChainColumn):
            columns.append(item)
            continue
        entry = {"kind": item} if isinstance(item, str) else dict(item)
        kind = entry.pop("kind", None)
        if kind not in CHAIN_KINDS:
            raise AshlarError(f"a chain column's kind is one of {', '.join(CHAIN_KINDS)}; received {kind!r}.")
        allowed = {
            "poisson": {"zero_inflated", "scale"},
            "negative_binomial": {"zero_inflated", "scale"},
            "hurdle": {"severity", "scale"},
            "histogram": {"bins"},
            "categorical": {"classes"},
        }.get(kind, set())
        unknown = sorted(set(entry) - allowed)
        if unknown:
            raise AshlarError(f"a {kind} column of a chain does not take {', '.join(unknown)}.")
        severity = str(entry.get("severity", "lognormal"))
        if severity not in HURDLE_SEVERITIES:
            raise AshlarError(f"hurdle severity must be one of {HURDLE_SEVERITIES}; received {severity!r}.")
        classes = entry.get("classes")
        if classes is not None and int(classes) < 2:
            raise AshlarError(f"a categorical column needs at least 2 classes; received {classes}.")
        bins = entry.get("bins")
        scale = float(entry.get("scale", 1.0))
        if not math.isfinite(scale) or scale <= 0:
            raise AshlarError(f"a chain column's scale must be positive; received {scale}.")
        columns.append(
            ChainColumn(
                kind=kind,
                zero_inflated=bool(entry.get("zero_inflated", False)),
                severity=severity,
                bins=None if bins is None else int(bins),
                classes=None if classes is None else int(classes),
                scale=scale,
            )
        )
    if not columns:
        raise AshlarError("a chain needs at least one column.")
    return tuple(columns)


def resolve_columns(columns: tuple[ChainColumn, ...], values) -> tuple[ChainColumn, ...]:
    """Fill in what the training values decide: a categorical column's classes and a count's scale.

    ``values`` is ``(rows, columns)`` on the raw scale; ``nan`` marks a cell
    that is not exact. Raises when a column's values are not of its kind.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ShapeError(f"this chain has {len(columns)} columns; y has shape {values.shape}.")
    resolved = []
    for index, column in enumerate(columns):
        cells = values[:, index]
        seen = cells[np.isfinite(cells)]
        where = f"column {index} of the chain ({column.kind})"
        if column.kind in COUNT_KINDS and (np.any(seen < 0) or np.any(seen != np.round(seen))):
            raise AshlarError(f"{where} models counts: its values must be non-negative whole numbers.")
        if column.kind == "hurdle" and np.any(seen < 0):
            raise AshlarError(f"{where} models an exact zero or a positive amount: its values must be non-negative.")
        if column.kind == "categorical":
            if np.any(seen < 0) or np.any(seen != np.round(seen)):
                raise AshlarError(f"{where} reads class codes 0, 1, 2, ...: its values must be non-negative whole numbers.")
            classes = column.classes
            if classes is None:
                classes = max(2, int(seen.max()) + 1 if seen.size else 2)
            elif seen.size and int(seen.max()) >= classes:
                raise AshlarError(f"{where} has {classes} classes; a value of {int(seen.max())} is outside them.")
            column = column._replace(classes=int(classes))
        if column.kind in (*COUNT_KINDS, "hurdle"):
            spread = float(np.std(np.log1p(seen))) if seen.size > 1 else 1.0
            column = column._replace(scale=spread if spread > 1e-3 else 1.0)
        resolved.append(column)
    return tuple(resolved)


def column_start(column: ChainColumn, values) -> jax.Array:
    """The bias a column's head starts from, set from its training values (``nan`` for a cell that is not exact)."""
    from ashlar.distributional import HISTOGRAM_BINS, initial_bias
    from ashlar.estimator import _hurdle_start

    values = np.asarray(values, dtype=np.float64)
    seen = values[np.isfinite(values)]
    if column.kind == "categorical":
        counts = np.bincount(seen.astype(np.int64), minlength=int(column.classes)).astype(np.float64) + 0.5
        return jnp.asarray(np.log(counts / counts.sum()), dtype=jnp.float32)
    if column.kind == "hurdle":
        sample = seen if seen.size else np.zeros(1)
        return jnp.asarray(_hurdle_start(np.maximum(sample, 0.0), column.severity), dtype=jnp.float32)
    log_mean = 0.0
    if column.kind in COUNT_KINDS:
        log_mean = float(np.log(max(float(np.mean(seen)) if seen.size else 1.0, 1e-3)))
    bins = HISTOGRAM_BINS if column.bins is None else int(column.bins)
    return initial_bias(column.kind, 1, bins, zero_inflated=bool(column.zero_inflated), log_mean=log_mean)


def encode_values(columns: tuple[ChainColumn, ...], values):
    """The fixed encoding of every column's value, ``... + (Σ features,)``, read by the later columns."""
    pieces = []
    for index, column in enumerate(columns):
        value = values[..., index]
        if column.kind == "categorical":
            pieces.append(jax.nn.one_hot(jnp.clip(value, 0, column.classes - 1).astype(jnp.int32), column.classes))
        elif column.scaled:
            pieces.append(jnp.clip(value, -_VALUE_CLIP, _VALUE_CLIP)[..., None])
        else:
            positive = jnp.maximum(value, 0.0)
            pieces.append(jnp.stack([jnp.log1p(positive) / column.scale, (positive == 0).astype(value.dtype)], axis=-1))
    return jnp.concatenate(pieces, axis=-1)


def prefix_mask(columns: tuple[ChainColumn, ...]) -> np.ndarray:
    """``(columns, features)``: column ``j`` reads the encodings of the columns before it only."""
    widths = [column.features for column in columns]
    owner = np.repeat(np.arange(len(columns)), widths)
    return (owner[None, :] < np.arange(len(columns))[:, None]).astype(np.float32)


def column_distribution(column: ChainColumn, params):
    """The distribution of one column from its head's parameters ``... + (width,)``."""
    from ashlar.distributions import Categorical, head_distribution, hurdle_distribution

    params = params[..., : column.width]
    if column.kind == "categorical":
        return Categorical(logits=params)
    if column.kind == "hurdle":
        return hurdle_distribution(params, column.severity)
    return head_distribution(column.kind, params)


class ChainHead(Module):
    """The ``chain`` task's head. Output ``(context, prev, out, skip)``, read by ``Chain``.

    ``context`` is ``(batch, member, columns * (width + hidden))``: for each
    column, its ``direct`` parameters (``width`` of them, padded to the widest
    column) and the ``pre`` activations of its conditioner, both linear in the
    trunk's hidden state. ``prev`` ``(member, columns, hidden, features)`` and
    ``out`` ``(member, columns, width, hidden)`` are the conditioner's weights,
    and ``skip`` ``(member, columns, width, features)`` a linear path from the
    earlier columns' encodings straight to the parameters, which starts at
    zero; all three are shared by every row. ``columns`` is the list of column kinds
    (``chain_columns``); ``hidden`` the conditioner's width, 32 by default;
    ``start`` each column's starting bias, which the estimator sets from the
    training columns.
    """

    direct: LinearEnsemble
    pre: LinearEnsemble
    prev: jax.Array
    out: jax.Array
    skip: jax.Array
    k: int = field(static=True)
    columns: tuple = field(static=True)
    hidden: int = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, columns=None, hidden=CHAIN_HIDDEN, start=None, **kwargs):
        from ashlar.distributional import initial_weight_scale

        del n_classes
        if kwargs:
            raise ShapeError(f"chain head received unknown arguments {sorted(kwargs)}.")
        columns = chain_columns(columns)
        if int(n_events) != len(columns):
            raise ShapeError(f"the chain names {len(columns)} columns; y has {n_events}.")
        if any(column.kind == "categorical" and column.classes is None for column in columns):
            raise ShapeError("a categorical column's class count is set from the training values before the head is built.")
        d, hidden = len(columns), int(hidden)
        if hidden < 1:
            raise ShapeError(f"hidden must be at least 1; received {hidden}.")
        width = max(column.width for column in columns)
        features = sum(column.features for column in columns)
        keys = jax.random.split(key, 4 + d)
        weights, biases = [], []
        for index, column in enumerate(columns):
            kind = "negative_binomial" if column.kind in ("hurdle", "categorical") else column.kind
            scale = 0.01 if column.kind == "hurdle" else initial_weight_scale(kind, int(d_in))
            weight = scale * jax.random.normal(keys[4 + index], (k, d_in, column.width), dtype=jnp.float32)
            bias = column_start(column, np.zeros(0)) if start is None else jnp.asarray(start[index], dtype=jnp.float32)
            pad = width - column.width
            weights.append(jnp.pad(weight, ((0, 0), (0, 0), (0, pad))))
            biases.append(jnp.pad(jnp.broadcast_to(bias, (k, column.width)), ((0, 0), (0, pad))))
        self.direct = LinearEnsemble(weight=jnp.concatenate(weights, axis=-1), bias=jnp.concatenate(biases, axis=-1))
        self.pre = LinearEnsemble(
            weight=jax.random.normal(keys[0], (k, d_in, d * hidden), dtype=jnp.float32) / math.sqrt(d_in),
            bias=jnp.zeros((k, d * hidden), dtype=jnp.float32),
        )
        self.prev = jax.random.normal(keys[1], (k, d, hidden, features), dtype=jnp.float32) / math.sqrt(max(features, 1))
        self.out = 0.01 * jax.random.normal(keys[2], (k, d, width, hidden), dtype=jnp.float32)
        self.skip = jnp.zeros((k, d, width, features), dtype=jnp.float32)
        self.k = int(k)
        self.columns = columns
        self.hidden = hidden

    def __call__(self, hidden, *, key):
        del key
        if hidden.ndim != 3 or hidden.shape[1] != self.k:
            raise ShapeError(f"head expected (batch, member, hidden) with member={self.k}; received {hidden.shape}.")
        return jnp.concatenate([self.direct(hidden), self.pre(hidden)], axis=-1), self.prev, self.out, self.skip


class Chain(Distribution):
    """The joint distribution of a chain of columns of different kinds, one per member and row.

    ``log_prob`` is ``Σ_j log p_j(y_j | y_<j, x)``, each term exact: a
    density for a continuous column, a mass for a count or a class, and for
    a hurdle the zero's probability or the amount's density. ``sample``
    draws the columns in order, each from its conditional given the draws
    before. ``log_box`` scores a label whose cells are exact except for
    classes known only to lie in a range, which it sums over, and the last
    column, which may be censored or an interval. Mean, variance, and
    covariance are Monte Carlo estimates from ``MOMENT_DRAWS`` fixed draws,
    and ``marginals()`` an ``Empirical`` of them.

    ``context`` is ``batch_shape + (columns * (width + hidden),)``, its last
    batch axis the members; ``prev`` and ``out`` are each member's
    conditioner weights and linear path, shared by every row (see ``ChainHead``).
    """

    context: jax.Array
    prev: jax.Array
    out: jax.Array
    skip: jax.Array
    columns: tuple = field(static=True)
    reparameterized: ClassVar[bool] = False
    shared_parameters: ClassVar[bool] = True

    def __init__(self, context, prev, out, skip, columns):
        self.context = jnp.asarray(context, dtype=jnp.float32)
        self.prev = jnp.asarray(prev, dtype=jnp.float32)
        self.out = jnp.asarray(out, dtype=jnp.float32)
        self.skip = jnp.asarray(skip, dtype=jnp.float32)
        self.columns = tuple(columns)
        members, d, width, hidden = self.out.shape
        if d != len(self.columns) or self.context.shape[-1] != d * (width + hidden):
            raise ShapeError(
                f"a chain of {len(self.columns)} columns needs context (..., {members}, {d * (width + hidden)}); "
                f"received {self.context.shape}."
            )

    @property
    def batch_shape(self):
        return self.context.shape[:-1]

    @property
    def event_shape(self):
        return (len(self.columns),)

    @property
    def support(self):
        return "mixed"

    def _index(self, index):
        return Chain(self.context[index], self.prev, self.out, self.skip, self.columns)

    def _params(self, column: int, context, values):
        """Head parameters of ``column`` given the earlier ``values`` (later ones are masked out)."""
        members, d, width, hidden = self.out.shape
        direct = context[..., column * width : (column + 1) * width]
        offset = d * width
        pre = context[..., offset + column * hidden : offset + (column + 1) * hidden]
        if column == 0:
            return direct + jnp.einsum("...kh,kph->...kp", jnp.tanh(pre), self.out[:, column])
        mask = jnp.asarray(prefix_mask(self.columns)[column])
        features = encode_values(self.columns, values) * mask
        units = jnp.tanh(pre + jnp.einsum("...kf,khf->...kh", features, self.prev[:, column]))
        linear = jnp.einsum("...kf,kpf->...kp", features, self.skip[:, column])
        return direct + linear + jnp.einsum("...kh,kph->...kp", units, self.out[:, column])

    def conditional(self, column: int, values):
        """The distribution of ``column`` given ``values`` of the columns before it, ``... + batch_shape``."""
        context, values = self._aligned(values)
        return column_distribution(self.columns[column], self._params(column, context, values))

    def _aligned(self, value):
        value = jnp.asarray(value, dtype=jnp.float32)
        batch = jnp.broadcast_shapes(value.shape[:-1], self.batch_shape)
        context = jnp.broadcast_to(self.context, batch + self.context.shape[-1:])
        return context, jnp.broadcast_to(value, batch + (len(self.columns),))

    def log_prob(self, value):
        context, value = self._aligned(value)
        total = 0.0
        for index, column in enumerate(self.columns):
            dist = column_distribution(column, self._params(index, context, value))
            total = total + dist.log_prob(value[..., index])
        return total

    def log_box(self, lower, upper, points=None):
        """Log-likelihood of a label whose cells are exact, a range of classes, or, in the last column, any interval.

        A class column whose cell is a range sums over the classes in it, so
        an unknown cause of a censored time, ``interval(-inf, inf)``, scores
        ``Σ_c p(c) P(T > t | c)``. Every other cell before the last must be
        exact; the estimator refuses a label that is not.
        """
        from ashlar.distributions import _bounds

        del points
        lower, upper = _bounds(lower, upper)
        d = len(self.columns)
        batch = jnp.broadcast_shapes(lower.shape[:-1], upper.shape[:-1], self.batch_shape)
        lower, upper = jnp.broadcast_to(lower, batch + (d,)), jnp.broadcast_to(upper, batch + (d,))
        context = jnp.broadcast_to(self.context, batch + self.context.shape[-1:])
        exact = lower == upper
        # Branches over the classes of earlier class columns: values (branches, ...batch, d), weight (branches, ...batch).
        values = jnp.where(exact, lower, 0.0)[None]
        weight = jnp.zeros((1,) + batch, dtype=jnp.float32)
        for index, column in enumerate(self.columns):
            low, high = lower[..., index], upper[..., index]
            dist = column_distribution(column, self._params(index, context[None], values))
            last = index == d - 1
            if column.kind == "categorical":
                codes = jnp.arange(column.classes, dtype=jnp.float32)
                logits = dist.logits  # (branches, ...batch, classes), normalized
                inside = (codes >= low[..., None]) & (codes <= high[..., None])
                logits = jnp.where(inside, logits, -jnp.inf)
                if last:
                    weight = weight + jax.nn.logsumexp(logits, axis=-1)
                    continue
                # One branch per class: (classes, branches, ...batch).
                weight = (weight[..., None] + logits)
                weight = jnp.moveaxis(weight, -1, 0).reshape((-1,) + batch)
                branches = values.shape[0]
                values = jnp.broadcast_to(values[None], (column.classes,) + values.shape)
                values = values.at[..., index].set(codes.reshape((-1,) + (1,) * (values.ndim - 2)))
                values = values.reshape((column.classes * branches,) + batch + (d,))
                continue
            if last:
                point = dist.log_prob(jnp.where(exact[..., index], low, 0.0))
                coarse = dist.log_interval(
                    jnp.where(exact[..., index], -jnp.inf, low), jnp.where(exact[..., index], jnp.inf, high)
                )
                weight = weight + jnp.where(exact[..., index], point, coarse)
                continue
            weight = weight + dist.log_prob(values[..., index])
        return jax.nn.logsumexp(weight, axis=0)

    def sample(self, key, sample_shape=()):
        shape = tuple(sample_shape) + self.batch_shape
        context = jnp.broadcast_to(self.context, shape + self.context.shape[-1:])
        values = jnp.zeros(shape + (len(self.columns),), dtype=jnp.float32)
        for index, column in enumerate(self.columns):
            key, draw_key = jax.random.split(key)
            dist = column_distribution(column, self._params(index, context, values))
            values = values.at[..., index].set(dist.sample(draw_key).astype(jnp.float32))
        return values

    def _moments(self):
        keys = jax.random.split(jax.random.PRNGKey(0), MOMENT_DRAWS // _MOMENT_CHUNK)
        d = len(self.columns)
        zero = (jnp.zeros(self.batch_shape + (d,)), jnp.zeros(self.batch_shape + (d, d)))

        def step(totals, key):
            draws = self.sample(key, (_MOMENT_CHUNK,))
            first = totals[0] + jnp.sum(draws, axis=0)
            second = totals[1] + jnp.einsum("n...i,n...j->...ij", draws, draws)
            return (first, second), None

        (first, second), _ = jax.lax.scan(step, zero, keys)
        mean = first / MOMENT_DRAWS
        return mean, second / MOMENT_DRAWS - mean[..., :, None] * mean[..., None, :]

    def mean(self):
        return self._moments()[0]

    def covariance(self):
        return self._moments()[1]

    def variance(self):
        return jnp.maximum(jnp.diagonal(self.covariance(), axis1=-2, axis2=-1), 0.0)

    def marginals(self):
        from ashlar.distributions import Empirical

        draws = self.sample(jax.random.PRNGKey(0), (MOMENT_DRAWS,))
        return Empirical(jnp.moveaxis(draws, 0, -1))


def chain_distribution(raw, columns) -> Chain:
    """The ``chain`` head's distribution from its output ``(context, prev, out, skip)``."""
    context, prev, out, skip = raw
    return Chain(context, prev, out, skip, columns)
