"""Entity effects: partial pooling of a grouping column, as an offset and as an embedding.

``ASRRegressor(task="gaussian", random_effects={"store": "intercept"})``
takes ``store`` out of the network's inputs and gives each store an offset
``b_store ~ N(0, tau^2)`` on the head's location, with ``tau`` learned. Each
store's offset has a variational posterior ``q(b_store) = N(m, s^2)``:

- training maximizes the evidence lower bound, the expected log likelihood
  of each row under ``q`` of its store, minus the KL divergence of every
  ``q`` from ``N(0, tau^2)`` once per pass over the training rows, so a store
  with three rows is pulled towards the average store and a store with three
  thousand rows is not;
- a row of a store seen in training predicts with ``b`` integrated over ``q``,
  and a row of a store not seen (or a missing store) integrates over
  ``N(0, tau^2)``, the population answer. Passing the column as missing gives
  the population distribution for any row.

``{"store": ["intercept", "price"]}`` adds a random slope on the scaled
numeric feature ``price``, ``b_store_price * z_price``, with its own
``tau``; several grouping columns add up (crossed or nested, ``store`` within
``region`` is two columns). Only the sum of the offsets moves the location,
and with independent normal posteriors that sum is normal, so the integral
is one-dimensional whatever the number of terms.

``"embedding"`` among the terms also feeds the network a pooled embedding of
the group, so a store can interact with the other features (a store whose
sales react to promotions more than most): ``e_store ~ N(0, diag(tau_e^2))``
with ``tau_e`` learned per dimension and a variational posterior per store,
drawn afresh on every training step and read at its mean at prediction. A
store with few rows sits near 0, the average store, where the network has
learned what an average store does; an unseen store reads 0.

The offset's integral is exact where it has a closed form: on the gaussian
and mixture heads at prediction a normal offset only widens each normal.
Every other head, and training, reads it on equal-weight normal nodes
folded into the member axis, which every objective already averages over
and every prediction already mixes over. The nodes are moment-matched, so
the expected log likelihood of a normal is exact.

``{"store": {"terms": ["intercept", "embedding"], "within": "region"}}``
nests stores in a broader class that is a grouping column of its own: the
store's offset adds to its region's, and its embedding to its region's
embedding, so a store with few rows is pulled towards its region rather than
towards the average store, and a new store of a known region starts there.

``random_effects_`` reports each ``tau`` and each group's posterior in the
units of the location (``y`` for the normal-family heads, the log mean for
counts), and ``with_context(X, y)`` updates the offsets' posteriors from new
rows (a Laplace step, exact for a single normal), giving a store that was
not in training its own offset from a few of its rows without refitting the
network.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module, field
from scipy import special

from ashlar.distributional import SPLINE_PIECES, flow_parameter_count
from ashlar.errors import AshlarError, ShapeError

# Tasks with an output that moves the whole distribution: a location, a log mean, a median, a shift.
RANDOM_EFFECT_TASKS = (
    "regression", "gaussian", "mixture", "flow", "student_t", "quantile", "quantile_spline",
    "poisson", "negative_binomial", "hurdle",
)
INTERCEPT = "intercept"
EMBEDDING = "embedding"
# Equal-weight normal nodes per row: training reads the expected log likelihood on them, and prediction
# the mixture over them for a head with no closed form.
TRAIN_NODES = 8
PREDICT_NODES = 16
# Half-normal prior on each tau, on the location's scale (a standardized target, or a log mean): weak.
TAU_PRIOR = 1.0
TAU_START = 0.5
_SCALE_FLOOR = 1e-4
_MIXTURE_WIDTH = 3
# The stream of a training step's key that draws the pooled embeddings, apart from dropout.
_EMBED_STREAM = 0xE77EC7


def _entries(spec) -> dict:
    if spec is None:
        return {}
    if not isinstance(spec, dict) or not spec:
        raise AshlarError(
            'random_effects is a dict of grouping columns to their terms, such as {"store": "intercept"}, '
            '{"store": ["intercept", "price", "embedding"]}, or, nested in a broader class, '
            '{"store": {"terms": ["intercept", "embedding"], "within": "region"}}.'
        )
    return spec


def check_random_effects(spec) -> dict[str, tuple[str, ...]]:
    """``{column: terms}``, checked: ``"intercept"`` first when it is there, then slopes, then ``"embedding"``.

    A value is ``"intercept"``, ``"embedding"``, a numeric column name for a
    random slope, a list of them, or ``{"terms": ..., "within": column}`` for a
    group nested in a broader one (see ``nesting``).
    """
    if spec is None:
        return {}
    checked = {}
    spec = _entries(spec)
    for column, value in spec.items():
        terms = value.get("terms") if isinstance(value, dict) else value
        if isinstance(value, dict) and set(value).difference({"terms", "within"}):
            raise AshlarError(f"random_effects[{column!r}] takes the keys 'terms' and 'within'.")
        items = [terms] if isinstance(terms, str) else terms
        if not isinstance(items, (list, tuple)) or not items or not all(isinstance(term, str) for term in items):
            raise AshlarError(
                f"random_effects[{column!r}] must be 'intercept', 'embedding', a numeric column, or a list of them."
            )
        if len(set(items)) != len(items):
            raise AshlarError(f"random_effects[{column!r}] names a term twice.")
        if str(column) in items:
            raise AshlarError(f"random_effects[{column!r}] cannot take a slope on its own grouping column.")
        ordered = [INTERCEPT] if INTERCEPT in items else []
        ordered += [term for term in items if term not in (INTERCEPT, EMBEDDING)]
        ordered += [EMBEDDING] if EMBEDDING in items else []
        checked[str(column)] = tuple(ordered)
    nesting(spec, checked)
    return checked


def nesting(spec, checked: dict | None = None) -> dict[str, str]:
    """``{column: broader column}`` for the groups declared ``within`` another grouping column.

    A nested group's offset already adds to its class's, so it is pulled
    towards the class's offset rather than 0. Its pooled embedding is added to
    its class's too, and the network reads the sum: a store with few rows reads
    nearly its class's embedding, and an unseen store of a known class reads
    exactly that. A nested group with an embedding needs a class with one.
    """
    if spec is None:
        return {}
    spec = _entries(spec)
    checked = check_random_effects(spec) if checked is None else checked
    parents = {}
    for column, value in spec.items():
        if not isinstance(value, dict) or value.get("within") is None:
            continue
        parent = value["within"]
        if not isinstance(parent, str) or parent not in checked or parent == str(column):
            raise AshlarError(
                f"random_effects[{column!r}]['within'] must name another grouping column of random_effects; "
                f"received {parent!r}."
            )
        if EMBEDDING in checked[str(column)] and EMBEDDING not in checked[parent]:
            raise AshlarError(
                f"{column!r} adds its embedding to {parent!r}'s, so {parent!r} needs 'embedding' among its terms too."
            )
        parents[str(column)] = parent
    for column in parents:
        seen, current = {column}, column
        while current in parents:
            current = parents[current]
            if current in seen:
                raise AshlarError(f"random_effects nests {column!r} in a cycle.")
            seen.add(current)
    return parents


def embedding_leaves(checked: dict, parents: dict) -> list[str]:
    """The embedded grouping columns the network reads: those no other embedded column is nested in."""
    embedded = [name for name, terms in checked.items() if EMBEDDING in terms]
    return [name for name in embedded if not any(parents.get(other) == name for other in embedded)]


def offset_terms(terms) -> tuple[str, ...]:
    """The terms that add to the location: the intercept and the slopes."""
    return tuple(term for term in terms if term != EMBEDDING)


def location_outputs(task: str, width: int) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    """The raw outputs an offset moves, and the raw scales it widens in closed form (``None``: no closed form).

    ``regression`` has no scale: its prediction is the mean, which an offset only moves.
    """
    width = int(width)
    if task == "regression":
        return tuple(range(width)), ()
    if task == "gaussian" and width == 2:
        return (0,), (1,)
    if task == "mixture" and width % _MIXTURE_WIDTH == 0:
        starts = tuple(range(0, width, _MIXTURE_WIDTH))
        return starts, tuple(start + 1 for start in starts)
    if task == "flow" and width == flow_parameter_count() + 2:
        return (flow_parameter_count(),), None
    if task in ("student_t", "quantile", "poisson", "negative_binomial") and width >= 1:
        return (0,), None
    if task == "quantile_spline" and width == 1 + 2 * SPLINE_PIECES:
        return (0,), None
    if task == "hurdle" and width == 3:
        return (1,), None
    raise AshlarError(
        f"random effects move a head's location, so they need one of {', '.join(RANDOM_EFFECT_TASKS)} "
        f"with one target column; this model is {task} with {width} outputs."
    )


def unit_nodes(count: int) -> np.ndarray:
    """``count`` equal-weight nodes of a standard normal: its quantiles at the cell midpoints, scaled to variance 1."""
    nodes = special.ndtri((np.arange(int(count)) + 0.5) / int(count))
    return (nodes / np.sqrt(np.mean(nodes**2))).astype(np.float32)


def positive(raw):
    """``softplus(raw) + 1e-4``: a standard deviation from its raw leaf."""
    return jax.nn.softplus(raw) + jnp.asarray(_SCALE_FLOOR, dtype=raw.dtype)


def raw_of(value):
    """Inverse of ``positive``, stable for large values."""
    shifted = jnp.maximum(jnp.asarray(value) - _SCALE_FLOOR, 1e-12)
    return shifted + jnp.log(-jnp.expm1(-shifted))


def _kl(mean, scale, tau):
    """KL divergence of ``N(mean, scale^2)`` from ``N(0, tau^2)``, elementwise."""
    return jnp.log(tau) - jnp.log(scale) + (jnp.square(scale) + jnp.square(mean)) / (2.0 * jnp.square(tau)) - 0.5


class RandomEffects(Module):
    """Variational group offsets on the head's location, and pooled group embeddings. See the module docstring.

    Per grouping column ``c`` with ``V`` categories and ``T`` offset terms:
    ``means[c]`` and ``raw_scales[c]`` are ``(V + 1, T)``, row 0 the unknown
    category, which reads the prior instead; ``raw_taus[c]`` is ``(T,)``.
    The pooled embeddings of the columns in ``embedded`` (positions among the
    grouping columns) have ``embed_means`` and ``embed_raw_scales`` of shape
    ``(V + 1, d)`` and ``embed_raw_taus`` of shape ``(d,)``. ``index_columns``
    are the columns of ``category_index`` that hold the groups, and
    ``slope_columns[c]`` the numeric column of each offset term, ``-1`` for
    the intercept. ``locations`` are the raw outputs an offset moves and
    ``scales`` the raw scales it widens in closed form at prediction, or
    ``None`` for nodes. ``kl_weight`` is one over the training rows' weight,
    so the penalty is the KL divergence per row.
    """

    means: tuple[jax.Array, ...]
    raw_scales: tuple[jax.Array, ...]
    raw_taus: tuple[jax.Array, ...]
    embed_means: tuple[jax.Array, ...]
    embed_raw_scales: tuple[jax.Array, ...]
    embed_raw_taus: tuple[jax.Array, ...]
    index_columns: tuple[int, ...] = field(static=True)
    slope_columns: tuple[tuple[int, ...], ...] = field(static=True)
    embedded: tuple[int, ...] = field(static=True)
    # Per grouping column, the position of the broader column it is nested in, or -1.
    parents: tuple[int, ...] = field(static=True)
    locations: tuple[int, ...] = field(static=True)
    scales: tuple[int, ...] | None = field(static=True)
    kl_weight: float = field(static=True)

    def __init__(
        self,
        vocab_sizes,
        index_columns,
        slope_columns,
        locations,
        scales,
        kl_weight: float,
        *,
        embedded=(),
        embedding_dim: int = 8,
        parents=None,
        key=None,
    ):
        if not (len(vocab_sizes) == len(index_columns) == len(slope_columns)) or not vocab_sizes:
            raise ShapeError("random effects need one vocabulary size, index column, and term list per grouping column.")
        start = float(np.log(np.expm1(TAU_START - _SCALE_FLOOR)))
        self.means = tuple(
            jnp.zeros((int(size) + 1, len(terms)), dtype=jnp.float32) for size, terms in zip(vocab_sizes, slope_columns)
        )
        self.raw_scales = tuple(
            jnp.full((int(size) + 1, len(terms)), start, dtype=jnp.float32) for size, terms in zip(vocab_sizes, slope_columns)
        )
        self.raw_taus = tuple(jnp.full((len(terms),), start, dtype=jnp.float32) for terms in slope_columns)
        self.embedded = tuple(int(position) for position in embedded)
        self.parents = tuple(-1 for _ in vocab_sizes) if parents is None else tuple(int(parent) for parent in parents)
        keys = jax.random.split(jax.random.PRNGKey(0) if key is None else key, max(1, len(self.embedded)))
        # Means start small and apart, so the network can tell groups apart from the first step.
        self.embed_means = tuple(
            0.02 * jax.random.normal(piece, (int(vocab_sizes[position]) + 1, int(embedding_dim)), dtype=jnp.float32)
            for position, piece in zip(self.embedded, keys)
        )
        self.embed_raw_scales = tuple(
            jnp.full((int(vocab_sizes[position]) + 1, int(embedding_dim)), start, dtype=jnp.float32)
            for position in self.embedded
        )
        self.embed_raw_taus = tuple(jnp.full((int(embedding_dim),), start, dtype=jnp.float32) for _ in self.embedded)
        self.index_columns = tuple(int(column) for column in index_columns)
        self.slope_columns = tuple(tuple(int(column) for column in terms) for terms in slope_columns)
        self.locations = tuple(int(position) for position in locations)
        self.scales = None if scales is None else tuple(int(position) for position in scales)
        self.kl_weight = float(kl_weight)

    @property
    def has_offsets(self) -> bool:
        return any(len(terms) for terms in self.slope_columns)

    @property
    def leaves(self) -> tuple[int, ...]:
        """The embedded columns the network reads: those no other embedded column is nested in."""
        return tuple(position for position in self.embedded if not any(self.parents[other] == position for other in self.embedded))

    @property
    def embedding_width(self) -> int:
        return sum(int(self.embed_means[self.embedded.index(position)].shape[1]) for position in self.leaves)

    def taus(self) -> tuple[jax.Array, ...]:
        return tuple(positive(raw) for raw in self.raw_taus)

    def embed_taus(self) -> tuple[jax.Array, ...]:
        return tuple(positive(raw) for raw in self.embed_raw_taus)

    def offsets(self, category_index, numeric_z):
        """Each row's offset mean and variance, ``(batch,)`` each, summed over the columns and terms."""
        mean = jnp.zeros((category_index.shape[0],), dtype=jnp.float32)
        variance = jnp.zeros_like(mean)
        for column, means, raw_scales, tau, terms in zip(
            self.index_columns, self.means, self.raw_scales, self.taus(), self.slope_columns
        ):
            if not terms:
                continue
            group = category_index[:, column]
            seen = (group > 0)[:, None]
            row_mean = jnp.where(seen, means[group], 0.0)
            row_sd = jnp.where(seen, positive(raw_scales[group]), tau[None, :])
            covariate = jnp.stack([jnp.ones_like(mean) if term < 0 else numeric_z[:, term] for term in terms], axis=1)
            mean = mean + jnp.sum(row_mean * covariate, axis=1)
            variance = variance + jnp.sum(jnp.square(row_sd * covariate), axis=1)
        return mean, variance

    def column_offsets(self, category_index, numeric_z) -> list[tuple[int, jax.Array]]:
        """``(column, mean)`` for each grouping column with offsets: its part of ``offsets``' mean, ``(batch,)``."""
        parts = []
        for column, means, terms in zip(self.index_columns, self.means, self.slope_columns):
            if not terms:
                continue
            group = category_index[:, column]
            row_mean = jnp.where((group > 0)[:, None], means[group], 0.0)
            covariate = jnp.stack(
                [jnp.ones((group.shape[0],), dtype=row_mean.dtype) if term < 0 else numeric_z[:, term] for term in terms],
                axis=1,
            )
            parts.append((column, jnp.sum(row_mean * covariate, axis=1)))
        return parts

    def embed(self, category_index, key):
        """The pooled embeddings, ``(batch, embedding_width)``: a posterior draw with a key, the posterior mean without.

        An unseen or missing group (index 0) reads the prior: a draw from it in training, its mean 0 at prediction.
        A nested group's vector is added to its class's, so an unseen store of a known class reads the class's.
        """
        own = {}
        for number, (position, means, raw_scales, tau) in enumerate(
            zip(self.embedded, self.embed_means, self.embed_raw_scales, self.embed_taus())
        ):
            group = category_index[:, self.index_columns[position]]
            seen = (group > 0)[:, None]
            mean = jnp.where(seen, means[group], 0.0)
            if key is None:
                own[position] = mean
                continue
            scale = jnp.where(seen, positive(raw_scales[group]), tau[None, :])
            noise = jax.random.normal(jax.random.fold_in(key, number), mean.shape, dtype=mean.dtype)
            own[position] = mean + scale * noise

        def block(position):
            parent = self.parents[position]
            return own[position] if parent < 0 else own[position] + block(parent)

        return jnp.concatenate([block(position) for position in self.leaves], axis=1)

    def penalty(self):
        """KL divergence of every seen group's posteriors from their priors, and the taus' half-normal prior, per row."""
        total = jnp.asarray(0.0, dtype=jnp.float32)
        parts = (
            (self.means, self.raw_scales, self.taus()),
            (self.embed_means, self.embed_raw_scales, self.embed_taus()),
        )
        for all_means, all_scales, all_taus in parts:
            for means, raw_scales, tau in zip(all_means, all_scales, all_taus):
                total = total + jnp.sum(_kl(means[1:], positive(raw_scales[1:]), tau))
                total = total + jnp.sum(jnp.square(tau)) / (2.0 * TAU_PRIOR**2)
        return total * self.kl_weight

    def shift(self, raw, mean, variance, nodes: int | None):
        """``raw`` with each row's offset: on ``nodes`` folded into the member axis, or in closed form for ``None``."""
        mask = np.zeros((raw.shape[-1],), dtype=np.float32)
        mask[list(self.locations)] = 1.0
        mask = jnp.asarray(mask)
        if nodes is None:
            moved = raw + mean[:, None, None] * mask
            if not self.scales:
                return moved
            positions = list(self.scales)
            widened = raw_of(jnp.sqrt(jnp.square(positive(moved[..., positions])) + variance[:, None, None]))
            return moved.at[..., positions].set(widened)
        points = jnp.asarray(unit_nodes(nodes))
        offsets = mean[:, None] + jnp.sqrt(variance)[:, None] * points[None, :]
        expanded = raw[:, :, None, :] + offsets[:, None, :, None] * mask
        return expanded.reshape(raw.shape[0], raw.shape[1] * int(nodes), raw.shape[-1])

    def __call__(self, raw, category_index, numeric_z, *, training: bool):
        if raw.ndim != 3:
            raise ShapeError(f"random effects read a head output (batch, member, output); received {raw.shape}.")
        if not self.has_offsets:
            return raw
        mean, variance = self.offsets(category_index, numeric_z)
        if training:
            return self.shift(raw, mean, variance, TRAIN_NODES)
        return self.shift(raw, mean, variance, None if self.scales is not None else PREDICT_NODES)


def embed_key(key):
    """The key that draws a training step's pooled embeddings, ``None`` at prediction."""
    return None if key is None else jax.random.fold_in(key, _EMBED_STREAM)


def report(module: RandomEffects, spec: dict, encoder, unit: float) -> dict:
    """``random_effects_``: per grouping column, ``tau`` per term and every seen group's posterior mean and sd.

    A nested column also names the column it is ``within``; its means are its deviations from its class's.
    """
    out = {}
    columns = {column.name: column for column in encoder.columns}
    names = list(spec)
    embedded = {position: number for number, position in enumerate(module.embedded)}
    for position, ((name, terms), means, raw_scales, tau) in enumerate(
        zip(spec.items(), module.means, module.raw_scales, module.taus())
    ):
        entry = {
            "within": None if module.parents[position] < 0 else names[module.parents[position]],
            "terms": list(offset_terms(terms)),
            "tau": np.asarray(tau, dtype=np.float64) * unit,
            "categories": list(columns[name].categories),
            "mean": np.asarray(means[1:], dtype=np.float64) * unit,
            "sd": np.asarray(positive(raw_scales[1:]), dtype=np.float64) * unit,
        }
        if position in embedded:
            number = embedded[position]
            entry["embedding_tau"] = np.asarray(module.embed_taus()[number], dtype=np.float64)
            entry["embedding"] = np.asarray(module.embed_means[number][1:], dtype=np.float64)
            entry["embedding_sd"] = np.asarray(positive(module.embed_raw_scales[number][1:]), dtype=np.float64)
        out[name] = entry
    return out


def laplace_update(loglik_grads, prior_mean, prior_var, covariate, groups, n_groups: int, steps: int = 25):
    """Gaussian posteriors of each group's offset terms from rows, by Newton steps on the log posterior.

    ``loglik_grads(offset)`` returns each row's first and second derivative of
    its log likelihood with respect to its location offset, ``(rows,)`` each.
    ``covariate`` is ``(rows, T)``, each term's multiplier on the row;
    ``groups`` maps rows to ``0..n_groups-1``; ``prior_mean`` and
    ``prior_var`` are ``(n_groups, T)``. Returns the posterior mean and the
    marginal variances, ``(n_groups, T)`` each. A normal likelihood is
    quadratic in the offset, so one step is exact there.
    """
    covariate = np.asarray(covariate, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    prior_mean = np.asarray(prior_mean, dtype=np.float64)
    prior_precision = 1.0 / np.asarray(prior_var, dtype=np.float64)
    terms = covariate.shape[1]

    def curvature(second):
        hessian = np.zeros((n_groups, terms, terms))
        hessian[:, np.arange(terms), np.arange(terms)] = -prior_precision
        np.add.at(hessian, groups, second[:, None, None] * covariate[:, :, None] * covariate[:, None, :])
        # Where the likelihood is not log-concave in the offset, the step keeps the prior's curvature.
        return _negative_definite(np.where(np.isfinite(hessian), hessian, 0.0))

    mean = prior_mean.copy()
    for _ in range(int(steps)):
        offset = np.sum(mean[groups] * covariate, axis=1)
        first, second = (np.asarray(value, dtype=np.float64) for value in loglik_grads(offset))
        gradient = -(mean - prior_mean) * prior_precision
        np.add.at(gradient, groups, np.where(np.isfinite(first), first, 0.0)[:, None] * covariate)
        step = np.linalg.solve(curvature(second), gradient[..., None])[..., 0]
        mean = mean - step
        if np.max(np.abs(step)) < 1e-6:
            break
    _, second = loglik_grads(np.sum(mean[groups] * covariate, axis=1))
    covariance = np.linalg.inv(-curvature(np.asarray(second, dtype=np.float64)))
    return mean, np.diagonal(covariance, axis1=1, axis2=2).copy()


def _negative_definite(hessian):
    """The Hessian with its eigenvalues capped below at a small negative number, so a Newton step climbs."""
    values, vectors = np.linalg.eigh(hessian)
    values = np.minimum(values, -1e-8)
    return np.einsum("gij,gj,gkj->gik", vectors, values, vectors)


__all__ = [
    "EMBEDDING",
    "INTERCEPT",
    "PREDICT_NODES",
    "RANDOM_EFFECT_TASKS",
    "TAU_PRIOR",
    "TRAIN_NODES",
    "RandomEffects",
    "check_random_effects",
    "embed_key",
    "embedding_leaves",
    "laplace_update",
    "location_outputs",
    "nesting",
    "offset_terms",
    "report",
    "unit_nodes",
]
