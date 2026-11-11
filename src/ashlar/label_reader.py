"""A label reader: a predictive distribution read from the training labels at prediction time.

The reader has no weights from the columns to the target. For a query row it first fits a trend, a ridge
regression of the context labels on features of the columns: a constant, each column as the normal scores of its
ranks and on its own robust scale, and learned ReLU units. The ridge is solved in closed form inside the forward
pass, which also gives every context row's leave-one-out residual through its leverage. Then label attention:
each head scores every context row by a learned squared distance between keys, with its own sharpness, the scores
are softmaxed over the context and the heads are mixed. The query's predictive distribution is its trend plus the
weighted empirical distribution of the context rows' leave-one-out residuals, a step CDF whose CRPS is exact.

Each column's weight in the trend and the keys is set in the forward pass from the context labels: how much of
their variance the column's bins explain (eta squared, against its value under no relation). A column that
carries the target counts for more on this table without a parameter learned per column.

Training minimizes that exact CRPS on batches of training rows. A step's query rows are never in its context, and
a random share of the other rows is dropped from it too, so a trained reader has seen contexts of many sizes.
Training stops on the validation rows' CRPS, which reads the training rows as context; afterwards the validation
rows join the context.

A network with a density head and a reader are two views of a table: the network fits a smooth map from the
columns, the reader the labels themselves. Neither wins everywhere: a target that is a linear function of the
columns, or one with few values, is the reader's; a smooth nonlinear surface is the network's. ``mixture_crps``
scores a two-model mixture at every weight in a grid, exactly, and ``mixture_quantiles`` reads its quantiles.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.special import ndtri

POOL_WEIGHTS = np.linspace(0.0, 1.0, 21)


class _Columns:
    """Each usable column as normal scores of its ranks among the fitted values, and on its own robust scale.

    A column with fewer than two distinct values is dropped; a missing cell reads as 0 in both views.
    """

    def __init__(self, x: np.ndarray):
        self.columns = []  # (index, sorted values, centre, spread)
        for j in range(x.shape[1]):
            values = np.sort(x[~np.isnan(x[:, j]), j])
            if len(values) < 2 or values[0] == values[-1]:
                continue
            centre = np.median(values)
            spread = (np.quantile(values, 0.75) - np.quantile(values, 0.25)) / 1.349
            self.columns.append((j, values, centre, spread if spread > 0 else values.std()))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """``(rows, 2 * kept)`` float32: the normal scores of every kept column, then the robust scales."""
        if not self.columns:  # nothing varies: one constant column keeps the shapes defined
            return np.zeros((len(x), 2), dtype=np.float32)
        scores, scales = [], []
        for j, values, centre, spread in self.columns:
            column = x[:, j]
            rank = (np.searchsorted(values, column, side="left") + np.searchsorted(values, column, side="right")) / (2 * len(values))
            scores.append(np.where(np.isnan(column), 0.0, ndtri(np.clip(rank, 0.5 / len(values), 1 - 0.5 / len(values)))))
            scales.append(np.where(np.isnan(column), 0.0, np.clip((column - centre) / spread, -6.0, 6.0)))
        return np.concatenate([np.stack(scores, axis=1), np.stack(scales, axis=1)], axis=1).astype(np.float32)


def _init(key, d, heads, key_size, width):
    k1, k2 = jax.random.split(key)
    return {
        "w1": jax.random.normal(k1, (d, width)) / np.sqrt(d),
        "b1": jnp.zeros(width),
        "proj": jax.random.normal(k2, (heads, d + width, key_size)) / np.sqrt(d + width),
        "t1": jax.random.normal(jax.random.fold_in(key, 7), (d, width)) / np.sqrt(d),
        "tb": jnp.zeros(width),
        # The ridge penalty starts at the number of trend features and is stored divided by 10, so Adam moves it faster.
        "log_ridge": jnp.asarray(np.log(1.0 + 2 * d + width) / 10.0),
        "log_sharp": jnp.log(jnp.geomspace(0.5, 16.0, heads)),
        "mix": jnp.zeros(heads),
        "rel": jnp.asarray([1.0, 0.5]),  # relevance = softplus(a + b * log(eta^2 / its value under no relation))
    }


def _relevance(params, onehot, y, mask):
    """Each column's multiplier from the context labels: how much of their variance its bins explain."""
    weight = mask.astype(y.dtype)
    count = jnp.einsum("ndb,n->db", onehot, weight)
    total = weight.sum()
    mean = (weight * y).sum() / total
    centered = (y - mean) * weight
    bin_mean = jnp.einsum("ndb,n->db", onehot, centered) / jnp.maximum(count, 1.0)
    eta = (count * bin_mean**2).sum(axis=1) / jnp.maximum((centered**2).sum(), 1e-6)
    signal = jnp.log(jnp.maximum(eta, 1e-8) * (total - 1.0) / (onehot.shape[2] - 1.0))
    rel = jax.nn.softplus(params["rel"][0] + params["rel"][1] * signal)
    return rel / jnp.sqrt(jnp.mean(rel**2) + 1e-8)


def _keys(params, z):
    feats = jnp.concatenate([z, jax.nn.relu(z @ params["w1"] + params["b1"])], axis=1)
    return jnp.einsum("nf,hfk->nhk", feats, params["proj"])


def _trend_features(params, z, raw):
    return jnp.concatenate([jnp.ones((len(z), 1), dtype=z.dtype), raw, z, jax.nn.relu(z @ params["t1"] + params["tb"])], axis=1)


def _attend(params, x_ctx, onehot_ctx, y_ctx, mask, x_query):
    """The context rows' leave-one-out trend residuals, the queries' trends, and ``(queries, context)`` weights.

    ``mask`` marks the usable context rows; the ridge fit and the attention read only those.
    """
    d = x_ctx.shape[1] // 2
    rel = _relevance(params, onehot_ctx, y_ctx, mask)
    z_ctx, raw_ctx = x_ctx[:, :d] * rel, x_ctx[:, d:] * rel
    z_query, raw_query = x_query[:, :d] * rel, x_query[:, d:] * rel
    weight = mask.astype(y_ctx.dtype)
    phi = _trend_features(params, z_ctx, raw_ctx)
    gram = (phi * weight[:, None]).T @ phi + jnp.exp(10.0 * params["log_ridge"]) * jnp.eye(phi.shape[1], dtype=phi.dtype)
    solved = jnp.linalg.solve(gram, phi.T)
    beta = solved @ (weight * y_ctx)
    leverage = jnp.einsum("np,pn->n", phi, solved)
    residual = (y_ctx - phi @ beta) / jnp.maximum(1.0 - weight * leverage, 1e-3)
    kq, kc = _keys(params, z_query), _keys(params, z_ctx)
    distance = (kq**2).sum(-1)[:, None, :] + (kc**2).sum(-1)[None, :, :] - 2.0 * jnp.einsum("qhk,chk->qch", kq, kc)
    logits = jnp.where(mask[None, :, None], -jnp.exp(params["log_sharp"]) * distance / kq.shape[-1], -1e30)
    weights = jnp.einsum("qch,h->qc", jax.nn.softmax(logits, axis=1), jax.nn.softmax(params["mix"]))
    return residual, _trend_features(params, z_query, raw_query) @ beta, weights


def _ecdf_crps(weights, y_ctx, y_query):
    """Exact CRPS of each query's weighted empirical CDF of the context values."""
    order = jnp.argsort(y_ctx)
    t = y_ctx[order]
    cdf = jnp.cumsum(weights[:, order], axis=1)[:, :-1]
    gap = t[1:] - t[:-1]
    below = jnp.clip(y_query[:, None] - t[None, :-1], 0.0, gap[None, :])
    inner = (cdf**2 * below + (1.0 - cdf) ** 2 * (gap[None, :] - below)).sum(axis=1)
    return inner + jax.nn.relu(t[0] - y_query) + jax.nn.relu(y_query - t[-1])


def _loss(params, x, onehot, y, query, drop):
    mask = drop.at[query].set(False)
    residual, trend, weights = _attend(params, x, onehot, y, mask, x[query])
    return _ecdf_crps(weights, residual, y[query] - trend).mean()


@jax.jit
def _step(params, state, x, onehot, y, query, drop, rate):
    value, grads = jax.value_and_grad(_loss)(params, x, onehot, y, query, drop)
    updates, state = optax.adam(1.0).update(grads, state)
    return jax.tree_util.tree_map(lambda p, u: p + rate * u, params, updates), state, value


@jax.jit
def _read(params, x_ctx, onehot_ctx, y_ctx, x_query):
    return _attend(params, x_ctx, onehot_ctx, y_ctx, jnp.ones(len(y_ctx), dtype=bool), x_query)


def _predict(params, x_ctx, onehot_ctx, y_ctx, x_query, chunk=128):
    """Residuals, and each query row's trend and weights, in fixed-size pieces so one compiled shape serves all."""
    trends, out = [], []
    residual = None
    for start in range(0, len(x_query), chunk):
        piece = x_query[start : start + chunk]
        padded = np.concatenate([piece, np.repeat(piece[:1], chunk - len(piece), axis=0)])
        residual, trend, weights = _read(params, x_ctx, onehot_ctx, y_ctx, jnp.asarray(padded))
        trends.append(np.asarray(trend)[: len(piece)])
        out.append(np.asarray(weights)[: len(piece)])
    return np.asarray(residual), np.concatenate(trends), np.concatenate(out)


class LabelReader:
    """Read a predictive distribution from the training labels (see the module docstring).

    ``fit(X, y, validation=(X_val, y_val))`` trains on ``X`` with ``X_val`` stopping training: a check every
    ``check_every`` steps, and ``patience`` checks without a better validation CRPS end it, keeping the best
    weights. Columns and the target are scaled on the training and validation rows together. ``atoms(X)`` returns
    each row's predictive distribution as atoms and masses, read from every fitted row (``context="all"``) or from
    the training rows only (``context="train"``, how training judged the validation rows). ``X`` is numeric with
    NaN for missing cells; a categorical column is read through its codes.
    """

    def __init__(self, *, heads=8, key_size=16, width=64, bins=8, max_steps=4000, learning_rate=2e-3, check_every=20,
                 patience=15, random_state=0):
        self.heads, self.key_size, self.width, self.bins = heads, key_size, width, bins
        self.max_steps, self.learning_rate = max_steps, learning_rate
        self.check_every, self.patience, self.random_state = check_every, patience, random_state

    def _onehot(self, x):
        edges = ndtri(np.linspace(0.0, 1.0, self.bins + 1)[1:-1])
        return np.eye(self.bins, dtype=np.float32)[np.searchsorted(edges, x[:, : x.shape[1] // 2])]

    def fit(self, X, y, validation):
        x_fit, y_fit = np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64).reshape(-1)
        x_val, y_val = np.asarray(validation[0], dtype=np.float64), np.asarray(validation[1], dtype=np.float64).reshape(-1)
        n = len(y_fit)
        self.columns_ = _Columns(np.concatenate([x_fit, x_val]))
        x_all = self.columns_(np.concatenate([x_fit, x_val]))
        y_all = np.concatenate([y_fit, y_val])
        self.center_, self.spread_ = float(y_all.mean()), float(y_all.std() + 1e-12)
        self.x_, self.onehot_ = x_all, self._onehot(x_all)
        self.y_ = ((y_all - self.center_) / self.spread_).astype(np.float32)
        self.n_train_ = n
        rng = np.random.default_rng(3000 + self.random_state)
        params = _init(jax.random.PRNGKey(self.random_state), x_all.shape[1] // 2, self.heads, self.key_size, self.width)
        state = optax.adam(1.0).init(params)
        batch = int(min(256, max(8, n // 8)))
        x_j, onehot_j, y_j = jnp.asarray(x_all[:n]), jnp.asarray(self.onehot_[:n]), jnp.asarray(self.y_[:n])
        best, best_loss, stale = params, np.inf, 0
        for index in range(self.max_steps):
            query = jnp.asarray(rng.choice(n, size=batch, replace=False))
            drop = jnp.asarray(rng.random(n) < rng.uniform(0.6, 1.0))
            params, state, _ = _step(params, state, x_j, onehot_j, y_j, query, drop, self.learning_rate)
            if (index + 1) % self.check_every == 0:
                residual, trend, w = _predict(params, x_j, onehot_j, y_j, x_all[n:])
                value = float(np.asarray(_ecdf_crps(jnp.asarray(w), jnp.asarray(residual), jnp.asarray(self.y_[n:] - trend))).mean())
                if value < best_loss:
                    best, best_loss, stale = params, value, 0
                else:
                    stale += 1
                    if stale >= self.patience:
                        break
        self.params_, self.steps_ = best, index + 1
        return self

    def atoms(self, X, context="all"):
        """``(values, masses)``, each ``(rows, context rows)`` float64: the rows' predictive distributions."""
        rows = slice(None) if context == "all" else slice(0, self.n_train_)
        residual, trend, weights = _predict(self.params_, jnp.asarray(self.x_[rows]), jnp.asarray(self.onehot_[rows]),
                                            jnp.asarray(self.y_[rows]), self.columns_(np.asarray(X, dtype=np.float64)))
        values = self.center_ + self.spread_ * (residual.astype(np.float64)[None, :] + trend.astype(np.float64)[:, None])
        weights = weights.astype(np.float64)
        return values, weights / weights.sum(axis=1, keepdims=True)


def mixture_crps(first, second, y, weights=POOL_WEIGHTS):
    """Mean exact CRPS of ``w * first + (1 - w) * second`` at each ``w``; each model is ``(atoms, masses)`` per row."""
    atoms = np.concatenate([first[0], second[0]], axis=1)
    order = np.argsort(atoms, axis=1)
    t = np.take_along_axis(atoms, order, axis=1)

    def cumulative(masses, leading):
        padded = np.zeros_like(atoms)
        if leading:
            padded[:, : masses.shape[1]] = masses
        else:
            padded[:, -masses.shape[1] :] = masses
        return np.cumsum(np.take_along_axis(padded, order, axis=1), axis=1)[:, :-1]

    cum_first, cum_second = cumulative(first[1], True), cumulative(second[1], False)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    gap = np.diff(t, axis=1)
    below = np.clip(y[:, None] - t[:, :-1], 0.0, gap)
    tails = np.maximum(t[:, 0] - y, 0.0) + np.maximum(y - t[:, -1], 0.0)
    out = []
    for w in weights:
        cdf = w * cum_first + (1.0 - w) * cum_second
        out.append(float(((cdf**2 * below + (1.0 - cdf) ** 2 * (gap - below)).sum(axis=1) + tails).mean()))
    return np.asarray(out)


def pool_weight(first, second, y, weights=POOL_WEIGHTS):
    """The weight on ``first`` with the lowest mean CRPS on these rows (the lowest such weight), and every score."""
    scores = mixture_crps(first, second, y, weights)
    return float(weights[int(np.argmin(scores))]), scores


def mixture_quantiles(first, second, weight, levels):
    """Quantiles of ``weight * first + (1 - weight) * second`` at ``levels``: per row, the smallest atom whose CDF
    reaches each level."""
    atoms = np.concatenate([first[0], second[0]], axis=1)
    masses = np.concatenate([weight * first[1], (1.0 - weight) * second[1]], axis=1)
    order = np.argsort(atoms, axis=1)
    t = np.take_along_axis(atoms, order, axis=1)
    cdf = np.cumsum(np.take_along_axis(masses, order, axis=1), axis=1)
    target = np.asarray(levels, dtype=np.float64) - 1e-12
    index = np.stack([np.searchsorted(row, target, side="left") for row in cdf])
    return np.take_along_axis(t, np.minimum(index, t.shape[1] - 1), axis=1)
