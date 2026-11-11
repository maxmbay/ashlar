"""Dressing: several cross-fitted networks and label readers, pooled, smoothed and corrected into one distribution.

A network with a density head fits a smooth map from the columns to the target. A label reader
(``ashlar.label_reader``) predicts a row from the training labels themselves. Neither wins on every table, and a
reader predicts point masses, so a pool of the two has no density. ``ASRDressed`` puts them together in one fit and
still returns one distribution with an exact CDF and an exact log density:

1. Members. The training rows are cut into ``members`` parts. Member ``k`` holds out part ``k`` and fits a network
   and a reader on the other parts, both stopped on part ``k``. A target with at most ``few_values`` distinct
   values gets an ordinal network over those values; any other target a ``flow`` network stopped on CRPS.
2. Pool weight. One weight ``w`` for the network against the reader, from a grid of 21, by the CRPS of every
   training row under the member that held it out.
3. Smoothing. Each reader's points become normals, with Silverman's bandwidth on the row's points times a factor
   from a grid of five, plus 1% of a wide normal at the training labels' mean and spread. One factor is picked by
   the held-out log score and one by the held-out CRPS. A member's pool, ``w`` times its network plus ``1 - w``
   times its smoothed reader, now has a density.
4. Corrector. One more network learns where a label falls inside its own row's pool: it is fitted to the label's
   normal score under the pool of the member that held the row out, on the held-out rows of every member but the
   last, and stopped on the last member's. A member's corrected CDF at ``y`` is the corrector's CDF at the normal
   score of ``y`` under that member's pool.
5. Gate. One weight, from the same grid, for the corrected pools against the pools smoothed at the CRPS factor, by
   the CRPS of the last member's held-out rows. A corrector that did not help on those rows gets weight 0.

The prediction is the gate times the equal mixture of the members' corrected pools, plus the rest times the equal
mixture of their smoothed pools. Bagging members is standard, and correcting a prediction on its own probability
scale is the idea behind probability calibration and behind composing models on that scale; the reader, and the
held-out bagged pool as the scale, are what this module adds.

The fit is eleven networks' worth of training (five networks, five readers, one corrector), so it is the slowest
model in Ashlar. One target column, exactly observed; no sample weights.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri
from sklearn.base import BaseEstimator

from ashlar import observed
from ashlar.errors import AshlarError
from ashlar.estimator import ASRModel
from ashlar.label_reader import POOL_WEIGHTS, LabelReader, mixture_crps, mixture_quantiles

# A density is read as atoms at these levels wherever a CRPS is searched over a grid; each atom carries the mass
# up to its level.
LEVELS = np.concatenate([[0.001, 0.0025], np.linspace(0.005, 0.995, 199), [0.9975, 0.999]])
LEVEL_MASS = np.diff(np.concatenate([[0.0], LEVELS]))
LEVEL_MASS[-1] += 1.0 - LEVELS[-1]
FACTORS = (1.0, 0.5, 0.25, 0.125, 2.0)  # bandwidth factors; the first wins a tie
FLOOR = 0.01  # the wide normal's share of a smoothed reader, at most
Z_GRID = np.linspace(-7.0, 7.0, 113)  # the stopping rows' pools are read at these normal scores
CRPS_ROWS = 120  # held-out rows per member in the CRPS factor's search
PIECE = 5_000_000  # array cells per piece when a smoothed reader is evaluated
PREDICT_ROWS = 512  # rows per piece at prediction time
MEMBER_SETTINGS = {"max_epochs": 1000, "cooldown_epochs": 3}

_READ = None


def _read(dist, t):
    """A network distribution's CDF and log density at ``t``, compiled once per shape."""
    global _READ
    if _READ is None:
        import jax

        _READ = jax.jit(lambda dist, t: (dist.cdf(t), dist.log_prob(t)))
    return _READ(dist, t)


def _take(X, index):
    return X.iloc[index] if hasattr(X, "iloc") else X[index]


def atoms_crps(atoms, masses, y):
    """Exact CRPS of each row's distribution on ``atoms`` (rows, m) with ``masses`` (rows, m), at ``y`` (rows,)."""
    order = np.argsort(atoms, axis=1)
    t = np.take_along_axis(atoms, order, 1)
    cdf = np.cumsum(np.take_along_axis(masses, order, 1), axis=1)[:, :-1]
    gap = np.diff(t, axis=1)
    below = np.clip(y[:, None] - t[:, :-1], 0.0, gap)
    return (cdf**2 * below + (1 - cdf) ** 2 * (gap - below)).sum(1) + np.maximum(t[:, 0] - y, 0) + np.maximum(y - t[:, -1], 0)


def _level_crps(atoms, masses, y):
    """The CRPS of each row's ``(atoms, masses)`` read through its quantiles at ``LEVELS``."""
    q = mixture_quantiles((atoms, masses), (atoms, masses), 1.0, LEVELS)
    return atoms_crps(q, np.broadcast_to(LEVEL_MASS, q.shape), y)


def _value_masses(F):
    """Each value's probability from the CDF ``F`` (rows, values - 1) at the values below the top one."""
    F = np.clip(F, 0.0, 1.0)
    return np.diff(np.concatenate([np.zeros((len(F), 1)), F, np.ones((len(F), 1))], axis=1), axis=1)


def _curve(first, second, y, chunk=100):
    """Mean CRPS of ``w * first + (1 - w) * second`` at every ``w`` in ``POOL_WEIGHTS``, a few rows at a time."""
    total = np.zeros(len(POOL_WEIGHTS))
    for start in range(0, len(y), chunk):
        rows = slice(start, start + chunk)
        total += len(y[rows]) * mixture_crps((first[0][rows], first[1][rows]), (second[0][rows], second[1][rows]), y[rows])
    return total / len(y)


def _union(models):
    """The equal mixture of ``(atoms, masses)`` models, as one ``(atoms, masses)``."""
    return (np.concatenate([np.asarray(a) for a, _ in models], axis=1),
            np.concatenate([np.asarray(m) for _, m in models], axis=1) / len(models))


def _combine(net, reader, weight):
    """Network atoms and reader atoms as one ``(atoms, masses)`` at ``weight`` on the network."""
    return (np.concatenate([np.asarray(net[0]), reader[0]], axis=1),
            np.concatenate([weight * np.asarray(net[1]), (1 - weight) * reader[1]], axis=1))


def _interp_rows(levels, table, u):
    """Each row's ``table`` (rows, len(levels)) read at its own probabilities ``u`` (rows, k)."""
    return np.stack([np.interp(row_u, levels, row) for row_u, row in zip(u, table, strict=True)])


def _normal_score(F, S):
    """Phi^-1 of a CDF value, read from the nearer tail (``S`` is the survival)."""
    return np.where(F < 0.5, ndtri(np.clip(F, 1e-300, 1.0)), -ndtri(np.clip(S, 1e-300, 1.0)))


def _floor_share(y, centre, spread, held_out_crps, cost):
    """The wide normal's share of a smoothed reader: ``FLOOR``, or less where that would cost more than ``cost``
    times the raw pool's held-out CRPS. A share ``e`` beside a sharp pool costs ``e^2`` times the wide normal's
    own CRPS on the labels."""
    if cost is None:
        return FLOOR
    score = (y - centre) / spread
    wide = spread * (score * (2 * ndtr(score) - 1) + 2 * np.exp(-0.5 * score**2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    return float(min(FLOOR, np.sqrt(float(cost) * held_out_crps / wide.mean())))


class _Smoothed:
    """Rows' reader distributions as densities: ``sum_j p_j N(a_j, h^2)`` at weight ``1 - floor`` plus ``floor`` of
    ``N(centre, spread^2)``. ``h`` is Silverman's rule on the row's weighted atoms, times ``factor``."""

    def __init__(self, atoms, masses, centre, spread, factor, floor=FLOOR):
        atoms, masses = np.asarray(atoms, np.float64), np.asarray(masses, np.float64)
        masses = np.where(masses >= 1e-12, masses, 0.0)
        width = int((masses > 0).sum(axis=1).max())
        top = np.argsort(-masses, axis=1)[:, :width]
        a, p = np.take_along_axis(atoms, top, 1), np.take_along_axis(masses, top, 1)
        p /= p.sum(axis=1, keepdims=True)
        mean = (p * a).sum(axis=1)
        sd = np.sqrt((p * (a - mean[:, None]) ** 2).sum(axis=1))
        order = np.argsort(a, axis=1)
        ordered, cum = np.take_along_axis(a, order, 1), np.cumsum(np.take_along_axis(p, order, 1), axis=1)
        rows = np.arange(len(a))
        iqr = (ordered[rows, np.minimum((cum < 0.75).sum(1), width - 1)]
               - ordered[rows, np.minimum((cum < 0.25).sum(1), width - 1)])
        scale = np.where(iqr > 0, np.minimum(sd, iqr / 1.349), sd)
        h = np.maximum(0.9 * scale * (1.0 / (p**2).sum(axis=1)) ** -0.2, 1e-5 * spread)
        self.h = np.maximum(h * factor, 1e-5 * spread)
        self.a, self.p, self.centre, self.spread, self.floor = a, p, centre, spread, float(floor)
        self.lo = np.minimum(ordered[:, 0] - 40 * self.h, centre - 40 * spread)
        self.hi = np.maximum(ordered[:, -1] + 40 * self.h, centre + 40 * spread)

    def terms(self, rows, y):
        """The CDF, survival and density at ``y`` (len(rows), k) for those rows."""
        F, S, f = np.empty(y.shape), np.empty(y.shape), np.empty(y.shape)
        step = max(1, PIECE // (y.shape[1] * self.a.shape[1]))
        for s in range(0, len(rows), step):
            r, piece = rows[s : s + step], slice(s, s + step)
            x = (y[piece, :, None] - self.a[r, None, :]) / self.h[r, None, None]
            p = self.p[r, None, :]
            F[piece] = (p * ndtr(x)).sum(-1)
            S[piece] = (p * ndtr(-x)).sum(-1)
            f[piece] = (p * np.exp(-0.5 * x * x)).sum(-1) / (np.sqrt(2 * np.pi) * self.h[r, None])
        b, floor = (y - self.centre) / self.spread, self.floor
        return ((1 - floor) * F + floor * ndtr(b), (1 - floor) * S + floor * ndtr(-b),
                (1 - floor) * f + floor * np.exp(-0.5 * b * b) / (np.sqrt(2 * np.pi) * self.spread))


class _Pool:
    """One member's pooled distribution on some rows: ``weight`` times the network plus the rest times the smoothed
    reader. ``network`` is the network's ``predict_dist`` on the rows, or, for a target with few values, its
    probability of each of ``values`` (rows, values)."""

    def __init__(self, network, reader, weight, values=None):
        self.network, self.reader, self.w, self.values = network, reader, float(weight), values

    def network_terms(self, y):
        """The network's CDF and density at ``y`` (rows, k)."""
        if self.values is not None:
            F = (self.network[:, None, :] * (self.values <= y[:, :, None])).sum(-1)
            return F, np.zeros_like(F)
        import jax.numpy as jnp

        F, log_f = _read(self.network, jnp.asarray(y.T))
        return np.asarray(F, np.float64).T, np.exp(np.asarray(log_f, np.float64).T)

    def terms(self, y):
        """The pooled CDF, survival and density at ``y`` (rows, k)."""
        Fn, fn = self.network_terms(y)
        Fr, Sr, fr = self.reader.terms(np.arange(len(y)), y)
        return self.w * Fn + (1 - self.w) * Fr, self.w * (1 - Fn) + (1 - self.w) * Sr, self.w * fn + (1 - self.w) * fr

    def terms_at(self, i, j, y):
        """``terms`` at the entries (i, j) only; the network is read on every row at once, the reader at those."""
        Fn, fn = (t[i, j] for t in self.network_terms(y))
        Fr, Sr, fr = (t[:, 0] for t in self.reader.terms(i, y[i, j][:, None]))
        return self.w * Fn + (1 - self.w) * Fr, self.w * (1 - Fn) + (1 - self.w) * Sr, self.w * fn + (1 - self.w) * fr

    def z(self, y):
        """The normal score of ``y`` (rows, k) under the pool."""
        F, S, _ = self.terms(y)
        return _normal_score(F, S)

    def log_pdf(self, y):
        """The pooled log density at ``y`` (rows, k), summed in logs so a far tail keeps its value; a density only."""
        import jax.numpy as jnp

        net = np.asarray(_read(self.network, jnp.asarray(y.T))[1], np.float64).T
        reader = np.log(np.maximum(self.reader.terms(np.arange(len(y)), y)[2], 1e-300))
        with np.errstate(divide="ignore"):
            return np.logaddexp(np.log(self.w) + net, np.log(1 - self.w) + reader)

    def probabilities(self):
        """Each value's pooled probability (rows, values); a target with few values only."""
        mids = np.broadcast_to((self.values[:-1] + self.values[1:]) / 2, (len(self.network), len(self.values) - 1)).copy()
        return self.w * self.network + (1 - self.w) * _value_masses(self.reader.terms(np.arange(len(mids)), mids)[0])


def _solve(pools, u, start, evaluate, tolerance):
    """Each row's ``y`` where an increasing CDF reaches ``u`` (rows, k), from ``start``: Newton steps inside a
    bisection bracket wide enough for every pool. ``evaluate(i, j, y)`` gives the CDF and density at the unsolved
    entries."""
    n, k = u.shape
    spread = pools[0].reader.spread
    y = np.array(start, dtype=np.float64)
    lo = np.repeat(np.minimum(np.min([p.reader.lo for p in pools], axis=0), y.min(axis=1) - spread)[:, None], k, axis=1)
    hi = np.repeat(np.maximum(np.max([p.reader.hi for p in pools], axis=0), y.max(axis=1) + spread)[:, None], k, axis=1)
    y = np.clip(y, lo, hi)
    active = np.ones((n, k), dtype=bool)
    for _ in range(100):
        i, j = np.nonzero(active)
        F, f = evaluate(i, j, y)
        g = F - u[i, j]
        done = (np.abs(g) <= tolerance) | (hi[i, j] - lo[i, j] <= 1e-12 * np.maximum(np.abs(y[i, j]), spread))
        active[i[done], j[done]] = False
        i, j, g, f = i[~done], j[~done], g[~done], f[~done]
        if not len(i):
            break
        lo[i, j] = np.where(g < 0, y[i, j], lo[i, j])
        hi[i, j] = np.where(g > 0, y[i, j], hi[i, j])
        step = y[i, j] - g / np.maximum(f, 1e-300)
        inside = (step > lo[i, j]) & (step < hi[i, j]) & np.isfinite(step)
        y[i, j] = np.where(inside, step, 0.5 * (lo[i, j] + hi[i, j]))
    return y


def _pool_quantiles(pools, u, start):
    """Quantiles at ``u`` (rows, k) of the equal mixture of ``pools``."""

    def evaluate(i, j, y):
        terms = [p.terms_at(i, j, y) for p in pools]
        return np.mean([t[0] for t in terms], axis=0), np.mean([t[2] for t in terms], axis=0)

    return _solve(pools, u, start, evaluate, 1e-9 + 2e-7 * max(p.w for p in pools))  # the network's CDF is float32


def _corrected_terms(pool, corrector, i, j, y):
    """The CDF and density at the entries (i, j) of ``y`` of the corrector's CDF at the normal score under ``pool``."""
    import jax.numpy as jnp

    G, S, g = pool.terms_at(i, j, y)
    z = _normal_score(G, S)
    full = np.zeros(y.shape)
    full[i, j] = z
    Fz, log_f = _read(corrector, jnp.asarray(full.T))
    log_f = np.asarray(log_f, np.float64).T[i, j]
    density = np.exp(log_f + np.log(np.maximum(g, 1e-300)) + 0.5 * z**2 + 0.5 * np.log(2 * np.pi))
    return np.asarray(Fz, np.float64).T[i, j], density


class _Corrector(ASRModel):
    """``ASRModel`` on the labels' normal scores under their pools, stopped on the stopping rows' CRPS in ``y``.

    ``stop_on`` is set before ``fit``: ``("flow", y at Z_GRID per stopping row, None, y)`` or
    ``("values", normal scores of the cell edges per stopping row, values, y)``.
    """

    stop_on = None

    def _validation_score(self, validation):
        import equinox as eqx
        import jax.numpy as jnp

        from ashlar.distributions import CRPS_LEVELS, Mixture, head_distribution
        from ashlar.train import EVAL_CHUNK_ROWS

        *features, _, _ = validation
        kind, table, values, y = self.stop_on
        task, n = self.task, len(y)
        pieces = [(s, tuple(a[s : s + EVAL_CHUNK_ROWS] for a in features)) for s in range(0, n, EVAL_CHUNK_ROWS)]
        if kind == "flow":
            alphas = (np.arange(CRPS_LEVELS) + 0.5) / CRPS_LEVELS
            levels = jnp.asarray(alphas.reshape(-1, 1), dtype=jnp.float32)

            @eqx.filter_jit
            def quantiles(core, *chunk):
                return Mixture(head_distribution(task, core(*chunk, key=None))).quantile(levels)

            def score(core):
                mean, std = (float(np.asarray(v, np.float64).reshape(-1)[0]) for v in self._target_affine())
                z = mean + std * np.concatenate([np.asarray(quantiles(core, *c), dtype=np.float64) for _, c in pieces], axis=1)
                position = np.clip((z.T - Z_GRID[0]) / (Z_GRID[1] - Z_GRID[0]), 0.0, len(Z_GRID) - 1.000001)
                index = np.floor(position).astype(int)
                frac, rows = position - index, np.arange(n)[:, None]
                gap = y[None, :] - (table[rows, index] * (1 - frac) + table[rows, index + 1] * frac).T
                return -float((2.0 * np.maximum(alphas[:, None] * gap, (alphas[:, None] - 1.0) * gap)).mean())

            return score

        @eqx.filter_jit
        def cdf(core, edges, *chunk):
            return Mixture(head_distribution(task, core(*chunk, key=None))).cdf(edges)

        def score(core):
            mean, std = (float(np.asarray(v, np.float64).reshape(-1)[0]) for v in self._target_affine())
            F = np.concatenate([np.asarray(cdf(core, jnp.asarray(((table[s : s + EVAL_CHUNK_ROWS] - mean) / std).T, dtype=jnp.float32), *c),
                                           dtype=np.float64) for s, c in pieces], axis=1).T
            return -float(atoms_crps(np.broadcast_to(values, (n, len(values))), _value_masses(F), y).mean())

        return score


class _Columns:
    """The columns as float64 for the reader: numbers as they are, any other column by the rank of its value among
    the training values (missing or unseen is NaN)."""

    def __init__(self, X):
        self.codes = {}
        if hasattr(X, "iloc"):
            for j in range(X.shape[1]):
                column = X.iloc[:, j]
                if getattr(column.dtype, "kind", "O") not in "biuf":
                    kept = column[~column.isna().to_numpy()]
                    self.codes[j] = np.unique(np.asarray([str(v) for v in kept.to_numpy(dtype=object)], dtype=object).astype(str))

    def __call__(self, X):
        if not hasattr(X, "iloc"):
            try:
                return np.asarray(X, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise AshlarError("pass a DataFrame when a column is not numeric.") from error
        out = np.empty((len(X), X.shape[1]), dtype=np.float64)
        for j in range(X.shape[1]):
            column = X.iloc[:, j]
            if j not in self.codes:
                out[:, j] = column.to_numpy(dtype=np.float64, na_value=np.nan)
                continue
            known, missing = self.codes[j], column.isna().to_numpy()
            text = np.asarray([str(v) for v in column.to_numpy(dtype=object)], dtype=object).astype(str)
            if not len(known):
                out[:, j] = np.nan
                continue
            index = np.minimum(np.searchsorted(known, text), len(known) - 1)
            out[:, j] = np.where((known[index] == text) & ~missing, index, np.nan)
        return out


class DressedDistribution:
    """A dressed model's prediction for some rows: what ``ASRDressed.predict_dist`` returns.

    One distribution per row. A value array's last axis runs over the rows, as with every Ashlar distribution:
    ``cdf(y)`` and ``log_prob(y)`` take ``(rows,)`` or ``(k, rows)``. ``log_prob`` is the exact log density, or
    the log probability for a target with few values (``values`` is then the array of values, and
    ``probabilities()`` each row's probability of each one).
    """

    def __init__(self, pieces, values):
        self._pieces, self.values = pieces, values
        self.batch_shape = (sum(piece.rows for piece in pieces),)

    def _apply(self, method, value):
        value = np.asarray(value, dtype=np.float64)
        full = np.broadcast_to(value, value.shape[:-1] + self.batch_shape if value.ndim else self.batch_shape)
        flat = full.reshape(-1, self.batch_shape[0]).T
        out, start = [], 0
        for piece in self._pieces:
            out.append(getattr(piece, method)(np.ascontiguousarray(flat[start : start + piece.rows])))
            start += piece.rows
        return np.concatenate(out, axis=0).T.reshape(full.shape)

    def cdf(self, value):
        """``P(Y <= value)``."""
        return self._apply("cdf", value)

    def log_prob(self, value):
        """The log density at ``value``; for a target with few values, the log probability (``-inf`` off them)."""
        return self._apply("log_prob", value)

    def quantile(self, levels):
        """Quantiles at ``levels`` (a number, or ``k`` of them), shape ``(rows,)`` or ``(k, rows)``.

        A density's quantile solves its CDF, to the precision of the networks' float32 CDFs; a target with few
        values returns the smallest value whose CDF reaches the level.
        """
        levels = np.asarray(levels, dtype=np.float64)
        if np.any(levels <= 0.0) or np.any(levels >= 1.0):
            raise AshlarError("quantile levels must lie strictly between 0 and 1.")
        return self._apply("quantile", levels.reshape(-1, 1) if levels.ndim == 1 else levels)

    def probabilities(self):
        """``(rows, len(values))``: each row's probability of each value; a target with few values only."""
        if self.values is None:
            raise AshlarError("probabilities() is for a target with few values; this one has a density.")
        return np.concatenate([piece.probabilities for piece in self._pieces], axis=0)

    def mean(self):
        """Each row's mean: exact for a target with few values, otherwise read from the quantiles at 203 levels."""
        if self.values is not None:
            return self.probabilities() @ self.values
        return self.quantile(LEVELS).T @ LEVEL_MASS

    def crps(self, value):
        """Each row's CRPS at ``value`` (rows,): exact for a target with few values, otherwise that of the
        quantiles at 203 levels, which is how a quantile benchmark reads a density."""
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if self.values is not None:
            proba = self.probabilities()
            return atoms_crps(np.broadcast_to(self.values, proba.shape), proba, value)
        q = self.quantile(LEVELS).T
        return atoms_crps(q, np.broadcast_to(LEVEL_MASS, q.shape), value)

    def sample(self, n: int = 100, seed: int = 0):
        """``n`` draws per row, shape ``(n, rows)``, by the quantile of uniform draws."""
        u = np.random.default_rng(int(seed)).uniform(1e-9, 1.0 - 1e-9, size=(int(n), self.batch_shape[0]))
        return self._apply("quantile", u)


class _DensityPiece:
    """The dressed density on a block of rows."""

    def __init__(self, pools, fallback, corrector, gate, table):
        self.pools, self.fallback, self.corrector, self.gate, self.table = pools, fallback, corrector, gate, table
        self.rows = len(table)

    def _terms(self, i, j, y):
        F, f = 0.0, 0.0
        for pool, other in zip(self.pools, self.fallback, strict=True):
            if self.gate > 0.0:
                Fc, fc = _corrected_terms(pool, self.corrector, i, j, y)
                F, f = F + self.gate * Fc, f + self.gate * fc
            if self.gate < 1.0:
                G, _, g = other.terms_at(i, j, y)
                F, f = F + (1 - self.gate) * G, f + (1 - self.gate) * g
        return F / len(self.pools), f / len(self.pools)

    def _all(self, y):
        i, j = np.nonzero(np.ones(y.shape, dtype=bool))
        F, f = self._terms(i, j, y)
        return F.reshape(y.shape), f.reshape(y.shape)

    def cdf(self, y):
        return np.clip(self._all(y)[0], 0.0, 1.0)

    def log_prob(self, y):
        import jax.numpy as jnp

        count, parts = np.log(len(self.pools)), []
        with np.errstate(divide="ignore"):
            if self.gate > 0.0:
                logs = []
                for pool in self.pools:
                    z = pool.z(y)
                    corrector = np.asarray(_read(self.corrector, jnp.asarray(z.T))[1], np.float64).T
                    logs.append(corrector + pool.log_pdf(y) + 0.5 * z**2 + 0.5 * np.log(2 * np.pi))
                parts.append(np.log(self.gate) + logsumexp(logs, axis=0) - count)
            if self.gate < 1.0:
                parts.append(np.log(1 - self.gate) + logsumexp([pool.log_pdf(y) for pool in self.fallback], axis=0) - count)
        return parts[0] if len(parts) == 1 else np.logaddexp(*parts)

    def quantile(self, u):
        start = _interp_rows(LEVELS, self.table, np.clip(u, LEVELS[0], LEVELS[-1]))
        tolerance = 3e-7 if self.gate > 0.0 else 1e-9 + 2e-7 * max(p.w for p in self.pools)
        return _solve([*self.pools, *self.fallback], u, start, self._terms, tolerance)


class _ValuePiece:
    """The dressed probabilities of a target with few values on a block of rows."""

    def __init__(self, probabilities, values):
        self.probabilities, self.values, self.rows = probabilities, values, len(probabilities)
        self.cum = np.cumsum(probabilities, axis=1)

    def cdf(self, y):
        index = np.searchsorted(self.values, y, side="right")
        padded = np.concatenate([np.zeros((self.rows, 1)), self.cum], axis=1)
        return np.clip(np.take_along_axis(padded, index, axis=1), 0.0, 1.0)

    def log_prob(self, y):
        index = np.minimum(np.searchsorted(self.values, y), len(self.values) - 1)
        with np.errstate(divide="ignore"):
            return np.where(self.values[index] == y, np.log(np.take_along_axis(self.probabilities, index, axis=1)), -np.inf)

    def quantile(self, u):
        index = np.stack([np.searchsorted(row, level - 1e-12, side="left") for row, level in zip(self.cum, u, strict=True)])
        return self.values[np.minimum(index, len(self.values) - 1)]


class ASRDressed(BaseEstimator):
    """Cross-fitted networks and label readers, pooled, smoothed and corrected into one distribution.

    See the module docstring for the five steps. ``fit(X, y)`` takes a table and one numeric target column;
    ``predict_dist(X)`` returns a ``DressedDistribution`` with an exact ``cdf`` and ``log_prob``.

    Parameters
    ----------
    members:
        How many cross-fitted members (default 5). Each holds out one part of the training rows.
    few_values:
        A target with at most this many distinct values (default 20) is treated as a set of values: its networks
        are ordinal, and the prediction is a probability for each value. ``0`` always fits a density.
    network:
        Settings for ``ashlar.tabm_piecewise``, applied to every member's network and to the corrector, on top of
        ``max_epochs=1000`` and ``cooldown_epochs=3``. ``task``, ``random_state`` and ``early_stopping_metric``
        are set by the model.
    reader:
        Settings for each member's ``ashlar.label_reader.LabelReader``, such as ``max_steps``.
    reader_rows:
        The most training rows a member's reader reads at once (default 4096). A larger part is sampled down to
        this many, because a reader's cost grows with the square of its rows. Parts above this size are untested.
    floor_cost:
        ``None`` (default) gives every smoothed reader 1% of the wide normal. A share, such as ``0.01``, lowers the
        wide normal's weight until its cost in CRPS is at most that share of the raw pool's held-out CRPS. The
        wide normal costs about 0.00006 of the labels' spread, which is nothing on most tables and several times
        the whole CRPS on a target the columns determine almost exactly.
    random_state:
        Seeds the split, the members and the corrector.

    Fitted attributes: ``values_`` (the target's values, or ``None`` for a density), ``pool_weight_`` (the
    network's weight against the reader), ``floor_`` (the wide normal's share), ``bandwidth_factor_`` and ``fallback_factor_`` (the smoothing factors
    by log score and by CRPS), ``gate_`` (the corrected pools' weight), ``members_`` (each member's network,
    reader and held-out rows), ``corrector_``, and ``history_`` (the searched curves, the readers' steps and the
    networks' best epochs).
    """

    def __init__(self, members: int = 5, few_values: int = 20, network: dict | None = None, reader: dict | None = None,
                 reader_rows: int = 4096, floor_cost: float | None = None, random_state: int = 0):
        self.members = members
        self.few_values = few_values
        self.network = network
        self.reader = reader
        self.reader_rows = reader_rows
        self.floor_cost = floor_cost
        self.random_state = random_state

    # ----------------------------------------------------------------------------------------------------- fitting

    def _network(self, task, seed, cls=None):
        from ashlar.recipes import tabm_piecewise

        settings = {**MEMBER_SETTINGS, **(self.network or {})}
        taken = {"task", "random_state", "early_stopping_metric"} & set(settings)
        if taken:
            raise AshlarError(f"ASRDressed sets {sorted(taken)} itself; leave them out of network=.")
        if task == "flow":
            settings["early_stopping_metric"] = "crps"
        model = tabm_piecewise(task=task, random_state=int(seed), **settings)
        return model if cls is None else cls(**model.get_params())

    def _check(self, X, y, sample_weight):
        if sample_weight is not None:
            raise AshlarError("ASRDressed does not take sample weights.")
        if observed.is_observed(y):
            raise AshlarError("ASRDressed needs exactly observed labels; censored and interval labels are not supported.")
        if isinstance(self.members, bool) or int(self.members) < 2:
            raise AshlarError("members must be at least 2.")
        if hasattr(X, "to_pandas") and not hasattr(X, "iloc"):
            X = X.to_pandas()
        if not hasattr(X, "iloc"):
            X = np.asarray(X)
            if X.ndim != 2:
                raise AshlarError("X must be a table: rows by columns.")
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise AshlarError("ASRDressed fits one target column; joint targets are not supported.")
        if y.dtype.kind not in "iuf":
            raise AshlarError("ASRDressed fits a numeric target; for classes, use ASRClassifier.")
        y = y.astype(np.float64)
        if not np.isfinite(y).all():
            raise AshlarError("the target must be finite.")
        if len(y) != len(X):
            raise AshlarError("X and y have different numbers of rows.")
        if len(y) < 20 * int(self.members):
            raise AshlarError(f"ASRDressed needs at least {20 * int(self.members)} rows for {int(self.members)} members.")
        return X, y

    def _parts(self, k, X, x, context):
        """Member ``k``'s network (its distribution, or its value probabilities) and reader atoms on these rows."""
        model, reader, _ = self.members_[k]
        if self.values_ is None:
            network = model.predict_dist(X)
        else:
            proba = np.asarray(model.predict_proba(X), dtype=np.float64)
            network = np.zeros((len(proba), len(self.values_)))
            network[:, np.asarray(model.classes_, dtype=np.int64)] = proba
            network /= network.sum(axis=1, keepdims=True)
        return network, reader.atoms(x, context=context)

    def _network_atoms(self, k, X, network):
        """Member ``k``'s network as ``(atoms, masses)``: its values, or its quantiles at ``LEVELS``."""
        if self.values_ is not None:
            return np.broadcast_to(self.values_, network.shape), network
        q = np.asarray(self.members_[k][0].predict_quantiles(X, LEVELS), dtype=np.float64)
        return q, np.broadcast_to(LEVEL_MASS, q.shape)

    def _smooth(self, network, reader, factor, weight=None):
        return _Pool(network, _Smoothed(*reader, self.centre_, self.spread_, factor, self.floor_),
                     self.pool_weight_ if weight is None else weight, self.values_)

    def _pool_crps(self, pool, raw, y):
        """Each row's CRPS of a smoothed pool, read at ``LEVELS`` (a density) or over its values."""
        if self.values_ is not None:
            return _level_crps(np.broadcast_to(self.values_, pool.network.shape), pool.probabilities(), y)
        levels = np.broadcast_to(LEVELS, (len(y), len(LEVELS)))
        q = _pool_quantiles([pool], levels, mixture_quantiles(raw, raw, 1.0, LEVELS))
        return atoms_crps(q, np.broadcast_to(LEVEL_MASS, q.shape), y)

    def fit(self, X, y, sample_weight=None):
        """Fit the members, the pool weight, the smoothing, the corrector and the gate (see the module docstring)."""
        X, y = self._check(X, y, sample_weight)
        n, count, seed = len(y), int(self.members), int(self.random_state)
        self.columns_ = _Columns(X)
        x = self.columns_(X)
        values = np.unique(y)
        self.values_ = values if self.few_values and values.size <= int(self.few_values) else None
        self.centre_, self.spread_ = float(y.mean()), float(y.std() + 1e-12)
        order = np.random.default_rng(2000 + seed).permutation(n)
        cuts = [int(k / count * n) for k in range(count + 1)]
        codes = np.searchsorted(values, y)
        self.members_, curves = [], []
        for k in range(count):
            held = np.sort(order[cuts[k] : cuts[k + 1]])
            part = np.sort(np.concatenate([order[: cuts[k]], order[cuts[k + 1] :]]))
            member_seed = seed + (k + 1) % count
            if self.values_ is not None:
                # An ordinal fit needs every validation value among its training rows: a value only the held-out
                # part has moves over.
                missing = ~np.isin(y[held], y[part])
                part, held = np.sort(np.concatenate([part, held[missing]])), held[~missing]
                if not len(held):
                    raise AshlarError("a member has no held-out rows left; the table is too small to dress.")
                model = self._network("ordinal", member_seed)
                model.fit(_take(X, part), codes[part], validation=(_take(X, held), codes[held]))
            else:
                model = self._network("flow", member_seed)
                model.fit(_take(X, part), y[part], validation=(_take(X, held), y[held]))
            rng = np.random.default_rng(4000 + seed + k)
            cap = int(self.reader_rows)
            read = part if len(part) <= cap else np.sort(rng.choice(part, cap, replace=False))
            check = held if len(held) <= cap else np.sort(rng.choice(held, cap, replace=False))
            reader = LabelReader(random_state=member_seed, **(self.reader or {})).fit(x[read], y[read], validation=(x[check], y[check]))
            self.members_.append((model, reader, held))
            curve = np.zeros(len(POOL_WEIGHTS))
            for rows in self._blocks(held):
                network, atoms = self._parts(k, _take(X, rows), x[rows], "train")
                curve += len(rows) * _curve(self._network_atoms(k, _take(X, rows), network), atoms, y[rows])
            curves.append(curve / len(held))
        sizes = np.asarray([len(held) for _, _, held in self.members_], dtype=np.float64)
        oof = (sizes[:, None] * np.stack(curves)).sum(axis=0) / sizes.sum()
        self.pool_weight_ = float(POOL_WEIGHTS[int(np.argmin(oof))])
        self.floor_ = _floor_share(y, self.centre_, self.spread_, float(oof.min()), self.floor_cost)
        self.history_ = {"weight_curve": oof.tolist(), "reader_steps": [int(reader.steps_) for _, reader, _ in self.members_],
                         "best_epochs": [int(model.history_.get("best_epoch", -1)) for model, _, _ in self.members_]}
        self._pick_factors(X, x, y)
        self._fit_corrector(X, x, y)
        return self

    @staticmethod
    def _blocks(index, size=PREDICT_ROWS):
        return [index[start : start + size] for start in range(0, len(index), size)]

    def _pick_factors(self, X, x, y):
        """The bandwidth factor by held-out log score, and the fallback's by held-out CRPS."""
        seed = int(self.random_state)
        scores, crps = np.zeros(len(FACTORS)), np.zeros(len(FACTORS))
        searched = FACTORS if self.pool_weight_ < 1.0 else FACTORS[:1]  # any factor does when the reader has no weight
        total = picked = 0
        for k, (_, _, held) in enumerate(self.members_):
            pick = np.sort(np.random.default_rng(3000 + seed + k).permutation(len(held))[:CRPS_ROWS])
            for rows in self._blocks(held):
                network, atoms = self._parts(k, _take(X, rows), x[rows], "train")
                for f, factor in enumerate(FACTORS):
                    pool = self._smooth(network, atoms, factor)
                    if self.values_ is None:
                        scores[f] += pool.log_pdf(y[rows][:, None]).sum()
                    else:
                        at = np.searchsorted(self.values_, y[rows])
                        scores[f] += np.log(np.maximum(pool.probabilities()[np.arange(len(rows)), at], 1e-300)).sum()
                total += len(rows)
            rows = held[pick]
            network, atoms = self._parts(k, _take(X, rows), x[rows], "train")
            raw = _combine(self._network_atoms(k, _take(X, rows), network), atoms, self.pool_weight_)
            for f, factor in enumerate(searched):
                crps[f] += self._pool_crps(self._smooth(network, atoms, factor), raw, y[rows]).sum()
            picked += len(rows)
        self.bandwidth_factor_ = float(FACTORS[int(np.argmax(scores))])
        self.fallback_factor_ = float(searched[int(np.argmin(crps[: len(searched)]))])
        self.history_["log_score_curve"] = (scores / total).tolist()
        self.history_["crps_curve"] = (crps[: len(searched)] / picked).tolist()

    def _fit_corrector(self, X, x, y):
        """The corrector on the held-out rows' normal scores, then the gate on the last member's held-out rows."""
        values, last = self.values_, len(self.members_) - 1
        mids = None if values is None else (values[:-1] + values[1:]) / 2
        fit_rows, labels = [], []
        stop = {"table": [], "corrected": None, "fallback": [], "base": [], "pools": []}
        for k, (_, _, held) in enumerate(self.members_):
            for rows in self._blocks(held):
                X_rows = _take(X, rows)
                network, atoms = self._parts(k, X_rows, x[rows], "train")
                pool = self._smooth(network, atoms, self.bandwidth_factor_)
                if values is None:
                    label = pool.z(y[rows][:, None])[:, 0]
                else:
                    edges = pool.z(np.broadcast_to(mids, (len(rows), len(mids))).copy())
                    padded = np.concatenate([np.full((len(rows), 1), -np.inf), edges, np.full((len(rows), 1), np.inf)], axis=1)
                    index = np.searchsorted(values, y[rows])
                    label = np.stack([padded[np.arange(len(rows)), index], padded[np.arange(len(rows)), index + 1]], axis=1)
                labels.append(label)
                fit_rows.append(rows)
                if k < last:
                    continue
                other = pool if self.fallback_factor_ == self.bandwidth_factor_ else self._smooth(network, atoms, self.fallback_factor_)
                if values is None:
                    raw = _combine(self._network_atoms(k, X_rows, network), atoms, self.pool_weight_)
                    raw_q = mixture_quantiles(raw, raw, 1.0, LEVELS)
                    z_levels = ndtr(np.broadcast_to(Z_GRID, (len(rows), len(Z_GRID))))
                    start = _interp_rows(LEVELS, raw_q, np.clip(z_levels, LEVELS[0], LEVELS[-1]))
                    stop["table"].append(_pool_quantiles([pool], z_levels, start))
                    levels = np.broadcast_to(LEVELS, (len(rows), len(LEVELS)))
                    base = _pool_quantiles([pool], levels, raw_q)
                    stop["base"].append(base)
                    stop["fallback"].append(base if other is pool else _pool_quantiles([other], levels, base))
                else:
                    stop["table"].append(edges)
                    stop["fallback"].append(other.probabilities())
                stop["pools"].append((rows, pool))
        held_last = self.members_[last][2]
        n_fit = sum(len(rows) for rows in fit_rows) - len(held_last)
        fit_index, stop_index = np.concatenate(fit_rows)[:n_fit], np.concatenate(fit_rows)[n_fit:]
        label = np.concatenate(labels)
        X_fit, X_stop, y_stop = _take(X, fit_index), _take(X, stop_index), y[stop_index]
        corrector = self._network("flow", int(self.random_state), cls=_Corrector)
        table = np.concatenate(stop["table"])
        if values is None:
            corrector.stop_on = ("flow", table, None, y_stop)
            corrector.fit(X_fit, label[:n_fit], validation=(X_stop, label[n_fit:]))
        else:
            corrector.stop_on = ("values", table, values, y_stop)
            corrector.fit(X_fit, observed.interval(label[:n_fit, 0], label[:n_fit, 1]),
                          validation=(X_stop, observed.interval(label[n_fit:, 0], label[n_fit:, 1])))
        corrector.stop_on = None
        self.corrector_ = corrector
        # The gate: the corrected pool against the pool at the CRPS factor, on the last member's held-out rows
        # under that member's own pool; the first lowest score is the smaller gate.
        corrected, at = [], 0
        for (rows, pool), base in zip(stop["pools"], stop["base"] or [None] * len(stop["pools"]), strict=True):
            if values is None:
                u = ndtr(np.asarray(corrector.predict_quantiles(_take(X, rows), LEVELS), dtype=np.float64))
                corrected.append(_pool_quantiles([pool], u, _interp_rows(LEVELS, base, np.clip(u, LEVELS[0], LEVELS[-1]))))
            else:
                edges = table[at : at + len(rows)]
                corrected.append(_value_masses(np.asarray(corrector.predict_dist(_take(X, rows)).cdf(edges.T), dtype=np.float64).T))
            at += len(rows)
        corrected, fallback = np.concatenate(corrected), np.concatenate(stop["fallback"])
        if values is None:
            mass = np.broadcast_to(LEVEL_MASS, corrected.shape)
            curve = _curve((corrected, mass), (fallback, mass), y_stop)
        else:
            grid = np.broadcast_to(values, corrected.shape)
            curve = _curve((grid, corrected), (grid, fallback), y_stop)
        self.gate_ = float(POOL_WEIGHTS[int(np.argmin(curve))])
        self.history_["gate_curve"] = curve.tolist()
        self.history_["corrector_epoch"] = int(corrector.history_.get("best_epoch", -1))

    # -------------------------------------------------------------------------------------------------- prediction

    def _require_fitted(self):
        if not hasattr(self, "gate_"):
            raise AshlarError("this ASRDressed is not fitted yet; call fit first.")

    def _piece(self, X, x):
        count, values, gate = len(self.members_), self.values_, self.gate_
        parts = [self._parts(k, X, x, "all") for k in range(count)]
        same = self.fallback_factor_ == self.bandwidth_factor_
        pools = [self._smooth(network, atoms, self.bandwidth_factor_) for network, atoms in parts] if gate > 0.0 or same else None
        fallback = pools if same else [self._smooth(network, atoms, self.fallback_factor_) for network, atoms in parts]
        if values is not None:
            proba = (1 - gate) * np.mean([p.probabilities() for p in fallback], axis=0)
            if gate > 0.0:
                mids = np.broadcast_to((values[:-1] + values[1:]) / 2, (len(x), len(values) - 1)).copy()
                dist = self.corrector_.predict_dist(X)
                proba = proba + gate * np.mean(
                    [_value_masses(np.asarray(dist.cdf(p.z(mids).T), dtype=np.float64).T) for p in pools], axis=0)
            return _ValuePiece(proba / proba.sum(axis=1, keepdims=True), values)
        raw = _union([_combine(self._network_atoms(k, X, network), atoms, self.pool_weight_)
                      for k, (network, atoms) in enumerate(parts)])
        table = mixture_quantiles(raw, raw, 1.0, LEVELS)
        corrector = self.corrector_.predict_dist(X) if gate > 0.0 else None
        return _DensityPiece(pools if pools is not None else fallback, fallback, corrector, gate, table)

    def predict_dist(self, X):
        """The predictive distribution of each row of ``X``, as a ``DressedDistribution``."""
        self._require_fitted()
        if hasattr(X, "to_pandas") and not hasattr(X, "iloc"):
            X = X.to_pandas()
        if not hasattr(X, "iloc"):
            X = np.asarray(X)
        x = self.columns_(X)
        index = np.arange(len(x))
        return DressedDistribution([self._piece(_take(X, rows), x[rows]) for rows in self._blocks(index)], self.values_)

    def predict(self, X):
        """Each row's predictive mean."""
        return self.predict_dist(X).mean()

    def predict_quantiles(self, X, levels=(0.1, 0.5, 0.9)):
        """Quantiles per row, shape ``(rows, len(levels))``."""
        return self.predict_dist(X).quantile(np.asarray(levels, dtype=np.float64).reshape(-1)).T

    def log_prob(self, X, y):
        """The log density of ``y`` under ``predict_dist(X)``, per row (the log probability for few values)."""
        return self.predict_dist(X).log_prob(np.asarray(y, dtype=np.float64).reshape(-1))

    def sample(self, X, n: int = 100, seed: int = 0):
        """``n`` draws per row from ``predict_dist(X)``, shape ``(n, rows)``."""
        return self.predict_dist(X).sample(n, seed)

    def score(self, X, y, sample_weight=None):
        """R² of the predictive mean, as a scikit-learn regressor returns."""
        from sklearn.metrics import r2_score

        return float(r2_score(np.asarray(y, dtype=np.float64).reshape(-1), self.predict(X), sample_weight=sample_weight))


def dressed(**settings) -> ASRDressed:
    """A dressed model: ``ASRDressed`` with ``members``, ``few_values``, ``reader``, ``reader_rows``, ``floor_cost``
    and ``random_state`` taken from ``settings`` and every other setting passed to its networks
    (``ashlar.tabm_piecewise``), as in ``ashlar.dressed(learning_rate=1e-3)``."""
    names = ("members", "few_values", "reader", "reader_rows", "floor_cost", "random_state")
    own = {name: settings.pop(name) for name in names if name in settings}
    return ASRDressed(network=settings or None, **own)
