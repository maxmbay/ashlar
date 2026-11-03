"""A density model of the numeric features, for inputs that are missing, noisy, or unusual.

A fitted model fills a missing numeric cell with the training median inside
JAX, and the predictive distribution is then as narrow as if the value were
known. ``InputModel`` is a Gaussian copula over the numeric features, fitted
on the training partition at ``fit``:

- each column is mapped to a standard normal score through its empirical
  quantiles (``KNOTS`` of them), piecewise linear and extended linearly past
  the ends;
- the scores share one correlation matrix, estimated from pairwise-complete
  rows and shrunk slightly towards the identity.

It gives three things:

- ``draw``: completions of a row's missing cells from their distribution
  given the row's observed cells (multiple imputation), and draws of a
  noisy cell around its measured value with the standard deviation the user
  supplies (errors in variables, with a flat prior on the true value);
- ``log_density``: the row's ``log p(x)`` over its observed numeric cells,
  in the units of ``X``, a score for shifted and out-of-distribution rows;
- the JSON form that ``save`` writes.

``ASRModel.predict_dist(X, input_draws=K)`` and
``ASRModel.with_uncertain_inputs`` run the network on ``K`` completions of
each row and return the equal-weight mixture over them and the members, so
every interval, quantile, and probability widens with what is not known
about the row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special

from ashlar.errors import AshlarError

KNOTS = 129
SHRINKAGE = 0.01
_MIN_SLOPE_FRACTION = 1e-3


@dataclass
class InputModel:
    """Gaussian copula over ``n`` numeric columns: per-column knots and one correlation matrix."""

    knots: np.ndarray  # (n, KNOTS) values, nondecreasing per column
    scores: np.ndarray  # (n, KNOTS) normal scores at the knots, increasing per column
    correlation: np.ndarray  # (n, n)

    @property
    def n_columns(self) -> int:
        return int(self.correlation.shape[0])

    # -------------------------------------------------------------- fit / json

    @classmethod
    def fit(cls, numeric_raw) -> InputModel:
        values = np.asarray(numeric_raw, dtype=np.float64)
        if values.ndim != 2:
            raise AshlarError("the input model is fitted on a (rows, numeric columns) array.")
        n = values.shape[1]
        levels = (np.arange(KNOTS) + 0.5) / KNOTS
        knots = np.zeros((n, KNOTS))
        scores = np.zeros((n, KNOTS))
        for j in range(n):
            column = values[:, j]
            finite = column[np.isfinite(column)]
            if finite.size == 0:
                finite = np.zeros(1)
            knots[j], scores[j] = _column_map(np.quantile(finite, levels), special.ndtri(levels))
        model = cls(knots=knots, scores=scores, correlation=np.eye(n))
        z = model.to_scores(values)
        model.correlation = _correlation(z)
        return model

    def to_json(self) -> dict:
        return {"knots": self.knots.tolist(), "scores": self.scores.tolist(), "correlation": self.correlation.tolist()}

    @classmethod
    def from_json(cls, payload) -> InputModel | None:
        if payload is None:
            return None
        return cls(
            knots=np.asarray(payload["knots"], dtype=np.float64).reshape(len(payload["correlation"]), -1),
            scores=np.asarray(payload["scores"], dtype=np.float64).reshape(len(payload["correlation"]), -1),
            correlation=np.asarray(payload["correlation"], dtype=np.float64),
        )

    # -------------------------------------------------------------- maps

    def to_scores(self, values) -> np.ndarray:
        """Normal scores of ``values`` ``(rows, n)``; NaN stays NaN."""
        values = np.asarray(values, dtype=np.float64)
        out = np.full(values.shape, np.nan)
        for j in range(self.n_columns):
            column = values[:, j]
            finite = np.isfinite(column)
            out[finite, j] = _extrapolate(column[finite], self.knots[j], self.scores[j])
        return out

    def from_scores(self, z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        out = np.empty(z.shape)
        for j in range(self.n_columns):
            out[..., j] = _extrapolate(z[..., j], self.scores[j], self.knots[j])
        return out

    def _log_slopes(self, values) -> np.ndarray:
        """``log dz/dx`` per finite cell, 0 where missing."""
        out = np.zeros(values.shape)
        for j in range(self.n_columns):
            column = values[:, j]
            finite = np.isfinite(column)
            out[finite, j] = np.log(_slope(column[finite], self.knots[j], self.scores[j]))
        return out

    # -------------------------------------------------------------- uses

    def log_density(self, numeric_raw) -> np.ndarray:
        """``log p(x)`` of each row's observed numeric cells, per row, in the units of ``X``.

        A row's missing cells are integrated out (the copula's marginal over
        its observed cells). A row with no observed numeric cell scores 0.
        """
        values = np.asarray(numeric_raw, dtype=np.float64)
        z = self.to_scores(values)
        observed = np.isfinite(z)
        out = self._log_slopes(values).sum(axis=1)
        for pattern, rows in _patterns(observed):
            index = np.flatnonzero(pattern)
            if index.size == 0:
                continue
            cov = self.correlation[np.ix_(index, index)]
            _sign, logdet = np.linalg.slogdet(cov)
            solved = np.linalg.solve(cov, z[np.ix_(rows, index)].T).T
            quad = np.sum(z[np.ix_(rows, index)] * solved, axis=1)
            out[rows] += -0.5 * (quad + logdet + index.size * np.log(2 * np.pi))
        return out

    def draw(self, numeric_raw, draws: int, rng, noise=None) -> np.ndarray:
        """``draws`` completions of every row, ``(draws, rows, n)``, in the units of ``X``.

        Missing cells are drawn from the copula given the row's observed
        cells; observed cells are kept. ``noise`` is ``(rows, n)`` standard
        deviations (0 where a cell is exact): such a cell is drawn as
        ``value + noise * N(0, 1)`` and its measured value conditions the
        missing cells. ``draws`` of a row without missing or noisy cells are
        all the row itself.
        """
        values = np.asarray(numeric_raw, dtype=np.float64)
        rows, n = values.shape
        out = np.broadcast_to(values, (int(draws), rows, n)).copy()
        z = self.to_scores(values)
        missing = ~np.isfinite(z)
        for pattern, members in _patterns(missing):
            lost = np.flatnonzero(pattern)
            if lost.size == 0:
                continue
            kept = np.flatnonzero(~pattern)
            cov = self.correlation
            if kept.size:
                gain = np.linalg.solve(cov[np.ix_(kept, kept)], cov[np.ix_(kept, lost)]).T  # (lost, kept)
                mean = z[np.ix_(members, kept)] @ gain.T
                conditional = cov[np.ix_(lost, lost)] - gain @ cov[np.ix_(kept, lost)]
            else:
                mean = np.zeros((members.size, lost.size))
                conditional = cov[np.ix_(lost, lost)]
            root = _psd_root(conditional)
            eps = rng.standard_normal((int(draws), members.size, lost.size))
            drawn = mean[None] + eps @ root.T
            completed = self.from_scores(_scatter(drawn, lost, n))
            out[:, members[:, None], lost[None, :]] = completed[..., lost]
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float64)
            noisy = np.isfinite(values) & (noise > 0)
            if noisy.any():
                eps = rng.standard_normal((int(draws), rows, n))
                out = np.where(noisy[None], values[None] + noise[None] * eps, out)
        return out


def _column_map(knots, scores):
    """Unique knot values, each with the mean score of its ties, so the map is a function."""
    unique, inverse = np.unique(knots, return_inverse=True)
    mean_scores = np.bincount(inverse, weights=scores) / np.bincount(inverse)
    if unique.size == 1:
        unique = np.array([unique[0] - 0.5, unique[0] + 0.5])
        mean_scores = np.array([-0.5, 0.5])
    # Pad back to a fixed width so every column stores KNOTS pairs; the repeats are harmless to interp.
    pad = KNOTS - unique.size
    return np.concatenate([unique, np.full(pad, unique[-1])]), np.concatenate([mean_scores, np.full(pad, mean_scores[-1])])


def _distinct(xs, ys):
    keep = np.concatenate([[True], np.diff(xs) > 0])
    return xs[keep], ys[keep]


def _extrapolate(values, xs, ys):
    """Piecewise-linear map through ``(xs, ys)``, continued past the ends with the end slopes."""
    xs, ys = _distinct(xs, ys)
    out = np.interp(values, xs, ys)
    if xs.size >= 2:
        low = values < xs[0]
        high = values > xs[-1]
        out[low] = ys[0] + (values[low] - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
        out[high] = ys[-1] + (values[high] - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    return out


def _slope(values, xs, ys):
    """``dy/dx`` of the map, floored so a tie in the knots (a discrete value) stays finite."""
    xs, ys = _distinct(xs, ys)
    slopes = np.diff(ys) / np.diff(xs)
    floor = _MIN_SLOPE_FRACTION * (ys[-1] - ys[0]) / max(xs[-1] - xs[0], 1e-12)
    segment = np.clip(np.searchsorted(xs, values, side="right") - 1, 0, slopes.size - 1)
    return np.maximum(slopes[segment], floor)


def _correlation(z) -> np.ndarray:
    n = z.shape[1]
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            both = np.isfinite(z[:, i]) & np.isfinite(z[:, j])
            if both.sum() > 2:
                a, b = z[both, i], z[both, j]
                denominator = a.std() * b.std()
                if denominator > 0:
                    corr[i, j] = corr[j, i] = float(np.mean((a - a.mean()) * (b - b.mean())) / denominator)
    corr = (1.0 - SHRINKAGE) * corr + SHRINKAGE * np.eye(n)
    eigenvalues, vectors = np.linalg.eigh(corr)
    corr = (vectors * np.maximum(eigenvalues, 1e-6)) @ vectors.T
    scale = np.sqrt(np.diag(corr))
    return corr / scale[:, None] / scale[None, :]


def _psd_root(cov) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    return vectors * np.sqrt(np.maximum(eigenvalues, 0.0))


def _patterns(mask):
    """``(pattern, rows)`` for each distinct row pattern of a boolean ``(rows, n)`` mask."""
    if mask.shape[0] == 0:
        return []
    unique, inverse = np.unique(mask, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    return [(unique[k], np.flatnonzero(inverse == k)) for k in range(unique.shape[0])]


def _scatter(drawn, lost, n):
    out = np.zeros((*drawn.shape[:-1], n))
    out[..., lost] = drawn
    return out
