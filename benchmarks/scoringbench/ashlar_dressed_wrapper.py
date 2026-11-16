"""Ashlar's dressed model (``ashlar.dressed()``), read through its quantiles."""

from __future__ import annotations

import numpy as np

from .base import DistributionPrediction, ProbabilisticWrapper
from .quantile_based import quantiles_to_distribution

# Quantile levels read from each row's predictive distribution: a fine grid through the body, plus the far tails.
LEVELS = np.concatenate([[0.001, 0.0025], np.linspace(0.005, 0.995, 199), [0.9975, 0.999]])
# The mass each quantile carries when the 203 of them stand in for a density; the mean is their weighted sum.
LEVEL_MASS = np.diff(np.concatenate([[0.0], LEVELS]))
LEVEL_MASS[-1] += 1.0 - LEVELS[-1]


class AshlarDressedWrapper(ProbabilisticWrapper):
    """Ashlar's dressed model: networks and label readers trained from scratch, pooled into one distribution.

    Five cross-fitted pairs of a network (a TabM-style ensemble with a flow head) and a label reader, which
    reads a row's distribution from the training labels. The pairs are pooled, the readers' points are smoothed
    into a density, and one more network corrects the pool; the correction follows condensité's PIT boosting.
    Nothing is pretrained and nothing is tuned: the entry runs ``ashlar.dressed()`` with its defaults.

    The prediction has an exact CDF, and its quantiles at 203 levels go to ScoringBench. A target with at most
    20 distinct training values gets a probability for each value.

    Code: https://github.com/maxmbay/ashlar (Apache-2.0), ``pip install ashlar-tabular``.
    Guide: https://github.com/maxmbay/ashlar/blob/main/docs/dressed.md

    Keyword arguments go to ``ashlar.dressed``.
    """

    def __init__(self, **settings):
        self.settings = settings
        self.model_ = None
        self._last = None

    def fit(self, X, y) -> "AshlarDressedWrapper":
        import ashlar

        y = np.asarray(y, dtype=np.float64).reshape(-1)
        finite = np.isfinite(y)
        if not finite.all():
            X = X.iloc[finite] if hasattr(X, "iloc") else np.asarray(X)[finite]
            y = y[finite]
        self._set_train_range(y)
        self._last = None
        self.model_ = ashlar.dressed(**self.settings).fit(X, y)
        return self

    def _read(self, X):
        """Each row's quantiles at LEVELS and its mean, kept for a second call on the same rows."""
        if self._last is not None and self._last[0] is X:
            return self._last[1], self._last[2]
        dist = self.model_.predict_dist(X)
        q = dist.quantile(LEVELS).T
        # A target with few values has an exact mean; a density's is read from its quantiles, as the model reads it.
        mean = q @ LEVEL_MASS if dist.values is None else dist.probabilities() @ dist.values
        self._last = (X, q, mean)
        return q, mean

    def predict(self, X) -> np.ndarray:
        return self._read(X)[1]

    def predict_distribution(self, X) -> DistributionPrediction:
        q, mean = self._read(X)
        return quantiles_to_distribution(
            q, LEVELS, mean=mean, y_range=self._y_train_range, train_range=self._y_train_range
        )
