"""Composable tabular models on JAX, Equinox, and Optax.

The import name is ``ashlar``. The distribution name is ``ashlar-tabular``.
PyPI project ``ashlar`` is the unrelated microscopy stitcher. This package does
not register either name.
"""

from ashlar._version import __version__
from ashlar.additive import AdditiveModel, DiagonalNeighborhood, GatedAdditive
from ashlar.estimator import TabularEstimator
from ashlar.forecast import FullHistoryForecaster, TimeSeriesForecaster
from ashlar.node import ObliviousEnsemble
from ashlar.predict import MeanParameterNormal, NormalMixture
from ashlar.priors import BartPriorEnsemble, GradientSparseNet, VariationalPairwise
from ashlar.rounds import HistogramBooster, LearnedNeighborhood, LinearLeafBooster
from ashlar.recipes import CLASSIFIER_RECIPE, DEFAULT_RECIPE, ORDINAL_RECIPE, TABM_PAPER, classifier, ordinal, residual_mlp, tabular, tabm
from ashlar.search import DEFAULT_SEARCH

__all__ = [
    "AdditiveModel",
    "BartPriorEnsemble",
    "CLASSIFIER_RECIPE",
    "DEFAULT_RECIPE",
    "DEFAULT_SEARCH",
    "DiagonalNeighborhood",
    "FullHistoryForecaster",
    "GatedAdditive",
    "GradientSparseNet",
    "HistogramBooster",
    "LearnedNeighborhood",
    "LinearLeafBooster",
    "ORDINAL_RECIPE",
    "MeanParameterNormal",
    "ObliviousEnsemble",
    "NormalMixture",
    "TABM_PAPER",
    "TabularEstimator",
    "TimeSeriesForecaster",
    "classifier",
    "ordinal",
    "residual_mlp",
    "tabular",
    "tabm",
    "VariationalPairwise",
    "__version__",
]
