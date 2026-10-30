"""Composable tabular models on JAX, Equinox, and Optax.

The import name is ``ashlar``. The distribution name is ``ashlar-tabular``.
PyPI project ``ashlar`` is the unrelated microscopy stitcher. This package does
not register either name.

The names in ``__all__`` are the public surface. Helpers stay on their modules.
Forecast heads are the ``head`` argument of the two forecasters: ``mse``
(default), ``quantile``, ``gaussian``, ``mixture``, ``flow``, and ``histogram``.
"""

from ashlar._version import __version__
from ashlar.additive import AdditiveModel, DiagonalNeighborhood, GatedAdditive
from ashlar.errors import AshlarError, ShapeError, WeightError
from ashlar.estimator import TabularEstimator
from ashlar.forecast import FullHistoryForecaster, TimeSeriesForecaster
from ashlar.node import ObliviousEnsemble
from ashlar.predict import MeanParameterNormal, NormalMixture
from ashlar.priors import BartPriorEnsemble, GradientSparseNet, VariationalPairwise
from ashlar.recipes import (
    CLASSIFIER_RECIPE,
    DEFAULT_RECIPE,
    ORDINAL_RECIPE,
    TABM_PAPER,
    TABM_PIECEWISE,
    TREE_RECIPE,
    classifier,
    ordinal,
    outcome_tree,
    residual_mlp,
    tabular,
    tabm,
    tabm_piecewise,
)
from ashlar.tree import OutcomeTree
from ashlar.rounds import HistogramBooster, LearnedNeighborhood, LinearLeafBooster
from ashlar.search import DEFAULT_SEARCH

__all__ = [
    "AdditiveModel",
    "AshlarError",
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
    "MeanParameterNormal",
    "NormalMixture",
    "ORDINAL_RECIPE",
    "ObliviousEnsemble",
    "OutcomeTree",
    "ShapeError",
    "TABM_PAPER",
    "TABM_PIECEWISE",
    "TREE_RECIPE",
    "TabularEstimator",
    "TimeSeriesForecaster",
    "VariationalPairwise",
    "WeightError",
    "__version__",
    "classifier",
    "ordinal",
    "outcome_tree",
    "residual_mlp",
    "tabular",
    "tabm",
    "tabm_piecewise",
]
