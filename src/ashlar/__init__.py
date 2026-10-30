"""Composable tabular models on JAX, Equinox, and Optax.

The import name is ``ashlar``. The distribution name is ``ashlar-tabular``.

The names in ``__all__`` are the public surface. Helpers stay on their modules.
Every model is ``ASR`` plus what it does, after LightGBM's ``LGBMRegressor``
and ``LGBMClassifier``: ``ASRRegressor``, ``ASRClassifier``,
``ASROrdinalClassifier``, ``ASROutcomeTree``, ``ASRForecaster``, and
``ASRHistoryForecaster``, all on the generic ``ASRModel``. The old names
still import, with a ``DeprecationWarning``.
Forecast heads are the ``head`` argument of the two forecasters: ``mse``
(default), ``quantile``, ``gaussian``, ``mixture``, ``flow``, ``histogram``, and ``student_t``.
"""

from ashlar._compat import renamed
from ashlar._version import __version__
from ashlar.additive import ASRAdditive, ASRDiagonalNeighborhood, ASRGatedAdditive
from ashlar.errors import AshlarError, ShapeError, WeightError
from ashlar.estimator import ASRClassifier, ASRModel, ASROrdinalClassifier, ASROutcomeTree, ASRRegressor
from ashlar.forecast import ASRForecaster, ASRHistoryForecaster
from ashlar.node import ASRObliviousEnsemble
from ashlar.predict import MeanParameterNormal, NormalMixture
from ashlar.priors import ASRBartPriorEnsemble, ASRGradientSparseNet, ASRVariationalPairwise
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
from ashlar.rounds import ASRHistogramBooster, ASRLearnedNeighborhood, ASRLinearLeafBooster
from ashlar.search import DEFAULT_SEARCH
from ashlar.tree import OutcomeTree

__all__ = [
    "ASRAdditive",
    "ASRBartPriorEnsemble",
    "ASRClassifier",
    "ASRDiagonalNeighborhood",
    "ASRForecaster",
    "ASRGatedAdditive",
    "ASRGradientSparseNet",
    "ASRHistogramBooster",
    "ASRHistoryForecaster",
    "ASRLearnedNeighborhood",
    "ASRLinearLeafBooster",
    "ASRModel",
    "ASRObliviousEnsemble",
    "ASROrdinalClassifier",
    "ASROutcomeTree",
    "ASRRegressor",
    "ASRVariationalPairwise",
    "AshlarError",
    "CLASSIFIER_RECIPE",
    "DEFAULT_RECIPE",
    "DEFAULT_SEARCH",
    "MeanParameterNormal",
    "NormalMixture",
    "ORDINAL_RECIPE",
    "OutcomeTree",
    "ShapeError",
    "TABM_PAPER",
    "TABM_PIECEWISE",
    "TREE_RECIPE",
    "WeightError",
    "__version__",
    "residual_mlp",
    "tabm",
    "tabm_piecewise",
]

# ``tabular``, ``classifier``, ``ordinal``, and ``outcome_tree`` still import
# from here and warn when called.
__getattr__ = renamed(
    __name__,
    globals(),
    {
        "TabularEstimator": "ASRModel",
        "TimeSeriesForecaster": "ASRForecaster",
        "FullHistoryForecaster": "ASRHistoryForecaster",
        "AdditiveModel": "ASRAdditive",
        "GatedAdditive": "ASRGatedAdditive",
        "DiagonalNeighborhood": "ASRDiagonalNeighborhood",
        "ObliviousEnsemble": "ASRObliviousEnsemble",
        "HistogramBooster": "ASRHistogramBooster",
        "LinearLeafBooster": "ASRLinearLeafBooster",
        "LearnedNeighborhood": "ASRLearnedNeighborhood",
        "BartPriorEnsemble": "ASRBartPriorEnsemble",
        "VariationalPairwise": "ASRVariationalPairwise",
        "GradientSparseNet": "ASRGradientSparseNet",
    },
)
