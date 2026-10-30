"""Versioned training recipes.

``TABM_PAPER`` records the architecture defaults of ``tabm`` 0.0.3
``TabM.make`` without numerical embeddings, plus the PyTabKit 1.7.3 TabM
training defaults that are visible on its constructor: learning rate ``2e-3``,
weight decay ``0``, batch size ``256``, and patience ``16``. It does not
reproduce a published dataset score. Quantile preprocessing, piecewise-linear
embeddings, and gradient clipping from the paper experiments are omitted.

When numerical embeddings are enabled, ``TabM.make`` switches to two blocks
and ``start_scaling_init='normal'``. ``tabm(num_embeddings='linear_relu')``
does the same. An explicit backbone override still wins, so the helper does
not silently replace a value the caller set.
"""

from __future__ import annotations

import warnings
from copy import deepcopy

from ashlar.estimator import ASRClassifier, ASRModel, ASROrdinalClassifier, ASROutcomeTree, ASRRegressor

# Constructor defaults of ``ASRModel``. ``fit(X, y)`` with no other
# arguments uses this path: residual MLP, median impute, robust scale with a
# smooth clip, a train-only z-score of regression labels, learned embeddings
# for inferred categoricals, no set columns, and no hyperparameter search.
# ``categorical_embeddings=[]`` restores one-hot. ``numeric_scaling='standard'``
# restores mean and standard deviation. ``standardize_target=False`` leaves
# regression labels on their original scale. Classification and ordinal keep
# that flag off.
DEFAULT_RECIPE = {
    "task": "regression",
    "backbone": "residual_mlp",
    "backbone_kwargs": None,
    "num_embeddings": "piecewise",
    "num_embedding_dim": 16,
    "categorical": None,
    "categorical_embeddings": None,
    "set_features": None,
    "categorical_embedding_dim": 8,
    "set_max_length": 8,
    "validation_size": 0.2,
    "split": "iid",
    "batch_size": "auto",
    "max_epochs": "auto",
    "patience": 16,
    "learning_rate": 2e-3,
    "learning_rate_schedule": "cosine",
    "standardize_target": True,
    "share_training_batches": True,
    "bottleneck": None,
    "histogram_variance": None,
    "soft_stages": None,
    "weight_decay": 1e-4,
    "feature_dropout": 0.0,
    "numeric_scaling": "robust",
    "prediction_policy": None,
    "random_state": 0,
}

TABM_PAPER = {
    "task": "regression",
    "backbone": "tabm",
    "backbone_kwargs": {
        "n_blocks": 3,
        "d_block": 512,
        "dropout": 0.1,
        "k": 32,
        "start_scaling_init": "random-signs",
    },
    "num_embeddings": None,
    "num_embedding_dim": 32,
    "learning_rate": 2e-3,
    "weight_decay": 0.0,
    "batch_size": 256,
    "patience": 16,
    "max_epochs": 200,
    "prediction_policy": "mean_members",
}

# The most accurate recipe in the ablations (examples/ablation_gbm_parity.py,
# config ``tabm_ple_quantile``): a 32-member TabM on piecewise-linear numeric
# embeddings of quantile-normalized features. About 5 to 10 times the cost of
# the default on CPU.
TABM_PIECEWISE = {
    "task": "regression",
    "backbone": "tabm",
    "backbone_kwargs": {
        "n_blocks": 2,
        "d_block": 256,
        "dropout": 0.1,
        "k": 32,
        "start_scaling_init": "normal",
    },
    "num_embeddings": "piecewise",
    "num_embedding_dim": 16,
    "numeric_scaling": "quantile",
    "learning_rate": 2e-3,
    "learning_rate_schedule": "constant",
    "weight_decay": 0.0,
    "batch_size": "auto",
    "max_epochs": "auto",
    "patience": 16,
}

TABM_PROBE = {
    "task": "regression",
    "backbone": "tabm",
    "backbone_kwargs": {
        "n_blocks": 2,
        "d_block": 16,
        "dropout": 0.0,
        "k": 4,
        "start_scaling_init": "random-signs",
    },
    "learning_rate": 2e-3,
    "weight_decay": 0.0,
    "batch_size": 16,
    "patience": 4,
    "max_epochs": 8,
    "validation_size": 0.25,
}

RESIDUAL_PROBE = {
    "task": "regression",
    "backbone": "residual_mlp",
    "backbone_kwargs": {"n_blocks": 1, "d_block": 16, "dropout": 0.0},
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 16,
    "patience": 4,
    "max_epochs": 8,
    "validation_size": 0.25,
}


def _merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def tabm(**overrides) -> ASRModel:
    """Paper-shaped TabM recipe, with caller overrides applied on top.

    ``task="classification"`` is legal. It keeps one shared minibatch for
    every member and does not z-score labels. The prediction policy becomes
    ``mean_probabilities`` unless you pass one. ``numeric_scaling`` stays the
    robust default; pass ``"quantile_tabr"`` when you want that map.
    ``share_training_batches=False`` stays regression-only.
    """
    params = deepcopy(TABM_PAPER)
    if overrides.get("num_embeddings"):
        explicit = overrides.get("backbone_kwargs") or {}
        if "n_blocks" not in explicit:
            params["backbone_kwargs"]["n_blocks"] = 2
        if "start_scaling_init" not in explicit:
            params["backbone_kwargs"]["start_scaling_init"] = "normal"
    task = overrides.get("task", params.get("task", "regression"))
    if task == "classification" and "prediction_policy" not in overrides:
        params["prediction_policy"] = "mean_probabilities"
    if task == "classification" and "standardize_target" not in overrides:
        params["standardize_target"] = False
    params = _merge(params, overrides)
    return ASRModel(**params)


CLASSIFIER_RECIPE = {
    **DEFAULT_RECIPE,
    "task": "classification",
    "prediction_policy": "mean_probabilities",
    "standardize_target": False,
}


ORDINAL_RECIPE = {
    **DEFAULT_RECIPE,
    "task": "ordinal",
    "prediction_policy": "mean_probabilities",
    "standardize_target": False,
}


def tabm_piecewise(**overrides) -> ASRModel:
    """TabM on piecewise-linear numeric embeddings: the accuracy recipe.

    Works for every task. Classification and ordinal turn the label z-score
    off and average member probabilities; a distribution task predicts the
    member mixture.
    """
    params = deepcopy(TABM_PIECEWISE)
    task = overrides.get("task", "regression")
    if task in {"classification", "ordinal"}:
        params.setdefault("prediction_policy", "mean_probabilities")
        if "standardize_target" not in overrides:
            params["standardize_target"] = False
    return ASRModel(**_merge(params, overrides))


def _renamed(old: str, new: str):
    warnings.warn(f"{old} is now {new}.", DeprecationWarning, stacklevel=3)


def ordinal(**overrides) -> ASROrdinalClassifier:
    """Deprecated: use ``ASROrdinalClassifier``."""
    _renamed("ordinal()", "ASROrdinalClassifier()")
    return ASROrdinalClassifier(**_merge(deepcopy(ORDINAL_RECIPE), overrides))


def classifier(**overrides) -> ASRClassifier:
    """Deprecated: use ``ASRClassifier``."""
    _renamed("classifier()", "ASRClassifier()")
    return ASRClassifier(**_merge(deepcopy(CLASSIFIER_RECIPE), overrides))


TREE_RECIPE = {
    **DEFAULT_RECIPE,
    "task": "tree",
    "prediction_policy": "mean_probabilities",
    "standardize_target": False,
    "head_kwargs": {"edge_prior": "empirical"},
}


def outcome_tree(tree, **overrides) -> ASROutcomeTree:
    """Deprecated: use ``ASROutcomeTree(tree)``."""
    _renamed("outcome_tree()", "ASROutcomeTree()")
    return ASROutcomeTree(**_merge(deepcopy(TREE_RECIPE), {"tree": tree, **overrides}))


def tabular(**overrides) -> ASRRegressor:
    """Deprecated: use ``ASRRegressor``, or ``ASRModel`` for a non-regression task."""
    _renamed("tabular()", "ASRRegressor()")
    params = _merge(deepcopy(DEFAULT_RECIPE), overrides)
    model = ASRRegressor if params["task"] in ASRRegressor._tasks else ASRModel
    return model(**params)


def residual_mlp(**overrides) -> ASRModel:
    """Small residual MLP. This is the interchangeable baseline, not a paper recipe."""
    params = _merge(deepcopy(RESIDUAL_PROBE), overrides)
    params.setdefault("max_epochs", 50)
    return ASRModel(**params)
