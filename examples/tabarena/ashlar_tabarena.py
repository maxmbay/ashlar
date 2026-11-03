"""TabArena (AutoGluon ``AbstractModel``) wrappers for Ashlar.

Two entries, one per recipe, so their configs get distinct names on the
leaderboard:

* ``AshlarModel`` (``ASR``): ``ASRRegressor`` / ``ASRClassifier`` defaults,
  a residual MLP on piecewise-linear numeric embeddings.
* ``AshlarTabMModel`` (``ASRTABM``): ``ashlar.tabm_piecewise``, the
  accuracy recipe (TabM ensemble on piecewise embeddings, quantile scaling).

This module must be importable (not ``__main__``) so Ray workers can unpickle
the classes when the runner is not in debug mode.

Known gaps against the TabArena contract, fine for a local Lite test but to
close before a submission: ``time_limit`` is not enforced (Ashlar has no
wall-clock budget, only ``max_epochs`` and ``patience``), and ``num_cpus`` is
not wired (JAX on CPU picks its thread count at import).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel

# Flat hyperparameters the search space samples; they go into backbone_kwargs.
_BACKBONE_KEYS = ("n_blocks", "d_block", "dropout", "k")


class AshlarModel(AbstractModel):
    """Ashlar: distributional neural networks for tabular data, in JAX.

    Codebase: https://github.com/maxmbay/ashlar (PyPI: ashlar-tabular)
    License: Apache-2.0
    """

    ag_key = "ASR"
    ag_name = "Ashlar"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    recipe = "default"

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        # Ashlar embeds non-numeric columns. AutoGluon hands categoricals as
        # pandas ``category``, whose integer categories would read as numeric,
        # so pass them as strings. Missing becomes its own "nan" level.
        for col in X.columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == bool:
                X[col] = X[col].astype(str).astype(object)
        return X

    def _build(self, params: dict):
        from ashlar import ASRClassifier, ASRRegressor, tabm_piecewise

        params = dict(params)
        backbone = {key: params.pop(key) for key in _BACKBONE_KEYS if key in params}
        classification = self.problem_type in ("binary", "multiclass")
        if self.recipe == "tabm_piecewise":
            if backbone:
                base = {"n_blocks": 2, "d_block": 256, "dropout": 0.1, "k": 32, "start_scaling_init": "normal"}
                params["backbone_kwargs"] = {**base, **backbone}
            return tabm_piecewise(task="classification" if classification else "regression", **params)
        if backbone:
            backbone.pop("k", None)
            params["backbone_kwargs"] = {"n_blocks": 3, "d_block": 256, "dropout": 0.1, **backbone}
        return ASRClassifier(**params) if classification else ASRRegressor(**params)

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        **kwargs,
    ) -> None:
        X = self.preprocess(X, is_train=True)
        y = np.asarray(y, dtype=np.float32 if self.problem_type == "regression" else np.int64)
        validation = None
        if X_val is not None and y_val is not None:
            X_val = self.preprocess(X_val)
            y_val = np.asarray(y_val, dtype=y.dtype)
            validation = (X_val, y_val)
        params = self._get_model_params()
        if not any(pd.api.types.is_numeric_dtype(X[c]) and X[c].dtype != bool for c in X.columns):
            # Ashlar raises on numeric embeddings with no numeric columns.
            params["num_embeddings"] = None
            # Quantile scaling (tabm_piecewise) also needs a numeric column.
            params["numeric_scaling"] = "robust"
        self.model = self._build(params)
        self.model.fit(X, y, validation=validation)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        if self.problem_type == "regression":
            return np.asarray(self.model.predict(X), dtype=np.float32)
        proba = np.asarray(self.model.predict_proba(X), dtype=np.float32)
        # AutoGluon labels are 0..K-1; align columns to them in case a class
        # was missing from a training fold.
        classes = list(np.asarray(self.model.classes_).astype(int))
        full = np.zeros((len(X), self.num_classes), dtype=np.float32)
        full[:, classes] = proba
        return self._convert_proba_to_unified_form(full)

    def _set_default_params(self) -> None:
        self._set_default_param_value("random_state", 0)

    def _get_default_auxiliary_params(self) -> dict:
        params = super()._get_default_auxiliary_params()
        params.update({"valid_raw_types": ["int", "float", "category", "bool"]})
        return params

    def _more_tags(self) -> dict:
        return {"can_refit_full": False}

    @classmethod
    def search_space(cls) -> dict:
        from autogluon.common.space import Categorical, Int, Real

        return {
            "learning_rate": Real(3e-4, 5e-3, log=True),
            "weight_decay": Categorical(0.0, 1e-5, 1e-4, 1e-3),
            "n_blocks": Int(1, 4),
            "d_block": Categorical(128, 256, 512),
            "dropout": Real(0.0, 0.4),
            "num_embedding_bins": Int(8, 96),
            "num_embedding_dim": Categorical(8, 16, 24),
        }

    @classmethod
    def config_generator(cls):
        from tabarena.utils.config_utils import ConfigGenerator

        return ConfigGenerator(model_cls=cls, manual_configs=[{}], search_space=cls.search_space())


class AshlarTabMModel(AshlarModel):
    """Ashlar's ``tabm_piecewise`` recipe."""

    ag_key = "ASRTABM"
    ag_name = "AshlarTabM"
    recipe = "tabm_piecewise"
