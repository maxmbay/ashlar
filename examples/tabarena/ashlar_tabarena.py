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

def _tabm_batch_size(n_train: int) -> int:
    """TabArena TabM's batch size rule (tabm_utils.get_tabm_auto_batch_size)."""
    for limit, size in ((2_800, 32), (4_500, 64), (6_400, 128), (32_000, 256)):
        if n_train < limit:
            return size
    return 512


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
    #: Defaults applied under the sampled hyperparameters.
    defaults: dict = {}
    #: TabArena TabM's batch size by training rows (32 below 2,800 up to 512 above 32k), instead of Ashlar's "auto".
    batch_rule: bool = False
    #: Pool categorical columns with at least this many training levels as random effects (intercept + embedding).
    pool_min_levels: int | None = None
    #: Early-stop binary tasks on validation ROC AUC, the metric TabArena scores them on.
    stop_on_auc: bool = False

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

        params = {**self.defaults, **params}
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
        if self.batch_rule and "batch_size" not in params:
            params["batch_size"] = _tabm_batch_size(len(X))
        if self.stop_on_auc and self.problem_type == "binary":
            params.setdefault("early_stopping_metric", "roc_auc")
        if self.pool_min_levels is not None and "random_effects" not in params:
            pooled = {
                col: ["intercept", "embedding"]
                for col in X.columns
                if X[col].dtype == object and X[col].nunique(dropna=True) >= self.pool_min_levels
            }
            if pooled:
                params["random_effects"] = pooled
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


# Train until early stopping (patience 16, best epoch restored) instead of the
# ``max_epochs="auto"`` budget of 20-100 epochs. Constant learning rate, since
# the cosine schedule's warmup and decay are spread over ``max_epochs``.
_RIP = {"max_epochs": 2000, "learning_rate_schedule": "constant"}


class AshlarRipModel(AshlarModel):
    """Ashlar defaults, trained until early stopping."""

    ag_key = "ASRRIP"
    ag_name = "AshlarRip"
    defaults = _RIP


class AshlarTabMRipModel(AshlarTabMModel):
    """``tabm_piecewise``, trained until early stopping."""

    ag_key = "ASRTABMRIP"
    ag_name = "AshlarTabMRip"
    defaults = _RIP


# Candidate defaults, all trained until early stopping. ``n_blocks``,
# ``d_block``, ``dropout`` and ``k`` go into backbone_kwargs (see _build).
SWEEP = {
    "base": ("default", {}),
    "deep": ("default", {"n_blocks": 5}),
    "wide": ("default", {"d_block": 512, "dropout": 0.15}),
    "bottleneck": ("default", {"bottleneck": True}),
    "lr1e3": ("default", {"learning_rate": 1e-3}),
    "batch128": ("default", {"batch_size": 128}),
    "regularized": ("default", {"dropout": 0.25, "weight_decay": 1e-3}),
    "tabm_wide": ("tabm_piecewise", {"n_blocks": 3, "d_block": 512}),
}
SWEEP_MODELS = {}
for _name, (_recipe, _overrides) in SWEEP.items():
    _key = "ASRS" + _name.upper().replace("_", "")
    _cls = type(
        f"AshlarSweep_{_name}",
        (AshlarModel,),
        {
            "ag_key": _key,
            "ag_name": f"AshlarSweep-{_name}",
            "recipe": _recipe,
            "defaults": {**_RIP, **_overrides},
            "__module__": __name__,
        },
    )
    globals()[_cls.__name__] = _cls
    SWEEP_MODELS[f"sweep_{_name}"] = _cls


# Round 4: Ashlar's own ideas on top of the TabM recipe with TabM's batch rule.
class AshlarTabMBsModel(AshlarTabMRipModel):
    """``tabm_piecewise`` until early stopping, with TabM's batch-size rule: the base for round 4."""

    ag_key = "ASRTABMBS"
    ag_name = "AshlarTabMBs"
    batch_rule = True


class AshlarTabMBsAucModel(AshlarTabMBsModel):
    ag_key = "ASRTABMBSAUC"
    ag_name = "AshlarTabMBs-auc"
    stop_on_auc = True


class AshlarTabMBsPooledModel(AshlarTabMBsModel):
    ag_key = "ASRTABMBSPOOL"
    ag_name = "AshlarTabMBs-pooled"
    pool_min_levels = 30


class AshlarTabMBsAvgModel(AshlarTabMBsModel):
    ag_key = "ASRTABMBSAVG"
    ag_name = "AshlarTabMBs-avg"
    defaults = {**_RIP, "weight_average": 0.8}


class AshlarTabMBsAllModel(AshlarTabMBsModel):
    ag_key = "ASRTABMBSALL"
    ag_name = "AshlarTabMBs-all"
    stop_on_auc = True
    pool_min_levels = 30
    defaults = {**_RIP, "weight_average": 0.8}


ROUND4_MODELS = {
    "tabm_bs": AshlarTabMBsModel,
    "tabm_bs_auc": AshlarTabMBsAucModel,
    "tabm_bs_pooled": AshlarTabMBsPooledModel,
    "tabm_bs_avg": AshlarTabMBsAvgModel,
    "tabm_bs_all": AshlarTabMBsAllModel,
}
