"""TabArena (AutoGluon ``AbstractModel``) wrappers for Ashlar.

Two entries, one per recipe, so their configs get distinct names on the
leaderboard:

* ``AshlarModel`` (``ASR``): ``ASRRegressor`` / ``ASRClassifier`` defaults,
  a residual MLP on piecewise-linear numeric embeddings.
* ``AshlarTabMModel`` (``ASRTABM``): ``ashlar.tabm_piecewise``, the
  accuracy recipe (TabM ensemble on piecewise embeddings, quantile scaling).

This module must be importable (not ``__main__``) so Ray workers can unpickle
the classes when the runner is not in debug mode.

``time_limit`` is passed to Ashlar's ``time_limit`` (85% of it, to leave room
for prediction): training stops at the first epoch end past it and keeps the
best checkpoint. ``num_cpus`` holds the process to that many cores through CPU
affinity (Linux only, and only before the process's first JAX computation;
see ``_limit_cpus``).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from autogluon.core.models import AbstractModel


def _limit_cpus(num_cpus) -> None:
    """Hold this process to ``num_cpus`` cores, which is how JAX on CPU can be limited.

    XLA's CPU runtime has no thread-count setting (``XLA_FLAGS`` thread flags
    are ignored by jax 0.12), but it sizes its thread pool from the process's
    CPU affinity when the pool is first made. So this works only before the
    process's first JAX computation, and only on Linux. Parallel workers
    spread over disjoint blocks of the allowed cores by process id.
    """
    if not num_cpus or not hasattr(os, "sched_setaffinity"):
        return
    allowed = sorted(os.sched_getaffinity(0))
    n = int(num_cpus)
    if n >= len(allowed):
        return
    start = (os.getpid() % (len(allowed) // n)) * n
    os.sched_setaffinity(0, allowed[start : start + n])


def _tabm_batch_size(n_rows: int, n_features: int) -> int:
    """TabArena TabM's batch size rule (tabm_utils.get_tabm_auto_batch_size): by rows, capped by width."""
    by_rows = next((size for limit, size in ((2_800, 32), (4_500, 64), (6_400, 128), (32_000, 256)) if n_rows < limit), 512)
    by_features = next((size for limit, size in ((256, 1024), (512, 512), (1024, 256), (2048, 64)) if n_features < limit), 32)
    return min(by_rows, by_features)


def _tempered(proba: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-12, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    out = np.exp(logits)
    return out / out.sum(axis=1, keepdims=True)


def _fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """The temperature in [0.25, 4] that minimises validation log loss."""
    from scipy.optimize import minimize_scalar

    rows = np.arange(len(y))

    def loss(log_t: float) -> float:
        return -np.mean(np.log(np.clip(_tempered(proba, np.exp(log_t))[rows, y], 1e-12, None)))

    return float(np.exp(minimize_scalar(loss, bounds=(np.log(0.25), np.log(4.0)), method="bounded").x))


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
    #: Early-stop on the metric TabArena scores, of the averaged ensemble, as TabArena's TabM does. Maps the problem
    #: type to an Ashlar early_stopping_metric; a missing type keeps the loss.
    stop_on_metric: dict | None = None
    #: Add a 0/1 categorical "<column>__missing" for each numeric column with missing training values (TabM's
    #: SimpleImputer(add_indicator=True)). Ashlar alone fills them with the median.
    missing_indicators: bool = False
    #: Fit a softmax temperature on the validation fold (classification).
    temperature_scaling: bool = False
    temperature_: float = 1.0

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, **kwargs) -> pd.DataFrame:
        X = super()._preprocess(X, **kwargs)
        X = X.copy()
        X.columns = [str(c) for c in X.columns]
        # Ashlar embeds non-numeric columns. AutoGluon hands categoricals as
        # pandas ``category``, whose integer categories would read as numeric,
        # so pass them as strings. Missing becomes its own "nan" level.
        if is_train:
            # AutoGluon recodes two-valued columns as uint8 0/1 ("bool" special type); keep them categorical so they
            # get an embedding, not piecewise numeric bins over two points.
            self._bool_features = [str(c) for c in self._feature_metadata.get_features(required_special_types=["bool"])]
        flags = set(getattr(self, "_bool_features", []))
        if is_train:
            self._missing_columns = [
                col
                for col in X.columns
                if self.missing_indicators
                and col not in flags
                and pd.api.types.is_numeric_dtype(X[col].dtype)
                and X[col].isna().any()
            ]
        for col in X.columns:
            if isinstance(X[col].dtype, pd.CategoricalDtype) or X[col].dtype == bool or col in flags:
                X[col] = X[col].astype(str).astype(object)
        for col in getattr(self, "_missing_columns", []):
            X[f"{col}__missing"] = np.where(X[col].isna(), "1", "0").astype(object)
        return X

    def _build(self, params: dict):
        from ashlar import ASRClassifier, ASRRegressor, tabm_piecewise

        params = {**self.defaults, **params}
        backbone = {key: params.pop(key) for key in _BACKBONE_KEYS if key in params}
        classification = self.problem_type in ("binary", "multiclass")
        if not classification:
            # Classification-only settings shared by a variant's defaults.
            params.pop("label_smoothing", None)
            if params.get("early_stopping_metric") in {"log_loss", "accuracy", "roc_auc"}:
                params.pop("early_stopping_metric")
        # A distribution head for regression (its mean is the point prediction).
        task = "classification" if classification else params.pop("task", "regression")
        params.pop("task", None)
        if self.recipe == "tabm_piecewise":
            if backbone:
                base = {"n_blocks": 2, "d_block": 256, "dropout": 0.1, "k": 32, "start_scaling_init": "normal"}
                params["backbone_kwargs"] = {**base, **backbone}
            return tabm_piecewise(task=task, **params)
        if backbone:
            backbone.pop("k", None)
            params["backbone_kwargs"] = {"n_blocks": 3, "d_block": 256, "dropout": 0.1, **backbone}
        if classification:
            return ASRClassifier(**params)
        return ASRRegressor(**params) if task == "regression" else ASRRegressor(task=task, **params)

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
        _limit_cpus(num_cpus)
        X = self.preprocess(X, is_train=True)
        y = np.asarray(y, dtype=np.float32 if self.problem_type == "regression" else np.int64)
        validation = None
        if X_val is not None and y_val is not None:
            X_val = self.preprocess(X_val)
            y_val = np.asarray(y_val, dtype=y.dtype)
            validation = (X_val, y_val)
        params = self._get_model_params()
        if self.batch_rule and "batch_size" not in params:
            params["batch_size"] = _tabm_batch_size(*X.shape)
        if self.stop_on_metric and self.problem_type in self.stop_on_metric:
            params.setdefault("early_stopping_metric", self.stop_on_metric[self.problem_type])
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
        if time_limit is not None and "time_limit" not in params:
            # Leave room for the post-training finish, pickling and out-of-fold prediction.
            params["time_limit"] = max(1.0, 0.85 * float(time_limit))
        self.model = self._build(params)
        self.model.fit(X, y, validation=validation)
        if self.temperature_scaling and validation is not None and self.problem_type != "regression":
            self.temperature_ = _fit_temperature(self._aligned_proba(X_val), y_val)

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        if self.problem_type == "regression":
            return np.asarray(self.model.predict(X), dtype=np.float32)
        full = self._aligned_proba(X)
        if self.temperature_ != 1.0:
            full = _tempered(full, self.temperature_)
        return self._convert_proba_to_unified_form(full.astype(np.float32))

    def _aligned_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.asarray(self.model.predict_proba(X), dtype=np.float64)
        # AutoGluon labels are 0..K-1; align columns to them in case a class
        # was missing from a training fold.
        classes = list(np.asarray(self.model.classes_).astype(int))
        full = np.zeros((len(X), self.num_classes), dtype=np.float64)
        full[:, classes] = proba
        return full

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
    ag_name = "AshlarTabMRipBs"
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


# Local sweep on the Mac (default architecture, then what the single-model
# neural baselines leave on the table), all trained until early stopping.
# ``Bs`` is TabM's batch-size rule.
def _variant(base, key, name, batch_rule=True, **params):
    return type(
        name + "Model",
        (base,),
        {"ag_key": key, "ag_name": name, "defaults": {**_RIP, **params}, "batch_rule": batch_rule, "__module__": __name__},
    )


AshlarRipBsModel = _variant(AshlarRipModel, "ASRBS", "AshlarRipBs")
AshlarRipBsDeepModel = _variant(AshlarRipModel, "ASRBSDEEP", "AshlarRipBsDeep", n_blocks=6)
AshlarRipBsWideModel = _variant(AshlarRipModel, "ASRBSWIDE", "AshlarRipBsWide", d_block=512)
AshlarRipBsNeckModel = _variant(AshlarRipModel, "ASRBSNECK", "AshlarRipBsNeck", bottleneck=True)
AshlarRipBsOneHotModel = _variant(AshlarRipModel, "ASRBSOH", "AshlarRipBsOneHot", categorical_embeddings=[])
AshlarTabMRipParityModel = _variant(
    AshlarTabMRipModel, "ASRTABMPAR", "AshlarTabMRipParity", d_block=512, weight_decay=3e-4, gradient_clip=1.0
)
AshlarTabMOneHotModel = _variant(AshlarTabMRipModel, "ASRTABMOH", "AshlarTabMOneHot", categorical_embeddings=[])
AshlarTabMPeriodicModel = _variant(AshlarTabMRipModel, "ASRTABMPER", "AshlarTabMPeriodic", num_embeddings="periodic")
# TabArena's TabM samples batches per member, for classification as well as regression.
AshlarTabMMemberBatchesModel = _variant(
    AshlarTabMRipModel, "ASRTABMMB", "AshlarTabMMemberBatches", share_training_batches=False
)
AshlarTabMQuantileTabrModel = _variant(
    AshlarTabMRipModel, "ASRTABMQT", "AshlarTabMQuantileTabr", numeric_scaling="quantile_tabr"
)
AshlarTabMKitchenModel = _variant(
    AshlarTabMRipModel,
    "ASRTABMKIT",
    "AshlarTabMKitchen",
    d_block=512,
    weight_decay=3e-4,
    gradient_clip=1.0,
    categorical_embeddings=[],
    share_training_batches=False,
)
# Temperature scaling on the bagging fold for log loss, and distribution heads
# (their mean as the point) for skewed or heteroscedastic regression targets.
AshlarTabMTempModel = _variant(AshlarTabMRipModel, "ASRTABMT", "AshlarTabMTemp")
AshlarTabMTempModel.temperature_scaling = True
AshlarTabMMixtureModel = _variant(AshlarTabMRipModel, "ASRTABMMIX", "AshlarTabMMixture", task="mixture")
AshlarTabMGaussianModel = _variant(AshlarTabMRipModel, "ASRTABMGAU", "AshlarTabMGaussian", task="gaussian")
# The submission config with per-member batches: the candidate fix for multiclass log loss.
AshlarTabMRipBsMemberModel = _variant(
    AshlarTabMRipModel, "ASRTABMBSMB", "AshlarTabMRipBsMember", share_training_batches=False
)

# Classification fixes for log loss: label smoothing (as RealMLP), and early
# stopping on the log loss of the averaged prediction instead of the mean
# member loss. Regression ignores both.
AshlarTabMRipBsSmoothModel = _variant(AshlarTabMRipModel, "ASRTABMBSLS", "AshlarTabMRipBsSmooth", label_smoothing=0.1)
AshlarTabMRipBsLogLossModel = _variant(
    AshlarTabMRipModel, "ASRTABMBSLL", "AshlarTabMRipBsLogLoss", early_stopping_metric="log_loss"
)
AshlarTabMRipBsSmoothLogLossModel = _variant(
    AshlarTabMRipModel,
    "ASRTABMBSLSLL",
    "AshlarTabMRipBsSmoothLogLoss",
    label_smoothing=0.1,
    early_stopping_metric="log_loss",
)

# From the audit of TabArena's TabM (Nov 4): it early-stops on the scored metric of the probability-averaged ensemble,
# adds missing-value indicators, and trains d_block 512 with weight decay 3e-4 and gradient clipping 1.0.
AshlarTabMRipBsMetricModel = _variant(AshlarTabMRipModel, "ASRTABMBSMET", "AshlarTabMRipBsMetric")
AshlarTabMRipBsMetricModel.stop_on_metric = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
AshlarTabMRipBsEnsembleLossModel = _variant(AshlarTabMRipModel, "ASRTABMBSENS", "AshlarTabMRipBsEnsembleLoss")
AshlarTabMRipBsEnsembleLossModel.stop_on_metric = {"multiclass": "log_loss", "regression": "rmse"}
AshlarTabMRipBsMissingModel = _variant(AshlarTabMRipModel, "ASRTABMBSNA", "AshlarTabMRipBsMissing")
AshlarTabMRipBsMissingModel.missing_indicators = True
AshlarTabMRipBsParityModel = _variant(
    AshlarTabMRipModel, "ASRTABMBSPAR", "AshlarTabMRipBsParity", d_block=512, weight_decay=3e-4, gradient_clip=1.0
)

LOCAL_SWEEP_MODELS = {
    "tabm_bs_metric": AshlarTabMRipBsMetricModel,
    "tabm_bs_ensloss": AshlarTabMRipBsEnsembleLossModel,
    "tabm_bs_missing": AshlarTabMRipBsMissingModel,
    "tabm_bs_parity": AshlarTabMRipBsParityModel,
    "tabm_bs_smooth": AshlarTabMRipBsSmoothModel,
    "tabm_bs_logloss": AshlarTabMRipBsLogLossModel,
    "tabm_bs_smooth_logloss": AshlarTabMRipBsSmoothLogLossModel,
    "rip_bs": AshlarRipBsModel,
    "rip_bs_deep": AshlarRipBsDeepModel,
    "rip_bs_wide": AshlarRipBsWideModel,
    "rip_bs_neck": AshlarRipBsNeckModel,
    "rip_bs_onehot": AshlarRipBsOneHotModel,
    "tabm_rip_bs": AshlarTabMBsModel,
    "tabm_rip_parity": AshlarTabMRipParityModel,
    "tabm_onehot": AshlarTabMOneHotModel,
    "tabm_periodic": AshlarTabMPeriodicModel,
    "tabm_member_batches": AshlarTabMMemberBatchesModel,
    "tabm_quantile_tabr": AshlarTabMQuantileTabrModel,
    "tabm_kitchen": AshlarTabMKitchenModel,
    "tabm_temp": AshlarTabMTempModel,
    "tabm_mixture": AshlarTabMMixtureModel,
    "tabm_gaussian": AshlarTabMGaussianModel,
    "tabm_bs_member": AshlarTabMRipBsMemberModel,
}
