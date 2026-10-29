"""Sklearn-shaped shell over an Equinox core.

The shell is mutable in the usual fitted-estimator way. ``core_`` and
``numeric_transform_`` stay ordinary JAX pytrees. ``prepare`` accepts the same
frames as ``fit`` and returns encoded arrays. ``forward_transformed`` accepts
raw numeric arrays, including NaNs, and differentiates through imputation and
the fitted numeric map. Category one-hot columns are a separate argument
because that lookup is a host operation. Learned categorical indices are too.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.base import BaseEstimator

from ashlar._version import __version__
from ashlar.core import PredictiveCore
from ashlar.errors import AshlarError, ShapeError, WeightError
from ashlar.heads import DistributionHead, LogitsHead, NormalHead, OrdinalHead, PointHead, SoftmaxHead
from ashlar.io import dump_ref, load_leaves, load_ref, save_leaves, save_opt_state, load_opt_state
from ashlar.layers import (
    DEFAULT_BOTTLENECK_WIDTH,
    CategoricalEmbeddings,
    FeatureTransformer,
    LateBottleneck,
    SOFT_STAGE_SHRINKAGE,
    SoftStageHistogram,
    LinearReLUEmbeddings,
    PeriodicEmbeddings,
    PiecewiseLinearEmbeddings,
    ResidualMLP,
    SetEmbeddings,
    SetEncoder,
    TabMBackbone,
    quantile_edges,
)
from ashlar.objectives import (
    classification_accuracy,
    classification_objective,
    normal_objective,
    make_distribution_objective,
    make_ordinal_expectation_objective,
    make_ordinal_histogram_objective,
    ordinal_objective,
    regression_objective,
)
from ashlar.distributional import QUANTILE_LEVELS, decode, log_density, point_values, sample_values
from ashlar.predict import distribution_from_normal_raw, mean_logits, mean_probabilities, split_normal_parameters
from ashlar.preprocess import NUMERIC_SCALINGS, FeatureEncoder, frame_columns, target_array
from ashlar.search import resolve_search, trial_grid
from ashlar.train import EVAL_CHUNK_ROWS, auto_batch_size, auto_max_epochs, fit_loop, make_optimizer
from ashlar.weights import REDUCTION, class_weight_multiplier, validate_sample_weight

_BACKBONES = {
    "residual_mlp": ResidualMLP,
    "tabm": TabMBackbone,
    "transformer": FeatureTransformer,
    "set": SetEncoder,
}
_FIELD_BACKBONES = (FeatureTransformer, SetEncoder)
_HEADS = {
    "point": PointHead,
    "logits": LogitsHead,
    "normal": NormalHead,
    "ordinal": OrdinalHead,
    "softmax": SoftmaxHead,
    "distribution": DistributionHead,
}
_ORDINAL_HEADS = {"ordinal", "softmax"}
_TASK_HEAD = {"regression": "point", "classification": "logits", "normal": "normal", "ordinal": "ordinal"}
_OBJECTIVES = {
    "regression": regression_objective,
    "classification": classification_objective,
    "normal": normal_objective,
    "ordinal": ordinal_objective,
}
_LABELED_TASKS = {"classification", "ordinal"}
# Tasks whose head is a whole predictive distribution from ``ashlar.distributional``.
DISTRIBUTION_TASKS = ("gaussian", "mixture", "flow", "histogram", "quantile")
_SCALED_TASKS = {"regression", *DISTRIBUTION_TASKS}
_TASK_HEAD.update({task: "distribution" for task in DISTRIBUTION_TASKS})
_NUM_EMBEDDINGS = {None, "linear_relu", "periodic", "piecewise", "piecewise_relu"}
# Bins per numeric column for ``num_embeddings="piecewise"``.
PIECEWISE_BINS = 48


def _squeeze_univariate(point):
    if getattr(point, "ndim", 0) == 2 and point.shape[-1] == 1:
        return point[:, 0]
    return point


def _point_from_raw(task, policy, raw, point_fn):
    if task == "regression":
        if raw.ndim != 3:
            raise ShapeError(f"regression output must have shape (batch, member, event); received {raw.shape}.")
        return _squeeze_univariate(jnp.mean(raw, axis=1))
    if task == "classification":
        if policy == "mean_probabilities":
            return mean_probabilities(raw)
        if policy == "mean_logits":
            return mean_logits(raw)
        raise AshlarError("classification prediction_policy must be 'mean_probabilities' or 'mean_logits'.")
    if task == "ordinal":
        if policy != "mean_probabilities":
            raise AshlarError("ordinal prediction_policy must be 'mean_probabilities'.")
        if raw.ndim != 3:
            raise ShapeError(f"ordinal log-probabilities must have shape (batch, member, class); received {raw.shape}.")
        probabilities = jnp.exp(raw)
        probabilities = probabilities / jnp.clip(jnp.sum(probabilities, axis=-1, keepdims=True), 1e-30)
        return jnp.mean(probabilities, axis=1)
    if task == "normal":
        loc, _scale = split_normal_parameters(raw)
        return _squeeze_univariate(jnp.mean(loc, axis=1))
    if task in DISTRIBUTION_TASKS:
        return jnp.mean(_member_points(task, raw), axis=1)
    if point_fn is None:
        raise AshlarError("a custom task requires point_fn(raw) -> predictions.")
    return point_fn(raw)


def _decode_members(task, raw):
    """Decode ``(batch, member, width)`` with every member as its own row."""
    batch, members, width = raw.shape
    return decode(task, raw.reshape(batch * members, width), 1), batch, members


def _member_points(task, raw):
    """Standardized point of each member, ``(batch, member)``. The mean for a mean-type head."""
    decoded, batch, members = _decode_members(task, raw)
    return point_values(task, decoded).reshape(batch, members)


def _dist_from_raw(task, policy, raw, dist_fn):
    if task == "normal":
        return distribution_from_normal_raw(raw, policy)
    if dist_fn is None:
        raise AshlarError("predict_dist requires a normal task or a custom dist_fn.")
    return dist_fn(raw)


class TabularEstimator(BaseEstimator):
    """Generic estimator. Architecture and task are constructor arguments, not subclasses.

    Parameters
    ----------
    task:
        ``regression``, ``classification``, ``ordinal``, ``normal``, or ``custom``.
        ``ordinal`` is a cumulative logit over numeric labels sorted by value.
        ``head_kwargs={"class_prior": "empirical"}`` freezes the cuts at a
        Jeffreys prior from the training counts. The default cuts stay learnable.
    backbone:
        ``residual_mlp``, ``tabm``, or an Equinox class with
        ``__init__(d_in, *, key, **kwargs)`` and attributes ``k`` and ``d_out``.
    prediction_policy:
        Classification: ``mean_probabilities`` (default) or ``mean_logits``.
        Ordinal: ``mean_probabilities``, the mean of the member class probabilities.
        Normal ``predict_dist``: ``mixture`` (default) or ``mean_parameters``.
        Regression point predictions average member outputs.
    split:
        ``iid``, ``temporal``, or ``group``. Preprocessing is fit on the
        training partition only. ``validation_size=0`` uses every row for
        training and disables early stopping. ``patience=0`` also disables
        early stopping while still holding out the validation slice.
    weight_decay:
        AdamW penalty. ``0`` turns it off. The default is ``1e-4``.
    feature_dropout:
        Probability of dropping each input field on a training step. ``0``
        turns it off. Prediction does not drop fields. The default is ``0``.
    numeric_scaling:
        ``robust`` (default) is median impute, median center, interquartile
        scale, then a smooth clip that saturates at ±3. ``standard`` is
        median impute, then mean and population standard deviation, with no
        clip. ``quantile`` and ``quantile_tabr`` replace that map with a
        train-only normal quantile transform. Scaling does not change the
        backbone, the learning rate, or the penalties. Linear-ReLU numeric
        embeddings stay off unless ``num_embeddings`` is set.
    learning_rate_schedule:
        ``constant`` (default) keeps the AdamW rate fixed. ``linear`` decays
        that same rate to zero across the epoch budget. ``surface`` warms up
        at ``1e-4`` for one epoch, capped at 20 steps, then sets the AdamW
        rate from the training-batch curvature. The optimizer stays AdamW.
        The default schedule is ``constant``.
    standardize_target:
        Regression only. ``None`` turns it on for regression and off for
        every other task. ``True`` fits the training mean and population
        standard deviation, trains on that z-score, and inverts it in
        ``predict``. A standard deviation below ``1e-6`` is replaced by 1.
        ``False`` leaves ``y`` on its original scale. Classification and
        ordinal stay off unless you pass ``True``, which raises.
    share_training_batches:
        ``True`` (default) shows every member the same minibatch. ``False``
        gives each TabM member its own shuffle. Prediction still averages
        members on the same rows. Regression only.
    bottleneck:
        ``None`` (default) leaves the head on the backbone width. ``True``
        inserts a per-member linear map of width 32 and a ReLU after the
        backbone and before the head. A positive integer uses that width.
        The map has to be narrower than the backbone. The residual default
        leaves this off.
    histogram_variance:
        ``None`` (default) trains the ordinal model on the true class.
        A positive number is the variance of a truncated Gaussian centered
        on that class, with one bin per class of width 1. The loss is the
        cross-entropy to that soft target, and the head must be ``softmax``.
        ``predict_score`` is still the expected label. The residual default
        leaves this off.
    expectation_weight:
        ``None`` (default) trains an ordinal model on the class negative
        log-likelihood alone. A non-negative number adds that many times the
        squared error of ``probabilities @ class_values``, the same expected
        label ``predict_score`` returns. Probabilities are the mean of the
        member probabilities. The residual default leaves this off, and it
        does not apply outside an ordinal task.
    soft_stages:
        ``None`` (default) leaves class scores to the head. A positive
        integer adds that many soft stumps on the numeric columns. Each
        stump has two class-logit leaves, mixed by a sigmoid gate. The
        stumps are summed with shrinkage 0.3 and added to the head's
        log-probabilities. The residual default leaves this off.
    sample_weight:
        Direct loss weights, accepted for every task. ``None`` means uniform
        weights. This is not row sampling.
    categorical_embeddings:
        ``None`` (default) learns an embedding for every categorical column
        that is not a set feature. Integer columns stay numeric unless named
        in ``categorical``, here, or ``set_features``. A list embeds those
        names and one-hots the other categoricals. ``[]`` one-hots every
        categorical. The constructor value is what ``get_params`` returns;
        the fitted names are ``embedded_columns_``.
    set_features:
        Column names whose cells are a list of categories. Each list is a
        variable-length set: a masked mean of learned token embeddings.
    """

    def __init__(
        self,
        task: str = "regression",
        backbone: str | type = "residual_mlp",
        backbone_kwargs: dict | None = None,
        head: str | type | None = None,
        head_kwargs: dict | None = None,
        objective=None,
        point_fn=None,
        dist_fn=None,
        num_embeddings: str | None = "piecewise",
        num_embedding_dim: int = 16,
        n_events: int | None = None,
        validation_size: float = 0.2,
        split: str = "iid",
        batch_size: int | str = "auto",
        max_epochs: int | str = "auto",
        patience: int = 16,
        learning_rate: float = 2e-3,
        learning_rate_schedule: str = "cosine",
        standardize_target: bool | None = None,
        share_training_batches: bool = True,
        bottleneck: int | bool | None = None,
        histogram_variance: float | None = None,
        expectation_weight: float | None = None,
        soft_stages: int | None = None,
        weight_decay: float = 1e-4,
        feature_dropout: float = 0.0,
        numeric_scaling: str = "robust",
        prediction_policy: str | None = None,
        class_weight=None,
        categorical: list[str] | None = None,
        categorical_embeddings: list[str] | None = None,
        set_features: list[str] | None = None,
        categorical_embedding_dim: int = 8,
        categorical_embedding_hidden: bool = False,
        set_max_length: int = 8,
        random_state: int = 0,
    ):
        self.task = task
        self.backbone = backbone
        self.backbone_kwargs = backbone_kwargs
        self.head = head
        self.head_kwargs = head_kwargs
        self.objective = objective
        self.point_fn = point_fn
        self.dist_fn = dist_fn
        self.num_embeddings = num_embeddings
        self.num_embedding_dim = num_embedding_dim
        self.n_events = n_events
        self.validation_size = validation_size
        self.split = split
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.learning_rate_schedule = learning_rate_schedule
        if standardize_target is None:
            standardize_target = task in _SCALED_TASKS
        self.standardize_target = bool(standardize_target)
        self.share_training_batches = share_training_batches
        self.bottleneck = bottleneck
        self.histogram_variance = histogram_variance
        self.expectation_weight = expectation_weight
        self.soft_stages = soft_stages
        self.weight_decay = weight_decay
        self.feature_dropout = feature_dropout
        self.numeric_scaling = numeric_scaling
        self.prediction_policy = prediction_policy
        self.class_weight = class_weight
        self.categorical = categorical
        self.categorical_embeddings = categorical_embeddings
        self.set_features = set_features
        self.categorical_embedding_dim = categorical_embedding_dim
        self.categorical_embedding_hidden = categorical_embedding_hidden
        self.set_max_length = set_max_length
        self.random_state = random_state

    def _check_config(self):
        if self.task not in {"regression", "classification", "ordinal", "normal", "custom", *DISTRIBUTION_TASKS}:
            raise AshlarError(
                "task must be 'regression', 'classification', 'ordinal', 'normal', 'custom', "
                "or a distribution: 'gaussian', 'mixture', 'flow', 'histogram', 'quantile'."
            )
        if self.split not in {"iid", "temporal", "group"}:
            raise AshlarError("split must be 'iid', 'temporal', or 'group'.")
        if self.random_state is None or int(self.random_state) < 0:
            raise AshlarError("random_state must be a non-negative integer.")
        if not 0.0 <= self.validation_size < 1.0:
            raise AshlarError("validation_size must be in [0, 1).")
        for name in ("max_epochs", "batch_size"):
            value = getattr(self, name)
            if value != "auto" and (isinstance(value, (str, bool)) or int(value) < 1):
                raise AshlarError(f"{name} must be a positive integer or 'auto'.")
        if self.patience < 0:
            raise AshlarError("patience must be non-negative.")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise AshlarError("learning_rate must be positive and weight_decay must be non-negative.")
        if self.learning_rate_schedule not in {"constant", "linear", "cosine", "surface"}:
            raise AshlarError(
                "learning_rate_schedule must be 'constant', 'linear', 'cosine', or 'surface'. AdamW is the only optimizer."
            )
        if not 0.0 <= float(self.feature_dropout) < 1.0:
            raise AshlarError("feature_dropout must be in [0, 1).")
        if self.numeric_scaling not in NUMERIC_SCALINGS:
            raise AshlarError(
                "numeric_scaling must be 'standard', 'robust', 'quantile', or 'quantile_tabr'."
            )
        if self.standardize_target and self.task not in _SCALED_TASKS:
            raise AshlarError(
                "standardize_target is a regression label z-score. It applies to regression and the distribution tasks."
            )
        if self.task in {"flow", "histogram"} and not self.standardize_target:
            raise AshlarError(f"the {self.task} head works on a standardized axis, so standardize_target must stay on.")
        if self.share_training_batches not in {True, False}:
            raise AshlarError("share_training_batches must be true or false.")
        if not self.share_training_batches and self.task != "regression":
            raise AshlarError("share_training_batches=False is implemented for regression.")
        if self.num_embeddings not in _NUM_EMBEDDINGS:
            raise AshlarError("num_embeddings must be None, 'linear_relu', 'periodic', 'piecewise', or 'piecewise_relu'.")
        head = self.head
        if isinstance(head, str) and self.task == "ordinal":
            if head not in _ORDINAL_HEADS:
                raise AshlarError("ordinal head must be 'ordinal' or 'softmax'.")
        elif isinstance(head, str) and self.task in _TASK_HEAD and head != _TASK_HEAD[self.task]:
            raise AshlarError(
                f"task {self.task!r} uses the { _TASK_HEAD[self.task]!r} head; received head {head!r}."
            )
        if self.task == "custom" and (head is None or self.objective is None or self.point_fn is None):
            raise AshlarError("a custom task requires a head class, an objective, and point_fn.")
        if self.task not in _LABELED_TASKS and self.class_weight is not None:
            raise AshlarError(
                "class_weight is only applied for classification and ordinal classification, and it is a loss multiplier, not a sampler."
            )
        policy = self._policy()
        if self.task == "classification" and policy not in {"mean_probabilities", "mean_logits"}:
            raise AshlarError("classification prediction_policy must be 'mean_probabilities' or 'mean_logits'.")
        if self.task == "ordinal" and policy != "mean_probabilities":
            raise AshlarError("ordinal prediction_policy must be 'mean_probabilities'.")
        if self.task == "normal" and policy not in {"mixture", "mean_parameters"}:
            raise AshlarError("normal prediction_policy must be 'mixture' or 'mean_parameters'.")
        if self.task == "regression" and policy != "mean_members":
            raise AshlarError("regression prediction_policy must be 'mean_members'.")
        if self.task in DISTRIBUTION_TASKS and policy != "mixture":
            raise AshlarError("a distribution task predicts the equal-weight mixture of its members: prediction_policy='mixture'.")
        if isinstance(self.backbone, str) and self.backbone not in _BACKBONES:
            raise AshlarError("backbone must be 'residual_mlp', 'tabm', 'transformer', 'set', or a class.")
        if int(self.categorical_embedding_dim) < 1 or int(self.set_max_length) < 1:
            raise AshlarError("categorical_embedding_dim and set_max_length must be positive.")
        self._bottleneck_width()
        if self.histogram_variance is not None:
            if self.task != "ordinal":
                raise AshlarError("histogram_variance is an ordinal soft target. It does not apply to other tasks.")
            variance = float(self.histogram_variance)
            if not np.isfinite(variance) or variance <= 0.0:
                raise AshlarError("histogram_variance must be positive and finite.")
            if self.head != "softmax":
                raise AshlarError(
                    "histogram_variance trains a histogram over class bins, so the ordinal head must be 'softmax'."
                )
        self._expectation_weight()
        self._soft_stage_count()
        kwargs = dict(self.backbone_kwargs or {})
        if "scaling_chunks" in kwargs or "arch_type" in kwargs or "feature_chunks" in kwargs:
            raise AshlarError(
                "scaling_chunks and feature_chunks are derived from the fitted schema, and arch_type "
                "other than batch-ensemble TabM is not implemented. Refusing to ignore those arguments."
            )

    def _policy(self) -> str:
        if self.prediction_policy is not None:
            return self.prediction_policy
        if self.task in _LABELED_TASKS:
            return "mean_probabilities"
        if self.task == "normal" or self.task in DISTRIBUTION_TASKS:
            return "mixture"
        return "mean_members"

    def _backbone_cls(self):
        if isinstance(self.backbone, str):
            return _BACKBONES[self.backbone]
        return self.backbone

    def _head_cls(self):
        head = self.head
        if head is None:
            head = _TASK_HEAD[self.task]
        if isinstance(head, str):
            return _HEADS[head]
        return head

    def fit(
        self,
        X,
        y,
        sample_weight=None,
        validation=None,
        groups=None,
        validation_sample_weight=None,
        resume: bool = False,
        search=None,
    ):
        """Fit or, with ``resume=True``, continue from ``core_`` and ``opt_state_``.

        ``search=None`` and ``search=False`` train with the constructor settings.
        ``search=True`` runs the small default grid, then trains with the winner.
        A dict widens that grid. See ``ashlar.search.DEFAULT_SEARCH``.
        """
        self._check_config()
        if resume and search not in (None, False):
            raise AshlarError("resume=True continues the current core. Omit search, or pass search=False.")
        self._trial = None
        if not resume:
            # A fresh fit resolves its own settings; a previous fit's must not leak in.
            self.resolved_ = None
        y_array = target_array(y)
        names, columns = frame_columns(X)
        n_rows = len(columns[0])
        if len(y_array) != n_rows:
            raise AshlarError(f"X has {n_rows} rows and y has {len(y_array)}.")
        weights = validate_sample_weight(sample_weight, n_rows)
        if validation is not None and groups is not None:
            raise AshlarError("pass either an explicit validation tuple or groups for a grouped split, not both.")
        if validation is not None and validation_sample_weight is not None and len(validation) == 3:
            raise AshlarError("validation sample weights were passed both as the third validation item and as validation_sample_weight.")
        train_idx, val_idx, val_pack = self._partition(
            names, columns, y_array, weights, validation, groups, validation_sample_weight
        )
        trials = None
        try:
            if search not in (None, False):
                if resume:
                    raise AshlarError("resume=True continues the current core. Omit search, or pass search=False.")
                spec = resolve_search(search)
                chosen, trials = self._run_search(
                    spec, names, columns, y_array, weights, train_idx, val_idx, val_pack
                )
                self._trial = chosen
            self._assign_after_success(names, columns, y_array, weights, train_idx, val_idx, val_pack, resume)
            self.resolved_ = self._resolved_training()
            self.search_ = None if trials is None else {"chosen": dict(self.resolved_), "trials": trials}
        finally:
            self._trial = None
        return self

    def _partition(self, names, columns, y_array, weights, validation, groups, validation_sample_weight):
        n_rows = len(columns[0])
        if validation is not None:
            if len(validation) not in {2, 3}:
                raise AshlarError("validation must be (X_val, y_val) or (X_val, y_val, sample_weight).")
            train_idx = np.arange(n_rows)
            val_y = target_array(validation[1])
            val_w = (
                validate_sample_weight(validation[2], len(val_y), name="validation sample_weight")
                if len(validation) == 3
                else validate_sample_weight(validation_sample_weight, len(val_y), name="validation_sample_weight")
            )
            return train_idx, None, (validation[0], val_y, val_w)
        if validation_sample_weight is not None:
            raise AshlarError("validation_sample_weight applies to an explicit validation tuple. Internal splits reuse sample_weight.")
        if self.validation_size == 0:
            return np.arange(n_rows), np.array([], dtype=int), None
        train_idx, val_idx = _split_indices(
            n_rows,
            y_array,
            mode=self.split,
            validation_size=self.validation_size,
            groups=groups,
            random_state=int(self.random_state),
            task=self.task,
        )
        return train_idx, val_idx, None

    def _assign_after_success(self, names, columns, y_array, weights, train_idx, val_idx, val_pack, resume):
        self._n_train = len(train_idx)
        train_frame = _take_rows(names, columns, train_idx)
        y_train = y_array[train_idx]
        w_train = weights[train_idx]
        classes, y_train_model, w_train = self._training_target(y_train, w_train)
        if self.task not in {"custom", *_LABELED_TASKS}:
            if not np.isfinite(np.asarray(y_train_model, dtype=np.float64)).all():
                raise AshlarError("regression and normal targets must be finite. Mask missing outcomes in a custom objective.")
        y_train_model = self._fit_target_scale(y_train_model)
        categorical = list(self._categorical_names(names, columns))
        encoder = FeatureEncoder.fit(
            train_frame,
            categorical=categorical,
            embeddings=self._embedding_argument(),
            set_features=list(self.set_features or []),
            set_max_length=int(self.set_max_length),
            numeric_scaling=self.numeric_scaling,
        )
        if self.num_embeddings is not None and encoder.n_numeric == 0:
            raise AshlarError("num_embeddings was requested, but the training partition has no numeric columns.")
        n_events, n_classes = self._output_sizes(y_train_model, classes)
        encoded = encoder.transform(train_frame)
        numeric_z = encoder.numeric_transform(encoded.numeric_raw)
        if resume:
            self._check_resume(encoder, n_events, n_classes)
            core = self.core_
            opt_state = self.opt_state_
            train_key = self._train_key
        else:
            init_key, train_key = jax.random.split(jax.random.PRNGKey(int(self.random_state)))
            core = self._build_core(
                encoder,
                n_events,
                n_classes,
                init_key,
                head_kwargs=self._head_kwargs_for_fit(y_train_model, n_classes),
                numeric_z=numeric_z,
            )
            opt_state = None
        y_device, w_device = _device_targets(self.task, y_train_model, w_train)
        validation = self._validation_device(
            encoder, names, columns, y_array, weights, val_idx, val_pack, classes, n_events
        )
        resolved = self._resolved_training()
        trained, opt_state, history = fit_loop(
            core,
            numeric_z,
            encoded.categorical_onehot,
            y_device,
            w_device,
            objective=self._objective(classes),
            learning_rate=float(resolved["learning_rate"]),
            learning_rate_schedule=self.learning_rate_schedule,
            share_training_batches=bool(self.share_training_batches),
            weight_decay=float(resolved["weight_decay"]),
            batch_size=int(resolved["batch_size"]),
            max_epochs=self._max_epochs(),
            patience=int(self.patience),
            key=train_key,
            validation=validation,
            opt_state=opt_state,
            category_index=encoded.category_index,
            set_tokens=encoded.set_tokens,
            set_mask=encoded.set_mask,
        )
        self.encoder_ = encoder
        self.embedded_columns_ = [column.name for column in encoder.embedding_columns]
        self.numeric_transform_ = encoder.numeric_transform
        self.core_ = trained
        self.opt_state_ = opt_state
        self.classes_ = None if classes is None else np.asarray(classes)
        self.n_events_ = n_events
        self.n_classes_ = n_classes
        self.history_ = history
        if "chosen_learning_rate" in history:
            self.learning_rate_ = float(history["chosen_learning_rate"])
        self.reduction_ = REDUCTION
        self.n_features_in_ = len(names)
        self._train_key = jax.random.fold_in(train_key, max(1, len(history["train_loss"])))
        self.is_fitted_ = True

    def _resolved_training(self) -> dict:
        trial = getattr(self, "_trial", None)
        if trial:
            kwargs = dict(self.backbone_kwargs or {})
            if isinstance(trial.get("backbone_kwargs"), dict):
                kwargs.update(trial["backbone_kwargs"])
            return {
                "learning_rate": float(trial.get("learning_rate", self.learning_rate)),
                "weight_decay": float(trial.get("weight_decay", self.weight_decay)),
                "feature_dropout": float(trial.get("feature_dropout", self.feature_dropout)),
                "batch_size": self._batch_size(trial.get("batch_size", self.batch_size)),
                "backbone_kwargs": kwargs or None,
            }
        saved = getattr(self, "resolved_", None)
        if saved:
            return saved
        return {
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "feature_dropout": float(self.feature_dropout),
            "batch_size": self._batch_size(self.batch_size),
            "backbone_kwargs": self.backbone_kwargs,
        }

    def _batch_size(self, value) -> int:
        if value == "auto":
            return auto_batch_size(getattr(self, "_n_train", 0))
        return int(value)

    def _max_epochs(self) -> int:
        if self.max_epochs == "auto":
            return auto_max_epochs(getattr(self, "_n_train", 0))
        return int(self.max_epochs)

    def _run_search(self, spec, names, columns, y_array, weights, train_idx, val_idx, val_pack):
        if val_pack is None and (val_idx is None or len(val_idx) == 0):
            raise AshlarError(
                "search needs a validation partition. Set validation_size > 0 or pass validation=(X_val, y_val). "
                "The no-search path can still train on every row."
            )
        trials = trial_grid(spec)
        records = []
        best = None
        best_loss = None
        best_score = None
        for index, trial in enumerate(trials):
            self._trial = trial
            try:
                outcome = self._trial_loss(
                    spec, names, columns, y_array, weights, train_idx, val_idx, val_pack, index
                )
            finally:
                self._trial = None
            record = {"params": {key: _json_ready(value) for key, value in trial.items()}, **outcome}
            records.append(record)
            if outcome["metric"] == "weighted_accuracy":
                score = outcome["val_score"]
                if best_score is None or score > best_score:
                    best_score = score
                    best = trial
            elif best_loss is None or outcome["val_loss"] < best_loss:
                best_loss = outcome["val_loss"]
                best = trial
        if best is None:
            raise AshlarError("the search grid is empty.")
        return best, records

    def _trial_loss(self, spec, names, columns, y_array, weights, train_idx, val_idx, val_pack, index):
        self._n_train = len(train_idx)
        train_frame = _take_rows(names, columns, train_idx)
        y_train = y_array[train_idx]
        w_train = weights[train_idx]
        classes, y_train_model, w_train = self._training_target(y_train, w_train)
        y_train_model = self._fit_target_scale(y_train_model)
        encoder = FeatureEncoder.fit(
            train_frame,
            categorical=list(self._categorical_names(names, columns)),
            embeddings=self._embedding_argument(),
            set_features=list(self.set_features or []),
            set_max_length=int(self.set_max_length),
            numeric_scaling=self.numeric_scaling,
        )
        n_events, n_classes = self._output_sizes(y_train_model, classes)
        encoded = encoder.transform(train_frame)
        numeric_z = encoder.numeric_transform(encoded.numeric_raw)
        init_key, train_key = jax.random.split(jax.random.PRNGKey(int(self.random_state)))
        core = self._build_core(
            encoder,
            n_events,
            n_classes,
            init_key,
            head_kwargs=self._head_kwargs_for_fit(y_train_model, n_classes),
            numeric_z=numeric_z,
        )
        y_device, w_device = _device_targets(self.task, y_train_model, w_train)
        validation = self._validation_device(
            encoder, names, columns, y_array, weights, val_idx, val_pack, classes, n_events
        )
        resolved = self._resolved_training()
        score_fn = None
        if self.task == "classification":
            policy = self._policy()

            def score_fn(model, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch, policy=policy):
                raw = model(z_batch, cat_batch, index_batch, token_batch, mask_batch, key=None)
                return classification_accuracy(raw, y_batch, w_batch, policy=policy)

        _trained, _state, history = fit_loop(
            core,
            numeric_z,
            encoded.categorical_onehot,
            y_device,
            w_device,
            objective=self._objective(classes),
            learning_rate=float(resolved["learning_rate"]),
            learning_rate_schedule=self.learning_rate_schedule,
            share_training_batches=bool(self.share_training_batches),
            weight_decay=float(resolved["weight_decay"]),
            batch_size=int(resolved["batch_size"]),
            max_epochs=int(spec["max_epochs"]),
            patience=int(spec["patience"]),
            key=jax.random.fold_in(train_key, index + 1),
            validation=validation,
            category_index=encoded.category_index,
            set_tokens=encoded.set_tokens,
            set_mask=encoded.set_mask,
            score_fn=score_fn,
        )
        if not history["val_loss"]:
            raise AshlarError("a search trial produced no validation loss.")
        result = {"val_loss": float(min(history["val_loss"])), "metric": "val_loss"}
        if self.task == "classification":
            if not history.get("val_score"):
                raise AshlarError("a classification search trial produced no accuracy score.")
            result = {
                "val_loss": float(min(history["val_loss"])),
                "val_score": float(max(history["val_score"])),
                "metric": "weighted_accuracy",
            }
        return result

    def _embedding_argument(self) -> list[str] | None:
        """``None`` embeds every non-set categorical. A list, including ``[]``, is explicit."""
        if self.categorical_embeddings is None:
            return None
        return list(self.categorical_embeddings)

    def _categorical_names(self, names, columns):
        requested = set(self.categorical or [])
        requested.update(self.categorical_embeddings or [])
        requested.update(self.set_features or [])
        unknown = requested.difference(names)
        if unknown:
            raise AshlarError(f"categorical columns {sorted(unknown)} are not in X.")
        for name, values in zip(names, columns, strict=True):
            if values.dtype == object or np.issubdtype(values.dtype, np.bool_) or not np.issubdtype(values.dtype, np.number):
                requested.add(name)
        return requested

    def _training_target(self, y_train, w_train):
        if self.task in _LABELED_TASKS:
            labels = _ordinal_label_vector(y_train) if self.task == "ordinal" else np.asarray(y_train)
            classes, encoded = np.unique(labels, return_inverse=True)
            kind = "ordinal classification" if self.task == "ordinal" else "classification"
            if len(classes) < 2:
                raise AshlarError(f"{kind} requires at least 2 classes in the training partition.")
            encoded = encoded.astype(np.int32)
            weights = _apply_class_weight(labels, encoded, len(classes), self.class_weight, w_train)
            return classes, encoded, weights
        if self.task == "custom":
            return None, np.asarray(y_train, dtype=np.float32), w_train
        return None, _regression_target(y_train, self.n_events), w_train

    def _output_sizes(self, y_train_model, classes):
        if self.task in _LABELED_TASKS:
            return 1, int(len(classes))
        if self.task == "custom":
            events = 1 if self.n_events is None else int(self.n_events)
            return events, 2
        events = 1 if np.asarray(y_train_model).ndim == 1 else int(np.asarray(y_train_model).shape[1])
        if self.n_events is not None and int(self.n_events) != events:
            raise ShapeError(f"n_events={self.n_events} does not match y with event count {events}.")
        return events, 2

    def _expectation_weight(self) -> float | None:
        value = self.expectation_weight
        if value is None:
            return None
        if isinstance(value, bool):
            raise AshlarError("expectation_weight must be a non-negative finite number, or None.")
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise AshlarError("expectation_weight must be a non-negative finite number, or None.") from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise AshlarError("expectation_weight must be a non-negative finite number, or None.")
        if weight == 0.0:
            return None
        if self.task != "ordinal":
            raise AshlarError(
                "expectation_weight adds squared error of the expected label to the ordinal likelihood. "
                "It does not apply to other tasks."
            )
        if self.histogram_variance is not None:
            raise AshlarError(
                "expectation_weight is added to the class negative log-likelihood. "
                "histogram_variance replaces that likelihood."
            )
        if self.objective is not None:
            raise AshlarError(
                "expectation_weight mixes with the built-in ordinal likelihood. A custom objective replaces that mix."
            )
        return weight

    def _objective(self, class_values=None):
        if self.objective is not None:
            return self.objective
        if self.task == "ordinal" and self.histogram_variance is not None:
            return make_ordinal_histogram_objective(float(self.histogram_variance))
        if self.task in DISTRIBUTION_TASKS:
            return make_distribution_objective(self.task)
        base = _OBJECTIVES[self.task]
        weight = self._expectation_weight()
        if weight is None:
            return base
        if class_values is None:
            raise AshlarError("expectation_weight needs the ordered class values from the training labels.")
        return make_ordinal_expectation_objective(np.asarray(class_values, dtype=np.float64), weight)

    def _head_kwargs_for_fit(self, y_encoded, n_classes: int) -> dict:
        kwargs = dict(self.head_kwargs or {})
        if self.task in DISTRIBUTION_TASKS:
            kwargs["kind"] = self.task
        if kwargs.get("class_prior") != "empirical":
            return kwargs
        if self.task != "ordinal":
            raise AshlarError("class_prior='empirical' is an ordinal cut prior. It does not apply to other tasks.")
        counts = np.bincount(np.asarray(y_encoded, dtype=np.int64).reshape(-1), minlength=int(n_classes)).astype(np.float64)
        total = float(counts.sum())
        if total <= 0.0:
            raise AshlarError("the empirical class prior needs at least one training label.")
        alpha = 0.5
        prior = (counts + alpha) / (total + alpha * int(n_classes))
        resolved = dict(kwargs)
        resolved["class_prior"] = [float(value) for value in prior.tolist()]
        self.class_prior_ = np.asarray(resolved["class_prior"], dtype=np.float64)
        return resolved

    def _head_kwargs_for_load(self) -> dict:
        kwargs = dict(self.head_kwargs or {})
        if self.task in DISTRIBUTION_TASKS:
            kwargs["kind"] = self.task
        return kwargs

    def _bottleneck_width(self) -> int | None:
        value = self.bottleneck
        if value is None or value is False:
            return None
        if value is True:
            return DEFAULT_BOTTLENECK_WIDTH
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise AshlarError(
                "bottleneck must be None, False, True, or a positive integer width. "
                "True uses the default width of 32."
            )
        return int(value)

    def _soft_stage_count(self) -> int | None:
        value = self.soft_stages
        if value is None or value is False:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
            raise AshlarError("soft_stages must be None or a positive integer.")
        if self.task != "ordinal":
            raise AshlarError("soft_stages is an ordinal leaf histogram. It does not apply to other tasks.")
        return int(value)

    def _build_core(
        self, encoder: FeatureEncoder, n_events: int, n_classes: int, key, head_kwargs=None, numeric_z=None
    ):
        """Initial core. ``numeric_z`` is the scaled training partition; piecewise bins come from it.

        Without it, as on ``load``, the bins are placeholders of the right shape.
        """
        emb_key, cat_key, set_key, backbone_key, head_key = jax.random.split(key, 5)
        embedding_dim = None
        if self.num_embeddings is not None:
            embedding_dim = int(self.num_embedding_dim)
        cat_dim = int(self.categorical_embedding_dim)
        chunks = encoder.feature_chunks(embedding_dim, cat_dim)
        d_in = sum(chunks)
        backbone_cls = self._backbone_cls()
        kwargs = dict(self._resolved_training()["backbone_kwargs"] or {})
        if backbone_cls is TabMBackbone:
            kwargs["scaling_chunks"] = chunks
            kwargs["share_training_batches"] = bool(self.share_training_batches)
        if backbone_cls in _FIELD_BACKBONES:
            kwargs["feature_chunks"] = chunks
        embeddings = None
        if self.num_embeddings == "linear_relu":
            embeddings = LinearReLUEmbeddings(encoder.n_numeric, embedding_dim, key=emb_key)
        elif self.num_embeddings == "periodic":
            embeddings = PeriodicEmbeddings(encoder.n_numeric, embedding_dim, key=emb_key)
        elif self.num_embeddings in {"piecewise", "piecewise_relu"}:
            if numeric_z is None:
                edges = jnp.tile(jnp.linspace(-1.0, 1.0, PIECEWISE_BINS + 1), (encoder.n_numeric, 1))
            else:
                edges = quantile_edges(numeric_z, PIECEWISE_BINS, seed=int(self.random_state))
            embeddings = PiecewiseLinearEmbeddings(
                edges, embedding_dim, key=emb_key, activation=self.num_embeddings == "piecewise_relu"
            )
        categorical_embeddings = _category_tables(
            encoder.embedding_columns, cat_dim, cat_key, hidden=bool(self.categorical_embedding_hidden)
        )
        set_embeddings = _category_tables(encoder.set_columns, cat_dim, set_key, set_encoder=True)
        backbone = backbone_cls(d_in, key=backbone_key, **kwargs)
        if not hasattr(backbone, "k") or not hasattr(backbone, "d_out"):
            raise AshlarError("a backbone must expose integer attributes k and d_out.")
        width = self._bottleneck_width()
        if width is None:
            bottleneck = None
            d_head = int(backbone.d_out)
        else:
            if int(width) >= int(backbone.d_out):
                raise ShapeError(
                    f"bottleneck width {int(width)} must be narrower than the backbone width {int(backbone.d_out)}."
                )
            bottleneck_key, head_key = jax.random.split(head_key)
            bottleneck = LateBottleneck(int(backbone.d_out), int(backbone.k), int(width), key=bottleneck_key)
            d_head = int(width)
        head = self._head_cls()(
            d_head,
            backbone.k,
            n_events=n_events,
            n_classes=n_classes,
            key=head_key,
            **(self._head_kwargs_for_load() if head_kwargs is None else dict(head_kwargs)),
        )
        stage_count = self._soft_stage_count()
        if stage_count is None:
            soft_histogram = None
        else:
            if int(encoder.n_numeric) < 1:
                raise ShapeError("soft_stages splits numeric columns, and this training partition has none.")
            soft_histogram = SoftStageHistogram(
                int(encoder.n_numeric),
                int(n_classes),
                int(stage_count),
                shrinkage=SOFT_STAGE_SHRINKAGE,
                key=jax.random.fold_in(key, 8),
            )
        return PredictiveCore(
            embeddings=embeddings,
            categorical_embeddings=categorical_embeddings,
            set_embeddings=set_embeddings,
            backbone=backbone,
            bottleneck=bottleneck,
            head=head,
            soft_histogram=soft_histogram,
            n_numeric=encoder.n_numeric,
            n_categorical=sum(encoder.categorical_widths),
            n_embedding_columns=len(encoder.embedding_columns),
            n_set_columns=len(encoder.set_columns),
            set_length=encoder.set_length,
            feature_dropout=float(self._resolved_training()["feature_dropout"]),
            feature_chunks=tuple(int(width) for width in chunks),
        )

    def _fit_target_scale(self, y):
        if not self.standardize_target or self.task not in _SCALED_TASKS:
            self.target_mean_ = None
            self.target_std_ = None
            return y
        array = np.asarray(y, dtype=np.float64)
        mean = np.mean(array, axis=0)
        std = np.asarray(np.std(array, axis=0), dtype=np.float64)
        std = np.where(std < 1e-6, np.asarray(1.0, dtype=np.float64), std)
        self.target_mean_ = float(mean) if np.ndim(mean) == 0 else np.asarray(mean, dtype=np.float64)
        self.target_std_ = float(std) if np.ndim(std) == 0 else np.asarray(std, dtype=np.float64)
        return self._apply_target_scale(array)

    def _apply_target_scale(self, y):
        if not self.standardize_target or self.target_mean_ is None:
            return y
        array = np.asarray(y, dtype=np.float64)
        return (array - np.asarray(self.target_mean_, dtype=np.float64)) / (
            np.asarray(self.target_std_, dtype=np.float64) + 1e-30
        )

    def _check_resume(self, encoder, n_events, n_classes):
        if not getattr(self, "is_fitted_", False):
            raise AshlarError("resume=True requires a fitted estimator. Call fit once before continuing.")
        if encoder.to_json() != self.encoder_.to_json() or n_events != self.n_events_ or n_classes != self.n_classes_:
            raise AshlarError("resume=True requires the same training schema, event count, and class set.")

    def _validation_device(self, encoder, names, columns, y_array, weights, val_idx, val_pack, classes, n_events):
        if val_pack is None and (val_idx is None or len(val_idx) == 0):
            return None
        if val_pack is None:
            frame = _take_rows(names, columns, val_idx)
            y_val = y_array[val_idx]
            w_val = weights[val_idx]
        else:
            frame, y_val, w_val = val_pack
        val_names, val_columns = frame_columns(frame)
        if len(val_columns[0]) != len(np.asarray(y_val)):
            raise AshlarError("validation X and y have different numbers of rows.")
        encoded = encoder.transform(frame)
        numeric_z = encoder.numeric_transform(encoded.numeric_raw)
        categorical = encoded.categorical_onehot
        if self.task in _LABELED_TASKS:
            labels = _ordinal_label_vector(y_val) if self.task == "ordinal" else np.asarray(y_val)
            y_encoded = _encode_known_labels(labels, classes)
            w_val = _apply_class_weight(labels, y_encoded, len(classes), self.class_weight, np.asarray(w_val, dtype=np.float64))
            y_device, w_device = _device_targets(self.task, y_encoded, w_val)
        elif self.task == "custom":
            y_device, w_device = _device_targets(self.task, np.asarray(y_val, dtype=np.float32), w_val)
        else:
            y_model = _regression_target(y_val, n_events)
            if not np.isfinite(np.asarray(y_model, dtype=np.float64)).all():
                raise AshlarError("validation targets must be finite for regression and normal tasks.")
            y_model = self._apply_target_scale(y_model)
            y_device, w_device = _device_targets(self.task, y_model, w_val)
        if float(jnp.sum(w_device)) <= 0.0:
            raise WeightError("validation weights sum to zero, so the early-stopping metric is undefined.")
        return (
            numeric_z,
            categorical,
            encoded.category_index,
            encoded.set_tokens,
            encoded.set_mask,
            y_device,
            w_device,
        )

    def prepare(self, X):
        """Return scaled numeric features and host-encoded one-hot categories.

        Learned category indices and set tokens are not in this pair. ``predict``
        reads them from the frame. Pass them to ``forward_raw`` when the fitted
        schema has those columns and you are not calling ``predict``.
        """
        self._require_fitted()
        encoded = self.encoder_.transform(X)
        return self.numeric_transform_(encoded.numeric_raw), encoded.categorical_onehot

    def forward_raw(self, numeric_raw, categorical_onehot, category_index=None, set_tokens=None, set_mask=None):
        """Run the core from raw numeric values. NaNs are imputed inside JAX."""
        self._require_fitted()
        transformed = self.numeric_transform_(jnp.asarray(numeric_raw, dtype=jnp.float32))
        categorical = jnp.asarray(categorical_onehot, dtype=jnp.float32)
        category_index, set_tokens, set_mask = self._side_inputs(
            transformed.shape[0], category_index, set_tokens, set_mask
        )
        return self.core_(transformed, categorical, category_index, set_tokens, set_mask, key=None)

    def forward_transformed(self, numeric_raw, categorical_onehot, category_index=None, set_tokens=None, set_mask=None):
        """Point predictions from raw numeric values, keeping JAX arrays."""
        raw = self.forward_raw(
            numeric_raw, categorical_onehot, category_index, set_tokens, set_mask
        )
        return _point_from_raw(self.task, self._policy(), raw, self.point_fn)

    def _side_inputs(self, n_rows, category_index, set_tokens, set_mask):
        needs_categories = self.core_.n_embedding_columns > 0
        needs_sets = self.core_.n_set_columns > 0
        if (needs_categories or needs_sets) and (
            category_index is None or (needs_sets and (set_tokens is None or set_mask is None))
        ):
            raise AshlarError(
                "this fitted core has learned categorical embeddings or set features. "
                "Pass category_index, set_tokens, and set_mask from encoder_.transform, or call predict."
            )
        if category_index is None:
            category_index = jnp.zeros((n_rows, 0), dtype=jnp.int32)
        if set_tokens is None:
            set_tokens = jnp.zeros((n_rows, 0, self.core_.set_length), dtype=jnp.int32)
            set_mask = jnp.zeros((n_rows, 0, self.core_.set_length), dtype=jnp.float32)
        return (
            jnp.asarray(category_index, dtype=jnp.int32),
            jnp.asarray(set_tokens, dtype=jnp.int32),
            jnp.asarray(set_mask, dtype=jnp.float32),
        )

    def _raw_from_frame(self, X):
        self._require_fitted()
        encoded = self.encoder_.transform(X)
        numeric_z = self.numeric_transform_(encoded.numeric_raw)
        inputs = (
            numeric_z,
            encoded.categorical_onehot,
            encoded.category_index,
            encoded.set_tokens,
            encoded.set_mask,
        )
        n_rows = int(numeric_z.shape[0])
        if n_rows <= EVAL_CHUNK_ROWS:
            return self.core_(*inputs, key=None)
        forward = _jit_forward()
        pieces = [
            forward(self.core_, *(array[start : start + EVAL_CHUNK_ROWS] for array in inputs))
            for start in range(0, n_rows, EVAL_CHUNK_ROWS)
        ]
        return jnp.concatenate(pieces, axis=0)

    def predict(self, X):
        raw = self._raw_from_frame(X)
        point = np.asarray(_point_from_raw(self.task, self._policy(), raw, self.point_fn))
        if self.task in _SCALED_TASKS and self.standardize_target:
            point = point * np.asarray(self.target_std_, dtype=np.float64) + np.asarray(self.target_mean_, dtype=np.float64)
        if self.task in _LABELED_TASKS:
            indices = np.argmax(point, axis=1)
            return self.classes_[indices]
        if point.ndim == 2 and point.shape[1] == 1:
            return point[:, 0]
        return point

    def score(self, X, y, sample_weight=None):
        """Accuracy for classification and ordinal tasks, R² otherwise.

        The same score a scikit-learn classifier or regressor returns, so
        ``cross_val_score`` and ``GridSearchCV`` work without ``scoring=``.
        """
        from sklearn.metrics import accuracy_score, r2_score

        prediction = self.predict(X)
        if self.task in _LABELED_TASKS:
            return float(accuracy_score(target_array(y), prediction, sample_weight=sample_weight))
        return float(r2_score(target_array(y), prediction, sample_weight=sample_weight))

    @property
    def _estimator_type(self):
        return "classifier" if self.task in _LABELED_TASKS else "regressor"

    def __sklearn_tags__(self):
        """Tag a classification or ordinal model as a classifier, anything else as a regressor.

        The tag follows ``task``, so ``classifier()`` stratifies folds and the
        probability scorers read ``predict_proba``.
        """
        from sklearn.utils import ClassifierTags, RegressorTags

        tags = super().__sklearn_tags__()
        if self.task in _LABELED_TASKS:
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
            tags.target_tags.required = True
        else:
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
            tags.target_tags.required = True
        return tags

    def predict_score(self, X):
        """Expected class value, ``probabilities @ classes_``.

        Available for ``task="ordinal"``. ``predict`` still returns the most
        likely label.
        """
        if self.task != "ordinal":
            raise AshlarError("predict_score is the expected class value for an ordinal task.")
        probabilities = np.asarray(self.predict_proba(X), dtype=np.float64)
        values = np.asarray(self.classes_, dtype=np.float64)
        return probabilities @ values

    def predict_proba(self, X):
        if self.task not in _LABELED_TASKS:
            raise AshlarError("predict_proba is available for classification and ordinal classification.")
        raw = self._raw_from_frame(X)
        return np.asarray(_point_from_raw(self.task, self._policy(), raw, self.point_fn))

    def _require_distribution(self, name: str, *, density: bool = False):
        if self.task not in DISTRIBUTION_TASKS:
            raise AshlarError(f"{name} needs a distribution task: {', '.join(DISTRIBUTION_TASKS)}.")
        if density and self.task == "quantile":
            raise AshlarError(f"the quantile head has three levels, not a density, so it has no {name}.")

    def _target_affine(self):
        if self.target_mean_ is None:
            return 0.0, 1.0
        return float(self.target_mean_), float(self.target_std_)

    def sample(self, X, n: int = 100, seed: int = 0):
        """``n`` draws per row from the predictive distribution, shape ``(n, rows)``, on the scale of ``y``.

        Each draw picks one member uniformly, then draws from that member's
        distribution, so the draws follow the equal-weight member mixture.
        """
        self._require_distribution("sample", density=True)
        raw = np.asarray(self.predict_raw(X))
        mean, std = self._target_affine()
        key = jax.random.PRNGKey(int(seed))
        pieces = []
        for start in range(0, raw.shape[0], 4096):
            key, member_key, draw_key = jax.random.split(key, 3)
            block = jnp.asarray(raw[start : start + 4096])
            rows, members, width = block.shape
            choice = jax.random.randint(member_key, (int(n), rows), 0, members)
            picked = jnp.take_along_axis(block[None], choice[..., None, None], axis=2)[:, :, 0, :]
            decoded = decode(self.task, picked.reshape(int(n) * rows, width), 1)
            draws = sample_values(self.task, decoded, draw_key, 1).reshape(int(n), rows)
            pieces.append(np.asarray(draws, dtype=np.float64))
        return np.concatenate(pieces, axis=1) * std + mean

    def log_prob(self, X, y):
        """Log density of ``y`` under the equal-weight member mixture, per row, on the scale of ``y``."""
        self._require_distribution("log_prob", density=True)
        raw = jnp.asarray(self.predict_raw(X))
        mean, std = self._target_affine()
        z = (np.asarray(target_array(y), dtype=np.float64).reshape(-1) - mean) / std
        decoded, batch, members = _decode_members(self.task, raw)
        values = jnp.repeat(jnp.asarray(z, dtype=raw.dtype), members)[:, None]
        per_member = log_density(self.task, decoded, values).reshape(batch, members)
        mixed = jax.nn.logsumexp(per_member, axis=1) - jnp.log(float(members))
        return np.asarray(mixed, dtype=np.float64) - np.log(std)

    def predict_quantiles(self, X, levels=QUANTILE_LEVELS, n_samples: int = 1000, seed: int = 0):
        """Quantiles per row, shape ``(rows, len(levels))``, on the scale of ``y``.

        A density head reads them from ``n_samples`` draws of the member
        mixture. The quantile head has its three trained levels, 0.1, 0.5, and
        0.9, averaged across members, and no others.
        """
        self._require_distribution("predict_quantiles")
        levels = tuple(float(level) for level in np.atleast_1d(levels))
        mean, std = self._target_affine()
        if self.task == "quantile":
            if levels != tuple(QUANTILE_LEVELS):
                raise AshlarError(f"the quantile head was trained at {QUANTILE_LEVELS} only.")
            decoded, batch, members = _decode_members(self.task, jnp.asarray(self.predict_raw(X)))
            values = np.asarray(decoded, dtype=np.float64).reshape(batch, members, 3).mean(axis=1)
            return values * std + mean
        draws = self.sample(X, n=n_samples, seed=seed)
        return np.quantile(draws, levels, axis=0).T

    def predict_dist(self, X, policy: str | None = None):
        raw = self._raw_from_frame(X)
        return _dist_from_raw(self.task, self._policy() if policy is None else policy, raw, self.dist_fn)

    def predict_raw(self, X):
        """Head output with the member axis preserved."""
        return self._raw_from_frame(X)

    def save(self, path: str | Path, *, save_optimizer: bool = True):
        self._require_fitted()
        directory = Path(path)
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": __version__,
            "distribution": "ashlar-tabular",
            "import": "ashlar",
            "reduction": REDUCTION,
            "task": self.task,
            "backbone": dump_ref(self.backbone),
            "backbone_kwargs": (getattr(self, "resolved_", None) or self._resolved_training())["backbone_kwargs"],
            "head": dump_ref(self.head) if not isinstance(self.head, str) else self.head,
            "head_kwargs": _saved_head_kwargs(self),
            "objective": dump_ref(self.objective),
            "point_fn": dump_ref(self.point_fn),
            "dist_fn": dump_ref(self.dist_fn),
            "num_embeddings": self.num_embeddings,
            "num_embedding_dim": self.num_embedding_dim,
            "n_events": self.n_events_,
            "n_classes": self.n_classes_,
            "prediction_policy": self._policy(),
            "learning_rate": (getattr(self, "resolved_", None) or self._resolved_training())["learning_rate"],
            "learning_rate_schedule": self.learning_rate_schedule,
            "standardize_target": bool(self.standardize_target),
            "share_training_batches": bool(self.share_training_batches),
            "bottleneck": self._bottleneck_width(),
            "histogram_variance": None if self.histogram_variance is None else float(self.histogram_variance),
            "expectation_weight": self._expectation_weight(),
            "soft_stages": self._soft_stage_count(),
            "target_mean": _json_ready(getattr(self, "target_mean_", None)),
            "target_std": _json_ready(getattr(self, "target_std_", None)),
            "weight_decay": (getattr(self, "resolved_", None) or self._resolved_training())["weight_decay"],
            "feature_dropout": (getattr(self, "resolved_", None) or self._resolved_training())["feature_dropout"],
            "numeric_scaling": self.numeric_scaling,
            "batch_size": (getattr(self, "resolved_", None) or self._resolved_training())["batch_size"],
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "validation_size": self.validation_size,
            "split": self.split,
            "random_state": int(self.random_state),
            "class_weight": _json_class_weight(self.class_weight),
            "classes": None if self.classes_ is None else [_python_scalar(value) for value in self.classes_.tolist()],
            "categorical": self.categorical,
            "categorical_embeddings": self.categorical_embeddings,
            "set_features": self.set_features,
            "categorical_embedding_dim": int(self.categorical_embedding_dim),
            "categorical_embedding_hidden": bool(self.categorical_embedding_hidden),
            "set_max_length": int(self.set_max_length),
            "search": getattr(self, "search_", None),
            "train_key": [int(value) for value in np.asarray(self._train_key).tolist()],
            "schema": self.encoder_.to_json(),
            "history": self.history_,
            "resolved_backbone_kwargs": _resolved_backbone_kwargs(self.core_),
        }
        (directory / "metadata.json").write_text(json.dumps(payload, indent=2))
        save_leaves(directory / "core.eqx", self.core_)
        save_leaves(directory / "numeric.eqx", self.numeric_transform_)
        if save_optimizer:
            save_opt_state(directory, self.opt_state_)
        return str(directory)

    @classmethod
    def load(cls, path: str | Path):
        directory = Path(path)
        payload = json.loads((directory / "metadata.json").read_text())
        estimator = cls(
            task=payload["task"],
            backbone=load_ref(payload["backbone"]),
            backbone_kwargs=payload["backbone_kwargs"],
            head=load_ref(payload["head"]),
            head_kwargs=payload["head_kwargs"],
            objective=load_ref(payload["objective"]),
            point_fn=load_ref(payload["point_fn"]),
            dist_fn=load_ref(payload["dist_fn"]),
            num_embeddings=payload["num_embeddings"],
            num_embedding_dim=payload["num_embedding_dim"],
            n_events=None if payload["task"] in _LABELED_TASKS else payload["n_events"],
            validation_size=payload["validation_size"],
            split=payload["split"],
            batch_size=payload["batch_size"],
            max_epochs=payload["max_epochs"],
            patience=payload["patience"],
            learning_rate=payload["learning_rate"],
            learning_rate_schedule=payload.get("learning_rate_schedule", "constant"),
            standardize_target=bool(payload.get("standardize_target", False)),
            share_training_batches=bool(payload.get("share_training_batches", True)),
            bottleneck=payload.get("bottleneck"),
            histogram_variance=payload.get("histogram_variance"),
            expectation_weight=payload.get("expectation_weight"),
            soft_stages=payload.get("soft_stages"),
            weight_decay=payload["weight_decay"],
            feature_dropout=payload.get("feature_dropout", 0.0),
            numeric_scaling=payload.get("numeric_scaling", "standard"),
            prediction_policy=payload["prediction_policy"],
            class_weight=payload["class_weight"],
            categorical=payload.get("categorical"),
            categorical_embeddings=payload.get("categorical_embeddings"),
            set_features=payload.get("set_features"),
            categorical_embedding_dim=payload.get("categorical_embedding_dim", 8),
            categorical_embedding_hidden=payload.get("categorical_embedding_hidden", False),
            set_max_length=payload.get("set_max_length", 8),
            random_state=payload["random_state"],
        )
        encoder = FeatureEncoder.from_json(payload["schema"])
        key = jax.random.PRNGKey(0)
        template = estimator._build_core(encoder, payload["n_events"], payload["n_classes"], key)
        estimator.encoder_ = encoder
        estimator.embedded_columns_ = [column.name for column in encoder.embedding_columns]
        estimator.numeric_transform_ = load_leaves(directory / "numeric.eqx", encoder.numeric_transform)
        estimator.encoder_.numeric_transform = estimator.numeric_transform_
        estimator.core_ = load_leaves(directory / "core.eqx", template)
        estimator.classes_ = None if payload["classes"] is None else np.asarray(payload["classes"])
        estimator.n_events_ = payload["n_events"]
        estimator.n_classes_ = payload["n_classes"]
        estimator.history_ = payload["history"]
        estimator.target_mean_ = payload.get("target_mean")
        estimator.target_std_ = payload.get("target_std")
        estimator.reduction_ = payload["reduction"]
        estimator.n_features_in_ = len(encoder.feature_names)
        estimator.is_fitted_ = True
        opt_path = directory / "optimizer.eqx"
        if opt_path.exists():
            import equinox as eqx

            rate = float(estimator.learning_rate)
            if estimator.learning_rate_schedule in {"linear", "cosine"}:
                rate = lambda count, rate=rate: rate  # same state layout as the fitted schedule
            template_state = make_optimizer(rate, float(estimator.weight_decay)).init(
                eqx.filter(estimator.core_, eqx.is_array)
            )
            estimator.opt_state_ = load_opt_state(directory, template_state)
        else:
            estimator.opt_state_ = None
        estimator._train_key = jnp.asarray(payload["train_key"], dtype=jnp.uint32)
        estimator.resolved_ = {
            "learning_rate": float(payload["learning_rate"]),
            "weight_decay": float(payload["weight_decay"]),
            "feature_dropout": float(payload.get("feature_dropout", estimator.feature_dropout)),
            "batch_size": int(payload["batch_size"]),
            "backbone_kwargs": payload["backbone_kwargs"],
        }
        estimator.search_ = payload.get("search")
        return estimator

    def _require_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise AshlarError("this estimator is not fitted.")


_FORWARD = None


def _jit_forward():
    """One compiled inference pass, shared by every fitted core."""
    global _FORWARD
    if _FORWARD is None:
        import equinox as eqx

        @eqx.filter_jit
        def forward(core, *inputs):
            return core(*inputs, key=None)

        _FORWARD = forward
    return _FORWARD


def _split_indices(n_rows, y, *, mode, validation_size, groups, random_state, task):
    rng = np.random.default_rng(random_state)
    if mode == "temporal":
        n_val = min(max(int(round(validation_size * n_rows)), 1), n_rows - 1)
        return np.arange(n_rows - n_val), np.arange(n_rows - n_val, n_rows)
    if mode == "group":
        if groups is None:
            raise AshlarError("split='group' requires groups aligned with the rows of X.")
        group_ids = np.asarray(groups)
        if len(group_ids) != n_rows:
            raise AshlarError("groups must have one entry per row.")
        unique = np.unique(group_ids)
        if len(unique) < 2:
            raise AshlarError("split='group' requires at least two groups.")
        order = rng.permutation(unique)
        n_val = min(max(int(round(validation_size * len(unique))), 1), len(unique) - 1)
        held = set(order[:n_val].tolist())
        val_idx = np.asarray([index for index, group in enumerate(group_ids.tolist()) if group in held], dtype=int)
        train_idx = np.asarray([index for index, group in enumerate(group_ids.tolist()) if group not in held], dtype=int)
        return train_idx, val_idx
    if task in _LABELED_TASKS:
        return _stratified_indices(np.asarray(y).reshape(-1), validation_size, random_state)
    order = rng.permutation(n_rows)
    n_val = min(max(int(round(validation_size * n_rows)), 1), n_rows - 1)
    return order[n_val:], order[:n_val]


def _stratified_indices(labels, validation_size, random_state):
    """Stratified iid split. A class with fewer than two rows stays in training.

    ``validation_size`` is the fraction of the remaining rows. The rare rows
    are not eligible for the validation partition, so the held-out fraction of
    the full sample is slightly smaller. A remainder that sklearn still cannot
    stratify raises.
    """
    from sklearn.model_selection import train_test_split

    labels = np.asarray(labels).reshape(-1)
    indices = np.arange(len(labels))
    _classes, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    rare = counts[inverse] < 2
    if not np.any(rare):
        return _sklearn_stratified(indices, labels, validation_size, random_state, train_test_split)

    common = indices[~rare]
    if len(common) < 2:
        raise AshlarError(
            "stratified iid split failed. Every class has fewer than two rows, so there is nothing left to stratify."
        )
    if len(np.unique(labels[~rare])) < 2:
        raise AshlarError(
            "stratified iid split failed. Only one class remains after classes with fewer than two rows stay in training."
        )
    train_common, val_idx = _sklearn_stratified(
        common, labels[common], validation_size, random_state, train_test_split
    )
    train_idx = np.concatenate([train_common, indices[rare]])
    return train_idx, val_idx


def _sklearn_stratified(indices, labels, validation_size, random_state, train_test_split):
    try:
        train_idx, val_idx = train_test_split(
            indices,
            test_size=validation_size,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError as exc:
        raise AshlarError(
            "stratified iid split failed. Use split='temporal', an explicit validation tuple, "
            "or a larger sample with at least two rows per class."
        ) from exc
    return np.asarray(train_idx), np.asarray(val_idx)


def _take_rows(names, columns, indices):
    return {name: np.asarray(values)[indices] for name, values in zip(names, columns, strict=True)}


def _regression_target(y, n_events):
    array = np.asarray(y, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if n_events is not None and n_events != 1 and (array.ndim != 2 or array.shape[1] != n_events):
        raise ShapeError(f"y must have shape (n_rows, {n_events}); received {array.shape}.")
    return array


def _device_targets(task, y, weights):
    w = jnp.asarray(weights, dtype=jnp.float32)
    if task in _LABELED_TASKS:
        return jnp.asarray(y, dtype=jnp.int32), w
    return jnp.asarray(y, dtype=jnp.float32), w


def _ordinal_label_vector(y):
    array = np.asarray(y)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.dtype == object or np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(array.dtype, np.number):
        raise AshlarError(
            "ordinal labels must be a vector of real numbers. The class order is their sorted numeric order, and predict_score is the expected label."
        )
    if not np.isfinite(np.asarray(array, dtype=np.float64)).all():
        raise AshlarError("ordinal labels must be finite.")
    return array


def _encode_known_labels(y, classes):
    y = np.asarray(y)
    index = np.searchsorted(classes, y)
    index = np.clip(index, 0, len(classes) - 1)
    if np.any(classes[index] != y):
        missing = np.unique(y[classes[index] != y]).tolist()
        raise AshlarError(f"validation labels {missing} were not present in the training partition.")
    return index.astype(np.int32)


def _apply_class_weight(y_original, y_encoded, n_classes, class_weight, sample_weight):
    labels = y_original if isinstance(class_weight, dict) else y_encoded
    multiplier = class_weight_multiplier(np.asarray(labels), n_classes, class_weight)
    if not np.isfinite(multiplier).all() or (multiplier < 0).any():
        raise WeightError("class-weight multipliers must be finite and non-negative.")
    return np.asarray(sample_weight, dtype=np.float64) * multiplier


def _python_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_class_weight(class_weight):
    if isinstance(class_weight, np.ndarray):
        return class_weight.tolist()
    return class_weight


def _saved_head_kwargs(estimator):
    payload = dict(estimator.head_kwargs or {})
    prior = getattr(estimator, "class_prior_", None)
    if prior is not None:
        payload["class_prior"] = [float(value) for value in np.asarray(prior, dtype=np.float64).tolist()]
    return payload or None


def _category_tables(columns, dim: int, key, *, set_encoder: bool = False, hidden: bool = False):
    if not columns:
        return None
    keys = jax.random.split(key, len(columns))
    tables = []
    layers = []
    for column, piece in zip(columns, keys, strict=True):
        vocab = 1 + len(column.categories)
        table_key, weight_key = jax.random.split(piece)
        tables.append(jax.random.normal(table_key, (vocab, dim), dtype=jnp.float32) * jnp.asarray(0.02, dtype=jnp.float32))
        if hidden:
            weight = jax.random.normal(weight_key, (dim, dim), dtype=jnp.float32) * jnp.asarray(dim**-0.5, dtype=jnp.float32)
            layers.append((weight, jnp.zeros((dim,), dtype=jnp.float32)))
    if set_encoder:
        return SetEmbeddings(tables=tuple(tables))
    return CategoricalEmbeddings(tables=tuple(tables), hidden=tuple(layers) if hidden else None)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    raise AshlarError(f"search parameters must be JSON values; received {value!r}.")


def _resolved_backbone_kwargs(core):
    backbone = core.backbone
    payload = {"k": int(backbone.k), "d_out": int(backbone.d_out), "dropout": float(backbone.dropout_rate)}
    if hasattr(backbone, "activation"):
        payload["activation"] = backbone.activation
    return payload
