"""Schema fitted only on the training partition.

Numeric imputation and standardization are a pure Equinox module, so a later
JAX computation can differentiate through them. Category lookup and one-hot
encoding stay on the host: gradients do not flow through a string or an
unknown-category decision.

Missing numeric values use the training median. The default map,
``numeric_scaling='robust'``, centers that filled column at its median,
divides by its interquartile range, and applies a smooth clip that saturates
at ±3. An interquartile range of 0 uses half the range of the filled column,
and a constant column uses divisor 1. ``numeric_scaling='standard'`` is the
mean and population standard deviation of the filled column, with no clip. A
population standard deviation below ``1e-6`` is replaced by 1. For robust
scaling the divisor is stored in the same ``std`` slot, and the mean is still
recorded. ``numeric_scaling='quantile'`` replaces that map: the same
median-filled training rows are fit with a ``QuantileTransformer`` to a
normal margin.

``numeric_clip`` sizes the robust clip, in interquartile units (3 by
default). ``numeric_clip=None`` turns every clip off, so values past the
training range keep moving: robust scaling becomes linear, and the quantile
maps continue past their outermost knots on a straight line, at the slope a
normal margin with the column's interquartile range would have, instead of
stopping at ±5.2. Standard scaling never clips.

``FeatureEncoder.fit(..., embeddings=None)`` learns an embedding for every
categorical column that is not a set feature. ``embeddings=[]`` one-hots
those columns. A non-empty list embeds the named columns and one-hots the
rest. Missing categorical values and categories absent from the training
vocabulary become index 0, whether that index is the unknown embedding row
or the one-hot unknown bucket. Known categories occupy indices ``1..V``. An
empty string is a real category, not a missing value.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox import Module
from sklearn.preprocessing import QuantileTransformer

from ashlar.errors import AshlarError

_QUANTILE_COUNT = 1000
_BOUNDS_THRESHOLD = 1e-7
_ROBUST_CLIP = 3.0
# Interquartile range of a standard normal, 2 * ppf(0.75): the quantile maps' linear tails use 1.349 / IQR.
_NORMAL_IQR = 1.3489795003921634
NUMERIC_SCALINGS = ("standard", "robust", "quantile", "quantile_tabr")
# scipy.stats.norm.ppf at sklearn's BOUNDS_THRESHOLD halo. JAX's float32 ppf
# of that edge is about 0.03 low, so the clip is the scipy value.
_CLIP_MIN = -5.199337582605575
_CLIP_MAX = 5.19933758270342


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    role: str
    categories: tuple[str, ...] = ()
    mean: float | None = None
    std: float | None = None
    median: float | None = None
    encoding: str = "onehot"
    max_length: int = 1


@dataclass(frozen=True)
class EncodedBatch:
    """Host-encoded arrays. Numeric NaNs are still present.

    ``category_index`` is ``(row, embedding column)`` with 0 = unknown.
    ``set_tokens`` and ``set_mask`` are ``(row, set column, position)``.
    Padding and unknown tokens share index 0; padding has mask 0 so it does
    not enter the mean.
    """

    numeric_raw: jax.Array
    categorical_onehot: jax.Array
    category_index: jax.Array
    set_tokens: jax.Array
    set_mask: jax.Array


class NumericTransform(Module):
    """Impute, then robust-scale, standardize, or apply a fitted normal quantile map.

    ``median``, ``mean``, and ``std`` have shape ``(n_numeric,)``. For
    ``scaling='robust'``, ``std`` stores the interquartile divisor and the
    forward centers on ``median``. For ``scaling='standard'``, ``std`` is the
    population standard deviation and the forward centers on ``mean``.
    Quantile knots have shape ``(n_quantiles, n_numeric)`` and are unused
    unless ``scaling`` is ``quantile`` or ``quantile_tabr``. The quantile
    forward follows ``QuantileTransformer._transform_col`` for
    ``output_distribution='normal'``. ``clip`` is the robust soft clip in
    interquartile units; ``None`` removes it, and gives the quantile maps
    linear tails past their outermost knots.
    """

    median: jax.Array
    mean: jax.Array
    std: jax.Array
    quantiles: jax.Array
    references: jax.Array
    scaling: str = eqx.field(static=True)
    clip: float | None = eqx.field(static=True, default=3.0)

    def __call__(self, raw):
        if raw.shape[-1] != self.mean.shape[0]:
            raise AshlarError(
                f"numeric transform expected {self.mean.shape[0]} columns; received {raw.shape[-1]}."
            )
        if self.scaling == "quantile_tabr":
            values = raw
        else:
            values = jnp.where(jnp.isnan(raw), self.median, raw)
        if self.scaling in {"quantile", "quantile_tabr"} and self.mean.shape[0] > 0:
            return _quantile_normal(values, self.quantiles, self.references, linear_tails=self.clip is None)
        if self.scaling == "robust":
            scaled = (values - self.median) / self.std
            if self.clip is None:
                return scaled
            return scaled / jnp.sqrt(1.0 + (scaled * scaled) / (self.clip * self.clip))
        return (values - self.mean) / self.std


def _normal_clips() -> tuple[float, float]:
    return _CLIP_MIN, _CLIP_MAX


def check_numeric_clip(numeric_clip, numeric_scaling: str):
    """Raise unless ``numeric_clip`` is ``None`` or a positive number that this scaling can use."""
    if numeric_clip is None:
        return
    if isinstance(numeric_clip, bool) or not isinstance(numeric_clip, (int, float, np.integer, np.floating)):
        raise AshlarError("numeric_clip must be a positive number or None.")
    if not np.isfinite(numeric_clip) or numeric_clip <= 0:
        raise AshlarError("numeric_clip must be a positive number or None.")
    if numeric_scaling != "robust" and float(numeric_clip) != _ROBUST_CLIP:
        raise AshlarError(
            f"numeric_clip sizes the robust soft clip. With numeric_scaling={numeric_scaling!r} it is 3.0, "
            "the scaling's own edge, or None, which lets values past the training range keep moving."
        )


def _tail_slope(knots, references):
    """Slope of a column's linear tails: a normal margin's, ``1.349 / IQR``, from the knots.

    A zero interquartile range uses half the range, and a constant column uses 1, as ``_robust_divisor`` does.
    """
    lower = jnp.interp(0.25, references, knots)
    upper = jnp.interp(0.75, references, knots)
    divisor = upper - lower
    divisor = jnp.where(divisor > 0, divisor, 0.5 * (knots[-1] - knots[0]))
    divisor = jnp.where(divisor > 0, divisor, 1.0)
    return _NORMAL_IQR / divisor


def _quantile_normal(filled, quantiles, references, linear_tails: bool = False):
    """Map ``(row, column)`` through per-column knots onto a clipped normal.

    ``linear_tails`` continues the map past the outermost knots on a straight
    line from the clip, at ``_tail_slope``, so a value beyond the training range
    still moves.
    """
    clip_min, clip_max = _normal_clips()

    def column(values, knots):
        lower_x = knots[0]
        upper_x = knots[-1]
        ascending = jnp.interp(values, knots, references)
        descending = jnp.interp(-values, -knots[::-1], -references[::-1])
        mapped = 0.5 * (ascending - descending)
        # Float32 cannot add 1e-7 to values at magnitude 1 or larger, so the
        # sklearn halo is widened to one ulp when that is the coarser step.
        lower_tol = jnp.maximum(jnp.asarray(_BOUNDS_THRESHOLD, dtype=values.dtype), jnp.spacing(lower_x))
        upper_tol = jnp.maximum(jnp.asarray(_BOUNDS_THRESHOLD, dtype=values.dtype), jnp.spacing(upper_x))
        lower = values - lower_tol < lower_x
        upper = values + upper_tol > upper_x
        mapped = jnp.where(upper, jnp.asarray(1.0, dtype=mapped.dtype), mapped)
        mapped = jnp.where(lower, jnp.asarray(0.0, dtype=mapped.dtype), mapped)
        normal = jnp.clip(jax.scipy.stats.norm.ppf(mapped), clip_min, clip_max)
        if not linear_tails:
            return normal
        slope = _tail_slope(knots, references)
        normal = jnp.where(values > upper_x, clip_max + (values - upper_x) * slope, normal)
        return jnp.where(values < lower_x, clip_min - (lower_x - values) * slope, normal)

    return jax.vmap(column, in_axes=(1, 1), out_axes=1)(filled, quantiles)


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return True
    except TypeError:
        return False
    return False


def _stringify(value) -> str:
    return str(value)


def frame_columns(data, names: list[str] | None = None):
    """Return ``(column_names, columns)`` where each column is an object array.

    Accepts a NumPy array, a pandas DataFrame, a Polars DataFrame, or a mapping.
    """
    if hasattr(data, "columns") and type(data).__module__.startswith("pandas"):
        frame_names = [str(name) for name in data.columns]
        columns = [np.asarray(data[name].to_numpy()) for name in data.columns]
        return frame_names, columns
    if hasattr(data, "columns") and type(data).__module__.startswith("polars"):
        frame_names = [str(name) for name in data.columns]
        columns = []
        for name in data.columns:
            series = data.get_column(name)
            if series.dtype.is_numeric():
                columns.append(np.asarray(series.to_numpy(), dtype=np.float64))
            else:
                columns.append(np.asarray(series.to_list(), dtype=object))
        return frame_names, columns
    if isinstance(data, dict):
        frame_names = [str(name) for name in data.keys()]
        columns = [np.asarray(data[name]) for name in data.keys()]
        return frame_names, columns
    array = np.asarray(data)
    if array.ndim != 2:
        raise AshlarError(f"array features must have shape (n_rows, n_columns); received {array.shape}.")
    frame_names = list(names) if names is not None else [f"x{i}" for i in range(array.shape[1])]
    if len(frame_names) != array.shape[1]:
        raise AshlarError("feature names do not match the number of columns.")
    columns = [array[:, index] for index in range(array.shape[1])]
    return frame_names, columns


def target_array(y) -> np.ndarray:
    if type(y).__module__.startswith("pandas"):
        return np.asarray(y)
    if type(y).__module__.startswith("polars"):
        return np.asarray(y.to_list(), dtype=object)
    return np.asarray(y)


def _numeric_column(values: np.ndarray) -> bool:
    if values.dtype == object:
        return False
    return np.issubdtype(values.dtype, np.number) and not np.issubdtype(values.dtype, np.bool_)


class FeatureEncoder:
    """Host schema plus a differentiable numeric transform."""

    def __init__(
        self,
        columns: tuple[ColumnSpec, ...],
        categorical: set[str] | None = None,
        *,
        numeric_scaling: str = "robust",
        quantiles: np.ndarray | None = None,
        references: np.ndarray | None = None,
        numeric_clip: float | None = _ROBUST_CLIP,
    ):
        if numeric_scaling not in NUMERIC_SCALINGS:
            raise AshlarError(
                "numeric_scaling must be 'standard', 'robust', 'quantile', or 'quantile_tabr'."
            )
        check_numeric_clip(numeric_clip, numeric_scaling)
        self.columns = columns
        self.categorical = set() if categorical is None else set(categorical)
        self.numeric_scaling = numeric_scaling
        self.numeric_clip = None if numeric_clip is None else float(numeric_clip)
        numeric = [column for column in columns if column.role == "numeric"]
        if numeric_scaling in {"quantile", "quantile_tabr"}:
            if quantiles is None or references is None:
                raise AshlarError("quantile scaling requires knots fit on the training partition.")
            quantile_knots = jnp.asarray(quantiles, dtype=jnp.float32)
            quantile_references = jnp.asarray(references, dtype=jnp.float32)
        else:
            quantile_knots = jnp.zeros((1, len(numeric)), dtype=jnp.float32)
            quantile_references = jnp.zeros((1,), dtype=jnp.float32)
        self.numeric_transform = NumericTransform(
            median=jnp.asarray([column.median for column in numeric], dtype=jnp.float32),
            mean=jnp.asarray([column.mean for column in numeric], dtype=jnp.float32),
            std=jnp.asarray([column.std for column in numeric], dtype=jnp.float32),
            quantiles=quantile_knots,
            references=quantile_references,
            scaling=numeric_scaling,
            clip=self.numeric_clip,
        )

    @property
    def feature_names(self) -> list[str]:
        return [column.name for column in self.columns]

    @property
    def n_numeric(self) -> int:
        return sum(column.role == "numeric" for column in self.columns)

    @property
    def categorical_widths(self) -> list[int]:
        return [
            1 + len(column.categories)
            for column in self.columns
            if column.role == "categorical" and column.encoding == "onehot"
        ]

    @property
    def embedding_columns(self) -> list[ColumnSpec]:
        return [column for column in self.columns if column.encoding == "embedding"]

    @property
    def set_columns(self) -> list[ColumnSpec]:
        return [column for column in self.columns if column.encoding == "set"]

    @property
    def set_length(self) -> int:
        if not self.set_columns:
            return 1
        return max(column.max_length for column in self.set_columns)

    def feature_chunks(self, numeric_embedding_dim: int | None, categorical_embedding_dim: int) -> list[int]:
        """Width of each field token after numeric and categorical encodings are concatenated."""
        chunks = []
        for column in self.columns:
            if column.role == "numeric":
                chunks.append(1 if numeric_embedding_dim is None else int(numeric_embedding_dim))
            elif column.encoding == "onehot":
                chunks.append(1 + len(column.categories))
            else:
                chunks.append(int(categorical_embedding_dim))
        return chunks

    @classmethod
    def fit(
        cls,
        data,
        *,
        categorical: list[str] | None = None,
        embeddings: list[str] | None = None,
        set_features: list[str] | None = None,
        set_max_length: int = 8,
        numeric_scaling: str = "robust",
        numeric_clip: float | None = _ROBUST_CLIP,
    ) -> "FeatureEncoder":
        names, columns = frame_columns(data)
        requested = set(categorical or [])
        embed_all = embeddings is None
        embedded = set() if embeddings is None else set(embeddings)
        sets = set(set_features or [])
        if set_max_length < 1:
            raise AshlarError("set_max_length must be positive.")
        if numeric_scaling not in NUMERIC_SCALINGS:
            raise AshlarError(
                "numeric_scaling must be 'standard', 'robust', 'quantile', or 'quantile_tabr'."
            )
        check_numeric_clip(numeric_clip, numeric_scaling)
        overlap = embedded.intersection(sets)
        if overlap:
            raise AshlarError(
                f"columns {sorted(overlap)} cannot be both a learned embedding and a set feature."
            )
        unknown = requested.union(embedded).union(sets).difference(names)
        if unknown:
            raise AshlarError(f"categorical columns {sorted(unknown)} are not in the frame.")
        specs = []
        for name, values in zip(names, columns, strict=True):
            categorical_column = (
                name in requested or name in embedded or name in sets or not _numeric_column(values)
            )
            if not categorical_column:
                numeric = np.asarray(values, dtype=np.float64)
                finite = np.isfinite(numeric)
                median = float(np.median(numeric[finite])) if finite.any() else 0.0
                filled = np.where(finite, numeric, median)
                mean = float(filled.mean()) if len(filled) else 0.0
                std = float(filled.std()) if len(filled) else 1.0
                if std < 1e-6:
                    std = 1.0
                if numeric_scaling == "robust":
                    std = _robust_divisor(filled)
                specs.append(ColumnSpec(name=name, role="numeric", mean=mean, std=std, median=median, encoding="numeric"))
            else:
                if name in sets:
                    encoding = "set"
                elif embed_all or name in embedded:
                    encoding = "embedding"
                else:
                    encoding = "onehot"
                categories = []
                seen = set()
                lengths = []
                for value in values.tolist():
                    tokens = _tokens(value) if encoding == "set" else ([] if _is_missing(value) else [_stringify(value)])
                    lengths.append(len(tokens))
                    for text in tokens:
                        if text not in seen:
                            seen.add(text)
                            categories.append(text)
                if not categories:
                    raise AshlarError(f"categorical column {name!r} has no observed categories in the training partition.")
                max_length = 1
                if encoding == "set":
                    observed = max(lengths) if lengths else 1
                    max_length = max(1, min(int(set_max_length), observed))
                specs.append(
                    ColumnSpec(
                        name=name,
                        role="categorical",
                        categories=tuple(categories),
                        encoding=encoding,
                        max_length=max_length,
                    )
                )
        if not specs:
            raise AshlarError("at least one feature column is required.")
        quantiles, references = (
            _fit_quantile_knots(names, columns, specs, numeric_scaling)
            if numeric_scaling in {"quantile", "quantile_tabr"}
            else (None, None)
        )
        return cls(
            tuple(specs),
            categorical=requested,
            numeric_scaling=numeric_scaling,
            quantiles=quantiles,
            references=references,
            numeric_clip=numeric_clip,
        )

    def transform(self, data) -> EncodedBatch:
        """Encode a frame. ``numeric_raw`` still contains NaNs.

        Call ``numeric_transform`` to impute and scale inside JAX.
        One-hot columns, learned-embedding indices, and set tokens are separate
        because they are different kinds of category.
        """
        names, columns = frame_columns(data)
        if names != self.feature_names:
            raise AshlarError(
                "feature columns do not match the fitted schema. "
                f"Expected {self.feature_names}; received {names}."
            )
        n_rows = len(columns[0]) if columns else 0
        numeric = []
        categorical = []
        embedding_indices = []
        set_tokens = []
        set_masks = []
        length = self.set_length
        for column, values in zip(self.columns, columns, strict=True):
            if len(values) != n_rows:
                raise AshlarError("feature columns have different lengths.")
            if column.role == "numeric":
                numeric.append(np.asarray(values, dtype=np.float32))
                continue
            lookup = {category: index + 1 for index, category in enumerate(column.categories)}
            if column.encoding == "onehot":
                indices = np.zeros(n_rows, dtype=np.int32)
                for row, value in enumerate(values.tolist()):
                    if _is_missing(value):
                        continue
                    indices[row] = lookup.get(_stringify(value), 0)
                onehot = np.zeros((n_rows, 1 + len(column.categories)), dtype=np.float32)
                onehot[np.arange(n_rows), indices] = 1.0
                categorical.append(onehot)
            elif column.encoding == "embedding":
                indices = np.zeros(n_rows, dtype=np.int32)
                for row, value in enumerate(values.tolist()):
                    if _is_missing(value):
                        continue
                    indices[row] = lookup.get(_stringify(value), 0)
                embedding_indices.append(indices)
            else:
                tokens = np.zeros((n_rows, length), dtype=np.int32)
                mask = np.zeros((n_rows, length), dtype=np.float32)
                for row, value in enumerate(values.tolist()):
                    observed = _tokens(value)[: column.max_length]
                    for position, text in enumerate(observed):
                        tokens[row, position] = lookup.get(text, 0)
                        mask[row, position] = 1.0
                set_tokens.append(tokens)
                set_masks.append(mask)
        numeric_raw = (
            np.column_stack(numeric).astype(np.float32) if numeric else np.zeros((n_rows, 0), dtype=np.float32)
        )
        categorical_onehot = (
            np.concatenate(categorical, axis=1).astype(np.float32)
            if categorical
            else np.zeros((n_rows, 0), dtype=np.float32)
        )
        category_index = (
            np.column_stack(embedding_indices).astype(np.int32)
            if embedding_indices
            else np.zeros((n_rows, 0), dtype=np.int32)
        )
        token_array = (
            np.stack(set_tokens, axis=1).astype(np.int32)
            if set_tokens
            else np.zeros((n_rows, 0, length), dtype=np.int32)
        )
        mask_array = (
            np.stack(set_masks, axis=1).astype(np.float32)
            if set_masks
            else np.zeros((n_rows, 0, length), dtype=np.float32)
        )
        return EncodedBatch(
            numeric_raw=jnp.asarray(numeric_raw),
            categorical_onehot=jnp.asarray(categorical_onehot),
            category_index=jnp.asarray(category_index),
            set_tokens=jnp.asarray(token_array),
            set_mask=jnp.asarray(mask_array),
        )

    def to_json(self) -> dict:
        payload = {
            "columns": [
                {
                    "name": column.name,
                    "role": column.role,
                    "categories": list(column.categories),
                    "mean": column.mean,
                    "std": column.std,
                    "median": column.median,
                    "encoding": column.encoding,
                    "max_length": column.max_length,
                }
                for column in self.columns
            ],
            "categorical": sorted(self.categorical),
            "numeric_scaling": self.numeric_scaling,
            "numeric_clip": self.numeric_clip,
        }
        if self.numeric_scaling in {"quantile", "quantile_tabr"}:
            payload["quantiles"] = np.asarray(self.numeric_transform.quantiles).tolist()
            payload["references"] = np.asarray(self.numeric_transform.references).tolist()
        return payload

    @classmethod
    def from_json(cls, payload: dict) -> "FeatureEncoder":
        columns = tuple(
            ColumnSpec(
                name=item["name"],
                role=item["role"],
                categories=tuple(item["categories"]),
                mean=item["mean"],
                std=item["std"],
                median=item["median"],
                encoding=item.get("encoding", "onehot"),
                max_length=int(item.get("max_length", 1)),
            )
            for item in payload["columns"]
        )
        return cls(
            columns,
            categorical=set(payload["categorical"]),
            numeric_scaling=payload.get("numeric_scaling", "standard"),
            quantiles=None if payload.get("quantiles") is None else np.asarray(payload["quantiles"], dtype=np.float64),
            references=None if payload.get("references") is None else np.asarray(payload["references"], dtype=np.float64),
            # A schema saved before numeric_clip existed used the robust clip of 3 and the quantile maps' edge.
            numeric_clip=payload.get("numeric_clip", _ROBUST_CLIP),
        )


def _robust_divisor(filled: np.ndarray) -> float:
    """Interquartile range of a median-filled training column.

    A zero interquartile range uses half the range. A constant column uses 1.
    The centered values are already 0, so the choice only keeps the divisor finite.
    """
    if filled.size == 0:
        return 1.0
    column = np.asarray(filled, dtype=np.float64)
    lower, upper = np.quantile(column, [0.25, 0.75], method="linear")
    divisor = float(upper - lower)
    if divisor == 0.0:
        divisor = 0.5 * float(column.max() - column.min())
    if divisor == 0.0:
        return 1.0
    return divisor


def _fit_quantile_knots(names, columns, specs, mode: str):
    """Fit a normal quantile map. It replaces mean/std.

    ``quantile`` median-fills, then uses ``min(1000, n_rows)`` knots and no
    noise. ``quantile_tabr`` is PyTabKit's ``TabrQuantileTransformer``:
    ``max(min(n_rows // 30, 1000), 10)`` knots, subsample ``1_000_000_000``,
    and Gaussian noise on the fit matrix only, scale
    ``1e-3 / max(column_std, 1e-3)``, ``random_state=0``. That path does not
    impute and does not z-score afterwards.
    """
    by_name = dict(zip(names, columns, strict=True))
    raw_columns = []
    for spec in specs:
        if spec.role != "numeric":
            continue
        numeric = np.asarray(by_name[spec.name], dtype=np.float64)
        if mode == "quantile_tabr" and not np.isfinite(numeric).all():
            raise AshlarError("quantile_tabr does not impute. Numeric training columns must be finite.")
        finite = np.isfinite(numeric)
        raw_columns.append(np.where(finite, numeric, spec.median))
    if not raw_columns:
        return None, None
    matrix = np.column_stack(raw_columns)
    if mode == "quantile_tabr":
        stds = np.std(matrix, axis=0, keepdims=True)
        noise_std = 1e-3 / np.maximum(stds, 1e-3)
        rng = np.random.default_rng(0)
        matrix = matrix + noise_std * rng.standard_normal(matrix.shape)
        n_quantiles = max(min(matrix.shape[0] // 30, _QUANTILE_COUNT), 10)
        subsample = 1_000_000_000
    else:
        n_quantiles = min(_QUANTILE_COUNT, matrix.shape[0])
        subsample = int(matrix.shape[0])
    transformer = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution="normal",
        subsample=subsample,
        random_state=0,
        copy=True,
    )
    transformer.fit(matrix)
    return np.asarray(transformer.quantiles_, dtype=np.float64), np.asarray(transformer.references_, dtype=np.float64)


def _tokens(value) -> list[str]:
    """Tokens of a set-valued cell. A string is one token. A list is many."""
    if _is_missing(value):
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if not _is_missing(item)]
    if isinstance(value, np.ndarray) and value.ndim > 0:
        return [str(item) for item in value.tolist() if not _is_missing(item)]
    return [str(value)]
