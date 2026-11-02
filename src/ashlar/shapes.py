"""Additive models for every output of a head, and the readout of their terms.

``ASRRegressor(task=..., backbone="additive")`` gives each column of ``X`` a
small network of its own (``ashlar.layers.AdditiveBackbone``), and
``backbone_kwargs={"pairs": [("hour", "workingday")]}`` adds one network per
pair of columns. The networks meet only in the head, which is linear in
them, so each of the head's outputs is an intercept plus one function per
term:

    output_p(x) = b_p + sum over terms t of f_tp(x_t)

on the scale the head's distribution reads the output on: a location, a
raw scale (through a softplus), a log mean, a logit. That is a generalized
additive model for location, scale, and shape (GAMLSS; Rigby and
Stasinopoulos, 2005) with networks for smoothers, on every built-in head
but the joint flow and the chain, which read the hidden layer through
networks of their own. A monotone feature keeps its own rising shape and a
grouping column of ``random_effects`` its offsets; both are additive too,
and the readout lists them as terms.

``model.shape_functions(X)`` returns each term's part of each output, row by
row, as a ``ShapeFunctions``.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ashlar.errors import AshlarError

# Heads that read the hidden layer through a network of their own: their outputs are not sums of terms.
ADDITIVE_REFUSED = ("multivariate_flow", "chain")
# Rows per pass when reading the terms.
_READ_ROWS = 4096


def check_pairs(pairs) -> tuple[tuple[str, str], ...]:
    """The additive backbone's ``pairs``, checked: ``((column, column), ...)``."""
    if pairs is None:
        return ()
    if isinstance(pairs, (str, bytes)) or not isinstance(pairs, (list, tuple)):
        raise AshlarError("pairs must be a list of [column, column] pairs, such as [['hour', 'workingday']].")
    checked = []
    seen = set()
    for pair in pairs:
        if (
            isinstance(pair, (str, bytes))
            or not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or not all(isinstance(name, str) for name in pair)
        ):
            raise AshlarError(f"each pair is two column names; received {pair!r}.")
        first, second = pair
        if first == second:
            raise AshlarError(f"a pair needs two different columns; received {list(pair)!r}.")
        key = frozenset(pair)
        if key in seen:
            raise AshlarError(f"the pair {list(pair)!r} is listed twice.")
        seen.add(key)
        checked.append((first, second))
    return tuple(checked)


def additive_terms(sources, pairs, excluded=frozenset()):
    """``(terms, columns)``: the fields each term reads, and its columns of ``X``.

    ``sources`` names the column of ``X`` behind each field, in the core's
    field order. Each column not in ``excluded`` is one term over all its
    fields, in order of first appearance, and each pair one more over both
    columns' fields.
    """
    fields: dict[str, list[int]] = {}
    for index, source in enumerate(sources):
        if source in excluded:
            continue
        fields.setdefault(source, []).append(index)
    if not fields:
        raise AshlarError("the additive backbone has no feature to read: every column is monotone or a grouping column.")
    terms = [tuple(indices) for indices in fields.values()]
    columns = [(source,) for source in fields]
    for first, second in pairs:
        missing = [name for name in (first, second) if name not in fields]
        if missing:
            raise AshlarError(
                f"pairs name columns of X that the additive backbone reads; {missing} are not "
                "(a monotone feature, a grouping column without an embedding, or not in X)."
            )
        terms.append(tuple(fields[first]) + tuple(fields[second]))
        columns.append((first, second))
    return tuple(terms), tuple(columns)


def _numbered(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index}" for index in range(1, count + 1))


def output_names(task: str, width: int, *, classes=None, cure: bool = False) -> tuple[str, ...]:
    """A name for each raw output of a head, in order; ``output_<i>`` where the layout is not known."""
    from ashlar.distributional import MIXTURE_COMPONENTS, SPLINE_PIECES, flow_parameter_count

    width = int(width)
    body = width - int(bool(cure))
    names: tuple[str, ...] | None = None
    if task == "regression":
        names = ("mean",) if body == 1 else _numbered("mean", body)
    elif task == "gaussian" and body == 2:
        names = ("loc", "scale")
    elif task == "student_t" and body == 3:
        names = ("loc", "scale", "df")
    elif task == "mixture" and body == 3 * MIXTURE_COMPONENTS:
        names = tuple(
            f"{part}_{component}" for component in range(1, MIXTURE_COMPONENTS + 1) for part in ("loc", "scale", "weight")
        )
    elif task == "flow" and body == flow_parameter_count() + 2:
        names = _numbered("spline", flow_parameter_count()) + ("shift", "scale")
    elif task == "histogram":
        names = _numbered("bin", body)
    elif task == "quantile" and body == 3:
        names = ("low", "middle_gap", "high_gap")
    elif task == "quantile_spline" and body == 1 + 2 * SPLINE_PIECES:
        names = ("median",) + _numbered("upper_slope", SPLINE_PIECES) + _numbered("lower_slope", SPLINE_PIECES)
    elif task == "poisson" and body in (1, 2):
        names = ("log_mean",) + (("zero_logit",) if body == 2 else ())
    elif task == "negative_binomial" and body in (2, 3):
        names = ("log_mean", "dispersion") + (("zero_logit",) if body == 3 else ())
    elif task == "hurdle" and body == 3:
        names = ("zero_logit", "severity_loc", "severity_scale")
    elif task == "normal" and body % 2 == 0:
        names = ("loc", "scale") if body == 2 else tuple(f"{part}_{e}" for e in range(1, body // 2 + 1) for part in ("loc", "scale"))
    elif task == "classification" and classes is not None and len(classes) == body:
        names = tuple(str(value) for value in classes)
    elif task == "ordinal" and body == 1:
        names = ("latent",)
    if names is None or len(names) != body:
        names = _numbered("output", body)
    return names + (("cure_logit",) if cure else ())


def _select(outputs: tuple[str, ...], output) -> int:
    if isinstance(output, (int, np.integer)) and not isinstance(output, bool):
        if not -len(outputs) <= int(output) < len(outputs):
            raise AshlarError(f"output {output} is out of range for {len(outputs)} outputs.")
        return int(output) % len(outputs)
    if output in outputs:
        return outputs.index(output)
    raise AshlarError(f"output must be one of {list(outputs)} or its position; received {output!r}.")


@dataclass(frozen=True)
class ShapeFunctions:
    """Each additive term's part of each head output, row by row.

    ``member_values`` is ``(rows, members, terms, outputs)`` and
    ``member_intercept`` ``(members, outputs)``: on every row and for every
    member, the intercept plus the terms is the head's output, before a random
    effect's spread is added. ``values`` and ``intercept`` average the
    members, and ``spread`` is the members' standard deviation, so a term the
    members disagree on shows it. Each term is centered on the rows read, its
    mean moved into the intercept, unless ``center=False``.

    ``names`` are the terms: a column, ``a:b`` for a pair, and ``column
    (random effect)`` for a grouping column's offsets; ``columns`` gives each
    term's columns of ``X``. ``outputs`` names the head's outputs. Location
    outputs are in the units of ``y``; the others are on the scale the head
    reads them on, such as a raw scale under a softplus, a log mean, or a logit.
    """

    names: tuple[str, ...]
    columns: tuple[tuple[str, ...], ...]
    outputs: tuple[str, ...]
    member_values: np.ndarray
    member_intercept: np.ndarray

    @property
    def values(self) -> np.ndarray:
        """``(rows, terms, outputs)``, the members' mean."""
        return self.member_values.mean(axis=1)

    @property
    def intercept(self) -> np.ndarray:
        """``(outputs,)``, the members' mean."""
        return self.member_intercept.mean(axis=0)

    @property
    def spread(self) -> np.ndarray:
        """``(rows, terms, outputs)``: the members' standard deviation, zero for one member."""
        return self.member_values.std(axis=1)

    def total(self) -> np.ndarray:
        """``(rows, outputs)``: the intercept plus every term, the members' mean of the head's outputs."""
        return self.intercept + self.values.sum(axis=1)

    def term(self, name: str, output=None) -> np.ndarray:
        """One term: ``(rows,)`` on ``output`` (a name or position), or ``(rows, outputs)`` without one."""
        if name not in self.names:
            raise AshlarError(f"term must be one of {list(self.names)}; received {name!r}.")
        values = self.values[:, self.names.index(name), :]
        return values if output is None else values[:, _select(self.outputs, output)]

    def importance(self, output=None) -> dict:
        """Mean absolute value of each centered term over the rows, largest first.

        ``{term: value}`` on one ``output``, or ``{term: (outputs,)}`` ordered
        by the first output without one.
        """
        values = self.values - self.values.mean(axis=0, keepdims=True)
        scores = np.abs(values).mean(axis=0)
        if output is not None:
            column = scores[:, _select(self.outputs, output)]
            order = np.argsort(-column, kind="stable")
            return {self.names[index]: float(column[index]) for index in order}
        order = np.argsort(-scores[:, 0], kind="stable")
        return {self.names[index]: scores[index] for index in order}

    def to_frame(self, output):
        """A pandas DataFrame of every term on one ``output``: a row per row read, a column per term."""
        import pandas as pd

        return pd.DataFrame(self.values[:, :, _select(self.outputs, output)], columns=list(self.names))


def read_shapes(model, X, *, center: bool = True) -> ShapeFunctions:
    """``model.shape_functions(X)``: see ``ShapeFunctions``."""
    from ashlar.effects import location_outputs
    from ashlar.heads import DistributionHead, HurdleHead, LogitsHead, NormalHead, OrdinalHead, PointHead
    from ashlar.layers import AdditiveBackbone, LinearEnsemble

    model._require_fitted()
    core = model.core_
    backbone = core.backbone
    if not isinstance(backbone, AdditiveBackbone):
        raise AshlarError("shape_functions reads the terms of an additive model; fit with backbone='additive'.")
    head = core.head
    linear = getattr(head, "output", None)
    if not isinstance(head, (PointHead, LogitsHead, OrdinalHead, NormalHead, DistributionHead, HurdleHead)) or not isinstance(
        linear, LinearEnsemble
    ):
        raise AshlarError(
            f"shape_functions reads a built-in one-column head's linear map; this model's head is {type(head).__name__}."
        )
    names = [name for name, _ in model._core_fields(model.encoder_)]
    terms, columns = model._additive_terms(names, (model._resolved_training()["backbone_kwargs"] or {}).get("pairs"))
    if tuple(terms) != tuple(backbone.terms):
        raise AshlarError("this model's terms do not match its schema; it was not built by this version of Ashlar.")
    members = int(backbone.k)
    count = len(terms)
    width = int(linear.weight.shape[-1])
    weight = linear.weight.reshape(members, count, int(backbone.d_term), width)
    encoded = model.encoder_.transform(model._with_cyclical(X))
    numeric_z = model.numeric_transform_(encoded.numeric_raw)
    inputs = (numeric_z, encoded.categorical_onehot, encoded.category_index, encoded.set_tokens, encoded.set_mask)
    n_rows = int(numeric_z.shape[0])
    blocks = []
    for start in range(0, max(n_rows, 1), _READ_ROWS):
        part = [array[start : start + _READ_ROWS] for array in inputs]
        outputs = backbone.term_outputs(core.fields(*part))
        blocks.append(np.asarray(jnp.einsum("bkto,ktop->bktp", outputs, weight), dtype=np.float64))
    values = np.concatenate(blocks, axis=0)[:n_rows]
    intercept = np.asarray(linear.bias, dtype=np.float64).reshape(members, width)
    term_names = [":".join(group) for group in columns]
    term_columns = [tuple(group) for group in columns]
    extra = []
    if core.monotone is not None:
        numeric = [column.name for column in model.encoder_.columns if column.role == "numeric"]
        parts = np.asarray(core.monotone.parts(numeric_z), dtype=np.float64)
        for position, column in enumerate(core.monotone.columns):
            extra.append(parts[:, :, position, :])
            term_names.append(numeric[column])
            term_columns.append((numeric[column],))
    effects = getattr(core, "random_effects", None)
    if effects is not None and effects.has_offsets:
        grouping = [column.name for column in model.encoder_.embedding_columns]
        numeric = [column.name for column in model.encoder_.columns if column.role == "numeric"]
        location = np.zeros((width,), dtype=np.float64)
        location[list(effects.locations)] = 1.0
        for (position, mean), slopes in zip(
            effects.column_offsets(encoded.category_index, numeric_z),
            [terms for terms in effects.slope_columns if terms],
        ):
            offset = np.asarray(mean, dtype=np.float64)[:, None, None] * location[None, None, :]
            extra.append(np.broadcast_to(offset, (n_rows, members, width)))
            name = grouping[position]
            term_names.append(f"{name} (random effect)")
            term_columns.append((name, *(numeric[slope] for slope in slopes if slope >= 0)))
    if extra:
        values = np.concatenate([values, np.stack(extra, axis=2)], axis=2)
    if center and n_rows:
        means = values.mean(axis=0)
        values = values - means[None]
        intercept = intercept + means.sum(axis=1)
    unit = np.ones((width,), dtype=np.float64)
    shift = np.zeros((width,), dtype=np.float64)
    if getattr(model, "target_std_", None) is not None:
        try:
            locations, _scales = location_outputs(model.task, width - int(bool(model._cure())))
        except AshlarError:
            locations = ()
        stds = np.broadcast_to(np.asarray(model.target_std_, dtype=np.float64), (width,)) if model.task == "regression" else None
        means_y = np.broadcast_to(np.asarray(model.target_mean_, dtype=np.float64), (width,)) if model.task == "regression" else None
        for index in locations:
            unit[index] = float(model.target_std_) if stds is None else float(stds[index])
            shift[index] = float(model.target_mean_) if means_y is None else float(means_y[index])
    values = values * unit
    intercept = intercept * unit + shift
    classes = getattr(model, "classes_", None)
    return ShapeFunctions(
        names=tuple(term_names),
        columns=tuple(term_columns),
        outputs=output_names(model.task, width, classes=classes, cure=bool(model._cure())),
        member_values=values,
        member_intercept=intercept,
    )
