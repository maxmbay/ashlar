"""A short Optax loop for tabular batches.

Klax and Solstice can train a generic Equinox module. Neither knows about a
fitted schema, direct loss weights, or member-wise TabM losses, and Klax's
built-in validation metric evaluates one larger batch rather than the full
validation partition. This loop is the smaller piece that encodes those
contracts. It is not a general training framework.

Training uses ``batch_weight_sum`` inside each minibatch. The monitored
validation loss uses the same formula on the whole validation partition.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox import (
    combine as eqx_combine,
    partition as eqx_partition,
    apply_updates,
    filter_grad,
    filter_jit,
    filter_jvp,
    filter_value_and_grad,
    filter as eqx_filter,
    is_array,
)

from ashlar.errors import AshlarError, WeightError

# The surface schedule is opt-in. Warmup uses a small AdamW rate, then one
# training batch sets the rate for the rest of the run. 10000 is a reference
# row count, the houses training cap, not a learning rate.
SURFACE_PROBE_RATE = 1e-4
SURFACE_REFERENCE_ROWS = 10_000
SURFACE_RATE_MIN = 1e-5
SURFACE_RATE_MAX = 1e-1
_STEEPNESS_FLOOR = 1e-8


# ``batch_size="auto"`` and ``max_epochs="auto"``. The batch doubles from 256
# for every doubling of the training rows past 50,000, up to 2,048. The epoch
# budget is about 20 million row passes, between 20 and 100 epochs.
AUTO_BATCH_BASE = 256
AUTO_BATCH_ROWS = 50_000
AUTO_BATCH_MAX = 2048
AUTO_ROW_PASSES = 20_000_000
AUTO_EPOCHS_MIN = 20
AUTO_EPOCHS_MAX = 100


def auto_batch_size(n_train: int) -> int:
    """Minibatch rows for a training partition of ``n_train`` rows."""
    import math

    if n_train <= AUTO_BATCH_ROWS:
        return AUTO_BATCH_BASE
    doublings = int(round(math.log2(n_train / AUTO_BATCH_ROWS)))
    return int(min(AUTO_BATCH_MAX, AUTO_BATCH_BASE * 2**doublings))


def auto_max_epochs(n_train: int) -> int:
    """Epoch budget for a training partition of ``n_train`` rows."""
    if n_train < 1:
        return AUTO_EPOCHS_MAX
    return int(min(AUTO_EPOCHS_MAX, max(AUTO_EPOCHS_MIN, round(AUTO_ROW_PASSES / n_train))))


# Share of the step budget spent warming up under the ``cosine`` schedule.
COSINE_WARMUP_SHARE = 0.05


def _learning_rate_arg(learning_rate, schedule, n_rows, batch_size, max_epochs):
    """AdamW learning rate. ``linear`` decays from ``learning_rate`` to 0.

    ``cosine`` warms up linearly from 0 over the first 5% of steps, then
    follows a half cosine down to 0. Either decay covers ``max_epochs`` times
    the number of minibatches in one pass. Early stopping does not stretch the
    schedule. ``constant`` leaves the rate unchanged.
    """
    if schedule in (None, "constant"):
        return learning_rate
    steps_per_epoch = max(1, (int(n_rows) + int(batch_size) - 1) // int(batch_size))
    total = steps_per_epoch * int(max_epochs)
    if schedule == "linear":
        return optax.linear_schedule(init_value=float(learning_rate), end_value=0.0, transition_steps=total)
    if schedule == "cosine":
        warmup = max(1, int(round(COSINE_WARMUP_SHARE * total)))
        return optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(learning_rate),
            warmup_steps=warmup,
            decay_steps=max(total, warmup + 1),
            end_value=0.0,
        )
    raise AshlarError("learning_rate_schedule must be 'constant', 'linear', 'cosine', or 'surface'.")


def decay_mask(params):
    """``True`` for every leaf AdamW may decay. Frozen fitted leaves, such as bin edges, are ``False``."""
    from ashlar.layers import FROZEN_FIELDS

    def keep(path, leaf):
        del leaf
        return not any(getattr(entry, "name", None) in FROZEN_FIELDS for entry in path)

    return jax.tree_util.tree_map_with_path(keep, params)


def make_optimizer(learning_rate, weight_decay: float, gradient_clip: float | None = None):
    """AdamW with frozen fitted leaves left out of the decay. Gradients there are already zero.

    ``gradient_clip`` rescales the whole gradient to that global norm first.
    """
    adamw = optax.adamw(learning_rate, weight_decay=weight_decay, mask=decay_mask)
    if gradient_clip is None:
        return adamw
    return optax.chain(optax.clip_by_global_norm(float(gradient_clip)), adamw)


def surface_learning_rate(n_train: int, curvature: float, grad_rms: float) -> float:
    """AdamW rate from one training batch's loss scale.

    Positive directional curvature ``uᵀHu`` is the steepness. Otherwise the
    root-mean-square gradient is the steepness. The rate is
    ``(n_train / 10000) / steepness``, clipped to ``[1e-5, 1e-1]``. A smaller
    training partition or a larger steepness lowers the rate.
    """
    steepness = float(curvature) if float(curvature) > 0.0 else float(grad_rms)
    if steepness < _STEEPNESS_FLOOR:
        steepness = _STEEPNESS_FLOOR
    raw = (float(n_train) / float(SURFACE_REFERENCE_ROWS)) / steepness
    if raw < SURFACE_RATE_MIN:
        return float(SURFACE_RATE_MIN)
    if raw > SURFACE_RATE_MAX:
        return float(SURFACE_RATE_MAX)
    return float(raw)


# An objective that draws (``objective.stochastic``, such as the energy score) takes a key. Its draws come
# from their own stream of the step's key, apart from dropout, and validation reads every epoch with the
# same fixed draws, so epochs compare on equal terms. Such an objective is validated in smaller chunks.
_DRAW_STREAM = 0x5C02E
_DRAWN_CHUNK_DIVISOR = 4


def _loss(objective, raw, y, sample_weight, key=None):
    """``objective(raw, y, sample_weight)``, with a key for one that draws; ``key=None`` reads the fixed draws."""
    if not getattr(objective, "stochastic", False):
        return objective(raw, y, sample_weight)
    if key is None:
        key = jax.random.PRNGKey(_DRAW_STREAM)
    else:
        key = jax.random.fold_in(key, _DRAW_STREAM)
    return objective(raw, y, sample_weight, key=key)


def _adamw_step(optimizer, objective):
    @filter_jit
    def step(model, state, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch, step_key):
        def loss_fn(candidate):
            raw = candidate(z_batch, cat_batch, index_batch, token_batch, mask_batch, key=step_key)
            return _loss(objective, raw, y_batch, w_batch, step_key)

        loss, grads = filter_value_and_grad(loss_fn)(model)
        arrays = eqx_filter(model, is_array)
        updates, state = optimizer.update(eqx_filter(grads, is_array), state, arrays)
        model = apply_updates(model, updates)
        return model, state, loss

    return step


def _stack_members(array, take, batch_size, k):
    """Gather ``take`` of shape ``(member, width)`` into ``(batch_size, member, ...)``.

    The remaining width is zero padding, matching the shared-batch loop.
    """
    picked = jnp.swapaxes(array[take], 0, 1)
    width = int(picked.shape[0])
    flat = picked.reshape((width * k,) + tuple(picked.shape[2:]))
    flat = _pad(flat, batch_size * k)
    return flat.reshape((batch_size, k) + tuple(picked.shape[2:]))


def _flat_members(stacked):
    """``(batch, member, ...)`` to ``(batch * member, ...)``, member fastest."""
    return stacked.reshape((stacked.shape[0] * stacked.shape[1],) + tuple(stacked.shape[2:]))


def _pad(array, length):
    if array.shape[0] == length:
        return array
    extra = length - array.shape[0]
    zeros = jnp.zeros((extra,) + tuple(array.shape[1:]), dtype=array.dtype)
    return jnp.concatenate([array, zeros], axis=0)


def fit_loop(
    core,
    numeric_z,
    categorical,
    y,
    sample_weight,
    *,
    objective,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    key,
    validation=None,
    opt_state=None,
    category_index=None,
    set_tokens=None,
    set_mask=None,
    score_fn=None,
    learning_rate_schedule: str = "constant",
    share_training_batches: bool = True,
    gradient_clip: float | None = None,
    units=None,
):
    """Train ``core`` and return the best validation checkpoint when validation exists.

    The optimizer is AdamW. ``learning_rate_schedule='linear'`` only replaces
    the scalar rate with a linear decay to zero. ``surface`` warms up at
    ``1e-4`` for one epoch, capped at 20 steps, then replaces the rate with
    ``surface_learning_rate``. The constructor rate is unused after that probe.

    ``units`` keeps observations that span several adjacent rows, such as
    group totals, whole inside each minibatch; see ``_fit_scanned``.
    """
    n_rows = int(numeric_z.shape[0])
    batch_size = max(1, min(int(batch_size), n_rows))
    surface = learning_rate_schedule == "surface"
    if units is not None and (surface or not (share_training_batches or int(core.backbone.k) == 1)):
        raise AshlarError("observations that span several rows train with shared minibatches and a fixed schedule, not surface.")
    if surface:
        optimizer = make_optimizer(SURFACE_PROBE_RATE, weight_decay, gradient_clip)
    else:
        optimizer = make_optimizer(
            _learning_rate_arg(learning_rate, learning_rate_schedule, n_rows, batch_size, max_epochs),
            weight_decay,
            gradient_clip,
        )
    params = eqx_filter(core, is_array)
    if opt_state is None:
        opt_state = optimizer.init(params)
    step = _adamw_step(optimizer, objective)

    @filter_jit
    def surface_stats(model, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch):
        """Directional curvature and RMS gradient, with dropout off."""

        def loss_fn(candidate):
            raw = candidate(z_batch, cat_batch, index_batch, token_batch, mask_batch, key=None)
            return _loss(objective, raw, y_batch, w_batch)

        grad_fn = filter_grad(loss_fn)
        grads = grad_fn(model)
        leaves = [leaf for leaf in jax.tree_util.tree_leaves(grads) if is_array(leaf)]
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        if not leaves:
            return zero, zero
        g_norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
        n_params = sum(leaf.size for leaf in leaves)
        grad_rms = g_norm / jnp.sqrt(jnp.asarray(n_params, dtype=g_norm.dtype))
        denom = jnp.maximum(g_norm, jnp.asarray(1e-12, dtype=g_norm.dtype))

        def _unit(leaf):
            if not is_array(leaf):
                return None
            return leaf / denom

        direction = jax.tree_util.tree_map(_unit, grads)
        _value, product = filter_jvp(grad_fn, (model,), (direction,))
        # A leaf with a stopped gradient, such as frozen bin edges, can come back
        # from the JVP as None; pair leaves by position and count it as zero.
        pairs = jax.tree_util.tree_map(
            lambda direction_leaf, product_leaf: (
                jnp.vdot(direction_leaf, product_leaf)
                if is_array(direction_leaf) and is_array(product_leaf)
                else jnp.asarray(0.0, dtype=jnp.float32)
            ),
            direction,
            product,
            is_leaf=lambda leaf: leaf is None,
        )
        curvature = sum(jax.tree_util.tree_leaves(pairs))
        return curvature, grad_rms

    @filter_jit
    def evaluate(model, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch):
        raw = model(z_batch, cat_batch, index_batch, token_batch, mask_batch, key=None)
        return _loss(objective, raw, y_batch, w_batch)

    if category_index is None:
        category_index = jnp.zeros((n_rows, 0), dtype=jnp.int32)
    if set_tokens is None:
        set_tokens = jnp.zeros((n_rows, 0, 1), dtype=jnp.int32)
        set_mask = jnp.zeros((n_rows, 0, 1), dtype=jnp.float32)
    member_count = int(core.backbone.k)
    if not surface and (share_training_batches or member_count == 1):
        return _fit_scanned(
            core,
            (numeric_z, categorical, category_index, set_tokens, set_mask, y, sample_weight),
            optimizer=optimizer,
            opt_state=opt_state,
            objective=objective,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            key=key,
            validation=validation,
            score_fn=score_fn,
            units=units,
        )
    history = {"train_loss": [], "val_loss": []}
    best_core = core
    best_state = opt_state
    best_val = None
    best_score = None
    stale = 0
    steps_per_epoch = max(1, (n_rows + batch_size - 1) // batch_size)
    warmup_limit = min(20, steps_per_epoch) if surface else 0
    warmup_seen = 0
    surface_ready = not surface

    for epoch in range(max_epochs):
        epoch_key = jax.random.fold_in(key, epoch)
        perm_key, batch_key = jax.random.split(epoch_key)
        member_count = int(core.backbone.k)
        unshared = (not share_training_batches) and member_count > 1
        if unshared:
            member_keys = jax.random.split(perm_key, member_count)
            order = jnp.stack([jax.random.permutation(member_key, n_rows) for member_key in member_keys])
        else:
            order = jax.random.permutation(perm_key, n_rows)
        train_losses = []
        cursor = 0
        step_index = 0
        while cursor < n_rows:
            if unshared:
                width = min(batch_size, n_rows - cursor)
                take = order[:, cursor : cursor + width]
                cursor += width
                weights = sample_weight[take]
                if float(jnp.sum(weights)) <= 0.0:
                    continue
                z_members = _stack_members(numeric_z, take, batch_size, member_count)
                y_members = _stack_members(y, take, batch_size, member_count)
                w_members = _stack_members(sample_weight, take, batch_size, member_count)
                z_batch = _flat_members(z_members)
                cat_batch = _flat_members(_stack_members(categorical, take, batch_size, member_count))
                index_batch = _flat_members(_stack_members(category_index, take, batch_size, member_count))
                token_batch = _flat_members(_stack_members(set_tokens, take, batch_size, member_count))
                mask_batch = _flat_members(_stack_members(set_mask, take, batch_size, member_count))
                y_batch = y_members
                w_batch = w_members
            else:
                take = order[cursor : cursor + batch_size]
                cursor += batch_size
                weights = sample_weight[take]
                if float(jnp.sum(weights)) <= 0.0:
                    continue
                z_batch = _pad(numeric_z[take], batch_size)
                cat_batch = _pad(categorical[take], batch_size)
                index_batch = _pad(category_index[take], batch_size)
                token_batch = _pad(set_tokens[take], batch_size)
                mask_batch = _pad(set_mask[take], batch_size)
                y_batch = _pad(y[take], batch_size)
                w_batch = _pad(weights, batch_size)
            step_key = jax.random.fold_in(batch_key, step_index)
            core, opt_state, loss = step(
                core, opt_state, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch, step_key
            )
            loss = float(loss)
            if loss != loss:
                raise WeightError("training loss is not finite. Check targets, weights, and the learning rate.")
            train_losses.append(loss)
            step_index += 1
            warmup_seen += 1
            if surface and not surface_ready and warmup_seen >= warmup_limit:
                curvature, grad_rms = surface_stats(
                    core, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch
                )
                curvature_value = float(curvature)
                grad_rms_value = float(grad_rms)
                if curvature_value != curvature_value or grad_rms_value != grad_rms_value:
                    raise WeightError("the loss-surface scale is not finite. Check targets, weights, and the probe rate.")
                chosen = surface_learning_rate(n_rows, curvature_value, grad_rms_value)
                optimizer = make_optimizer(chosen, weight_decay, gradient_clip)
                step = _adamw_step(optimizer, objective)
                opt_state = optimizer.init(eqx_filter(core, is_array))
                surface_ready = True
                history["chosen_learning_rate"] = chosen
                history["surface_curvature"] = curvature_value
                history["surface_grad_rms"] = grad_rms_value
                history["surface_n_train"] = int(n_rows)
                history["surface_warmup_steps"] = int(warmup_seen)
                history["surface_probe_rate"] = float(SURFACE_PROBE_RATE)
        if not train_losses:
            raise WeightError("every training batch has zero total weight, so there is nothing to optimize.")
        history["train_loss"].append(float(np_mean(train_losses)))
        if validation is None:
            best_core, best_state = core, opt_state
            continue
        val_loss = float(evaluate(core, *validation))
        if val_loss != val_loss:
            raise WeightError("validation loss is not finite.")
        history["val_loss"].append(val_loss)
        if score_fn is None:
            improved = best_val is None or val_loss < best_val
            if improved:
                best_val = val_loss
        else:
            metric = float(score_fn(core, *validation))
            if metric != metric:
                raise WeightError("validation score is not finite.")
            history.setdefault("val_score", []).append(metric)
            improved = best_score is None or metric > best_score
            if improved:
                best_score = metric
        if improved:
            best_core, best_state = core, opt_state
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
    if surface and not surface_ready:
        raise WeightError("the surface schedule finished before its warmup, so no learning rate was set.")
    return best_core, best_state, history


def np_mean(values: list[float]) -> float:
    return sum(values) / len(values)


# Rows per forward pass when a whole partition is scored. A partition at or
# below this size is one pass, so small-table losses are unchanged.
EVAL_CHUNK_ROWS = 8192


def pack_units(first, size, order, batch_size: int, steps: int | None = None):
    """Rows of whole observations, in ``order``, packed into batches of ``batch_size`` rows.

    ``first`` and ``size`` give each observation's adjacent rows (a group
    total, or one row). An observation never straddles two batches, and its
    rows stay adjacent and in order, so an objective can find each one from a
    flag on its first row. Returns ``(index, keep)`` of shape
    ``(steps, batch_size)``: row indices, and ``False`` on padding, whose index
    is 0. Each batch starts its observations within its first
    ``batch_size - largest + 1`` slots, which fixes the number of batches.
    """
    first = np.asarray(first, dtype=np.int64)
    size = np.asarray(size, dtype=np.int64)
    order = np.asarray(order, dtype=np.int64)
    largest = int(size.max())
    if largest > batch_size:
        raise ValueError(f"an observation of {largest} rows does not fit in a batch of {batch_size}.")
    capacity = batch_size - largest + 1
    total = int(size.sum())
    if steps is None:
        steps = max(1, (total - 1) // capacity + 1)
    sizes = size[order]
    start = np.cumsum(sizes) - sizes
    batch = start // capacity
    lead = np.searchsorted(batch, np.arange(steps), side="left")
    lead_start = start[np.minimum(lead, len(start) - 1)]
    slot = start - lead_start[batch]
    unit = np.repeat(np.arange(len(order)), sizes)
    within = np.arange(total) - np.repeat(start, sizes)
    index = np.zeros((steps, batch_size), dtype=np.int32)
    keep = np.zeros((steps, batch_size), dtype=bool)
    index[batch[unit], slot[unit] + within] = first[order][unit] + within
    keep[batch[unit], slot[unit] + within] = True
    return index, keep


def chunked_weighted(fn, arrays, weights, chunk_rows: int = EVAL_CHUNK_ROWS):
    """``fn`` over row chunks, recombined with the ``batch_weight_sum`` reduction.

    ``fn(*chunk)`` returns a weighted mean over its rows. The chunks are
    recombined by their weight mass, which equals ``fn`` on the whole partition
    for any objective of the form ``sum(w * loss) / sum(w)``.
    """
    n_rows = int(weights.shape[0])
    if n_rows <= chunk_rows:
        return float(fn(*arrays))
    total = 0.0
    mass = 0.0
    for start in range(0, n_rows, chunk_rows):
        piece = tuple(array[start : start + chunk_rows] for array in arrays)
        chunk_mass = float(jnp.sum(weights[start : start + chunk_rows]))
        if chunk_mass <= 0.0:
            continue
        total += float(fn(*piece)) * chunk_mass
        mass += chunk_mass
    return total / mass if mass > 0.0 else float("nan")


def _fit_scanned(
    core,
    data,
    *,
    optimizer,
    opt_state,
    objective,
    batch_size: int,
    max_epochs: int,
    patience: int,
    key,
    validation,
    score_fn,
    units=None,
):
    """Shared-minibatch training with one compiled scan per epoch.

    Same permutation, minibatches, dropout keys, early stopping, and
    checkpoint as the step-by-step loop. A minibatch with zero weight mass
    leaves the model and optimizer state unchanged. The last minibatch is
    padded with zero-weight rows, as before.

    ``units`` is ``(first, size)`` of observations that span adjacent rows,
    such as group totals. Each epoch then shuffles observations instead of
    rows and packs them whole into minibatches (``pack_units``), with at
    least two of the largest per batch.
    """
    n_rows = int(data[0].shape[0])
    if units is not None:
        first, size = (np.asarray(part, dtype=np.int64) for part in units)
        largest = int(size.max())
        batch_size = min(max(int(batch_size), 2 * largest), n_rows)
        steps_per_epoch = max(1, (n_rows - 1) // (batch_size - largest + 1) + 1)
    else:
        steps_per_epoch = max(1, (n_rows + batch_size - 1) // batch_size)
    padded = steps_per_epoch * batch_size
    params, static = eqx_partition(core, is_array)

    def scan_epoch(params, state, index, mask, batch_key):
        def body(carry, inputs):
            params, state = carry
            take, keep, step_index = inputs
            batch = tuple(array[take] for array in data)
            *features, y_batch, w_batch = batch
            w_batch = jnp.where(keep, w_batch, jnp.zeros_like(w_batch))
            step_key = jax.random.fold_in(batch_key, step_index)

            def loss_fn(candidate_params):
                candidate = eqx_combine(candidate_params, static)
                raw = candidate(*features, key=step_key)
                return _loss(objective, raw, y_batch, w_batch, step_key)

            has_mass = jnp.sum(w_batch) > 0

            def update(_):
                loss, grads = jax.value_and_grad(loss_fn)(params)
                updates, new_state = optimizer.update(grads, state, params)
                return optax.apply_updates(params, updates), new_state, loss

            def skip(_):
                return params, state, jnp.asarray(0.0, dtype=jnp.float32)

            new_params, new_state, loss = jax.lax.cond(has_mass, update, skip, None)
            return (new_params, new_state), (loss, has_mass)

        steps = jnp.arange(steps_per_epoch)
        (params, state), (losses, seen) = jax.lax.scan(body, (params, state), (index, mask, steps))
        return params, state, losses, seen

    @filter_jit
    def run_epoch(params, state, order, batch_key):
        valid = jnp.arange(padded) < n_rows
        index = jnp.where(valid, jnp.pad(order, (0, padded - n_rows)), 0)
        index = index.reshape(steps_per_epoch, batch_size)
        mask = valid.reshape(steps_per_epoch, batch_size)
        return scan_epoch(params, state, index, mask, batch_key)

    run_packed_epoch = filter_jit(scan_epoch)

    @filter_jit
    def evaluate(model, *batch):
        *features, y_batch, w_batch = batch
        raw = model(*features, key=None)
        return _loss(objective, raw, y_batch, w_batch)

    eval_rows = EVAL_CHUNK_ROWS
    if getattr(objective, "stochastic", False):
        eval_rows = max(256, EVAL_CHUNK_ROWS // _DRAWN_CHUNK_DIVISOR // max(1, int(core.backbone.k)))
    history = {"train_loss": [], "val_loss": []}
    best_core = core
    best_state = opt_state
    best_val = None
    best_score = None
    stale = 0
    for epoch in range(max_epochs):
        epoch_key = jax.random.fold_in(key, epoch)
        perm_key, batch_key = jax.random.split(epoch_key)
        if units is None:
            order = jax.random.permutation(perm_key, n_rows)
            params, opt_state, losses, seen = run_epoch(params, opt_state, order, batch_key)
        else:
            order = np.asarray(jax.random.permutation(perm_key, len(first)))
            index, keep = pack_units(first, size, order, batch_size, steps_per_epoch)
            params, opt_state, losses, seen = run_packed_epoch(
                params, opt_state, jnp.asarray(index), jnp.asarray(keep), batch_key
            )
        seen = np.asarray(losses)[np.asarray(seen)]
        if seen.size == 0:
            raise WeightError("every training batch has zero total weight, so there is nothing to optimize.")
        if not np.isfinite(seen).all():
            raise WeightError("training loss is not finite. Check targets, weights, and the learning rate.")
        history["train_loss"].append(float(seen.mean()))
        core = eqx_combine(params, static)
        if validation is None:
            best_core, best_state = core, opt_state
            continue
        val_loss = chunked_weighted(lambda *chunk: evaluate(core, *chunk), validation, validation[-1], eval_rows)
        if val_loss != val_loss:
            raise WeightError("validation loss is not finite.")
        history["val_loss"].append(val_loss)
        if score_fn is None:
            improved = best_val is None or val_loss < best_val
            if improved:
                best_val = val_loss
        else:
            metric = chunked_weighted(lambda *chunk: score_fn(core, *chunk), validation, validation[-1])
            if metric != metric:
                raise WeightError("validation score is not finite.")
            history.setdefault("val_score", []).append(metric)
            improved = best_score is None or metric > best_score
            if improved:
                best_score = metric
        if improved:
            best_core, best_state = core, opt_state
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
    return best_core, best_state, history
