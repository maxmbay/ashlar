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
import optax
from equinox import (
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


def _learning_rate_arg(learning_rate, schedule, n_rows, batch_size, max_epochs):
    """AdamW learning rate. ``linear`` decays from ``learning_rate`` to 0.

    The decay covers ``max_epochs`` times the number of minibatches in one
    pass. Early stopping does not stretch the schedule. ``constant`` leaves
    the rate unchanged.
    """
    if schedule in (None, "constant"):
        return learning_rate
    if schedule != "linear":
        raise AshlarError("learning_rate_schedule must be 'constant' or 'linear'.")
    steps_per_epoch = max(1, (int(n_rows) + int(batch_size) - 1) // int(batch_size))
    return optax.linear_schedule(
        init_value=float(learning_rate),
        end_value=0.0,
        transition_steps=steps_per_epoch * int(max_epochs),
    )


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


def _adamw_step(optimizer, objective):
    @filter_jit
    def step(model, state, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch, step_key):
        def loss_fn(candidate):
            raw = candidate(z_batch, cat_batch, index_batch, token_batch, mask_batch, key=step_key)
            return objective(raw, y_batch, w_batch)

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
):
    """Train ``core`` and return the best validation checkpoint when validation exists.

    The optimizer is AdamW. ``learning_rate_schedule='linear'`` only replaces
    the scalar rate with a linear decay to zero. ``surface`` warms up at
    ``1e-4`` for one epoch, capped at 20 steps, then replaces the rate with
    ``surface_learning_rate``. The constructor rate is unused after that probe.
    """
    n_rows = int(numeric_z.shape[0])
    batch_size = max(1, min(int(batch_size), n_rows))
    surface = learning_rate_schedule == "surface"
    if surface:
        optimizer = optax.adamw(SURFACE_PROBE_RATE, weight_decay=weight_decay)
    else:
        optimizer = optax.adamw(
            _learning_rate_arg(learning_rate, learning_rate_schedule, n_rows, batch_size, max_epochs),
            weight_decay=weight_decay,
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
            return objective(raw, y_batch, w_batch)

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
        dir_leaves = [leaf for leaf in jax.tree_util.tree_leaves(direction) if is_array(leaf)]
        prod_leaves = [leaf for leaf in jax.tree_util.tree_leaves(product) if is_array(leaf)]
        curvature = sum(
            jnp.vdot(direction_leaf, product_leaf)
            for direction_leaf, product_leaf in zip(dir_leaves, prod_leaves, strict=True)
        )
        return curvature, grad_rms

    @filter_jit
    def evaluate(model, z_batch, cat_batch, index_batch, token_batch, mask_batch, y_batch, w_batch):
        raw = model(z_batch, cat_batch, index_batch, token_batch, mask_batch, key=None)
        return objective(raw, y_batch, w_batch)

    if category_index is None:
        category_index = jnp.zeros((n_rows, 0), dtype=jnp.int32)
    if set_tokens is None:
        set_tokens = jnp.zeros((n_rows, 0, 1), dtype=jnp.int32)
        set_mask = jnp.zeros((n_rows, 0, 1), dtype=jnp.float32)
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
                optimizer = optax.adamw(chosen, weight_decay=weight_decay)
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
