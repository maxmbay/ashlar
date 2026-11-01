"""Persistence for importable components and array pytrees.

Custom heads and objectives are stored as ``module:qualname`` and imported on
load. Closures and classes defined in ``__main__`` are rejected. Optimizer
moments are array leaves restored into a freshly initialized Optax state.
Load only models you trust: loading imports the recorded Python callables.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import equinox as eqx

from ashlar.errors import AshlarError


def dump_ref(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(key): dump_ref(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dump_ref(value) for value in obj]
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module or not qualname or module == "__main__":
        raise AshlarError(
            "custom classes and functions must be defined in an importable module so a saved "
            f"model can be reloaded. Received {obj!r}."
        )
    return {"__callable__": f"{module}:{qualname}"}


def load_ref(value):
    if isinstance(value, dict) and set(value) == {"__callable__"}:
        module_name, qualname = value["__callable__"].split(":", 1)
        current = importlib.import_module(module_name)
        for part in qualname.split("."):
            current = getattr(current, part)
        return current
    if isinstance(value, dict):
        return {key: load_ref(item) for key, item in value.items()}
    if isinstance(value, list):
        return [load_ref(item) for item in value]
    return value


def save_leaves(path: Path, tree) -> None:
    eqx.tree_serialise_leaves(path, tree)


def load_leaves(path: Path, template):
    return eqx.tree_deserialise_leaves(path, template)


def save_opt_state(directory: Path, opt_state) -> None:
    """Save optimizer moments. ``load_opt_state`` needs a freshly initialized template."""
    eqx.tree_serialise_leaves(directory / "optimizer.eqx", opt_state)


def load_opt_state(directory: Path, template):
    return eqx.tree_deserialise_leaves(directory / "optimizer.eqx", template)
