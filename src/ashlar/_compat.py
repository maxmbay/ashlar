"""Old public names, kept importable with a ``DeprecationWarning``."""

from __future__ import annotations

import warnings


def renamed(module: str, namespace: dict, names: dict[str, str]):
    """Module ``__getattr__`` that maps each old name in ``names`` to its new one."""

    def __getattr__(name):
        if name in names:
            warnings.warn(f"{name} is now {names[name]}.", DeprecationWarning, stacklevel=2)
            return namespace[names[name]]
        raise AttributeError(f"module {module!r} has no attribute {name!r}")

    return __getattr__
