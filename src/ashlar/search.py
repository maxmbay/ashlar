"""Bounded hyperparameter search. It runs only when ``fit(..., search=...)`` asks.

``search=None`` and ``search=False`` leave training on the constructor settings.
``search=True`` uses ``DEFAULT_SEARCH``. A dict widens that default: keys replace
the matching default, and ``space`` replaces the default space entirely.

The grid is the cartesian product of the lists in ``space``, in key order.
When the product is longer than ``n_trials``, the first ``n_trials`` combinations
in that product order are kept. This is deterministic. It is not a random search.

Allowed space keys are ``learning_rate``, ``weight_decay``, ``feature_dropout``,
``batch_size``, and ``backbone_kwargs``. ``backbone_kwargs`` values are dicts merged over the
estimator's own backbone kwargs. The winning values are the ones used for the
final train. Trial budgets (``max_epochs``, ``patience``) are not the final train.

Regression and normal trials are ranked by validation loss, lower better.
Classification trials are ranked by weighted accuracy, higher better. That
score uses the same effective row weights as the weighted cross-entropy. It
does not apply class weights a second time, and it is not a squared-error loss.
The final train after a classification search still minimizes that weighted
cross-entropy. Omitting ``search`` does not compute the accuracy score.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product

from ashlar.errors import AshlarError

DEFAULT_SEARCH = {
    "n_trials": 2,
    "max_epochs": 2,
    "patience": 0,
    "space": {"weight_decay": [1e-4, 1e-3]},
}

_SPACE_KEYS = {"learning_rate", "weight_decay", "feature_dropout", "batch_size", "backbone_kwargs"}


def resolve_search(search) -> dict:
    """Return a concrete search spec. ``True`` selects ``DEFAULT_SEARCH``."""
    if search is True:
        return deepcopy(DEFAULT_SEARCH)
    if not isinstance(search, dict):
        raise AshlarError("search must be True or a dict. Omit it, or pass False, to train without a search.")
    spec = deepcopy(DEFAULT_SEARCH)
    if "space" in search:
        if not isinstance(search["space"], dict) or not search["space"]:
            raise AshlarError("search['space'] must be a non-empty dict of lists.")
        unknown = set(search["space"]).difference(_SPACE_KEYS)
        if unknown:
            raise AshlarError(
                f"search space keys {sorted(unknown)} are not supported. "
                f"Use {sorted(_SPACE_KEYS)}."
            )
        spec["space"] = {key: list(value) for key, value in search["space"].items()}
    for key in ("n_trials", "max_epochs", "patience"):
        if key in search:
            spec[key] = search[key]
    if int(spec["n_trials"]) < 1 or int(spec["max_epochs"]) < 1 or int(spec["patience"]) < 0:
        raise AshlarError("search n_trials and max_epochs must be positive, and patience must be non-negative.")
    return spec


def trial_grid(spec: dict) -> list[dict]:
    """The combinations that will actually be trained, capped at ``n_trials``."""
    space = spec["space"]
    keys = list(space)
    axes = []
    for key in keys:
        values = space[key]
        if isinstance(values, dict) or not isinstance(values, (list, tuple)) or len(values) == 0:
            raise AshlarError(f"search space[{key!r}] must be a non-empty list.")
        axes.append(list(values))
    combinations = [dict(zip(keys, values, strict=True)) for values in product(*axes)]
    return combinations[: int(spec["n_trials"])]
