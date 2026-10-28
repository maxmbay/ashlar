"""Errors raised before an estimator mutates fitted state."""


class AshlarError(ValueError):
    """An Ashlar input or configuration is invalid."""


class WeightError(AshlarError):
    """Sample weights are missing, non-finite, negative, or the wrong shape."""


class ShapeError(AshlarError):
    """A tensor axis does not match the contract. Axes are never silently broadcast."""
