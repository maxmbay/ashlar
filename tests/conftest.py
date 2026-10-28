"""Unit tests run on CPU unless the caller already chose a backend."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
