"""Unit tests run on CPU unless the caller already chose a backend."""

import gc
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture(autouse=True, scope="module")
def _release_compiled_programs():
    """Drop the compiled programs of each test module when it ends.

    A worker runs hundreds of small fits, each compiling programs of its own,
    and JAX keeps every one for the life of the process. A worker of the full
    suite grew past 4 GB and was killed for it.
    """
    yield
    jax.clear_caches()
    gc.collect()
