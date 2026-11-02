"""Unit tests run on CPU unless the caller already chose a backend.

Most of a test's time is XLA compiling small programs, and JAX compiles each
operation it runs eagerly on its own. The tests' programs are tiny, so LLVM's
optimizations cost more than they save. Level 1 compiles them several times
faster than the default and runs them about as fast. Set
``XLA_FLAGS`` yourself to override it.
"""

import gc
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# scikit-learn's histogram trees spin up an OpenMP thread per core. Beside JAX in
# every pytest-xdist worker, those threads fight for the cores: one 0.4 s test
# took 100 s that way.
os.environ.setdefault("OMP_NUM_THREADS", "1")
if "xla_backend_optimization_level" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_backend_optimization_level=1").strip()

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
