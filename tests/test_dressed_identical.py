"""The dressed model's faster fit and prediction give the numbers of 0.1.0, bit for bit.

``dressed_0_1_0.py`` beside this file is ``src/ashlar/dressed.py`` as released in 0.1.0, unchanged. The three other
functions the speedups changed are kept below as released: the bisection of ``ashlar.distributions`` (it now stops
once no bracket moves), the label reader's attention (now cut into a side that reads the context alone and a side
that reads the queries) and ``LabelReader.atoms`` (it now keeps the context side). Each test runs the released code
beside the current code on the same data and compares the results as bytes, so one unit in the last place, a NaN
where a number belongs or the sign of a zero counts as a difference. Where both hold a NaN its bits are not compared
(``_same``). The fits are tiny, and pieces and blocks are cut small so that the threads share every read.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import pickle
import threading
import time
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import ashlar
import ashlar.distributions as distributions
import ashlar.label_reader as label_reader
from ashlar import ASRDressed
from ashlar.distributional import flow_parameter_count
from ashlar.distributions import _BISECTION_STEPS, Flow, Mixture, Normal
from ashlar.errors import AshlarError
from ashlar.label_reader import _keys, _predict, _relevance, _trend_features
from micro import TABM_NET

import dressed_0_1_0 as R  # src/ashlar/dressed.py as released in 0.1.0

D = importlib.import_module("ashlar.dressed")  # the module; ``ashlar.dressed`` is also the function

NETWORK = {"backbone_kwargs": TABM_NET, "num_embedding_dim": 4, "num_embedding_bins": 8, "max_epochs": 3}
READER = {"heads": 2, "key_size": 4, "width": 8, "max_steps": 60}
BLOCK = 50  # rows per block in the fits and predictions below, so each runs several


# ---------------------------------------------------------------------------- the other functions, as released


def _released_bisection_quantile(dist, q):
    """``ashlar.distributions._bisection_quantile`` as released: 80 halvings, always."""
    q = jnp.asarray(q, dtype=jnp.float32)
    frozen = jax.lax.stop_gradient(dist)
    shape = jnp.broadcast_shapes(q.shape, frozen.batch_shape)
    level = jnp.broadcast_to(jax.lax.stop_gradient(q), shape)
    inner = jnp.clip(level, 1e-7, 1.0 - 1e-7)
    mean = jnp.broadcast_to(frozen.mean(), shape)
    sd = jnp.broadcast_to(frozen.stddev(), shape)
    low = mean - sd * jnp.sqrt((1.0 - inner) / inner) - 1e-6
    high = mean + sd * jnp.sqrt(inner / (1.0 - inner)) + 1e-6

    def step(_, bounds):
        low, high = bounds
        middle = 0.5 * (low + high)
        above = frozen.cdf(middle) >= level
        return jnp.where(above, low, middle), jnp.where(above, middle, high)

    low, high = jax.lax.fori_loop(0, _BISECTION_STEPS, step, (low, high))
    x0 = jax.lax.stop_gradient(0.5 * (low + high))
    try:
        density = jnp.exp(dist.log_prob(x0))
    except NotImplementedError:
        # No density, as for draws: the bisection answer, with no gradient.
        refined = x0
    else:
        gap = dist.cdf(x0) - q
        newton = jnp.where(density > 1e-12, gap / jnp.maximum(density, 1e-12), 0.0)
        refined = x0 - newton + jax.lax.stop_gradient(newton)
    if dist.support in ("nonnegative", "positive"):
        # An atom at zero, or a lower edge the bracket straddles, is answered exactly.
        refined = jnp.where(frozen.cdf(jnp.zeros_like(x0)) >= level, 0.0, jnp.maximum(refined, 0.0))
    else:
        refined = jnp.where(level <= 0.0, -jnp.inf, refined)
    if dist.defective:
        # A level above the mass that ever happens is never reached.
        refined = jnp.where(frozen.cdf(jnp.full_like(x0, jnp.inf)) < level, jnp.inf, refined)
    return jnp.where(level >= 1.0, jnp.inf, refined)


def _released_attend(params, x_ctx, onehot_ctx, y_ctx, mask, x_query):
    """``ashlar.label_reader._attend`` as released, in one piece."""
    d = x_ctx.shape[1] // 2
    rel = _relevance(params, onehot_ctx, y_ctx, mask)
    z_ctx, raw_ctx = x_ctx[:, :d] * rel, x_ctx[:, d:] * rel
    z_query, raw_query = x_query[:, :d] * rel, x_query[:, d:] * rel
    weight = mask.astype(y_ctx.dtype)
    phi = _trend_features(params, z_ctx, raw_ctx)
    gram = (phi * weight[:, None]).T @ phi + jnp.exp(10.0 * params["log_ridge"]) * jnp.eye(phi.shape[1], dtype=phi.dtype)
    solved = jnp.linalg.solve(gram, phi.T)
    beta = solved @ (weight * y_ctx)
    leverage = jnp.einsum("np,pn->n", phi, solved)
    residual = (y_ctx - phi @ beta) / jnp.maximum(1.0 - weight * leverage, 1e-3)
    kq, kc = _keys(params, z_query), _keys(params, z_ctx)
    distance = (kq**2).sum(-1)[:, None, :] + (kc**2).sum(-1)[None, :, :] - 2.0 * jnp.einsum("qhk,chk->qch", kq, kc)
    logits = jnp.where(mask[None, :, None], -jnp.exp(params["log_sharp"]) * distance / kq.shape[-1], -1e30)
    weights = jnp.einsum("qch,h->qc", jax.nn.softmax(logits, axis=1), jax.nn.softmax(params["mix"]))
    return residual, _trend_features(params, z_query, raw_query) @ beta, weights


def _released_atoms(self, X, context="all"):
    """``LabelReader.atoms`` as released: the context side read again with every chunk of queries."""
    rows = slice(None) if context == "all" else slice(0, self.n_train_)
    residual, trend, weights = _predict(self.params_, jnp.asarray(self.x_[rows]), jnp.asarray(self.onehot_[rows]),
                                        jnp.asarray(self.y_[rows]), self.columns_(np.asarray(X, dtype=np.float64)))
    values = self.center_ + self.spread_ * (residual.astype(np.float64)[None, :] + trend.astype(np.float64)[:, None])
    weights = weights.astype(np.float64)
    return values, weights / weights.sum(axis=1, keepdims=True)


@contextlib.contextmanager
def _released_code():
    """The released bisection, attention and reader atoms in place of the current ones. Every compiled program is
    dropped on the way in and on the way out, so no program traced with one version runs under the other."""
    traced = {"bisection": 0, "attention": 0}

    def bisection(dist, q):
        traced["bisection"] += 1
        return _released_bisection_quantile(dist, q)

    def attend(*args):
        traced["attention"] += 1
        return _released_attend(*args)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(distributions, "_bisection_quantile", bisection)
        patch.setattr(label_reader, "_attend", attend)
        patch.setattr(label_reader.LabelReader, "atoms", _released_atoms)
        jax.clear_caches()
        try:
            yield traced
        finally:
            jax.clear_caches()


# ---------------------------------------------------------------------------------------------------- helpers


def _vector_extensions():
    """The widest vector extension numpy uses on this machine, for a message."""
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:
        return "numpy's vector extension unknown"
    return "numpy uses " + next((name for name in ("AVX512F", "AVX2", "SSE42", "ASIMD") if features.get(name)), "no vector extension")


def _same(a, b):
    """The same shape, type and bytes. Where both hold a NaN, its bits are not compared: IEEE 754 leaves the sign and
    the payload of a NaN result open, so two codes that agree on every number can differ in a NaN's bits. A warning
    says so when it happens. A NaN in one and a number in the other is a difference."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.tobytes() == b.tobytes():
        return True
    if a.dtype.kind != "f":
        return False
    nan = np.isnan(a)
    if not (np.array_equal(nan, np.isnan(b)) and np.where(nan, 0, a).tobytes() == np.where(nan, 0, b).tobytes()):
        return False
    warnings.warn(f"the two versions agree on every number and differ in the bits of a NaN: {_differences(a, b)}", stacklevel=2)
    return True


def _differences(a, b):
    """What differs between two results, for a message."""
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shapes {a.shape} and {b.shape}, types {a.dtype} and {b.dtype}"
    if a.dtype.kind != "f":
        return f"{int((a != b).sum())} of {a.size} entries differ"
    bits = a.view(f"u{a.itemsize}") != b.view(f"u{a.itemsize}")
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    numbers = bits & ~nan_a & ~nan_b
    with np.errstate(invalid="ignore"):
        gap = np.where(numbers, np.abs(a.astype(np.float64) - b.astype(np.float64)), 0.0)
    return (f"{int(bits.sum())} of {a.size} entries differ in their bits: {int((bits & nan_a & nan_b).sum())} are NaN in both,"
            f" {int((bits & (nan_a ^ nan_b)).sum())} NaN in one, {int((numbers & (a == b)).sum())} zeros of opposite sign and"
            f" {int((numbers & (a != b)).sum())} other numbers (largest difference {float(np.nan_to_num(gap, nan=np.inf).max()) if a.size else 0.0});"
            f" {_vector_extensions()}")


@pytest.fixture(scope="module", autouse=True)
def _small_blocks():
    """Blocks of ``BLOCK`` rows in both versions, and small pieces of the readers' sums in the current one."""
    blocks = staticmethod(lambda index, size=BLOCK: [index[start : start + size] for start in range(0, len(index), size)])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(R.ASRDressed, "_blocks", blocks)
        patch.setattr(D.ASRDressed, "_blocks", blocks)
        patch.setattr(D, "PIECE", 4_000)
        yield


def _table(n=260, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=list("abc"))
    y = (X["a"] + np.sin(2.0 * X["b"]) + 0.3 * rng.normal(size=n)).to_numpy()
    return X, y


def _case(model, settings):
    """A shallow copy of ``model`` with another gate, pool weight or fallback factor."""
    other = model.__class__.__new__(model.__class__)
    other.__dict__.update(model.__dict__)
    for name, value in settings.items():
        if name == "factors":
            apart = 0.5 if other.bandwidth_factor_ != 0.5 else 0.25
            other.fallback_factor_ = other.bandwidth_factor_ if value == "equal" else apart
        else:
            setattr(other, name, value)
    return other


class _AsCurrent(pickle.Unpickler):
    """Loads a pickle made by the released module into the current classes, as a pickle saved by 0.1.0 loads."""

    def find_class(self, module, name):
        return super().find_class("ashlar.dressed" if module == R.__name__ else module, name)


def _current(released, threads):
    """A released object, pickled and loaded as the current version would load it; a model reads with ``threads``."""
    other = _AsCurrent(io.BytesIO(pickle.dumps(released))).load()
    if threads is not None:
        other.threads = threads
    return other


def _values(y):
    """Three values per row around ``y``, with a NaN, infinities and a far value among them."""
    values = np.stack([y, y + 0.7, np.where(np.arange(len(y)) % 9 == 3, np.nan, y - 2.0)])
    values[1, :3] = [np.inf, -np.inf, y[2] + 1e3]
    return values


def _reads(model, X, y):
    """Every read of ``model.predict_dist(X)``, in a fixed order."""
    u = np.random.default_rng(1).uniform(1e-9, 1 - 1e-9, size=(7, len(y)))
    dist = model.predict_dist(X)
    reads = {"cdf": dist.cdf(_values(y)), "log_prob": dist.log_prob(_values(y))}
    if model.values_ is not None:
        reads["probabilities"] = dist.probabilities()
    reads["quantile at 3 levels, after a pickle"] = pickle.loads(pickle.dumps(dist)).quantile([0.1, 0.5, 0.9])
    reads["quantile at 203 levels"] = dist.quantile(D.LEVELS)
    reads["quantile at 7 levels per row"] = dist.quantile(u)
    reads["mean"] = dist.mean()
    reads["crps"] = dist.crps(y)
    reads["sample"] = dist.sample(5, seed=1)
    return reads


def _fitted(model):
    """Every number a fitted dressed model holds, as ``(what, value)``."""

    def arrays(tree):
        return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_array))]

    numbers = [model.pool_weight_, model.floor_, model.bandwidth_factor_, model.fallback_factor_, model.gate_, model.centre_,
               model.spread_, None if model.values_ is None else model.values_.tolist()]
    out = [("history_", repr(model.history_)), ("fitted numbers", repr(numbers))]
    for k, (network, reader, held) in enumerate(model.members_):
        out += [(f"member {k}'s held-out rows", held), (f"member {k}'s network history", repr(network.history_)),
                (f"member {k}'s reader steps", repr(reader.steps_))]
        out += [(f"member {k}'s network weights {i}", a) for i, a in enumerate(arrays(network.core_))]
        out += [(f"member {k}'s reader parameters {i}", a) for i, a in enumerate(arrays(reader.params_))]
    out += [(f"the corrector's weights {i}", a) for i, a in enumerate(arrays(model.corrector_.core_))]
    return out + [("the corrector's history", repr(model.corrector_.history_))]


def _assert_same_fit(model, released, label):
    got, expected = _fitted(model), _fitted(released)
    assert [what for what, _ in got] == [what for what, _ in expected], label
    for (what, value), (_, reference) in zip(got, expected, strict=True):
        same = value == reference if isinstance(value, str) else _same(value, reference)
        assert same, f"{label}: {what} differs" + ("" if isinstance(value, str) else f": {_differences(value, reference)}")


def _assert_same_reads(model, X, y, reads, cases, label):
    for case, settings in cases.items():
        got = _reads(_case(model, settings), X, y)
        for name, expected in reads[case].items():
            assert _same(got[name], expected), f"{label}, {case}, {name}: {_differences(got[name], expected)}"


# ------------------------------------------------------------------------------------------------- fast tests


def test_the_readers_sums_are_the_released_ones(monkeypatch):
    rng = np.random.default_rng(0)
    rows, width, centre, spread = 23, 97, 0.1, 1.3
    atoms = np.round(rng.normal(size=(rows, width)), 1)  # with ties
    masses = rng.dirichlet(np.full(width, 0.3), size=rows)
    masses[masses < 0.004] = 0.0  # each row keeps its own number of points
    masses[3] = np.where(np.arange(width) == 7, 1.0, 0.0)  # one point with all of a row's mass
    masses[5] = 1e-13  # no point above the cut: the row's sums are NaN
    with np.errstate(invalid="ignore"):
        new, old = D._Smoothed(atoms, masses, centre, spread, 0.5, 0.01), R._Smoothed(atoms, masses, centre, spread, 0.5, 0.01)
        for factor in (0.5, 0.125, 2.0):
            other, fresh = new.at_factor(factor), R._Smoothed(atoms, masses, centre, spread, factor, 0.01)
            assert all(_same(getattr(other, name), getattr(fresh, name)) for name in ("a", "p", "h", "lo", "hi"))
    index = np.concatenate([rng.integers(0, rows, size=150), np.full(140, 4), [5, 3], rng.integers(0, rows, size=10)])
    y = centre + spread * rng.normal(size=(len(index), 3))
    y[::37, 0], y[1::41, 1], y[2::43, 2], y[3, 0] = np.nan, np.inf, -np.inf, centre + 60 * spread
    for at in (y, y[:, :1]):
        with np.errstate(invalid="ignore"):
            reference = old.terms(index, at)
        for piece in (700, 20_000):  # pieces of a few entries, and pieces that hold many entries of one row
            monkeypatch.setattr(D, "PIECE", piece)
            for threads in (1, 3):
                for parts in ("F", "S", "f", "FS", "Ff", "Sf", "FSf"):
                    with np.errstate(invalid="ignore"):
                        got = new.terms(index, at, parts, threads)
                    for part, value, expected in zip("FSf", got, reference, strict=True):
                        if part not in parts:
                            assert value is None
                        else:
                            assert _same(value, expected), f"{part} of {parts}, {threads} threads, pieces of {piece} cells"
    assert all(_same(g, r[:1]) for g, r in zip(new.terms(index[:1], y[:1]), old.terms(index[:1], y[:1]), strict=True))


def test_the_bisection_stops_at_the_released_answer():
    rng = np.random.default_rng(0)
    flows = Mixture(Flow(rng.normal(size=(40, 4, flow_parameter_count() + 2)).astype(np.float32)))
    loc = jnp.asarray(3.0 * rng.normal(size=(40, 3)), dtype=jnp.float32)
    scale = jnp.asarray(rng.uniform(0.01, 2.0, size=(40, 3)), dtype=jnp.float32)
    levels = np.concatenate([[1e-7, 0.001, 0.0025], np.linspace(0.005, 0.995, 21), [0.999, 1.0 - 1e-7, np.nan]])
    levels = jnp.asarray(levels.reshape(-1, 1), dtype=jnp.float32)
    current, released = jax.jit(lambda dist, q: dist.quantile(q)), jax.jit(_released_bisection_quantile)
    for dist in (flows, Mixture(Normal(loc, scale))):
        assert _same(current(dist, levels), released(dist, levels))
    inner = levels[3:-1]

    def total(loc, quantile):
        return quantile(Mixture(Normal(loc, scale)), inner).sum()

    assert _same(jax.grad(total)(loc, lambda d, q: d.quantile(q)), jax.grad(total)(loc, _released_bisection_quantile))
    stacked = jnp.stack([loc, loc + 1.0, 2.0 * loc])
    assert _same(jax.vmap(lambda loc: Mixture(Normal(loc, scale)).quantile(inner))(stacked),
                 jax.vmap(lambda loc: _released_bisection_quantile(Mixture(Normal(loc, scale)), inner))(stacked))


def test_a_nan_level_is_refused():
    dist = D.DressedDistribution([], None)
    for levels in (np.nan, [0.5, np.nan], [[0.5], [np.nan]], 0.0, [0.5, 1.0]):
        with pytest.raises(AshlarError, match="strictly between 0 and 1"):
            dist.quantile(levels)


def test_the_threads_setting(monkeypatch):
    monkeypatch.delenv("ASHLAR_THREADS", raising=False)
    assert D.default_threads() == min(D.MAX_THREADS, D._cpus()) and D._check_threads(None) == D.default_threads()
    assert D._check_threads(3) == 3 and D._check_threads(-1) == D._cpus()
    monkeypatch.setenv("ASHLAR_THREADS", "2")
    assert D.default_threads() == 2 and D._check_threads(None) == 2
    for wrong in (0, -2, 1.5, "many"):
        with pytest.raises(AshlarError, match="threads"):
            D._check_threads(wrong)
    assert ASRDressed(threads=2).get_params()["threads"] == 2 and ashlar.dressed(threads=3).threads == 3
    assert ashlar.dressed(threads=3).network is None
    # A model or a distribution pickled before the setting existed has no ``threads`` of its own.
    model, dist = ASRDressed(members=3), D.DressedDistribution([], None)
    del model.__dict__["threads"], dist.__dict__["threads"]
    again = pickle.loads(pickle.dumps(model))
    assert model.threads is None and again.threads is None and dist.threads is None
    assert again.get_params()["threads"] is None and "threads" not in repr(again)


def test_runs_inside_runs_take_every_task_once_and_finish(monkeypatch):
    done, lock = [], threading.Lock()

    def task(k, threads, inner=False):
        def run():
            if inner:  # a task that itself runs tasks on the same pool, as two blocks of quantiles do
                D._run([task(1000 + 10 * k + j, threads) for j in range(5)], threads)
            time.sleep(0.001)
            with lock:
                done.append(k)
        return run

    for threads in (2, 3):
        # A new process's pool: no more workers than these runs ask for, so every worker can be a waiting caller.
        monkeypatch.setattr(D, "_POOL", {"pid": None, "executor": None, "workers": 0, "lock": threading.Lock(), "reads": None})
        done.clear()
        runner = threading.Thread(target=D._run, args=([task(k, threads, inner=k % 4 == 0) for k in range(40)], threads), daemon=True)
        runner.start()
        runner.join(60)
        assert not runner.is_alive(), f"runs inside runs on {threads} threads did not finish"
        assert sorted(done) == list(range(40)) + [1000 + 10 * k + j for k in range(0, 40, 4) for j in range(5)]


def test_after_a_failure_no_task_starts():
    started = []

    def fail():
        started.append("failed")
        raise ValueError("this task fails")

    def slow():
        started.append("slow")
        time.sleep(0.05)

    with pytest.raises(ValueError, match="this task fails"):
        D._run([fail] + [slow] * 20, 2)
    assert 1 <= len(started) <= 2  # the other thread finishes the task it holds and starts no other


# ------------------------------------------------------------------------------------------------- slow tests


DENSITY_CASES = {
    "as fitted": {},
    "gate 0": {"gate_": 0.0, "pool_weight_": 0.5, "factors": "apart"},
    "gate 0.6": {"gate_": 0.6, "pool_weight_": 0.5, "factors": "apart"},
    "gate 1": {"gate_": 1.0, "pool_weight_": 0.5, "factors": "apart"},
    "gate 0.6, factors equal": {"gate_": 0.6, "pool_weight_": 0.5, "factors": "equal"},
}
VALUE_CASES = {"as fitted": {}, "gate 0": {"gate_": 0.0, "pool_weight_": 0.5, "factors": "apart"},
               "gate 0.6": {"gate_": 0.6, "pool_weight_": 0.5, "factors": "apart"}, "gate 1": {"gate_": 1.0, "pool_weight_": 0.5}}


@pytest.fixture(scope="module")
def density():
    X, y = _table()
    with _released_code() as traced:
        released = R.ASRDressed(members=2, network=NETWORK, reader=READER, reader_rows=64).fit(X.iloc[:200], y[:200])
        reads = {case: _reads(_case(released, settings), X.iloc[200:], y[200:]) for case, settings in DENSITY_CASES.items()}
        saved = pickle.dumps(_case(released, DENSITY_CASES["gate 0.6"]).predict_dist(X.iloc[200:]))
    assert traced["bisection"] and traced["attention"]  # the released functions ran
    return X, y, released, reads, saved


@pytest.fixture(scope="module")
def few_values():
    X, y = _table(seed=1)
    X["kind"] = np.where(X["c"] > 0, "up", "down")  # read by the networks and, by code, by the readers
    # Adding +0.0 turns the -0.0 that rounding gives into +0.0: with both zeros among the labels, the released fit's
    # np.unique keeps either one, so its ``values_`` holds a zero of either sign from one fit to the next.
    y = np.round(np.clip(y, -2.0, 2.0)) + 0.0
    with _released_code() as traced:
        released = R.ASRDressed(members=3, network=NETWORK, reader=READER).fit(X.iloc[:200], y[:200])
        reads = {case: _reads(_case(released, settings), X.iloc[200:], y[200:]) for case, settings in VALUE_CASES.items()}
    assert traced["attention"]
    return X, y, released, reads


@pytest.mark.slow
def test_a_density_fit_and_every_read_are_the_released_ones(density):
    X, y, released, reads, saved = density
    assert released.values_ is None
    for threads in (1, 3):  # the released model, saved and loaded by the current version
        _assert_same_reads(_current(released, threads), X.iloc[200:], y[200:], reads, DENSITY_CASES, f"{threads} threads")
    dist = _AsCurrent(io.BytesIO(saved)).load()  # a distribution saved by 0.1.0
    assert isinstance(dist, D.DressedDistribution) and dist.threads is None
    expected = reads["gate 0.6"]
    assert _same(dist.cdf(_values(y[200:])), expected["cdf"]) and _same(dist.log_prob(_values(y[200:])), expected["log_prob"])
    assert _same(dist.quantile(D.LEVELS), expected["quantile at 203 levels"])
    for threads in (1, 3):
        model = ASRDressed(members=2, network=NETWORK, reader=READER, reader_rows=64, threads=threads).fit(X.iloc[:200], y[:200])
        _assert_same_fit(model, released, f"the fit on {threads} threads")
    with pytest.MonkeyPatch.context() as patch:  # as on a table too large for the fit to keep its held-out reads
        patch.setattr(D, "KEPT_CELLS", 0)
        model = ASRDressed(members=2, network=NETWORK, reader=READER, reader_rows=64, threads=3).fit(X.iloc[:200], y[:200])
    _assert_same_fit(model, released, "the fit that reads its held-out rows again")


@pytest.mark.slow
def test_a_few_values_fit_and_every_read_are_the_released_ones(few_values):
    X, y, released, reads = few_values
    assert released.values_ is not None
    _assert_same_reads(_current(released, 3), X.iloc[200:], y[200:], reads, VALUE_CASES, "3 threads")
    model = ASRDressed(members=3, network=NETWORK, reader=READER, threads=2).fit(X.iloc[:200], y[:200])
    _assert_same_fit(model, released, "the fit on 2 threads")
    with pytest.MonkeyPatch.context() as patch:  # as on a table too large for the fit to keep its held-out reads
        patch.setattr(D, "KEPT_CELLS", 0)
        model = ASRDressed(members=3, network=NETWORK, reader=READER, threads=2).fit(X.iloc[:200], y[:200])
    _assert_same_fit(model, released, "the fit that reads its held-out rows again")
    # Labels with zeros of both signs: the fit reports 0.0 and is otherwise the fit on the labels above.
    mixed = np.where((y == 0.0) & (np.arange(len(y)) % 2 == 0), -0.0, y)
    zeros = mixed[:200][mixed[:200] == 0.0]
    assert np.signbit(zeros).any() and not np.signbit(zeros).all()
    model = ASRDressed(members=3, network=NETWORK, reader=READER, threads=2).fit(X.iloc[:200], mixed[:200])
    assert not np.signbit(model.values_[model.values_ == 0.0]).any()
    _assert_same_fit(model, released, "the fit on labels with zeros of both signs")
