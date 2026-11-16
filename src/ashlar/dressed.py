"""Dressing: several cross-fitted networks and label readers, pooled, smoothed and corrected into one distribution.

A network with a density head fits a smooth map from the columns to the target. A label reader
(``ashlar.label_reader``) predicts a row from the labels of the training rows it reads. Neither wins on every table,
and a reader predicts point masses, so a pool of the two has no density. ``ASRDressed`` puts them together in one
fit and still returns one distribution with an exact CDF and an exact log density:

1. Members. The training rows are cut into ``members`` parts. Member ``k`` holds out part ``k`` and fits a network
   and a reader on the other parts, both stopped on part ``k``. The reader trains on at most ``reader_rows`` of
   those rows and is stopped on at most as many of part ``k``. A target with at most ``few_values`` distinct values
   gets an ordinal network over those values, stopped on log loss, and a held-out row whose value the other parts
   lack moves into training. Any other target gets a ``flow`` network stopped on CRPS.
2. Pool weight. One weight ``w`` for the network against the reader, from a grid of 21, by the CRPS of every
   held-out row under the member that held it out. The pool scored is the unsmoothed one, with the network read as
   atoms at ``LEVELS``; a tie goes to the smaller weight.
3. Smoothing. Each reader's points become normals, with Silverman's rule on the row's weighted points (Kish's
   effective sample size ``1 / sum(p^2)`` as the count) times a factor from a grid of five. Each smoothed reader
   gives 1% of its mass to a wide normal at the training labels' mean and spread, so ``0.01 * (1 - w)`` of the
   pool. One factor is picked by the log score of every held-out row and one by the CRPS of ``CRPS_ROWS`` held-out
   rows per member. A member's pool, ``w`` times its network plus ``1 - w`` times its smoothed reader, now has a
   density.
4. Corrector. One more network, reading the columns only, learns where a label falls inside its own row's pool. It
   is fitted by likelihood to the label's normal score under the pool, smoothed at the log-score factor, of the
   member that held the row out, on the held-out rows of every member but the last. It keeps the epoch with the
   lowest CRPS, in ``y``, of the corrected pools on the last member's held-out rows. A member's corrected CDF at
   ``y`` is the corrector's CDF at the normal score of ``y`` under that member's pool.
5. Gate. One weight, from the same grid, for the corrected pools against the pools smoothed at the CRPS factor:
   the one with the lowest CRPS on the last member's held-out rows, the rows that stopped the corrector, with a tie
   going to the smaller weight. The CRPS of a mixture can be below that of both its parts, so the gate can be
   positive where the corrected pools alone score worse.

The prediction is the gate times the equal mixture of the members' corrected pools, plus the rest times the equal
mixture of their smoothed pools. At prediction each reader reads its held-out rows as well as its training rows;
every number above was picked with readers that read their training rows only.

The members are cross-fitted, the pool is a linear pool whose weight is picked on held-out rows (stacking), the
smoothing is kernel dressing, and the corrector and gate follow the PIT boosting step of the condensite package,
here on normal scores, with one network and a weight picked by CRPS. What this module adds is the label reader and
its pairing with a network in the pool. ``docs/dressed.md`` has the references.

The fit is eleven networks' worth of training (five networks, five readers, one corrector), so it is the slowest
model in Ashlar. One target column, exactly observed; no sample weights.
"""

from __future__ import annotations

import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Empty, SimpleQueue

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri
from sklearn.base import BaseEstimator, RegressorMixin

from ashlar import observed
from ashlar.errors import AshlarError
from ashlar.estimator import ASRModel
from ashlar.label_reader import POOL_WEIGHTS, LabelReader, mixture_crps, mixture_quantiles

# A density is read as atoms at these levels wherever a CRPS is searched over a grid; each atom carries the mass
# up to its level.
LEVELS = np.concatenate([[0.001, 0.0025], np.linspace(0.005, 0.995, 199), [0.9975, 0.999]])
LEVEL_MASS = np.diff(np.concatenate([[0.0], LEVELS]))
LEVEL_MASS[-1] += 1.0 - LEVELS[-1]
FACTORS = (1.0, 0.5, 0.25, 0.125, 2.0)  # bandwidth factors; the first wins a tie
FLOOR = 0.01  # the wide normal's share of a smoothed reader, at most
Z_GRID = np.linspace(-7.0, 7.0, 113)  # the stopping rows' pools are read at these normal scores
CRPS_ROWS = 120  # held-out rows per member in the CRPS factor's search
PIECE = 131_072  # array cells (entries x atoms) per piece of work when a smoothed reader is evaluated
PREDICT_ROWS = 512  # rows per piece at prediction time
MEMBER_SETTINGS = {"max_epochs": 1000, "cooldown_epochs": 3}
MAX_THREADS = 8  # the most threads a fit or a prediction uses unless told otherwise
KEPT_CELLS = 32_000_000  # a fit keeps its held-out reads up to this many reader cells (rows x points), about 0.5 GB

_READ = None
_POOL = {"pid": None, "executor": None, "workers": 0, "lock": threading.Lock(), "reads": None}


def default_threads() -> int:
    """The threads a fit or a prediction uses when ``threads`` is ``None``: the ``ASHLAR_THREADS`` environment
    variable if it is set, otherwise the CPUs this process may run on, at most ``MAX_THREADS``."""
    setting = os.environ.get("ASHLAR_THREADS", "").strip()
    if setting:
        return _check_threads(setting)
    return max(1, min(MAX_THREADS, _cpus()))


def _cpus():
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS and Windows have no affinity call
        return os.cpu_count() or 1


def _check_threads(threads) -> int:
    """``threads`` as a count: a positive number, ``-1`` for every CPU this process may run on, or ``None`` for
    ``default_threads()``."""
    if threads is None:
        return default_threads()
    try:
        count = int(threads)
    except (TypeError, ValueError) as error:
        raise AshlarError(f"threads must be a positive number, -1 or None; received {threads!r}.") from error
    if count == -1:
        return _cpus()
    if count < 1 or count != float(threads):
        raise AshlarError(f"threads must be a positive number, -1 or None; received {threads!r}.")
    return count


def _executor(workers):
    """The process's one pool of worker threads, with at least ``workers`` of them. It is made on first use, grows
    when more are asked for, and is made anew in a child process after a fork. Nothing holds it but this module, so
    it never reaches a pickle."""
    with _POOL["lock"]:
        if _POOL["pid"] != os.getpid() or _POOL["workers"] < workers:
            # A replaced pool is not shut down: a caller may still hold it, and its idle threads end with it.
            _POOL.update(pid=os.getpid(), workers=workers, executor=ThreadPoolExecutor(workers, thread_name_prefix="ashlar-dressed"))
        return _POOL["executor"]


def _reads():
    """A few threads that only start JAX reads and wait for them: XLA runs reads from different threads at once, but
    one thread's reads one after another."""
    with _POOL["lock"]:
        if _POOL["reads"] is None or _POOL["reads"][0] != os.getpid():
            _POOL["reads"] = (os.getpid(), ThreadPoolExecutor(8, thread_name_prefix="ashlar-dressed-read"))
        return _POOL["reads"][1]


def _after_fork():
    _POOL.update(pid=None, executor=None, workers=0, lock=threading.Lock(), reads=None)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _run(tasks, threads):
    """Call every task, at most ``threads`` at a time, the calling thread being one of them; each thread takes the
    next task when it is free. Each task writes its own entries of its outputs and computes them as it would alone,
    so no result depends on ``threads`` or on which thread ran what. Once a task fails, no thread starts another,
    and the failure is raised when the tasks already running are done."""
    lanes = min(int(threads), len(tasks))
    if lanes <= 1:
        for task in tasks:
            task()
        return
    queue, failed = SimpleQueue(), threading.Event()
    for task in tasks:
        queue.put(task)

    def lane():
        try:
            while not failed.is_set():
                try:
                    task = queue.get_nowait()
                except Empty:
                    return
                task()
        except BaseException:
            failed.set()
            raise

    executor = _executor(lanes - 1)
    futures = [executor.submit(lane) for _ in range(lanes - 1)]
    try:
        lane()
    finally:
        # A lane that never started has nothing left to take, so it is cancelled and not waited for: a cancelled
        # future counts as done only once a worker takes it off the queue, and every worker may be a caller waiting
        # here, in a run started from inside another. A lane that started ends when the queue is empty.
        wait([future for future in futures if not future.cancel()])
    for future in futures:
        if not future.cancelled():
            future.result()


def _reader():
    """The compiled read of a network distribution's CDF and log density, made once (compiled once per shape)."""
    global _READ
    if _READ is None:
        import jax

        _READ = jax.jit(lambda dist, t: (dist.cdf(t), dist.log_prob(t)))
    return _READ


def _read(dist, t):
    """A network distribution's CDF and log density at ``t``, compiled once per shape."""
    return _reader()(dist, t)


def _take(X, index):
    return X.iloc[index] if hasattr(X, "iloc") else X[index]


def atoms_crps(atoms, masses, y):
    """Exact CRPS of each row's distribution on ``atoms`` (rows, m) with ``masses`` (rows, m), at ``y`` (rows,)."""
    order = np.argsort(atoms, axis=1)
    t = np.take_along_axis(atoms, order, 1)
    cdf = np.cumsum(np.take_along_axis(masses, order, 1), axis=1)[:, :-1]
    gap = np.diff(t, axis=1)
    below = np.clip(y[:, None] - t[:, :-1], 0.0, gap)
    return (cdf**2 * below + (1 - cdf) ** 2 * (gap - below)).sum(1) + np.maximum(t[:, 0] - y, 0) + np.maximum(y - t[:, -1], 0)


def _level_crps(atoms, masses, y):
    """The CRPS of each row's ``(atoms, masses)`` read through its quantiles at ``LEVELS``."""
    q = mixture_quantiles((atoms, masses), (atoms, masses), 1.0, LEVELS)
    return atoms_crps(q, np.broadcast_to(LEVEL_MASS, q.shape), y)


def _value_masses(F):
    """Each value's probability from the CDF ``F`` (rows, values - 1) at the values below the top one."""
    F = np.clip(F, 0.0, 1.0)
    return np.diff(np.concatenate([np.zeros((len(F), 1)), F, np.ones((len(F), 1))], axis=1), axis=1)


def _curve(first, second, y, chunk=100):
    """Mean CRPS of ``w * first + (1 - w) * second`` at every ``w`` in ``POOL_WEIGHTS``, a few rows at a time."""
    total = np.zeros(len(POOL_WEIGHTS))
    for start in range(0, len(y), chunk):
        rows = slice(start, start + chunk)
        total += len(y[rows]) * mixture_crps((first[0][rows], first[1][rows]), (second[0][rows], second[1][rows]), y[rows])
    return total / len(y)


def _union(models):
    """The equal mixture of ``(atoms, masses)`` models, as one ``(atoms, masses)``."""
    return (np.concatenate([np.asarray(a) for a, _ in models], axis=1),
            np.concatenate([np.asarray(m) for _, m in models], axis=1) / len(models))


def _combine(net, reader, weight):
    """Network atoms and reader atoms as one ``(atoms, masses)`` at ``weight`` on the network."""
    return (np.concatenate([np.asarray(net[0]), reader[0]], axis=1),
            np.concatenate([weight * np.asarray(net[1]), (1 - weight) * reader[1]], axis=1))


def _network_quantiles(model, network, X):
    """A member network's quantiles at ``LEVELS`` as ``(atoms, masses)``, read from ``network``, its ``predict_dist``
    on ``X``, block by block as ``predict_quantiles`` reads them, without running the network a second time."""
    if getattr(model, "calibration_", None) is not None:
        q = np.asarray(model.predict_quantiles(X, LEVELS), dtype=np.float64)
    else:
        import jax.numpy as jnp

        from ashlar.estimator import _chunks, _jit_quantile

        levels = jnp.asarray(tuple(float(level) for level in LEVELS), dtype=jnp.float32)[:, None]
        q = np.concatenate([np.asarray(_jit_quantile()(part, levels), dtype=np.float64).T
                            for _, part in _chunks(network, model._chunk_rows())], axis=0)
    return q, np.broadcast_to(LEVEL_MASS, q.shape)


def _interp_rows(levels, table, u):
    """Each row's ``table`` (rows, len(levels)) read at its own probabilities ``u`` (rows, k)."""
    return np.stack([np.interp(row_u, levels, row) for row_u, row in zip(u, table, strict=True)])


def _normal_score(F, S):
    """Phi^-1 of a CDF value, read from the nearer tail (``S`` is the survival)."""
    return np.where(F < 0.5, ndtri(np.clip(F, 1e-300, 1.0)), -ndtri(np.clip(S, 1e-300, 1.0)))


def _floor_share(y, centre, spread, held_out_crps, cost):
    """The wide normal's share of a smoothed reader: ``FLOOR``, or less where that would cost more than ``cost``
    times the raw pool's held-out CRPS. A share ``e`` mixed into a pool that is a point mass at the label costs
    ``e^2`` times the wide normal's own CRPS there. The share of the pool is ``e * (1 - w)``; the rule leaves that
    factor out, so it errs low."""
    if cost is None:
        return FLOOR
    score = (y - centre) / spread
    wide = spread * (score * (2 * ndtr(score) - 1) + 2 * np.exp(-0.5 * score**2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))
    return float(min(FLOOR, np.sqrt(float(cost) * held_out_crps / wide.mean())))


class _Smoothed:
    """Rows' reader distributions as densities: ``sum_j p_j N(a_j, h^2)`` at weight ``1 - floor`` plus ``floor`` of
    ``N(centre, spread^2)``. ``h`` is Silverman's rule on the row's weighted atoms, with Kish's effective sample size
    ``1 / sum_j p_j^2`` as the count, times ``factor``."""

    def __init__(self, atoms, masses, centre, spread, factor, floor=FLOOR):
        atoms, masses = np.asarray(atoms, np.float64), np.asarray(masses, np.float64)
        masses = np.where(masses >= 1e-12, masses, 0.0)
        width = int((masses > 0).sum(axis=1).max())
        top = np.argsort(-masses, axis=1)[:, :width]
        a, p = np.take_along_axis(atoms, top, 1), np.take_along_axis(masses, top, 1)
        p /= p.sum(axis=1, keepdims=True)
        mean = (p * a).sum(axis=1)
        sd = np.sqrt((p * (a - mean[:, None]) ** 2).sum(axis=1))
        order = np.argsort(a, axis=1)
        ordered, cum = np.take_along_axis(a, order, 1), np.cumsum(np.take_along_axis(p, order, 1), axis=1)
        rows = np.arange(len(a))
        iqr = (ordered[rows, np.minimum((cum < 0.75).sum(1), width - 1)]
               - ordered[rows, np.minimum((cum < 0.25).sum(1), width - 1)])
        scale = np.where(iqr > 0, np.minimum(sd, iqr / 1.349), sd)
        self._rule = np.maximum(0.9 * scale * (1.0 / (p**2).sum(axis=1)) ** -0.2, 1e-5 * spread)  # h at factor 1
        self.a, self.p, self.centre, self.spread, self.floor = a, p, centre, spread, float(floor)
        self._first, self._last = ordered[:, 0].copy(), ordered[:, -1].copy()
        self._set_factor(factor)

    def _set_factor(self, factor):
        self.h = np.maximum(self._rule * factor, 1e-5 * self.spread)
        self.lo = np.minimum(self._first - 40 * self.h, self.centre - 40 * self.spread)
        self.hi = np.maximum(self._last + 40 * self.h, self.centre + 40 * self.spread)

    def at_factor(self, factor):
        """The same rows smoothed at another bandwidth factor: the numbers of a fresh build, without its sorts. The
        atoms and masses are shared, not copied."""
        other = copy.copy(self)
        other._set_factor(factor)
        return other

    def terms(self, rows, y, parts="FSf", threads=1):
        """The CDF, survival and density at ``y`` (len(rows), k) for those rows; ``None`` for a part left out of
        ``parts``."""
        return _kernel_sums([(self, rows, y, parts)], threads)[0]

    def _ends(self):
        """Each row's atoms that carry mass come first (they are ordered by mass); this is one past the last. A row
        with no mass (its masses are then NaN) keeps every atom, as the full sum would."""
        ends = getattr(self, "_mass_ends", None)
        if ends is None:
            carried = self.p > 0
            ends = np.where(carried.any(axis=1), carried.shape[1] - np.argmax(carried[:, ::-1], axis=1), carried.shape[1])
            self._mass_ends = ends
        return ends

    def _pieces(self, rows, y, parts, out):
        """The work of ``terms`` as tasks, each on at most ``PIECE`` cells and, where a row has many entries, on
        that row alone."""
        size = max(1, PIECE // (y.shape[1] * self.a.shape[1]))
        self._ends()  # made once here, not by every task
        bounds, start = [], 0
        if len(rows):
            change = np.flatnonzero(rows[1:] != rows[:-1]) + 1
            for first, last in zip(np.concatenate([[0], change]), np.concatenate([change, [len(rows)]]), strict=True):
                if last - start <= size:
                    continue  # this row's entries join the open piece
                if first > start:
                    bounds.append((start, first))
                    start = first
                while last - start > size:
                    bounds.append((start, start + size))
                    start += size
            bounds.append((start, len(rows)))
        return [lambda bound=bound: self._sum_piece(rows, y, parts, out, *bound) for bound in bounds]

    def _sum_piece(self, rows, y, parts, out, start, stop):
        """The kernel sums of entries ``start:stop`` before the floor. Each entry's terms, their order and their sum
        are those of the plain sum over all of the row's atoms: the atoms after a row's last atom with mass each add
        an exact 0, so they are written as 0 without being evaluated, and a row read by many entries is broadcast,
        not copied per entry."""
        r = rows[start:stop]
        width = self.a.shape[1]
        if (r == r[0]).all():
            row = r[0]
            end = int(self._ends()[row])
            p = self.p[row, :end]
            x = (y[start:stop, :, None] - self.a[row, :end]) / self.h[row]
            scale = np.sqrt(2 * np.pi) * self.h[row]
        else:
            end = int(self._ends()[r].max())
            p = self.p[r, None, :end]
            x = (y[start:stop, :, None] - self.a[r, None, :end]) / self.h[r, None, None]
            scale = np.sqrt(2 * np.pi) * self.h[r, None]
        products = np.zeros(x.shape[:-1] + (width,))
        kept, scratch = products[..., :end], np.empty(x.shape)
        if "F" in parts:
            np.multiply(p, ndtr(x, out=scratch), out=kept)
            out["F"][start:stop] = products.sum(-1)
        if "S" in parts:
            np.multiply(p, ndtr(np.negative(x, out=scratch), out=scratch), out=kept)
            out["S"][start:stop] = products.sum(-1)
        if "f" in parts:
            np.multiply(x, -0.5, out=scratch)
            np.multiply(scratch, x, out=scratch)
            np.multiply(p, np.exp(scratch, out=scratch), out=kept)
            out["f"][start:stop] = products.sum(-1) / scale

    def _finish(self, y, out):
        """The floor's share added to the sums in ``out``."""
        b, floor = (y - self.centre) / self.spread, self.floor
        F = (1 - floor) * out["F"] + floor * ndtr(b) if "F" in out else None
        S = (1 - floor) * out["S"] + floor * ndtr(-b) if "S" in out else None
        f = (1 - floor) * out["f"] + floor * np.exp(-0.5 * b * b) / (np.sqrt(2 * np.pi) * self.spread) if "f" in out else None
        return F, S, f


def _kernel_sums(jobs, threads=1):
    """``_Smoothed.terms`` for several jobs ``(smoothed, rows, y, parts)`` in one batch of work on ``threads``."""
    outs, tasks = [], []
    for smoothed, rows, y, parts in jobs:
        out = {part: np.empty(y.shape) for part in "FSf" if part in parts}
        outs.append(out)
        tasks.extend(smoothed._pieces(np.asarray(rows), y, parts, out))
    _run(tasks, threads)
    return [smoothed._finish(y, out) for (smoothed, _, y, _), out in zip(jobs, outs, strict=True)]


class _Pool:
    """One member's pooled distribution on some rows: ``weight`` times the network plus the rest times the smoothed
    reader. ``network`` is the network's ``predict_dist`` on the rows, or, for a target with few values, its
    probability of each of ``values`` (rows, values)."""

    def __init__(self, network, reader, weight, values=None):
        self.network, self.reader, self.w, self.values = network, reader, float(weight), values

    def at_factor(self, factor):
        """The same pool with its reader smoothed at another bandwidth factor (``_Smoothed.at_factor``)."""
        return _Pool(self.network, self.reader.at_factor(factor), self.w, self.values)

    def network_terms(self, y):
        """The network's CDF and density at ``y`` (rows, k)."""
        if self.values is not None:
            F = (self.network[:, None, :] * (self.values <= y[:, :, None])).sum(-1)
            return F, np.zeros_like(F)
        import jax.numpy as jnp

        F, log_f = _read(self.network, jnp.asarray(y.T))
        return np.asarray(F, np.float64).T, np.exp(np.asarray(log_f, np.float64).T)

    def terms(self, y, parts="FSf", threads=1):
        """The pooled CDF, survival and density at ``y`` (rows, k); ``None`` for a part left out of ``parts``."""
        Fn, fn = self.network_terms(y)
        return self._pooled(Fn, fn, self.reader.terms(np.arange(len(y)), y, parts, threads))

    def terms_at(self, i, j, y, parts="FSf", threads=1):
        """``terms`` at the entries (i, j) only; the network is read on every row at once, the reader at those."""
        Fn, fn = (t[i, j] for t in self.network_terms(y))
        reader = self.reader.terms(i, y[i, j][:, None], parts, threads)
        return self._pooled(Fn, fn, [None if t is None else t[:, 0] for t in reader])

    def _pooled(self, Fn, fn, reader):
        Fr, Sr, fr = reader
        return (None if Fr is None else self.w * Fn + (1 - self.w) * Fr,
                None if Sr is None else self.w * (1 - Fn) + (1 - self.w) * Sr,
                None if fr is None else self.w * fn + (1 - self.w) * fr)

    def z(self, y, threads=1):
        """The normal score of ``y`` (rows, k) under the pool."""
        F, S, _ = self.terms(y, "FS", threads)
        return _normal_score(F, S)

    def network_log_pdf(self, y):
        """The network's log density at ``y`` (rows, k)."""
        import jax.numpy as jnp

        return np.asarray(_read(self.network, jnp.asarray(y.T))[1], np.float64).T

    def log_pdf(self, y, net=None, threads=1):
        """The pooled log density at ``y`` (rows, k), summed in logs so a far tail keeps its value; a density only.
        ``net`` is ``network_log_pdf(y)`` when it is already read."""
        net = self.network_log_pdf(y) if net is None else net
        reader = np.log(np.maximum(self.reader.terms(np.arange(len(y)), y, "f", threads)[2], 1e-300))
        with np.errstate(divide="ignore"):
            return np.logaddexp(np.log(self.w) + net, np.log(1 - self.w) + reader)

    def probabilities(self, threads=1):
        """Each value's pooled probability (rows, values); a target with few values only."""
        mids = np.broadcast_to((self.values[:-1] + self.values[1:]) / 2, (len(self.network), len(self.values) - 1)).copy()
        F = self.reader.terms(np.arange(len(mids)), mids, "F", threads)[0]
        return self.w * self.network + (1 - self.w) * _value_masses(F)


def _solve(pools, u, start, evaluate, tolerance):
    """Each row's ``y`` where an increasing CDF reaches ``u`` (rows, k), from ``start``: Newton steps inside a
    bisection bracket wide enough for every pool. ``evaluate(i, j, y)`` gives the CDF and density at the unsolved
    entries."""
    n, k = u.shape
    spread = pools[0].reader.spread
    y = np.array(start, dtype=np.float64)
    lo = np.repeat(np.minimum(np.min([p.reader.lo for p in pools], axis=0), y.min(axis=1) - spread)[:, None], k, axis=1)
    hi = np.repeat(np.maximum(np.max([p.reader.hi for p in pools], axis=0), y.max(axis=1) + spread)[:, None], k, axis=1)
    y = np.clip(y, lo, hi)
    active = np.ones((n, k), dtype=bool)
    for _ in range(100):
        i, j = np.nonzero(active)
        F, f = evaluate(i, j, y)
        g = F - u[i, j]
        done = (np.abs(g) <= tolerance) | (hi[i, j] - lo[i, j] <= 1e-12 * np.maximum(np.abs(y[i, j]), spread))
        active[i[done], j[done]] = False
        i, j, g, f = i[~done], j[~done], g[~done], f[~done]
        if not len(i):
            break
        lo[i, j] = np.where(g < 0, y[i, j], lo[i, j])
        hi[i, j] = np.where(g > 0, y[i, j], hi[i, j])
        step = y[i, j] - g / np.maximum(f, 1e-300)
        inside = (step > lo[i, j]) & (step < hi[i, j]) & np.isfinite(step)
        y[i, j] = np.where(inside, step, 0.5 * (lo[i, j] + hi[i, j]))
    return y


def _pool_quantiles(pools, u, start, threads=1):
    """Quantiles at ``u`` (rows, k) of the equal mixture of ``pools``."""

    def evaluate(i, j, y):
        terms = [p.terms_at(i, j, y, "Ff", threads) for p in pools]
        return np.mean([t[0] for t in terms], axis=0), np.mean([t[2] for t in terms], axis=0)

    return _solve(pools, u, start, evaluate, 1e-9 + 2e-7 * max(p.w for p in pools))  # the network's CDF is float32


class _Started:
    """A network's CDF and log density at ``y`` (rows, k), started at once (from a thread of its own when ``threads``
    is above 1, so several reads run together); ``at`` waits for them and reads the entries (i, j). It reads every
    row and level, as ``_Pool.network_terms`` does: a network's float32 results depend on the shape it reads."""

    def __init__(self, network, y, threads=1):
        import jax.numpy as jnp

        t, read = jnp.asarray(y.T), _reader()  # the compiled read is made here, in one thread
        if threads > 1:
            self._future = _reads().submit(lambda: tuple(np.asarray(v) for v in read(network, t)))
        else:
            self._future, self._values = None, read(network, t)

    def values(self):
        """The CDF and the log density on every entry, ``(k, rows)``."""
        if self._future is not None:
            self._values, self._future = self._future.result(), None
        return self._values

    def at(self, i, j):
        """The CDF and the log density at the entries (i, j), as float64."""
        F, log_f = self.values()
        return np.asarray(F)[j, i].astype(np.float64), np.asarray(log_f)[j, i].astype(np.float64)


def _scores(G, upper_survival):
    """``_normal_score`` with the survival given only where it is read: at the entries whose CDF is not below 0.5."""
    z = np.empty(G.shape)
    lower = G < 0.5
    z[lower] = ndtri(np.clip(G[lower], 1e-300, 1.0))
    z[~lower] = -ndtri(np.clip(upper_survival, 1e-300, 1.0))
    return z


class _Corrector(ASRModel):
    """``ASRModel`` on the labels' normal scores under their pools, stopped on the stopping rows' CRPS in ``y``.

    ``stop_on`` is set before ``fit``: ``("flow", y at Z_GRID per stopping row, None, y)`` or
    ``("values", normal scores of the cell edges per stopping row, values, y)``.
    """

    stop_on = None

    def _validation_score(self, validation):
        import equinox as eqx
        import jax.numpy as jnp

        from ashlar.distributions import CRPS_LEVELS, Mixture, head_distribution
        from ashlar.train import EVAL_CHUNK_ROWS

        *features, _, _ = validation
        kind, table, values, y = self.stop_on
        task, n = self.task, len(y)
        pieces = [(s, tuple(a[s : s + EVAL_CHUNK_ROWS] for a in features)) for s in range(0, n, EVAL_CHUNK_ROWS)]
        if kind == "flow":
            alphas = (np.arange(CRPS_LEVELS) + 0.5) / CRPS_LEVELS
            levels = jnp.asarray(alphas.reshape(-1, 1), dtype=jnp.float32)

            @eqx.filter_jit
            def quantiles(core, *chunk):
                return Mixture(head_distribution(task, core(*chunk, key=None))).quantile(levels)

            def score(core):
                mean, std = (float(np.asarray(v, np.float64).reshape(-1)[0]) for v in self._target_affine())
                z = mean + std * np.concatenate([np.asarray(quantiles(core, *c), dtype=np.float64) for _, c in pieces], axis=1)
                position = np.clip((z.T - Z_GRID[0]) / (Z_GRID[1] - Z_GRID[0]), 0.0, len(Z_GRID) - 1.000001)
                index = np.floor(position).astype(int)
                frac, rows = position - index, np.arange(n)[:, None]
                gap = y[None, :] - (table[rows, index] * (1 - frac) + table[rows, index + 1] * frac).T
                return -float((2.0 * np.maximum(alphas[:, None] * gap, (alphas[:, None] - 1.0) * gap)).mean())

            return score

        @eqx.filter_jit
        def cdf(core, edges, *chunk):
            return Mixture(head_distribution(task, core(*chunk, key=None))).cdf(edges)

        def score(core):
            mean, std = (float(np.asarray(v, np.float64).reshape(-1)[0]) for v in self._target_affine())
            F = np.concatenate([np.asarray(cdf(core, jnp.asarray(((table[s : s + EVAL_CHUNK_ROWS] - mean) / std).T, dtype=jnp.float32), *c),
                                           dtype=np.float64) for s, c in pieces], axis=1).T
            return -float(atoms_crps(np.broadcast_to(values, (n, len(values))), _value_masses(F), y).mean())

        return score


class _Columns:
    """The columns as float64 for the reader: numbers as they are, any other column by the rank of its value among
    the training values (missing or unseen is NaN)."""

    def __init__(self, X):
        self.codes = {}
        if hasattr(X, "iloc"):
            for j in range(X.shape[1]):
                column = X.iloc[:, j]
                if getattr(column.dtype, "kind", "O") not in "biuf":
                    kept = column[~column.isna().to_numpy()]
                    self.codes[j] = np.unique(np.asarray([str(v) for v in kept.to_numpy(dtype=object)], dtype=object).astype(str))

    def __call__(self, X):
        if not hasattr(X, "iloc"):
            try:
                return np.asarray(X, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise AshlarError("pass a DataFrame when a column is not numeric.") from error
        out = np.empty((len(X), X.shape[1]), dtype=np.float64)
        for j in range(X.shape[1]):
            column = X.iloc[:, j]
            if j not in self.codes:
                out[:, j] = column.to_numpy(dtype=np.float64, na_value=np.nan)
                continue
            known, missing = self.codes[j], column.isna().to_numpy()
            text = np.asarray([str(v) for v in column.to_numpy(dtype=object)], dtype=object).astype(str)
            if not len(known):
                out[:, j] = np.nan
                continue
            index = np.minimum(np.searchsorted(known, text), len(known) - 1)
            out[:, j] = np.where((known[index] == text) & ~missing, index, np.nan)
        return out


class DressedDistribution:
    """A dressed model's prediction for some rows: what ``ASRDressed.predict_dist`` returns.

    One distribution per row. A value array's last axis runs over the rows, as with every Ashlar distribution:
    ``cdf(y)`` and ``log_prob(y)`` take ``(rows,)`` or ``(k, rows)``. ``log_prob`` is the exact log density, or
    the log probability for a target with few values (``values`` is then the array of values, and
    ``probabilities()`` each row's probability of each one).

    ``threads`` is how many threads a read may use (``None``: ``default_threads()``); no result depends on it.
    """

    threads = None  # a distribution saved before this setting existed reads it as the default

    def __init__(self, pieces, values, threads=None):
        self._pieces, self.values, self.threads = pieces, values, threads
        self.batch_shape = (sum(piece.rows for piece in pieces),)

    def _apply(self, method, value):
        value = np.asarray(value, dtype=np.float64)
        full = np.broadcast_to(value, value.shape[:-1] + self.batch_shape if value.ndim else self.batch_shape)
        flat = full.reshape(-1, self.batch_shape[0]).T
        threads = _check_threads(self.threads)
        starts = np.cumsum([0] + [piece.rows for piece in self._pieces])
        out = [None] * len(self._pieces)

        def solve(k):
            out[k] = getattr(self._pieces[k], method)(np.ascontiguousarray(flat[starts[k] : starts[k + 1]]), threads=threads)

        # Quantiles of two blocks at a time: one block's network reads and last rounds run while the other's sums do.
        # Each block is solved as it would be alone.
        _run([lambda k=k: solve(k) for k in range(len(self._pieces))], min(2, threads) if method == "quantile" else 1)
        return np.concatenate(out, axis=0).T.reshape(full.shape)

    def _level_quantiles(self):
        """The quantiles at ``LEVELS``, ``(203, rows)``, solved once: ``mean`` and ``crps`` both read them."""
        q = getattr(self, "_at_levels", None)
        if q is None:
            q = self._at_levels = self.quantile(LEVELS)
        return q

    def cdf(self, value):
        """``P(Y <= value)``."""
        return self._apply("cdf", value)

    def log_prob(self, value):
        """The log density at ``value``; for a target with few values, the log probability (``-inf`` off them)."""
        return self._apply("log_prob", value)

    def quantile(self, levels):
        """Quantiles at ``levels`` (a number, or ``k`` of them), shape ``(rows,)`` or ``(k, rows)``.

        A density's quantile solves its CDF, to the precision of the networks' float32 CDFs; a target with few
        values returns the smallest value whose CDF reaches the level.
        """
        levels = np.asarray(levels, dtype=np.float64)
        if not np.all((levels > 0.0) & (levels < 1.0)):  # a NaN level is refused too
            raise AshlarError("quantile levels must lie strictly between 0 and 1.")
        return self._apply("quantile", levels.reshape(-1, 1) if levels.ndim == 1 else levels)

    def probabilities(self):
        """``(rows, len(values))``: each row's probability of each value; a target with few values only."""
        if self.values is None:
            raise AshlarError("probabilities() is for a target with few values; this one has a density.")
        return np.concatenate([piece.probabilities for piece in self._pieces], axis=0)

    def mean(self):
        """Each row's mean: exact for a target with few values, otherwise read from the quantiles at 203 levels."""
        if self.values is not None:
            return self.probabilities() @ self.values
        return self._level_quantiles().T @ LEVEL_MASS

    def crps(self, value):
        """Each row's CRPS at ``value`` (rows,): exact for a target with few values, otherwise that of the
        quantiles at 203 levels read as atoms, each carrying the mass from the level below (the last also the mass
        above it). ScoringBench reads the same quantiles as a CDF linear between them, with the outer 0.1% on each
        side at the outermost quantiles; for a normal the two readings differ by under 0.01%."""
        value = np.asarray(value, dtype=np.float64).reshape(-1)
        if self.values is not None:
            proba = self.probabilities()
            return atoms_crps(np.broadcast_to(self.values, proba.shape), proba, value)
        q = self._level_quantiles().T
        return atoms_crps(q, np.broadcast_to(LEVEL_MASS, q.shape), value)

    def sample(self, n: int = 100, seed: int = 0):
        """``n`` draws per row, shape ``(n, rows)``, by the quantile of uniform draws."""
        u = np.random.default_rng(int(seed)).uniform(1e-9, 1.0 - 1e-9, size=(int(n), self.batch_shape[0]))
        return self._apply("quantile", u)


class _DensityPiece:
    """The dressed density on a block of rows. ``table``, the quantile solver's start, may be a function that builds
    it: only ``quantile`` reads it, so it is built on the first quantile read."""

    def __init__(self, pools, fallback, corrector, gate, table, rows):
        self.pools, self.fallback, self.corrector, self.gate, self.table = pools, fallback, corrector, gate, table
        self.rows = rows

    def _start_table(self):
        if callable(self.table):
            self.table = self.table()
        return self.table

    def __getstate__(self):
        self._start_table()  # a pickle holds the table, not the function that builds it
        return self.__dict__

    def _sums(self, pools, rows, y, parts, done, threads):
        """The readers' sums of ``pools`` not yet in ``done`` (by reader), in one batch; a reader shared by two
        pools is summed once."""
        readers = {id(pool.reader): pool.reader for pool in pools if id(pool.reader) not in done}
        jobs = [(reader, rows, y, parts) for reader in readers.values()]
        done.update(zip(readers, _kernel_sums(jobs, threads), strict=True))

    def _start(self, pools, y, threads):
        """Every network of ``pools`` started at ``y``, once each (a member's pool and fallback share one)."""
        reads = {}
        for pool in pools:
            if id(pool.network) not in reads:
                reads[id(pool.network)] = _Started(pool.network, y, threads)
        return reads

    def _terms(self, i, j, y, density=True, threads=1):
        """The CDF and (with ``density``) the density at the entries (i, j) of ``y``: per member, the gate times its
        corrected pool plus the rest times its fallback, summed in member order. Each network is read once per call,
        while the readers' sums run; the survival is summed only where a normal score reads it."""
        gate, members = self.gate, len(self.pools)
        corrected = list(self.pools) if gate > 0.0 else []
        plain = list(self.fallback) if gate < 1.0 else []
        reads = self._start(corrected + plain, y, threads)
        yy, parts, sums = y[i, j][:, None], "Ff" if density else "F", {}
        self._sums(corrected or plain, i, yy, parts, sums, threads)
        network = {key: read.at(i, j) for key, read in reads.items()}
        scores, started = [], []
        if corrected:
            pooled = []
            for pool in corrected:
                Fn, log_fn = network[id(pool.network)]
                Fr, _, fr = sums[id(pool.reader)]
                G = pool.w * Fn + (1 - pool.w) * Fr[:, 0]
                g = pool.w * np.exp(log_fn) + (1 - pool.w) * fr[:, 0] if density else None
                pooled.append((G, g, ~(G < 0.5)))
            survival = _kernel_sums([(pool.reader, i[upper], yy[upper], "S") for pool, (_, _, upper) in zip(corrected, pooled, strict=True)],
                                    threads)
            for pool, (G, g, upper), (_, Sr, _) in zip(corrected, pooled, survival, strict=True):
                Fn = network[id(pool.network)][0]
                z = _scores(G, pool.w * (1 - Fn[upper]) + (1 - pool.w) * Sr[:, 0])
                full = np.zeros(y.shape)
                full[i, j] = z
                scores.append((z, g))
                started.append(_Started(self.corrector, full, threads))
        self._sums(plain, i, yy, parts, sums, threads)  # runs while JAX reads the corrector
        F, f = 0.0, 0.0
        for k, (pool, other) in enumerate(zip(self.pools, self.fallback, strict=True)):
            if gate > 0.0:
                Fc, log_f = started[k].at(i, j)
                F = F + gate * Fc
                if density:
                    z, g = scores[k]
                    f = f + gate * np.exp(log_f + np.log(np.maximum(g, 1e-300)) + 0.5 * z**2 + 0.5 * np.log(2 * np.pi))
            if gate < 1.0:
                Fn, log_fn = network[id(other.network)]
                Fr, _, fr = sums[id(other.reader)]
                F = F + (1 - gate) * (other.w * Fn + (1 - other.w) * Fr[:, 0])
                if density:
                    f = f + (1 - gate) * (other.w * np.exp(log_fn) + (1 - other.w) * fr[:, 0])
        return F / members, (f / members if density else None)

    def _all(self, y, density=True, threads=1):
        i, j = np.nonzero(np.ones(y.shape, dtype=bool))
        F, f = self._terms(i, j, y, density, threads)
        return F.reshape(y.shape), (f.reshape(y.shape) if density else None)

    def cdf(self, y, threads=1):
        return np.clip(self._all(y, density=False, threads=threads)[0], 0.0, 1.0)

    def log_prob(self, y, threads=1):
        """The log density, summed in logs as ``_Pool.log_pdf`` does, from one read of each network and from each
        reader's sums of only the parts used."""
        gate, count, parts = self.gate, np.log(len(self.pools)), []
        corrected = list(self.pools) if gate > 0.0 else []
        plain = list(self.fallback) if gate < 1.0 else []
        reads = self._start(corrected + plain, y, threads)
        rows, sums, densities = np.arange(len(y)), {}, {}
        self._sums(corrected, rows, y, "Ff", sums, threads)
        self._sums(plain, rows, y, "f", sums, threads)
        network = {key: tuple(np.asarray(v).T.astype(np.float64) for v in read.values()) for key, read in reads.items()}

        def log_pdf(pool):
            """``_Pool.log_pdf`` from the read and the sums above."""
            if id(pool) not in densities:
                net, fr = network[id(pool.network)][1], sums[id(pool.reader)][2]
                with np.errstate(divide="ignore"):
                    densities[id(pool)] = np.logaddexp(np.log(pool.w) + net, np.log(1 - pool.w) + np.log(np.maximum(fr, 1e-300)))
            return densities[id(pool)]

        with np.errstate(divide="ignore"):
            if corrected:
                pooled = []
                for pool in corrected:
                    G = pool.w * network[id(pool.network)][0] + (1 - pool.w) * sums[id(pool.reader)][0]
                    pooled.append((G, np.nonzero(~(G < 0.5))))
                survival = _kernel_sums([(pool.reader, at[0], y[at][:, None], "S") for pool, (_, at) in zip(corrected, pooled, strict=True)],
                                        threads)
                scores = [_scores(G, pool.w * (1 - network[id(pool.network)][0][at]) + (1 - pool.w) * Sr[:, 0])
                          for pool, (G, at), (_, Sr, _) in zip(corrected, pooled, survival, strict=True)]
                started = [_Started(self.corrector, z, threads) for z in scores]  # all at once
                logs = []
                for pool, z, read in zip(corrected, scores, started, strict=True):
                    corrector = np.asarray(read.values()[1], np.float64).T
                    logs.append(corrector + log_pdf(pool) + 0.5 * z**2 + 0.5 * np.log(2 * np.pi))
                parts.append(np.log(gate) + logsumexp(logs, axis=0) - count)
            if plain:
                parts.append(np.log(1 - gate) + logsumexp([log_pdf(pool) for pool in plain], axis=0) - count)
        return parts[0] if len(parts) == 1 else np.logaddexp(*parts)

    def quantile(self, u, threads=1):
        start = _interp_rows(LEVELS, self._start_table(), np.clip(u, LEVELS[0], LEVELS[-1]))
        tolerance = 3e-7 if self.gate > 0.0 else 1e-9 + 2e-7 * max(p.w for p in self.pools)

        def evaluate(i, j, y):
            return self._terms(i, j, y, threads=threads)

        return _solve([*self.pools, *self.fallback], u, start, evaluate, tolerance)


class _ValuePiece:
    """The dressed probabilities of a target with few values on a block of rows."""

    def __init__(self, probabilities, values):
        self.probabilities, self.values, self.rows = probabilities, values, len(probabilities)
        self.cum = np.cumsum(probabilities, axis=1)

    def cdf(self, y, threads=1):
        index = np.searchsorted(self.values, y, side="right")
        padded = np.concatenate([np.zeros((self.rows, 1)), self.cum], axis=1)
        return np.clip(np.take_along_axis(padded, index, axis=1), 0.0, 1.0)

    def log_prob(self, y, threads=1):
        index = np.minimum(np.searchsorted(self.values, y), len(self.values) - 1)
        with np.errstate(divide="ignore"):
            return np.where(self.values[index] == y, np.log(np.take_along_axis(self.probabilities, index, axis=1)), -np.inf)

    def quantile(self, u, threads=1):
        index = np.stack([np.searchsorted(row, level - 1e-12, side="left") for row, level in zip(self.cum, u, strict=True)])
        return self.values[np.minimum(index, len(self.values) - 1)]


class ASRDressed(RegressorMixin, BaseEstimator):
    """Cross-fitted networks and label readers, pooled, smoothed and corrected into one distribution.

    See the module docstring for the five steps. ``fit(X, y)`` takes a table and one numeric target column;
    ``predict_dist(X)`` returns a ``DressedDistribution`` with an exact ``cdf`` and ``log_prob``.

    Parameters
    ----------
    members:
        How many cross-fitted members (default 5). Each holds out one part of the training rows.
    few_values:
        A target with at most this many distinct values (default 20) is treated as a set of values: its networks
        are ordinal and stopped on held-out log loss, and the prediction is a probability for each value. ``0``
        always fits a density.
    network:
        Settings for ``ashlar.tabm_piecewise``, applied to every member's network and to the corrector, on top of
        ``max_epochs=1000`` and ``cooldown_epochs=3``. ``task``, ``random_state`` and ``early_stopping_metric``
        are set by the model.
    reader:
        Settings for each member's ``ashlar.label_reader.LabelReader``, such as ``max_steps``.
    reader_rows:
        The most training rows a member's reader trains on (default 4096), and separately the most held-out rows it
        is stopped on. A larger set is sampled down to this many, because a reader's cost grows with the square of
        its rows. At prediction a reader reads both sets, so up to twice this many rows. Parts above this size are
        untested.
    floor_cost:
        ``None`` (default) gives every smoothed reader 1% of the wide normal, ``0.01 * (1 - w)`` of the pool. A
        share, such as ``0.01``, lowers the wide normal's weight until its cost in CRPS is at most that share of the
        raw pool's held-out CRPS, the cost taken as for a pool that is a point mass at the label (``_floor_share``).
        For normally spread labels that cost is at most about 0.00006 of their spread: nothing on most tables, and
        more than the whole CRPS where the pool's CRPS is below that.
    random_state:
        Seeds the split, the members, the rows sampled for the readers and for the CRPS factor, and the corrector.
    threads:
        Threads for the fit and for prediction: a positive number, ``-1`` for every CPU this process may run on, or
        ``None`` (default) for the ``ASHLAR_THREADS`` environment variable if set, otherwise the CPUs at most
        ``MAX_THREADS`` (8). The fit trains up to this many members at once and then splits the readers' sums over
        as many threads; prediction splits the sums, solves two blocks of rows at a time and, above 1, issues each
        round's network reads from threads of their own, so XLA runs them at once. No fitted number and no
        prediction depends on it. A ``DressedDistribution`` takes it from the model and can be given its own
        (``dist.threads = 1``). JAX's own threads are JAX's setting.

    Fitted attributes: ``values_`` (the target's values, or ``None`` for a density), ``pool_weight_`` (the
    network's weight against the reader), ``floor_`` (the wide normal's share), ``bandwidth_factor_`` and ``fallback_factor_`` (the smoothing factors
    by log score and by CRPS), ``gate_`` (the corrected pools' weight), ``members_`` (each member's network,
    reader and held-out rows), ``corrector_``, and ``history_`` (the searched curves, the readers' steps and the
    networks' best epochs).
    """

    threads = None  # a model saved before this setting existed reads it as the default

    def __init__(self, members: int = 5, few_values: int = 20, network: dict | None = None, reader: dict | None = None,
                 reader_rows: int = 4096, floor_cost: float | None = None, random_state: int = 0, threads: int | None = None):
        self.members = members
        self.few_values = few_values
        self.network = network
        self.reader = reader
        self.reader_rows = reader_rows
        self.floor_cost = floor_cost
        self.random_state = random_state
        self.threads = threads

    # ----------------------------------------------------------------------------------------------------- fitting

    def _network(self, task, seed, cls=None):
        from ashlar.recipes import tabm_piecewise

        settings = {**MEMBER_SETTINGS, **(self.network or {})}
        taken = {"task", "random_state", "early_stopping_metric"} & set(settings)
        if taken:
            raise AshlarError(f"ASRDressed sets {sorted(taken)} itself; leave them out of network=.")
        if task == "flow":
            settings["early_stopping_metric"] = "crps"
        model = tabm_piecewise(task=task, random_state=int(seed), **settings)
        return model if cls is None else cls(**model.get_params())

    def _check(self, X, y, sample_weight):
        if sample_weight is not None:
            raise AshlarError("ASRDressed does not take sample weights.")
        if observed.is_observed(y):
            raise AshlarError("ASRDressed needs exactly observed labels; censored and interval labels are not supported.")
        if isinstance(self.members, bool) or int(self.members) < 2:
            raise AshlarError("members must be at least 2.")
        if hasattr(X, "to_pandas") and not hasattr(X, "iloc"):
            X = X.to_pandas()
        if not hasattr(X, "iloc"):
            X = np.asarray(X)
            if X.ndim != 2:
                raise AshlarError("X must be a table: rows by columns.")
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[1] == 1:
            y = y[:, 0]
        if y.ndim != 1:
            raise AshlarError("ASRDressed fits one target column; joint targets are not supported.")
        if y.dtype.kind not in "iuf":
            raise AshlarError("ASRDressed fits a numeric target; for classes, use ASRClassifier.")
        y = y.astype(np.float64)
        if not np.isfinite(y).all():
            raise AshlarError("the target must be finite.")
        if len(y) != len(X):
            raise AshlarError("X and y have different numbers of rows.")
        if len(y) < 20 * int(self.members):
            raise AshlarError(f"ASRDressed needs at least {20 * int(self.members)} rows for {int(self.members)} members.")
        return X, y

    def _parts(self, model, reader, X, x, context):
        """A member's network (its distribution, or its value probabilities) and reader atoms on these rows."""
        if self.values_ is None:
            network = model.predict_dist(X)
        else:
            proba = np.asarray(model.predict_proba(X), dtype=np.float64)
            network = np.zeros((len(proba), len(self.values_)))
            network[:, np.asarray(model.classes_, dtype=np.int64)] = proba
            network /= network.sum(axis=1, keepdims=True)
        return network, reader.atoms(x, context=context)

    def _network_atoms(self, model, X, network):
        """A member's network as ``(atoms, masses)``: its values, or its quantiles at ``LEVELS`` read from
        ``network``, its ``predict_dist(X)``."""
        if self.values_ is not None:
            return np.broadcast_to(self.values_, network.shape), network
        return _network_quantiles(model, network, X)

    def _smooth(self, network, reader, factor, weight=None):
        return _Pool(network, _Smoothed(*reader, self.centre_, self.spread_, factor, self.floor_),
                     self.pool_weight_ if weight is None else weight, self.values_)

    def _pool_crps(self, pool, y, start, threads=1):
        """Each row's CRPS of a smoothed pool, read at ``LEVELS`` from ``start``, the raw pool's quantiles there (a
        density), or over its values (``start`` is then ``None``)."""
        if self.values_ is not None:
            return _level_crps(np.broadcast_to(self.values_, pool.network.shape), pool.probabilities(threads), y)
        levels = np.broadcast_to(LEVELS, (len(y), len(LEVELS)))
        q = _pool_quantiles([pool], levels, start, threads)
        return atoms_crps(q, np.broadcast_to(LEVEL_MASS, q.shape), y)

    def fit(self, X, y, sample_weight=None):
        """Fit the members, the pool weight, the smoothing, the corrector and the gate (see the module docstring)."""
        X, y = self._check(X, y, sample_weight)
        threads = _check_threads(self.threads)
        n, count, seed = len(y), int(self.members), int(self.random_state)
        self.columns_ = _Columns(X)
        x = self.columns_(X)
        values = np.unique(y)
        self.values_ = values if self.few_values and values.size <= int(self.few_values) else None
        self.centre_, self.spread_ = float(y.mean()), float(y.std() + 1e-12)
        order = np.random.default_rng(2000 + seed).permutation(n)
        cuts = [int(k / count * n) for k in range(count + 1)]
        codes = np.searchsorted(values, y)
        done = [None] * count
        # A held-out row's read holds 16 bytes for every point its reader reads. Up to ``KEPT_CELLS`` of them the fit
        # keeps every member's reads for the factor search and the corrector; on a larger table those two steps read
        # the rows again, to the same numbers.
        keep = n * min(int(self.reader_rows), n) <= KEPT_CELLS

        def member(k):
            """Member ``k``'s network and reader, fitted, and its reads of its held-out rows, block by block."""
            held = np.sort(order[cuts[k] : cuts[k + 1]])
            part = np.sort(np.concatenate([order[: cuts[k]], order[cuts[k + 1] :]]))
            member_seed = seed + (k + 1) % count
            if self.values_ is not None:
                # An ordinal fit needs every validation value among its training rows: a value only the held-out
                # part has moves over.
                missing = ~np.isin(y[held], y[part])
                part, held = np.sort(np.concatenate([part, held[missing]])), held[~missing]
                if not len(held):
                    raise AshlarError("a member has no held-out rows left; the table is too small to dress.")
                model = self._network("ordinal", member_seed)
                model.fit(_take(X, part), codes[part], validation=(_take(X, held), codes[held]))
            else:
                model = self._network("flow", member_seed)
                model.fit(_take(X, part), y[part], validation=(_take(X, held), y[held]))
            rng = np.random.default_rng(4000 + seed + k)
            cap = int(self.reader_rows)
            read = part if len(part) <= cap else np.sort(rng.choice(part, cap, replace=False))
            check = held if len(held) <= cap else np.sort(rng.choice(held, cap, replace=False))
            reader = LabelReader(random_state=member_seed, **(self.reader or {})).fit(x[read], y[read], validation=(x[check], y[check]))
            curve, reads = np.zeros(len(POOL_WEIGHTS)), []
            for rows in self._blocks(held):
                X_rows = _take(X, rows)
                network, atoms = self._parts(model, reader, X_rows, x[rows], "train")
                net = self._network_atoms(model, X_rows, network)
                curve += len(rows) * _curve(net, atoms, y[rows])
                if keep:
                    reads.append((rows, network, atoms, net))
            done[k] = (model, reader, held), curve / len(held), reads if keep else None

        # Up to ``threads`` members fit at once, each as it would alone; none of them sums a reader. The factor
        # search and the corrector then split their readers' sums over as many threads, so the fit keeps at most
        # ``threads`` threads of its own busy.
        _run([lambda k=k: member(k) for k in range(count)], threads)
        self.members_ = [fitted for fitted, _, _ in done]
        reads = [member_reads for _, _, member_reads in done]
        sizes = np.asarray([len(held) for _, _, held in self.members_], dtype=np.float64)
        oof = (sizes[:, None] * np.stack([curve for _, curve, _ in done])).sum(axis=0) / sizes.sum()
        self.pool_weight_ = float(POOL_WEIGHTS[int(np.argmin(oof))])
        self.floor_ = _floor_share(y, self.centre_, self.spread_, float(oof.min()), self.floor_cost)
        self.history_ = {"weight_curve": oof.tolist(), "reader_steps": [int(reader.steps_) for _, reader, _ in self.members_],
                         "best_epochs": [int(model.history_.get("best_epoch", -1)) for model, _, _ in self.members_]}
        self._pick_factors(X, x, y, reads, threads)
        self._fit_corrector(X, x, y, reads, threads)
        return self

    @staticmethod
    def _blocks(index, size=PREDICT_ROWS):
        return [index[start : start + size] for start in range(0, len(index), size)]

    def _held_reads(self, k, X, x, kept, quantiles=False):
        """Member ``k``'s reads of its held-out rows, block by block, as ``(rows, network, reader atoms, network
        atoms)``: the ones ``fit`` kept, or, where it kept none (``KEPT_CELLS``), the same reads made again, with the
        network's atoms only if ``quantiles``."""
        if kept[k] is not None:
            yield from kept[k]
            return
        model, reader, held = self.members_[k]
        for rows in self._blocks(held):
            X_rows = _take(X, rows)
            network, atoms = self._parts(model, reader, X_rows, x[rows], "train")
            yield rows, network, atoms, (self._network_atoms(model, X_rows, network) if quantiles else None)

    def _pick_factors(self, X, x, y, reads, threads=1):
        """The bandwidth factor by the log score of every held-out row, and the fallback's by the CRPS of
        ``CRPS_ROWS`` held-out rows per member, read at ``LEVELS``. ``reads`` holds each member's reads of its
        held-out blocks, where ``fit`` kept them (``_held_reads``)."""
        seed = int(self.random_state)
        scores, crps = np.zeros(len(FACTORS)), np.zeros(len(FACTORS))
        searched = FACTORS if self.pool_weight_ < 1.0 else FACTORS[:1]  # any factor does when the reader has no weight
        total = picked = 0
        for k, (model, reader, held) in enumerate(self.members_):
            pick = np.sort(np.random.default_rng(3000 + seed + k).permutation(len(held))[:CRPS_ROWS])
            for rows, network, atoms, _ in self._held_reads(k, X, x, reads):
                # Every factor shares one sort of the readers' points and one read of the network's log density.
                first = self._smooth(network, atoms, FACTORS[0])
                net = first.network_log_pdf(y[rows][:, None]) if self.values_ is None else None
                for f, factor in enumerate(FACTORS):
                    pool = first if f == 0 else first.at_factor(factor)
                    if self.values_ is None:
                        scores[f] += pool.log_pdf(y[rows][:, None], net, threads).sum()
                    else:
                        at = np.searchsorted(self.values_, y[rows])
                        scores[f] += np.log(np.maximum(pool.probabilities(threads)[np.arange(len(rows)), at], 1e-300)).sum()
                total += len(rows)
            rows = held[pick]
            X_rows = _take(X, rows)
            network, atoms = self._parts(model, reader, X_rows, x[rows], "train")
            start = None
            if self.values_ is None:  # the raw pool's quantiles start every factor's solve
                raw = _combine(self._network_atoms(model, X_rows, network), atoms, self.pool_weight_)
                start = mixture_quantiles(raw, raw, 1.0, LEVELS)
            first = self._smooth(network, atoms, searched[0])
            for f, factor in enumerate(searched):
                pool = first if f == 0 else first.at_factor(factor)
                crps[f] += self._pool_crps(pool, y[rows], start, threads).sum()
            picked += len(rows)
        self.bandwidth_factor_ = float(FACTORS[int(np.argmax(scores))])
        self.fallback_factor_ = float(searched[int(np.argmin(crps[: len(searched)]))])
        self.history_["log_score_curve"] = (scores / total).tolist()
        self.history_["crps_curve"] = (crps[: len(searched)] / picked).tolist()

    def _fit_corrector(self, X, x, y, reads, threads=1):
        """The corrector on the held-out rows' normal scores, then the gate on the last member's held-out rows.
        ``reads`` holds each member's reads of its held-out blocks, where ``fit`` kept them (``_held_reads``)."""
        values, last = self.values_, len(self.members_) - 1
        mids = None if values is None else (values[:-1] + values[1:]) / 2
        fit_rows, labels = [], []
        stop = {"table": [], "corrected": None, "fallback": [], "base": [], "pools": []}
        for k in range(len(self.members_)):
            for rows, network, atoms, net in self._held_reads(k, X, x, reads, quantiles=values is None and k == last):
                pool = self._smooth(network, atoms, self.bandwidth_factor_)
                if values is None:
                    label = pool.z(y[rows][:, None], threads)[:, 0]
                else:
                    edges = pool.z(np.broadcast_to(mids, (len(rows), len(mids))).copy(), threads)
                    padded = np.concatenate([np.full((len(rows), 1), -np.inf), edges, np.full((len(rows), 1), np.inf)], axis=1)
                    index = np.searchsorted(values, y[rows])
                    label = np.stack([padded[np.arange(len(rows)), index], padded[np.arange(len(rows)), index + 1]], axis=1)
                labels.append(label)
                fit_rows.append(rows)
                if k < last:
                    continue
                other = pool if self.fallback_factor_ == self.bandwidth_factor_ else pool.at_factor(self.fallback_factor_)
                if values is None:
                    raw = _combine(net, atoms, self.pool_weight_)
                    raw_q = mixture_quantiles(raw, raw, 1.0, LEVELS)
                    z_levels = ndtr(np.broadcast_to(Z_GRID, (len(rows), len(Z_GRID))))
                    start = _interp_rows(LEVELS, raw_q, np.clip(z_levels, LEVELS[0], LEVELS[-1]))
                    stop["table"].append(_pool_quantiles([pool], z_levels, start, threads))
                    levels = np.broadcast_to(LEVELS, (len(rows), len(LEVELS)))
                    base = _pool_quantiles([pool], levels, raw_q, threads)
                    stop["base"].append(base)
                    stop["fallback"].append(base if other is pool else _pool_quantiles([other], levels, base, threads))
                else:
                    stop["table"].append(edges)
                    stop["fallback"].append(other.probabilities(threads))
                stop["pools"].append((rows, pool))
        held_last = self.members_[last][2]
        n_fit = sum(len(rows) for rows in fit_rows) - len(held_last)
        fit_index, stop_index = np.concatenate(fit_rows)[:n_fit], np.concatenate(fit_rows)[n_fit:]
        label = np.concatenate(labels)
        X_fit, X_stop, y_stop = _take(X, fit_index), _take(X, stop_index), y[stop_index]
        corrector = self._network("flow", int(self.random_state), cls=_Corrector)
        table = np.concatenate(stop["table"])
        if values is None:
            corrector.stop_on = ("flow", table, None, y_stop)
            corrector.fit(X_fit, label[:n_fit], validation=(X_stop, label[n_fit:]))
        else:
            corrector.stop_on = ("values", table, values, y_stop)
            corrector.fit(X_fit, observed.interval(label[:n_fit, 0], label[:n_fit, 1]),
                          validation=(X_stop, observed.interval(label[n_fit:, 0], label[n_fit:, 1])))
        corrector.stop_on = None
        self.corrector_ = corrector
        # The gate: the corrected pool against the pool at the CRPS factor, on the last member's held-out rows
        # under that member's own pool, the rows that stopped the corrector; the first lowest score is the smaller
        # gate. A mixture's CRPS can be below that of both its parts, so the gate can be positive where the
        # corrected pool alone scores worse.
        corrected, at = [], 0
        for (rows, pool), base in zip(stop["pools"], stop["base"] or [None] * len(stop["pools"]), strict=True):
            if values is None:
                u = ndtr(np.asarray(corrector.predict_quantiles(_take(X, rows), LEVELS), dtype=np.float64))
                corrected.append(_pool_quantiles([pool], u, _interp_rows(LEVELS, base, np.clip(u, LEVELS[0], LEVELS[-1])), threads))
            else:
                edges = table[at : at + len(rows)]
                corrected.append(_value_masses(np.asarray(corrector.predict_dist(_take(X, rows)).cdf(edges.T), dtype=np.float64).T))
            at += len(rows)
        corrected, fallback = np.concatenate(corrected), np.concatenate(stop["fallback"])
        if values is None:
            mass = np.broadcast_to(LEVEL_MASS, corrected.shape)
            curve = _curve((corrected, mass), (fallback, mass), y_stop)
        else:
            grid = np.broadcast_to(values, corrected.shape)
            curve = _curve((grid, corrected), (grid, fallback), y_stop)
        self.gate_ = float(POOL_WEIGHTS[int(np.argmin(curve))])
        self.history_["gate_curve"] = curve.tolist()
        self.history_["corrector_epoch"] = int(corrector.history_.get("best_epoch", -1))

    # -------------------------------------------------------------------------------------------------- prediction

    def _require_fitted(self):
        if not hasattr(self, "gate_"):
            raise AshlarError("this ASRDressed is not fitted yet; call fit first.")

    def _piece(self, X, x, threads=1):
        values, gate = self.values_, self.gate_
        # Each reader reads its held-out rows as well as its training rows here ("all"); fit's searches read "train".
        parts = [self._parts(model, reader, X, x, "all") for model, reader, _ in self.members_]
        same = self.fallback_factor_ == self.bandwidth_factor_
        pools = [self._smooth(network, atoms, self.bandwidth_factor_) for network, atoms in parts] if gate > 0.0 or same else None
        if same:
            fallback = pools
        elif pools is not None:  # the second smoothing reuses the first's sorted points
            fallback = [pool.at_factor(self.fallback_factor_) for pool in pools]
        else:
            fallback = [self._smooth(network, atoms, self.fallback_factor_) for network, atoms in parts]
        if values is not None:
            proba = (1 - gate) * np.mean([p.probabilities(threads) for p in fallback], axis=0)
            if gate > 0.0:
                mids = np.broadcast_to((values[:-1] + values[1:]) / 2, (len(x), len(values) - 1)).copy()
                dist = self.corrector_.predict_dist(X)
                proba = proba + gate * np.mean(
                    [_value_masses(np.asarray(dist.cdf(p.z(mids, threads).T), dtype=np.float64).T) for p in pools], axis=0)
            return _ValuePiece(proba / proba.sum(axis=1, keepdims=True), values)
        models, weight = [model for model, _, _ in self.members_], self.pool_weight_

        def table():
            """The quantile solver's start: the raw pools' quantiles, built on the first quantile read."""
            raw = _union([_combine(_network_quantiles(model, network, X), atoms, weight)
                          for model, (network, atoms) in zip(models, parts, strict=True)])
            return mixture_quantiles(raw, raw, 1.0, LEVELS)

        corrector = self.corrector_.predict_dist(X) if gate > 0.0 else None
        return _DensityPiece(pools if pools is not None else fallback, fallback, corrector, gate, table, len(x))

    def predict_dist(self, X):
        """The predictive distribution of each row of ``X``, as a ``DressedDistribution``."""
        self._require_fitted()
        if hasattr(X, "to_pandas") and not hasattr(X, "iloc"):
            X = X.to_pandas()
        if not hasattr(X, "iloc"):
            X = np.asarray(X)
        x = self.columns_(X)
        index = np.arange(len(x))
        threads = _check_threads(self.threads)
        return DressedDistribution([self._piece(_take(X, rows), x[rows], threads) for rows in self._blocks(index)], self.values_,
                                   threads=self.threads)

    def predict(self, X):
        """Each row's predictive mean."""
        return self.predict_dist(X).mean()

    def predict_quantiles(self, X, levels=(0.1, 0.5, 0.9)):
        """Quantiles per row, shape ``(rows, len(levels))``."""
        return self.predict_dist(X).quantile(np.asarray(levels, dtype=np.float64).reshape(-1)).T

    def log_prob(self, X, y):
        """The log density of ``y`` under ``predict_dist(X)``, per row (the log probability for few values)."""
        return self.predict_dist(X).log_prob(np.asarray(y, dtype=np.float64).reshape(-1))

    def sample(self, X, n: int = 100, seed: int = 0):
        """``n`` draws per row from ``predict_dist(X)``, shape ``(n, rows)``."""
        return self.predict_dist(X).sample(n, seed)

    def score(self, X, y, sample_weight=None):
        """R² of the predictive mean, as a scikit-learn regressor returns."""
        from sklearn.metrics import r2_score

        return float(r2_score(np.asarray(y, dtype=np.float64).reshape(-1), self.predict(X), sample_weight=sample_weight))


def dressed(**settings) -> ASRDressed:
    """A dressed model: ``ASRDressed`` with ``members``, ``few_values``, ``reader``, ``reader_rows``, ``floor_cost``,
    ``random_state`` and ``threads`` taken from ``settings`` and every other setting passed to its networks
    (``ashlar.tabm_piecewise``), as in ``ashlar.dressed(learning_rate=1e-3)``."""
    names = ("members", "few_values", "reader", "reader_rows", "floor_cost", "random_state", "threads")
    own = {name: settings.pop(name) for name in names if name in settings}
    return ASRDressed(network=settings or None, **own)
