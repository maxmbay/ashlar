"""Nested outcomes: one softmax per internal node of an outcome tree.

A spec is a nested dict. A key maps to a dict (a subtree), a list (leaf
names, or dicts for deeper subtrees), or ``None`` (a leaf)::

    {"swing": {"contact": ["in_play", "foul"], "miss": None},
     "take": ["called_strike", "ball"]}

The root is unnamed. Every internal node needs at least two children, and
every name must be unique, so a label is one node name. A leaf's probability
is the product of the branch probabilities on its path. A label may stop at
an internal node: the loss is then the log-probability of that node, so a
row that only says ``swing`` still trains the root split.
"""

from __future__ import annotations

import json
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from equinox import field

from ashlar.errors import AshlarError, ShapeError
from ashlar.heads import _EnsembleHead
from ashlar.layers import LinearEnsemble, _rsqrt_uniform
from ashlar.weights import weighted_mean

ROOT = None


class OutcomeTree:
    """Parsed outcome tree. Node 0 is the root; nodes are numbered depth first.

    ``names[i]`` is the node name (``None`` for the root), ``parent[i]`` its
    parent (``-1`` for the root), and ``children[i]`` its ordered children.
    Each internal node owns one edge per child; ``edge_parent`` and
    ``edge_child`` list them in internal-node order. ``paths`` is the
    ``(node, edge)`` 0/1 matrix of the edges from the root to each node.
    """

    def __init__(self, spec):
        if isinstance(spec, OutcomeTree):
            spec = spec.spec
        if isinstance(spec, str):
            spec = json.loads(spec)
        if not isinstance(spec, dict):
            raise AshlarError("an outcome tree is a dict of {name: subtree}, at least two names at the root.")
        self.spec = _normal_spec(spec)
        names: list = [ROOT]
        parent = [-1]
        children: list[list[int]] = [[]]

        def visit(node: int, subtree: dict):
            for name, child in subtree.items():
                index = len(names)
                names.append(name)
                parent.append(node)
                children.append([])
                children[node].append(index)
                if child is not None:
                    visit(index, child)

        visit(0, self.spec)
        if len(set(names[1:])) != len(names) - 1:
            seen, repeated = set(), set()
            for name in names[1:]:
                (repeated if name in seen else seen).add(name)
            raise AshlarError(f"outcome tree names must be unique; repeated {sorted(repeated)}.")
        self.names = tuple(names)
        self.parent = tuple(parent)
        self.children = tuple(tuple(c) for c in children)
        self.internal = tuple(i for i, c in enumerate(self.children) if c)
        self.leaves = tuple(i for i, c in enumerate(self.children) if not c)
        edge_parent, edge_child = [], []
        for node in self.internal:
            for child in self.children[node]:
                edge_parent.append(node)
                edge_child.append(child)
        self.edge_parent = tuple(edge_parent)
        self.edge_child = tuple(edge_child)
        edge_of_child = {child: e for e, child in enumerate(edge_child)}
        paths = np.zeros((len(names), len(edge_child)), dtype=np.float32)
        for node in range(1, len(names)):
            walk = node
            while walk != 0:
                paths[node, edge_of_child[walk]] = 1.0
                walk = self.parent[walk]
        self.paths = paths
        width = max(len(self.children[node]) for node in self.internal)
        slots = np.zeros((len(self.internal), width), dtype=np.int32)
        mask = np.zeros((len(self.internal), width), dtype=bool)
        e = 0
        for row, node in enumerate(self.internal):
            for slot in range(len(self.children[node])):
                slots[row, slot] = e
                mask[row, slot] = True
                e += 1
        self.slots = slots
        self.slot_mask = mask
        self.index = {name: i for i, name in enumerate(self.names) if i}

    @property
    def n_nodes(self) -> int:
        return len(self.names)

    @property
    def n_edges(self) -> int:
        return len(self.edge_child)

    @property
    def leaf_names(self) -> list:
        return [self.names[i] for i in self.leaves]

    def to_json(self) -> str:
        return json.dumps(self.spec)

    def node(self, name) -> int:
        """Node index of ``name``. ``None`` is the root."""
        if name is None:
            return 0
        if name not in self.index:
            raise AshlarError(f"{name!r} is not a node of the outcome tree; nodes are {list(self.index)}.")
        return self.index[name]

    def path(self, name) -> list:
        """Node names from the first branch down to ``name``."""
        walk, out = self.node(name), []
        while walk != 0:
            out.append(self.names[walk])
            walk = self.parent[walk]
        return out[::-1]

    def encode(self, labels) -> np.ndarray:
        """Node index per label. ``None``, ``NaN``, and ``""`` are the root: no information."""
        out = np.empty(len(labels), dtype=np.int32)
        for row, label in enumerate(labels):
            if label is None or label == "" or (isinstance(label, float) and np.isnan(label)):
                out[row] = 0
            else:
                out[row] = self.node(label)
        return out

    def node_log_probabilities(self, logits):
        """``(..., edge)`` branch logits to ``(..., node)`` log-probabilities.

        Each internal node's edges get a log-softmax over that node's
        children. A node's log-probability is the sum of the edge
        log-probabilities on its path, so the root is 0.
        """
        if logits.shape[-1] != self.n_edges:
            raise ShapeError(f"tree logits need {self.n_edges} edges on the last axis; received {logits.shape}.")
        grouped = jnp.take(logits, jnp.asarray(self.slots), axis=-1)
        grouped = jnp.where(jnp.asarray(self.slot_mask), grouped, -jnp.inf)
        log_branch = jax.nn.log_softmax(grouped, axis=-1)
        rows = np.repeat(np.arange(len(self.internal)), self.slot_mask.sum(axis=1))
        cols = np.concatenate([np.arange(n) for n in self.slot_mask.sum(axis=1)])
        edges = log_branch[..., rows, cols]
        return edges @ jnp.asarray(self.paths.T, dtype=edges.dtype)

    def conditional(self, node_probabilities, name):
        """``P(child | name)`` for each child of ``name``, from node marginals ``(rows, node)``."""
        node = self.node(name)
        kids = list(self.children[node])
        if not kids:
            raise AshlarError(f"{name!r} is a leaf; it has no conditional distribution.")
        marginal = np.asarray(node_probabilities)[:, kids]
        return marginal / np.clip(marginal.sum(axis=1, keepdims=True), 1e-30, None)


def _normal_spec(spec) -> dict:
    """``{name: dict | None}`` with lists expanded and names checked."""
    if isinstance(spec, dict):
        items = list(spec.items())
    elif isinstance(spec, (list, tuple)):
        items = []
        for entry in spec:
            if isinstance(entry, dict):
                items.extend(entry.items())
            else:
                items.append((entry, None))
    else:
        raise AshlarError(f"an outcome subtree is a dict, a list, or None; received {type(spec).__name__}.")
    if len(items) < 2:
        raise AshlarError(f"every internal node of an outcome tree needs at least two children; received {items!r}.")
    out = {}
    for name, child in items:
        if not isinstance(name, str) or not name:
            raise AshlarError(f"outcome tree node names must be non-empty strings; received {name!r}.")
        if name in out:
            raise AshlarError(f"outcome tree names must be unique; repeated [{name!r}].")
        out[name] = None if child is None else _normal_spec(child)
    return out


@lru_cache(maxsize=64)
def parse_tree(spec_json: str) -> OutcomeTree:
    return OutcomeTree(json.loads(spec_json))


def tree_json(spec) -> str:
    """Canonical JSON for a spec, an ``OutcomeTree``, or a JSON string."""
    return OutcomeTree(spec).to_json()


class TreeHead(_EnsembleHead):
    """One branch softmax per internal node on a shared hidden state.

    Output ``(batch, member, node)`` log-probabilities; the root column is 0.
    ``edge_prior`` (optional) is one probability per edge in internal-node
    order; the bias starts at its log, so the untrained model predicts the
    training frequencies of each branch.
    """

    tree: str = field(static=True)

    def __init__(self, d_in, k, *, n_events, n_classes, key, tree, edge_prior=None, **kwargs):
        del n_events
        if kwargs:
            raise ShapeError(f"tree head received unknown arguments {sorted(kwargs)}.")
        parsed = parse_tree(tree_json(tree))
        if n_classes != parsed.n_nodes:
            raise ShapeError(f"the outcome tree has {parsed.n_nodes} nodes; the head was asked for {n_classes}.")
        weight_key, bias_key = jax.random.split(key)
        width = parsed.n_edges
        if edge_prior is None:
            bias = _rsqrt_uniform(bias_key, (k, width), d_in, jnp.float32)
        else:
            prior = jnp.log(jnp.clip(jnp.asarray(edge_prior, dtype=jnp.float32), 1e-6, 1.0))
            if prior.shape != (width,):
                raise ShapeError(f"edge_prior needs {width} entries; received {prior.shape}.")
            bias = jnp.broadcast_to(prior, (k, width))
        self.output = LinearEnsemble(
            weight=_rsqrt_uniform(weight_key, (k, d_in, width), d_in, jnp.float32),
            bias=jnp.asarray(bias),
        )
        self.k = k
        self.tree = parsed.to_json()

    def __call__(self, hidden, *, key):
        del key
        self._check(hidden)
        return parse_tree(self.tree).node_log_probabilities(self.output(hidden))


def tree_objective(node_log_probs, y, sample_weight):
    """Mean member negative log-probability of the labelled node, weighted over rows.

    ``y`` is a node index. A leaf label scores the full path; an internal label
    scores only the branches above it. A root label scores 0.
    """
    if node_log_probs.ndim != 3:
        raise ShapeError(f"tree log-probabilities must be (batch, member, node); received {node_log_probs.shape}.")
    index = jnp.asarray(y).astype(jnp.int32)[:, None, None]
    chosen = jnp.take_along_axis(node_log_probs, index, axis=-1)[..., 0]
    return weighted_mean(-jnp.mean(chosen, axis=1), sample_weight)


def empirical_edge_prior(tree: OutcomeTree, nodes, weights, alpha: float = 0.5) -> list[float]:
    """Branch frequencies with a Jeffreys pseudo-count, one per edge.

    A row labelled at a node counts once for every edge on its path, so a
    partial label informs the splits above it.
    """
    nodes = np.asarray(nodes, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    reach = np.zeros(tree.n_nodes, dtype=np.float64)
    np.add.at(reach, nodes, weights)
    # Mass that reaches a node: labels at it or anywhere below it. Children
    # are numbered after their parent, so one reverse pass accumulates.
    below = reach.copy()
    for node in reversed(range(1, tree.n_nodes)):
        below[tree.parent[node]] += below[node]
    prior = []
    for node in tree.internal:
        kids = tree.children[node]
        counts = np.asarray([below[c] for c in kids]) + alpha
        prior.extend((counts / counts.sum()).tolist())
    return prior
