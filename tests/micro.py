"""Micro settings: the least data and compute that still runs each code path.

A test here checks that the code is right, not that a model is good. So a fit
uses a one-block, 16-wide network, a few epochs, and a few hundred rows. Where a
test compares two fits (a censored fit against a naive one, say), the effect is
made large enough that this tiny network shows it. Recovery of a parameter to a
tight tolerance belongs in the benchmarks under ``examples/``.

Most of a fit's time in a test is XLA compiling it, not training, so every fit
with these settings costs about a second however many rows it sees.
"""

from __future__ import annotations

# One block, 16 wide, no dropout. ``residual_mlp`` is the default backbone.
NET = {"n_blocks": 1, "d_block": 16, "dropout": 0.0}

# The same for TabM, with four members so member disagreement is defined.
TABM_NET = {"n_blocks": 1, "d_block": 16, "k": 4}

# Few piecewise bins and a narrow numeric embedding.
_EMBEDDING = {"num_embedding_dim": 4, "num_embedding_bins": 8}

# Keyword arguments for ``ASRRegressor``, ``ASRClassifier``, ``ASRModel`` and friends.
MICRO = {"backbone_kwargs": NET, **_EMBEDDING, "random_state": 0}
MICRO_TABM = {"backbone": "tabm", "backbone_kwargs": TABM_NET, **_EMBEDDING, "random_state": 0}

# A forecaster with a short window. Its transformer's width is fixed by the library.
FORECAST = {"lookback": 16, "horizon": 4, "random_state": 0}
