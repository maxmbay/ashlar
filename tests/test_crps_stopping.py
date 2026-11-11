"""early_stopping_metric='crps': the kept checkpoint is the one with the best validation CRPS of the member mixture."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from ashlar import ASRModel
from ashlar import observed
from ashlar.distributions import Normal, quantile_crps
from ashlar.errors import AshlarError
from micro import MICRO_TABM


def _normal_crps(y, loc, scale):
    z = (y - loc) / scale
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))


def test_quantile_crps_matches_the_closed_form_for_a_normal():
    loc, scale = jnp.asarray([0.0, 1.0, -2.0]), jnp.asarray([1.0, 0.5, 3.0])
    y = jnp.asarray([0.3, 2.5, -2.0])
    got = np.asarray(quantile_crps(Normal(loc, scale), y))
    want = np.asarray([_normal_crps(float(a), float(b), float(c)) for a, b, c in zip(y, loc, scale, strict=True)])
    assert np.allclose(got, want, rtol=5e-3)


def test_a_mixture_fit_stops_on_validation_crps_and_other_tasks_refuse_it():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 3)), columns=list("abc"))
    y = (X["a"] + 0.5 * rng.standard_t(3, size=500)).to_numpy()
    model = ASRModel(**MICRO_TABM, task="mixture", max_epochs=4, early_stopping_metric="crps")
    model.fit(X.iloc[:400], y[:400], validation=(X.iloc[400:], y[400:]))
    q = np.asarray(model.predict_quantiles(X.iloc[400:], [0.1, 0.5, 0.9]))
    assert q.shape == (100, 3) and np.all(np.isfinite(q)) and np.all(np.diff(q, axis=1) >= 0)
    scores = model.history_["val_score"]
    assert len(scores) == 4 and all(score < 0 for score in scores)
    assert model.history_["best_epoch"] == int(np.argmax(scores)) + 1
    for task in ("regression", "quantile"):
        with pytest.raises(AshlarError):
            ASRModel(**MICRO_TABM, task=task, early_stopping_metric="crps").fit(X, y)
    with pytest.raises(AshlarError):
        ASRModel(**MICRO_TABM, task="mixture", early_stopping_metric="accuracy").fit(X, y)


def test_crps_stopping_refuses_censored_labels_and_survives_a_save(tmp_path):
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(300, 2)), columns=list("ab"))
    y = (X["a"] + 0.3 * rng.normal(size=300)).to_numpy()
    with pytest.raises(AshlarError, match="exactly observed"):
        ASRModel(**MICRO_TABM, task="mixture", max_epochs=2, early_stopping_metric="crps").fit(X, observed.censored(y, np.arange(300) % 3 == 0))
    model = ASRModel(**MICRO_TABM, task="mixture", max_epochs=2, early_stopping_metric="crps").fit(X, y)
    model.save(tmp_path / "model")
    assert ASRModel.load(tmp_path / "model").early_stopping_metric == "crps"
