"""Forecasting as a table: origins as rows, the horizon as a joint outcome."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from ashlar import ASRRegressor
from ashlar.distributions import LowRankMultivariateNormal
from ashlar.errors import AshlarError
from ashlar.horizon import horizon_frame, peak, stockout
from micro import MICRO


def _panel(series=3, steps=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for item in range(series):
        level = 10.0 * (item + 1)
        promo = rng.integers(0, 2, steps)
        for step in range(steps):
            rows.append(
                {
                    "store": f"s{item}",
                    "day": step,
                    "sales": level + 3.0 * promo[step] + rng.normal(),
                    "promo": promo[step],
                    "customers": level + rng.normal(),
                    "kind": "big" if item == 2 else "small",
                }
            )
    # Days out of order within each store; the frame sorts them by ``time``.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=0).sort_values("store", kind="stable")


def test_frame_rows_hold_the_lags_the_future_covariates_and_the_path():
    data = _panel()
    frame = horizon_frame(
        data, target="sales", lookback=4, horizon=3, series="store", time="day",
        known=["promo"], past=["customers"], static=["kind"],
    )
    # Each series: origins at days 3 .. 36, with three days ahead observed.
    assert len(frame.X) == 3 * 34 and frame.Y.shape == (3 * 34, 3)
    assert frame.target_columns == ("sales_step1", "sales_step2", "sales_step3")
    ordered = data.sort_values(["store", "day"])
    first = ordered[ordered["store"] == "s0"].reset_index(drop=True)
    row = frame.X.iloc[0]
    assert frame.index.loc[0, "series"] == "s0" and frame.index.loc[0, "origin"] == 3
    np.testing.assert_allclose([row[f"sales_lag{j}"] for j in (1, 2, 3, 4)], first["sales"].to_numpy()[[3, 2, 1, 0]])
    np.testing.assert_allclose([row[f"promo_step{h}"] for h in (1, 2, 3)], first["promo"].to_numpy()[[4, 5, 6]])
    np.testing.assert_allclose([row[f"customers_lag{j}"] for j in (1, 4)], first["customers"].to_numpy()[[3, 0]])
    np.testing.assert_allclose(frame.Y[0], first["sales"].to_numpy()[[4, 5, 6]])
    assert row["kind"] == "small" and frame.X.iloc[-1]["kind"] == "big"


def test_bounds_stride_open_ends_and_scale():
    data = _panel(series=1)
    frame = horizon_frame(data, target="sales", lookback=4, horizon=3, time="day", stride=5, start=10, end=30)
    assert frame.index["origin"].tolist() == [13, 18, 23, 28]
    open_end = horizon_frame(data, target="sales", lookback=4, horizon=3, time="day", complete=False)
    assert open_end.index["origin"].iloc[-1] == 39 and np.isnan(open_end.Y[-1]).all() and np.isnan(open_end.Y[-2, 2])
    scaled = horizon_frame(data, target="sales", lookback=4, horizon=3, time="day", scale="mean")
    plain = horizon_frame(data, target="sales", lookback=4, horizon=3, time="day")
    np.testing.assert_allclose(scaled.unscale(scaled.Y), plain.Y)
    np.testing.assert_allclose(scaled.unscale(np.stack([scaled.Y, scaled.Y]))[1], plain.Y)
    np.testing.assert_allclose(np.exp(scaled.X["sales_log_scale"]), scaled.index["scale"])
    for bad in (
        dict(lookback=0, horizon=3),
        dict(lookback=4, horizon=3, known=["nope"]),
        dict(lookback=4, horizon=3, known=["promo"], past=["promo"]),
        dict(lookback=4, horizon=3, scale="max"),
        dict(lookback=60, horizon=3),
    ):
        with pytest.raises(AshlarError):
            horizon_frame(data, target="sales", time="day", **bad)


def test_missing_targets_drop_origins_and_stay_missing_lags():
    data = _panel(series=1).sort_values("day").reset_index(drop=True)
    data.loc[10, "sales"] = np.nan
    frame = horizon_frame(data, target="sales", lookback=4, horizon=3, time="day", scale="mean")
    origins = frame.index["origin"].to_numpy()
    # Origins 7, 8, and 9 would forecast day 10, so they go; 10 to 13 keep it as a missing lag.
    assert not np.isin([7, 8, 9], origins).any() and np.isin([6, 10, 13], origins).all()
    row = frame.X[frame.index["origin"] == 12].iloc[0]
    assert np.isnan(row["sales_lag3"]) and np.isfinite(row["sales_lag1"])
    expected = np.abs(data["sales"].to_numpy()[[12, 11, 9]]).mean()
    assert np.isclose(frame.index.loc[frame.index["origin"] == 12, "scale"].iloc[0], expected)
    open_end = horizon_frame(data, target="sales", lookback=4, horizon=3, time="day", complete=False)
    assert np.isnan(open_end.Y[open_end.index["origin"] == 8][0, 1])


def test_path_functions():
    paths = np.asarray([[1.0, 5.0, 2.0], [4.0, 4.0, 4.0], [0.5, 0.5, 0.5]])
    np.testing.assert_allclose(peak(paths), [5.0, 4.0, 0.5])
    lasts = stockout(7.5)
    assert lasts.__name__ == "stockout_7.5"
    np.testing.assert_allclose(np.asarray(lasts(paths)), [3.0, 2.0, 4.0])


def test_a_joint_head_forecasts_the_path_and_calibrates_its_peak():
    # Mechanics only: a tiny network trained for two epochs, so the checks are shapes, bounds, and order.
    data = _panel(series=3, steps=60, seed=1)
    frame = horizon_frame(data, target="sales", lookback=7, horizon=4, series="store", time="day", known=["promo"])
    origin = frame.index["origin"].to_numpy()
    train, calibration, test = origin < 35, (origin >= 35) & (origin < 45), origin >= 45
    model = ASRRegressor(task="multivariate_gaussian", max_epochs=2, **MICRO)
    model.fit(frame.X[train], frame.Y[train])
    lasts = stockout(60.0)
    model.calibrate(frame.X[calibration], frame.Y[calibration], target=[peak, lasts])
    rows = int(test.sum())
    assert np.asarray(model.sample(frame.X[test], n=8, seed=0)).shape == (8, rows, 4)
    interval = np.asarray(model.predict_interval(frame.X[test], 0.8, target=peak))
    assert interval.shape == (rows, 2) and np.all(interval[:, 0] <= interval[:, 1])
    exceed = np.asarray([model.predict_exceedance(frame.X[test], step, target=lasts) for step in (1.0, 2.0, 3.0)])
    assert np.all((exceed >= 0.0) & (exceed <= 1.0)) and np.all(np.diff(exceed, axis=0) <= 1e-9)


def test_the_joint_normal_density_holds_a_variance_at_its_floor():
    # A step the features make certain, such as a day a store is known to be closed, drives its variance to the
    # floor. The low-rank normal's capacitance then lost every digit in float32: its log density came out wrong or
    # NaN, and a training run on lags alone stopped on a loss that was not finite.
    rng = np.random.default_rng(0)
    factor = rng.normal(size=(6, 2)).astype(np.float32)
    factor[:, 1] = factor[:, 0] + 1e-3 * rng.normal(size=6)
    diag = np.array([1e-8, 1e-8, 1e-8, 0.5, 1.0, 2.0], dtype=np.float32)
    value = (factor @ rng.normal(size=2) + np.sqrt(diag) * rng.normal(size=6)).astype(np.float32)
    got = float(LowRankMultivariateNormal(jnp.zeros(6), jnp.asarray(factor), jnp.asarray(diag)).log_prob(jnp.asarray(value)))
    covariance = factor.astype(np.float64) @ factor.T.astype(np.float64) + np.diag(diag.astype(np.float64))
    expected = stats.multivariate_normal(np.zeros(6), covariance).logpdf(value.astype(np.float64))
    assert abs(got - expected) < 0.01 * abs(expected)
