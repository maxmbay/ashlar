"""Calibration beyond the marginal: per group, in bins of member disagreement, and joint regions."""

from __future__ import annotations

import numpy as np
import pytest

from ashlar import ASRModel, ASRRegressor
from ashlar.calibrate import (
    MIN_GROUP_ROWS,
    density_pit,
    disagreement_edges,
    disagreement_keys,
    group_key,
    group_keys,
    region_level,
    region_threshold,
)
from ashlar.errors import AshlarError, ShapeError


def _three_spreads(n, seed):
    """``y`` whose noise is 0.3, 1, or 3 by a group the features do not show."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    group = rng.integers(0, 3, n)
    y = x[:, 0] + np.array([0.3, 1.0, 3.0])[group] * rng.normal(size=n)
    return x, y, group


def _within(interval, y, group):
    return np.array([np.mean((y[group == k] >= interval[group == k, 0]) & (y[group == k] <= interval[group == k, 1])) for k in range(3)])


def _shared_shock(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3)).astype(np.float32)
    shock = rng.normal(size=n) * (0.5 + np.abs(x[:, 2]))
    y = np.stack([x[:, 0] + shock, -x[:, 1] + shock, x[:, 0] * x[:, 1] + shock], axis=1) + 0.3 * rng.normal(size=(n, 3))
    return x, y * np.array([1.0, 10.0, 3.0])


def test_group_keys_read_a_whole_number_the_same_way_whatever_its_type():
    assert group_key(3) == group_key(3.0) == group_key(np.int64(3)) == group_key(np.float32(3.0)) == "3"
    assert group_key(2.5) == "2.5" and group_key("north") == "north" and group_key(True) == "True"
    assert list(group_keys(np.array([1.0, 2.0, 1.0]))) == ["1", "2", "1"]
    values = np.random.default_rng(0).uniform(size=1000)
    edges = disagreement_edges(values, 100)
    assert len(edges) == 4 and np.all(np.diff(edges) > 0)
    keys = disagreement_keys(values, edges)
    assert sorted(set(keys)) == ["0", "1", "2", "3", "4"] and all(150 < np.sum(keys == k) < 250 for k in set(keys))
    assert len(disagreement_edges(values[:300], 100)) == 2
    with pytest.raises(AshlarError):
        disagreement_edges(values[:150], 100)


def test_a_region_holds_every_outcome_whose_density_pit_is_below_its_level():
    rng = np.random.default_rng(0)
    drawn = rng.normal(size=(500, 2000))
    seen = rng.normal(size=2000)
    pit = density_pit(drawn, seen, rng=rng)
    # Uniform when the outcome is another draw.
    assert abs(np.mean(pit < 0.3) - 0.3) < 0.03
    level = 0.8
    inside = seen >= region_threshold(drawn, level)
    assert np.all(inside[pit <= level])
    assert abs(inside.mean() - level) < 0.03
    assert region_level(None, 0.8) == 0.8


def test_a_calibration_per_group_holds_within_each_group():
    x, y, group = _three_spreads(1500, 0)
    x_cal, y_cal, group_cal = _three_spreads(900, 1)
    x_new, y_new, group_new = _three_spreads(3000, 2)
    model = ASRRegressor(task="gaussian", max_epochs=5, random_state=0).fit(x, y)
    model.calibrate(x_cal, y_cal)
    pooled = _within(model.predict_interval(x_new, 0.8), y_new, group_new)
    # One calibration of every row holds 80% on average: far more in the quiet group, far less in the loud one.
    assert pooled[0] > 0.95 and pooled[2] < 0.6
    model.calibrate(x_cal, y_cal, groups=group_cal)
    assert sorted(model.calibration_["groups"]) == ["0", "1", "2"]
    grouped = _within(model.predict_interval(x_new, 0.8, groups=group_new), y_new, group_new)
    assert np.all(np.abs(grouped - 0.8) < 0.05), grouped
    probability = model.predict_exceedance(x_new, 2.0, groups=group_new)
    assert abs(probability[group_new == 2].mean() - np.mean(y_new[group_new == 2] > 2.0)) < 0.03
    with pytest.raises(AshlarError):
        model.predict_interval(x_new, 0.8)
    with pytest.raises(ShapeError):
        model.predict_interval(x_new, 0.8, groups=group_new[:10])
    assert model.predict_interval(x_new, 0.8, calibrated=False).shape == (3000, 2)
    # The raw head ignores the labels, so a call need not drop them to read it.
    assert np.allclose(model.predict_interval(x_new, 0.8, calibrated=False, groups=group_new), model.predict_interval(x_new, 0.8, calibrated=False))
    model.calibrate(x_cal, y_cal)
    with pytest.raises(AshlarError):
        model.predict_interval(x_new, 0.8, groups=group_new)


@pytest.mark.slow
def test_groups_from_a_column_fall_back_to_every_row_and_save():
    x, y, group = _three_spreads(2000, 0)
    x_cal, y_cal, group_cal = _three_spreads(900, 1)
    x_new, y_new, group_new = _three_spreads(1000, 2)
    # A fourth group too small for a table of its own reads the table of every row.
    group_cal = np.where(np.arange(len(group_cal)) < 30, 3, group_cal)
    group_new = np.where(np.arange(len(group_new)) < 30, 3, group_new)
    frame = lambda x, g: {"a": x[:, 0], "b": x[:, 1], "c": x[:, 2], "site": g.astype(np.float64)}
    model = ASRRegressor(task="gaussian", max_epochs=5, random_state=0).fit(frame(x, group), y)
    model.calibrate(frame(x_cal, group_cal), y_cal, groups="site")
    assert sorted(model.calibration_["groups"]) == ["0", "1", "2"]
    assert model.calibration_["by"] == {"column": "site"} and model.calibration_["min_rows"] == MIN_GROUP_ROWS
    # An integer column at prediction finds the tables a float column made.
    rows = frame(x_new, group_new)
    rows["site"] = group_new.astype(np.int64)
    interval = model.predict_interval(rows, 0.8)
    small = group_new == 3
    pooled = ASRModel.load(model.save("/tmp/ashlar-groups-test")).predict_interval(rows, 0.8)
    assert np.allclose(pooled, interval, atol=1e-4)
    every_row = model.calibrate(frame(x_cal, group_cal), y_cal).predict_interval(rows, 0.8)
    assert np.allclose(interval[small], every_row[small], atol=1e-4)
    assert not np.allclose(interval[~small], every_row[~small], atol=1e-3)
    with pytest.raises(AshlarError):
        model.calibrate(frame(x_cal, group_cal), y_cal, groups="nope")
    with pytest.raises(AshlarError):
        model.calibrate(frame(x_cal, group_cal), y_cal, groups="site", localize="members")
    with pytest.raises(AshlarError):
        model.calibrate(frame(x_cal, group_cal), y_cal, groups="site", min_group_rows=5)
    with pytest.raises(AshlarError):
        model.calibrate(frame(x_cal, group_cal), y_cal, localize="members")
    model.calibrate(frame(x_cal, group_cal), y_cal, groups="site")
    with pytest.raises(AshlarError):
        model.predict_interval(rows, 0.8, groups=group_new)


@pytest.mark.slow
def test_member_bins_calibrate_each_level_of_disagreement():
    x, y, _group = _three_spreads(2000, 0)
    x_cal, y_cal, _ = _three_spreads(1000, 1)
    x_new, y_new, _ = _three_spreads(2000, 2)
    model = ASRRegressor(
        task="gaussian", backbone="tabm", backbone_kwargs={"n_blocks": 1, "d_block": 32, "k": 8}, max_epochs=5, random_state=0
    ).fit(x, y)
    model.calibrate(x_cal, y_cal, localize="members")
    assert len(model.calibration_["by"]["members"]) == 4 and sorted(model.calibration_["groups"]) == ["0", "1", "2", "3", "4"]
    interval = model.predict_interval(x_new, 0.8)
    assert abs(np.mean((y_new >= interval[:, 0]) & (y_new <= interval[:, 1])) - 0.8) < 0.04
    assert model.predict_quantiles(x_new[:7], [0.1, 0.9]).shape == (7, 2)
    assert model.predict_exceedance(x_new[:7], 0.0).shape == (7,)
    restored = ASRModel.load(model.save("/tmp/ashlar-members-test"))
    assert np.allclose(restored.predict_interval(x_new[:50], 0.8), interval[:50], atol=1e-4)


@pytest.mark.slow
@pytest.mark.parametrize("task, kwargs", [("multivariate_gaussian", {}), ("multivariate_flow", {})])
def test_a_joint_region_covers_after_calibration(task, kwargs):
    x, y = _shared_shock(2000, 0)
    x_cal, y_cal = _shared_shock(1500, 1)
    x_new, y_new = _shared_shock(3000, 2)
    model = ASRRegressor(task=task, max_epochs=15, random_state=0, **kwargs).fit(x, y)
    density = np.asarray(model.log_prob(x_new, y_new))
    raw = model.predict_region(x_new, 0.8, n_samples=500)
    model.calibrate(x_cal, y_cal, n_samples=500)
    assert len(model.calibration_["region"]["values"]) == 1500
    calibrated = model.predict_region(x_new, 0.8, n_samples=500)
    assert abs(np.mean(density >= calibrated) - 0.8) < 0.03
    assert np.allclose(model.predict_region(x_new, 0.8, n_samples=500, calibrated=False), raw)
    # A calibration by group calibrates each group's region and each group's total.
    group = np.arange(len(y_cal)) % 2
    model.calibrate(x_cal, y_cal, n_samples=500, groups=group, min_group_rows=200)
    assert model.predict_region(x_new[:20], 0.8, n_samples=500, groups=np.arange(20) % 2).shape == (20,)
    assert model.predict_interval(x_new[:20], 0.8, target="sum", groups=np.arange(20) % 2).shape == (20, 2)
    with pytest.raises(AshlarError):
        ASRRegressor(task="gaussian", max_epochs=1).fit(x, y[:, 0]).predict_region(x_new)
