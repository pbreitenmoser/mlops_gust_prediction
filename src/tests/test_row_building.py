"""Property and example tests for row building and path consistency (src/common/features.py).

Properties 4 to 6 of the design plus hand-calculated examples.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.common import config
from src.common.features import (
    build_feature_rows,
    compute_aggregated_features,
    compute_latest_aggregates,
    to_grid,
)
from src.tests.strategies import PROPERTY_SETTINGS, make_row, readings_frames

_TS = config.EVENT_TIME
_STEP = pd.Timedelta(minutes=config.INTERVAL_MIN)
_OUTPUT_COLUMNS = [config.PRIMARY_KEY, _TS] + config.FEATURE_COLUMNS + [config.LABEL_COLUMN]
_NUMERIC = [c for c in config.RAW_COLUMNS if c not in ("station_abbr", _TS)]


def _agg_present(grid: pd.DataFrame) -> pd.Series:
    return ~compute_aggregated_features(grid).isna().any(axis=1)


# Feature: wind-gust-forecasting-mlops, Property 4: Row retention keeps unlabeled rows
# **Validates: Requirements 4.5, 5.2, 6.1, 6.13**
@PROPERTY_SETTINGS
@given(frame=readings_frames(max_steps=120, max_gap_prob=0.1, max_nan_prob=0.05))
def test_property_4_row_retention_keeps_unlabeled_rows(frame):
    rows, stats = build_feature_rows(frame)
    grid = to_grid(frame)
    expected_ts = list(grid.index[_agg_present(grid)])

    assert list(rows[_TS]) == expected_ts
    assert list(rows.columns) == _OUTPUT_COLUMNS
    assert (rows[config.PRIMARY_KEY] == config.STATION).all()
    assert rows[_TS].is_unique
    # Aggregated features are never null in kept rows.
    assert not rows[config.AGGREGATED_FEATURES].isna().any().any()

    labeled = int(rows[config.LABEL_COLUMN].notna().sum())
    assert stats.rows == len(rows)
    assert stats.labeled == labeled
    assert stats.unlabeled == len(rows) - labeled
    assert stats.labeled + stats.unlabeled == stats.rows
    assert stats.dropped_missing == len(grid) - len(rows)


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return np.allclose(
        a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64"), rtol=0, atol=1e-9, equal_nan=True
    )


# Feature: wind-gust-forecasting-mlops, Property 5: Labels are stable as data grows and unlabeled rows get filled in
# **Validates: Requirements 6.3, 6.12**
@PROPERTY_SETTINGS
@given(data=st.data(), frame=readings_frames(min_steps=2, max_steps=120, max_gap_prob=0.1, max_nan_prob=0.05))
def test_property_5_labels_stable_as_data_grows(data, frame):
    timestamps = sorted(frame[_TS])
    cut = data.draw(st.sampled_from(timestamps))
    prefix = frame[frame[_TS] <= cut]

    prefix_rows, _ = build_feature_rows(prefix)
    full_rows, _ = build_feature_rows(frame)

    assert prefix_rows[_TS].is_unique and full_rows[_TS].is_unique
    full_by_ts = full_rows.set_index(_TS)
    gust = dict(zip(frame[_TS], frame["fu3010z1"]))

    for _, row in prefix_rows.iterrows():
        ts = row[_TS]
        assert ts in full_by_ts.index  # features depend on the past only, so the row stays
        ext = full_by_ts.loc[ts]
        if not math.isnan(row[config.LABEL_COLUMN]):
            # Known values never change.
            assert ext[config.LABEL_COLUMN] == row[config.LABEL_COLUMN]
            same = np.allclose(
                row[config.FEATURE_COLUMNS].to_numpy(dtype="float64"),
                ext[config.FEATURE_COLUMNS].to_numpy(dtype="float64"),
                rtol=0, atol=1e-9, equal_nan=True,
            )
            assert same
        else:
            future = [gust.get(ts + k * _STEP, math.nan) for k in range(1, config.HORIZON_STEPS + 1)]
            if all(not math.isnan(v) for v in future):
                assert not math.isnan(ext[config.LABEL_COLUMN])
                assert ext[config.LABEL_COLUMN] == max(future)


# Feature: wind-gust-forecasting-mlops, Property 6: Pipeline path and inference path agree on aggregated features
# **Validates: Requirements 11.3, 11.4**
@PROPERTY_SETTINGS
@given(data=st.data(), frame=readings_frames(min_steps=37, max_steps=120, max_gap_prob=0.05, max_nan_prob=0.03))
def test_property_6_pipeline_and_inference_paths_agree(data, frame):
    t = data.draw(st.sampled_from(sorted(frame[_TS])))
    truncated = frame[frame[_TS] <= t]

    latest = compute_latest_aggregates(truncated)
    assert list(latest.index) == config.AGGREGATED_FEATURES

    rows, _ = build_feature_rows(frame)
    match = rows[rows[_TS] == t]
    if len(match) == 1:
        pipeline = match.iloc[0][config.AGGREGATED_FEATURES].to_numpy(dtype="float64")
        assert not np.isnan(latest.to_numpy(dtype="float64")).any()
        assert np.allclose(pipeline, latest.to_numpy(dtype="float64"), rtol=0, atol=1e-9)
    else:
        # The pipeline dropped the row because some feature is null; the inference path
        # must show the same null (and the row is not silently valid there).
        assert latest.isna().any()
        full_agg = compute_aggregated_features(to_grid(frame)).loc[t]
        assert np.array_equal(full_agg.isna().to_numpy(), latest.isna().to_numpy())
        both = ~full_agg.isna().to_numpy()
        assert np.allclose(
            full_agg.to_numpy(dtype="float64")[both],
            latest.to_numpy(dtype="float64")[both],
            rtol=0, atol=1e-9,
        )


# Feature: wind-gust-forecasting-mlops, Property 6: Short sequences have null features for uncovered windows in both paths
# **Validates: Requirements 11.3, 11.4**
@PROPERTY_SETTINGS
@given(frame=readings_frames(max_steps=36, max_gap_prob=0.0, max_nan_prob=0.0))
def test_property_6_short_sequences_null_in_both_paths(frame):
    n = len(frame)
    latest = compute_latest_aggregates(frame)
    full = compute_aggregated_features(to_grid(frame)).iloc[-1]
    assert np.array_equal(latest.isna().to_numpy(), full.isna().to_numpy())
    for label, w in config.WINDOWS.items():
        if n < w:
            assert math.isnan(latest[f"gust_max_{label}"])
            assert math.isnan(latest[f"wind_mean_{label}"])
    if n <= config.WINDOWS["3h"]:
        assert math.isnan(latest["pressure_change_3h"])
    rows, _ = build_feature_rows(frame)
    if n < config.WINDOWS["6h"]:
        assert rows.empty  # fewer than 36 readings: no complete 6 h window anywhere
    else:
        assert len(rows) == 1  # exactly 36 readings: only the last grid point is complete


# ---------------------------------------------------------------------------
# Hand-calculated examples
# ---------------------------------------------------------------------------

_START = pd.Timestamp("2024-03-01 00:00", tz="UTC")


def _build(n: int, drop: set[int] = frozenset(), gust_nan: set[int] = frozenset()) -> pd.DataFrame:
    """Reading i (0-based): gust=i (90 at i=37), wind=2i, pressure=1000+0.5i, other columns 0."""
    rows = []
    for i in range(n):
        if i in drop:
            continue
        values = {c: 0.0 for c in _NUMERIC}
        values["fu3010z1"] = 90.0 if i == 37 else float(i)
        values["fu3010z0"] = 2.0 * i
        values["pp0qffs0"] = 1000.0 + 0.5 * i
        if i in gust_nan:
            values["fu3010z1"] = float("nan")
        rows.append(make_row(_START + i * _STEP, values))
    return pd.DataFrame(rows, columns=config.RAW_COLUMNS)


def _at(rows: pd.DataFrame, i: int) -> pd.Series:
    match = rows[rows[_TS] == _START + i * _STEP]
    assert len(match) == 1
    return match.iloc[0]


def test_example_aggregated_features_hand_calculated():
    rows, stats = build_feature_rows(_build(45))
    # First complete 6 h window ends at reading 35.
    assert list(rows[_TS]) == [_START + i * _STEP for i in range(35, 45)]
    assert stats.rows == 10

    # i=35: gust 0..35 -> 1h/3h/6h max = 35; wind=2i: mean(2*30..2*35)=65,
    # mean(2*18..2*35)=53, mean(2*0..2*35)=35; pressure change = 0.5*18 = 9
    r = _at(rows, 35)
    assert r["gust_max_1h"] == 35 and r["gust_max_3h"] == 35 and r["gust_max_6h"] == 35
    assert r["wind_mean_1h"] == pytest.approx(65.0)
    assert r["wind_mean_3h"] == pytest.approx(53.0)
    assert r["wind_mean_6h"] == pytest.approx(35.0)
    assert r["pressure_change_3h"] == pytest.approx(9.0)

    # i=37: spike 90 is inside every window; wind means 2i-5, 2i-17, 2i-35.
    r = _at(rows, 37)
    assert (r["gust_max_1h"], r["gust_max_3h"], r["gust_max_6h"]) == (90, 90, 90)
    assert r["wind_mean_1h"] == pytest.approx(69.0)
    assert r["wind_mean_3h"] == pytest.approx(57.0)
    assert r["wind_mean_6h"] == pytest.approx(39.0)

    # i=43: spike (i=37) leaves the 1h window (38..43) but stays in 3h and 6h.
    r = _at(rows, 43)
    assert r["gust_max_1h"] == 43
    assert r["gust_max_3h"] == 90 and r["gust_max_6h"] == 90
    assert r["wind_mean_1h"] == pytest.approx(81.0)
    assert r["pressure_change_3h"] == pytest.approx(9.0)


def test_example_labels_hand_calculated():
    rows, stats = build_feature_rows(_build(45))
    # Label(i) = max gust of readings i+1..i+6.
    assert _at(rows, 35)[config.LABEL_COLUMN] == 90  # window 36..41 contains the spike
    assert _at(rows, 36)[config.LABEL_COLUMN] == 90  # 37..42
    assert _at(rows, 37)[config.LABEL_COLUMN] == 43  # 38..43, own value 90 excluded
    assert _at(rows, 38)[config.LABEL_COLUMN] == 44  # 39..44
    # Last six readings have no complete future window.
    for i in range(39, 45):
        assert math.isnan(_at(rows, i)[config.LABEL_COLUMN])
    assert (stats.labeled, stats.unlabeled) == (4, 6)


def test_example_incomplete_future_window_gives_kept_null_label_rows():
    # Reading 42 is absent: windows needing it have no label; aggregates of 42+ are dropped.
    rows, stats = build_feature_rows(_build(45, drop={42}))
    assert list(rows[_TS]) == [_START + i * _STEP for i in range(35, 42)]
    assert _at(rows, 35)[config.LABEL_COLUMN] == 90  # window 36..41 is complete
    for i in range(36, 42):
        assert math.isnan(_at(rows, i)[config.LABEL_COLUMN])  # every window includes 42
    assert (stats.labeled, stats.unlabeled) == (1, 6)


def test_example_missing_future_gust_gives_kept_null_label_rows():
    # Reading 40 exists but fu3010z1 is missing.
    rows, stats = build_feature_rows(_build(45, gust_nan={40}))
    assert list(rows[_TS]) == [_START + i * _STEP for i in range(35, 40)]
    for i in range(35, 40):  # each future window 36..41 ... 39..45 contains reading 40
        assert math.isnan(_at(rows, i)[config.LABEL_COLUMN])
    assert (stats.labeled, stats.unlabeled) == (0, 5)
    assert stats.affected_raw_rows == 1


def test_example_pipeline_and_inference_paths_agree():
    frame = _build(45)
    rows, _ = build_feature_rows(frame)
    truncated = frame[frame[_TS] <= _START + 40 * _STEP]
    latest = compute_latest_aggregates(truncated)

    expected = {
        "gust_max_1h": 90.0, "gust_max_3h": 90.0, "gust_max_6h": 90.0,
        "wind_mean_1h": 75.0, "wind_mean_3h": 63.0, "wind_mean_6h": 45.0,
        "pressure_change_3h": 9.0,
    }
    for name, value in expected.items():
        assert latest[name] == pytest.approx(value, abs=1e-9)
    stored = _at(rows, 40)
    for name in config.AGGREGATED_FEATURES:
        assert stored[name] == pytest.approx(latest[name], abs=1e-9)
