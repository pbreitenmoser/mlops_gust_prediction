"""Property tests for metrics, selection, inference row assembly and schema check.

Strategies are self-contained in this file.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.common import config
from src.common.features import (
    FeatureSchemaError,
    build_inference_row,
    validate_feature_columns,
)
from src.common.metrics import mae, persistence_baseline_rmse, rmse, time_split
from src.common.selection import (
    drop_unlabeled,
    select_insert_window,
    select_latest_row,
    select_latest_version,
)

PBT = settings(max_examples=100, deadline=None)
TS = config.EVENT_TIME
LABEL = config.LABEL_COLUMN
BASE = pd.Timestamp("2024-01-01", tz="UTC")

finite = st.floats(min_value=-100, max_value=200, allow_nan=False, allow_infinity=False)
maybe_nan = st.one_of(finite, st.just(float("nan")))


@st.composite
def unique_timestamps(draw, min_size=1, max_size=40):
    offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=20_000),
            min_size=min_size,
            max_size=max_size,
            unique=True,
        )
    )
    return [BASE + pd.Timedelta(minutes=10 * o) for o in offsets]


@st.composite
def labeled_frames(draw, min_size=1, max_size=40):
    """Frame in arbitrary row order with unique timestamps, nullable label and gust_now."""
    stamps = draw(unique_timestamps(min_size=min_size, max_size=max_size))
    n = len(stamps)
    return pd.DataFrame(
        {
            TS: stamps,
            "gust_now": draw(st.lists(maybe_nan, min_size=n, max_size=n)),
            LABEL: draw(st.lists(maybe_nan, min_size=n, max_size=n)),
        }
    )


# Feature: wind-gust-forecasting-mlops, Property 13: Null labels are dropped and counted
# Validates: Requirements 7.2
@PBT
@given(df=labeled_frames())
def test_property_13_null_labels_dropped_and_counted(df):
    labeled, dropped = drop_unlabeled(df)
    assert dropped == int(df[LABEL].isna().sum())
    assert len(labeled) + dropped == len(df)
    assert labeled[LABEL].notna().all()
    expected = df[df[LABEL].notna()]
    pd.testing.assert_frame_equal(labeled, expected)


# Feature: wind-gust-forecasting-mlops, Property 14: Time split is ordered, proportional and unshuffled
# Validates: Requirements 7.3, 7.4, 7.6, 7.7, 8.6
@PBT
@given(df=labeled_frames(min_size=2))
def test_property_14_time_split(df):
    train, test = time_split(df)
    n = len(df)
    n_train = math.floor(0.8 * n)
    assert len(train) == n_train
    assert len(test) == n - n_train
    assert len(train) > 0 and len(test) > 0
    # Strictly ordered: last train < first test; both internally ascending
    assert train[TS].max() < test[TS].min()
    assert train[TS].is_monotonic_increasing
    assert test[TS].is_monotonic_increasing
    # Partition: same set of rows, none lost or duplicated
    combined = pd.concat([train, test])
    assert sorted(combined[TS]) == sorted(df[TS])
    assert combined[TS].is_unique
    # Train is exactly the oldest n_train timestamps
    assert list(train[TS]) == sorted(df[TS])[:n_train]


def test_property_14_single_row_raises():
    df = pd.DataFrame({TS: [BASE], LABEL: [1.0], "gust_now": [1.0]})
    with pytest.raises(ValueError):
        time_split(df)


# Feature: wind-gust-forecasting-mlops, Property 15: Metrics match their definitions
# Validates: Requirements 8.2, 8.3
@PBT
@given(
    pairs=st.lists(st.tuples(finite, finite), min_size=1, max_size=50),
)
def test_property_15_metrics_match_definitions(pairs):
    y_true = [p[0] for p in pairs]
    y_pred = [p[1] for p in pairs]
    n = len(pairs)
    expected_rmse = math.sqrt(sum((a - b) ** 2 for a, b in pairs) / n)
    expected_mae = sum(abs(a - b) for a, b in pairs) / n
    assert rmse(y_true, y_pred) == pytest.approx(expected_rmse, rel=1e-9, abs=1e-9)
    assert mae(y_true, y_pred) == pytest.approx(expected_mae, rel=1e-9, abs=1e-9)
    assert mae(y_true, y_pred) <= rmse(y_true, y_pred) + 1e-9


@PBT
@given(df=labeled_frames())
def test_property_15_persistence_baseline(df):
    valid = df.dropna(subset=["gust_now", LABEL])
    if valid.empty:
        with pytest.raises(ValueError):
            persistence_baseline_rmse(df)
        return
    value, used = persistence_baseline_rmse(df)
    assert used == len(valid)
    diffs = [(g - l) ** 2 for g, l in zip(valid["gust_now"], valid[LABEL])]
    assert value == pytest.approx(math.sqrt(sum(diffs) / len(diffs)), rel=1e-9, abs=1e-9)


# Feature: wind-gust-forecasting-mlops, Property 16: Latest model version is the maximum
# Validates: Requirements 9.3, 10.4
@PBT
@given(versions=st.lists(st.integers(min_value=1, max_value=10_000), min_size=1, max_size=30, unique=True))
def test_property_16_latest_version_is_max(versions):
    models = [SimpleNamespace(version=v) for v in versions]
    chosen = select_latest_version(models)
    assert chosen.version == max(versions)


def test_property_16_empty_raises():
    with pytest.raises(ValueError, match="model not found"):
        select_latest_version([])


# Feature: wind-gust-forecasting-mlops, Property 17: Latest reading and latest stored row selection
# Validates: Requirements 10.1, 10.3, 10.11
@PBT
@given(df=labeled_frames(), data=st.data())
def test_property_17_latest_row_is_newest_regardless_of_order_or_label(df, data):
    perm = data.draw(st.permutations(list(range(len(df)))))
    shuffled = df.iloc[list(perm)].reset_index(drop=True)
    row = select_latest_row(shuffled)
    assert row[TS] == df[TS].max()
    # The matching row is returned whole, whether labeled or not
    expected = df[df[TS] == df[TS].max()].iloc[0]
    if pd.isna(expected[LABEL]):
        assert pd.isna(row[LABEL])
    else:
        assert row[LABEL] == expected[LABEL]


def test_property_17_empty_raises():
    with pytest.raises(ValueError):
        select_latest_row(pd.DataFrame({TS: [], LABEL: []}))


# Feature: wind-gust-forecasting-mlops, Property 7 (window part): insert window
# Validates: Requirements 6.5
@PBT
@given(df=labeled_frames())
def test_property_7_insert_window(df):
    selected = select_insert_window(df)
    newest = df[TS].max()
    cutoff = newest - pd.Timedelta(days=config.INCREMENTAL_INSERT_DAYS)
    assert newest in set(selected[TS])
    assert (selected[TS] >= cutoff).all()
    # Nothing at or after the cutoff is lost
    assert len(selected) == int((df[TS] >= cutoff).sum())


# Feature: wind-gust-forecasting-mlops, Property 18: Inference row assembly
# Validates: Requirements 10.2, 10.5, 10.10
@st.composite
def stored_and_live(draw):
    stored = {c: draw(finite) for c in config.AGGREGATED_FEATURES}
    # Stored realtime values and the label must not leak into the result
    stored.update({c: draw(finite) for c in config.REALTIME_FEATURES})
    stored[LABEL] = draw(finite)
    stored[TS] = BASE
    stored["station"] = config.STATION
    raw = {
        c: draw(finite)
        for c in config.RAW_COLUMNS
        if c not in ("station_abbr", TS)
    }
    raw["dkl010z0"] = draw(st.floats(min_value=0, max_value=360))
    live = pd.Series(raw)
    live["station_abbr"] = config.STATION
    live[TS] = BASE + pd.Timedelta(hours=1)
    return pd.Series(stored), live


@PBT
@given(pair=stored_and_live())
def test_property_18_inference_row_assembly(pair):
    stored, live = pair
    row = build_inference_row(stored, live)
    assert list(row.columns) == config.FEATURE_COLUMNS
    assert len(row) == 1
    assert LABEL not in row.columns
    for c in config.AGGREGATED_FEATURES:
        assert row[c].iloc[0] == stored[c]
    rad = math.radians(live["dkl010z0"])
    assert row["gust_now"].iloc[0] == live["fu3010z1"]
    assert row["wind_speed_now"].iloc[0] == live["fu3010z0"]
    assert row["wind_dir_sin"].iloc[0] == pytest.approx(math.sin(rad), abs=1e-9)
    assert row["wind_dir_cos"].iloc[0] == pytest.approx(math.cos(rad), abs=1e-9)
    assert row["temp_now"].iloc[0] == live["tre200s0"]
    assert row["humidity_now"].iloc[0] == live["ure200s0"]
    assert row["pressure_now"].iloc[0] == live["pp0qffs0"]
    assert row["precip_now"].iloc[0] == live["rre150z0"]
    assert row["radiation_now"].iloc[0] == live["gre000z0"]


def test_property_18_missing_live_value_is_nan():
    stored = pd.Series({c: 1.0 for c in config.AGGREGATED_FEATURES})
    live = pd.Series({"fu3010z1": 5.0, "fu3010z0": 2.0, "dkl010z0": 90.0})
    row = build_inference_row(stored, live)
    assert row["gust_now"].iloc[0] == 5.0
    assert np.isnan(row["temp_now"].iloc[0])
    assert list(row.columns) == config.FEATURE_COLUMNS


# Feature: wind-gust-forecasting-mlops, Property 19: Feature column schema check
# Validates: Requirements 11.5
def _is_exact(cols):
    return list(cols) == config.FEATURE_COLUMNS


@PBT
@given(
    cols=st.one_of(
        st.permutations(config.FEATURE_COLUMNS),
        st.lists(st.sampled_from(config.FEATURE_COLUMNS + ["extra", LABEL]), max_size=20),
        st.just(list(config.FEATURE_COLUMNS)),
    )
)
def test_property_19_schema_check(cols):
    cols = list(cols)
    if _is_exact(cols):
        validate_feature_columns(cols)
    else:
        with pytest.raises(FeatureSchemaError):
            validate_feature_columns(cols)


def test_property_19_error_names_problems():
    cols = list(config.FEATURE_COLUMNS)
    with pytest.raises(FeatureSchemaError, match="missing"):
        validate_feature_columns(cols[:-1])
    with pytest.raises(FeatureSchemaError, match="unexpected"):
        validate_feature_columns(cols + ["extra"])
    with pytest.raises(FeatureSchemaError, match="misordered"):
        validate_feature_columns(cols[::-1])
