"""Property tests for aggregated features and labels (src/common/features.py).

The oracle is a brute-force, dictionary-based implementation that looks up each
timestamp explicitly, so absent rows and NaN are handled without any grid logic.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from src.common import config
from src.common.features import (
    compute_aggregated_features,
    compute_label,
    to_grid,
)
from src.tests.strategies import (
    PROPERTY_SETTINGS,
    future_edits,
    past_edits,
    readings_frames,
)

_TS = config.EVENT_TIME
_STEP = pd.Timedelta(minutes=config.INTERVAL_MIN)


def _lookup(frame: pd.DataFrame, column: str) -> dict:
    return dict(zip(frame[_TS], frame[column]))


def _window(values: dict, t: pd.Timestamp, n: int):
    """Values at t-(n-1)*10min ... t, or None if a reading is absent or missing."""
    out = []
    for k in range(n):
        v = values.get(t - k * _STEP)
        if v is None or math.isnan(v):
            return None
        out.append(v)
    return out


def _expected_agg(frame: pd.DataFrame, t: pd.Timestamp) -> dict:
    gust = _lookup(frame, "fu3010z1")
    wind = _lookup(frame, "fu3010z0")
    pres = _lookup(frame, "pp0qffs0")
    exp = {}
    for label, w in config.WINDOWS.items():
        g = _window(gust, t, w)
        exp[f"gust_max_{label}"] = max(g) if g else math.nan
        m = _window(wind, t, w)
        exp[f"wind_mean_{label}"] = sum(m) / w if m else math.nan
    now = pres.get(t, math.nan)
    before = pres.get(t - config.WINDOWS["3h"] * _STEP, math.nan)
    exp["pressure_change_3h"] = now - before  # NaN if either is NaN/absent
    return exp


def _assert_same(actual, expected: dict) -> None:
    for name, exp in expected.items():
        got = actual[name]
        if math.isnan(exp):
            assert math.isnan(got), f"{name}: expected NaN, got {got}"
        else:
            assert not math.isnan(got), f"{name}: expected {exp}, got NaN"
            assert math.isclose(got, exp, rel_tol=1e-9, abs_tol=1e-9), f"{name}: {got} != {exp}"


# Feature: wind-gust-forecasting-mlops, Property 1: Aggregated features match the definition
# **Validates: Requirements 4.1, 4.2, 4.3, 4.5, 4.7, 11.4**
@PROPERTY_SETTINGS
@given(frame=readings_frames(max_steps=120))
def test_property_1_aggregated_features_match_definition(frame):
    grid = to_grid(frame)
    agg = compute_aggregated_features(grid)
    assert list(agg.columns) == config.AGGREGATED_FEATURES
    # Every grid timestamp (including gap rows) is checked against the oracle.
    for t in grid.index:
        _assert_same(agg.loc[t], _expected_agg(frame, t))


# Feature: wind-gust-forecasting-mlops, Property 2: Aggregated features use only past data
# **Validates: Requirements 4.4**
@PROPERTY_SETTINGS
@given(data=st.data(), frame=readings_frames(max_steps=120))
def test_property_2_aggregated_features_use_only_past_data(data, frame):
    t = data.draw(st.sampled_from(list(frame[_TS])), label="t")
    edited = data.draw(future_edits(frame, t), label="edited")

    before = compute_aggregated_features(to_grid(frame)).loc[t]
    after = compute_aggregated_features(to_grid(edited)).loc[t]
    pd.testing.assert_series_equal(before, after, check_exact=True)


def _expected_label(frame: pd.DataFrame, t: pd.Timestamp) -> float:
    gust = _lookup(frame, "fu3010z1")
    future = [gust.get(t + k * _STEP) for k in range(1, config.HORIZON_STEPS + 1)]
    if any(v is None or math.isnan(v) for v in future):
        return math.nan
    return max(future)


# Feature: wind-gust-forecasting-mlops, Property 3: Label is the maximum of the next six gusts, else null
# **Validates: Requirements 5.1, 5.2, 5.3, 5.6**
@PROPERTY_SETTINGS
@given(data=st.data(), frame=readings_frames(max_steps=120, max_gap_prob=0.15, max_nan_prob=0.15))
def test_property_3_label_is_max_of_next_six_gusts(data, frame):
    grid = to_grid(frame)
    label = compute_label(grid)
    assert label.name == config.LABEL_COLUMN

    for t in grid.index:
        exp = _expected_label(frame, t)
        got = label.loc[t]
        if math.isnan(exp):
            assert math.isnan(got), f"{t}: expected NaN, got {got}"
        else:
            assert got == exp, f"{t}: {got} != {exp}"

    # Changing readings at or before t (< t + 10 min) never changes the label at t.
    t = data.draw(st.sampled_from(list(grid.index)), label="t")
    edited = data.draw(past_edits(frame, t), label="edited")
    after = compute_label(to_grid(edited))
    a, b = label.loc[t], after.loc[t]
    assert (math.isnan(a) and math.isnan(b)) or a == b


# Feature: wind-gust-forecasting-mlops, Property 3: Label is null when no six valid future readings exist
# **Validates: Requirements 5.2, 5.6**
@PROPERTY_SETTINGS
@given(frame=readings_frames(max_steps=config.HORIZON_STEPS))
def test_property_3_short_sequences_have_only_null_labels(frame):
    # At most six grid steps: no timestamp can have six future readings.
    label = compute_label(to_grid(frame))
    assert label.isna().all()
    assert len(label) == len(to_grid(frame))
    assert not np.any(label.notna())
