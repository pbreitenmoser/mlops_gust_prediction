"""Shared Hypothesis strategies for the property tests.

`readings_frames()` builds raw Readings (same columns as the parsed MeteoSwiss CSV)
on a 10-minute UTC grid with absent rows (gaps) and NaN values.
"""

from __future__ import annotations

import pandas as pd
from hypothesis import settings
from hypothesis import strategies as st

from src.common import config

# One settings object for every property test (max_examples=100).
PROPERTY_SETTINGS = settings(max_examples=100, deadline=None)

_NUMERIC_COLUMNS = [c for c in config.RAW_COLUMNS if c not in ("station_abbr", config.EVENT_TIME)]
_STEP = pd.Timedelta(minutes=config.INTERVAL_MIN)

_START = st.datetimes(
    min_value=pd.Timestamp("2020-01-01").to_pydatetime(),
    max_value=pd.Timestamp("2030-01-01").to_pydatetime(),
).map(lambda d: pd.Timestamp(d).floor(f"{config.INTERVAL_MIN}min").tz_localize("UTC"))

# Small values keep float arithmetic well behaved; integers make sums exact.
_VALUE = st.integers(min_value=-50, max_value=150).map(float)


@st.composite
def value_for(draw, nan_prob: float) -> float:
    """A float value, or NaN with probability `nan_prob`."""
    if draw(st.floats(min_value=0, max_value=1)) < nan_prob:
        return float("nan")
    return draw(_VALUE)


def make_row(ts: pd.Timestamp, values: dict) -> dict:
    row = {"station_abbr": config.STATION, config.EVENT_TIME: ts}
    row.update({c: values[c] for c in _NUMERIC_COLUMNS})
    return row


@st.composite
def readings_frames(
    draw,
    min_steps: int = 1,
    max_steps: int = 120,
    max_gap_prob: float = 0.3,
    max_nan_prob: float = 0.3,
) -> pd.DataFrame:
    """Raw Readings spanning up to `max_steps` grid steps, with gaps and NaN.

    The first step is always present, so the frame is never empty. Gap and NaN
    probabilities are drawn per example (often 0) so that fully valid windows,
    gappy windows and NaN-heavy windows all occur.
    """
    start = draw(_START)
    steps = draw(st.integers(min_value=min_steps, max_value=max_steps))
    gap_prob = draw(st.sampled_from([0.0, 0.0, max_gap_prob / 3, max_gap_prob]))
    nan_prob = draw(st.sampled_from([0.0, 0.0, max_nan_prob / 3, max_nan_prob]))

    rows = []
    for i in range(steps):
        if i > 0 and draw(st.floats(min_value=0, max_value=1)) < gap_prob:
            continue
        values = {c: draw(value_for(nan_prob)) for c in _NUMERIC_COLUMNS}
        rows.append(make_row(start + i * _STEP, values))
    return pd.DataFrame(rows, columns=config.RAW_COLUMNS)


@st.composite
def future_edits(draw, frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Return `frame` with arbitrary changes strictly after `t`.

    Future readings are changed (values/NaN), removed, and new ones are appended
    (including after a gap). Readings at or before `t` are untouched.
    """
    past = frame[frame[config.EVENT_TIME] <= t]
    future = frame[frame[config.EVENT_TIME] > t]
    kept = []
    for _, row in future.iterrows():
        action = draw(st.sampled_from(["keep", "change", "drop"]))
        if action == "drop":
            continue
        values = row.to_dict()
        if action == "change":
            for c in _NUMERIC_COLUMNS:
                values[c] = draw(value_for(0.2))
        kept.append(values)
    last = frame[config.EVENT_TIME].max()
    extra = draw(st.integers(min_value=0, max_value=20))
    offset = draw(st.integers(min_value=1, max_value=5))
    for i in range(extra):
        values = {c: draw(value_for(0.2)) for c in _NUMERIC_COLUMNS}
        kept.append(make_row(last + (offset + i) * _STEP, values))
    future_df = pd.DataFrame(kept, columns=config.RAW_COLUMNS)
    if future_df.empty:
        return past.reset_index(drop=True)
    return pd.concat([past, future_df], ignore_index=True)


@st.composite
def past_edits(draw, frame: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Return `frame` with arbitrary value changes at timestamps <= `t` (no rows added/removed)."""
    out = frame.copy()
    mask = out[config.EVENT_TIME] <= t
    for idx in out.index[mask]:
        for c in _NUMERIC_COLUMNS:
            if draw(st.booleans()):
                out.at[idx, c] = draw(value_for(0.2))
    return out
