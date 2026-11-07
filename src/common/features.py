"""Pure feature, label and row-building logic (no I/O, no Hopsworks).

Source of the data: MeteoSwiss. Everything is index-based on a complete 10-minute
UTC grid, so a rolling window of ``w`` rows is exactly ``(t - w*10 min, t]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from src.common import config

logger = logging.getLogger(__name__)

_TS = config.EVENT_TIME
_GUST = "fu3010z1"
_WIND = "fu3010z0"
_DIR = "dkl010z0"
_PRESSURE = "pp0qffs0"
# Raw columns that feed the Aggregated_Features.
_AGGREGATE_SOURCES = [_GUST, _WIND, _PRESSURE]
_PRESSURE_LAG_STEPS = config.WINDOWS["3h"]
_LATEST_SLICE = max(config.WINDOWS.values()) + 1  # 37 readings


class FeatureSchemaError(Exception):
    """Raised when feature columns differ from config.FEATURE_COLUMNS."""


@dataclass(frozen=True)
class FeatureRowStats:
    """Counts describing the result of build_feature_rows."""

    rows: int  # rows returned (labeled + unlabeled)
    labeled: int
    unlabeled: int
    dropped_missing: int  # grid rows dropped because an aggregated feature is missing
    affected_raw_rows: int  # raw readings with a missing value in an aggregate source column


def to_grid(readings: pd.DataFrame) -> pd.DataFrame:
    """Return readings on a complete 10-minute grid indexed by `reference_timestamp`.

    Duplicates are removed (last wins), rows are sorted, and absent timestamps
    become all-NaN rows. Only numeric raw columns are kept.
    """
    numeric = [c for c in config.RAW_COLUMNS if c not in ("station_abbr", _TS)]
    if readings.empty:
        empty = pd.DataFrame(columns=numeric, dtype="float64")
        empty.index = pd.DatetimeIndex([], tz="UTC", name=_TS)
        return empty
    df = readings.drop_duplicates(subset=_TS, keep="last").sort_values(_TS)
    df = df.set_index(_TS)[numeric].astype("float64")
    full = pd.date_range(
        df.index.min(), df.index.max(), freq=f"{config.INTERVAL_MIN}min", name=_TS
    )
    return df.reindex(full)


def compute_aggregated_features(grid: pd.DataFrame) -> pd.DataFrame:
    """Seven Aggregated_Features, indexed like `grid`.

    NaN where the window is incomplete (start of data) or contains a missing value.
    """
    out = pd.DataFrame(index=grid.index)
    for label, w in config.WINDOWS.items():
        out[f"gust_max_{label}"] = grid[_GUST].rolling(window=w, min_periods=w).max()
    for label, w in config.WINDOWS.items():
        out[f"wind_mean_{label}"] = grid[_WIND].rolling(window=w, min_periods=w).mean()
    out["pressure_change_3h"] = grid[_PRESSURE] - grid[_PRESSURE].shift(_PRESSURE_LAG_STEPS)
    return out[config.AGGREGATED_FEATURES]


def compute_realtime_columns(grid: pd.DataFrame) -> pd.DataFrame:
    """Nine realtime columns (the Reading's own values), indexed like `grid`."""
    rad = np.deg2rad(grid[_DIR])
    out = pd.DataFrame(
        {
            "gust_now": grid[_GUST],
            "wind_speed_now": grid[_WIND],
            "wind_dir_sin": np.sin(rad),
            "wind_dir_cos": np.cos(rad),
            "temp_now": grid["tre200s0"],
            "humidity_now": grid["ure200s0"],
            "pressure_now": grid[_PRESSURE],
            "precip_now": grid["rre150z0"],
            "radiation_now": grid["gre000z0"],
        },
        index=grid.index,
    )
    return out[config.REALTIME_FEATURES]


def compute_label(grid: pd.DataFrame) -> pd.Series:
    """Max gust over t+10 ... t+60 min; NaN unless all six future readings exist."""
    future = pd.concat(
        [grid[_GUST].shift(-k) for k in range(1, config.HORIZON_STEPS + 1)], axis=1
    )
    label = future.max(axis=1, skipna=False)
    label.name = config.LABEL_COLUMN
    return label


def build_feature_rows(readings: pd.DataFrame) -> tuple[pd.DataFrame, FeatureRowStats]:
    """Build Feature_Group rows (features + label) from raw readings.

    Drops rows with any missing Aggregated_Feature; keeps unlabeled rows.
    """
    grid = to_grid(readings)
    aggregated = compute_aggregated_features(grid)
    realtime = compute_realtime_columns(grid)
    label = compute_label(grid)

    # Only count readings that actually exist (not grid-filled gaps).
    present = readings.drop_duplicates(subset=_TS, keep="last")
    affected_raw = int(present[_AGGREGATE_SOURCES].isna().any(axis=1).sum()) if len(present) else 0

    frame = pd.concat([aggregated, realtime], axis=1)
    frame[config.LABEL_COLUMN] = label
    complete = ~aggregated.isna().any(axis=1)
    dropped = int((~complete).sum())
    frame = frame[complete]

    frame = frame.reset_index()
    frame.insert(0, config.PRIMARY_KEY, config.STATION)
    frame = frame[
        [config.PRIMARY_KEY, _TS] + config.FEATURE_COLUMNS + [config.LABEL_COLUMN]
    ].reset_index(drop=True)

    labeled = int(frame[config.LABEL_COLUMN].notna().sum())
    stats = FeatureRowStats(
        rows=len(frame),
        labeled=labeled,
        unlabeled=len(frame) - labeled,
        dropped_missing=dropped,
        affected_raw_rows=affected_raw,
    )
    logger.info(
        "Feature rows: %d (labeled=%d, unlabeled=%d); dropped %d grid rows with missing "
        "aggregated features; %d raw rows affected by missing values",
        stats.rows, stats.labeled, stats.unlabeled, stats.dropped_missing,
        stats.affected_raw_rows,
    )
    return frame, stats


def compute_latest_aggregates(readings: pd.DataFrame) -> pd.Series:
    """Aggregated features at the newest timestamp, from the trailing 37 grid rows only."""
    grid = to_grid(readings)
    if grid.empty:
        raise ValueError("No readings to compute aggregates from")
    aggregated = compute_aggregated_features(grid.tail(_LATEST_SLICE))
    return aggregated.iloc[-1]


def build_inference_row(stored_row: pd.Series, latest_reading: pd.Series) -> pd.DataFrame:
    """One-row frame in FEATURE_COLUMNS order.

    Aggregated values come from the stored row, realtime values from the live Reading.
    The label column is never included.
    """
    numeric = [c for c in config.RAW_COLUMNS if c not in ("station_abbr", _TS)]
    live = pd.DataFrame([latest_reading.reindex(numeric)]).astype("float64")
    realtime = compute_realtime_columns(live).iloc[0]
    values = {c: stored_row[c] for c in config.AGGREGATED_FEATURES}
    values.update({c: realtime[c] for c in config.REALTIME_FEATURES})
    return pd.DataFrame([values], columns=config.FEATURE_COLUMNS).astype("float64")


def validate_feature_columns(columns: Sequence[str]) -> None:
    """Raise FeatureSchemaError unless `columns` equals FEATURE_COLUMNS (names and order)."""
    cols = list(columns)
    expected = config.FEATURE_COLUMNS
    if cols == expected:
        return
    missing = [c for c in expected if c not in cols]
    unexpected = [c for c in cols if c not in expected]
    problems = []
    if missing:
        problems.append(f"missing: {missing}")
    if unexpected:
        problems.append(f"unexpected: {unexpected}")
    if not missing and not unexpected:
        problems.append(f"misordered: got {cols}, expected {expected}")
    raise FeatureSchemaError("Feature columns differ from FEATURE_COLUMNS; " + "; ".join(problems))
