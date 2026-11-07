"""Pure selection helpers (no Hopsworks imports, no I/O)."""

from __future__ import annotations

from typing import Iterable, TypeVar

import pandas as pd

from src.common.config import EVENT_TIME, INCREMENTAL_INSERT_DAYS, LABEL_COLUMN

T = TypeVar("T")


def drop_unlabeled(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows with a null label. Returns (labeled rows, number dropped)."""
    labeled = df.dropna(subset=[LABEL_COLUMN])
    return labeled, len(df) - len(labeled)


def select_latest_row(df: pd.DataFrame) -> pd.Series:
    """Return the row with the newest `reference_timestamp`, labeled or not."""
    if df.empty:
        raise ValueError("No stored feature rows available")
    ts = pd.to_datetime(df[EVENT_TIME], utc=True)
    return df.loc[ts.idxmax()]


def select_latest_version(models: Iterable[T]) -> T:
    """Return the model object with the highest integer `version`."""
    models = list(models)
    if not models:
        raise ValueError("model not found")
    return max(models, key=lambda m: int(m.version))


def select_insert_window(df: pd.DataFrame, days: int = INCREMENTAL_INSERT_DAYS) -> pd.DataFrame:
    """Rows with `reference_timestamp >= newest - days`; always contains the newest row."""
    if df.empty:
        return df.copy()
    ts = pd.to_datetime(df[EVENT_TIME], utc=True)
    return df[ts >= ts.max() - pd.Timedelta(days=days)].copy()
