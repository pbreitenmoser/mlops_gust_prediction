"""Time split and regression metrics. Pure functions, no I/O."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.common.config import EVENT_TIME, LABEL_COLUMN


def time_split(df: pd.DataFrame, train_frac: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split by `reference_timestamp`; never shuffles.

    n_train = floor(train_frac * n). Raises ValueError if either part is empty.
    """
    ordered = df.sort_values(EVENT_TIME, kind="stable").reset_index(drop=True)
    n = len(ordered)
    n_train = math.floor(train_frac * n)
    if n_train == 0 or n - n_train == 0:
        raise ValueError(f"Cannot split {n} rows into non-empty train and test sets")
    return ordered.iloc[:n_train].copy(), ordered.iloc[n_train:].copy()


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    diff = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(diff)))


def persistence_baseline_rmse(test_df: pd.DataFrame) -> tuple[float, int]:
    """RMSE of predicting the label with the stored `gust_now`.

    Rows with null `gust_now` (or null label) are skipped. Returns (rmse, rows_used).
    """
    valid = test_df.dropna(subset=["gust_now", LABEL_COLUMN])
    if valid.empty:
        raise ValueError("No test rows with non-null gust_now for the persistence baseline")
    return rmse(valid[LABEL_COLUMN], valid["gust_now"]), len(valid)
