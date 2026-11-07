"""Feature pipeline: build Feature_Group rows from MeteoSwiss data and upsert them.

Modules (not names) are imported so tests can monkeypatch module attributes.
Source of the data: MeteoSwiss.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.common import config, features, hopsworks_io, logging_utils, meteoswiss, selection, settings

logger = logging.getLogger(__name__)

MODES = ("backfill", "incremental")


def _log_summary(rows: pd.DataFrame) -> None:
    ts = pd.to_datetime(rows[config.EVENT_TIME], utc=True)
    labeled = int(rows[config.LABEL_COLUMN].notna().sum())
    logger.info(
        "Inserted rows=%d min_event_time=%s max_event_time=%s labeled=%d unlabeled=%d",
        len(rows), ts.min().isoformat(), ts.max().isoformat(), labeled, len(rows) - labeled,
    )


def _insert(project_settings: settings.Settings, rows: pd.DataFrame) -> None:
    """Zero rows: leave the Feature_Group untouched and fail (Requirement 6.9)."""
    if rows.empty:
        raise RuntimeError("No feature rows to insert; Feature_Group left unchanged")
    with logging_utils.step("hopsworks_login"):
        project = hopsworks_io.login(project_settings)
    with logging_utils.step("insert_rows", records=len(rows)):
        fg = hopsworks_io.get_feature_group(project)
        hopsworks_io.insert_rows(fg, rows)
    _log_summary(rows)


def run_backfill(project_settings: settings.Settings) -> pd.DataFrame:
    """Historical files for TRAINING_YEARS plus Recent_File; insert all rows."""
    with logging_utils.step("download_backfill") as info:
        readings = meteoswiss.load_backfill_readings(config.TRAINING_YEARS)  # checks year coverage first
        info.records = len(readings)
    with logging_utils.step("build_features") as info:
        rows, _stats = features.build_feature_rows(readings)
        info.records = len(rows)
    _insert(project_settings, rows)
    return rows


def run_incremental(project_settings: settings.Settings) -> pd.DataFrame:
    """Recent_File plus Now_File merged; insert the trailing INCREMENTAL_INSERT_DAYS."""
    with logging_utils.step("download_incremental") as info:
        readings = meteoswiss.load_recent_and_now()
        info.records = len(readings)
    with logging_utils.step("build_features") as info:
        rows, _stats = features.build_feature_rows(readings)
        rows = selection.select_insert_window(rows, config.INCREMENTAL_INSERT_DAYS)
        info.records = len(rows)
    _insert(project_settings, rows)
    return rows


def run(mode: str) -> pd.DataFrame:
    """Load settings first (no Hopsworks call before it), then run the selected mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r}")
    with logging_utils.step("load_settings"):
        project_settings = settings.load_settings()
        logging_utils.register_secret(project_settings.api_key)
    if mode == "backfill":
        return run_backfill(project_settings)
    return run_incremental(project_settings)
