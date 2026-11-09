"""Feature pipeline: build Feature_Group rows from MeteoSwiss data and upsert them.

Modules (not names) are imported so tests can monkeypatch module attributes.
Source of the data: MeteoSwiss.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src.common import (
    config,
    features,
    hopsworks_io,
    logging_utils,
    meteoswiss,
    report,
    selection,
    settings,
)

logger = logging.getLogger(__name__)

MODES = ("backfill", "incremental")

# Features shown in the quality summary (a few key ones as a sanity check).
QUALITY_FEATURES = ["gust_max_1h", "gust_max_6h", "wind_mean_3h", "pressure_change_3h", "gust_now"]
# Raw source columns checked for missing values.
RAW_SOURCE_COLUMNS = [c for c in config.RAW_COLUMNS if c not in ("station_abbr", config.EVENT_TIME)]


def _report_header(mode: str) -> None:
    report.banner(f"FEATURE PIPELINE ({mode})")
    report.kv("Mode", mode)
    report.kv("Station", config.STATION)
    report.kv("Started (UTC)", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    report.kv("Feature group", f"{config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION}")
    if mode == "backfill":
        years = list(config.TRAINING_YEARS)
        report.kv("Data scope", f"historical {years[0]}-{years[-1]} + recent file")
    else:
        report.kv("Data scope", f"recent + now file, upsert last {config.INCREMENTAL_INSERT_DAYS} days")


def _report_cleaning(
    readings: pd.DataFrame, stats: features.FeatureRowStats, rows: pd.DataFrame
) -> None:
    """Cleaning and feature step: what went in, what was dropped, what came out."""
    report.section("Cleaning and feature step")
    ts = pd.to_datetime(readings[config.EVENT_TIME], utc=True)
    report.kv("Raw readings", f"{len(readings):,}")
    report.kv("Raw time range (UTC)", f"{report.fmt_ts(ts.min())} to {report.fmt_ts(ts.max())}")
    report.kv("Raw rows w/ missing value", f"{stats.affected_raw_rows:,} (wind, gust or pressure)")
    report.kv("Dropped (incomplete window)", f"{stats.dropped_missing:,} grid rows")
    report.kv("Feature rows built", f"{stats.rows:,}")
    report.kv("  labeled", f"{stats.labeled:,}")
    report.kv("  unlabeled (label null)", f"{stats.unlabeled:,}")
    if len(rows) != stats.rows:
        report.kv("Rows after insert window", f"{len(rows):,}")
    missing = {c: int(readings[c].isna().sum()) for c in RAW_SOURCE_COLUMNS if c in readings}
    missing = {c: n for c, n in missing.items() if n}
    if missing:
        report.line("Missing values per raw column: " + ", ".join(f"{c}={n:,}" for c, n in missing.items()))
    else:
        report.line("Missing values per raw column: none")


def _report_quality(rows: pd.DataFrame) -> None:
    """Quality summary: min / mean / max of a few key features and of the label."""
    report.section("Quality summary")
    if rows.empty:
        report.line("No rows.")
        return
    cols = [c for c in QUALITY_FEATURES + [config.LABEL_COLUMN] if c in rows]
    report.table(
        ["column (km/h or unit)", "min", "mean", "max", "null"],
        [
            [
                c,
                report.fmt_num(rows[c].min()),
                report.fmt_num(rows[c].mean()),
                report.fmt_num(rows[c].max()),
                int(rows[c].isna().sum()),
            ]
            for c in cols
        ],
    )


def _report_write(rows: pd.DataFrame, duration_s: float) -> None:
    """Write step: rows upserted into the feature group."""
    ts = pd.to_datetime(rows[config.EVENT_TIME], utc=True)
    labeled = int(rows[config.LABEL_COLUMN].notna().sum())
    report.section("Write step")
    report.kv("Operation", "upsert (existing rows are replaced, not duplicated)")
    report.kv("Rows upserted", f"{len(rows):,}")
    report.kv("  labeled", f"{labeled:,}")
    report.kv("  unlabeled", f"{len(rows) - labeled:,}")
    report.kv("Time range (UTC)", f"{report.fmt_ts(ts.min())} to {report.fmt_ts(ts.max())}")
    report.kv("Target", f"{config.FEATURE_GROUP_NAME} v{config.FEATURE_GROUP_VERSION}")
    report.kv("Write duration", report.fmt_duration(duration_s))


def _report_footer(rows: pd.DataFrame, started: float) -> None:
    report.banner("FEATURE PIPELINE: SUCCESS")
    report.kv("Rows upserted", f"{len(rows):,}")
    report.kv("Total runtime", report.fmt_duration(time.monotonic() - started))


def _insert(project_settings: settings.Settings, rows: pd.DataFrame) -> None:
    """Zero rows: leave the Feature_Group untouched and fail (Requirement 6.9)."""
    if rows.empty:
        raise RuntimeError("No feature rows to insert; Feature_Group left unchanged")
    with logging_utils.step("hopsworks_login"):
        project = hopsworks_io.login(project_settings)
    start = time.monotonic()
    with logging_utils.step("insert_rows", records=len(rows)):
        fg = hopsworks_io.get_feature_group(project)
        hopsworks_io.insert_rows(fg, rows)
    _report_write(rows, time.monotonic() - start)


def run_backfill(project_settings: settings.Settings) -> pd.DataFrame:
    """Historical files for TRAINING_YEARS plus Recent_File; insert all rows."""
    with logging_utils.step("download_backfill") as info:
        readings = meteoswiss.load_backfill_readings(config.TRAINING_YEARS)  # checks year coverage first
        info.records = len(readings)
    with logging_utils.step("build_features") as info:
        rows, stats = features.build_feature_rows(readings)
        info.records = len(rows)
    _report_cleaning(readings, stats, rows)
    _report_quality(rows)
    _insert(project_settings, rows)
    return rows


def run_incremental(project_settings: settings.Settings) -> pd.DataFrame:
    """Recent_File plus Now_File merged; insert the trailing INCREMENTAL_INSERT_DAYS."""
    with logging_utils.step("download_incremental") as info:
        readings = meteoswiss.load_recent_and_now()
        info.records = len(readings)
    with logging_utils.step("build_features") as info:
        rows, stats = features.build_feature_rows(readings)
        rows = selection.select_insert_window(rows, config.INCREMENTAL_INSERT_DAYS)
        info.records = len(rows)
    _report_cleaning(readings, stats, rows)
    _report_quality(rows)
    _insert(project_settings, rows)
    return rows


def run(mode: str) -> pd.DataFrame:
    """Load settings first (no Hopsworks call before it), then run the selected mode."""
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode!r}")
    started = time.monotonic()
    _report_header(mode)
    with logging_utils.step("load_settings"):
        project_settings = settings.load_settings()
        logging_utils.register_secret(project_settings.api_key)
    rows = run_backfill(project_settings) if mode == "backfill" else run_incremental(project_settings)
    _report_footer(rows, started)
    return rows
