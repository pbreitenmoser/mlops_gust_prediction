"""Inference pipeline: forecast the max gust of the next hour at station BER.

Prints a report to stdout on success (inputs, forecast vs context, runtime, a
highlighted forecast box and the one-line result). Logs go to stderr. Every
error raises before anything is printed, so a failing run produces no output.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common import config, hopsworks_io, meteoswiss, report
from src.common.features import (
    build_inference_row,
    compute_latest_aggregates,
    validate_feature_columns,
)
from src.common.logging_utils import step
from src.common.selection import select_latest_row
from src.common.settings import Settings

logger = logging.getLogger(__name__)

MODEL_FILE = "gust_model.joblib"
FEATURE_COLUMNS_FILE = "feature_columns.json"
SKEW_TOLERANCE = 1e-6


class InferenceError(Exception):
    """A run-stopping condition with a clear message (no secrets)."""


def newest_reading(now_df: pd.DataFrame) -> pd.Series:
    """Return the Reading with the maximum timestamp of the Now_File."""
    if now_df is None or now_df.empty:
        raise InferenceError("no live data available (Now_File is empty)")
    ts = pd.to_datetime(now_df[config.EVENT_TIME], utc=True)
    return now_df.loc[ts.idxmax()]


def feature_age_hours(stored_ts: object, reading_ts: object) -> float:
    """Age in hours of the stored row relative to the latest Reading."""
    stored = pd.Timestamp(stored_ts)
    reading = pd.Timestamp(reading_ts)
    stored = stored.tz_localize("UTC") if stored.tzinfo is None else stored.tz_convert("UTC")
    reading = reading.tz_localize("UTC") if reading.tzinfo is None else reading.tz_convert("UTC")
    return (reading - stored).total_seconds() / 3600.0


def is_stale(age_hours: float, limit_hours: float = config.STALE_FEATURE_HOURS) -> bool:
    """True when the stored row is older than the limit (strictly greater)."""
    return age_hours > limit_hours


def check_aggregates_available(stored_row: pd.Series) -> None:
    """Raise if any aggregated value of the stored row is null."""
    missing = [c for c in config.AGGREGATED_FEATURES if c not in stored_row or pd.isna(stored_row[c])]
    if missing:
        raise InferenceError(f"required feature value unavailable: {missing}")


def warn_missing_realtime(row: pd.DataFrame) -> list[str]:
    """Log a warning for realtime columns that are NaN; return their names."""
    missing = [c for c in config.REALTIME_FEATURES if pd.isna(row.iloc[0][c])]
    if missing:
        logger.warning("Realtime values missing in live Reading (passed as NaN): %s", missing)
    return missing


def skew_differences(
    stored_row: pd.Series, recomputed: pd.Series, tol: float = SKEW_TOLERANCE
) -> list[str]:
    """Names of aggregated features where stored and recomputed values differ."""
    diffs = []
    for col in config.AGGREGATED_FEATURES:
        a, b = stored_row[col], recomputed[col]
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b) or abs(float(a) - float(b)) > tol:
            diffs.append(col)
    return diffs


def skew_diagnostic(now_df: pd.DataFrame, stored_row: pd.Series) -> None:
    """Optional check: recompute aggregates from the Now_File when 37 contiguous readings exist.

    Only meaningful when the stored row is the one at the newest Reading.
    """
    needed = max(config.WINDOWS.values()) + 1
    ts = pd.to_datetime(now_df[config.EVENT_TIME], utc=True).sort_values().drop_duplicates()
    tail = ts.tail(needed)
    if len(tail) < needed:
        logger.info("Skew diagnostic skipped: fewer than %d readings in Now_File", needed)
        return
    span = tail.iloc[-1] - tail.iloc[0]
    if span != pd.Timedelta(minutes=config.INTERVAL_MIN * (needed - 1)):
        logger.info("Skew diagnostic skipped: last %d readings are not contiguous", needed)
        return
    stored_ts = pd.Timestamp(stored_row[config.EVENT_TIME])
    stored_ts = stored_ts.tz_localize("UTC") if stored_ts.tzinfo is None else stored_ts.tz_convert("UTC")
    if stored_ts != tail.iloc[-1]:
        logger.info("Skew diagnostic skipped: stored row is not at the newest Reading")
        return
    recomputed = compute_latest_aggregates(now_df)
    diffs = skew_differences(stored_row, recomputed)
    if diffs:
        logger.warning("Training-serving skew: stored and recomputed aggregates differ: %s", diffs)
    else:
        logger.info("Skew diagnostic: stored aggregates match recomputed values")


def format_output_line(station: str, reference_time: object, forecast: float) -> str:
    """e.g. 'BER 2026-11-01T12:30:00+00:00 forecast_max_gust_next_1h=47.3 km/h'."""
    ts = pd.Timestamp(reference_time)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return f"{station} {ts.isoformat()} forecast_max_gust_next_1h={float(forecast):.1f} km/h"


def load_model_artifacts(model_dir: Path) -> tuple[object, list[str]]:
    """Load the joblib model and the ordered feature column list."""
    model_dir = Path(model_dir)
    model_path = _find(model_dir, MODEL_FILE)
    columns_path = _find(model_dir, FEATURE_COLUMNS_FILE)
    model = joblib.load(model_path)
    columns = json.loads(columns_path.read_text(encoding="utf-8"))
    return model, list(columns)


def _find(model_dir: Path, name: str) -> Path:
    direct = model_dir / name
    if direct.is_file():
        return direct
    for candidate in model_dir.rglob(name):
        if candidate.is_file():
            return candidate
    raise InferenceError(f"model artifact {name} not found in {model_dir}")


@dataclass
class InferenceResult:
    """Everything the terminal report needs; nothing is printed while it is built."""

    line: str
    forecast: float
    reference_time: pd.Timestamp
    stored_time: pd.Timestamp
    stored_age_hours: float
    inputs: pd.Series  # the 16 model inputs in FEATURE_COLUMNS order
    model_version: object = None


def _signed(value: float) -> str:
    return f"{value:+.1f}"


def print_report(result: InferenceResult, runtime_s: float) -> None:
    """Print inputs, forecast vs context, runtime and the highlighted forecast."""
    x = result.inputs
    report.banner("INFERENCE PIPELINE")
    report.section("Inputs used")
    report.kv("Station", config.STATION)
    report.kv("Model", f"{config.MODEL_NAME} v{result.model_version}")
    report.kv("Live reading (UTC)", report.fmt_ts(result.reference_time))
    report.kv("Stored row (UTC)", f"{report.fmt_ts(result.stored_time)} ({result.stored_age_hours:.1f} h older)")
    report.line()
    report.line("Aggregated (from Feature Store):")
    for c in config.AGGREGATED_FEATURES:
        report.kv(f"  {c}", report.fmt_num(x[c]))
    report.line("Realtime (from live reading):")
    for c in config.REALTIME_FEATURES:
        report.kv(f"  {c}", report.fmt_num(x[c], 3))

    report.section("Forecast vs context (km/h)")
    f = result.forecast
    report.table(
        ["reference", "value", "forecast diff"],
        [
            [name, report.fmt_num(x[c], 1), _signed(f - x[c]) if pd.notna(x[c]) else "n/a"]
            for name, c in (
                ("gust now", "gust_now"),
                ("max gust last 1 h", "gust_max_1h"),
                ("max gust last 3 h", "gust_max_3h"),
                ("max gust last 6 h", "gust_max_6h"),
            )
        ],
    )
    report.line()
    report.kv("Forecast max gust next 1 h", f"{f:.1f} km/h")
    report.kv("Total runtime", report.fmt_duration(runtime_s))

    report.highlight(
        [
            "GUST FORECAST",
            "",
            f"{config.STATION}  |  reference time {report.fmt_ts(result.reference_time)} UTC",
            f"max gust in the next hour: {f:.1f} km/h",
        ]
    )
    # Machine-readable line, unchanged format.
    print(result.line, flush=True)


def run_inference(settings: Settings) -> InferenceResult:
    """Run all steps and return the result (nothing is printed here)."""
    with step("fetch_now_file"):
        try:
            now_df = meteoswiss.load_now()
        except meteoswiss.DataError as exc:
            if "empty" in str(exc).lower() or "no data rows" in str(exc).lower():
                raise InferenceError(f"no live data available: {exc}") from None
            raise InferenceError(f"Now_File unreachable or unusable: {exc}") from None
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Now_File unreachable: {exc}") from None
        latest = newest_reading(now_df)
        reference_time = pd.Timestamp(latest[config.EVENT_TIME])

    with step("read_stored_features"):
        project = hopsworks_io.login(settings)
        fg = hopsworks_io.get_feature_group(project)
        fv = hopsworks_io.get_feature_view(project, fg)
        since = reference_time - pd.Timedelta(days=config.INFERENCE_LOOKBACK_DAYS)
        rows = hopsworks_io.read_latest_rows(fv, since)
        if rows is None or rows.empty:
            raise InferenceError("no stored features available in the lookback window")
        stored = select_latest_row(rows)
        check_aggregates_available(stored)
        age = feature_age_hours(stored[config.EVENT_TIME], reference_time)
        if is_stale(age):
            logger.warning(
                "Stored features are %.1f hours older than the latest Reading (limit %d h)",
                age,
                config.STALE_FEATURE_HOURS,
            )

    with step("load_model"):
        _model_info, model_dir = hopsworks_io.download_latest_model(project)
        model, saved_columns = load_model_artifacts(model_dir)
        validate_feature_columns(saved_columns)

    with step("predict"):
        try:
            skew_diagnostic(now_df, stored)
        except Exception as exc:  # noqa: BLE001 - diagnostic only
            logger.warning("Skew diagnostic failed: %s", exc)
        row = build_inference_row(stored, latest)
        validate_feature_columns(list(row.columns))
        warn_missing_realtime(row)
        prediction = model.predict(row[saved_columns])
        forecast = float(np.asarray(prediction).ravel()[0])
        if not np.isfinite(forecast):
            raise InferenceError("model returned a non-finite forecast")

    return InferenceResult(
        line=format_output_line(config.STATION, reference_time, forecast),
        forecast=forecast,
        reference_time=reference_time,
        stored_time=pd.Timestamp(stored[config.EVENT_TIME]),
        stored_age_hours=age,
        inputs=row.iloc[0],
        model_version=getattr(_model_info, "version", "?"),
    )


def run_pipeline_main(settings: Settings) -> None:
    """Entry for run_pipeline: print the report only after everything succeeded."""
    started = time.monotonic()
    # Hopsworks client libraries print progress text to stdout; send it to stderr so
    # stdout only carries the report (and stays empty when a run fails).
    with contextlib.redirect_stdout(sys.stderr):
        result = run_inference(settings)
    print_report(result, time.monotonic() - started)
