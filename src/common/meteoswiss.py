"""Download, parse and merge MeteoSwiss 10-minute CSV files (station BER).

Source: MeteoSwiss. All data is kept in memory; no partial files are written.
"""

from __future__ import annotations

import io
import logging
import time
from datetime import datetime, timezone
from typing import Iterable

import pandas as pd
import requests

from src.common import config

logger = logging.getLogger(__name__)

_NUMERIC_COLUMNS = [
    c for c in config.RAW_COLUMNS if c not in ("station_abbr", "reference_timestamp")
]
_TIMESTAMP_FORMAT = "%d.%m.%Y %H:%M"
_RETRY_BACKOFF_S = 2.0


class DataError(Exception):
    """Raised for unusable MeteoSwiss data (missing column, empty file, bad timestamp, ...)."""


class DownloadError(Exception):
    """Raised when a file cannot be downloaded after all attempts."""


def download_csv(url: str, timeout_s: int) -> bytes:
    """Download a file into memory, with up to DOWNLOAD_ATTEMPTS attempts.

    Logs URL and reason of each failure. Raises DownloadError when all attempts fail.
    """
    last_reason = "unknown"
    for attempt in range(1, config.DOWNLOAD_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=timeout_s)
            if response.status_code != 200:
                raise requests.HTTPError(f"HTTP status {response.status_code}")
            return response.content
        except requests.RequestException as exc:
            last_reason = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Download failed (attempt %d/%d): url=%s reason=%s",
                attempt, config.DOWNLOAD_ATTEMPTS, url, last_reason,
            )
            if attempt < config.DOWNLOAD_ATTEMPTS:
                time.sleep(_RETRY_BACKOFF_S)
    raise DownloadError(
        f"Download failed after {config.DOWNLOAD_ATTEMPTS} attempts: url={url} reason={last_reason}"
    )


def parse_csv(raw: bytes, source: str) -> pd.DataFrame:
    """Parse one MeteoSwiss CSV into a frame with the configured RAW_COLUMNS only.

    `reference_timestamp` becomes tz-aware UTC (config.SOURCE_TZ). Row numbers in
    errors are 1-based data rows (the header is not counted).
    """
    if not raw or not raw.strip():
        raise DataError(f"{source}: file is empty")
    try:
        df = pd.read_csv(
            io.BytesIO(raw), sep=";", dtype=str, keep_default_na=False,
            encoding="utf-8-sig",
        )
    except pd.errors.EmptyDataError as exc:
        raise DataError(f"{source}: file is empty") from exc
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataError(f"{source}: cannot parse CSV: {exc}") from exc

    for column in config.RAW_COLUMNS:
        if column not in df.columns:
            raise DataError(f"{source}: missing column {column}")
    df = df[config.RAW_COLUMNS].copy()
    if df.empty:
        raise DataError(f"{source}: file contains no data rows")

    ts = pd.to_datetime(
        df["reference_timestamp"].str.strip(), format=_TIMESTAMP_FORMAT, errors="coerce"
    )
    bad = ts.isna()
    if bad.any():
        pos = int(bad.to_numpy().argmax())
        raise DataError(
            f"{source}: row {pos + 1}: cannot parse reference_timestamp "
            f"{df['reference_timestamp'].iloc[pos]!r} (expected dd.mm.yyyy HH:MM)"
        )
    df["reference_timestamp"] = ts.dt.tz_localize(config.SOURCE_TZ).dt.tz_convert("UTC")

    for column in _NUMERIC_COLUMNS:
        values = df[column].str.strip().replace("", pd.NA)
        try:
            df[column] = pd.to_numeric(values, errors="raise").astype("float64")
        except (ValueError, TypeError) as exc:
            raise DataError(f"{source}: column {column} has non-numeric values: {exc}") from exc
    df["station_abbr"] = df["station_abbr"].str.strip()
    return df.reset_index(drop=True)


def merge_readings(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate, dedupe on timestamp (last frame wins) and sort chronologically."""
    if not frames:
        raise DataError("No readings to merge")
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset="reference_timestamp", keep="last")
    return merged.sort_values("reference_timestamp").reset_index(drop=True)


def check_year_coverage(years: Iterable[int], available_years: Iterable[int]) -> None:
    """Raise DataError naming the uncovered years. Must run before any download."""
    available = set(available_years)
    missing = sorted(set(years) - available)
    if missing:
        raise DataError(
            f"Training years not covered by Historical_Files plus Recent_File: {missing}; "
            f"available: {min(available)}-{max(available)}"
        )


def _fetch(file_name: str, timeout_s: int) -> pd.DataFrame:
    return parse_csv(download_csv(config.BASE_URL + file_name, timeout_s), file_name)


def load_backfill_readings(years: Iterable[int]) -> pd.DataFrame:
    """Historical_Files covering `years` (rows restricted to those years) plus the Recent_File."""
    years = sorted(set(years))
    current_year = datetime.now(timezone.utc).year
    check_year_coverage(years, config.available_years(current_year))

    frames: list[pd.DataFrame] = []
    historical_years = [y for y in years if y < current_year]
    for decade in sorted({y // 10 * 10 for y in historical_years}):
        df = _fetch(config.historical_file_name(decade), config.DOWNLOAD_TIMEOUT_S)
        wanted = set(historical_years)
        frames.append(df[df["reference_timestamp"].dt.year.isin(wanted)])
    frames.append(_fetch(config.RECENT_FILE, config.DOWNLOAD_TIMEOUT_S))
    return merge_readings(*frames)


def load_recent_and_now() -> pd.DataFrame:
    """Recent_File and Now_File merged; the Now_File value wins on overlaps."""
    recent = _fetch(config.RECENT_FILE, config.DOWNLOAD_TIMEOUT_S)
    now = _fetch(config.NOW_FILE, config.NOW_TIMEOUT_S)
    return merge_readings(recent, now)


def load_now() -> pd.DataFrame:
    """Now_File only (30 s timeout)."""
    return _fetch(config.NOW_FILE, config.NOW_TIMEOUT_S)
