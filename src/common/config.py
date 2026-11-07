"""Single source of truth for all constants shared by the pipelines.

Source of the data: MeteoSwiss.
"""

# --- Station and data source -------------------------------------------------
STATION = "BER"
BASE_URL = "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/ber/"
RECENT_FILE = "ogd-smn_ber_t_recent.csv"
NOW_FILE = "ogd-smn_ber_t_now.csv"
# Time zone of `reference_timestamp` in the source files.
# Verified (task 2.1) in the MeteoSwiss Open Data documentation,
# https://opendatadocs.meteoswiss.ch/general/download ("How date/time, time
# intervals and missing values are represented"): "Date/Time is expressed as
# dd.mm.yyyy HH:MM. All reference time stamps at MeteoSwiss are in UTC!"
# For granularity t the timestamp marks the END of the 10-minute interval
# (ReferenceTS 16:00 = 15:50:01 to 16:00:00).
SOURCE_TZ = "UTC"

# Historical_Files of the 10-minute granularity (granularity t) at station BER,
# verified against the STAC item
# https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items/ber
# on the date of task 2.1. Files are split by decade; only these exist:
#   ogd-smn_ber_t_historical_2000-2009.csv  (first row 01.02.2004 00:00)
#   ogd-smn_ber_t_historical_2010-2019.csv
#   ogd-smn_ber_t_historical_2020-2029.csv  (last row 31.12.2025 23:50)
# "historical" runs until 31 December of last year, "recent" from 1 January of
# this year until yesterday, "now" from yesterday 12 UTC to the latest reading.
HISTORICAL_DECADES = (2000, 2010, 2020)
# First calendar year fully covered by the Historical_Files (2004 starts in February).
FIRST_FULL_HISTORICAL_YEAR = 2005


def historical_file_name(decade_start: int) -> str:
    """Return the Historical_File name for a decade, e.g. 2020 -> ..._2020-2029.csv."""
    if decade_start not in HISTORICAL_DECADES:
        raise ValueError(
            f"No Historical_File for decade {decade_start}; available: {list(HISTORICAL_DECADES)}"
        )
    return f"ogd-smn_ber_t_historical_{decade_start}-{decade_start + 9}.csv"


def available_years(current_year: int) -> list[int]:
    """Return the calendar years that can be loaded.

    Fully covered historical years up to last year, plus the current year via the
    Recent_File.
    """
    return list(range(FIRST_FULL_HISTORICAL_YEAR, current_year + 1))


# --- Time grid and horizon ---------------------------------------------------
TRAINING_YEARS = range(2023, 2026)  # subset of the covered years; 2026 comes from the recent file
INTERVAL_MIN = 10
HORIZON_STEPS = 6  # 60 min
INCREMENTAL_INSERT_DAYS = 2

# --- Hopsworks names ---------------------------------------------------------
FEATURE_GROUP_NAME = "bern_wind_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "bern_wind_gust_fv"
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "bern_wind_gust_model"

# --- Columns -----------------------------------------------------------------
PRIMARY_KEY = "station"
EVENT_TIME = "reference_timestamp"
LABEL_COLUMN = "gust_max_next_1h"

AGGREGATED_FEATURES = [
    "gust_max_1h",
    "gust_max_3h",
    "gust_max_6h",
    "wind_mean_1h",
    "wind_mean_3h",
    "wind_mean_6h",
    "pressure_change_3h",
]
REALTIME_FEATURES = [
    "gust_now",
    "wind_speed_now",
    "wind_dir_sin",
    "wind_dir_cos",
    "temp_now",
    "humidity_now",
    "pressure_now",
    "precip_now",
    "radiation_now",
]
# The single ordered list of model input columns.
FEATURE_COLUMNS = AGGREGATED_FEATURES + REALTIME_FEATURES

RAW_COLUMNS = [
    "station_abbr",
    "reference_timestamp",
    "fu3010z1",
    "fu3010z0",
    "dkl010z0",
    "tre200s0",
    "ure200s0",
    "pp0qffs0",
    "prestas0",
    "gre000z0",
    "rre150z0",
]

# Window lengths in readings (10-minute grid).
WINDOWS = {"1h": 6, "3h": 18, "6h": 36}

# --- Timeouts and limits -----------------------------------------------------
DOWNLOAD_TIMEOUT_S = 60
NOW_TIMEOUT_S = 30
DOWNLOAD_ATTEMPTS = 3
MODEL_UPLOAD_TIMEOUT_S = 120
TRAINING_TIMEOUT_S = 300
STALE_FEATURE_HOURS = 24
INFERENCE_LOOKBACK_DAYS = 7
