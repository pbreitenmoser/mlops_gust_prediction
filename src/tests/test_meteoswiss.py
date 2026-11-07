"""Tests for src/common/meteoswiss.py (data access)."""
from __future__ import annotations

from unittest import mock

import numpy as np
import pandas as pd
import pytest
import requests
from hypothesis import given, settings, strategies as st

from src.common import config, meteoswiss
from src.common.meteoswiss import DataError, DownloadError

PBT = settings(max_examples=100, deadline=None)
NUMERIC = [c for c in config.RAW_COLUMNS if c not in ("station_abbr", "reference_timestamp")]


# ---------- helpers ----------
def _to_csv(timestamps, values) -> bytes:
    """MeteoSwiss-style CSV: ';' delimiter, dd.mm.yyyy HH:MM, empty field = missing."""
    lines = [";".join(config.RAW_COLUMNS)]
    for ts, row in zip(timestamps, values):
        cells = ["BER", ts.strftime("%d.%m.%Y %H:%M")]
        cells += ["" if v is None else repr(v) for v in row]
        lines.append(";".join(cells))
    return ("\n".join(lines) + "\n").encode("utf-8")


_timestamps = st.lists(
    st.datetimes(min_value=pd.Timestamp("2000-01-01").to_pydatetime(),
                 max_value=pd.Timestamp("2035-01-01").to_pydatetime()),
    min_size=1, max_size=15, unique_by=lambda d: d.replace(second=0, microsecond=0),
).map(lambda ds: [pd.Timestamp(d).floor("min") for d in ds])

_value = st.one_of(st.none(), st.floats(min_value=-500, max_value=5000, allow_nan=False,
                                        allow_infinity=False).map(lambda f: round(f, 3)))


@st.composite
def _table(draw):
    ts = draw(_timestamps)
    rows = [draw(st.lists(_value, min_size=len(NUMERIC), max_size=len(NUMERIC))) for _ in ts]
    return ts, rows


# ---------- Property 7 (merge part) ----------
# Feature: wind-gust-forecasting-mlops, Property 7: Merging recent and now data
# **Validates: Requirements 6.5**
@PBT
@given(
    recent_ts=st.sets(st.integers(0, 60), min_size=1, max_size=30),
    now_ts=st.sets(st.integers(0, 60), min_size=1, max_size=30),
    data=st.data(),
)
def test_property_7_merge_recent_and_now(recent_ts, now_ts, data):
    base = pd.Timestamp("2025-01-01", tz="UTC")

    def frame(idx, offset):
        order = data.draw(st.permutations(sorted(idx)))
        return pd.DataFrame({
            "reference_timestamp": [base + pd.Timedelta(minutes=10 * i) for i in order],
            "v": [float(i + offset) for i in order],
        })

    recent, now = frame(recent_ts, 0.0), frame(now_ts, 1000.0)
    merged = meteoswiss.merge_readings(recent, now)

    ts = merged["reference_timestamp"]
    assert ts.is_monotonic_increasing and ts.is_unique
    assert set(ts) == set(recent["reference_timestamp"]) | set(now["reference_timestamp"])
    now_vals = dict(zip(now["reference_timestamp"], now["v"]))
    recent_vals = dict(zip(recent["reference_timestamp"], recent["v"]))
    for t, v in zip(merged["reference_timestamp"], merged["v"]):
        assert v == (now_vals[t] if t in now_vals else recent_vals[t])


# ---------- Property 8 ----------
# Feature: wind-gust-forecasting-mlops, Property 8: CSV parse round trip
# **Validates: Requirements 3.3, 3.5**
@PBT
@given(_table())
def test_property_8_csv_round_trip(table):
    ts, rows = table
    parsed = meteoswiss.parse_csv(_to_csv(ts, rows), "x.csv")

    assert list(parsed.columns) == config.RAW_COLUMNS
    assert str(parsed["reference_timestamp"].dt.tz) == "UTC"
    assert [t.tz_localize(None) for t in parsed["reference_timestamp"]] == list(ts)
    expected = np.array([[np.nan if v is None else v for v in r] for r in rows], dtype=float)
    np.testing.assert_allclose(parsed[NUMERIC].to_numpy(dtype=float), expected, equal_nan=True)


# ---------- Property 9 ----------
# Feature: wind-gust-forecasting-mlops, Property 9: Missing required column is reported
# **Validates: Requirements 3.8, 3.10**
@PBT
@given(table=_table(), column=st.sampled_from(config.RAW_COLUMNS),
       name=st.text("abcdefghij_.", min_size=1, max_size=8))
def test_property_9_missing_column_reported(table, column, name):
    ts, rows = table
    df = pd.read_csv(pd.io.common.BytesIO(_to_csv(ts, rows)), sep=";", dtype=str,
                     keep_default_na=False).drop(columns=[column])
    raw = df.to_csv(sep=";", index=False).encode()
    source = f"{name}.csv"
    with pytest.raises(DataError) as exc:
        meteoswiss.parse_csv(raw, source)
    assert column in str(exc.value) and source in str(exc.value)


@PBT
@given(table=_table(), data=st.data())
def test_property_9_bad_timestamp_reports_row(table, data):
    ts, rows = table
    row_idx = data.draw(st.integers(0, len(ts) - 1))
    lines = _to_csv(ts, rows).decode().splitlines()
    cells = lines[row_idx + 1].split(";")
    cells[1] = data.draw(st.sampled_from(["2025-01-01 10:00", "garbage", "32.13.2025 99:99"]))
    lines[row_idx + 1] = ";".join(cells)
    with pytest.raises(DataError) as exc:
        meteoswiss.parse_csv("\n".join(lines).encode(), "bad.csv")
    msg = str(exc.value)
    assert "bad.csv" in msg and f"row {row_idx + 1}" in msg


# ---------- Property 10 ----------
# Feature: wind-gust-forecasting-mlops, Property 10: Year range coverage check
# **Validates: Requirements 3.9, 3.11**
@PBT
@given(requested=st.sets(st.integers(1990, 2035), max_size=10),
       available=st.sets(st.integers(1990, 2035), min_size=1, max_size=20))
def test_property_10_year_coverage(requested, available):
    uncovered = sorted(requested - available)
    if not uncovered:
        meteoswiss.check_year_coverage(requested, available)
        return
    with pytest.raises(DataError) as exc:
        meteoswiss.check_year_coverage(requested, available)
    msg = str(exc.value)
    listed = msg[msg.index("[") + 1: msg.index("]")]
    assert [int(y) for y in listed.split(",")] == uncovered


# ---------- Unit: download retry ----------
def _ok(content=b"data"):
    r = mock.Mock(spec=requests.Response)
    r.status_code = 200
    r.content = content
    return r


def test_download_retries_three_times_then_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(meteoswiss.requests, "get",
                           side_effect=requests.ConnectionError("boom")) as get, \
         mock.patch.object(meteoswiss.time, "sleep") as sleep:
        with pytest.raises(DownloadError) as exc:
            meteoswiss.download_csv("https://example.test/a.csv", 5)
    assert get.call_count == config.DOWNLOAD_ATTEMPTS == 3
    assert sleep.call_count == 2
    assert "https://example.test/a.csv" in str(exc.value)
    assert list(tmp_path.iterdir()) == []  # no partial output


def test_download_retries_on_http_error_then_succeeds():
    bad = mock.Mock(spec=requests.Response)
    bad.status_code = 503
    with mock.patch.object(meteoswiss.requests, "get", side_effect=[bad, _ok(b"abc")]) as get, \
         mock.patch.object(meteoswiss.time, "sleep"):
        assert meteoswiss.download_csv("https://example.test/a.csv", 5) == b"abc"
    assert get.call_count == 2
