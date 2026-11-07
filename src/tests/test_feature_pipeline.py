"""Unit tests for the feature pipeline CLI and orchestration (Requirements 6.5, 6.7, 6.9)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.common import config, hopsworks_io, meteoswiss
from src.feature_pipeline import main as fp_main
from src.feature_pipeline import pipeline
from src.tests.strategies import make_row

_STEP = pd.Timedelta(minutes=config.INTERVAL_MIN)
_NUMERIC = [c for c in config.RAW_COLUMNS if c not in ("station_abbr", config.EVENT_TIME)]


def _readings(n: int, end: str = "2026-03-10 12:00") -> pd.DataFrame:
    """n contiguous synthetic readings on the 10-minute UTC grid ending at `end`."""
    last = pd.Timestamp(end, tz="UTC")
    rows = []
    for i in range(n):
        ts = last - (n - 1 - i) * _STEP
        rows.append(make_row(ts, {c: float(i % 20) + 1.0 for c in _NUMERIC}))
    return pd.DataFrame(rows, columns=config.RAW_COLUMNS)


@pytest.fixture
def hops(monkeypatch):
    """Fake Hopsworks I/O recording calls; fake settings env values."""
    monkeypatch.setenv("HOPSWORKS_API_KEY", "fake-key")
    monkeypatch.setenv("HOPSWORKS_PROJECT_NAME", "fake-project")
    calls = {"login": 0, "inserted": []}

    def login(_settings):
        calls["login"] += 1
        return object()

    monkeypatch.setattr(hopsworks_io, "login", login)
    monkeypatch.setattr(hopsworks_io, "get_feature_group", lambda _project: object())
    monkeypatch.setattr(
        hopsworks_io, "insert_rows", lambda _fg, rows: calls["inserted"].append(rows.copy())
    )
    return calls


@pytest.mark.parametrize("argv", [[], ["bogus"], ["Backfill"]])
def test_invalid_or_missing_mode_exits_2_naming_modes(argv, hops, monkeypatch, capsys):
    def fail(*_a, **_k):
        raise AssertionError("pipeline must not run")

    monkeypatch.setattr(pipeline, "run", fail)
    with pytest.raises(SystemExit) as exc:
        fp_main.main(argv)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "backfill" in err and "incremental" in err
    assert hops["login"] == 0
    assert hops["inserted"] == []


def test_zero_computed_rows_exits_nonzero_without_insert(hops, monkeypatch):
    # Fewer readings than the 6h window needs: no row can be built.
    monkeypatch.setattr(meteoswiss, "load_recent_and_now", lambda: _readings(10))
    assert fp_main.main(["incremental"]) != 0
    assert hops["login"] == 0
    assert hops["inserted"] == []


def test_incremental_keeps_newest_now_file_reading(hops, monkeypatch):
    readings = _readings(400)  # > 2 days, so the window selection actually trims
    newest = readings[config.EVENT_TIME].max()
    monkeypatch.setattr(meteoswiss, "load_recent_and_now", lambda: readings)

    assert fp_main.main(["incremental"]) == 0
    assert hops["login"] == 1
    assert len(hops["inserted"]) == 1
    inserted = hops["inserted"][0]
    ts = pd.to_datetime(inserted[config.EVENT_TIME], utc=True)
    assert ts.max() == newest
    assert ts.min() >= newest - pd.Timedelta(days=config.INCREMENTAL_INSERT_DAYS)
    # The newest reading has no future label yet but is still inserted.
    newest_row = inserted[ts == newest].iloc[0]
    assert pd.isna(newest_row[config.LABEL_COLUMN])
