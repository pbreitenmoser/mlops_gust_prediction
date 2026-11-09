"""Unit tests for the inference pipeline (src/inference_pipeline).

Covers staleness warning, failing runs without output, and output rounding.
Requirements: 10.5, 10.6, 10.7, 10.9. No network; Hopsworks and MeteoSwiss are mocked.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from src.common import config, hopsworks_io, meteoswiss
from src.common.settings import Settings
from src.inference_pipeline import main as inference_main
from src.inference_pipeline import pipeline

_TS = config.EVENT_TIME
_READING_TS = pd.Timestamp("2026-03-10 12:30:00", tz="UTC")
_SETTINGS = Settings(api_key="dummy-key", project_name="dummy-project")


def _now_df() -> pd.DataFrame:
    row = {c: 5.0 for c in config.RAW_COLUMNS if c not in ("station_abbr", _TS)}
    row.update({"station_abbr": config.STATION, _TS: _READING_TS, "dkl010z0": 90.0})
    return pd.DataFrame([row], columns=config.RAW_COLUMNS)


def _stored_rows(age_hours: float) -> pd.DataFrame:
    row = {c: 10.0 for c in config.FEATURE_COLUMNS}
    row.update({config.PRIMARY_KEY: config.STATION, _TS: _READING_TS - pd.Timedelta(hours=age_hours)})
    return pd.DataFrame([row])


def _save_model(model_dir, constant: float) -> None:
    model = DummyRegressor(strategy="constant", constant=constant)
    model.fit(pd.DataFrame([[0.0] * len(config.FEATURE_COLUMNS)], columns=config.FEATURE_COLUMNS), [constant])
    joblib.dump(model, model_dir / pipeline.MODEL_FILE)
    (model_dir / pipeline.FEATURE_COLUMNS_FILE).write_text(json.dumps(config.FEATURE_COLUMNS))


@pytest.fixture
def mock_env(monkeypatch, tmp_path):
    """Mock all external calls; returns a configure(age_hours, constant) function."""

    def configure(age_hours: float = 1.0, constant: float = 30.0, now=None) -> None:
        _save_model(tmp_path, constant)
        monkeypatch.setattr(meteoswiss, "load_now", lambda: _now_df() if now is None else now)
        monkeypatch.setattr(hopsworks_io, "login", lambda settings: SimpleNamespace())
        monkeypatch.setattr(hopsworks_io, "get_feature_group", lambda project: object())
        monkeypatch.setattr(hopsworks_io, "get_feature_view", lambda project, fg: object())
        monkeypatch.setattr(
            hopsworks_io, "read_latest_rows", lambda fv, since: _stored_rows(age_hours)
        )
        monkeypatch.setattr(
            hopsworks_io, "download_latest_model", lambda project: (object(), tmp_path)
        )

    return configure


def _stale_warnings(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "Stored features" in r.getMessage()
    ]


def test_staleness_warning_at_25_hours_still_forecasts(mock_env, caplog):
    mock_env(age_hours=25, constant=30.0)
    with caplog.at_level(logging.WARNING):
        line = pipeline.run_inference(_SETTINGS).line
    assert len(_stale_warnings(caplog)) == 1
    assert "forecast_max_gust_next_1h=30.0 km/h" in line


def test_no_staleness_warning_at_23_hours_and_forecast_produced(mock_env, caplog):
    mock_env(age_hours=23, constant=30.0)
    with caplog.at_level(logging.WARNING):
        line = pipeline.run_inference(_SETTINGS).line
    assert _stale_warnings(caplog) == []
    assert "forecast_max_gust_next_1h=30.0 km/h" in line


@pytest.mark.parametrize(
    "load_now",
    [
        pytest.param(lambda: pd.DataFrame(columns=config.RAW_COLUMNS), id="empty-frame"),
        pytest.param(
            lambda: (_ for _ in ()).throw(meteoswiss.DataError("file is empty")), id="data-error"
        ),
        pytest.param(
            lambda: (_ for _ in ()).throw(meteoswiss.DownloadError("unreachable")),
            id="unreachable",
        ),
    ],
)
def test_empty_or_unreachable_now_file_exits_nonzero_without_output(
    mock_env, monkeypatch, capsys, load_now
):
    mock_env()
    monkeypatch.setattr(meteoswiss, "load_now", load_now)
    monkeypatch.setattr(inference_main, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(inference_main, "load_settings", lambda: _SETTINGS)

    assert inference_main.main() != 0
    assert capsys.readouterr().out == ""


def test_successful_run_prints_report_with_result_line_and_exits_zero(mock_env, monkeypatch, capsys):
    mock_env(constant=30.0)
    monkeypatch.setattr(inference_main, "setup_logging", lambda *a, **k: None)
    monkeypatch.setattr(inference_main, "load_settings", lambda: _SETTINGS)

    assert inference_main.main() == 0
    out = capsys.readouterr().out
    assert "Inputs used" in out
    assert "Forecast vs context" in out
    assert "GUST FORECAST" in out
    assert "Total runtime" in out
    assert out.splitlines()[-1] == (
        f"BER {_READING_TS.isoformat()} forecast_max_gust_next_1h=30.0 km/h"
    )


@pytest.mark.parametrize(
    ("constant", "expected"),
    [(47.26, "47.3"), (47.24, "47.2"), (12.0, "12.0"), (0.04, "0.0")],
)
def test_forecast_is_rounded_to_one_decimal(mock_env, constant, expected):
    mock_env(constant=constant)
    line = pipeline.run_inference(_SETTINGS).line
    assert line.endswith(f"forecast_max_gust_next_1h={expected} km/h")


def test_format_output_line_rounds_to_one_decimal():
    line = pipeline.format_output_line("BER", _READING_TS, 47.349)
    assert line == "BER 2026-03-10T12:30:00+00:00 forecast_max_gust_next_1h=47.3 km/h"
