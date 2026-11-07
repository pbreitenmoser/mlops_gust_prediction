"""Unit tests for the training pipeline (atomic save, too few rows, empty model name).

No network: Hopsworks access is replaced by monkeypatching `hopsworks_io` attributes.

Validates: Requirements 7.7, 8.4, 8.7, 9.5
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from src.common import config
from src.common.config import EVENT_TIME, FEATURE_COLUMNS, LABEL_COLUMN, PRIMARY_KEY
from src.common.settings import Settings
from src.training_pipeline import main as training_main
from src.training_pipeline import pipeline
from src.training_pipeline.pipeline import (
    MODEL_FILE_NAME,
    TrainingError,
    run_training,
    save_model_atomically,
)

SETTINGS = Settings(api_key="dummy-key", project_name="dummy-project")


def make_frame(n_rows: int, n_unlabeled: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.random((n_rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df.insert(0, PRIMARY_KEY, "BER")
    df[EVENT_TIME] = pd.date_range("2024-01-01", periods=n_rows, freq="10min", tz="UTC")
    df[LABEL_COLUMN] = rng.random(n_rows) * 50
    if n_unlabeled:
        df.loc[df.index[-n_unlabeled:], LABEL_COLUMN] = np.nan
    return df


@pytest.fixture
def fake_hopsworks(monkeypatch):
    """Replace hopsworks_io functions; returns a dict recording register_model calls."""
    calls = {"register": 0}
    frame_holder = {"frame": make_frame(100)}

    monkeypatch.setattr(pipeline.hopsworks_io, "login", lambda settings: object())
    monkeypatch.setattr(pipeline.hopsworks_io, "get_feature_group", lambda project: object())
    monkeypatch.setattr(pipeline.hopsworks_io, "get_feature_view", lambda project, fg: object())
    monkeypatch.setattr(
        pipeline.hopsworks_io, "read_training_frame", lambda fv: frame_holder["frame"]
    )

    def _register(*args, **kwargs):
        calls["register"] += 1

    monkeypatch.setattr(pipeline.hopsworks_io, "register_model", _register)
    calls["frame_holder"] = frame_holder
    return calls


def _main_with(monkeypatch, tmp_path) -> int:
    monkeypatch.setattr(training_main, "load_settings", lambda: SETTINGS)
    monkeypatch.setattr(
        training_main, "run_training", lambda settings: run_training(settings, tmp_path)
    )
    return training_main.main()


def test_atomic_save_keeps_previous_model_when_dump_fails(tmp_path, monkeypatch):
    target = tmp_path / MODEL_FILE_NAME
    target.write_bytes(b"previous-model")

    def failing_dump(obj, path, *args, **kwargs):
        with open(path, "wb") as fh:
            fh.write(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(pipeline.joblib, "dump", failing_dump)

    with pytest.raises(OSError):
        save_model_atomically({"m": 1}, target)

    assert target.read_bytes() == b"previous-model"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_save_writes_loadable_model(tmp_path):
    target = save_model_atomically({"m": 1}, tmp_path / MODEL_FILE_NAME)
    assert joblib.load(target) == {"m": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_too_few_labeled_rows_raises_and_writes_no_model(tmp_path, fake_hopsworks):
    # 1 labeled row -> floor(0.8 * 1) == 0 training rows -> cannot split.
    fake_hopsworks["frame_holder"]["frame"] = make_frame(10, n_unlabeled=9)

    with pytest.raises(TrainingError, match="Too few labeled rows"):
        run_training(SETTINGS, tmp_path)

    assert not (tmp_path / MODEL_FILE_NAME).exists()
    assert fake_hopsworks["register"] == 0


def test_too_few_labeled_rows_main_returns_non_zero(tmp_path, monkeypatch, fake_hopsworks):
    fake_hopsworks["frame_holder"]["frame"] = make_frame(10, n_unlabeled=9)

    assert _main_with(monkeypatch, tmp_path) != 0
    assert not (tmp_path / MODEL_FILE_NAME).exists()


def test_empty_model_name_skips_upload_and_fails(tmp_path, monkeypatch, fake_hopsworks):
    monkeypatch.setattr(config, "MODEL_NAME", "  ")

    with pytest.raises(TrainingError, match="model name is empty"):
        run_training(SETTINGS, tmp_path)

    assert fake_hopsworks["register"] == 0


def test_empty_model_name_main_returns_non_zero(tmp_path, monkeypatch, fake_hopsworks):
    monkeypatch.setattr(config, "MODEL_NAME", "")

    assert _main_with(monkeypatch, tmp_path) != 0
    assert fake_hopsworks["register"] == 0


def test_valid_run_registers_model(tmp_path, fake_hopsworks):
    """Sanity check that the fixtures drive a successful run (guards the negative tests)."""
    metrics = run_training(SETTINGS, tmp_path)

    assert set(metrics) == {"rmse", "mae", "baseline_rmse"}
    assert (tmp_path / MODEL_FILE_NAME).exists()
    assert fake_hopsworks["register"] == 1
