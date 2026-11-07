"""Tests for settings, logging and repository hygiene."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from hypothesis import given, settings as hsettings, strategies as st

from src.common import logging_utils
from src.common.logging_utils import REDACTION, RedactingFormatter, run_pipeline
from src.common.settings import API_KEY_VAR, PROJECT_VAR, ConfigError, load_settings

REPO_ROOT = Path(__file__).resolve().parents[2]

_values = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789_", min_size=12, max_size=30)
_blank = st.sampled_from(["", " ", "   ", "\t"])


# Feature: wind-gust-forecasting-mlops, Property 11: Environment validation
@hsettings(max_examples=100, deadline=None)
@given(
    key_state=st.one_of(st.none(), _blank, _values),
    project_state=st.one_of(st.none(), _blank, _values),
)
def test_property_11_environment_validation(key_state, project_state, tmp_path_factory):
    """**Validates: Requirements 1.9, 2.1, 2.2**"""
    env = {}
    if key_state is not None:
        env[API_KEY_VAR] = key_state
    if project_state is not None:
        env[PROJECT_VAR] = project_state
    no_file = tmp_path_factory.getbasetemp() / "does-not-exist.env"

    expected_missing = [
        name
        for name, val in ((API_KEY_VAR, key_state), (PROJECT_VAR, project_state))
        if val is None or not val.strip()
    ]
    with mock.patch.dict(os.environ, env, clear=False):
        for var in (API_KEY_VAR, PROJECT_VAR):
            if var not in env:
                os.environ.pop(var, None)
        if expected_missing:
            with pytest.raises(ConfigError) as exc:
                load_settings(no_file)
            message = str(exc.value)
            for name in expected_missing:
                assert name in message
            for name in {API_KEY_VAR, PROJECT_VAR} - set(expected_missing):
                assert name not in message
            for val in (key_state, project_state):
                if val and val.strip():
                    assert val.strip() not in message
        else:
            s = load_settings(no_file)
            assert s.api_key == key_state.strip()
            assert s.project_name == project_state.strip()
            assert "api_key='***'" in repr(s)
            assert "api_key='***'" in str(s)


def test_process_environment_wins_over_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(f"{API_KEY_VAR}=file_key_value\n{PROJECT_VAR}=file_project\n")
    monkeypatch.setenv(API_KEY_VAR, "process_key_value")
    monkeypatch.delenv(PROJECT_VAR, raising=False)
    try:
        s = load_settings(env_file)
    finally:
        os.environ.pop(PROJECT_VAR, None)
    assert s.api_key == "process_key_value"
    assert s.project_name == "file_project"


# Feature: wind-gust-forecasting-mlops, Property 12: Secrets are redacted from log output
@hsettings(max_examples=100, deadline=None)
@given(
    secret=_values,
    prefix=st.text(alphabet="ABCDEFG .:", max_size=10),
    suffix=st.text(alphabet="ABCDEFG .:", max_size=10),
    style=st.sampled_from(["plain", "kv", "json", "exception"]),
)
def test_property_12_secrets_redacted(secret, prefix, suffix, style):
    """**Validates: Requirements 2.3, 12.6**"""
    formatter = RedactingFormatter([secret])
    exc_info = None
    if style == "kv":
        msg = f"{prefix} api_key={secret} {suffix}"
    elif style == "json":
        msg = f'{prefix} {{"token": "{secret}"}} {suffix}'
    else:
        msg = f"{prefix}{secret}{suffix}"
    if style == "exception":
        try:
            raise ValueError(f"failed with {secret}")
        except ValueError:
            import sys

            exc_info = sys.exc_info()
    record = logging.LogRecord("gust", logging.ERROR, __file__, 1, msg, None, exc_info)
    out = formatter.format(record)
    assert secret not in out
    assert REDACTION in out


def _run_and_capture(caplog, fn, name="test_pipe"):
    caplog.clear()
    with caplog.at_level(logging.INFO, logger=logging_utils.LOGGER_NAME):
        code = run_pipeline(name, fn)
    return code, [r.getMessage() for r in caplog.records]


def test_run_pipeline_success_logs_start_and_end(caplog):
    code, msgs = _run_and_capture(caplog, lambda: None)
    assert code == 0
    assert "pipeline=test_pipe event=start" in msgs[0] and "time_utc=" in msgs[0]
    assert "pipeline=test_pipe event=end" in msgs[-1]
    assert "outcome=success" in msgs[-1] and msgs[-1].count("Z") >= 1


def test_run_pipeline_failure_returns_nonzero_and_logs_error(caplog):
    def boom():
        raise RuntimeError("kaputt")

    code, msgs = _run_and_capture(caplog, boom)
    assert code != 0
    assert any("RuntimeError" in m and "kaputt" in m for m in msgs)
    assert "event=start" in msgs[0]
    assert "event=end" in msgs[-1] and "outcome=failure" in msgs[-1]


def test_run_pipeline_system_exit_codes(caplog):
    def exit0():
        raise SystemExit(0)

    def exit2():
        raise SystemExit(2)

    assert _run_and_capture(caplog, exit0)[0] == 0
    assert _run_and_capture(caplog, exit2)[0] != 0


def test_env_example_has_exactly_the_two_keys():
    lines = [
        ln.strip()
        for ln in (REPO_ROOT / "src" / ".env.example").read_text().splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    keys = [ln.split("=", 1)[0] for ln in lines]
    assert sorted(keys) == sorted([API_KEY_VAR, PROJECT_VAR])
    values = dict(ln.split("=", 1) for ln in lines)
    assert values[API_KEY_VAR] == ""


def test_gitignore_lists_src_env():
    entries = [ln.strip() for ln in (REPO_ROOT / ".gitignore").read_text().splitlines()]
    assert "src/.env" in entries
    result = subprocess.run(
        ["git", "check-ignore", "-q", "src/.env"], cwd=REPO_ROOT, check=False
    )
    assert result.returncode == 0


def test_isolation_fixture_blocks_sockets_and_env():
    assert API_KEY_VAR not in os.environ and PROJECT_VAR not in os.environ
    with pytest.raises(RuntimeError):
        socket.create_connection(("example.com", 80), timeout=1)
    s = socket.socket()
    try:
        with pytest.raises(RuntimeError):
            s.connect(("127.0.0.1", 9))
    finally:
        s.close()
