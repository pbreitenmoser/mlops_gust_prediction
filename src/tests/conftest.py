"""Shared test isolation: no network sockets, no Hopsworks environment variables."""

from __future__ import annotations

import os
import socket

import pytest

HOPSWORKS_ENV_VARS = ("HOPSWORKS_API_KEY", "HOPSWORKS_PROJECT_NAME")


class NetworkAccessBlocked(RuntimeError):
    """Raised when a test tries to open a network connection."""


def _blocked(*_args, **_kwargs):
    raise NetworkAccessBlocked("Network access is blocked in tests")


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch):
    """Block sockets and clear Hopsworks env vars; fail (never skip) if isolation is unmet."""
    for var in HOPSWORKS_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)

    leftovers = [v for v in HOPSWORKS_ENV_VARS if v in os.environ]
    if leftovers:
        pytest.fail(f"Isolation unmet: env vars still set: {leftovers}")
    if socket.socket.connect is not _blocked or socket.create_connection is not _blocked:
        pytest.fail("Isolation unmet: socket blocking is not active")
    yield
