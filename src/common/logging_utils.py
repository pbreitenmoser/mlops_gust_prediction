"""Logging with secret redaction, step timing and the pipeline runner."""

from __future__ import annotations

import contextvars
import logging
import re
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Iterator

REDACTION = "***REDACTED***"
LOGGER_NAME = "gust"

# key=value / key: value / "key": "value" style secrets.
_SECRET_PATTERN = re.compile(
    r"""(?ix)
    (?P<key>["']?(?:api[_-]?key|apikey|password|passwd|secret|token|authorization)["']?)
    (?P<sep>\s*[:=]\s*)
    (?P<quote>["']?)
    (?:bearer\s+)?
    (?P<value>[^\s"',;&]+)
    """
)

_current_pipeline: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_pipeline", default="-"
)
_current_step: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_step", default="-"
)

log = logging.getLogger(LOGGER_NAME)


class RedactingFormatter(logging.Formatter):
    """Formats the full record (including traceback) and redacts secrets."""

    def __init__(self, secrets: Iterable[str] = (), fmt: str | None = None) -> None:
        super().__init__(
            fmt or "%(asctime)s %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        self.converter = time.gmtime
        self._secrets: list[str] = []
        for secret in secrets:
            self.add_secret(secret)

    def add_secret(self, secret: str | None) -> None:
        if secret and secret.strip():
            self._secrets.append(secret)
            # longest first so that overlapping secrets are fully replaced
            self._secrets.sort(key=len, reverse=True)

    def redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, REDACTION)
        return _SECRET_PATTERN.sub(
            lambda m: f"{m.group('key')}{m.group('sep')}{m.group('quote')}{REDACTION}",
            text,
        )

    def format(self, record: logging.LogRecord) -> str:
        return self.redact(super().format(record))


_formatter: RedactingFormatter | None = None


def setup_logging(secrets: Iterable[str] = (), level: int = logging.INFO) -> RedactingFormatter:
    """Install one root handler with a RedactingFormatter and return the formatter."""
    global _formatter
    formatter = RedactingFormatter(secrets)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)
    _formatter = formatter
    return formatter


def register_secret(secret: str | None) -> None:
    """Add a secret learned after setup (e.g. after load_settings)."""
    if _formatter is not None:
        _formatter.add_secret(secret)


class StepInfo:
    """Mutable holder so the body of a step can report its record count."""

    def __init__(self, records: int | None = None) -> None:
        self.records = records


@contextmanager
def step(name: str, records: int | None = None) -> Iterator[StepInfo]:
    """Log step name, outcome, duration in seconds and record count; re-raise on failure."""
    info = StepInfo(records)
    token = _current_step.set(name)
    start = time.monotonic()
    try:
        yield info
    except BaseException:
        _log_step(name, "failure", time.monotonic() - start, info.records)
        raise
    else:
        _log_step(name, "success", time.monotonic() - start, info.records)
        _current_step.reset(token)


def _log_step(name: str, outcome: str, duration: float, records: int | None) -> None:
    msg = f"step={name} outcome={outcome} duration_s={duration:.2f}"
    if records is not None:
        msg += f" records={records}"
    (log.info if outcome == "success" else log.error)(msg)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_pipeline(name: str, fn: Callable[[], Any]) -> int:
    """Run fn, log UTC start/end entries and outcome, and return the exit code."""
    token = _current_pipeline.set(name)
    step_token = _current_step.set("-")
    log.info(f"pipeline={name} event=start time_utc={_utc_now()}")
    exit_code = 1
    try:
        fn()
        exit_code = 0
    except SystemExit as exc:  # treat sys.exit(0/None) as success
        exit_code = 0 if exc.code in (None, 0) else 1
        if exit_code:
            log.error(f"pipeline={name} step={_current_step.get()} error: exit {exc.code!r}")
    except Exception as exc:
        log.error(
            f"pipeline={name} step={_current_step.get()} error: {type(exc).__name__}: {exc}",
            exc_info=True,
        )
    finally:
        outcome = "success" if exit_code == 0 else "failure"
        log.info(f"pipeline={name} event=end time_utc={_utc_now()} outcome={outcome}")
        _current_pipeline.reset(token)
        _current_step.reset(step_token)
    return exit_code


def run_with_timeout(fn: Callable[[], Any], seconds: float) -> Any:
    """Run fn in a daemon worker thread; raise TimeoutError if it exceeds `seconds`."""
    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - propagated to the caller
            result["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"operation did not finish within {seconds} seconds")
    if "error" in result:
        raise result["error"]
    return result.get("value")
