"""Environment loading and validation. Makes no network call."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

API_KEY_VAR = "HOPSWORKS_API_KEY"
PROJECT_VAR = "HOPSWORKS_PROJECT_NAME"
DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


class ConfigError(Exception):
    """Raised when required configuration is missing. Never contains values."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    project_name: str

    def __repr__(self) -> str:
        return f"Settings(api_key='***', project_name={self.project_name!r})"

    __str__ = __repr__


def load_settings(env_path: Path = DEFAULT_ENV_PATH) -> Settings:
    """Load settings; the process environment wins over the optional .env file."""
    env_path = Path(env_path)
    if env_path.is_file():
        load_dotenv(env_path, override=False)

    missing = []
    values = {}
    for key in (API_KEY_VAR, PROJECT_VAR):
        value = os.environ.get(key)
        if value is None or not value.strip():
            missing.append(key)
        else:
            values[key] = value.strip()
    if missing:
        raise ConfigError("Missing or empty: " + ", ".join(missing))
    return Settings(api_key=values[API_KEY_VAR], project_name=values[PROJECT_VAR])
