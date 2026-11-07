"""CLI entry: uv run python -m src.inference_pipeline.main"""

from __future__ import annotations

import sys

from src.common.logging_utils import register_secret, run_pipeline, setup_logging
from src.common.settings import load_settings
from src.inference_pipeline.pipeline import run_pipeline_main


def _run() -> None:
    settings = load_settings()
    register_secret(settings.api_key)
    run_pipeline_main(settings)


def main() -> int:
    setup_logging()
    return run_pipeline("inference_pipeline", _run)


if __name__ == "__main__":
    sys.exit(main())
