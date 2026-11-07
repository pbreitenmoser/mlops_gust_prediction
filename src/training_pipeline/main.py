"""Entry point: uv run python -m src.training_pipeline.main"""

from __future__ import annotations

import sys

from src.common.logging_utils import run_pipeline, setup_logging
from src.common.settings import load_settings
from src.training_pipeline.pipeline import run_training


def main() -> int:
    formatter = setup_logging()

    def _run() -> None:
        settings = load_settings()
        formatter.add_secret(settings.api_key)
        run_training(settings)

    return run_pipeline("training_pipeline", _run)


if __name__ == "__main__":
    sys.exit(main())
