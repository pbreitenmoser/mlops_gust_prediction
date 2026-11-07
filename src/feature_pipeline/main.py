"""CLI: uv run python -m src.feature_pipeline.main backfill|incremental"""

from __future__ import annotations

import argparse
import sys

from src.common import logging_utils
from src.feature_pipeline import pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="feature_pipeline", description="Wind gust feature pipeline")
    parser.add_argument("mode", choices=list(pipeline.MODES), help="backfill or incremental")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)  # argparse exits with code 2 on a missing/invalid mode
    logging_utils.setup_logging()
    return logging_utils.run_pipeline("feature_pipeline", lambda: pipeline.run(args.mode))


if __name__ == "__main__":
    sys.exit(main())
