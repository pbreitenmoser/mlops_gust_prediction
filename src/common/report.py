"""Small helpers for the human-readable run reports printed to the terminal.

Reports go to stdout, log lines (logging_utils) go to stderr. Everything is
flushed immediately so the report and the log lines stay in order.
"""

from __future__ import annotations

import sys
from typing import Iterable, Sequence

import pandas as pd

WIDTH = 72


def _out(text: str = "") -> None:
    print(text, file=sys.stdout, flush=True)


def banner(title: str) -> None:
    """Heavy double rule with a centered title, used for run header and footer."""
    _out()
    _out("=" * WIDTH)
    _out(title.center(WIDTH).rstrip())
    _out("=" * WIDTH)


def section(title: str) -> None:
    """Light rule with a left-aligned title."""
    _out()
    _out(f"--- {title} ".ljust(WIDTH, "-"))


def kv(key: str, value: object, width: int = 24) -> None:
    _out(f"  {key:<{width}} {value}")


def line(text: str = "") -> None:
    _out(f"  {text}" if text else "")


def table(header: Sequence[str], rows: Iterable[Sequence[object]], first_width: int = 24) -> None:
    """Print a simple left/right aligned table (first column left, others right)."""
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(str(h)), *(len(r[i]) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(header)]
    widths[0] = max(widths[0], first_width)
    head = "  " + str(header[0]).ljust(widths[0]) + "  " + "  ".join(
        str(h).rjust(widths[i + 1]) for i, h in enumerate(header[1:])
    )
    _out(head)
    _out("  " + "-" * (len(head) - 2))
    for r in rows:
        _out("  " + r[0].ljust(widths[0]) + "  " + "  ".join(
            c.rjust(widths[i + 1]) for i, c in enumerate(r[1:])
        ))


def highlight(lines: Sequence[str]) -> None:
    """Print lines inside a bold box so the key result stands out among the logs."""
    inner = max(len(s) for s in lines) + 4
    inner = max(inner, WIDTH - 2)
    _out()
    _out("#" * (inner + 2))
    _out("#" + " " * inner + "#")
    for s in lines:
        _out("#" + s.center(inner) + "#")
    _out("#" + " " * inner + "#")
    _out("#" * (inner + 2))


def fmt_ts(value: object) -> str:
    """UTC timestamp as 'YYYY-MM-DD HH:MM'."""
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%d %H:%M")


def fmt_num(value: object, digits: int = 2) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def fmt_duration(seconds: float) -> str:
    return f"{seconds:.1f} s" if seconds < 120 else f"{int(seconds // 60)} min {seconds % 60:.0f} s"
