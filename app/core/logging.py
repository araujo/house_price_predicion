"""Logging setup (stub — extend with structured logging as needed)."""

import logging
import sys
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger. Call once at application startup."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format=_DEFAULT_FORMAT,
        stream=sys.stdout,
    )
