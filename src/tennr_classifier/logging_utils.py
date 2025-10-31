"""Logging helpers for the Tennr classifier pipeline."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from config import Settings


def configure_logging(settings: Settings, *, force: bool = False) -> None:
    """Configure root logging handlers and levels using provided settings."""
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    handlers = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=force,
    )
    if settings.debug:
        logging.getLogger().setLevel(logging.DEBUG)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger instance, initializing basic config if needed."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.StreamHandler(sys.stdout)],
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )
    return logging.getLogger(name or "tennr_classifier")
