"""Logging setup for ClutterField experiments and library code.

Provides a single `setup_logging` entrypoint that configures a console
handler (and optionally a file handler) with a consistent format, so that
experiment scripts and library modules share the same logging behavior.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_dir: str | Path | None = None) -> logging.Logger:
    """Configure the root ClutterField logger.

    Args:
        level: Logging level name (e.g. "DEBUG", "INFO", "WARNING").
        log_dir: If provided, a `clutterfield.log` file handler is added
            under this directory, in addition to the console handler.

    Returns:
        The configured "clutterfield" logger.
    """
    logger = logging.getLogger("clutterfield")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir_path / "clutterfield.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger namespaced under "clutterfield".

    Args:
        name: Suffix identifying the module or component, typically
            `__name__` of the calling module.

    Returns:
        A logger named "clutterfield.<name>".
    """
    return logging.getLogger(f"clutterfield.{name}")
