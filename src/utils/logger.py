from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str = "risk_engine") -> logging.Logger:
    """Create a configured logger for the project."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(handler)
    return logger
