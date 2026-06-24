"""
Virtual Chemistry Lab API - Logging Configuration
Loguru setup with structured logging and rotation.
"""

import sys
from pathlib import Path
from loguru import logger
from app.config import settings


def setup_logging():
    """Configure Loguru logging with rotation and structured output."""
    # Remove default handler
    logger.remove()

    # Console handler with colored output
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=settings.LOG_FORMAT,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler with rotation
    log_dir = Path("/tmp/chem_lab/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "chem_lab_{time:YYYY-MM-DD}.log",
        rotation="00:00",  # Rotate at midnight
        retention="30 days",
        compression="zip",
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
    )

    # Error file handler
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        compression="zip",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
    )

    logger.info("Logging configured successfully")


# Setup logging on import
setup_logging()
