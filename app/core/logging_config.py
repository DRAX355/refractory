"""
app/core/logging_config.py
===========================
Structured logging setup using loguru.
Call `setup_logging()` once at startup.
"""
import sys
from loguru import logger
from app.core.config import settings


def setup_logging() -> None:
    """Configure loguru for the application."""
    logger.remove()  # Remove default handler

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stdout,
        format=log_format,
        level=settings.log_level.upper(),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    logger.add(
        "logs/refractoryai_{time:YYYY-MM-DD}.log",
        format=log_format,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    logger.info(
        f"Logging initialised — level={settings.log_level}, env={settings.app_env}"
    )
