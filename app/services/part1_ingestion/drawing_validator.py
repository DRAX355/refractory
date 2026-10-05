"""
app/services/part1_ingestion/drawing_validator.py
==================================================
Part 1 — Validates that an uploaded file is a plausible engineering drawing.

Checks:
  - File size within limits
  - Extension is supported
  - File is not corrupted (quick integrity check)
  - Minimum resolution for raster images
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from app.core.config import settings
from app.models.vessel_geometry import DrawingFormat
from app.services.part1_ingestion.file_handler import EXTENSION_MAP


MIN_IMAGE_WIDTH  = 800   # pixels
MIN_IMAGE_HEIGHT = 600   # pixels


class ValidationError(Exception):
    """Raised when a drawing file fails validation."""


def validate_upload(
    file_path: Path,
    file_size_bytes: int,
) -> None:
    """
    Validate an uploaded drawing file before processing.

    Args:
        file_path:        Path to the saved upload.
        file_size_bytes:  Size of the file in bytes.

    Raises:
        ValidationError: with a human-readable message on any check failure.
    """
    _check_file_size(file_size_bytes)
    _check_extension(file_path)
    _check_file_integrity(file_path)
    logger.info(f"Validation passed for '{file_path.name}'")


def validate_images(images: list[np.ndarray]) -> list[str]:
    """
    Check extracted images for quality issues.

    Returns:
        List of warning strings (empty = no issues).
    """
    warnings: list[str] = []

    if not images:
        warnings.append("No images could be extracted from the drawing.")
        return warnings

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        if w < MIN_IMAGE_WIDTH or h < MIN_IMAGE_HEIGHT:
            warnings.append(
                f"Page {i + 1} resolution ({w}x{h}) is low — OCR may be inaccurate. "
                f"Recommended minimum: {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}."
            )

    return warnings


# ── Private Checks ─────────────────────────────────────────────────────────────

def _check_file_size(size_bytes: int) -> None:
    max_bytes = settings.max_upload_bytes
    if size_bytes > max_bytes:
        raise ValidationError(
            f"File size ({size_bytes / 1024 / 1024:.1f} MB) exceeds limit "
            f"({settings.max_upload_size_mb} MB)."
        )
    if size_bytes == 0:
        raise ValidationError("Uploaded file is empty.")


def _check_extension(path: Path) -> None:
    ext = path.suffix.lower()
    if ext not in EXTENSION_MAP:
        supported = ", ".join(EXTENSION_MAP.keys())
        raise ValidationError(
            f"Unsupported file extension '{ext}'. "
            f"Supported extensions: {supported}"
        )


def _check_file_integrity(path: Path) -> None:
    """Quick integrity check — attempts to read the first few bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(8)
        if len(header) < 4:
            raise ValidationError(f"File '{path.name}' appears to be truncated.")
    except OSError as exc:
        raise ValidationError(
            f"Cannot read file '{path.name}': {exc}"
        ) from exc
