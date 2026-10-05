"""
app/services/part1_ingestion/ingestion_service.py
==================================================
Part 1 — Orchestrates the full drawing ingestion pipeline:

  1. Save uploaded bytes to disk
  2. Validate the file
  3. Load / convert to numpy image(s)
  4. Return an IngestionResult for downstream use (Part 2)
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from app.core.config import settings
from app.models.vessel_geometry import DrawingFormat, IngestionResponse
from app.services.part1_ingestion.file_handler import load_drawing, detect_format
from app.services.part1_ingestion.drawing_validator import (
    validate_upload,
    validate_images,
    ValidationError,
)


@dataclass
class IngestionResult:
    """
    Internal result object produced by Part 1.
    Passed directly into Part 2 (extraction pipeline).
    """
    job_id: str
    file_path: Path
    filename: str
    format: DrawingFormat
    images: list[np.ndarray]          # one image per drawing page
    file_size_bytes: int
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.images)

    def to_response(self) -> IngestionResponse:
        return IngestionResponse(
            job_id=self.job_id,
            filename=self.filename,
            format=self.format,
            file_size_bytes=self.file_size_bytes,
            page_count=self.page_count,
            message=(
                "Drawing ingested successfully."
                if not self.warnings
                else f"Ingested with {len(self.warnings)} warning(s)."
            ),
        )


async def ingest_drawing(
    file_bytes: bytes,
    filename: str,
) -> IngestionResult:
    """
    Entry point for Part 1 — accepts raw bytes of an uploaded drawing file.

    Args:
        file_bytes: Raw file content from the HTTP upload.
        filename:   Original filename (used to detect format).

    Returns:
        IngestionResult with loaded images and metadata.

    Raises:
        ValidationError: if the file fails validation.
        RuntimeError:    if loading or conversion fails.
    """
    job_id = str(uuid.uuid4())
    logger.info(f"[Part 1] Starting ingestion — job_id={job_id}, file={filename}")

    # ── Step 1: Save to disk ──────────────────────────────────────────────────
    save_dir = settings.outputs_dir / job_id
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / filename

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    file_size = len(file_bytes)
    logger.debug(f"[Part 1] Saved to {file_path} ({file_size} bytes)")

    # ── Step 2: Validate ──────────────────────────────────────────────────────
    validate_upload(file_path, file_size)

    # ── Step 3: Load / Convert ────────────────────────────────────────────────
    images, fmt = load_drawing(file_path)

    # ── Step 4: Image quality warnings ───────────────────────────────────────
    warnings = validate_images(images)
    if warnings:
        for w in warnings:
            logger.warning(f"[Part 1] {w}")

    result = IngestionResult(
        job_id=job_id,
        file_path=file_path,
        filename=filename,
        format=fmt,
        images=images,
        file_size_bytes=file_size,
        warnings=warnings,
    )

    logger.info(
        f"[Part 1] Ingestion complete — job_id={job_id}, "
        f"format={fmt}, pages={result.page_count}"
    )
    return result
