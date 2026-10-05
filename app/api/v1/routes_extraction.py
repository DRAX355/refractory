"""
app/api/v1/routes_extraction.py
=================================
Part 2 — REST API routes for drawing intelligence extraction.

Endpoints:
  POST /api/v1/extraction/process
    - Accepts: file upload (runs Part 1 + Part 2 in one call)
    - Returns: ExtractionResponse with full VesselGeometry JSON

  POST /api/v1/extraction/process/{job_id}
    - Accepts: job_id from a previous Part 1 ingestion
    - Returns: ExtractionResponse (loads saved images from disk)

  GET /api/v1/extraction/{job_id}/result
    - Returns the saved VesselGeometry JSON for a completed job
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.models.vessel_geometry import ExtractionResponse, VesselGeometry
from app.services.part1_ingestion.drawing_validator import ValidationError
from app.services.part1_ingestion.ingestion_service import ingest_drawing
from app.services.part2_extraction.extraction_service import extract_vessel_geometry

router = APIRouter(
    prefix="/extraction",
    tags=["Part 2 — Drawing Intelligence (Extraction)"],
)


@router.post(
    "/process",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and fully process a drawing (Part 1 + Part 2)",
    description=(
        "One-shot endpoint: upload a drawing file and receive the full "
        "structured VesselGeometry extraction result in a single call. "
        "Internally runs Part 1 ingestion then Part 2 extraction."
    ),
)
async def process_drawing(
    file: UploadFile = File(
        ...,
        description="Engineering drawing file (PDF, DXF, DWG, PNG, JPG, TIFF)",
    ),
) -> ExtractionResponse:
    """
    Upload a drawing and run the full extraction pipeline (Part 1 + Part 2).
    Returns a complete VesselGeometry model.
    """
    logger.info(f"[API] Full process request — file={file.filename}")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # ── Part 1: Ingest ─────────────────────────────────────────────────────────
    try:
        ingestion_result = await ingest_drawing(
            file_bytes=file_bytes,
            filename=file.filename or "drawing",
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        error_msg = str(exc)
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY if "ODA" in error_msg else status.HTTP_500_INTERNAL_SERVER_ERROR
        if status_code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.exception("Ingestion error")
        else:
            logger.warning(f"Ingestion error: {exc}")
        
        raise HTTPException(
            status_code=status_code,
            detail=f"Drawing ingestion failed: {exc}",
        )

    # ── Part 2: Extract ────────────────────────────────────────────────────────
    try:
        extraction_response = await extract_vessel_geometry(ingestion_result)
    except Exception as exc:
        logger.exception("Extraction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drawing extraction failed: {exc}",
        )

    return extraction_response


@router.post(
    "/process/{job_id}",
    response_model=ExtractionResponse,
    status_code=status.HTTP_200_OK,
    summary="Run Part 2 extraction on an already-ingested drawing",
    description=(
        "Run Part 2 extraction on a drawing that was previously uploaded "
        "via POST /api/v1/ingestion/upload. Provide the job_id from that response."
    ),
)
async def process_by_job_id(job_id: str) -> ExtractionResponse:
    """Re-run Part 2 extraction on a previously ingested drawing."""
    job_dir = settings.outputs_dir / job_id
    if not job_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found. Please upload the drawing first.",
        )

    # Find the original drawing file
    drawing_files = [
        f for f in job_dir.iterdir()
        if f.is_file() and f.suffix.lower() in
        {".pdf", ".dxf", ".dwg", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    ]
    if not drawing_files:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No drawing file found for job '{job_id}'.",
        )

    drawing_path = drawing_files[0]
    with open(drawing_path, "rb") as f:
        file_bytes = f.read()

    ingestion_result = await ingest_drawing(
        file_bytes=file_bytes,
        filename=drawing_path.name,
    )
    # Preserve original job_id
    ingestion_result.job_id = job_id

    try:
        return await extract_vessel_geometry(ingestion_result)
    except Exception as exc:
        logger.exception("Re-extraction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {exc}",
        )


@router.get(
    "/{job_id}/result",
    response_model=VesselGeometry,
    summary="Get the extraction result for a completed job",
)
async def get_extraction_result(job_id: str) -> VesselGeometry:
    """Return the saved VesselGeometry JSON for a completed extraction job."""
    output_file = settings.outputs_dir / job_id / "vessel_geometry.json"

    if not output_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No extraction result found for job '{job_id}'. "
                "Run extraction first via POST /api/v1/extraction/process."
            ),
        )

    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return VesselGeometry(**data)
