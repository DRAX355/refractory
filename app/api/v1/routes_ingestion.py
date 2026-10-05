"""
app/api/v1/routes_ingestion.py
================================
Part 1 — REST API routes for drawing upload and ingestion.

Endpoint:
  POST /api/v1/ingestion/upload
    - Accepts multipart/form-data with a drawing file
    - Returns: IngestionResponse (job_id, format, page_count, etc.)
    - The job_id is used to chain into Part 2 extraction

  GET /api/v1/ingestion/{job_id}/status
    - Check the status of an ingestion job
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import settings
from app.models.vessel_geometry import DrawingFormat, IngestionResponse
from app.services.part1_ingestion.drawing_validator import ValidationError
from app.services.part1_ingestion.ingestion_service import ingest_drawing

router = APIRouter(
    prefix="/ingestion",
    tags=["Part 1 — Drawing Ingestion"],
)


@router.post(
    "/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an engineering drawing",
    description=(
        "Upload a naked-shell engineering drawing in PDF, DXF, DWG, PNG, JPG, or TIFF format. "
        "The file is validated, loaded, and converted to images ready for Part 2 extraction."
    ),
)
async def upload_drawing(
    file: UploadFile = File(
        ...,
        description="Engineering drawing file (PDF, DXF, DWG, PNG, JPG, TIFF)",
    ),
    vessel_type_hint: str = Form(
        default="BOF",
        description="Expected vessel type — used as a hint for downstream processing",
    ),
) -> IngestionResponse:
    """
    Upload a drawing file and run Part 1 ingestion.

    Returns job_id which can be used to trigger Part 2 extraction via
    `POST /api/v1/extraction/process/{job_id}`.
    """
    logger.info(f"Upload request — filename={file.filename}, content_type={file.content_type}")

    # Read uploaded bytes
    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    try:
        result = await ingest_drawing(
            file_bytes=file_bytes,
            filename=file.filename or "drawing_upload",
        )
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.exception(f"Ingestion failed for '{file.filename}'")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drawing ingestion failed: {exc}",
        )

    return result.to_response()


@router.get(
    "/{job_id}/status",
    summary="Check ingestion job status",
    description="Returns whether the job output exists on disk.",
)
async def get_job_status(job_id: str) -> JSONResponse:
    """Check whether a given job_id has been processed."""
    job_dir = settings.outputs_dir / job_id
    if not job_dir.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )

    geometry_file = job_dir / "vessel_geometry.json"
    return JSONResponse(content={
        "job_id": job_id,
        "ingestion_complete": True,
        "extraction_complete": geometry_file.exists(),
        "output_path": str(geometry_file) if geometry_file.exists() else None,
    })
