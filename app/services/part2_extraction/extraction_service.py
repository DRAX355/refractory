"""
app/services/part2_extraction/extraction_service.py
====================================================
Part 2 — Orchestrates the full Drawing Intelligence pipeline.

Source priority per format:
  DXF  → Step 0: Direct DXF entity parse (exact CAD geometry)
  PDF  → Step 0: PDF vector path extraction (exact drawn paths)
         Step 2b: PDF embedded text (OCR backup)
         Step 2c: Spatially-positioned dimension labels
  ALL  → Step 1: Grounding DINO structural element detection
         Step 2: Tesseract OCR (for scanned/rasterised drawings)
         Step 3: OpenCV geometry analysis (contour + zone detection)

The assembler is called with ALL available sources and uses the best one.
NO hardcoded fallbacks — if geometry cannot be extracted, the profile is
empty and the UI shows an "insufficient data" state.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from loguru import logger

from app.core.config import settings
from app.models.vessel_geometry import ExtractionResponse, VesselGeometry, DrawingFormat
from app.services.part1_ingestion.ingestion_service import IngestionResult
from app.services.part2_extraction.dino_detector import detect_drawing_elements
from app.services.part2_extraction.ocr_extractor import run_ocr, OCRResult, _classify_text
from app.services.part2_extraction.opencv_processor import process_geometry
from app.services.part2_extraction.geometry_assembler import assemble_vessel_geometry

from app.utils.unit_parser import parse_dimension, parse_capacity_tons


async def extract_vessel_geometry(
    ingestion_result: IngestionResult,
) -> ExtractionResponse:
    """
    Run the full Part 2 extraction pipeline on an ingested drawing.
    Returns ExtractionResponse with the VesselGeometry and timing info.
    """
    start_time = time.perf_counter()
    job_id     = ingestion_result.job_id
    warnings: list[str] = list(ingestion_result.warnings)

    # Geometry sources — populated below depending on file format
    dxf_profile      = None   # DXFProfile from direct DXF entity parse
    pdf_path_profile = None   # PDFPathProfile from PDF vector paths
    positioned_dims: list = []

    logger.info(
        f"[Part 2] Starting extraction — job_id={job_id}, "
        f"format={ingestion_result.format}, pages={ingestion_result.page_count}"
    )

    primary_image = ingestion_result.images[0]

    # ── Step 1: Grounding DINO ─────────────────────────────────────────────────
    logger.info("[Part 2] Step 1 — Grounding DINO detection…")
    try:
        dino_detections = detect_drawing_elements(primary_image)
    except Exception as exc:
        logger.warning(f"[Part 2] DINO detection failed: {exc}")
        warnings.append(f"DINO detection skipped: {exc}")
        dino_detections = []

    # ── Step 2: DocTR OCR ──────────────────────────────────────────────────
    logger.info("[Part 2] Step 2 — DocTR OCR extraction…")
    try:
        ocr_result = run_ocr(primary_image, dino_regions=dino_detections)
    except Exception as exc:
        logger.warning(f"[Part 2] OCR failed: {exc}")
        warnings.append(f"OCR extraction skipped: {exc}")
        ocr_result = OCRResult()

    # ── Step 0a: DXF direct geometry (highest priority for DXF files) ─────────
    if ingestion_result.format == DrawingFormat.DXF:
        logger.info("[Part 2] Step 0a — Direct DXF entity extraction…")
        
        # 1. Exact Recon Pipeline (Generates exact 3D Mesh)
        try:
            logger.info("[Part 2] Running exact Recon pipeline...")
            from app.services.part2_extraction.recon.dxf_reader import read_dxf_scene
            from app.services.part2_extraction.recon.reconstruct import reconstruct
            from app.services.part2_extraction.recon.mesh_builder import build_mesh
            
            scene = read_dxf_scene(ingestion_result.file_path)
            rec = reconstruct(scene)
            if rec.ok:
                out_dir = settings.outputs_dir / job_id
                out_dir.mkdir(parents=True, exist_ok=True)
                mesh_path = out_dir / "mesh.glb"
                build_mesh(rec, filepath=str(mesh_path))
                warnings.append("Recon pipeline: Exact 3D mesh generated (mesh.glb).")
            else:
                logger.warning("[Part 2] Exact Recon pipeline failed to build solid.")
        except Exception as exc:
            logger.warning(f"[Part 2] Exact Recon pipeline failed: {exc}")

        # 2. Parametric DXF Profile (for VesselGeometry contract)
        try:
            from app.services.part2_extraction.dxf_geometry_extractor import extract_dxf_profile
            dxf_profile = extract_dxf_profile(ingestion_result.file_path, text_polygons=ocr_result.text_polygons)
            if dxf_profile.success:
                logger.info(
                    f"[Part 2] DXF profile: {len(dxf_profile.points)} pts, "
                    f"h={dxf_profile.total_height_mm:.0f}mm"
                )
                warnings.append(
                    f"DXF direct extraction: {len(dxf_profile.points)} profile points"
                )
            else:
                logger.info("[Part 2] DXF direct extraction found no vessel geometry")
        except Exception as exc:
            logger.warning(f"[Part 2] DXF direct extraction failed: {exc}")

    # ── Step 0b: PDF vector path extraction ───────────────────────────────────
    if ingestion_result.format == DrawingFormat.PDF:
        logger.info("[Part 2] Step 0b — PDF vector path extraction…")
        
        # 2. Parametric PDF Profile (for VesselGeometry contract)
        try:
            from app.utils.pdf_path_extractor import extract_pdf_path_profile
            pdf_path_profile = extract_pdf_path_profile(ingestion_result.file_path, text_polygons=ocr_result.text_polygons)
            if pdf_path_profile.success:
                logger.info(
                    f"[Part 2] PDF path profile: {len(pdf_path_profile.points)} pts, "
                    f"h={pdf_path_profile.total_height_mm:.0f}mm"
                )
                warnings.append(
                    f"PDF vector paths: {len(pdf_path_profile.points)} profile points"
                )
            else:
                logger.info("[Part 2] PDF path extraction found no vessel geometry")
        except Exception as exc:
            logger.warning(f"[Part 2] PDF path extraction failed: {exc}")



    # ── Step 3: OpenCV geometry ────────────────────────────────────────────────
    logger.info("[Part 2] Step 3 — OpenCV geometry analysis…")
    try:
        # Provide scale hint from OCR outer diameter if available
        known_dim_mm = None
        if ocr_result.parsed_dimensions:
            dims_mm = [v for v, u in ocr_result.parsed_dimensions if u == "mm"]
            if dims_mm:
                known_dim_mm = max(dims_mm)

        opencv_result = process_geometry(primary_image, known_dimension_mm=known_dim_mm)
    except Exception as exc:
        logger.warning(f"[Part 2] OpenCV failed: {exc}")
        warnings.append(f"OpenCV skipped: {exc}")
        from app.services.part2_extraction.opencv_processor import OpenCVGeometryResult
        opencv_result = OpenCVGeometryResult()

    # ── Assemble geometry from all sources ─────────────────────────────────────
    vessel_geometry = assemble_vessel_geometry(
        ocr_result=ocr_result,
        opencv_result=opencv_result,
        dino_detections=dino_detections,
        source_file=ingestion_result.filename,
        source_format=ingestion_result.format,
        dxf_profile=dxf_profile,
        pdf_path_profile=pdf_path_profile,
        positioned_dims=positioned_dims,
    )
    
    if pdf_path_profile and pdf_path_profile.success and pdf_path_profile.points:
        vessel_geometry.extra_metadata["pdf_paths"] = pdf_path_profile.points

    _save_output(vessel_geometry, job_id)

    elapsed = time.perf_counter() - start_time
    logger.info(
        f"[Part 2] Extraction complete — job_id={job_id}, "
        f"elapsed={elapsed:.2f}s, "
        f"confidence={vessel_geometry.extraction_confidence:.2f}, "
        f"segments={len(vessel_geometry.shell.profile.segments)}, "
        f"method={vessel_geometry.shell.profile.extraction_method}"
    )

    n_segs  = len(vessel_geometry.shell.profile.segments)
    method  = vessel_geometry.shell.profile.extraction_method
    conf    = vessel_geometry.extraction_confidence

    if method == "none" or n_segs == 0:
        msg = (
            "Extraction complete but no vessel geometry could be read from this drawing. "
            "Ensure the file contains readable vector geometry or clear dimension annotations."
        )
    elif conf >= 0.7:
        msg = f"Extraction successful — {n_segs} segments extracted via {method}."
    else:
        msg = f"Extraction complete with partial data — {n_segs} segments via {method}. Manual review recommended."

    return ExtractionResponse(
        job_id=job_id,
        vessel_geometry=vessel_geometry,
        processing_time_seconds=round(elapsed, 3),
        warnings=warnings,
        message=msg,
    )


def _save_output(geometry: VesselGeometry, job_id: str) -> None:
    """Persist the VesselGeometry JSON and Excel to the outputs directory."""
    out_dir = settings.outputs_dir / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / "vessel_geometry.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(geometry.model_dump(mode="json"), f, indent=2, default=str)
    logger.info(f"[Part 2] Output saved: {out_path}")

    # Generate Excel Output
    try:
        import pandas as pd
        excel_path = out_dir / "vessel_geometry.xlsx"
        
        segments_data = []
        for s in geometry.shell.profile.segments:
            seg_dict = s.model_dump()
            seg_dict["segment_type"] = seg_dict.get("segment_type", "")
            segments_data.append(seg_dict)
            
        segments_df = pd.DataFrame(segments_data)
        
        nozzles_data = []
        for n in geometry.nozzles:
            n_dict = n.model_dump()
            if n_dict.get("bounding_box"):
                n_dict["bbox_x_min"] = n_dict["bounding_box"]["x_min"]
                n_dict["bbox_y_min"] = n_dict["bounding_box"]["y_min"]
                n_dict["bbox_x_max"] = n_dict["bounding_box"]["x_max"]
                n_dict["bbox_y_max"] = n_dict["bounding_box"]["y_max"]
                del n_dict["bounding_box"]
            nozzles_data.append(n_dict)
            
        nozzles_df = pd.DataFrame(nozzles_data)
        
        meta_data = {
            "source_file": [geometry.source_file],
            "source_format": [geometry.source_format],
            "vessel_type": [geometry.vessel_type],
            "drawing_number": [geometry.drawing_number],
            "revision": [geometry.revision],
            "outer_diameter_mm": [geometry.shell.outer_diameter_mm.value if geometry.shell.outer_diameter_mm else None],
            "outer_diameter_raw_text": [geometry.shell.outer_diameter_mm.raw_text if geometry.shell.outer_diameter_mm else None],
            "inner_diameter_mm": [geometry.shell.inner_diameter_mm.value if geometry.shell.inner_diameter_mm else None],
            "inner_diameter_raw_text": [geometry.shell.inner_diameter_mm.raw_text if geometry.shell.inner_diameter_mm else None],
            "total_length_mm": [geometry.shell.total_length_mm.value if geometry.shell.total_length_mm else None],
            "total_length_raw_text": [geometry.shell.total_length_mm.raw_text if geometry.shell.total_length_mm else None],
            "profile_total_height_mm": [geometry.shell.profile.total_height_mm],
            "profile_max_outer_radius_mm": [geometry.shell.profile.max_outer_radius_mm],
            "extraction_method": [geometry.shell.profile.extraction_method]
        }
        
        # Flatten segments directly into metadata for immediate visibility
        for i, s in enumerate(geometry.shell.profile.segments):
            meta_data[f"segment_{i+1}_label"] = [s.label]
            meta_data[f"segment_{i+1}_height_mm"] = [s.height_mm]
            meta_data[f"segment_{i+1}_raw_text"] = [s.height_raw_text]

        meta_df = pd.DataFrame(meta_data)
        
        pdf_paths = geometry.extra_metadata.get("pdf_paths")
        pdf_paths_df = pd.DataFrame(pdf_paths, columns=["y", "r"]) if pdf_paths else pd.DataFrame()
        
        with pd.ExcelWriter(excel_path) as writer:
            meta_df.to_excel(writer, sheet_name="Metadata", index=False)
            if not segments_df.empty:
                segments_df.to_excel(writer, sheet_name="Segments", index=False)
            if not nozzles_df.empty:
                nozzles_df.to_excel(writer, sheet_name="Nozzles", index=False)
            if not pdf_paths_df.empty:
                pdf_paths_df.to_excel(writer, sheet_name="PDF Paths", index=False)
                
        logger.info(f"[Part 2] Excel output saved: {excel_path}")
    except ImportError:
        logger.warning("[Part 2] pandas not installed, skipping Excel export.")
    except Exception as e:
        logger.warning(f"[Part 2] Failed to save Excel output: {e}")

