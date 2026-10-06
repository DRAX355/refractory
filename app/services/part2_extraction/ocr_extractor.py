"""
app/services/part2_extraction/ocr_extractor.py
===============================================
Part 2 — Tesseract OCR: Dimension & Text Extraction.

Extracts:
  - Vessel dimensions (diameters, lengths) with unit parsing
  - Operating conditions (vessel capacity, e.g. "180T BOF")
  - Drawing metadata (drawing number, revision, title)
  - Nozzle labels and annotations
  - All raw text strings (for downstream NLP / RAG in Part 3)

Pipeline per image:
  1. Preprocess image (grayscale → denoise → threshold)
  2. Run Tesseract with engineering drawing config
  3. Parse dimension values using unit_parser
  4. Classify text into categories
"""
from __future__ import annotations

import re

import numpy as np
import pytesseract
from loguru import logger

from app.core.config import settings
from app.models.vessel_geometry import (
    Dimension,
    DimensionUnit,
    OperatingConditions,
    VesselType,
)
from app.utils.image_utils import preprocess_for_ocr, scale_image, crop_region
from app.utils.unit_parser import parse_dimension, parse_capacity_tons
from app.models.vessel_geometry import BoundingBox


# ── Tesseract configuration ────────────────────────────────────────────────────

def _configure_tesseract() -> None:
    """Set Tesseract binary path from settings."""
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


# PSM 6  = uniform block of text
# PSM 11 = sparse text (scattered dimension labels on drawings)
# PSM 7  = single text line (good for individual dimension annotations)
# PSM 13 = raw line — no OSD, treat as single line
TESSERACT_CONFIG_SPARSE  = r'--psm 11 -l eng --oem 3'
TESSERACT_CONFIG_BLOCK   = r'--psm 6  -l eng --oem 3'
# Numeric-focused config: whitelist digits, dot, comma, slash, degree, Ø symbol
TESSERACT_CONFIG_NUMERIC = (
    r'--psm 11 -l eng --oem 3 '
    r'-c tessedit_char_whitelist=0123456789.,/\xd8\xf8\u00d8\u00f8\u03a6%-: '
)


# ── Regex patterns for drawing-specific text ────────────────────────────────────

_RE_DRAWING_NUMBER = re.compile(
    r"\b([A-Z]{2,}-[A-Z0-9]+-\d{3,}(?:-[A-Z0-9]+)?)\b"
)
_RE_REVISION = re.compile(r"\bRev(?:ision)?\s*([A-Z0-9]+)\b", re.IGNORECASE)
_RE_VESSEL_TYPE = re.compile(r"\b(BOF|EAF|LADLE|RH\s*DEGASSER)\b", re.IGNORECASE)

NOZZLE_KEYWORDS = {
    "tap hole", "tap pad", "taphole",
    "charge pad", "charge door",
    "oxygen lance", "o2 lance",
    "running pad", "rest area",
    "trunnion",
}


# ── Public API ─────────────────────────────────────────────────────────────────

class OCRResult:
    """Container for all OCR-extracted data from one drawing image."""

    def __init__(self) -> None:
        self.raw_texts: list[str] = []
        self.dimension_strings: list[str] = []
        self.parsed_dimensions: list[tuple[float, str]] = []  # (value, unit)
        self.drawing_number: str = ""
        self.revision: str = ""
        self.vessel_type_hint: str = ""
        self.operating_conditions: OperatingConditions = OperatingConditions()
        self.nozzle_labels: list[str] = []
        
        # New for DocTR-based spatial masking
        from shapely.geometry import Polygon
        self.text_polygons: list[Polygon] = []


def run_ocr(
    image_bgr: np.ndarray,
    dino_regions: list[BoundingBox] | None = None,
) -> OCRResult:
    """
    Run full OCR pipeline on a drawing image using DocTR.
    """
    result = OCRResult()
    
    logger.debug("[OCR] Running DocTR on full image...")
    from app.services.part2_extraction.doctr_extractor import run_doctr_on_image
    
    # Downscale image if too large (DocTR is slow on 9000x9000)
    h, w = image_bgr.shape[:2]
    import cv2
    img_for_doctr = image_bgr
    scale = 1.0
    if max(h, w) > 4000:
        scale = 4000 / max(h, w)
        img_for_doctr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        
    doctr_result = run_doctr_on_image(img_for_doctr)
    
    # 1. Add words from standard orientation
    from shapely.affinity import scale as shapely_scale
    for word in doctr_result.words:
        poly = word.polygon
        if scale != 1.0:
            poly = shapely_scale(poly, xfact=1/scale, yfact=1/scale, origin=(0,0))
            
        result.text_polygons.append(poly)
        result.raw_texts.append(word.value)
        
    # 2. Add words from 90-degree CCW rotated orientation (for vertical text like diameters)
    img_rot = cv2.rotate(img_for_doctr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    doctr_rot_result = run_doctr_on_image(img_rot)
    
    h_rot, w_rot = img_rot.shape[:2]
    # Original (scaled) dimensions
    orig_w, orig_h = img_for_doctr.shape[1], img_for_doctr.shape[0]
    
    for word in doctr_rot_result.words:
        # Map polygon back to unrotated space
        # Transformation: orig_x = orig_w - y_rot, orig_y = x_rot
        mapped_coords = []
        for (x_rot, y_rot) in word.polygon.exterior.coords:
            orig_x = orig_w - y_rot
            orig_y = x_rot
            mapped_coords.append((orig_x, orig_y))
            
        from shapely.geometry import Polygon
        poly = Polygon(mapped_coords)
        if scale != 1.0:
            poly = shapely_scale(poly, xfact=1/scale, yfact=1/scale, origin=(0,0))
            
        result.text_polygons.append(poly)
        result.raw_texts.append(word.value)
        
    logger.debug(f"[OCR] Full image — {len(result.raw_texts)} unique text blocks extracted via DocTR (including CCW rotated)")

    # ── Region-specific OCR (DINO crops) ─────────────────────────────────────
    # Not needed with DocTR since it natively detects all dense text globally.
    if dino_regions:
        logger.debug(f"[OCR] Skipping regional DINO OCR, DocTR handles full image.")

    # ── Parse extracted text ──────────────────────────────────────────────────
    _parse_all_texts(result)

    logger.info(
        f"[OCR] Complete — raw_lines={len(result.raw_texts)}, "
        f"dimensions={len(result.parsed_dimensions)}, "
        f"vessel_type={result.vessel_type_hint or 'unknown'}"
    )
    return result


def _classify_text(line: str, result: "OCRResult") -> None:
    """Classify a single text line into OCRResult fields (drawing num, revision, vessel type, etc.)."""
    if not result.drawing_number:
        m = _RE_DRAWING_NUMBER.search(line)
        if m: result.drawing_number = m.group(1)
    if not result.revision:
        m = _RE_REVISION.search(line)
        if m: result.revision = m.group(0)
    if not result.vessel_type_hint:
        m = _RE_VESSEL_TYPE.search(line)
        if m: result.vessel_type_hint = m.group(0).upper().replace(" ", "_")
    line_lower = line.lower()
    for kw in NOZZLE_KEYWORDS:
        if kw in line_lower and line not in result.nozzle_labels:
            result.nozzle_labels.append(line); break


def _parse_all_texts(result: OCRResult) -> None:
    """Classify and parse all raw text lines."""
    for line in result.raw_texts:

        # Drawing number
        if not result.drawing_number:
            m = _RE_DRAWING_NUMBER.search(line)
            if m:
                result.drawing_number = m.group(1)

        # Revision
        if not result.revision:
            m = _RE_REVISION.search(line)
            if m:
                result.revision = m.group(0)

        # Vessel type
        if not result.vessel_type_hint:
            m = _RE_VESSEL_TYPE.search(line)
            if m:
                result.vessel_type_hint = m.group(0).upper().replace(" ", "_")

        # Capacity (e.g. "180T BOF")
        capacity = parse_capacity_tons(line)
        if capacity is not None and result.operating_conditions.vessel_capacity_tons is None:
            result.operating_conditions.vessel_capacity_tons = capacity
            result.operating_conditions.additional_notes.append(line)

        # Dimensions
        parsed = parse_dimension(line)
        if parsed:
            result.dimension_strings.append(line)
            result.parsed_dimensions.append(parsed)
        else:
            # Handle Tesseract merging multiple vertical dimensions into one horizontal line
            parts = [p for p in line.split() if p]
            if len(parts) > 1:
                for part in parts:
                    parsed_part = parse_dimension(part)
                    if parsed_part:
                        result.dimension_strings.append(part)
                        result.parsed_dimensions.append(parsed_part)

        # Nozzle labels
        line_lower = line.lower()
        for kw in NOZZLE_KEYWORDS:
            if kw in line_lower and line not in result.nozzle_labels:
                result.nozzle_labels.append(line)
                break


def make_dimension_model(
    value: float,
    unit_str: str,
    raw_text: str = "",
    confidence: float = 0.8,
) -> Dimension:
    """Helper: build a Dimension model from parsed values."""
    try:
        unit = DimensionUnit(unit_str)
    except ValueError:
        unit = DimensionUnit.UNKNOWN
    return Dimension(value=value, unit=unit, raw_text=raw_text, confidence=confidence)
