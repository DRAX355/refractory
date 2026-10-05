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


def run_ocr(
    image_bgr: np.ndarray,
    dino_regions: list[BoundingBox] | None = None,
) -> OCRResult:
    """
    Run full OCR pipeline on a drawing image.

    Args:
        image_bgr:     OpenCV BGR image.
        dino_regions:  Optional DINO-detected regions to crop and OCR separately.

    Returns:
        OCRResult with all extracted text and parsed values.
    """
    _configure_tesseract()
    result = OCRResult()

    # ── Full-image OCR (dynamically upscaled) ─────────────────────────────────
    h, w = image_bgr.shape[:2]
    # For large CAD PDFs (e.g. 9000x6000), 3x upscaling causes MemoryError.
    # We only upscale if the image is small/low-DPI.
    scale_factor = 1.0 if w > 5000 else (1.5 if w > 3000 else 3.0)
    logger.debug(f"[OCR] Base image {w}x{h}, scaling by {scale_factor}x for OCR")
    
    upscaled = scale_image(image_bgr, scale=scale_factor)
    preprocessed = preprocess_for_ocr(upscaled)

    logger.debug("[OCR] Running Tesseract on full image...")

    # Pass 1 — sparse (catches scattered annotations & dimension strings)
    full_text_sparse = pytesseract.image_to_string(
        preprocessed, config=TESSERACT_CONFIG_SPARSE
    )
    # Pass 2 — numeric whitelist (catches dimension numbers missed by Pass 1)
    # Use the original upscaled (not binarised) for better digit recognition
    upscaled_gray = scale_image(
        __import__('cv2').cvtColor(image_bgr, __import__('cv2').COLOR_BGR2GRAY),
        scale=2.0
    )
    full_text_numeric = pytesseract.image_to_string(
        upscaled_gray, config=TESSERACT_CONFIG_NUMERIC
    )

    # Pass 3 — Rotated numeric whitelist (catches vertical dimension text on the left/right axes)
    rotated = __import__('cv2').rotate(upscaled_gray, __import__('cv2').ROTATE_90_CLOCKWISE)
    full_text_rotated = pytesseract.image_to_string(
        rotated, config=TESSERACT_CONFIG_NUMERIC
    )

    all_raw = full_text_sparse + "\n" + full_text_numeric + "\n" + full_text_rotated
    lines = [ln.strip() for ln in all_raw.splitlines() if ln.strip()]
    # Deduplicate while preserving order
    seen: set[str] = set()
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            result.raw_texts.append(ln)
    logger.debug(f"[OCR] Full image — {len(result.raw_texts)} unique text lines extracted")

    # ── Region-specific OCR (DINO crops) ─────────────────────────────────────
    if dino_regions:
        for bbox in dino_regions:
            crop = crop_region(
                image_bgr,
                (int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)),
            )
            if crop.size == 0:
                continue
            crop_up = scale_image(crop, scale=2.0)
            crop_pre = preprocess_for_ocr(crop_up)
            region_text = pytesseract.image_to_string(
                crop_pre, config=TESSERACT_CONFIG_BLOCK
            )
            region_lines = [ln.strip() for ln in region_text.splitlines() if ln.strip()]
            result.raw_texts.extend(region_lines)
            logger.debug(
                f"[OCR] Region '{bbox.label}' — {len(region_lines)} lines extracted"
            )

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
