"""
app/utils/pdf_text_extractor.py
================================
Direct text extraction from vector/CAD PDFs using pdfplumber.

CAD-generated PDFs (from AutoCAD, FreeCAD, etc.) store text as actual
embedded text objects — NOT as rasterised pixels. Tesseract OCR cannot
read these (it reads pixel patterns), but pdfplumber reads the PDF text
layer directly, giving perfect accuracy.

This module is used as the PRIMARY text source for PDF drawings.
Tesseract OCR is used as fallback for scanned/rasterised drawings.

NEW — extract_segment_dimensions():
    Leverages pdfplumber's word-level (x, y) coordinate data to associate
    dimension labels with their spatial position in the drawing.  This lets
    us map "2416" → a vertical span on the drawing, giving us per-segment
    heights that drive the 3D zone render.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

from loguru import logger


def extract_pdf_text(pdf_path: str | Path) -> list[str]:
    """
    Extract all text strings from a vector PDF using pdfplumber.

    Returns:
        List of non-empty text strings found in the PDF.
        Returns empty list if pdfplumber is not installed or no text found.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("[PDF-Text] pdfplumber not installed — falling back to OCR only")
        return []

    path = Path(pdf_path)
    if not path.exists():
        logger.warning(f"[PDF-Text] File not found: {path}")
        return []

    texts: list[str] = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract words with their text content
                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                )
                page_texts = [w["text"].strip() for w in words if w["text"].strip()]
                logger.debug(
                    f"[PDF-Text] Page {page_num}: {len(page_texts)} words extracted"
                )
                texts.extend(page_texts)

                # Also get raw text blocks (catches multi-word annotations)
                raw = page.extract_text(x_tolerance=3, y_tolerance=3)
                if raw:
                    for line in raw.splitlines():
                        line = line.strip()
                        if line and line not in texts:
                            texts.append(line)

    except Exception as exc:
        logger.warning(f"[PDF-Text] pdfplumber extraction failed: {exc}")
        return []

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    logger.info(f"[PDF-Text] Extracted {len(unique)} unique text items from PDF")
    return unique


# ── Spatially-aware segment dimension extraction ──────────────────────────────

class PositionedDimension(NamedTuple):
    """A numeric dimension value together with its page-coordinate position."""
    value: float          # parsed mm value
    unit: str             # "mm" | "cm" | "m" | "inch"
    raw_text: str         # raw OCR/PDF string
    x: float              # page x-coordinate (pt)
    y: float              # page y-coordinate (pt) — measured from top of page
    page_width: float     # page width in pt (for normalising)
    page_height: float    # page height in pt (for normalising)


_RE_DIA      = re.compile(r'[Øø\u00d8\u03a6]?\s*([\d]+(?:[.,]\d+)?)\s*(?:mm)?', re.IGNORECASE)
_RE_NUMBER   = re.compile(r'\b(\d{3,6}(?:\.\d+)?)\b')
_RE_CAPACITY = re.compile(r'(\d+(?:\.\d+)?)\s*[Tt]\b')
_RE_UNIT_DIM = re.compile(
    r'([Øø\u00d8\u03a6])?\s*([\d.,]+)\s*(mm|cm|m|inch|in|")?',
    re.IGNORECASE,
)


def extract_segment_dimensions(pdf_path: str | Path) -> list[PositionedDimension]:
    """
    Extract all numeric dimensions from a vector PDF together with their
    page (x, y) coordinates.

    Each word in the PDF has a known bounding box.  We use the y-coordinate
    (vertical position on page) to infer which axial section of the vessel the
    dimension label belongs to.  Combined with known total length, this lets
    the geometry assembler assign per-segment heights.

    Returns:
        List of PositionedDimension, sorted by y-coordinate (top→bottom of page,
        i.e. vessel top→bottom in a typical GA drawing orientation).
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("[PDF-Segments] pdfplumber not available")
        return []

    path = Path(pdf_path)
    if not path.exists():
        return []

    results: list[PositionedDimension] = []

    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                pw = float(page.width)
                ph = float(page.height)

                words = page.extract_words(
                    x_tolerance=3,
                    y_tolerance=3,
                    keep_blank_chars=False,
                )

                for word in words:
                    text = word["text"].strip()
                    if not text:
                        continue

                    # Word centre coordinates (top-left origin)
                    wx = float(word.get("x0", 0) + word.get("x1", 0)) / 2.0
                    wy = float(word.get("top", word.get("y0", 0)))

                    parsed = _parse_dim_from_text(text)
                    if parsed is None:
                        continue

                    value, unit = parsed
                    results.append(
                        PositionedDimension(
                            value=value,
                            unit=unit,
                            raw_text=text,
                            x=wx,
                            y=wy,
                            page_width=pw,
                            page_height=ph,
                        )
                    )

    except Exception as exc:
        logger.warning(f"[PDF-Segments] Extraction failed: {exc}")
        return []

    # Sort top→bottom (ascending y = nearer top of page)
    results.sort(key=lambda d: d.y)
    logger.info(f"[PDF-Segments] Extracted {len(results)} positioned dimensions")
    return results


def _parse_dim_from_text(text: str) -> tuple[float, str] | None:
    """
    Parse a single PDF word into (value_mm, unit).

    Returns None if the text is not a recognisable dimension.
    """
    text = text.strip().replace(",", "")

    m = _RE_UNIT_DIM.fullmatch(text)
    if m:
        try:
            val = float(m.group(2))
            unit_raw = m.group(3) or "mm"
            unit = _norm_unit(unit_raw)
            if val > 0:
                return _to_mm(val, unit), "mm"
        except (ValueError, TypeError):
            pass

    m = _RE_NUMBER.fullmatch(text)
    if m:
        try:
            val = float(m.group(1))
            # Only keep values plausible for engineering dimensions
            if 50 <= val <= 120_000:
                return val, "mm"
        except ValueError:
            pass

    return None


def _norm_unit(raw: str) -> str:
    mapping = {
        "mm": "mm", "cm": "cm", "m": "m",
        '"': "inch", "inch": "inch", "in": "inch",
    }
    return mapping.get(raw.lower().strip(), "mm")


def _to_mm(value: float, unit: str) -> float:
    if unit == "cm":
        return value * 10
    if unit == "m":
        return value * 1000
    if unit == "inch":
        return value * 25.4
    return value


# ── Legacy helper (unchanged) ─────────────────────────────────────────────────

def classify_pdf_texts(texts: list[str]) -> dict[str, list[float]]:
    """
    Classify extracted PDF texts into dimension categories using
    value-range heuristics for BOF vessels.

    Returns a dict with keys: diameters, lengths, capacities, all_numbers
    """
    diameters:  list[float] = []
    lengths:    list[float] = []
    capacities: list[float] = []
    all_nums:   list[float] = []

    for t in texts:
        # Capacity (e.g. "180T", "200 T")
        cm = _RE_CAPACITY.search(t)
        if cm:
            capacities.append(float(cm.group(1)))

        # All numbers in the text
        for nm in _RE_NUMBER.finditer(t):
            val = float(nm.group(1).replace(',', '.'))
            all_nums.append(val)

            # BOF outer diameters: 5000–15000mm
            if 5000 <= val <= 15000:
                if re.search(r'[Øø]', t[:nm.start()+5]):
                    diameters.append(val)
                else:
                    lengths.append(val)
            # Segment lengths / partial dimensions: 100–5000mm
            elif 100 <= val < 5000:
                lengths.append(val)

    return {
        "diameters":  sorted(set(diameters), reverse=True),
        "lengths":    sorted(set(lengths), reverse=True),
        "capacities": capacities,
        "all_numbers": sorted(set(all_nums), reverse=True),
    }
