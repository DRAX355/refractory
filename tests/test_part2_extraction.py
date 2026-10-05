"""
tests/test_part2_extraction.py
================================
Unit tests for Part 2 — Drawing Intelligence (Extraction).

Tests:
  - Unit parser (dimension/capacity parsing)
  - OpenCV geometry processor
  - OCR extractor (with mocked Tesseract)
  - Geometry assembler
  - Full extraction API endpoint (Part 1 + Part 2)

Run with:
  pytest tests/test_part2_extraction.py -v
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from app.main import app
from app.models.vessel_geometry import (
    BoundingBox,
    VesselGeometry,
    VesselType,
    DrawingFormat,
)
from app.services.part2_extraction.ocr_extractor import OCRResult, _parse_all_texts
from app.services.part2_extraction.opencv_processor import process_geometry
from app.services.part2_extraction.geometry_assembler import assemble_vessel_geometry
from app.utils.unit_parser import parse_dimension, parse_capacity_tons

client = TestClient(app)


# ── Fixtures ────────────────────────────────────────────────────────────────────

def make_drawing_image(width=1200, height=900) -> np.ndarray:
    """Create a synthetic drawing image with circles and lines."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Draw a large circle (vessel cross-section)
    cx, cy = width // 2, height // 2
    import cv2
    cv2.circle(img, (cx, cy), 300, (0, 0, 0), 3)
    cv2.circle(img, (cx, cy), 200, (0, 0, 0), 2)
    # Draw vessel outline (rectangle)
    cv2.rectangle(img, (50, 200), (1150, 700), (0, 0, 0), 3)
    return img


@pytest.fixture
def drawing_image() -> np.ndarray:
    return make_drawing_image()


@pytest.fixture
def sample_png_bytes() -> bytes:
    img = Image.fromarray(make_drawing_image())
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Unit Parser Tests ─────────────────────────────────────────────────────────

class TestUnitParser:
    def test_diameter_with_symbol(self):
        result = parse_dimension("Ø7570")
        assert result is not None
        assert result[0] == pytest.approx(7570.0)
        assert result[1] == "mm"

    def test_dimension_with_mm_suffix(self):
        result = parse_dimension("10000mm")
        assert result is not None
        assert result[0] == pytest.approx(10000.0)

    def test_plain_number_assumed_mm(self):
        result = parse_dimension("97948")
        assert result is not None
        assert result[0] == pytest.approx(97948.0)
        assert result[1] == "mm"

    def test_capacity_180t(self):
        result = parse_capacity_tons("180T BOF")
        assert result == pytest.approx(180.0)

    def test_capacity_200t(self):
        result = parse_capacity_tons("200t Ladle")
        assert result == pytest.approx(200.0)

    def test_unparseable_returns_none(self):
        assert parse_dimension("ABCD XYZ") is None

    def test_inch_dimension(self):
        result = parse_dimension('24"')
        assert result is not None
        assert result[1] == "inch"


# ── OCR Extractor Tests ───────────────────────────────────────────────────────

class TestOCRExtractor:
    def test_parse_bof_drawing_text(self):
        """Test OCR text parsing logic directly without Tesseract."""
        result = OCRResult()
        result.raw_texts = [
            "ZZ-BOF-180-001",
            "Rev B",
            "180T BOF Vessel",
            "Ø10000",
            "Ø7570",
            "97948",
            "Tap Hole",
            "Charge Pad",
        ]
        _parse_all_texts(result)

        assert result.drawing_number == "ZZ-BOF-180-001"
        assert "Rev" in result.revision
        assert result.vessel_type_hint == "BOF"
        assert result.operating_conditions.vessel_capacity_tons == pytest.approx(180.0)
        assert len(result.parsed_dimensions) >= 3
        assert any("tap hole" in label.lower() for label in result.nozzle_labels)
        assert any("charge pad" in label.lower() for label in result.nozzle_labels)


# ── OpenCV Processor Tests ────────────────────────────────────────────────────

class TestOpenCVProcessor:
    def test_detects_circles(self, drawing_image):
        result = process_geometry(drawing_image)
        # Should detect at least one circle (vessel bore)
        assert len(result.circles) >= 1

    def test_detects_lines(self, drawing_image):
        result = process_geometry(drawing_image)
        assert len(result.lines) >= 1

    def test_detects_contours(self, drawing_image):
        result = process_geometry(drawing_image)
        assert len(result.contours) >= 1

    def test_scale_computed_with_known_dim(self, drawing_image):
        result = process_geometry(
            drawing_image,
            known_dimension_mm=10000.0,
            known_dimension_px=600.0,
        )
        assert result.scale_px_per_mm == pytest.approx(0.06)


# ── Geometry Assembler Tests ──────────────────────────────────────────────────

class TestGeometryAssembler:
    def test_assemble_complete_geometry(self, drawing_image):
        import cv2
        from app.services.part2_extraction.opencv_processor import process_geometry

        ocr = OCRResult()
        ocr.raw_texts = [
            "ZZ-BOF-180-001", "Rev B", "180T BOF",
            "Ø10000", "Ø7570", "97948",
            "Tap Hole", "Charge Pad",
        ]
        _parse_all_texts(ocr)

        opencv_res = process_geometry(drawing_image)
        dino = [
            BoundingBox(x_min=10, y_min=10, x_max=200, y_max=100,
                        confidence=0.9, label="title block"),
            BoundingBox(x_min=500, y_min=300, x_max=600, y_max=400,
                        confidence=0.75, label="tap hole"),
        ]

        geometry = assemble_vessel_geometry(
            ocr_result=ocr,
            opencv_result=opencv_res,
            dino_detections=dino,
            source_file="test_bof.png",
            source_format=DrawingFormat.PNG,
        )

        assert isinstance(geometry, VesselGeometry)
        assert geometry.vessel_type == VesselType.BOF
        assert geometry.drawing_number == "ZZ-BOF-180-001"
        assert geometry.operating_conditions.vessel_capacity_tons == pytest.approx(180.0)
        assert geometry.shell.outer_diameter_mm is not None
        assert geometry.shell.inner_diameter_mm is not None
        assert geometry.extraction_confidence > 0.0

    def test_low_confidence_on_empty_data(self):
        from app.services.part2_extraction.opencv_processor import OpenCVGeometryResult
        geometry = assemble_vessel_geometry(
            ocr_result=OCRResult(),
            opencv_result=OpenCVGeometryResult(),
            dino_detections=[],
            source_file="empty.png",
            source_format=DrawingFormat.PNG,
        )
        assert geometry.extraction_confidence < 0.5


# ── API Endpoint Tests ────────────────────────────────────────────────────────

class TestExtractionAPI:
    @patch("app.services.part2_extraction.dino_detector.detect_drawing_elements")
    @patch("pytesseract.image_to_string")
    def test_process_endpoint(self, mock_ocr, mock_dino, sample_png_bytes):
        """Test full Part 1 + Part 2 pipeline via API with mocked DINO + OCR."""
        mock_dino.return_value = []
        mock_ocr.return_value = "180T BOF\nZZ-BOF-180-001\nRev B\nO10000\nO7570\n97948"

        response = client.post(
            "/api/v1/extraction/process",
            files={"file": ("bof_drawing.png", sample_png_bytes, "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "vessel_geometry" in data
        assert "job_id" in data
        assert "processing_time_seconds" in data

    def test_get_result_not_found(self):
        response = client.get("/api/v1/extraction/nonexistent-id/result")
        assert response.status_code == 404
