"""
app/models/vessel_geometry.py
==============================
Pydantic data models for the Structured Vessel Geometry.

This is the **contract model** between Part 1/2 (Extraction) and
downstream parts (3 — AI Assistant, 4 — Design Engine, 5 — Geometry Engine).

All other teams should import these models without modification.
Changes MUST be backwards-compatible (add fields, never remove).
"""
from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class VesselType(str, Enum):
    BOF = "BOF"           # Basic Oxygen Furnace
    EAF = "EAF"           # Electric Arc Furnace
    LADLE = "LADLE"       # Ladle / Transfer Vessel
    RH_DEGASSER = "RH"    # RH Degasser
    UNKNOWN = "UNKNOWN"


class DrawingFormat(str, Enum):
    PDF = "pdf"
    DWG = "dwg"
    DXF = "dxf"
    PNG = "png"
    JPG = "jpg"
    TIFF = "tiff"
    UNKNOWN = "unknown"


class DimensionUnit(str, Enum):
    MM = "mm"
    CM = "cm"
    INCH = "inch"
    UNKNOWN = "unknown"


# ── Sub-models ────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Pixel or normalised bounding box from DINO detection."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    label: str = ""


class Dimension(BaseModel):
    """A single extracted dimension with value and unit."""
    value: float
    unit: DimensionUnit = DimensionUnit.MM
    raw_text: str = ""          # original OCR text, e.g. "Ø7570"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class Nozzle(BaseModel):
    """A nozzle or opening on the vessel shell."""
    label: str = ""             # e.g. "Tap Hole", "Charge Pad", "Oxygen Lance"
    diameter_mm: float | None = None
    position_mm: float | None = None   # axial position from reference end
    angle_deg: float | None = None     # angular position
    bounding_box: BoundingBox | None = None


class ViewRegion(BaseModel):
    """A detected drawing view (front, side, section, plan)."""
    view_type: str = ""         # e.g. "front_elevation", "section_AA", "plan"
    bounding_box: BoundingBox | None = None
    extracted_text: list[str] = Field(default_factory=list)


class OperatingConditions(BaseModel):
    """Process / operating parameters extracted from drawing notes."""
    vessel_capacity_tons: float | None = None   # e.g. 180.0 for 180T BOF
    heat_number: str | None = None
    steel_grade: str | None = None
    tapping_temperature_c: float | None = None
    additional_notes: list[str] = Field(default_factory=list)


# ── Vessel Profile Segment models ──────────────────────────────────────────────

class SegmentType(str, Enum):
    """Geometric character of a vessel shell segment."""
    FLAT_BOTTOM = "flat_bottom"   # closed circular plate at base
    DOME        = "dome"          # hemispherical / dished lower head
    CYLINDER    = "cylinder"      # straight cylindrical barrel
    CONE        = "cone"          # tapered / conical frustum
    NECK        = "neck"          # narrow cylindrical mouth at top
    TRANSITION  = "transition"    # short curved blending region
    UNKNOWN     = "unknown"


class VesselSegment(BaseModel):
    """
    One discrete axial zone of the vessel shell with real extracted dimensions.

    Segments are ordered bottom to top (index 0 = vessel bottom).
    Each segment is a frustum — a cylinder is a degenerate frustum
    where radius_top_mm == radius_bottom_mm.
    """
    index: int = 0                     # 0-based order from bottom
    segment_type: SegmentType = SegmentType.UNKNOWN
    label: str = ""                    # human label, e.g. "Barrel", "Lower Dome"

    # Axial extent (along the vessel axis)
    height_mm: float = 0.0            # actual extracted axial height of this segment
    height_raw_text: str = ""         # OCR source text / dimension label

    # Outer-shell radii at bottom and top face of this frustum
    radius_bottom_mm: float = 0.0     # outer shell radius at the bottom face
    radius_top_mm: float = 0.0       # outer shell radius at the top face

    # Derived / computed
    slope_angle_deg: float = 0.0      # half-cone angle (0 deg = perfect cylinder)
    is_extracted: bool = False        # True = real dimension; False = estimate


class VesselProfile(BaseModel):
    """
    Complete axial profile of the vessel, assembled from ordered VesselSegments.
    This is the primary data structure driving the 3D parametric render.
    Segments are ordered bottom to top.
    """
    segments: list[VesselSegment] = Field(default_factory=list)
    total_height_mm: float = 0.0      # sum of all segment heights
    max_outer_radius_mm: float = 0.0  # largest outer shell radius in profile
    extraction_method: str = "estimated"  # "extracted" | "estimated" | "hybrid"


class VesselShellGeometry(BaseModel):
    """Core shell geometry dimensions of the vessel."""
    outer_diameter_mm: Dimension | None = None      # e.g. Ø10000
    inner_diameter_mm: Dimension | None = None      # e.g. Ø7570
    total_length_mm: Dimension | None = None        # e.g. 97948 (as shown)
    shell_thickness_mm: Dimension | None = None
    trunnion_diameter_mm: Dimension | None = None
    bottom_thickness_mm: Dimension | None = None
    cone_angle_deg: float | None = None

    # ── Segment-level profile ─────────────────────────────────────────────────
    # Ordered list of shell segments (bottom→top).
    # Each segment has its own height, radii, and type so the 3D render is
    # built from real extracted dimensions rather than hardcoded proportional guesses.
    profile: VesselProfile = Field(default_factory=VesselProfile)


# ── Primary Contract Model ────────────────────────────────────────────────────

class VesselGeometry(BaseModel):
    """
    Structured Vessel Geometry — primary output of Part 2.
    This is the contract model consumed by all downstream parts (3–7).

    Usage:
        from app.models.vessel_geometry import VesselGeometry
    """
    # ── Metadata ────────────────────────────────────────
    source_file: str = ""                    # original uploaded filename
    source_format: DrawingFormat = DrawingFormat.UNKNOWN
    vessel_type: VesselType = VesselType.BOF
    drawing_number: str = ""
    revision: str = ""
    dimension_unit: DimensionUnit = DimensionUnit.MM

    # ── Geometry ─────────────────────────────────────────
    shell: VesselShellGeometry = Field(default_factory=VesselShellGeometry)
    nozzles: list[Nozzle] = Field(default_factory=list)

    # ── Operating Conditions ─────────────────────────────
    operating_conditions: OperatingConditions = Field(
        default_factory=OperatingConditions
    )

    # ── Extraction Metadata ──────────────────────────────
    detected_views: list[ViewRegion] = Field(default_factory=list)
    raw_ocr_texts: list[str] = Field(default_factory=list)
    dino_detections: list[BoundingBox] = Field(default_factory=list)
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # ── Pass-through for downstream parts ───────────────
    extra_metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_schema_extra = {
            "example": {
                "source_file": "BOF_180T_GA_drawing.pdf",
                "source_format": "pdf",
                "vessel_type": "BOF",
                "drawing_number": "ZZ-BOF-180-001",
                "revision": "Rev B",
                "dimension_unit": "mm",
                "shell": {
                    "outer_diameter_mm": {"value": 10000, "unit": "mm", "raw_text": "Ø10000"},
                    "inner_diameter_mm": {"value": 7570, "unit": "mm", "raw_text": "Ø7570"},
                    "total_length_mm": {"value": 97948, "unit": "mm", "raw_text": "97948"},
                    "profile": {
                        "segments": [
                            {"index": 0, "segment_type": "flat_bottom", "label": "Bottom Cap",
                             "height_mm": 150, "radius_bottom_mm": 0, "radius_top_mm": 5000,
                             "is_extracted": True},
                            {"index": 1, "segment_type": "dome", "label": "Lower Dome",
                             "height_mm": 2416, "radius_bottom_mm": 5000, "radius_top_mm": 5000,
                             "is_extracted": True},
                            {"index": 2, "segment_type": "cylinder", "label": "Barrel",
                             "height_mm": 4884, "radius_bottom_mm": 5000, "radius_top_mm": 5000,
                             "is_extracted": True},
                            {"index": 3, "segment_type": "cone", "label": "Shoulder",
                             "height_mm": 1965, "radius_bottom_mm": 5000, "radius_top_mm": 2600,
                             "slope_angle_deg": 35.2, "is_extracted": True},
                            {"index": 4, "segment_type": "neck", "label": "Mouth",
                             "height_mm": 530, "radius_bottom_mm": 2600, "radius_top_mm": 2600,
                             "is_extracted": True},
                        ],
                        "total_height_mm": 9945,
                        "max_outer_radius_mm": 5000,
                        "extraction_method": "extracted",
                    },
                },
                "operating_conditions": {"vessel_capacity_tons": 180.0},
                "extraction_confidence": 0.82,
            }
        }


# ── API Response Wrappers ─────────────────────────────────────────────────────

class IngestionResponse(BaseModel):
    """Part 1 API response after successful drawing upload."""
    job_id: str
    filename: str
    format: DrawingFormat
    file_size_bytes: int
    page_count: int = 1
    message: str = "Drawing ingested successfully."


class ExtractionResponse(BaseModel):
    """Part 2 API response after drawing intelligence extraction."""
    job_id: str
    vessel_geometry: VesselGeometry
    processing_time_seconds: float
    warnings: list[str] = Field(default_factory=list)
    message: str = "Extraction completed successfully."
