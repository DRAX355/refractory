"""
app/services/part2_extraction/recon/scene.py
============================================
Reader-independent description of a 2-D engineering drawing.

Every reader (DXF, vector PDF, raster) produces a ``Scene``.  Geometry is stored
in *drawing units* (``Scene.mm_per_unit`` converts to millimetres) with a
**y-up** orientation, so downstream code never needs to know where the data came
from.  ``Scene.exact`` tells the reconstruction whether the coordinates are CAD
exact (DXF / vector PDF) or measured from pixels (raster).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

Pt = tuple[float, float]

ROLE_GEOMETRY = "geometry"
ROLE_CENTERLINE = "centerline"
ROLE_HIDDEN = "hidden"
ROLE_ANNOTATION = "annotation"


@dataclass
class Prim:
    """A flattened curve (line, arc, polyline, spline …) as an ordered point list."""
    pts: list[Pt]
    role: str = ROLE_GEOMETRY
    layer: str = ""
    linetype: str = ""
    kind: str = "poly"          # line | arc | circle | poly | spline
    closed: bool = False

    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return min(xs), min(ys), max(xs), max(ys)

    def length(self) -> float:
        return sum(math.dist(self.pts[i], self.pts[i + 1]) for i in range(len(self.pts) - 1))


@dataclass
class DimRec:
    """One dimension annotation with the points it measures."""
    id: str = ""
    kind: str = "linear"        # linear | diameter | radius | angular | unknown
    value: float = 0.0          # millimetres (or degrees for angular)
    unit: str = "mm"
    text: str = ""              # text actually drawn / read
    p1: Pt | None = None        # first measured point (scene units)
    p2: Pt | None = None        # second measured point (scene units)
    text_pos: Pt | None = None
    direction: float | None = None   # radians; axis the distance is measured along
    source: str = ""            # dxf_dimension | ocr | derived | override
    confidence: float = 1.0
    status: str = "ok"          # ok | corrected | measured | override | unreadable
    note: str = ""


@dataclass
class Scene:
    prims: list[Prim] = field(default_factory=list)
    dims: list[DimRec] = field(default_factory=list)
    texts: list[tuple[str, Pt]] = field(default_factory=list)
    mm_per_unit: float = 1.0
    exact: bool = True                  # True for CAD-exact coordinates
    tolerance_mm: float = 0.05          # geometric accuracy of the coordinates
    source: str = ""                    # dxf | pdf_vector | raster
    warnings: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ── helpers ────────────────────────────────────────────────────────────
    def geometry(self) -> list[Prim]:
        return [p for p in self.prims if p.role == ROLE_GEOMETRY]

    def centerlines(self) -> list[Prim]:
        return [p for p in self.prims if p.role == ROLE_CENTERLINE]

    def bbox(self, roles: tuple[str, ...] = (ROLE_GEOMETRY, ROLE_CENTERLINE)):
        xs: list[float] = []
        ys: list[float] = []
        for p in self.prims:
            if p.role in roles:
                b = p.bbox()
                xs += [b[0], b[2]]
                ys += [b[1], b[3]]
        if not xs:
            return None
        return min(xs), min(ys), max(xs), max(ys)
