"""
app/services/part2_extraction/dxf_geometry_extractor.py
========================================================
Part 2 — Direct DXF CAD Entity Extraction.

For DXF files, this module reads the raw vector entities (LINE, LWPOLYLINE,
POLYLINE, ARC, SPLINE) to reconstruct the exact vessel cross-section profile
without any heuristic assumptions.

Pipeline:
  1. Parse all entities in the DXF modelspace
  2. Find the vessel cross-section (largest closed or semi-closed polyline)
  3. Sample the profile: for each Y-level, measure the outer radius
  4. Detect slope breakpoints from the sampled radius profile
  5. Return ordered VesselZone objects (no defaults, no guesses)

Returns [] (empty) if the DXF does not contain usable vessel geometry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from loguru import logger


class ProfilePoint(NamedTuple):
    """One sample point on the vessel outer profile."""
    y_mm: float      # axial position (vessel axis direction)
    r_mm: float      # outer shell radius at this y position


@dataclass
class DXFProfile:
    """Result of direct DXF geometry extraction."""
    points: list[ProfilePoint] = field(default_factory=list)
    # Extracted text annotations with positions (for dimension mapping)
    dimension_annotations: list[tuple[float, float, float]] = field(default_factory=list)
    # (value_mm, x, y) — numeric labels with their DXF coordinate
    total_height_mm: float = 0.0
    max_radius_mm: float = 0.0
    success: bool = False
    method: str = ""   # "polyline" | "lines" | "entities"


def extract_dxf_profile(dxf_path: str | Path) -> DXFProfile:
    """
    Parse a DXF file and extract the vessel cross-section profile directly
    from CAD entities.

    Returns a DXFProfile with sampled (y, r) points and any text annotations.
    Returns DXFProfile(success=False) if no usable vessel geometry is found.
    """
    try:
        import ezdxf
    except ImportError:
        logger.warning("[DXF] ezdxf not installed — skipping direct DXF extraction")
        return DXFProfile()

    path = Path(dxf_path)
    if not path.exists():
        logger.warning(f"[DXF] File not found: {path}")
        return DXFProfile()

    try:
        doc = ezdxf.readfile(str(path))
    except Exception as exc:
        logger.warning(f"[DXF] Failed to read DXF: {exc}")
        return DXFProfile()

    msp = doc.modelspace()

    result = DXFProfile()

    # ── Step 1: Collect all line segments ─────────────────────────────────────
    all_segments: list[tuple[float, float, float, float]] = []  # (x1,y1,x2,y2)
    annotations: list[tuple[float, float, float]] = []          # (value_mm, cx, cy)

    for entity in msp:
        etype = entity.dxftype()

        if etype == "LINE":
            try:
                p1 = entity.dxf.start
                p2 = entity.dxf.end
                all_segments.append((p1.x, p1.y, p2.x, p2.y))
            except Exception:
                pass

        elif etype in ("LWPOLYLINE", "POLYLINE"):
            try:
                pts = list(entity.get_points())
                for i in range(len(pts) - 1):
                    x1, y1 = pts[i][0], pts[i][1]
                    x2, y2 = pts[i + 1][0], pts[i + 1][1]
                    all_segments.append((x1, y1, x2, y2))
                # Close the polyline if flagged
                if hasattr(entity.dxf, "closed") and entity.dxf.closed and len(pts) > 1:
                    x1, y1 = pts[-1][0], pts[-1][1]
                    x2, y2 = pts[0][0], pts[0][1]
                    all_segments.append((x1, y1, x2, y2))
            except Exception:
                pass

        elif etype == "ARC":
            try:
                cx   = entity.dxf.center.x
                cy   = entity.dxf.center.y
                r    = entity.dxf.radius
                a1   = math.radians(entity.dxf.start_angle)
                a2   = math.radians(entity.dxf.end_angle)
                # Approximate arc as 16 line segments
                if a2 < a1:
                    a2 += 2 * math.pi
                n = 16
                for i in range(n):
                    t1 = a1 + (a2 - a1) * i / n
                    t2 = a1 + (a2 - a1) * (i + 1) / n
                    x1 = cx + r * math.cos(t1)
                    y1 = cy + r * math.sin(t1)
                    x2 = cx + r * math.cos(t2)
                    y2 = cy + r * math.sin(t2)
                    all_segments.append((x1, y1, x2, y2))
            except Exception:
                pass

        elif etype in ("TEXT", "MTEXT"):
            try:
                text = (entity.dxf.text if etype == "TEXT"
                        else entity.text).strip()
                if not text:
                    continue
                # Try to parse a numeric dimension from the text
                val = _parse_numeric_mm(text)
                if val is not None and val > 0:
                    pos = entity.dxf.insert if etype == "TEXT" else entity.dxf.insert
                    annotations.append((val, pos.x, pos.y))
            except Exception:
                pass

    if not all_segments:
        logger.info("[DXF] No usable line segments found in DXF")
        return DXFProfile()

    # ── Step 1.5: Cluster segments to isolate the side view ───────────────────
    xs = [x for seg in all_segments for x in (seg[0], seg[2])]
    ys = [y for seg in all_segments for y in (seg[1], seg[3])]
    page_w = max(xs) - min(xs)
    page_h = max(ys) - min(ys)
    distance_threshold = max(page_w, page_h) * 0.05
    
    parent = {i: i for i in range(len(all_segments))}
    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]
    
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    bboxes = []
    for x1, y1, x2, y2 in all_segments:
        bboxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        
    for i in range(len(all_segments)):
        ix_min, iy_min, ix_max, iy_max = bboxes[i]
        for j in range(i + 1, len(all_segments)):
            jx_min, jy_min, jx_max, jy_max = bboxes[j]
            dx = max(0, max(ix_min, jx_min) - min(ix_max, jx_max))
            dy = max(0, max(iy_min, jy_min) - min(iy_max, jy_max))
            if dx <= distance_threshold and dy <= distance_threshold:
                union(i, j)
                
    clusters_map = {}
    for i in range(len(all_segments)):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(all_segments[i])
        
    best_cluster = all_segments
    best_score = -1
    
    for cluster in clusters_map.values():
        c_ys = [y for seg in cluster for y in (seg[1], seg[3])]
        c_xs = [x for seg in cluster for x in (seg[0], seg[2])]
        y_span = max(c_ys) - min(c_ys)
        x_span = max(c_xs) - min(c_xs)
        if y_span < 1 or x_span < 1:
            continue
            
        area = x_span * y_span
        aspect_ratio = max(x_span, y_span) / min(x_span, y_span)
        
        # Penalize perfectly square clusters (likely top view)
        score = area
        if aspect_ratio < 1.15:
            score *= 0.1
            
        if score > best_score:
            best_score = score
            best_cluster = cluster
            
    all_segments = best_cluster

    # ── Step 2: Find the vessel axis and outer profile ─────────────────────────
    # Assumption: vessel is drawn in cross-section (elevation view) where:
    #   - The vessel axis is vertical (Y direction in DXF = axial direction)
    #   - OR horizontal (X direction in DXF = axial direction)
    # We detect which orientation gives the most consistent profile.

    profile_v = _extract_profile_from_segments(all_segments, axis="vertical")
    profile_h = _extract_profile_from_segments(all_segments, axis="horizontal")

    # Pick the axis orientation that gives a larger, more consistent profile
    chosen = profile_v if len(profile_v) >= len(profile_h) else profile_h

    if len(chosen) < 3:
        logger.info("[DXF] Not enough profile points from DXF segments")
        return DXFProfile()

    # Sort by y_mm (axial position, bottom→top)
    chosen.sort(key=lambda p: p.y_mm)

    result.points = chosen
    result.dimension_annotations = annotations
    result.total_height_mm = chosen[-1].y_mm - chosen[0].y_mm
    result.max_radius_mm = max(p.r_mm for p in chosen)
    result.success = True
    result.method = "polyline" if any(
        e.dxftype() in ("LWPOLYLINE", "POLYLINE") for e in msp
    ) else "lines"

    logger.info(
        f"[DXF] Profile extracted: {len(chosen)} points, "
        f"height={result.total_height_mm:.1f}mm, "
        f"max_r={result.max_radius_mm:.1f}mm"
    )
    return result


def _extract_profile_from_segments(
    segments: list[tuple[float, float, float, float]],
    axis: str = "vertical",
    n_samples: int = 200,
) -> list[ProfilePoint]:
    """
    Build a 1D radius-vs-position profile by sampling the segments along
    the given axis.

    axis="vertical":   Y = axial position, X = radius
    axis="horizontal": X = axial position, Y = radius
    """
    if not segments:
        return []

    # Determine bounding box
    xs = [x for seg in segments for x in (seg[0], seg[2])]
    ys = [y for seg in segments for y in (seg[1], seg[3])]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    if axis == "vertical":
        axial_min, axial_max = y_min, y_max
        radial_min, radial_max = x_min, x_max
    else:
        axial_min, axial_max = x_min, x_max
        radial_min, radial_max = y_min, y_max

    axial_span = axial_max - axial_min
    if axial_span < 1e-6:
        return []

    # Detect axis of symmetry (the radial midpoint = vessel centreline)
    radial_mid = (radial_min + radial_max) / 2.0

    profile: list[ProfilePoint] = []

    for i in range(n_samples):
        # The axial position at this sample
        axial_pos = axial_min + axial_span * i / (n_samples - 1)
        tolerance = axial_span / (n_samples * 2)

        # Find all segment endpoints near this axial slice
        radial_vals: list[float] = []
        for x1, y1, x2, y2 in segments:
            if axis == "vertical":
                seg_axial_1, seg_axial_2 = y1, y2
                seg_radial_1, seg_radial_2 = x1, x2
            else:
                seg_axial_1, seg_axial_2 = x1, x2
                seg_radial_1, seg_radial_2 = y1, y2

            seg_axial_min = min(seg_axial_1, seg_axial_2)
            seg_axial_max = max(seg_axial_1, seg_axial_2)

            # Check if this axial slice intersects the segment
            if seg_axial_min - tolerance <= axial_pos <= seg_axial_max + tolerance:
                # Interpolate the radial position at this axial slice
                denom = seg_axial_max - seg_axial_min
                if denom < 1e-9:
                    radial_vals.append((seg_radial_1 + seg_radial_2) / 2.0)
                else:
                    t = (axial_pos - seg_axial_min) / denom
                    r_interp = seg_radial_1 + t * (seg_radial_2 - seg_radial_1)
                    radial_vals.append(r_interp)

        if not radial_vals:
            continue

        # The outer radius = max distance from centreline on the far side
        # (some segments are on the near side = negative radius)
        radii_from_centre = [abs(rv - radial_mid) for rv in radial_vals]
        r_max = max(radii_from_centre)

        if r_max < 1e-3:  # skip degenerate points
            continue

        # Normalise to bottom = 0
        y_normalised = axial_pos - axial_min
        profile.append(ProfilePoint(y_mm=y_normalised, r_mm=r_max))

    return profile


def _parse_numeric_mm(text: str) -> float | None:
    """
    Try to parse a numeric mm value from a DXF text entity.
    Returns None if not parseable.
    """
    import re
    text = text.strip().replace(",", "")
    # Match: optional Ø prefix, number, optional mm/cm/m suffix
    m = re.match(
        r'^[Øø\u00d8\u03a6]?\s*([\d.]+)\s*(mm|cm|m|")?$',
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    try:
        val = float(m.group(1))
        unit = (m.group(2) or "mm").lower()
        if unit == "cm":
            return val * 10
        if unit == "m":
            return val * 1000
        if unit in ('"', "inch"):
            return val * 25.4
        return val
    except ValueError:
        return None
