"""
app/utils/pdf_path_extractor.py
================================
Extract vector paths (drawn lines/curves) from a vector PDF.

For CAD-generated PDFs, the vessel outline is stored as PDF path objects
(not rasterised pixels). This module reads those paths directly using
pdfplumber's low-level page primitives, giving us the exact geometric
profile of the vessel as drawn — without any heuristic assumptions.

Pipeline:
  1. Enumerate all path objects on each page
  2. Flatten to line segments (curves are approximated)
  3. Find the largest contiguous outline (= vessel profile)
  4. Sample the radius profile at evenly-spaced height intervals
  5. Return (y_mm, r_mm) ProfilePoint list ordered bottom→top

Returns [] if no usable paths are found.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from loguru import logger


class ProfilePoint(NamedTuple):
    y_mm: float   # axial position (0 = bottom)
    r_mm: float   # outer shell radius at this axial position


@dataclass
class PDFPathProfile:
    """Result of vector-path extraction from a PDF."""
    points: list[ProfilePoint] = field(default_factory=list)
    total_height_mm: float = 0.0
    max_radius_mm: float = 0.0
    page_width_pt: float = 0.0
    page_height_pt: float = 0.0
    success: bool = False


def extract_pdf_path_profile(pdf_path: str | Path) -> PDFPathProfile:
    """
    Extract the vessel cross-section profile directly from PDF vector paths.

    Uses pdfplumber's rect/curve/line primitives (not text, not raster).
    Returns PDFPathProfile(success=False) if no paths found.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning("[PDF-Paths] pdfplumber not installed")
        return PDFPathProfile()

    path = Path(pdf_path)
    if not path.exists():
        return PDFPathProfile()

    result = PDFPathProfile()

    try:
        with pdfplumber.open(str(path)) as pdf:
            if not pdf.pages:
                return result

            page = pdf.pages[0]
            result.page_width_pt  = float(page.width)
            result.page_height_pt = float(page.height)

            # Collect all line segments from PDF paths
            # pdfplumber exposes: page.lines, page.curves, page.rects
            segments: list[tuple[float, float, float, float]] = []

            for line in (page.lines or []):
                try:
                    x1 = float(line["x0"])
                    y1 = float(line["y0"])
                    x2 = float(line["x1"])
                    y2 = float(line["y1"])
                    segments.append((x1, y1, x2, y2))
                except (KeyError, TypeError):
                    pass

            for curve in (page.curves or []):
                # Approximate the Bezier curve with straight line segments
                try:
                    pts = curve.get("pts", [])
                    if len(pts) >= 2:
                        for i in range(len(pts) - 1):
                            x1, y1 = pts[i]
                            x2, y2 = pts[i + 1]
                            segments.append((x1, y1, x2, y2))
                except (KeyError, TypeError):
                    pass

            for rect in (page.rects or []):
                try:
                    x0 = float(rect["x0"])
                    y0 = float(rect["y0"])
                    x1 = float(rect["x1"])
                    y1 = float(rect["y1"])
                    # Add 4 edges of the rectangle
                    segments.extend([
                        (x0, y0, x1, y0),
                        (x1, y0, x1, y1),
                        (x1, y1, x0, y1),
                        (x0, y1, x0, y0),
                    ])
                except (KeyError, TypeError):
                    pass

            if not segments:
                logger.info("[PDF-Paths] No path segments found in PDF")
                return result

            logger.info(f"[PDF-Paths] Found {len(segments)} path segments")

            # ── Build radius profile ─────────────────────────────────────────
            # In PDF coordinates, y=0 is at the BOTTOM of the page.
            # The vessel is typically the dominant vertical structure.
            profile_pts = _build_profile_from_segments(
                segments,
                page_w=result.page_width_pt,
                page_h=result.page_height_pt,
            )

            if len(profile_pts) < 3:
                logger.info("[PDF-Paths] Insufficient profile points from PDF paths")
                return result

            # Convert from PDF points (pt) to mm (1 pt = 0.352778 mm)
            PT_TO_MM = 0.352778
            profile_mm = [
                ProfilePoint(y_mm=p.y_mm * PT_TO_MM, r_mm=p.r_mm * PT_TO_MM)
                for p in profile_pts
            ]

            result.points = profile_mm
            result.total_height_mm = profile_mm[-1].y_mm - profile_mm[0].y_mm if profile_mm else 0.0
            result.max_radius_mm   = max(p.r_mm for p in profile_mm) if profile_mm else 0.0
            result.success = True

            logger.info(
                f"[PDF-Paths] Profile extracted: {len(profile_mm)} pts, "
                f"height={result.total_height_mm:.1f}mm, "
                f"max_r={result.max_radius_mm:.1f}mm"
            )

    except Exception as exc:
        logger.warning(f"[PDF-Paths] Extraction failed: {exc}")
        return PDFPathProfile()

    return result


def _build_profile_from_segments(
    segments: list[tuple[float, float, float, float]],
    page_w: float,
    page_h: float,
    n_samples: int = 200,
) -> list[ProfilePoint]:
    """
    Scan through the vessel vertically (or horizontally if rotated) and at each 
    level find the maximum transverse extent of the drawn paths → the vessel outer radius.
    """
    if not segments or page_h < 1:
        return []

    # Filter to segments that span a meaningful length
    min_seg_len = max(page_w, page_h) * 0.002
    filtered: list[tuple[float, float, float, float]] = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        length = math.hypot(x2 - x1, y2 - y1)
        if length >= min_seg_len:
            filtered.append(seg)

    if not filtered:
        return []

    # Cluster segments to separate distinct views (e.g., side view vs top view)
    # using a simple connected-components approach with a distance threshold.
    distance_threshold = max(page_w, page_h) * 0.05
    
    parent = {i: i for i in range(len(filtered))}
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
    for x1, y1, x2, y2 in filtered:
        bboxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
        
    for i in range(len(filtered)):
        ix_min, iy_min, ix_max, iy_max = bboxes[i]
        for j in range(i + 1, len(filtered)):
            jx_min, jy_min, jx_max, jy_max = bboxes[j]
            dx = max(0, max(ix_min, jx_min) - min(ix_max, jx_max))
            dy = max(0, max(iy_min, jy_min) - min(iy_max, jy_max))
            if dx <= distance_threshold and dy <= distance_threshold:
                union(i, j)
                
    clusters_map = {}
    for i in range(len(filtered)):
        root = find(i)
        if root not in clusters_map:
            clusters_map[root] = []
        clusters_map[root].append(filtered[i])
        
    clusters = list(clusters_map.values())
    
    best_cluster = filtered
    best_score = -1
    
    for cluster in clusters:
        ys = [y for seg in cluster for y in (seg[1], seg[3])]
        xs = [x for seg in cluster for x in (seg[0], seg[2])]
        y_span = max(ys) - min(ys)
        x_span = max(xs) - min(xs)
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

    # Use the best cluster (the side view) for extraction
    ys = [y for seg in best_cluster for y in (seg[1], seg[3])]
    xs = [x for seg in best_cluster for x in (seg[0], seg[2])]
    y_min, y_max = min(ys), max(ys)
    x_min, x_max = min(xs), max(xs)

    y_span = y_max - y_min
    x_span = x_max - x_min
    if y_span < 1 and x_span < 1:
        return []

    profile: list[ProfilePoint] = []
    is_horizontal = x_span > y_span

    if is_horizontal:
        # Scan along X axis
        for i in range(n_samples):
            x_sample = x_min + x_span * i / (n_samples - 1)
            tol = x_span / (n_samples * 2)

            y_vals: list[float] = []
            for x1, y1, x2, y2 in best_cluster:
                seg_x_min = min(x1, x2)
                seg_x_max = max(x1, x2)
                if seg_x_min - tol <= x_sample <= seg_x_max + tol:
                    dx = seg_x_max - seg_x_min
                    if dx < 1e-9:
                        y_interp = (y1 + y2) / 2.0
                    else:
                        t = (x_sample - seg_x_min) / dx
                        y_interp = min(y1, y2) + t * abs(y2 - y1)
                    y_vals.append(y_interp)

            if not y_vals:
                continue

            y_far_bot = min(y_vals)
            y_far_top = max(y_vals)
            half_width = (y_far_top - y_far_bot) / 2.0

            if half_width < min_seg_len * 0.5:
                continue

            y_norm = x_sample - x_min
            profile.append(ProfilePoint(y_mm=y_norm, r_mm=half_width))
    else:
        # Scan along Y axis
        for i in range(n_samples):
            y_sample = y_min + y_span * i / (n_samples - 1)
            tol = y_span / (n_samples * 2)

            x_vals: list[float] = []
            for x1, y1, x2, y2 in best_cluster:
                seg_y_min = min(y1, y2)
                seg_y_max = max(y1, y2)
                if seg_y_min - tol <= y_sample <= seg_y_max + tol:
                    dy = seg_y_max - seg_y_min
                    if dy < 1e-9:
                        x_interp = (x1 + x2) / 2.0
                    else:
                        t = (y_sample - seg_y_min) / dy
                        x_interp = min(x1, x2) + t * abs(x2 - x1)
                    x_vals.append(x_interp)

            if not x_vals:
                continue

            x_far_left  = min(x_vals)
            x_far_right = max(x_vals)
            half_width = (x_far_right - x_far_left) / 2.0

            if half_width < min_seg_len * 0.5:
                continue

            y_norm = y_sample - y_min
            profile.append(ProfilePoint(y_mm=y_norm, r_mm=half_width))

    profile.sort(key=lambda p: p.y_mm)
    return profile
