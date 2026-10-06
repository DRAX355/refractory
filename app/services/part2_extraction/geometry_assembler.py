"""
app/services/part2_extraction/geometry_assembler.py
====================================================
Part 2 — Geometry Assembler.

Takes the outputs of DINO, OCR, OpenCV, DXF direct parse, and PDF path
extraction and assembles them into the final `VesselGeometry` structured model.

NO HARDCODED DEFAULTS. Every dimension in the output is either:
  a) directly extracted from the drawing, or
  b) left as None / empty so the UI can show "insufficient data"

Source priority for the vessel profile (shell segments):
  1. DXF direct entity parse    → exact vector geometry (most accurate)
  2. PDF vector paths            → exact drawn paths from PDF
  3. OpenCV contour analysis     → pixel-level silhouette analysis
  4. PDF text positioned dims    → text labels with spatial coordinates

If none of the above yield a usable profile, shell.profile.segments is empty
and extraction_method = "none" — the UI should show a "cannot render" state
rather than silently showing wrong geometry.
"""
from __future__ import annotations

import math
import statistics
from loguru import logger

from app.models.vessel_geometry import (
    BoundingBox,
    Dimension,
    DimensionUnit,
    Nozzle,
    SegmentType,
    VesselGeometry,
    VesselProfile,
    VesselSegment,
    VesselShellGeometry,
    VesselType,
    ViewRegion,
    DrawingFormat,
)
from app.services.part2_extraction.ocr_extractor import OCRResult, make_dimension_model
from app.services.part2_extraction.opencv_processor import OpenCVGeometryResult


def assemble_vessel_geometry(
    ocr_result: OCRResult,
    opencv_result: OpenCVGeometryResult,
    dino_detections: list[BoundingBox],
    source_file: str,
    source_format: DrawingFormat,
    dxf_profile=None,      # DXFProfile | None
    pdf_path_profile=None, # PDFPathProfile | None
) -> VesselGeometry:
    """
    Assemble a VesselGeometry from Part 2 extraction results.
    NO default/estimated fallbacks — only real extracted data is used.
    """
    logger.info("[Assembler] Assembling VesselGeometry from extraction results…")

    geometry = VesselGeometry(
        source_file=source_file,
        source_format=source_format,
    )

    geometry.vessel_type    = _resolve_vessel_type(ocr_result)
    geometry.drawing_number = ocr_result.drawing_number
    geometry.revision       = ocr_result.revision

    geometry.shell = _assemble_shell(
        ocr=ocr_result,
        opencv=opencv_result,
        dxf_profile=dxf_profile,
        pdf_path_profile=pdf_path_profile,
    )

    geometry.operating_conditions = ocr_result.operating_conditions
    geometry.nozzles               = _assemble_nozzles(dino_detections, ocr_result)
    geometry.detected_views        = _build_view_regions(dino_detections, ocr_result)
    geometry.raw_ocr_texts         = ocr_result.raw_texts
    geometry.dino_detections       = dino_detections
    geometry.extraction_confidence = _compute_confidence(geometry, ocr_result)

    n_segs = len(geometry.shell.profile.segments)
    method = geometry.shell.profile.extraction_method
    logger.info(
        f"[Assembler] Done — type={geometry.vessel_type}, "
        f"conf={geometry.extraction_confidence:.2f}, "
        f"segments={n_segs}, method={method}"
    )
    return geometry


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_vessel_type(ocr: OCRResult) -> VesselType:
    hint = ocr.vessel_type_hint.upper()
    mapping = {
        "BOF": VesselType.BOF,
        "EAF": VesselType.EAF,
        "LADLE": VesselType.LADLE,
        "RH_DEGASSER": VesselType.RH_DEGASSER,
        "RH": VesselType.RH_DEGASSER,
    }
    return mapping.get(hint, VesselType.UNKNOWN)


def _assemble_shell(
    ocr: OCRResult,
    opencv: OpenCVGeometryResult,
    dxf_profile,
    pdf_path_profile,
) -> VesselShellGeometry:
    shell = VesselShellGeometry()

    # ── Classify OCR/PDF text dimensions into diameter vs length ──────────────
    # We DON'T assume fixed ranges. Instead we collect all values, sort them,
    # and classify by relative size (the largest round value is likely OD,
    # the next is ID, the total span is the length).
    all_dims_mm: list[tuple[float, str, str]] = []  # (value_mm, unit, raw_text)
    for (value, unit), raw in zip(ocr.parsed_dimensions, ocr.dimension_strings):
        vm = _to_mm(value, unit)
        if vm is not None and vm > 0:
            all_dims_mm.append((vm, unit, raw))

    # Sort by value descending
    all_dims_mm.sort(key=lambda x: x[0], reverse=True)

    # Heuristic-free classification: let the data decide
    diameters = [(v, u, r) for v, u, r in all_dims_mm if _looks_like_diameter(v, u, r)]
    lengths    = [(v, u, r) for v, u, r in all_dims_mm if not _looks_like_diameter(v, u, r)]

    # If no explicit diameters are found, fallback to physics heuristics:
    # The largest dimension is usually the total length, the second largest is the outer diameter.
    if not diameters and len(lengths) >= 2:
        # Move the second largest length to diameters ONLY if it's > 50% of the total length.
        # Otherwise, it might be a tall segment height!
        if lengths[1][0] > lengths[0][0] * 0.5:
            diameters.append(lengths.pop(1))
    elif not diameters and len(lengths) == 1:
        # If there's only one dimension, we can't be sure, but let's assume it's length
        pass

    if diameters:
        v, u, raw = diameters[0]
        shell.outer_diameter_mm = make_dimension_model(v, u, raw, confidence=0.85)
        logger.debug(f"[Assembler] Outer dia: {v}{u} ('{raw}')")

    if len(diameters) >= 2:
        # Inner diameter = largest diameter that is smaller than outer
        outer_mm = shell.outer_diameter_mm.value if shell.outer_diameter_mm else float("inf")
        for v, u, raw in diameters[1:]:
            if v < outer_mm:
                shell.inner_diameter_mm = make_dimension_model(v, u, raw, confidence=0.8)
                logger.debug(f"[Assembler] Inner dia: {v}{u} ('{raw}')")
                break

    if lengths:
        v, u, raw = lengths[0]
        shell.total_length_mm = make_dimension_model(v, u, raw, confidence=0.75)
        logger.debug(f"[Assembler] Total length: {v}{u} ('{raw}')")

    # OpenCV circle cross-validation (only if scale is known)
    if opencv.circles and opencv.scale_px_per_mm and shell.inner_diameter_mm is None:
        largest = opencv.circles[0]
        dia_mm = round(largest.radius_px * 2 / opencv.scale_px_per_mm, 0)
        logger.debug(f"[Assembler] OpenCV circle dia estimate: {dia_mm}mm")
        outer_mm = shell.outer_diameter_mm.value if shell.outer_diameter_mm else 0
        # Only use if it is smaller than the outer diameter (= inner dia)
        if 0 < dia_mm < outer_mm:
            shell.inner_diameter_mm = Dimension(
                value=dia_mm,
                unit=DimensionUnit.MM,
                raw_text=f"~{dia_mm}mm (OpenCV estimate)",
                confidence=0.45,
            )

    # ── Build the per-segment profile (priority: DXF > PDF paths > OpenCV) ────
    shell.profile = _assemble_profile(
        shell=shell,
        opencv=opencv,
        dxf_profile=dxf_profile,
        pdf_path_profile=pdf_path_profile,
        ocr=ocr,
    )

    return shell


def _looks_like_diameter(value_mm: float, unit: str, raw_text: str) -> bool:
    """
    Return True if this dimension is explicitly marked as a diameter.
    Signals:
      - raw text starts with Ø/Phi/DIA
      - raw text starts with '0' followed by digits (OCR misread of Ø)
    """
    import re
    if re.search(r"[Øø\u00d8\u03a6]|DIA", raw_text, re.IGNORECASE):
        return True
    # Catch OCR errors where 'Ø' is misread as '0'
    if re.match(r"^0\d{3,}", raw_text):
        return True
    return False

def _assemble_profile(
    shell: VesselShellGeometry,
    opencv: OpenCVGeometryResult,
    dxf_profile,
    pdf_path_profile,
    ocr: OCRResult = None,
) -> VesselProfile:
    """
    Build VesselProfile from extracted data only.
    Priority: DXF > PDF paths > OpenCV zones + positioned dims > positioned dims alone
    Returns empty profile if no usable data.
    """
    profile = VesselProfile()

    outer_r = shell.outer_diameter_mm.value / 2.0 if shell.outer_diameter_mm else None
    total_h = shell.total_length_mm.value if shell.total_length_mm else None

    # ── Priority 1: DXF direct geometry ───────────────────────────────────────
    if dxf_profile and dxf_profile.success and len(dxf_profile.points) >= 3:
        segs = _segments_from_profile_points(
            profile_points=dxf_profile.points,
            known_outer_r=outer_r,
            known_total_h=total_h,
            ocr=ocr,
        )
        if segs:
            profile.segments = segs
            profile.extraction_method = "dxf_direct"
            profile.total_height_mm   = sum(s.height_mm for s in segs)
            profile.max_outer_radius_mm = max(
                max(s.radius_bottom_mm, s.radius_top_mm) for s in segs
            )
            _renumber(segs)
            _compute_slopes(segs)
            logger.info(f"[Assembler] Profile from DXF: {len(segs)} segments")
            return profile

    # ── Priority 2: PDF vector paths ───────────────────────────────────────────
    if pdf_path_profile and pdf_path_profile.success and len(pdf_path_profile.points) >= 3:
        segs = _segments_from_profile_points(
            profile_points=pdf_path_profile.points,
            known_outer_r=outer_r,
            known_total_h=total_h,
            ocr=ocr,
        )
        if segs:
            profile.segments = segs
            profile.extraction_method = "pdf_paths"
            profile.total_height_mm   = sum(s.height_mm for s in segs)
            profile.max_outer_radius_mm = max(
                max(s.radius_bottom_mm, s.radius_top_mm) for s in segs
            )
            _renumber(segs)
            _compute_slopes(segs)
            logger.info(f"[Assembler] Profile from PDF paths: {len(segs)} segments")
            return profile

    # ── Priority 3: OpenCV zones (only if scale is known OR total_h is known) ──
    if opencv.vessel_zones and len(opencv.vessel_zones) >= 2:
        segs = _segments_from_opencv_zones(
            opencv=opencv,
            outer_r_mm=outer_r,
            total_h_mm=total_h,
            positioned_dims=positioned_dims,
            ocr=ocr,
        )
        if segs:
            profile.segments = segs
            profile.extraction_method = (
                "opencv+dims" if any(s.is_extracted for s in segs) else "opencv"
            )
            profile.total_height_mm = sum(s.height_mm for s in segs)
            profile.max_outer_radius_mm = max(
                max(s.radius_bottom_mm, s.radius_top_mm) for s in segs
            ) if segs else 0.0
            _renumber(segs)
            _compute_slopes(segs)
            logger.info(
                f"[Assembler] Profile from OpenCV: {len(segs)} segments, "
                f"method={profile.extraction_method}"
            )
            return profile

    # ── Priority 4: PDF positioned dims only ──────────────────────────────────
    if positioned_dims and total_h:
        segs = _segments_from_positioned_dims(
            dims=positioned_dims,
            outer_r_mm=outer_r,
            total_h_mm=total_h,
        )
        if segs:
            profile.segments = segs
            profile.extraction_method = "pdf_text_dims"
            profile.total_height_mm   = sum(s.height_mm for s in segs)
            profile.max_outer_radius_mm = outer_r or max(
                max(s.radius_bottom_mm, s.radius_top_mm) for s in segs
            )
            _renumber(segs)
            _compute_slopes(segs)
            logger.info(f"[Assembler] Profile from positioned dims: {len(segs)} segments")
            return profile

    # ── No usable data — return empty (UI will show "cannot render") ──────────
    logger.warning(
        "[Assembler] No usable geometry data for profile — "
        "segments list is empty. UI should show 'insufficient data'."
    )
    profile.extraction_method = "none"
    return profile


def _segments_from_profile_points(
    profile_points: list,
    known_outer_r: float | None,
    known_total_h: float | None,
    ocr: OCRResult = None,
) -> list[VesselSegment]:
    """
    Convert a list of (y_mm, r_mm) ProfilePoints into VesselSegments by:
    1. Detecting slope-change breakpoints in the radius profile
    2. If known_total_h is given, scale the y-axis accordingly
    3. If positioned_dims are given, use them to assign real mm heights
    """
    if len(profile_points) < 3:
        return []

    pts = sorted(profile_points, key=lambda p: p.y_mm)

    raw_h = pts[-1].y_mm - pts[0].y_mm
    h_scale = (known_total_h / raw_h) if (known_total_h and raw_h > 0) else None

    raw_r_max = max(p.r_mm for p in pts)
    r_scale = (known_outer_r / raw_r_max) if (known_outer_r and raw_r_max > 0) else None

    if h_scale is not None and r_scale is None:
        r_scale = h_scale
    elif r_scale is not None and h_scale is None:
        h_scale = r_scale
    elif h_scale is None and r_scale is None:
        h_scale = 1.0
        r_scale = 1.0

    # Apply scale
    scaled = [(p.y_mm * h_scale, p.r_mm * r_scale) for p in pts]

    # Find breakpoints in the radius derivative
    breakpoint_ys = _find_breakpoints_from_profile(scaled)

    # Build zones from breakpoints
    boundaries = [scaled[0][0]] + breakpoint_ys + [scaled[-1][0]]
    segments: list[VesselSegment] = []

    for i in range(len(boundaries) - 1):
        y_bot = boundaries[i]
        y_top = boundaries[i + 1]
        h_mm  = y_top - y_bot
        if h_mm < 1.0:
            continue

        # Sample radius at bottom and top of this zone
        r_bot = _interpolate_r(scaled, y_bot)
        r_top = _interpolate_r(scaled, y_top)

        seg_type = _classify_from_radii(r_bot, r_top, h_mm)

        segments.append(VesselSegment(
            index=i,
            segment_type=seg_type,
            label=_label_for_type(seg_type, i),
            height_mm=round(h_mm, 1),
            radius_bottom_mm=round(r_bot, 1),
            radius_top_mm=round(r_top, 1),
            is_extracted=True,
        ))

    # Match the explicit OCR numbers to the segments to remove 'hallucinated' / NTS generic values
    if segments and known_total_h and ocr:
        # Fallback: We have no position data, but we have parsed dimensions.
        # If the number of extracted lengths equals the number of segments, try applying them.
        all_dims = []
        for v, u in ocr.parsed_dimensions:
            if u == "mm" and v < known_total_h * 0.9:
                if known_outer_r and abs(v - known_outer_r * 2) < 1.0:
                    continue
                if v not in all_dims:
                    all_dims.append(v)
        
        # Find the subset of dimensions that best sum to the total height
        from itertools import combinations
        best_subset = []
        best_err = float('inf')
        
        for r in range(1, min(9, len(all_dims) + 1)):
            for subset in combinations(all_dims, r):
                s_sum = sum(subset)
                err = abs(s_sum - known_total_h) / max(known_total_h, 1)
                if err < best_err:
                    best_err = err
                    best_subset = list(subset)
                    
        if best_err < 0.05:
            new_segments = []
            y_curr = scaled[0][0]
            
            for i, h in enumerate(best_subset):
                y_bot = y_curr
                y_top = y_curr + h
                
                r_bot = _interpolate_r(scaled, min(y_bot, scaled[-1][0]))
                r_top = _interpolate_r(scaled, min(y_top, scaled[-1][0]))
                seg_type = _classify_from_radii(r_bot, r_top, h)
                
                new_segments.append(VesselSegment(
                    index=i,
                    segment_type=seg_type,
                    label=_label_for_type(seg_type, i),
                    height_mm=h,
                    height_raw_text=str(h),
                    radius_bottom_mm=round(r_bot, 1),
                    radius_top_mm=round(r_top, 1),
                    is_extracted=True,
                ))
                y_curr = y_top
            
            logger.info(f"[Assembler] Fallback: Replaced {len(segments)} CAD segments with {len(new_segments)} exact OCR segments (err={best_err:.3f}).")
            return new_segments

    return segments


def _find_breakpoints_from_profile(
    scaled: list[tuple[float, float]],
    min_zone_height: float | None = None,
    slope_change_ratio: float = 0.15,
) -> list[float]:
    """
    Detect y positions where the radius slope changes significantly.
    slope_change_ratio: fractional change in slope relative to max radius
    that triggers a new zone boundary.
    """
    if len(scaled) < 4:
        return []

    total_h  = scaled[-1][0] - scaled[0][0]
    max_r    = max(r for _, r in scaled)
    min_zone = min_zone_height or (total_h * 0.03)  # at least 3% of height per zone

    breakpoints: list[float] = []
    last_bp = scaled[0][0]

    for i in range(1, len(scaled) - 1):
        y_prev, r_prev = scaled[i - 1]
        y_curr, r_curr = scaled[i]
        y_next, r_next = scaled[i + 1]

        dy_before = y_curr - y_prev
        dy_after  = y_next - y_curr

        if dy_before < 1e-9 or dy_after < 1e-9:
            continue

        drdy_before = (r_curr - r_prev) / dy_before
        drdy_after  = (r_next - r_curr) / dy_after

        slope_change = abs(drdy_after - drdy_before)
        threshold    = max_r * slope_change_ratio / max(total_h / 100, 1)

        if slope_change > threshold and (y_curr - last_bp) > min_zone:
            breakpoints.append(y_curr)
            last_bp = y_curr

    return breakpoints


def _interpolate_r(
    scaled: list[tuple[float, float]],
    y_target: float,
) -> float:
    """Linear interpolation of r at y_target from the sorted profile."""
    for i in range(len(scaled) - 1):
        y1, r1 = scaled[i]
        y2, r2 = scaled[i + 1]
        if y1 <= y_target <= y2:
            if abs(y2 - y1) < 1e-9:
                return (r1 + r2) / 2
            t = (y_target - y1) / (y2 - y1)
            return r1 + t * (r2 - r1)
    # Clamp to endpoints
    if y_target <= scaled[0][0]:
        return scaled[0][1]
    return scaled[-1][1]


def _apply_annotation_dims(
    segments: list[VesselSegment],
    annotation_dims: list[tuple[float, float, float]],
    total_h: float | None,
) -> None:
    """
    Try to match text dimension annotations to segments by proximity.
    annotation_dims: list of (value_mm, x, y) in DXF/page coordinates.
    """
    if not annotation_dims or not segments:
        return
    # Build cumulative y positions for segments (bottom→top)
    cumulative_y = [0.0]
    for s in segments:
        cumulative_y.append(cumulative_y[-1] + s.height_mm)

    seg_total = cumulative_y[-1]
    if seg_total < 1:
        return

    # For each annotation, find which segment it falls in (by y fraction)
    if not total_h:
        return

    # Map annotation y-coordinates to segment indices
    ann_y_min = min(a[2] for a in annotation_dims)
    ann_y_max = max(a[2] for a in annotation_dims)
    ann_y_span = max(ann_y_max - ann_y_min, 1.0)

    for val_mm, ax, ay in annotation_dims:
        # Normalise the annotation y to [0,1] fraction of total height
        ann_frac = (ay - ann_y_min) / ann_y_span
        y_mm_pos = ann_frac * seg_total

        # Find which segment this corresponds to
        for si, seg in enumerate(segments):
            seg_y_bot = cumulative_y[si]
            seg_y_top = cumulative_y[si + 1]
            if seg_y_bot <= y_mm_pos <= seg_y_top:
                # Only override if the value is plausible for this segment
                h_frac = seg.height_mm / seg_total if seg_total > 0 else 1
                if 0.005 <= val_mm / seg_total <= 0.8:
                    seg.height_mm = round(val_mm, 1)
                    seg.height_raw_text = str(round(val_mm, 0))
                    seg.is_extracted = True
                break


def _segments_from_opencv_zones(
    opencv: OpenCVGeometryResult,
    outer_r_mm: float | None,
    total_h_mm: float | None,
    positioned_dims: list,
    ocr: OCRResult = None,
) -> list[VesselSegment]:
    """Convert OpenCV VesselZone objects to VesselSegments."""
    zones = opencv.vessel_zones
    if not zones:
        return []

    total_h_px = sum(z.height_px for z in zones)
    if total_h_px <= 0:
        return []

    max_r_px = max(max(z.radius_bottom_px, z.radius_top_px) for z in zones)
    if max_r_px <= 0:
        return []

    # We NEED either scale or total_h to convert pixel heights to mm
    if opencv.scale_px_per_mm:
        h_px_to_mm = lambda px: px / opencv.scale_px_per_mm
        r_px_to_mm = lambda px: px / opencv.scale_px_per_mm
    elif total_h_mm:
        h_px_to_mm = lambda px: (px / total_h_px) * total_h_mm
        r_px_to_mm = lambda px: (px / max_r_px) * (outer_r_mm or max_r_px)
    else:
        # No scale reference — cannot convert to mm
        logger.warning("[Assembler] OpenCV zones: no scale or total_h — cannot convert to mm")
        return []

    # Build lookup from positioned dims for enriching zone heights
    pdim_by_zone: dict[int, list[float]] = {}
    if positioned_dims and total_h_mm:
        page_h = positioned_dims[0].page_height if positioned_dims else 0.0
        y_top_total = min(z.y_top_px for z in zones)
        y_bot_total = max(z.y_bottom_px for z in zones)
        span = max(y_bot_total - y_top_total, 1)
        for z_idx, z in enumerate(zones):
            frac_top = (z.y_top_px - y_top_total) / span
            frac_bot = (z.y_bottom_px - y_top_total) / span
            for pdim in positioned_dims:
                norm_y = (pdim.y / page_h) if page_h > 0 else 0.5
                if frac_top <= norm_y <= frac_bot:
                    pdim_by_zone.setdefault(z_idx, []).append(pdim.value)

    segments: list[VesselSegment] = []
    for zi, zone in enumerate(reversed(zones)):
        orig_zi = len(zones) - 1 - zi
        h_mm = h_px_to_mm(zone.height_px)

        zone_vals = pdim_by_zone.get(orig_zi, [])
        plausible = [v for v in zone_vals if 10 <= v <= (total_h_mm or 1e9) * 0.6]
        is_extracted = False
        if plausible:
            h_mm = statistics.median(plausible)
            is_extracted = True

        r_bot_mm = r_px_to_mm(zone.radius_bottom_px)
        r_top_mm = r_px_to_mm(zone.radius_top_px)

        seg_type = _map_zone_type(zone.zone_type)

        segments.append(VesselSegment(
            index=zi,
            segment_type=seg_type,
            label=_label_for_type(seg_type, zi),
            height_mm=round(h_mm, 1),
            radius_bottom_mm=round(r_bot_mm, 1),
            radius_top_mm=round(r_top_mm, 1),
            is_extracted=is_extracted,
        ))

    # Re-scale heights to match known total_h if available
    if total_h_mm:
        raw_sum = sum(s.height_mm for s in segments)
        if raw_sum > 0:
            scale = total_h_mm / raw_sum
            for s in segments:
                s.height_mm = round(s.height_mm * scale, 1)

    if not positioned_dims and segments and total_h_mm and ocr:
        all_dims = []
        for v, u in ocr.parsed_dimensions:
            if u == "mm" and v < total_h_mm * 0.9:
                if outer_r_mm and abs(v - outer_r_mm * 2) < 1.0:
                    continue
                if v not in all_dims:
                    all_dims.append(v)
        
        # Find the subset of dimensions that best sum to the total height
        from itertools import combinations
        best_subset = []
        best_err = float('inf')
        
        for r in range(1, min(9, len(all_dims) + 1)):
            for subset in combinations(all_dims, r):
                s_sum = sum(subset)
                err = abs(s_sum - total_h_mm) / max(total_h_mm, 1)
                if err < best_err:
                    best_err = err
                    best_subset = list(subset)
                    
        if best_err < 0.05:
            new_segments = []
            y_curr = 0
            
            # Map radii by interpolating on the generated segments
            cumulative_y = [0]
            for s in segments: cumulative_y.append(cumulative_y[-1] + s.height_mm)
            
            def _interp_r_cv(y):
                for i, s in enumerate(segments):
                    if cumulative_y[i] <= y <= cumulative_y[i+1]:
                        frac = (y - cumulative_y[i]) / max(s.height_mm, 1)
                        return s.radius_bottom_mm + frac * (s.radius_top_mm - s.radius_bottom_mm)
                return segments[-1].radius_top_mm

            for i, h in enumerate(best_subset):
                y_bot = y_curr
                y_top = y_curr + h
                
                r_bot = _interp_r_cv(y_bot)
                r_top = _interp_r_cv(y_top)
                seg_type = _classify_from_radii(r_bot, r_top, h)
                
                new_segments.append(VesselSegment(
                    index=i,
                    segment_type=seg_type,
                    label=_label_for_type(seg_type, i),
                    height_mm=h,
                    height_raw_text=str(h),
                    radius_bottom_mm=round(r_bot, 1),
                    radius_top_mm=round(r_top, 1),
                    is_extracted=True,
                ))
                y_curr = y_top
            
            logger.info(f"[Assembler] OpenCV Fallback: Replaced {len(segments)} CAD segments with {len(new_segments)} exact OCR segments (err={best_err:.3f}).")
            return new_segments

    return segments


def _segments_from_positioned_dims(
    dims: list,
    outer_r_mm: float | None,
    total_h_mm: float,
) -> list[VesselSegment]:
    """
    Build segments purely from PDF positioned dimension text labels.
    Only used when no geometric source is available.
    """
    # Filter to values that are plausible segment heights
    height_dims = [
        d for d in dims
        if 10 <= d.value <= total_h_mm * 0.7
    ]
    if not height_dims:
        return []

    height_dims.sort(key=lambda d: d.y)

    segments: list[VesselSegment] = []
    n = len(height_dims)
    for i, pdim in enumerate(height_dims):
        # Radius: if outer_r is known, use it; otherwise leave as 0
        r = outer_r_mm or 0.0
        seg_type = (
            SegmentType.DOME if i == 0 else
            SegmentType.NECK if i == n - 1 else
            SegmentType.CYLINDER
        )
        segments.append(VesselSegment(
            index=i,
            segment_type=seg_type,
            label=_label_for_type(seg_type, i),
            height_mm=round(pdim.value, 1),
            height_raw_text=pdim.raw_text,
            radius_bottom_mm=round(r, 1),
            radius_top_mm=round(r, 1),
            is_extracted=True,
        ))

    return segments


def _classify_from_radii(r_bot: float, r_top: float, h_mm: float) -> SegmentType:
    """Classify a zone purely from its geometry — no vessel-type assumptions."""
    if h_mm < 5:
        return SegmentType.FLAT_BOTTOM
    if r_bot < 1e-3 or r_top < 1e-3:
        return SegmentType.FLAT_BOTTOM
    ratio = min(r_bot, r_top) / max(r_bot, r_top) if max(r_bot, r_top) > 0 else 1.0
    delta_r = abs(r_top - r_bot)
    slope_deg = math.degrees(math.atan2(delta_r, h_mm))
    if slope_deg < 1.5:
        return SegmentType.CYLINDER
    if ratio < 0.3:
        return SegmentType.NECK
    if slope_deg > 8:
        return SegmentType.CONE
    return SegmentType.DOME


def _map_zone_type(zone_type_str: str) -> SegmentType:
    mapping = {
        "cylinder":    SegmentType.CYLINDER,
        "cone":        SegmentType.CONE,
        "dome":        SegmentType.DOME,
        "neck":        SegmentType.NECK,
        "flat_bottom": SegmentType.FLAT_BOTTOM,
        "transition":  SegmentType.TRANSITION,
    }
    return mapping.get(zone_type_str, SegmentType.UNKNOWN)


def _label_for_type(seg_type: SegmentType, index: int) -> str:
    labels = {
        SegmentType.FLAT_BOTTOM: "Bottom Cap",
        SegmentType.DOME:        "Lower Dome",
        SegmentType.CYLINDER:    "Barrel",
        SegmentType.CONE:        "Taper / Shoulder",
        SegmentType.NECK:        "Neck / Mouth",
        SegmentType.TRANSITION:  "Transition",
        SegmentType.UNKNOWN:     f"Zone {index + 1}",
    }
    return labels.get(seg_type, f"Zone {index + 1}")


def _renumber(segments: list[VesselSegment]) -> None:
    for i, s in enumerate(segments):
        s.index = i


def _compute_slopes(segments: list[VesselSegment]) -> None:
    for s in segments:
        if s.height_mm > 0:
            delta_r = abs(s.radius_top_mm - s.radius_bottom_mm)
            s.slope_angle_deg = round(math.degrees(math.atan2(delta_r, s.height_mm)), 2)
        else:
            s.slope_angle_deg = 0.0


def _assemble_nozzles(
    dino_detections: list[BoundingBox],
    ocr: OCRResult,
) -> list[Nozzle]:
    nozzles: list[Nozzle] = []
    nozzle_labels = {"tap hole", "charge pad", "oxygen lance port", "nozzle opening"}
    for bbox in dino_detections:
        if bbox.label.lower() in nozzle_labels:
            nozzles.append(Nozzle(label=bbox.label, bounding_box=bbox))
    for label in ocr.nozzle_labels:
        if not any(label.lower() in n.label.lower() for n in nozzles):
            nozzles.append(Nozzle(label=label))
    logger.debug(f"[Assembler] Nozzles: {len(nozzles)}")
    return nozzles


def _build_view_regions(
    dino_detections: list[BoundingBox],
    ocr: OCRResult,
) -> list[ViewRegion]:
    view_labels = {"section view", "front elevation view"}
    return [
        ViewRegion(view_type=bbox.label, bounding_box=bbox)
        for bbox in dino_detections
        if bbox.label.lower() in view_labels
    ]


def _compute_confidence(geometry: VesselGeometry, ocr: OCRResult) -> float:
    scores: list[float] = []
    scores.append(0.9 if geometry.shell.outer_diameter_mm else 0.05)
    scores.append(0.8 if geometry.shell.inner_diameter_mm else 0.05)
    scores.append(0.7 if geometry.shell.total_length_mm   else 0.05)
    scores.append(0.9 if geometry.operating_conditions.vessel_capacity_tons else 0.2)
    scores.append(0.8 if geometry.drawing_number else 0.2)
    scores.append(min(1.0, len(ocr.raw_texts) / 50))

    method = geometry.shell.profile.extraction_method
    method_scores = {
        "dxf_direct":    1.0,
        "pdf_paths":     0.95,
        "opencv+dims":   0.75,
        "opencv":        0.55,
        "pdf_text_dims": 0.45,
        "none":          0.0,
    }
    scores.append(method_scores.get(method, 0.0))

    extracted_count = sum(1 for s in geometry.shell.profile.segments if s.is_extracted)
    total_count = len(geometry.shell.profile.segments)
    if total_count > 0:
        scores.append(extracted_count / total_count)
    else:
        scores.append(0.0)

    return round(statistics.mean(scores), 3)


def _to_mm(value: float, unit: str) -> float | None:
    try:
        if unit == "cm":   return value * 10
        if unit == "m":    return value * 1000
        if unit == "inch": return value * 25.4
        return value
    except Exception:
        return None
