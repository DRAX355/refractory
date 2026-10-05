"""
app/services/part2_extraction/opencv_processor.py
==================================================
Part 2 — OpenCV Geometry Processing.

Processes the raw drawing image to extract structural geometry:
  1. Line detection (Hough Lines) — identify dimension lines, shell outline
  2. Circle/ellipse detection — identify circular cross-sections, nozzle bores
  3. Contour analysis — extract vessel outline and zone boundaries
  4. Scale estimation — map pixel distances to real dimensions using known dims
  5. Zone detection — split the vessel profile into discrete segments where
     the radius or slope changes, driven by the vessel contour profile curve

Outputs structured geometry primitives used by the geometry assembler.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DetectedLine:
    """A line segment detected in the drawing."""
    x1: int
    y1: int
    x2: int
    y2: int
    length_px: float = 0.0
    angle_deg: float = 0.0   # 0° = horizontal, 90° = vertical


@dataclass
class DetectedCircle:
    """A circle detected in the drawing (Hough Circles)."""
    cx: int
    cy: int
    radius_px: int


@dataclass
class DetectedContour:
    """A structural contour (polygon approximation)."""
    points: np.ndarray
    area_px: float
    perimeter_px: float
    bounding_rect: tuple[int, int, int, int]   # (x, y, w, h)
    aspect_ratio: float


@dataclass
class VesselZone:
    """
    A discrete axial zone of the vessel where the outer profile changes slope.
    Zones are ordered bottom→top in image Y coordinates (top of image = vessel top).
    """
    y_top_px: int          # pixel y-coordinate of the top boundary of this zone
    y_bottom_px: int       # pixel y-coordinate of the bottom boundary of this zone
    height_px: float       # zone height in pixels
    radius_left_px: float  # outer shell radius at left edge (pixel distance from axis)
    radius_right_px: float # outer shell radius at right edge (average of both sides)
    radius_top_px: float   # radius at the narrow end (top of zone)
    radius_bottom_px: float # radius at the wide end (bottom of zone)
    slope_deg: float       # taper angle in degrees (0 = cylinder)
    zone_type: str = "unknown"  # cylinder | cone | dome | neck | flat_bottom


@dataclass
class OpenCVGeometryResult:
    """All geometry primitives extracted from one drawing image."""
    lines: list[DetectedLine] = field(default_factory=list)
    circles: list[DetectedCircle] = field(default_factory=list)
    contours: list[DetectedContour] = field(default_factory=list)
    # Estimated scale: pixels per mm (None if not determined)
    scale_px_per_mm: float | None = None
    # Largest detected bounding rectangle (likely the vessel outline)
    vessel_bounding_rect: tuple[int, int, int, int] | None = None
    # ── NEW: Discrete axial zones extracted from vessel profile ───────────────
    vessel_zones: list[VesselZone] = field(default_factory=list)
    # Vessel axis in pixel space (x coordinate of symmetry axis)
    vessel_axis_x_px: int | None = None


# ── Public API ────────────────────────────────────────────────────────────────

def process_geometry(
    image_bgr: np.ndarray,
    known_dimension_mm: float | None = None,
    known_dimension_px: float | None = None,
) -> OpenCVGeometryResult:
    """
    Run the full OpenCV geometry extraction pipeline.

    Args:
        image_bgr:             Input drawing image.
        known_dimension_mm:    A known real-world dimension (mm) for scale calc.
        known_dimension_px:    Pixel length corresponding to known_dimension_mm.

    Returns:
        OpenCVGeometryResult with detected lines, circles, contours, and zones.
    """
    result = OpenCVGeometryResult()

    # ── Preprocessing ─────────────────────────────────────────────────────────
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)

    # ── 1. Line detection ─────────────────────────────────────────────────────
    result.lines = _detect_lines(edges)
    logger.debug(f"[OpenCV] Detected {len(result.lines)} line segments")

    # ── 2. Circle detection ───────────────────────────────────────────────────
    result.circles = _detect_circles(gray)
    logger.debug(f"[OpenCV] Detected {len(result.circles)} circles")

    # ── 3. Contour analysis ───────────────────────────────────────────────────
    result.contours = _detect_contours(edges, image_bgr.shape[:2])
    logger.debug(f"[OpenCV] Detected {len(result.contours)} significant contours")

    # ── 4. Vessel bounding rectangle ──────────────────────────────────────────
    result.vessel_bounding_rect = _find_vessel_bounding_rect(result.contours)

    # ── 5. Scale estimation ───────────────────────────────────────────────────
    if known_dimension_mm and known_dimension_px:
        result.scale_px_per_mm = known_dimension_px / known_dimension_mm
        logger.info(
            f"[OpenCV] Scale estimated: {result.scale_px_per_mm:.4f} px/mm"
        )
    elif result.vessel_bounding_rect and known_dimension_mm:
        # Estimate from the bounding rect's larger dimension
        _, _, w, h = result.vessel_bounding_rect
        larger_px = max(w, h)
        result.scale_px_per_mm = larger_px / known_dimension_mm
        logger.info(
            f"[OpenCV] Scale estimated from bounding rect: "
            f"{result.scale_px_per_mm:.4f} px/mm"
        )

    # ── 6. Vessel zone detection (NEW) ────────────────────────────────────────
    result.vessel_zones, result.vessel_axis_x_px = _detect_vessel_zones(
        gray, edges, result.vessel_bounding_rect
    )
    logger.info(
        f"[OpenCV] Detected {len(result.vessel_zones)} vessel zones "
        f"from profile contour"
    )

    logger.info("[OpenCV] Geometry processing complete.")
    return result


def pixels_to_mm(
    pixels: float, scale_px_per_mm: float | None
) -> float | None:
    """Convert a pixel measurement to mm using the scale factor."""
    if scale_px_per_mm is None or scale_px_per_mm == 0:
        return None
    return round(pixels / scale_px_per_mm, 1)


# ── Private helpers ────────────────────────────────────────────────────────────

def _detect_lines(edges: np.ndarray) -> list[DetectedLine]:
    """Detect line segments using Probabilistic Hough Transform."""
    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10,
    )

    lines: list[DetectedLine] = []
    if raw_lines is None:
        return lines

    for seg in raw_lines:
        # OpenCV 4: seg shape is (1, 4) → seg[0]
        # OpenCV 5: seg shape is (4,)   → seg directly
        coords = seg[0] if seg.ndim == 2 else seg
        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        angle = float(np.degrees(np.arctan2(dy, dx))) % 180
        lines.append(DetectedLine(x1, y1, x2, y2, length, angle))

    # Sort by length descending (longest lines first = vessel outlines)
    lines.sort(key=lambda l: l.length_px, reverse=True)
    return lines


def _detect_circles(gray: np.ndarray) -> list[DetectedCircle]:
    """Detect circles using Hough Circle Transform."""
    h, w = gray.shape
    min_radius = max(10, min(h, w) // 50)
    max_radius = min(h, w) // 3

    raw = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_radius * 2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    circles: list[DetectedCircle] = []
    if raw is None:
        return circles

    for c in np.round(raw[0, :]).astype(int):
        circles.append(DetectedCircle(cx=int(c[0]), cy=int(c[1]), radius_px=int(c[2])))

    # Sort by radius descending (largest first = vessel bore)
    circles.sort(key=lambda c: c.radius_px, reverse=True)
    return circles


def _detect_contours(
    edges: np.ndarray, image_shape: tuple[int, int]
) -> list[DetectedContour]:
    """Find and analyse significant contours in the edge image."""
    h, w = image_shape
    total_area = h * w

    contours_raw, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    result: list[DetectedContour] = []
    for cnt in contours_raw:
        area = cv2.contourArea(cnt)
        if area < total_area * 0.001:   # ignore tiny noise
            continue

        perimeter = cv2.arcLength(cnt, closed=True)
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / ch if ch > 0 else 0.0

        result.append(
            DetectedContour(
                points=cnt,
                area_px=area,
                perimeter_px=perimeter,
                bounding_rect=(x, y, cw, ch),
                aspect_ratio=aspect,
            )
        )

    # Sort by area descending
    result.sort(key=lambda c: c.area_px, reverse=True)
    return result


def _find_vessel_bounding_rect(
    contours: list[DetectedContour],
) -> tuple[int, int, int, int] | None:
    """
    Estimate the vessel body bounding rectangle.

    Heuristic: look for a large contour with high aspect ratio
    (vessel shell is much wider than it is tall in elevation view).
    """
    for cnt in contours:
        if cnt.aspect_ratio > 2.5:   # wide elongated shape = vessel elevation
            return cnt.bounding_rect
    # Fallback: return largest contour bounding rect
    if contours:
        return contours[0].bounding_rect
    return None


# ── Vessel zone detection ──────────────────────────────────────────────────────

def _detect_vessel_zones(
    gray: np.ndarray,
    edges: np.ndarray,
    vessel_rect: tuple[int, int, int, int] | None,
) -> tuple[list[VesselZone], int | None]:
    """
    Detect discrete axial zones of the vessel by analysing its horizontal
    outer-profile envelope row-by-row.

    Strategy:
      1. Crop the image to the vessel bounding rectangle.
      2. For each row of pixels (= one axial level of the vessel),
         measure the horizontal extent of the vessel boundary pixels.
      3. Track where the radius (half-width) changes slope significantly —
         these are zone boundaries (cylinder↔cone↔dome transitions).
      4. Group rows into segments of similar slope.

    Returns:
        (zones, axis_x_px) — zones are ordered top→bottom (image coords).
        Returns ([], None) if detection fails.
    """
    if vessel_rect is None:
        logger.debug("[OpenCV-Zones] No vessel bounding rect — skipping zone detection")
        return [], None

    vx, vy, vw, vh = vessel_rect
    img_h, img_w = gray.shape[:2]

    # Clamp to image boundaries
    x0 = max(0, vx)
    y0 = max(0, vy)
    x1 = min(img_w, vx + vw)
    y1 = min(img_h, vy + vh)

    if (x1 - x0) < 20 or (y1 - y0) < 20:
        logger.debug("[OpenCV-Zones] Vessel rect too small for zone analysis")
        return [], None

    # ── Build the vessel edge image in the cropped region ─────────────────────
    crop_edges = edges[y0:y1, x0:x1]

    # Use a dilated mask to bridge small gaps in the edge drawing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(crop_edges, kernel, iterations=2)

    crop_h, crop_w = dilated.shape
    if crop_h < 10 or crop_w < 10:
        return [], None

    is_horizontal = crop_w > crop_h * 1.1
    if is_horizontal:
        dilated = dilated.T
        crop_h, crop_w = dilated.shape

    # ── Extract left-edge and right-edge x-positions per row ──────────────────
    # For each row (axial cross-section), find the leftmost and rightmost
    # white (edge) pixel → these are the vessel shell boundary pixels.
    left_edge_x  = np.full(crop_h, -1, dtype=np.float32)
    right_edge_x = np.full(crop_h, -1, dtype=np.float32)

    for row_idx in range(crop_h):
        row = dilated[row_idx, :]
        nonzero = np.where(row > 0)[0]
        if len(nonzero) >= 2:
            left_edge_x[row_idx]  = float(nonzero[0])
            right_edge_x[row_idx] = float(nonzero[-1])

    # ── Fill gaps (rows where no edge pixels were found) ─────────────────────
    left_edge_x  = _fill_gaps(left_edge_x)
    right_edge_x = _fill_gaps(right_edge_x)

    # Axis is the centre of the vessel profile
    valid = (left_edge_x >= 0) & (right_edge_x >= 0)
    if valid.sum() < 10:
        logger.debug("[OpenCV-Zones] Too few valid edge rows")
        return [], None

    axis_x_crop = float(np.mean(
        (left_edge_x[valid] + right_edge_x[valid]) / 2.0
    ))
    
    if is_horizontal:
        axis_x_img = int(y0 + axis_x_crop)
    else:
        axis_x_img = int(x0 + axis_x_crop)

    # Radius profile: half-width from axis to right edge (more stable side)
    radius_px_profile = np.where(
        right_edge_x >= 0,
        right_edge_x - axis_x_crop,
        -1.0,
    ).astype(np.float32)

    # Smooth the radius profile to reduce OCR annotation noise
    radius_smooth = _smooth_1d(radius_px_profile, window=max(5, crop_h // 60))

    # ── Detect slope-change breakpoints ───────────────────────────────────────
    breakpoints = _find_slope_breakpoints(radius_smooth, crop_h)
    logger.debug(f"[OpenCV-Zones] Found {len(breakpoints)} breakpoints at rows: {breakpoints}")

    # ── Build VesselZone objects from breakpoints ──────────────────────────────
    zones: list[VesselZone] = []
    boundaries = [0] + breakpoints + [crop_h - 1]

    for i in range(len(boundaries) - 1):
        row_top = boundaries[i]
        row_bot = boundaries[i + 1]
        if row_bot - row_top < 3:
            continue

        r_top = float(radius_smooth[row_top]) if radius_smooth[row_top] > 0 else 0.0
        r_bot = float(radius_smooth[row_bot]) if radius_smooth[row_bot] > 0 else 0.0

        # Mean radius
        seg_radii = radius_smooth[row_top:row_bot + 1]
        r_mean = float(np.mean(seg_radii[seg_radii > 0])) if (seg_radii > 0).any() else 0.0

        h_px = float(row_bot - row_top)

        # Slope angle (half-cone angle)
        if h_px > 0:
            delta_r = abs(r_top - r_bot)
            slope_deg = math.degrees(math.atan2(delta_r, h_px))
        else:
            slope_deg = 0.0

        # Classify zone type
        zone_type = _classify_zone(r_top, r_bot, h_px, slope_deg)

        zone = VesselZone(
            y_top_px=int(y0 + row_top),
            y_bottom_px=int(y0 + row_bot),
            height_px=h_px,
            radius_left_px=r_mean,
            radius_right_px=r_mean,
            radius_top_px=r_top,
            radius_bottom_px=r_bot,
            slope_deg=slope_deg,
            zone_type=zone_type,
        )
        zones.append(zone)

    logger.info(
        f"[OpenCV-Zones] Assembled {len(zones)} zones from {len(breakpoints)} breakpoints"
    )
    return zones, axis_x_img


def _fill_gaps(arr: np.ndarray, missing_val: float = -1.0) -> np.ndarray:
    """Linear interpolation to fill rows where no edge was detected."""
    result = arr.copy()
    n = len(result)
    i = 0
    while i < n:
        if result[i] == missing_val:
            # Find next valid value
            j = i + 1
            while j < n and result[j] == missing_val:
                j += 1
            # Find previous valid value
            prev_val = result[i - 1] if i > 0 else result[j] if j < n else 0.0
            next_val = result[j] if j < n else prev_val
            # Interpolate
            span = j - i + 1
            for k in range(i, min(j, n)):
                t = (k - i + 1) / span
                result[k] = prev_val * (1.0 - t) + next_val * t
            i = j
        else:
            i += 1
    return result


def _smooth_1d(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """Simple moving-average smooth of a 1D float array."""
    if window < 2:
        return arr.copy()
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.astype(np.float32)


def _find_slope_breakpoints(
    radius_profile: np.ndarray,
    n_rows: int,
    min_zone_height: int = 20,
    slope_change_threshold: float = 1.5,
) -> list[int]:
    """
    Find rows where the slope of the radius profile changes significantly.

    Uses a sliding-window first-derivative approach:
      - Compute dr/dy in a small window before and after each row.
      - Where |slope_after - slope_before| exceeds threshold → breakpoint.

    Returns list of row indices (sorted ascending).
    """
    if len(radius_profile) < min_zone_height * 2:
        return []

    # Derivative of radius w.r.t. row index
    w = max(3, min_zone_height // 4)
    drdy = np.gradient(radius_profile.astype(np.float64))

    # Smooth the derivative to reduce noise
    drdy_smooth = _smooth_1d(drdy.astype(np.float32), window=w * 2)

    breakpoints: list[int] = []
    last_bp = 0

    for i in range(w, n_rows - w):
        slope_before = float(drdy_smooth[max(0, i - w)])
        slope_after  = float(drdy_smooth[min(n_rows - 1, i + w)])
        change = abs(slope_after - slope_before)

        if change >= slope_change_threshold and (i - last_bp) >= min_zone_height:
            breakpoints.append(i)
            last_bp = i

    return breakpoints


def _classify_zone(
    r_top: float,
    r_bottom: float,
    height_px: float,
    slope_deg: float,
) -> str:
    """Classify a vessel zone from its geometry."""
    r_ratio = r_top / r_bottom if r_bottom > 0 else 1.0

    if height_px < 5:
        return "flat_bottom"
    if slope_deg < 2.0:
        return "cylinder"
    if r_ratio < 0.4 and slope_deg > 20:
        return "neck"
    if slope_deg > 5.0:
        return "cone"
    return "dome"
