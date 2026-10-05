"""
app/services/part2_extraction/section_reader.py
================================================
Exact reader for *axisymmetric section drawings* (DXF).

Works on drawings that show a vessel as a section through its axis of revolution,
including DXFs that were converted from PDF (everything on layer 0, text exploded
to glyph polylines, dimension lines drawn as plain coloured lines).

Pipeline
--------
 1. read every stroke (LINE / POLYLINE / ARC / ... flattened) with its colour
 2. split strokes into  geometry  |  dimension / extension / axis lines  |  glyphs
 3. find the axis of revolution (long centre line)
 4. recover the real scale: OCR every dimension string (glyphs are rendered back
    to an image) and tie it to the pair of extension lines it measures
 5. ray-cast the geometry from the axis (even-odd) -> exact wall cross-section
 6. segment the outer envelope into dome / cone / cylinder pieces
 7. detect tilted nozzles from their centre line and measure them in their own frame

Nothing here is hard-coded to one drawing: every number comes from the file.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from shapely.geometry import Polygon
from shapely.ops import unary_union

Pt = tuple[float, float]


# ═════════════════════════════════ data classes ══════════════════════════════

@dataclass
class Stroke:
    pts: list[Pt]
    color: int
    src: str = "LINE"

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.pts[:-1], self.pts[1:]))

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.pts]
        ys = [p[1] for p in self.pts]
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class TextItem:
    text: str
    cx: float
    cy: float
    horizontal: bool          # reading direction is along the drawing X axis
    h: float
    source: str               # "glyph-ocr" | "text-entity"
    conf: float = 0.0
    candidates: list[float] = field(default_factory=list)


@dataclass
class DimRec:
    text: str
    value_mm: float
    measured_units: float
    orientation: str          # "axial" | "radial"
    is_diameter: bool
    consistent: bool = True
    source: str = ""


@dataclass
class SegmentRec:
    kind: str                 # dome | cone | cylinder | transition
    label: str
    z0: float
    z1: float
    r_out0: float
    r_out1: float
    r_in0: float = 0.0
    r_in1: float = 0.0


@dataclass
class NozzleRec:
    label: str
    axis_z_mm: float                  # where the nozzle axis meets the vessel axis
    tilt_deg: float                   # angle between nozzle axis and the vessel's radial direction
    direction: tuple[float, float]    # (axial, radial) unit vector, pointing outwards
    r_in_mm: float
    r_out_mm: float
    u_start_mm: float                 # start of the pipe wall along the nozzle axis
    flange_u0_mm: float
    flange_u1_mm: float
    flange_r_mm: float
    radial_side: int                  # -1 = drawn below the vessel axis, +1 above


@dataclass
class SectionResult:
    ok: bool = False
    reason: str = ""
    scale_mm_per_unit: float = 1.0
    scale_source: str = "assumed"
    scale_confidence: float = 0.0
    axis_horizontal: bool = True
    axis_coord: float = 0.0
    z_origin: float = 0.0              # paper x of the left-most geometry point
    body: object = None                # shapely geometry in (z, r) mm
    cavity: object = None              # shapely geometry (z, r) mm: empty space adjacent to the axis
    segments: list[SegmentRec] = field(default_factory=list)
    nozzles: list[NozzleRec] = field(default_factory=list)
    dims: list[DimRec] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_length_mm: float = 0.0
    max_outer_radius_mm: float = 0.0
    shell_thickness_mm: float = 0.0
    inner_diameter_mm: float = 0.0
    outer_diameter_mm: float = 0.0
    confidence: float = 0.0
    # debug overlay data (paper units)
    debug_strokes: list[Stroke] = field(default_factory=list)


# ═════════════════════════════════ loading ═══════════════════════════════════

_PATH_TYPES = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE"}


def _eff_color(doc, e, parent: int | None = None) -> int:
    c = e.dxf.color if e.dxf.hasattr("color") else 256
    if c == 256:
        try:
            return abs(doc.layers.get(e.dxf.layer).dxf.color)
        except Exception:
            return 7
    if c == 0:
        return parent if parent is not None else 7
    return c


def _flatten(e) -> list[list[Pt]]:
    from ezdxf import path as ezpath
    out: list[list[Pt]] = []
    p = ezpath.make_path(e)
    for sp in p.sub_paths():
        pts = [(float(v.x), float(v.y)) for v in sp.flattening(0.02, segments=8)]
        if len(pts) >= 2:
            out.append(pts)
    return out


def _collect(doc, entities, strokes, texts, parent_color=None, depth=0):
    for e in entities:
        t = e.dxftype()
        try:
            if t in _PATH_TYPES:
                col = _eff_color(doc, e, parent_color)
                for pts in _flatten(e):
                    strokes.append(Stroke(pts, col, t))
            elif t == "INSERT" and depth < 3:
                col = _eff_color(doc, e, parent_color)
                _collect(doc, e.virtual_entities(), strokes, texts, col, depth + 1)
            elif t in ("TEXT", "MTEXT"):
                raw = e.dxf.text if t == "TEXT" else e.plain_text()
                raw = (raw or "").strip()
                if not raw:
                    continue
                ins = e.dxf.insert
                rot = float(e.dxf.rotation) if (t == "TEXT" and e.dxf.hasattr("rotation")) else (
                    float(getattr(e.dxf, "rotation", 0.0)) if t == "MTEXT" else 0.0)
                h = float(e.dxf.height) if e.dxf.hasattr("height") else (
                    float(getattr(e.dxf, "char_height", 1.0)))
                horiz = abs(math.sin(math.radians(rot))) < 0.5
                texts.append(TextItem(raw, ins.x, ins.y, horiz, h, "text-entity", 1.0))
        except Exception as exc:                       # noqa: BLE001
            logger.debug(f"[Section] skipped {t}: {exc}")


def _touching_components(strokes: list[Stroke], tol: float) -> list[list[int]]:
    n = len(strokes)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    ends = np.array([[s.pts[0], s.pts[-1]] for s in strokes], dtype=float)  # n,2,2
    bb = np.array([s.bbox for s in strokes], dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            # quick reject by bbox gap
            if (bb[j, 0] - bb[i, 2] > tol or bb[i, 0] - bb[j, 2] > tol or
                    bb[j, 1] - bb[i, 3] > tol or bb[i, 1] - bb[j, 3] > tol):
                continue
            if _poly_touch(strokes[i].pts, strokes[j].pts, tol):
                par[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _poly_touch(a: list[Pt], b: list[Pt], tol: float) -> bool:
    """True if any end-point of one polyline lies within tol of the other polyline."""
    for p in (a[0], a[-1]):
        if _pt_poly_dist(p, b) <= tol:
            return True
    for p in (b[0], b[-1]):
        if _pt_poly_dist(p, a) <= tol:
            return True
    return False


def _pt_poly_dist(p: Pt, poly: list[Pt]) -> float:
    best = 1e18
    px, py = p
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        dx, dy = x2 - x1, y2 - y1
        d2 = dx * dx + dy * dy
        t = 0.0 if d2 < 1e-18 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / d2))
        best = min(best, math.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return best


# ═════════════════════════════════ OCR of glyphs ═════════════════════════════

_NUM_RE = re.compile(r"^\d+(\.\d+)?$")


def _render_glyphs(strokes: list[Stroke], rot_deg: float, px_h: int = 54):
    from PIL import Image, ImageDraw
    th = math.radians(rot_deg)
    c, s = math.cos(th), math.sin(th)
    pts_all = []
    for st in strokes:
        pts_all.append([(x * c - y * s, x * s + y * c) for x, y in st.pts])
    xs = [p[0] for pl in pts_all for p in pl]
    ys = [p[1] for pl in pts_all for p in pl]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if h < 1e-6 or w < 1e-6:
        return None
    k = px_h / h
    W, H = int(w * k) + 60, int(h * k) + 60
    if W > 3000:
        k *= 3000 / W
        W, H = int(w * k) + 60, int(h * k) + 60
    img = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(img)
    lw = max(2, int(px_h * 0.09))
    for pl in pts_all:
        pp = [(30 + (x - min(xs)) * k, 30 + (max(ys) - y) * k) for x, y in pl]
        dr.line(pp, fill=0, width=lw, joint="curve")
    return img


def _ocr_strokes(strokes: list[Stroke], horizontal: bool, tesseract_cmd: str | None):
    """Return list of (text, conf) candidates for a glyph cluster."""
    try:
        import pytesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    except Exception:                                      # noqa: BLE001
        return []
    rots = [0.0] if horizontal else [90.0, -90.0]
    res = []
    for rot in rots:
        img = _render_glyphs(strokes, rot)
        if img is None:
            continue
        for psm in (7, 8):
            try:
                cfg = f"--psm {psm} -c tessedit_char_whitelist=0123456789."
                d = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
                words = [(t, float(cf)) for t, cf in zip(d["text"], d["conf"]) if t.strip() and float(cf) >= 0]
                if not words:
                    continue
                txt = "".join(w[0] for w in words).strip().strip(".")
                conf = sum(w[1] for w in words) / len(words)
                if txt:
                    res.append((txt, conf))
            except Exception as exc:                        # noqa: BLE001
                logger.debug(f"[Section] tesseract failed: {exc}")
                return res
    return res


# ═════════════════════════════════ main entry ════════════════════════════════

def read_section(path: str | Path, tesseract_cmd: str | None = None) -> SectionResult:
    import ezdxf

    res = SectionResult()
    try:
        doc = ezdxf.readfile(str(path))
    except Exception as exc:                                # noqa: BLE001
        res.reason = f"cannot read DXF: {exc}"
        return res

    strokes: list[Stroke] = []
    texts: list[TextItem] = []
    _collect(doc, doc.modelspace(), strokes, texts)
    strokes = [s for s in strokes if s.length > 1e-6]
    if len(strokes) < 8:
        res.reason = "too few strokes"
        return res

    allx = [p[0] for s in strokes for p in s.pts]
    ally = [p[1] for s in strokes for p in s.pts]
    diag = math.hypot(max(allx) - min(allx), max(ally) - min(ally))

    # ── colour classes ───────────────────────────────────────────────────────
    by_color_len = Counter()
    for s in strokes:
        by_color_len[s.color] += s.length
    geom_color = 7 if by_color_len.get(7, 0) > 0 else by_color_len.most_common(1)[0][0]
    geom_all = [s for s in strokes if s.color == geom_color]
    other = [s for s in strokes if s.color != geom_color]

    # ── glyphs vs geometry : small touching components are text ──────────────
    touch_tol = 0.0007 * diag
    glyph_limit = 0.022 * diag
    comps = _touching_components(geom_all, touch_tol)
    geom: list[Stroke] = []
    glyphs: list[Stroke] = []
    for c in comps:
        xs = [p[0] for i in c for p in geom_all[i].pts]
        ys = [p[1] for i in c for p in geom_all[i].pts]
        ext = max(max(xs) - min(xs), max(ys) - min(ys))
        (geom if ext > glyph_limit else glyphs).extend(geom_all[i] for i in c)
    if not geom:
        res.reason = "no geometry strokes"
        return res

    # ── axis of revolution ──────────────────────────────────────────────────
    gx0 = min(p[0] for s in geom for p in s.pts)
    gx1 = max(p[0] for s in geom for p in s.pts)
    gy0 = min(p[1] for s in geom for p in s.pts)
    gy1 = max(p[1] for s in geom for p in s.pts)
    axis = _find_axis(other, (gx0, gy0, gx1, gy1))
    axis_color = None
    if axis is None:
        # fall back: assume horizontal axis at the mid-height of the geometry
        axis = ("h", (gy0 + gy1) / 2.0, None)
        res.warnings.append("No centre line found - axis assumed at mid-height of the geometry.")
    orient, y0, axis_color = axis

    if orient == "v":                      # make the axis horizontal by swapping x/y
        def sw(s: Stroke) -> Stroke:
            return Stroke([(p[1], p[0]) for p in s.pts], s.color, s.src)
        strokes = [sw(s) for s in strokes]
        geom = [sw(s) for s in geom]
        glyphs = [sw(s) for s in glyphs]
        other = [sw(s) for s in other]
        texts = [TextItem(t.text, t.cy, t.cx, not t.horizontal, t.h, t.source, t.conf) for t in texts]
        gx0, gx1, gy0, gy1 = gy0, gy1, gx0, gx1
        res.axis_horizontal = False
    res.axis_coord = y0
    res.debug_strokes = strokes

    # ── bridge small gaps in the geometry (PDF-to-CAD conversion cuts curves) ─
    geom = _bridge_gaps(geom, tol=0.0020 * diag)

    # ── scale from dimensions ───────────────────────────────────────────────
    other_dims = [s for s in other if s.color != axis_color]
    _recover_scale(res, glyphs, texts, other_dims, diag, tesseract_cmd, (gx0, gy0, gx1, gy1), y0)

    S = res.scale_mm_per_unit
    z_origin = min(p[0] for s in geom for p in s.pts)
    res.z_origin = z_origin

    # ── body cross-section by even-odd ray casting from the axis ────────────
    half = _pick_half(geom, y0)
    body_info = _raycast_body(geom, y0, half)
    if body_info is None:
        res.reason = "could not build the wall cross-section"
        return res
    slabs, odd_slabs = body_info
    if odd_slabs:
        res.warnings.append(f"{odd_slabs} cross-section slice(s) had an odd number of boundary crossings.")

    body_paper, cavity_paper = _slabs_to_polygons(slabs)
    res.body = _scale_geom(body_paper, z_origin, S)
    res.cavity = _scale_geom(cavity_paper, z_origin, S)

    # ── metrics + segments ──────────────────────────────────────────────────
    res.total_length_mm = (gx1 - z_origin) * S
    _segment_profile(res, slabs, z_origin, S)
    if not res.segments:
        res.reason = "no segments recovered"
        return res
    res.max_outer_radius_mm = max(max(s.r_out0, s.r_out1) for s in res.segments)

    cyl = [s for s in res.segments if s.kind == "cylinder"]
    if cyl:
        main = max(cyl, key=lambda s: (s.z1 - s.z0))
        res.outer_diameter_mm = 2 * main.r_out0
        res.inner_diameter_mm = 2 * main.r_in0
        res.shell_thickness_mm = main.r_out0 - main.r_in0

    # ── nozzles ─────────────────────────────────────────────────────────────
    try:
        res.nozzles = _detect_nozzles(geom, other, axis_color, y0, half, z_origin, S, diag)
    except Exception as exc:                                # noqa: BLE001
        logger.warning(f"[Section] nozzle detection failed: {exc}")
        res.warnings.append(f"Nozzle detection failed: {exc}")

    # ── confidence ──────────────────────────────────────────────────────────
    conf = 0.55 + 0.35 * res.scale_confidence
    if odd_slabs:
        conf -= 0.1
    res.confidence = max(0.1, min(0.98, conf))
    res.texts = [d.text for d in res.dims]
    res.ok = True
    return res


# ═════════════════════════════════ helpers ═══════════════════════════════════

def _find_axis(other: list[Stroke], geom_bbox):
    gx0, gy0, gx1, gy1 = geom_bbox
    groups: dict[tuple[str, float, int], list[tuple[float, float]]] = defaultdict(list)
    for s in other:
        if len(s.pts) != 2:
            continue
        (x1, y1), (x2, y2) = s.pts
        if abs(y1 - y2) < 1e-3 * max(1.0, abs(x2 - x1)) and abs(x2 - x1) > 0:
            groups[("h", round(y1, 1), s.color)].append((min(x1, x2), max(x1, x2)))
        elif abs(x1 - x2) < 1e-3 * max(1.0, abs(y2 - y1)) and abs(y2 - y1) > 0:
            groups[("v", round(x1, 1), s.color)].append((min(y1, y2), max(y1, y2)))
    best, best_score = None, 0.0
    for (o, c, col), spans in groups.items():
        lo = min(a for a, _ in spans)
        hi = max(b for _, b in spans)
        total = sum(b - a for a, b in spans)
        if o == "h":
            ext, cross_lo, cross_hi = gx1 - gx0, gy0, gy1
        else:
            ext, cross_lo, cross_hi = gy1 - gy0, gx0, gx1
        if not (cross_lo < c < cross_hi) or len(spans) < 2 and total < 0.5 * ext:
            continue
        span = hi - lo
        if span < 0.6 * ext:
            continue
        mid_off = abs(c - (cross_lo + cross_hi) / 2.0) / max(cross_hi - cross_lo, 1e-9)
        score = (total / ext) * (1.0 - min(mid_off, 0.9))
        if score > best_score:
            best, best_score = (o, c, col), score
    return best


def _bridge_gaps(geom: list[Stroke], tol: float) -> list[Stroke]:
    """Join dangling end-points that are nearly coincident (mutual nearest)."""
    ends = []
    for i, s in enumerate(geom):
        ends.append((s.pts[0], i, 0))
        ends.append((s.pts[-1], i, 1))
    # degree of each endpoint = how many other strokes touch it
    def touches(p, skip):
        for j, s in enumerate(geom):
            if j == skip:
                continue
            if _pt_poly_dist(p, s.pts) < 1e-3:
                return True
        return False

    dangling = [(p, i, k) for (p, i, k) in ends if not touches(p, i)]
    used = set()
    out = list(geom)
    for a, (pa, ia, ka) in enumerate(dangling):
        best = None
        for b, (pb, ib, kb) in enumerate(dangling):
            if a == b or ia == ib:
                continue
            d = math.dist(pa, pb)
            if d < tol and (best is None or d < best[0]):
                best = (d, b)
        if best is None:
            continue
        b = best[1]
        pb, ib, kb = dangling[b]
        # mutual nearest?
        mb = min(((math.dist(pb, q), c) for c, (q, ic, kc) in enumerate(dangling) if c != b and ic != ib),
                 default=(1e9, None))
        if mb[1] != a or (min(a, b), max(a, b)) in used:
            continue
        used.add((min(a, b), max(a, b)))
        out.append(Stroke([pa, pb], geom[ia].color, "BRIDGE"))
    return out


def _pick_half(geom: list[Stroke], y0: float) -> int:
    """+1 = upper half is the cleaner (body) half, -1 = lower half."""
    up = sum(s.length for s in geom if all(p[1] >= y0 - 1e-6 for p in s.pts))
    lo = sum(s.length for s in geom if all(p[1] <= y0 + 1e-6 for p in s.pts))
    return +1 if up <= lo else -1


def _raycast_body(geom: list[Stroke], y0: float, half: int):
    segs = []
    seen = set()
    for si, s in enumerate(geom):
        for a, b in zip(s.pts[:-1], s.pts[1:]):
            x1, r1 = a[0], (a[1] - y0) * half
            x2, r2 = b[0], (b[1] - y0) * half
            if r1 < -1e-9 and r2 < -1e-9:
                continue
            if abs(x2 - x1) < 1e-9:
                continue
            if r1 < 0 or r2 < 0:
                t = (0.0 - r1) / (r2 - r1)
                xm = x1 + t * (x2 - x1)
                if r1 < 0:
                    x1, r1 = xm, 0.0
                else:
                    x2, r2 = xm, 0.0
                if abs(x2 - x1) < 1e-9:
                    continue
            if x1 > x2:
                x1, r1, x2, r2 = x2, r2, x1, r1
            key = (round(x1, 3), round(r1, 3), round(x2, 3), round(r2, 3))
            if key in seen:
                continue
            seen.add(key)
            segs.append((x1, r1, x2, r2, si))
    if not segs:
        return None
    arr = np.array([(a, b, c, d) for a, b, c, d, _ in segs], dtype=float)
    ids = [s[4] for s in segs]
    xs = sorted({round(v, 6) for v in np.concatenate([arr[:, 0], arr[:, 2]])})
    slabs = []
    odd = 0
    for xa, xb in zip(xs[:-1], xs[1:]):
        if xb - xa < 1e-6:
            continue
        mask = (arr[:, 0] <= xa + 1e-7) & (arr[:, 2] >= xb - 1e-7)
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            continue
        cr = []
        for k in idx:
            x1, r1, x2, r2 = arr[k]
            ra = r1 + (r2 - r1) * (xa - x1) / (x2 - x1)
            rb = r1 + (r2 - r1) * (xb - x1) / (x2 - x1)
            cr.append((ra, rb, ids[k]))
        cr.sort(key=lambda t: t[0] + t[1])
        pre_zero = False
        if len(cr) % 2 == 1:
            cr.insert(0, (0.0, 0.0, None))
            pre_zero = True
            # a leading zero is only legitimate at the dome apex; count it for diagnostics
            odd += 1 if xa > xs[0] + 0.05 * (xs[-1] - xs[0]) else 0
        slabs.append({"xa": xa, "xb": xb, "cr": cr, "pre_zero": pre_zero})
    return slabs, odd


def _slabs_to_polygons(slabs):
    body_polys, cav_polys = [], []
    for sl in slabs:
        cr = sl["cr"]
        xa, xb = sl["xa"], sl["xb"]
        for k in range(0, len(cr) - 1, 2):
            lo, hi = cr[k], cr[k + 1]
            p = Polygon([(xa, lo[0]), (xb, lo[1]), (xb, hi[1]), (xa, hi[0])])
            if p.area > 1e-12:
                body_polys.append(p if p.is_valid else p.buffer(0))
        if not sl["pre_zero"] and cr:
            c = Polygon([(xa, 0.0), (xb, 0.0), (xb, cr[0][1]), (xa, cr[0][0])])
            if c.area > 1e-12:
                cav_polys.append(c if c.is_valid else c.buffer(0))
    body = unary_union(body_polys).buffer(0) if body_polys else Polygon()
    cav = unary_union(cav_polys).buffer(0) if cav_polys else Polygon()
    return body, cav


def _scale_geom(g, z_origin: float, S: float):
    from shapely import affinity
    if g is None or g.is_empty:
        return g
    g = affinity.translate(g, xoff=-z_origin)
    return affinity.scale(g, xfact=S, yfact=S, origin=(0, 0))


# ─────────────────────────── scale recovery from dimensions ──────────────────

def _group_strings(glyphs: list[Stroke], tol_glyph_comp: float, diag: float):
    comps = _touching_components(glyphs, tol_glyph_comp) if glyphs else []
    cb = []
    for c in comps:
        xs = [p[0] for i in c for p in glyphs[i].pts]
        ys = [p[1] for i in c for p in glyphs[i].pts]
        cb.append((min(xs), min(ys), max(xs), max(ys), c))
    if not cb:
        return []
    hs = sorted(max(b[2] - b[0], b[3] - b[1]) for b in cb)
    h_med = hs[int(len(hs) * 0.75)] if hs else 1.0
    gap = 0.50 * h_med
    n = len(cb)
    par = list(range(n))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            dx = max(0.0, max(cb[i][0], cb[j][0]) - min(cb[i][2], cb[j][2]))
            dy = max(0.0, max(cb[i][1], cb[j][1]) - min(cb[i][3], cb[j][3]))
            if math.hypot(dx, dy) <= gap:
                par[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for g in groups.values():
        idx = [k for i in g for k in cb[i][4]]
        sts = [glyphs[k] for k in idx]
        xs = [p[0] for s in sts for p in s.pts]
        ys = [p[1] for s in sts for p in s.pts]
        out.append((sts, (min(xs), min(ys), max(xs), max(ys))))
    return out


def _recover_scale(res: SectionResult, glyphs, texts, other_dims, diag, tesseract_cmd, geom_bbox, y0):
    items: list[TextItem] = []

    # text entities (real CAD text) -------------------------------------------
    for t in texts:
        s = t.text.replace(",", "").replace("Ø", "").replace("ø", "").replace("⌀", "").strip()
        if _NUM_RE.match(s):
            t2 = TextItem(t.text, t.cx, t.cy, t.horizontal, t.h, "text-entity", 1.0, [float(s)])
            items.append(t2)

    # exploded text (glyph polylines) -> OCR ----------------------------------
    for sts, bb in _group_strings(glyphs, 0.0007 * diag, diag):
        w, h = bb[2] - bb[0], bb[3] - bb[1]
        if len(sts) < 1 or max(w, h) < 0.004 * diag:
            continue
        horizontal = w >= h
        cands = _ocr_strokes(sts, horizontal, tesseract_cmd)
        values: list[float] = []
        best_conf = 0.0
        best_txt = ""
        for txt, conf in cands:
            if not _NUM_RE.match(txt):
                continue
            vals = [float(txt)]
            digits = txt.replace(".", "")
            if len(digits) >= 4 and txt[0] != "." and txt[0] in "0689":   # leading Ø read as a digit
                vals.append(float(txt[1:]))
            for v in vals:
                if v not in values:
                    values.append(v)
            if conf > best_conf:
                best_conf, best_txt = conf, txt
        if values:
            items.append(TextItem(best_txt, (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2,
                                  horizontal, min(w, h) if horizontal else min(w, h),
                                  "glyph-ocr", best_conf / 100.0, values))

    if not items:
        res.warnings.append("No readable dimension text - assuming DXF units are millimetres (scale 1.0).")
        return

    # horizontal / vertical dimension rows from 'other' strokes -------------------
    hor = [s for s in other_dims if len(s.pts) == 2 and abs(s.pts[0][1] - s.pts[1][1]) < 1e-3 * max(1, diag)]
    ver = [s for s in other_dims if len(s.pts) == 2 and abs(s.pts[0][0] - s.pts[1][0]) < 1e-3 * max(1, diag)]

    assoc = []        # (item, M_units, orientation)
    for it in items:
        if it.horizontal:      # reading along X: measures an axial length
            M = _bracket(it.cx, it.cy, it.h, hor, ver, along=0)
            ori = "axial"
        else:                  # vertical text: measures a radial/diametral length
            M = _bracket(it.cy, it.cx, it.h, ver, hor, along=1)
            ori = "radial"
        if M:
            assoc.append((it, M, ori))

    if not assoc:
        res.warnings.append("Dimension text found but could not be tied to extension lines - scale assumed 1.0.")
        return

    # consensus --------------------------------------------------------------
    cand = []   # (scale, weight, assoc_index, value)
    for k, (it, M, ori) in enumerate(assoc):
        logger.info(f"[Section] dim text '{it.text}' cands={it.candidates} M={M:.2f} ({ori})")
        for v in it.candidates:
            if v > 0 and M > 0:
                cand.append((v / M, M, k, v))
    if not cand:
        res.warnings.append("No usable dimension values - scale assumed 1.0.")
        return
    best = None
    for s0, w0, k0, v0 in cand:
        members = {}
        for s1, w1, k1, v1 in cand:
            if abs(s1 - s0) / s0 <= 0.012:
                if k1 not in members or abs(s1 - s0) < abs(members[k1][0] - s0):
                    members[k1] = (s1, w1, v1)
        weight = sum(m[1] for m in members.values())
        if best is None or (len(members), weight) > (len(best[0]), best[1]):
            best = (members, weight, s0)
    members, weight, s0 = best
    # scale from the largest measured length in the consensus (smallest relative error)
    kmax = max(members, key=lambda k: members[k][1])
    scale = members[kmax][2] / assoc[kmax][1]
    # cross-check estimate
    ratios = [m[0] for m in members.values()]
    spread = (max(ratios) - min(ratios)) / scale if len(ratios) > 1 else 0.0

    res.scale_mm_per_unit = scale
    res.scale_source = f"{len(members)} dimension(s) agree (OCR + extension lines)"
    res.scale_confidence = min(1.0, 0.45 + 0.18 * len(members)) * (1.0 - min(spread * 20, 0.5))

    for k, (it, M, ori) in enumerate(assoc):
        if k in members:
            val = members[k][2]
            ok = True
        else:
            val = M * scale
            ok = False
            res.warnings.append(f"Dimension '{it.text}' disagrees with the drawing scale (measured {M*scale:.1f}).")
        txt = f"{val:g}"
        res.dims.append(DimRec(text=txt, value_mm=val, measured_units=M, orientation=ori,
                               is_diameter=(ori == "radial"), consistent=ok, source=it.source))


def _bracket(c_along, c_cross, h, parallel, perpendicular, along: int):
    """
    Tie a text at (c_along, c_cross) to the pair of extension lines it measures.
    `parallel` are dimension-line strokes parallel to the reading direction,
    `perpendicular` are extension-line strokes. Returns measured length in drawing units.
    """
    # 1. dimension line row: nearest parallel stroke on the text's cross coordinate
    row = None
    for s in parallel:
        (x1, y1), (x2, y2) = s.pts
        a1, a2 = (x1, x2) if along == 0 else (y1, y2)
        cr = y1 if along == 0 else x1
        lo, hi = min(a1, a2), max(a1, a2)
        d = abs(cr - c_cross)
        if d > 3.5 * h + 1.0:
            continue
        # the row should extend near the text along its reading direction
        if lo - 3.0 * h <= c_along <= hi + 3.0 * h and (row is None or d < row[0]):
            row = (d, cr)
    if row is None:
        return None
    row_c = row[1]
    # 2. extension lines touching that row
    exts = []
    for s in perpendicular:
        (x1, y1), (x2, y2) = s.pts
        a = x1 if along == 0 else y1
        lo, hi = (min(y1, y2), max(y1, y2)) if along == 0 else (min(x1, x2), max(x1, x2))
        if lo - 0.9 <= row_c <= hi + 0.9:
            exts.append(a)
    exts = sorted({round(a, 2) for a in exts})
    below = [a for a in exts if a <= c_along]
    above = [a for a in exts if a >= c_along]
    if not below or not above:
        return None
    a, b = max(below), min(above)
    return (b - a) if b - a > 1e-6 else None


# ─────────────────────────────── profile segmentation ────────────────────────

def _segment_profile(res: SectionResult, slabs, z_origin: float, S: float):
    # envelope per slab: outermost crossing + the first (inner) boundary
    env = []
    for sl in slabs:
        cr = sl["cr"]
        first, last = cr[0], cr[-1]
        env.append({
            "za": (sl["xa"] - z_origin) * S, "zb": (sl["xb"] - z_origin) * S,
            "ro_a": last[0] * S, "ro_b": last[1] * S, "sid": last[2],
            "ri_a": first[0] * S, "ri_b": first[1] * S, "pre": sl["pre_zero"],
            "pairs": [(cr[k], cr[k + 1]) for k in range(0, len(cr) - 1, 2)],
        })
    if not env:
        return
    # group consecutive slabs by the stroke forming the outer boundary
    pieces = []
    for e in env:
        if pieces and pieces[-1][0] == e["sid"]:
            pieces[-1][1].append(e)
        else:
            pieces.append((e["sid"], [e]))

    total = env[-1]["zb"] - env[0]["za"]
    infos = []
    for sid, group in pieces:
        pts = [(group[0]["za"], group[0]["ro_a"])]
        for g in group:
            pts.append((g["zb"], g["ro_b"]))
        infos.append({"pts": pts, "group": group})

    def classify(pts):
        (z0, r0), (z1, r1) = pts[0], pts[-1]
        dz, dr = z1 - z0, r1 - r0
        L = math.hypot(dz, dr)
        if L < 1e-9:
            return "straight", 0.0
        dev = max(abs((p[1] - r0) * dz - (p[0] - z0) * dr) / L for p in pts)
        slope = math.degrees(math.atan2(dr, dz)) if dz > 1e-9 else 90.0 * (1 if dr >= 0 else -1)
        return ("straight" if dev <= 0.0016 * total + 2.5 else "curved"), slope

    for inf in infos:
        inf["type"], inf["slope"] = classify(inf["pts"])

    # absorb slivers into the longer neighbour
    thr = 0.003 * total
    changed = True
    while changed and len(infos) > 1:
        changed = False
        for i, inf in enumerate(infos):
            span = inf["pts"][-1][0] - inf["pts"][0][0]
            if span < thr:
                j = i - 1 if (i > 0 and (i == len(infos) - 1 or
                                          (infos[i - 1]["pts"][-1][0] - infos[i - 1]["pts"][0][0]) >=
                                          (infos[i + 1]["pts"][-1][0] - infos[i + 1]["pts"][0][0]))) else i + 1
                if j < 0 or j >= len(infos):
                    continue
                a, b = (infos[j], inf) if j < i else (inf, infos[j])
                merged = {"pts": a["pts"] + b["pts"], "group": a["group"] + b["group"]}
                merged["type"], merged["slope"] = classify(merged["pts"])
                infos[min(i, j):max(i, j) + 1] = [merged]
                changed = True
                break

    # merge compatible neighbours
    def mergeable(a, b):
        if a["type"] == "curved" and b["type"] == "curved":
            return True
        if a["type"] == "straight" and b["type"] == "straight":
            return abs(a["slope"] - b["slope"]) < 1.5 and \
                abs(a["pts"][-1][1] - b["pts"][0][1]) < 0.002 * total + 3
        return False

    merged_infos = [infos[0]]
    for inf in infos[1:]:
        if mergeable(merged_infos[-1], inf):
            m = merged_infos[-1]
            m["pts"] = m["pts"] + inf["pts"]
            m["group"] = m["group"] + inf["group"]
            m["type"], m["slope"] = classify(m["pts"])
            if m["type"] == "straight" and inf["type"] == "curved":
                m["type"] = "curved"
        else:
            merged_infos.append(inf)

    segs: list[SegmentRec] = []
    for inf in merged_infos:
        pts = inf["pts"]
        z0, z1 = pts[0][0], pts[-1][0]
        ro0, ro1 = pts[0][1], pts[-1][1]
        g0, g1 = inf["group"][0], inf["group"][-1]
        ri0 = g0["ri_a"] if not g0["pre"] else 0.0
        ri1 = g1["ri_b"] if not g1["pre"] else 0.0
        if inf["type"] == "curved":
            kind = "dome" if (z0 < 0.02 * total and not segs) else "transition"
        else:
            kind = "cylinder" if abs(inf["slope"]) < 1.5 else "cone"
        segs.append(SegmentRec(kind, "", z0, z1, ro0, ro1, ri0, ri1))

    # labels
    cyl = [s for s in segs if s.kind == "cylinder"]
    main = max(cyl, key=lambda s: s.z1 - s.z0) if cyl else None
    n_cone = 0
    for s in segs:
        if s.kind == "dome":
            s.label = "Dished head"
        elif s.kind == "cone":
            n_cone += 1
            s.label = "Cone"
        elif s.kind == "cylinder":
            if s is main:
                s.label = "Cylindrical shell"
            elif main and max(s.r_out0, s.r_out1) > main.r_out0 * 1.02:
                s.label = "End flange"
            else:
                s.label = "End collar"
        else:
            s.label = "Transition"
    if n_cone == 2:
        cones = [s for s in segs if s.kind == "cone"]
        cones[0].label, cones[1].label = "Left cone", "Right cone"
    res.segments = segs


# ──────────────────────────────── nozzles ────────────────────────────────────

def _detect_nozzles(geom, other, axis_color, y0, half, z_origin, S, diag) -> list[NozzleRec]:
    # tilted centre lines of the same colour as the vessel axis
    tilted = []
    for s in other:
        if s.color != axis_color or len(s.pts) != 2:
            continue
        (x1, y1), (x2, y2) = s.pts
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy)
        if L < 1e-9:
            continue
        ang = abs(math.degrees(math.atan2(dy, dx)))
        ang = min(ang, 180 - ang)                 # 0..90 from the horizontal axis
        if 3.0 < ang < 87.0:
            tilted.append((s, ang))
    if not tilted:
        return []

    # cluster by direction + offset
    lines = []
    for s, ang in tilted:
        (x1, y1), (x2, y2) = s.pts
        d = np.array([x2 - x1, y2 - y1], float)
        d /= np.linalg.norm(d)
        if d[1] < 0 or (d[1] == 0 and d[0] < 0):
            d = -d
        nrm = np.array([-d[1], d[0]])
        off = float(nrm @ np.array([x1, y1]))
        for ln in lines:
            if abs(ln["d"] @ d) > math.cos(math.radians(1.5)) and abs(ln["off"] - float(ln["n"] @ np.array([x1, y1]))) < 0.0035 * diag:
                ln["pts"].extend([s.pts[0], s.pts[1]])
                break
        else:
            lines.append({"d": d, "n": nrm, "off": off, "pts": [s.pts[0], s.pts[1]]})

    noz = []
    for idx, ln in enumerate(lines):
        pts = np.array(ln["pts"], float)
        c = pts.mean(axis=0)
        u, s_, vt = np.linalg.svd(pts - c)
        d = vt[0]
        if abs(d[1]) < 1e-9:
            continue
        # intersection with the vessel axis y = y0
        t0 = (y0 - c[1]) / d[1]
        P0 = c + t0 * d
        # orient d outward (towards the farthest end point from P0)
        far = pts[np.argmax(np.linalg.norm(pts - P0, axis=1))]
        if (far - P0) @ d < 0:
            d = -d
        side = 1 if d[1] > 0 else -1
        if side != -half and False:
            pass
        n_vec = np.array([-d[1], d[0]])
        # tilt from the radial direction
        tilt = math.degrees(math.acos(min(1.0, abs(d[1]))))

        def uv(p):
            q = np.array(p) - P0
            return float(q @ d), float(q @ n_vec)

        par_v, par_u, perp_u, perp_v = [], [], [], []
        for s in geom:
            if s.src == "BRIDGE" or len(s.pts) < 2:
                continue
            ys = [p[1] for p in s.pts]
            if side > 0 and min(ys) < y0 - 1e-6:
                continue
            if side < 0 and max(ys) > y0 + 1e-6:
                continue
            (x1, y1), (x2, y2) = s.pts[0], s.pts[-1]
            v = np.array([x2 - x1, y2 - y1], float)
            L = np.linalg.norm(v)
            if L < 0.008 * diag:
                if L > 0:
                    pass
                else:
                    continue
            cosang = abs(float(v @ d)) / max(L, 1e-12)
            a = uv(s.pts[0])
            b = uv(s.pts[-1])
            if cosang > math.cos(math.radians(2.5)) and L >= 0.008 * diag:
                par_v.append((abs((a[1] + b[1]) / 2), min(a[0], b[0]), max(a[0], b[0])))
            elif cosang < math.sin(math.radians(2.5)) and L >= 0.008 * diag:
                perp_u.append((a[0] + b[0]) / 2)
                perp_v.extend([abs(a[1]), abs(b[1])])
        if not par_v:
            continue

        def cluster(vals, tol):
            vals = sorted(vals)
            out = []
            for v in vals:
                if out and v - out[-1][-1] <= tol:
                    out[-1].append(v)
                else:
                    out.append([v])
            return [sum(c) / len(c) for c in out]

        tol_c = 0.0020 * diag
        vcl = cluster([p[0] for p in par_v], tol_c)
        r_in = vcl[0]
        r_out = vcl[1] if len(vcl) > 1 else vcl[0]
        # pipe wall starts at the smallest u among parallel strokes
        u_start = min(p[1] for p in par_v)
        ucl = cluster(perp_u, tol_c) if perp_u else []
        ucl = [u for u in ucl if u > u_start]
        if ucl:
            fu0, fu1 = ucl[0], ucl[-1]
            fr = max(perp_v) if perp_v else r_out
        else:
            fu0 = fu1 = max(p[2] for p in par_v)
            fr = r_out
        noz.append(NozzleRec(
            label=f"Nozzle N{idx + 1}",
            axis_z_mm=float((P0[0] - z_origin) * S),
            tilt_deg=float(tilt),
            direction=(float(d[0]), float(d[1])),          # (axial, paper-up) unit vector
            r_in_mm=float(r_in * S), r_out_mm=float(r_out * S),
            u_start_mm=float(u_start * S),
            flange_u0_mm=float(fu0 * S), flange_u1_mm=float(fu1 * S),
            flange_r_mm=float(fr * S),
            radial_side=side,
        ))
    return noz
