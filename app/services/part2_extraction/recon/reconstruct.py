"""
app/services/part2_extraction/recon/reconstruct.py
==================================================
Scene  →  axisymmetric solid cross-sections (exact) + features + dimension checks.

Idea
----
A pressure vessel / converter drawing is a *section through the axis of
revolution*.  Instead of guessing "cylinder + cone + dome" we recover the
**exact planar faces** drawn in the section:

 1. find the axis of revolution (centre-line, or mirror symmetry as fallback)
 2. express every geometry curve in axial/radial coordinates  (z, r)  [mm]
 3. node + close gaps + polygonize the line work  →  planar faces
 4. the face touching the axis is the *cavity*; connected thin faces are the
    *solid* cross-section (shell wall, flanges, end plates, linings …)
 5. the half that is cleaner is the revolve body; whatever the other half has in
    addition (tap-hole spout …) becomes a **feature** with its own axis
 6. each face boundary is segmented into exact line / arc elements
 7. every dimension is checked against the recovered geometry

The result is handed to ``mesh_builder`` which revolves the faces with CSG.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from loguru import logger
from shapely import affinity
from shapely.geometry import (GeometryCollection, LineString, MultiLineString, MultiPolygon,
                              Point, Polygon, box)
from shapely.ops import linemerge, nearest_points, polygonize_full, unary_union
from shapely.strtree import STRtree

from .scene import DimRec, Scene

Pt = tuple[float, float]


# ═══════════════════════════════ data classes ════════════════════════════════

@dataclass
class Element:
    """One exact boundary element in (z, r) millimetres."""
    kind: str                       # line | arc
    p0: Pt
    p1: Pt
    center: Pt | None = None
    radius: float | None = None
    ccw: bool | None = None
    side: str = "outer"             # outer | inner
    label: str = ""


@dataclass
class SubFeature:
    """Axisymmetric add-on with its own axis (tap-hole spout, nozzle …)."""
    label: str
    origin: Pt                      # on the main axis  (z, r=0)
    direction: Pt                   # unit vector (dz, dr), pointing away from the main axis
    tilt_deg: float                 # angle between feature axis and the main-axis normal
    side: int                       # +1 / -1 : half of the section the feature is drawn in
    profile_sr: list[list[Pt]] = field(default_factory=list)   # polygon rings in (s, rho>=0)
    holes_sr: list[list[list[Pt]]] = field(default_factory=list)
    bore_radius: float = 0.0
    outer_radius: float = 0.0
    s_end: float = 0.0
    plate_radius: float = 0.0
    plate_thickness: float = 0.0
    symmetry_error_pct: float = 0.0


@dataclass
class DimCheck:
    id: str
    kind: str
    stated: float
    measured: float | None
    deviation: float | None
    status: str                     # ok | endpoint_off_geometry | override | corrected | measured
    note: str = ""


@dataclass
class Recon:
    ok: bool = False
    axis_source: str = ""
    axis_origin: Pt = (0.0, 0.0)
    axis_dir: Pt = (1.0, 0.0)
    flipped: bool = False
    length_mm: float = 0.0
    max_radius_mm: float = 0.0
    main_polys: list[Polygon] = field(default_factory=list)     # (z, r>=0) solid section
    cavity: Polygon | MultiPolygon | None = None
    features: list[SubFeature] = field(default_factory=list)
    outer_chains: list[list[Element]] = field(default_factory=list)
    inner_chains: list[list[Element]] = field(default_factory=list)
    main_chain: list[Element] = field(default_factory=list)     # outer chain from the closed end
    wall_thickness_mm: float | None = None
    dim_checks: list[DimCheck] = field(default_factory=list)
    symmetry_error_pct: float = 0.0
    accuracy_mm: float = 0.05
    warnings: list[str] = field(default_factory=list)
    half_used: str = "upper"
    # geometry in scene coordinates helpers (for UI overlays)
    frame: "Frame | None" = None


# ═══════════════════════════════ coordinate frame ════════════════════════════

class Frame:
    """scene (x, y)  →  (z along axis, r signed radial)   in millimetres."""

    def __init__(self, origin: Pt, u: Pt, k: float, flip: bool = False, z0: float = 0.0):
        self.O = origin
        self.u = u
        n = (-u[1], u[0])
        if n[1] < -1e-9 or (abs(n[1]) <= 1e-9 and n[0] < 0):   # "r > 0" = visually up
            n = (-n[0], -n[1])
        self.n = n
        self.k = k
        self.flip = flip
        self.z0 = z0

    def __call__(self, p: Pt) -> Pt:
        vx, vy = p[0] - self.O[0], p[1] - self.O[1]
        s = vx * self.u[0] + vy * self.u[1]
        t = vx * self.n[0] + vy * self.n[1]
        z = (-s if self.flip else s) * self.k - self.z0
        return (z, t * self.k)

    def inverse(self, zr: Pt) -> Pt:
        z, r = zr
        s = -(z + self.z0) / self.k if self.flip else (z + self.z0) / self.k
        t = r / self.k
        return (self.O[0] + s * self.u[0] + t * self.n[0],
                self.O[1] + s * self.u[1] + t * self.n[1])


# ═══════════════════════════════ axis detection ══════════════════════════════

def _merged_centerlines(scene: Scene, ext: float) -> list[tuple[Pt, Pt]]:
    segs: list[tuple[float, float, float, float, Pt, Pt]] = []
    for p in scene.centerlines():
        a, b = p.pts[0], p.pts[-1]
        if math.dist(a, b) < 1e-9:
            continue
        th = math.atan2(b[1] - a[1], b[0] - a[0]) % math.pi
        c = -a[0] * math.sin(th) + a[1] * math.cos(th)
        segs.append((th, c, 0.0, 0.0, a, b))
    groups: list[list[tuple]] = []
    for s in segs:
        for g in groups:
            dth = abs(g[0][0] - s[0])
            dth = min(dth, math.pi - dth)
            if dth < 0.012 and abs(g[0][1] - s[1]) < 0.004 * ext:
                g.append(s)
                break
        else:
            groups.append([s])
    out = []
    for g in groups:
        th = g[0][0]
        d = (math.cos(th), math.sin(th))
        pts = [q for s in g for q in (s[4], s[5])]
        prm = [q[0] * d[0] + q[1] * d[1] for q in pts]
        a = pts[int(np.argmin(prm))]
        b = pts[int(np.argmax(prm))]
        out.append((a, b))
    out.sort(key=lambda ab: -math.dist(*ab))
    return out


def _symmetry_axis(geom_pts: np.ndarray, ext: float):
    from scipy.spatial import cKDTree
    x0, y0 = geom_pts.min(0)
    x1, y1 = geom_pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tree = cKDTree(geom_pts)
    best = (-1.0, None)
    for name, mirror, a, b in (
        ("h", lambda P: np.c_[P[:, 0], 2 * cy - P[:, 1]], (x0, cy), (x1, cy)),
        ("v", lambda P: np.c_[2 * cx - P[:, 0], P[:, 1]], (cx, y0), (cx, y1)),
    ):
        d, _ = tree.query(mirror(geom_pts))
        score = float(np.mean(d < 0.006 * ext))
        if score > best[0]:
            best = (score, (a, b))
    return best


def find_axis(scene: Scene):
    bb = scene.bbox((("geometry"),))
    ext = max(bb[2] - bb[0], bb[3] - bb[1])
    cls = _merged_centerlines(scene, ext)
    if cls and math.dist(*cls[0]) >= 0.35 * ext:
        main, subs = cls[0], [c for c in cls[1:] if math.dist(*c) >= 0.015 * ext]
        return main, subs, "centerline"
    pts = np.array([q for p in scene.geometry() for q in p.pts], dtype=float)
    score, ab = _symmetry_axis(pts, ext)
    if score >= 0.55:
        return ab, [c for c in cls if math.dist(*c) >= 0.015 * ext], f"symmetry({score:.2f})"
    # fallback: the long side of the bounding box through its centre
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    if (bb[2] - bb[0]) >= (bb[3] - bb[1]):
        ab = ((bb[0], cy), (bb[2], cy))
    else:
        ab = ((cx, bb[1]), (cx, bb[3]))
    return ab, [], "bbox"


# ═══════════════════════════════ helpers ═════════════════════════════════════

def _polys(g) -> list[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, (MultiPolygon, GeometryCollection)):
        out = []
        for h in g.geoms:
            out += _polys(h)
        return out
    return []


def _lines(g) -> list[LineString]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    if hasattr(g, "geoms"):
        out = []
        for h in g.geoms:
            out += _lines(h)
        return out
    return []


def _nverts(g) -> int:
    return sum(len(p.exterior.coords) + sum(len(i.coords) for i in p.interiors) for p in _polys(g))


def _mirror(g):
    return affinity.scale(g, xfact=1.0, yfact=-1.0, origin=(0, 0))


def _close_gaps_and_polygonize(lines: list[LineString], tol: float):
    merged = unary_union(lines)
    edges = _lines(merged)
    q = max(tol * 0.01, 1e-9)

    def key(c):
        return (round(c[0] / q), round(c[1] / q))

    cnt: Counter = Counter()
    for e in edges:
        cnt[key(e.coords[0])] += 1
        cnt[key(e.coords[-1])] += 1
    dang = []
    for e in edges:
        for c in (e.coords[0], e.coords[-1]):
            if cnt[key(c)] == 1:
                dang.append((c, e))
    add: list[LineString] = []
    if dang and edges:
        tree = STRtree(edges)
        for c, own in dang:
            P = Point(c)
            best = None
            for i in tree.query(P.buffer(tol)):
                e = edges[int(i)]
                if e is own:
                    continue
                d = e.distance(P)
                if d <= tol and (best is None or d < best[0]):
                    best = (d, e)
            if best is not None and best[0] > 1e-9:
                qpt = nearest_points(best[1], P)[0]
                add.append(LineString([c, (qpt.x, qpt.y)]))
    if add:
        merged = unary_union(edges + add)
    polys, _cuts, _dang, _inv = polygonize_full(merged)
    return _polys(polys), len(add)


def _fit_circle(pts):
    P = np.asarray(pts, dtype=float)
    m = P.mean(0)
    Q = P - m
    x, y = Q[:, 0], Q[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones(len(Q))]
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx + m[0], cy + m[1], r, float(res.max())


def _turn_deg(a: Pt, b: Pt, c: Pt) -> float:
    v1 = (b[0] - a[0], b[1] - a[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    return math.degrees(math.atan2(v1[0] * v2[1] - v1[1] * v2[0],
                                   v1[0] * v2[0] + v1[1] * v2[1]))


def segment_chain(pts: list[Pt], tol: float, corner_deg: float = 25.0) -> list[Element]:
    """Segment a polyline into exact line / arc elements."""
    P: list[Pt] = [pts[0]]
    for q in pts[1:]:
        if math.dist(q, P[-1]) > 1e-9:
            P.append(q)
    n = len(P)
    if n < 2:
        return []
    br = [0] + [i for i in range(1, n - 1) if abs(_turn_deg(P[i - 1], P[i], P[i + 1])) >= corner_deg] + [n - 1]
    out: list[Element] = []
    for a, b in zip(br[:-1], br[1:]):
        out += _fit_run(P[a:b + 1], tol)
    return out


def _fit_run(run: list[Pt], tol: float) -> list[Element]:
    p0, pn = run[0], run[-1]
    if len(run) == 2:
        return [Element("line", p0, pn)]
    d = np.array(pn) - np.array(p0)
    L = float(np.hypot(*d))
    arr = np.asarray(run)
    if L < 1e-12:
        dev = np.hypot(*(arr - np.array(p0)).T)
    else:
        dev = np.abs(np.cross(d, arr - np.array(p0))) / L
    if dev.max() <= tol:
        return [Element("line", p0, pn)]
    if len(run) >= 5:
        turns = [_turn_deg(run[i - 1], run[i], run[i + 1]) for i in range(1, len(run) - 1)]
        same_sign = all(t > 0 for t in turns) or all(t < 0 for t in turns)
        if same_sign:
            cx, cy, r, res = _fit_circle(run)
            sweep = sum(abs(t) for t in turns)
            if res <= max(tol, 2e-4 * r) and sweep < 340 and r < 1e7:
                return [Element("arc", p0, pn, (cx, cy), r, ccw=turns[0] > 0)]
    k = int(np.argmax(dev))
    if k in (0, len(run) - 1):
        return [Element("line", p0, pn)]
    return _fit_run(run[:k + 1], tol) + _fit_run(run[k:], tol)


def _chains_from_lines(geom, tol: float, side: str) -> list[list[Element]]:
    out = []
    for ls in _lines(linemerge(geom) if not isinstance(geom, LineString) else geom):
        pts = [(float(x), float(y)) for x, y in ls.coords]
        if len(pts) < 2:
            continue
        if pts[0][0] > pts[-1][0]:
            pts.reverse()
        els = segment_chain(pts, tol)
        for e in els:
            e.side = side
        if els:
            out.append(els)
    out.sort(key=lambda c: c[0].p0[0])
    return out


# ═══════════════════════════════ main entry ══════════════════════════════════

def reconstruct(scene: Scene) -> Recon:
    rec = Recon(warnings=list(scene.warnings))
    geo = scene.geometry()
    if not geo:
        rec.warnings.append("No geometry in scene.")
        return rec
    k = scene.mm_per_unit
    bb = scene.bbox(("geometry",))
    ext_u = max(bb[2] - bb[0], bb[3] - bb[1])

    # ── 1. axis ──────────────────────────────────────────────────────────────
    (A, B), subs, src = find_axis(scene)
    u = (B[0] - A[0], B[1] - A[1])
    ul = math.hypot(*u)
    u = (u[0] / ul, u[1] / ul)
    if u[0] < -1e-9 or (abs(u[0]) <= 1e-9 and u[1] < 0):
        u = (-u[0], -u[1])
    rec.axis_source = src
    frame = Frame(A, u, k)

    # ── 2. geometry in (z, r) ────────────────────────────────────────────────
    def to_lines(fr: Frame) -> list[LineString]:
        out = []
        for p in geo:
            pts = [fr(q) for q in p.pts]
            if len(pts) >= 2 and LineString(pts).length > 1e-9:
                out.append(LineString(pts))
        return out

    L0 = to_lines(frame)
    allg = unary_union(L0)
    zmin, rmin, zmax, rmax = allg.bounds
    Lz = zmax - zmin
    w = max(0.004 * Lz, 1e-6)
    r_lo = abs(max((_b for _b in (abs(v) for v in allg.intersection(box(zmin, rmin - 1, zmin + w, rmax + 1)).bounds)
                    if not math.isinf(_b)), default=0.0))
    r_hi = abs(max((_b for _b in (abs(v) for v in allg.intersection(box(zmax - w, rmin - 1, zmax, rmax + 1)).bounds)
                    if not math.isinf(_b)), default=0.0))
    flip = r_lo > 1.15 * r_hi
    frame = Frame(A, u, k, flip=flip, z0=0.0)
    L1 = to_lines(frame)
    g1 = unary_union(L1)
    zmin = g1.bounds[0]
    frame = Frame(A, u, k, flip=flip, z0=zmin)
    L1 = to_lines(frame)
    g1 = unary_union(L1)
    zmin, rmin, zmax, rmax = g1.bounds
    Lz = zmax - zmin
    rec.flipped, rec.length_mm, rec.frame = flip, Lz, frame
    rec.axis_origin, rec.axis_dir = A, u
    rec.max_radius_mm = max(abs(rmin), abs(rmax))
    tol = max(scene.tolerance_mm, 0.02)
    rec.accuracy_mm = scene.tolerance_mm
    gap_tol = max(tol * 2.0, 2e-4 * Lz) if scene.exact else max(tol * 2.5, 2e-3 * Lz)

    # ── 3. faces ─────────────────────────────────────────────────────────────
    axis_ls = LineString([(zmin - 1.0, 0.0), (zmax + 1.0, 0.0)])
    faces, n_closed = _close_gaps_and_polygonize(L1 + [axis_ls], gap_tol)
    if n_closed:
        rec.warnings.append(f"{n_closed} small gap(s) in the line work were closed "
                            f"(tolerance {gap_tol:.2f} mm).")
    if not faces:
        rec.warnings.append("Line work does not form closed regions – cannot build solid section.")
        return rec

    ax_len = axis_ls.length
    upper = [f for f in faces if f.representative_point().y > 0]
    lower = [f for f in faces if f.representative_point().y < 0]

    def pick_cavity(fs: list[Polygon]):
        if not fs:
            return []
        ov = [(f.boundary.intersection(axis_ls).length, f) for f in fs]
        mx = max(o for o, _ in ov)
        if mx < 0.15 * Lz:
            return []
        return [f for o, f in ov if o >= 0.15 * mx and o >= 0.02 * Lz]

    def solids_of(fs: list[Polygon], cav: list[Polygon]):
        t_max = 0.18 * rec.max_radius_mm
        rest = [f for f in fs if all(f is not c for c in cav)]
        thin = [f for f in rest if f.area > 1e-9 and 2 * f.area / max(f.length, 1e-9) <= t_max]
        if not cav:
            return thin
        kept: list[Polygon] = []
        frontier = list(cav)
        pool = list(thin)
        tol_s = gap_tol * 0.25
        changed = True
        while changed and pool:
            changed = False
            for f in list(pool):
                for c in frontier:
                    if f.boundary.intersection(c.boundary).length > tol_s:
                        kept.append(f)
                        frontier.append(f)
                        pool.remove(f)
                        changed = True
                        break
        return kept

    cav_u, cav_l = pick_cavity(upper), pick_cavity(lower)
    sol_u = solids_of(upper, cav_u)
    sol_l = solids_of(lower, cav_l)
    S_u = unary_union(sol_u).buffer(0) if sol_u else Polygon()
    S_l = unary_union(sol_l).buffer(0) if sol_l else Polygon()
    if S_u.is_empty and S_l.is_empty:
        rec.warnings.append("No solid wall cross-section could be identified.")
        return rec
    C_u = unary_union(cav_u).buffer(0) if cav_u else None
    C_l = unary_union(cav_l).buffer(0) if cav_l else None

    # ── 4. choose the revolve half ───────────────────────────────────────────
    if S_l.is_empty:
        main_sign = +1
    elif S_u.is_empty:
        main_sign = -1
    else:
        nu, nl = _nverts(S_u), _nverts(S_l)
        main_sign = +1 if nu <= nl * 1.03 else -1
    S_main = S_u if main_sign > 0 else S_l
    C_main = C_u if main_sign > 0 else C_l
    S_other = S_l if main_sign > 0 else S_u
    if main_sign < 0:
        S_main_pos = _mirror(S_main)
        C_main_pos = _mirror(C_main) if C_main is not None else None
    else:
        S_main_pos, C_main_pos = S_main, C_main
    rec.half_used = "upper" if main_sign > 0 else "lower"
    if C_main_pos is None or C_main_pos.is_empty:
        rec.warnings.append("Interior cavity could not be isolated (line work not closed) – "
                            "tap-hole union with the cavity is skipped.")
        C_main_pos = None
    rec.main_polys = _polys(S_main_pos)
    rec.cavity = C_main_pos

    if not S_other.is_empty:
        sym = S_main_pos.symmetric_difference(S_other if main_sign < 0 else _mirror(S_other))
        rec.symmetry_error_pct = round(100.0 * sym.area / max(S_main_pos.area, 1e-9), 2)
    # ── 5. features (what the other half has in addition) ────────────────────
    if not S_other.is_empty:
        _detect_features(rec, S_main_pos, S_other, main_sign, subs, frame, tol, gap_tol)

    # ── 6. exact boundary elements ───────────────────────────────────────────
    ax_b = axis_ls.buffer(max(tol, 1e-6) * 2)
    bnd = unary_union([p.boundary for p in rec.main_polys])
    inner_b = (C_main_pos.boundary if C_main_pos is not None else None)
    cav_buf = inner_b.buffer(max(gap_tol * 0.1, 1e-4)) if inner_b is not None else None
    outer_geo = bnd.difference(ax_b)
    if cav_buf is not None:
        outer_geo = outer_geo.difference(cav_buf)
    el_tol = max(tol * 1.2, 0.05)
    rec.outer_chains = _chains_from_lines(outer_geo, el_tol, "outer") if not outer_geo.is_empty else []
    if inner_b is not None:
        rec.inner_chains = _chains_from_lines(inner_b.difference(ax_b), el_tol, "inner")
    if rec.outer_chains:
        rec.main_chain = max(rec.outer_chains, key=lambda c: abs(c[-1].p1[0] - c[0].p0[0]))
    # wall thickness (median normal distance inner→outer along the biggest chain)
    rec.wall_thickness_mm = _wall_thickness(rec)

    # ── 7. dimension checks ──────────────────────────────────────────────────
    rec.dim_checks = _check_dims(scene, frame, g1, rec, gap_tol)
    rec.ok = True
    return rec


def _wall_thickness(rec: Recon) -> float | None:
    if rec.cavity is None or not rec.main_polys or not rec.main_chain:
        return None
    cav_b = rec.cavity.boundary
    ds = []
    for e in rec.main_chain:
        if e.kind != "line":
            continue
        n = max(3, int(math.dist(e.p0, e.p1) / max(rec.length_mm / 40, 1)))
        for i in range(n):
            t = (i + 0.5) / n
            p = Point(e.p0[0] + t * (e.p1[0] - e.p0[0]), e.p0[1] + t * (e.p1[1] - e.p0[1]))
            if p.y > 1e-6:
                ds.append(cav_b.distance(p))
    return float(np.median(ds)) if ds else None


# ═══════════════════════════════ features ════════════════════════════════════

def _detect_features(rec: Recon, S_main_pos, S_other, main_sign, subs, frame: Frame, tol, gap_tol):
    feat_sign = -main_sign
    M_signed = S_main_pos if feat_sign > 0 else _mirror(S_main_pos)
    extra = S_other.difference(M_signed.buffer(gap_tol * 1.5)).buffer(0)
    min_area = max(20 * tol * tol, 3e-5 * S_main_pos.area)
    extras = [p for p in _polys(extra) if p.area > min_area]
    if not extras:
        return

    sub_axes = []
    for a_u, b_u in subs:
        a, b = frame(a_u), frame(b_u)
        if abs(b[1] - a[1]) < 1e-6 * max(rec.length_mm, 1):
            continue                                    # parallel to the main axis
        t = -a[1] / (b[1] - a[1])
        o = (a[0] + t * (b[0] - a[0]), 0.0)
        far = a if abs(a[1]) > abs(b[1]) else b
        d = (far[0] - o[0], far[1] - o[1])
        dl = math.hypot(*d)
        if dl < 1e-9 or (1 if far[1] > 0 else -1) != feat_sign:
            continue
        sub_axes.append((o, (d[0] / dl, d[1] / dl), dl))
    if not sub_axes:
        rec.warnings.append(
            f"{len(extras)} asymmetric region(s) found in the {'lower' if feat_sign < 0 else 'upper'} "
            f"half but no feature centre-line – they are not modelled.")
        return

    assigned: dict[int, list[Polygon]] = {i: [] for i in range(len(sub_axes))}
    for p in extras:
        best, bi = 1e18, 0
        for i, (o, d, dl) in enumerate(sub_axes):
            seg = LineString([o, (o[0] + d[0] * dl, o[1] + d[1] * dl)])
            dd = p.distance(seg)
            if dd < best:
                best, bi = dd, i
        assigned[bi].append(p)

    for i, (o, d, dl) in enumerate(sub_axes):
        polys = assigned[i]
        if not polys:
            continue
        nd = (-d[1], d[0])
        mat = [d[0], d[1], nd[0], nd[1], -(o[0] * d[0] + o[1] * d[1]), -(o[0] * nd[0] + o[1] * nd[1])]
        # shapely affine: x' = a x + b y + xoff ; y' = d x + e y + yoff
        m = [d[0], d[1], nd[0], nd[1], mat[4], mat[5]]
        G = affinity.affine_transform(unary_union(polys), m)       # (s, rho)
        Gp = G.intersection(box(-1e9, 0, 1e9, 1e9))
        Gn = _mirror(G.intersection(box(-1e9, -1e9, 1e9, 0)))
        half = Gp if Gp.area >= Gn.area else Gn
        other = Gn if Gp.area >= Gn.area else Gp
        half = half.buffer(0)
        if half.is_empty:
            continue
        sym_err = 100.0 * half.symmetric_difference(other).area / max(half.area, 1e-9)

        s0, _, s_end, rho_max = half.bounds
        # tube slice (probe at 15 % of the feature length from the shell)
        probe = s0 + 0.15 * (s_end - s0)
        sl = half.intersection(LineString([(probe, -1.0), (probe, rho_max + 1.0)]))
        ivals = sorted((min(l.coords[0][1], l.coords[-1][1]), max(l.coords[0][1], l.coords[-1][1]))
                       for l in _lines(sl))
        if not ivals:
            continue
        r_i, r_o = max(ivals, key=lambda iv: iv[1] - iv[0])
        r_i = max(r_i, 0.0)
        ext_rect = Polygon([(0.0, r_i), (probe + 1e-6, r_i), (probe + 1e-6, r_o), (0.0, r_o)])
        prof = unary_union([half, ext_rect]).buffer(0)
        # plate = last slice
        last = half.intersection(LineString([(s_end - 0.25 * (s_end - s0) * 0.0 - 1e-6 * s_end, 0.0),
                                             (s_end - 1e-6 * s_end, 0.0)]))
        plate_r = rho_max
        # estimate plate thickness: s-extent where rho extent > r_o*1.02
        wide = [s for s in np.linspace(s0, s_end, 200)
                if half.intersection(LineString([(s, 0), (s, rho_max + 1)])).bounds[3] > r_o * 1.03
                if not half.intersection(LineString([(s, 0), (s, rho_max + 1)])).is_empty]
        plate_t = (max(wide) - min(wide)) if wide else 0.0

        feat = SubFeature(
            label="Tap hole / nozzle spout",
            origin=o,
            direction=d,
            tilt_deg=round(math.degrees(math.atan2(d[0], abs(d[1]))) * (1 if feat_sign < 0 else -1), 3),
            side=feat_sign,
            bore_radius=round(r_i, 3),
            outer_radius=round(r_o, 3),
            s_end=round(s_end, 3),
            plate_radius=round(plate_r, 3),
            plate_thickness=round(plate_t, 3),
            symmetry_error_pct=round(sym_err, 2),
        )
        for pg in _polys(prof):
            feat.profile_sr.append([(float(x), float(y)) for x, y in pg.exterior.coords])
            feat.holes_sr.append([[(float(x), float(y)) for x, y in h.coords] for h in pg.interiors])
        rec.features.append(feat)
        logger.info(f"[Recon] feature: bore r={r_i:.1f} outer r={r_o:.1f} s_end={s_end:.1f} "
                    f"tilt={feat.tilt_deg}° sym_err={sym_err:.1f}%")


# ═══════════════════════════════ dimension checks ════════════════════════════

def _check_dims(scene: Scene, frame: Frame, geom, rec: Recon, snap_tol: float) -> list[DimCheck]:
    out: list[DimCheck] = []
    mirror_geom = unary_union([geom, _mirror(geom)])
    tol_pt = max(snap_tol * 2.0, 0.5)
    for d in scene.dims:
        if d.p1 is None or d.p2 is None:
            out.append(DimCheck(d.id, d.kind, d.value, None, None, d.status, d.note))
            continue
        if d.kind == "angular":
            out.append(DimCheck(d.id, d.kind, d.value, None, None, d.status, d.note))
            continue
        z1, z2 = frame(d.p1), frame(d.p2)
        off = []
        for z in (z1, z2):
            dist = mirror_geom.distance(Point(z))
            off.append(dist)
        if d.direction is not None:
            dv = (math.cos(d.direction), math.sin(d.direction))
            a, b = d.p1, d.p2
            meas = abs((b[0] - a[0]) * dv[0] + (b[1] - a[1]) * dv[1]) * scene.mm_per_unit
        else:
            meas = math.dist(d.p1, d.p2) * scene.mm_per_unit
        if d.kind == "radius":
            meas = meas
        dev = abs(meas - d.value)
        status = d.status
        note = d.note
        if max(off) > tol_pt and not scene.exact:
            pass
        elif max(off) > max(tol_pt, 1e-3 * rec.length_mm):
            status = "endpoint_off_geometry" if status == "ok" else status
            note = (note + " " if note else "") + (
                f"Dimension reference point lies {max(off):.1f} mm from the drawn geometry.")
        out.append(DimCheck(d.id, d.kind, d.value, round(meas, 3), round(dev, 3), status, note))
    return out
