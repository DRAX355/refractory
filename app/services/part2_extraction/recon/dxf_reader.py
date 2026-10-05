"""
app/services/part2_extraction/recon/dxf_reader.py
=================================================
Exact DXF reader.  Produces a ``Scene`` straight from CAD entities – no pixels,
no heuristics on geometry.  DIMENSION entities are read as *data* (value + the
two measured points) and are never mixed into the geometry.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from loguru import logger

from .scene import (
    ROLE_ANNOTATION, ROLE_CENTERLINE, ROLE_GEOMETRY, ROLE_HIDDEN,
    DimRec, Prim, Scene,
)

# $INSUNITS → millimetres per drawing unit
_INSUNITS_TO_MM = {
    0: 1.0, 1: 25.4, 2: 304.8, 3: 1_609_344.0, 4: 1.0, 5: 10.0, 6: 1000.0,
    7: 1e6, 8: 25.4e-6, 9: 0.0254, 10: 914.4, 14: 100.0, 15: 1e4, 16: 1e5,
}

_RE_CENTER_LT = re.compile(r"CENTER|DASHDOT|PHANTOM|AXIS|CHAIN|CENTRE", re.I)
_RE_HIDDEN_LT = re.compile(r"HIDDEN|DASHED|DASH(?!DOT)|DOT", re.I)
_RE_CENTER_LAYER = re.compile(r"cent|axis|(^|[^a-z])cl([^a-z]|$)|symm|chain", re.I)
_RE_ANNOT_LAYER = re.compile(
    r"dim|defpoint|text|anno|note|hatch|title|border|frame|table|bom|balloon|"
    r"leader|revision|symbol|viewport|grid", re.I)
_RE_HIDDEN_LAYER = re.compile(r"hidden|hid\b|dashed|phantom", re.I)


def read_dxf_scene(path: str | Path) -> Scene:
    import ezdxf
    from ezdxf import path as ezpath

    path = Path(path)
    try:
        doc = ezdxf.readfile(str(path))
    except Exception as exc:                       # damaged file → try recovery
        logger.warning(f"[DXF] readfile failed ({exc}); trying recover mode")
        from ezdxf import recover
        doc, _aud = recover.readfile(str(path))

    msp = doc.modelspace()
    ins = int(doc.header.get("$INSUNITS", 0) or 0)
    mm_per_unit = _INSUNITS_TO_MM.get(ins, 1.0)

    scene = Scene(mm_per_unit=mm_per_unit, exact=True, source="dxf")
    if ins == 0:
        scene.warnings.append("DXF has no $INSUNITS – assuming millimetres.")
    scene.meta["insunits"] = ins

    layer_lt: dict[str, str] = {}
    for lay in doc.layers:
        layer_lt[lay.dxf.name] = str(lay.dxf.get("linetype", "Continuous"))

    # flatten tolerance: 0.02 mm sagitta, expressed in drawing units
    tol_units = 0.02 / mm_per_unit

    def classify(e) -> str:
        layer = str(e.dxf.get("layer", "0"))
        lt = str(e.dxf.get("linetype", "BYLAYER"))
        if lt.upper() in ("BYLAYER", ""):
            lt = layer_lt.get(layer, "Continuous")
        if _RE_CENTER_LT.search(lt) or _RE_CENTER_LAYER.search(layer):
            return ROLE_CENTERLINE, layer, lt
        if _RE_ANNOT_LAYER.search(layer):
            return ROLE_ANNOTATION, layer, lt
        if _RE_HIDDEN_LT.search(lt) or _RE_HIDDEN_LAYER.search(layer):
            return ROLE_HIDDEN, layer, lt
        return ROLE_GEOMETRY, layer, lt

    def walk(layout, depth=0):
        for e in layout:
            t = e.dxftype()
            if t == "INSERT" and depth < 6:
                try:
                    yield from walk(list(e.virtual_entities()), depth + 1)
                except Exception as exc:
                    logger.debug(f"[DXF] INSERT explode failed: {exc}")
            else:
                yield e

    n_ent = 0
    for e in walk(msp):
        t = e.dxftype()
        n_ent += 1
        try:
            if t in ("LINE", "ARC", "CIRCLE", "ELLIPSE", "SPLINE", "LWPOLYLINE", "POLYLINE"):
                role, layer, lt = classify(e)
                kind = {"LINE": "line", "ARC": "arc", "CIRCLE": "circle",
                        "SPLINE": "spline"}.get(t, "poly")
                p = ezpath.make_path(e)
                for sp in p.sub_paths():
                    pts = [(v.x, v.y) for v in sp.flattening(tol_units, segments=4)]
                    if len(pts) >= 2:
                        closed = bool(sp.is_closed) or t == "CIRCLE" or (
                            len(pts) > 2 and math.dist(pts[0], pts[-1]) < 1e-9)
                        scene.prims.append(Prim(pts=pts, role=role, layer=layer,
                                                linetype=lt, kind=kind, closed=closed))
            elif t in ("TEXT", "MTEXT"):
                txt = e.dxf.text if t == "TEXT" else e.plain_text()
                txt = (txt or "").strip()
                if txt:
                    ip = e.dxf.insert
                    scene.texts.append((txt, (float(ip.x), float(ip.y))))
            elif t == "DIMENSION":
                rec = _read_dimension(e, mm_per_unit, len(scene.dims))
                if rec is not None:
                    scene.dims.append(rec)
        except Exception as exc:
            logger.debug(f"[DXF] entity {t} skipped: {exc}")

    logger.info(f"[DXF] {n_ent} entities → {len(scene.prims)} prims, "
                f"{len(scene.dims)} dims, {len(scene.texts)} texts, {mm_per_unit} mm/unit")
    if not scene.geometry():
        scene.warnings.append("DXF contains no usable geometry entities in model space.")
        return scene

    _keep_main_view(scene)
    return scene


# ── Dimension entities ─────────────────────────────────────────────────────────

def _read_dimension(e, mm_per_unit: float, idx: int) -> DimRec | None:
    dxf = e.dxf
    dtype = int(getattr(e, "dimtype", dxf.get("dimtype", 0)) or 0) & 7
    text = str(dxf.get("text", "") or "")

    def P(name):
        v = dxf.get(name)
        return None if v is None else (float(v.x), float(v.y))

    meas = None
    try:
        am = dxf.get("actual_measurement")
        if am:
            meas = float(am)
    except Exception:
        pass
    if meas is None:
        try:
            meas = float(e.get_measurement())
        except Exception:
            meas = None

    lfac = 1.0
    try:
        lfac = float(e.override().get("dimlfac", 1.0) or 1.0)
    except Exception:
        pass

    rec = DimRec(id=f"D{idx + 1}", text=text, source="dxf_dimension", confidence=1.0)
    rec.text_pos = P("text_midpoint")

    if dtype in (0, 1):                                  # linear / aligned
        p1, p2 = P("defpoint2"), P("defpoint3")
        if p1 is None or p2 is None:
            return None
        rec.kind, rec.p1, rec.p2 = "linear", p1, p2
        if dtype == 1:
            rec.direction = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        else:
            rec.direction = math.radians(float(dxf.get("angle", 0.0) or 0.0))
        d = (math.cos(rec.direction), math.sin(rec.direction))
        geo = abs((p2[0] - p1[0]) * d[0] + (p2[1] - p1[1]) * d[1])
        val = (meas if meas else geo) * lfac
        rec.value = val * mm_per_unit
    elif dtype in (3, 4):                                # diameter / radius
        pa, pb = P("defpoint"), P("defpoint4")
        if pa is None or pb is None:
            return None
        rec.kind = "diameter" if dtype == 3 else "radius"
        rec.p1, rec.p2 = pa, pb
        rec.direction = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        rec.value = (meas if meas else math.dist(pa, pb)) * lfac * mm_per_unit
    elif dtype in (2, 5):                                # angular
        rec.kind, rec.unit = "angular", "deg"
        rec.value = float(meas) if meas else 0.0
        if meas and meas < 2 * math.pi + 1e-6 and abs(meas) <= 6.2832:
            rec.value = math.degrees(meas)  # some writers store radians
        rec.p1, rec.p2 = P("defpoint2"), P("defpoint3")
    else:
        return None

    if text and text.replace("<>", "").strip() not in ("", "<>"):
        m = re.search(r"(\d+(?:\.\d+)?)", text.replace(",", ""))
        if m and "<>" not in text:
            shown = float(m.group(1))
            if rec.value and abs(shown - rec.value) > max(0.5, 1e-3 * rec.value):
                rec.status = "override"
                rec.note = (f"Dimension text '{text}' overrides the measured value "
                            f"{rec.value:.2f}")
    return rec


# ── Select the view that contains the vessel ──────────────────────────────────

def _keep_main_view(scene: Scene) -> None:
    """Drawings often hold several views + title block.  Keep the cluster of
    geometry that contains the main centre-line (or the biggest cluster)."""
    prims = [p for p in scene.prims if p.role in (ROLE_GEOMETRY, ROLE_CENTERLINE, ROLE_HIDDEN)]
    if len(prims) < 2:
        return
    bb = np.array([p.bbox() for p in prims], dtype=float)           # x0,y0,x1,y1
    ext = max(bb[:, 2].max() - bb[:, 0].min(), bb[:, 3].max() - bb[:, 1].min())
    gap = 0.03 * ext
    n = len(prims)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        dx = np.maximum(0.0, np.maximum(bb[i, 0], bb[:, 0]) - np.minimum(bb[i, 2], bb[:, 2]))
        dy = np.maximum(0.0, np.maximum(bb[i, 1], bb[:, 1]) - np.minimum(bb[i, 3], bb[:, 3]))
        for j in np.nonzero((dx <= gap) & (dy <= gap))[0]:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[ri] = rj

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    if len(groups) == 1:
        return

    def score(idx: list[int]) -> float:
        has_cl = any(prims[i].role == ROLE_CENTERLINE and prims[i].length() > 0.3 * ext for i in idx)
        tot = sum(prims[i].length() for i in idx if prims[i].role == ROLE_GEOMETRY)
        return tot * (10.0 if has_cl else 1.0)

    best = max(groups.values(), key=score)
    keep = {id(prims[i]) for i in best}
    kb = bb[best]
    x0, y0, x1, y1 = kb[:, 0].min(), kb[:, 1].min(), kb[:, 2].max(), kb[:, 3].max()
    m = 0.35 * max(x1 - x0, y1 - y0)

    dropped = len(prims) - len(best)
    scene.prims = [p for p in scene.prims
                   if p.role != ROLE_ANNOTATION and id(p) in keep]

    def inside(pt):
        return pt is not None and x0 - m <= pt[0] <= x1 + m and y0 - m <= pt[1] <= y1 + m

    scene.dims = [d for d in scene.dims if inside(d.p1) or inside(d.p2)]
    scene.texts = [t for t in scene.texts if inside(t[1])]
    scene.warnings.append(f"Multiple drawing views found – using the vessel view "
                          f"({dropped} entities in other views ignored).")
