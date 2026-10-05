"""
app/services/part2_extraction/section_mesh.py
==============================================
Exact 3D solid from a SectionResult:

  * revolve the recovered wall cross-section (z, r) about the vessel axis
  * cut each nozzle bore through the shell, add the nozzle pipe wall + flange
  * export GLB (axis along +X, mm) and a debug overlay PNG
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh
from loguru import logger

try:
    import manifold3d as m3d
except ImportError:                                           # pragma: no cover
    m3d = None

from .section_reader import SectionResult


def _contours(geom) -> list[np.ndarray]:
    polys = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for p in polys:
        if p.is_empty or p.area < 1e-6:
            continue
        # manifold revolve: x = radius, y = axial position
        out.append(np.array([(r, z) for z, r in p.exterior.coords[:-1]], dtype=float))
        for ring in p.interiors:
            out.append(np.array([(r, z) for z, r in ring.coords[:-1]], dtype=float))
    return out


def _revolve(geom, segments: int):
    cs = m3d.CrossSection(_contours(geom), m3d.FillRule.EvenOdd)
    return m3d.Manifold.revolve(cs, segments)


def _cyl_along(direction_m: np.ndarray, origin_m: np.ndarray, u0: float, u1: float, radius: float, seg: int):
    """Cylinder with axis through `origin_m` along `direction_m`, spanning u0..u1."""
    h = u1 - u0
    cyl = m3d.Manifold.cylinder(h, radius, radius, seg)          # along +z, base at z=0
    e3 = direction_m / np.linalg.norm(direction_m)
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.cross(e3, e1)
    t = origin_m + e3 * u0
    mat = [
        [e1[0], e2[0], e3[0], t[0]],
        [e1[1], e2[1], e3[1], t[1]],
        [e1[2], e2[2], e3[2], t[2]],
    ]
    return cyl.transform(mat)


def build_solid(res: SectionResult, segments: int = 160):
    """Return (trimesh.Trimesh, info dict) or (None, reason)."""
    if m3d is None:
        return None, "manifold3d is not installed"
    if res.body is None or res.body.is_empty:
        return None, "empty cross-section"

    body = _revolve(res.body, segments)
    info = {"nozzles_applied": 0}

    for nz in res.nozzles:
        try:
            # manifold space: x = radial (paper up), y = axial, z = depth
            d_m = np.array([nz.direction[1], nz.direction[0], 0.0])
            o_m = np.array([0.0, nz.axis_z_mm, 0.0])
            bore = _cyl_along(d_m, o_m, 0.0, nz.flange_u0_mm + 0.5, nz.r_in_mm, 96)
            body = body - bore
            u_a = max(0.0, nz.u_start_mm)
            ring = _cyl_along(d_m, o_m, u_a, nz.flange_u0_mm + 0.01, nz.r_out_mm, 96) - \
                _cyl_along(d_m, o_m, u_a - 1.0, nz.flange_u0_mm + 1.0, nz.r_in_mm, 96)
            if res.cavity is not None and not res.cavity.is_empty:
                ring = ring - _revolve(res.cavity, segments)
            flange = _cyl_along(d_m, o_m, nz.flange_u0_mm, max(nz.flange_u1_mm, nz.flange_u0_mm + 1.0),
                                nz.flange_r_mm, 128)
            body = body + ring + flange
            info["nozzles_applied"] += 1
        except Exception as exc:                              # noqa: BLE001
            logger.warning(f"[Mesh] nozzle {nz.label} failed: {exc}")

    mesh = body.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3]
    faces = np.asarray(mesh.tri_verts)
    # manifold (x=radial, y=axial, z=depth) -> world (X=axial, Y=up, Z=depth); proper rotation
    R = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    # NOTE: y_world = -x_manifold would flip "paper up"; undo so the nozzle stays on the drawn side
    R = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])   # det = +1
    verts = verts @ R.T
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    tm.apply_translation(-tm.bounds.mean(axis=0))
    info.update({
        "vertices": int(len(tm.vertices)),
        "faces": int(len(tm.faces)),
        "watertight": bool(tm.is_watertight),
        "volume_m3": float(abs(tm.volume)) / 1e9,
        "bounds_mm": [list(map(float, b)) for b in tm.bounds],
    })
    return tm, info


def export_glb(res: SectionResult, path: str | Path, segments: int = 160) -> dict:
    tm, info = build_solid(res, segments)
    if tm is None:
        return {"ok": False, "reason": info}
    tm.export(str(path))
    info["ok"] = True
    logger.info(f"[Mesh] {path}: {info['vertices']} verts, watertight={info['watertight']}")
    return info


def save_debug_png(res: SectionResult, path: str | Path):
    """Overlay the recovered section on the drawing for visual verification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely import affinity

    fig, ax = plt.subplots(figsize=(15, 9))
    for s in res.debug_strokes:
        xs = [p[0] for p in s.pts]
        ys = [p[1] for p in s.pts]
        ax.plot(xs, ys, color="#9aa7a2", lw=0.4)
    S = res.scale_mm_per_unit
    polys = list(res.body.geoms) if hasattr(res.body, "geoms") else [res.body]
    for p in polys:
        if p.is_empty:
            continue
        q = affinity.scale(p, xfact=1 / S, yfact=1 / S, origin=(0, 0))
        q = affinity.translate(q, xoff=res.z_origin)
        # upper copy at the drawn axis (half unknown -> draw both)
        for sign in (1, -1):
            xs = [pt[0] for pt in q.exterior.coords]
            ys = [res.axis_coord + sign * pt[1] for pt in q.exterior.coords]
            ax.fill(xs, ys, color="#00B87A", alpha=0.45, lw=0)
    ax.set_aspect("equal")
    ax.set_title(f"scale {S:.3f} mm/unit - {res.scale_source}")
    fig.savefig(str(path), dpi=110, bbox_inches="tight")
    plt.close(fig)
