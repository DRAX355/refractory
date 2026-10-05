"""
app/services/part2_extraction/recon/pdf_reader.py
=================================================
Exact PDF vector path reader. Produces a ``Scene`` straight from PDF paths
using pdfplumber.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from loguru import logger

from .scene import (
    ROLE_GEOMETRY, Prim, Scene,
)

def read_pdf_scene(pdf_path: str | Path) -> Scene:
    try:
        import pdfplumber
    except ImportError:
        logger.warning("[PDF-Reader] pdfplumber not installed")
        return Scene()

    path = Path(pdf_path)
    # 1 pt = 0.352778 mm
    scene = Scene(mm_per_unit=0.352778, exact=True, source="pdf_vector")

    with pdfplumber.open(str(path)) as pdf:
        if not pdf.pages:
            return scene
        
        # We process the first page
        page = pdf.pages[0]
        
        # The coordinates in pdfplumber have y=0 at the top, increasing downwards.
        # But we need them to be consistent (e.g. y-up).
        # We can just leave them and let reconstruct.py's axis detection handle it.
        # However, it's usually better to flip y so it matches standard CAD (y-up).
        page_h = float(page.height)

        def to_pt(x, y):
            # flip y so y=0 is at the bottom
            return (float(x), page_h - float(y))

        for line in (page.lines or []):
            try:
                p1 = to_pt(line["x0"], line["y0"])
                p2 = to_pt(line["x1"], line["y1"])
                scene.prims.append(Prim(pts=[p1, p2], role=ROLE_GEOMETRY, kind="line"))
            except (KeyError, TypeError):
                pass
                
        for curve in (page.curves or []):
            try:
                pts = curve.get("pts", [])
                if len(pts) >= 2:
                    p = [to_pt(pt[0], pt[1]) for pt in pts]
                    scene.prims.append(Prim(pts=p, role=ROLE_GEOMETRY, kind="poly"))
            except (KeyError, TypeError):
                pass
                
        for rect in (page.rects or []):
            try:
                x0, y0 = float(rect["x0"]), float(rect["y0"])
                x1, y1 = float(rect["x1"]), float(rect["y1"])
                # In pdfplumber, rects often have y0 < y1 (top-down)
                pts = [
                    to_pt(x0, y0),
                    to_pt(x1, y0),
                    to_pt(x1, y1),
                    to_pt(x0, y1),
                    to_pt(x0, y0)
                ]
                scene.prims.append(Prim(pts=pts, role=ROLE_GEOMETRY, kind="poly", closed=True))
            except (KeyError, TypeError):
                pass

    logger.info(f"[PDF-Reader] Extracted {len(scene.prims)} geometric primitives.")
    if not scene.prims:
        scene.warnings.append("PDF contains no usable vector geometry.")
        return scene

    _keep_main_view(scene)
    return scene


def _keep_main_view(scene: Scene) -> None:
    """Isolate the largest cluster of geometry (the vessel view) if there are multiple views."""
    prims = [p for p in scene.prims if p.role == ROLE_GEOMETRY]
    if len(prims) < 2:
        return
        
    bb = np.array([p.bbox() for p in prims], dtype=float)           # x0,y0,x1,y1
    ext = max(bb[:, 2].max() - bb[:, 0].min(), bb[:, 3].max() - bb[:, 1].min())
    gap = 0.05 * ext
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
        return sum(prims[i].length() for i in idx)

    best = max(groups.values(), key=score)
    keep = {id(prims[i]) for i in best}
    
    dropped = len(prims) - len(best)
    scene.prims = [p for p in scene.prims if id(p) in keep]
    
    scene.warnings.append(f"Multiple drawing views found – using the largest view "
                          f"({dropped} entities in other views ignored).")
