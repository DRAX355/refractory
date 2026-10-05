"""
app/services/part2_extraction/recon/mesh_builder.py
===================================================
Constructs a 3D mesh from a reconstructed axisymmetric 2D section.
"""

from __future__ import annotations

import math
import numpy as np
import trimesh
from loguru import logger

try:
    import manifold3d as m3d
except ImportError:
    m3d = None

from .reconstruct import Recon


def build_mesh(rec: Recon, segments: int = 64, filepath: str | None = None) -> trimesh.Trimesh | None:
    """
    Builds a 3D mesh from the Recon object using CSG (via manifold3d).
    Revolves the main solid profile around the Z axis (which maps to Y in manifold's space).
    Adds any sub-features with correct orientation.
    Returns a trimesh.Trimesh object and optionally saves it to `filepath`.
    """
    if m3d is None:
        logger.error("manifold3d is not installed. Cannot build 3D mesh.")
        return None

    if not rec.ok or not rec.main_polys:
        logger.warning("Reconstruction failed or no solid geometry found. Cannot build mesh.")
        return None

    logger.info(f"Building 3D mesh with {segments} segments per revolution...")

    # 1. Revolve the main body
    # Manifold revolve expects X=radius, Y=axis.
    # Our 2D coordinates are (z, r). We map to (r, z) -> (X, Y).
    contours = []
    
    for poly in rec.main_polys:
        # exterior
        ext_coords = np.array(poly.exterior.coords)
        ext_zr = np.c_[ext_coords[:, 1], ext_coords[:, 0]]
        contours.append(ext_zr)
        
        # interiors
        for interior in poly.interiors:
            int_coords = np.array(interior.coords)
            int_zr = np.c_[int_coords[:, 1], int_coords[:, 0]]
            contours.append(int_zr)
            
    if not contours:
        logger.warning("No valid contours to revolve.")
        return None

    cross_section = m3d.CrossSection(contours, m3d.FillRule.EvenOdd)
    main_solid = m3d.Manifold.revolve(cross_section, segments)

    # 2. Add features (SubFeature)
    if rec.features:
        logger.info(f"Adding {len(rec.features)} feature(s) to the mesh...")
        for feat in rec.features:
            feat_contours = []
            for ring in feat.profile_sr:
                r_coords = np.array(ring)
                feat_zr = np.c_[r_coords[:, 1], r_coords[:, 0]]
                feat_contours.append(feat_zr)
            
            for rings in feat.holes_sr:
                for ring in rings:
                    r_coords = np.array(ring)
                    feat_zr = np.c_[r_coords[:, 1], r_coords[:, 0]]
                    feat_contours.append(feat_zr)
                    
            if not feat_contours:
                continue
                
            feat_cs = m3d.CrossSection(feat_contours, m3d.FillRule.EvenOdd)
            feat_solid = m3d.Manifold.revolve(feat_cs, segments)
            
            # The feature was revolved around its local Y axis.
            # Local Y needs to align with the direction vector (dr, dz) in the main (X, Y) plane.
            # feat.direction is (dz, dr).
            dz = feat.direction[0]
            dr = feat.direction[1]
            z_0 = feat.origin[0]

            # Rotation matrix to align local Y with (dr, dz, 0)
            mat = [
                [dz, dr, 0.0, 0.0],
                [-dr, dz, 0.0, z_0],
                [0.0, 0.0, 1.0, 0.0]
            ]
            
            feat_solid = feat_solid.transform(mat)
            main_solid = main_solid + feat_solid

    # 3. Convert to trimesh
    mesh = main_solid.to_mesh()
    verts = mesh.vert_properties[:, :3]
    faces = mesh.tri_verts
    
    # trimesh currently has Z-up. We should probably rotate the mesh so that the original Z axis
    # (which is currently Y) becomes Z, and X/Z become X/Y.
    # We constructed it with axis of revolution = Y.
    # Let's rotate -90 deg around X so that +Y goes to +Z.
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    
    # Rotate Y to Z
    rot_matrix = trimesh.transformations.rotation_matrix(
        angle=-math.pi / 2, 
        direction=[1, 0, 0], 
        point=[0, 0, 0]
    )
    tm.apply_transform(rot_matrix)

    logger.info(f"Mesh built successfully: {len(tm.vertices)} vertices, {len(tm.faces)} faces.")

    if filepath:
        tm.export(filepath)
        logger.info(f"Mesh exported to {filepath}")

    return tm
