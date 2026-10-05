"""
app/services/part2_extraction/recon
===================================
Exact, dimension-aware vessel reconstruction (Part 2).

Pipeline
--------
    drawing (DXF / vector PDF / raster image)
        └─► reader      → ``Scene``   (primitives + dimensions + texts, in drawing units)
            └─► reconstruct → ``Recon``  (axis, solid cross-sections, cavity, features, checks)
                └─► mesh_builder → watertight 3-D model (GLB / STL) + validation report

* ``scene.py``          – neutral data model shared by every reader
* ``dxf_reader.py``     – exact CAD entity reader (LINE/ARC/POLYLINE/SPLINE/INSERT/DIMENSION/TEXT)
* ``pdf_reader.py``     – vector PDF reader (pdfplumber paths + words)
* ``raster_reader.py``  – scanned / screenshot reader (ridge skeleton + OCR + calibration)
* ``reconstruct.py``    – scene → axisymmetric solid cross-sections + features + checks
* ``mesh_builder.py``   – CSG model build with manifold3d, export + validation
"""
