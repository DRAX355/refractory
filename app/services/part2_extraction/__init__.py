"""
app/services/part2_extraction/__init__.py
==========================================
Part 2 — Drawing Intelligence (Extraction) public API.

Usage:
    from app.services.part2_extraction import extract_vessel_geometry

    geometry = await extract_vessel_geometry(ingestion_result)
"""
from app.services.part2_extraction.extraction_service import extract_vessel_geometry

__all__ = ["extract_vessel_geometry"]
