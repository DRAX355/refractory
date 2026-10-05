"""
app/services/part1_ingestion/__init__.py
=========================================
Part 1 — Input Engineering Drawing public API.

Usage:
    from app.services.part1_ingestion import ingest_drawing

    result = await ingest_drawing(file_bytes, filename)
"""
from app.services.part1_ingestion.ingestion_service import ingest_drawing

__all__ = ["ingest_drawing"]
