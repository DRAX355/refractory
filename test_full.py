"""Test full extraction pipeline on the uploaded drawing."""
import sys, asyncio
sys.path.insert(0, '.')
from pathlib import Path
from app.services.part1_ingestion.ingestion_service import ingest_drawing
from app.services.part2_extraction.extraction_service import extract_vessel_geometry

async def test():
    pdfs = list(Path('data/outputs').rglob('*.pdf'))
    if not pdfs:
        print("No PDF found.")
        sys.exit(1)
        
    pdf = pdfs[0]
    print(f"Testing full pipeline on: {pdf}")
    
    with open(pdf, 'rb') as f:
        file_bytes = f.read()
        
    ing_res = await ingest_drawing(file_bytes, pdf.name)
    print(f"Ingested. Pages: {ing_res.page_count}")
    
    ext_res = await extract_vessel_geometry(ing_res)
    geo = ext_res.vessel_geometry
    
    print("\n--- EXTRACTION RESULTS ---")
    print(f"Confidence: {geo.extraction_confidence}")
    print(f"Type: {geo.vessel_type}")
    print(f"Drawing No: {geo.drawing_number}")
    print(f"Outer Dia: {geo.shell.outer_diameter_mm}")
    print(f"Inner Dia: {geo.shell.inner_diameter_mm}")
    print(f"Length: {geo.shell.total_length_mm}")
    print(f"Capacity: {geo.operating_conditions.vessel_capacity_tons}")
    print("\nRaw Texts Found:")
    for t in geo.raw_ocr_texts:
        print(f"  {repr(t)}")

if __name__ == '__main__':
    asyncio.run(test())
