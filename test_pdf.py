"""Test pdfplumber on the actual uploaded drawing."""
import sys, glob
sys.path.insert(0, '.')
from pathlib import Path

# Find the uploaded PDF
pdfs = list(Path('data/outputs').rglob('*.pdf'))
if not pdfs:
    print("No PDF found in data/outputs")
    sys.exit(1)

pdf = pdfs[0]
print(f"Testing: {pdf}")

from app.utils.pdf_text_extractor import extract_pdf_text, classify_pdf_texts
texts = extract_pdf_text(pdf)
print(f"\n=== {len(texts)} text items extracted ===")
for t in texts:
    print(f"  {repr(t)}")

print("\n=== Classification ===")
cats = classify_pdf_texts(texts)
for k, v in cats.items():
    print(f"  {k}: {v}")
