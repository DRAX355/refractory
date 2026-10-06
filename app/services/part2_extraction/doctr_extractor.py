"""
app/services/part2_extraction/doctr_extractor.py
================================================
Uses DocTR (Document Text Recognition) instead of Tesseract.
Provides exact bounding polygons for every detected word, which is critical
for intersecting with CAD geometry to erase dimension lines.
"""

import numpy as np
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from shapely.geometry import Polygon
from dataclasses import dataclass
from typing import List

@dataclass
class DocTRWord:
    value: str
    polygon: Polygon
    confidence: float

@dataclass
class DocTRResult:
    words: List[DocTRWord]
    
    def get_all_polygons(self) -> List[Polygon]:
        return [w.polygon for w in self.words]

# Load model globally to avoid reloading on every request
# We use lightweight mobilenet architectures for CPU compatibility
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = ocr_predictor(det_arch='db_mobilenet_v3_large', reco_arch='crnn_mobilenet_v3_small', pretrained=True)
    return _predictor

def run_doctr_on_image(image_bgr: np.ndarray) -> DocTRResult:
    """
    Runs DocTR on a BGR numpy image (e.g. from cv2).
    Returns a list of words and their exact Shapely polygons.
    """
    predictor = get_predictor()
    
    # DocTR expects RGB images or DocumentFile
    # Convert BGR to RGB
    import cv2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # DocTR expects a list of arrays
    out = predictor([image_rgb])
    
    h, w = image_rgb.shape[:2]
    
    words = []
    
    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    # Geometry is ((xmin, ymin), (xmax, ymax)) in relative coordinates [0, 1]
                    (xmin_rel, ymin_rel), (xmax_rel, ymax_rel) = word.geometry
                    
                    xmin, ymin = xmin_rel * w, ymin_rel * h
                    xmax, ymax = xmax_rel * w, ymax_rel * h
                    
                    # Create a Shapely polygon
                    poly = Polygon([
                        (xmin, ymin),
                        (xmax, ymin),
                        (xmax, ymax),
                        (xmin, ymax)
                    ])
                    
                    words.append(DocTRWord(
                        value=word.value,
                        polygon=poly,
                        confidence=word.confidence
                    ))
                    
    return DocTRResult(words=words)
