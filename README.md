# AI-Driven Refractory Lining Design — RefractoryAI
## ZeroZeta | End-to-End Solution Architecture

> **From Engineering Drawing to CAD Deliverables**

---

## Architecture Overview

| Part | Module | Description | Status |
|------|--------|-------------|--------|
| **1** | `part1_ingestion` | Accept PDF/DWG/DXF/Image drawings | ✅ Done |
| **2** | `part2_extraction` | DINO + OCR + OpenCV → VesselGeometry | ✅ Done |
| 3 | `ai_assistant` | SmolLM2 + RAG (Local LLM) | 🔲 Pending |
| 4 | `design_engine` | Refractory Engineering Rules Engine | 🔲 Pending |
| 5 | `geometry_engine` | Parametric FreeCAD geometry | 🔲 Pending |
| 6 | `cad_generation` | FreeCAD TechDraw + ezdxf export | 🔲 Pending |
| 7 | `outputs` | GA/Section/Component drawings + BOM | 🔲 Pending |

---

## Project Structure

```
refractoryAI/
├── app/
│   ├── main.py                                # FastAPI app entry point
│   ├── api/v1/
│   │   ├── routes_ingestion.py                # Part 1 REST API
│   │   └── routes_extraction.py               # Part 2 REST API
│   ├── core/
│   │   ├── config.py                          # Settings (pydantic-settings)
│   │   └── logging_config.py                  # Loguru structured logging
│   ├── models/
│   │   └── vessel_geometry.py                 # ★ CONTRACT MODEL (all parts use this)
│   ├── services/
│   │   ├── part1_ingestion/
│   │   │   ├── ingestion_service.py           # Part 1 orchestrator
│   │   │   ├── file_handler.py                # Multi-format file loader
│   │   │   └── drawing_validator.py           # File validation
│   │   └── part2_extraction/
│   │       ├── extraction_service.py          # Part 2 orchestrator
│   │       ├── dino_detector.py               # Grounding DINO detection
│   │       ├── ocr_extractor.py               # Tesseract OCR
│   │       ├── opencv_processor.py            # OpenCV geometry
│   │       └── geometry_assembler.py          # Fusion → VesselGeometry
│   └── utils/
│       ├── image_utils.py                     # Shared image helpers
│       └── unit_parser.py                     # Dimension string parser
├── tests/
│   ├── test_part1_ingestion.py
│   └── test_part2_extraction.py
├── data/
│   ├── sample_drawings/                        # Put test drawings here
│   └── outputs/                                # Extraction results (JSON)
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Prerequisites — Install These First

### 1. Tesseract OCR (required for Part 2 OCR)
- **Windows**: https://github.com/UB-Mannheim/tesseract/wiki
  - Install to `C:\Program Files\Tesseract-OCR\`
  - Add to PATH

### 2. Poppler (required for PDF ingestion)
- **Windows**: https://github.com/oschwartz10612/poppler-windows/releases/
  - Extract to `C:\poppler\`
  - Add `C:\poppler\Library\bin` to PATH

### 3. ODA File Converter (required for DWG files — optional)
- **Download**: https://www.opendesign.com/guestfiles/oda_file_converter
- Only needed if you're working with DWG files

---

## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate      # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env
# Edit .env: set TESSERACT_CMD and POPPLER_PATH paths

# 4. Run the API server
uvicorn app.main:app --reload --port 8000
```

API Documentation: http://localhost:8000/docs

---

## API Endpoints

### Part 1 — Drawing Ingestion
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingestion/upload` | Upload drawing (PDF/DXF/DWG/Image) |
| `GET`  | `/api/v1/ingestion/{job_id}/status` | Check job status |

### Part 2 — Drawing Intelligence
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/extraction/process` | Upload + extract in one call |
| `POST` | `/api/v1/extraction/process/{job_id}` | Re-run extraction on ingested job |
| `GET`  | `/api/v1/extraction/{job_id}/result` | Get saved VesselGeometry JSON |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `GET`  | `/docs` | Swagger UI |

---

## Contract Model (for Parts 3–7)

```python
from app.models.vessel_geometry import VesselGeometry

# VesselGeometry contains:
# - shell: VesselShellGeometry (outer_diameter, inner_diameter, total_length...)
# - nozzles: list[Nozzle]
# - operating_conditions: OperatingConditions (vessel_capacity_tons, ...)
# - vessel_type: VesselType (BOF, EAF, LADLE, ...)
# - drawing_number, revision
# - raw_ocr_texts: list[str]      ← useful for Part 3 RAG
# - dino_detections: list[BoundingBox]
# - extraction_confidence: float
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## For Other Team Members (Parts 3–7)

1. Import the contract model: `from app.models.vessel_geometry import VesselGeometry`
2. Call Part 2 API: `POST /api/v1/extraction/process` with a drawing file
3. The JSON response contains the full `VesselGeometry` object
4. Or load from disk: `data/outputs/{job_id}/vessel_geometry.json`
