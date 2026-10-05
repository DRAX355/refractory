"""
app/core/config.py
==================
Global application settings loaded from environment variables / .env file.
Uses pydantic-settings for validation and type coercion.
"""
from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"
    app_title: str = "RefractoryAI — Drawing Intelligence API"
    app_version: str = "0.1.0"

    # ── External Tool Paths ───────────────────────────────────
    # Tesseract OCR binary path
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # Poppler bin path for pdf2image (empty = auto-detect on Linux)
    poppler_path: str = ""

    # ── Model Config ─────────────────────────────────────────
    grounding_dino_model: str = "IDEA-Research/grounding-dino-base"
    device: str = "cpu"  # "cpu" | "cuda"

    # ── Data Directories ─────────────────────────────────────
    sample_drawings_dir: Path = Path("data/sample_drawings")
    outputs_dir: Path = Path("data/outputs")

    # ── Upload Limits ─────────────────────────────────────────
    max_upload_size_mb: int = 50

    @field_validator("outputs_dir", "sample_drawings_dir", mode="before")
    @classmethod
    def ensure_directories_exist(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


# Singleton — import this everywhere
settings = Settings()
