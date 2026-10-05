"""
app/utils/image_utils.py
=========================
Image loading, preprocessing, and format utilities shared across services.
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from loguru import logger


def load_image_as_numpy(source: str | Path | bytes) -> np.ndarray:
    """
    Load an image from a file path or raw bytes into an OpenCV-compatible
    BGR numpy array.

    Args:
        source: file path, Path object, or raw bytes.

    Returns:
        numpy ndarray in BGR format (uint8).

    Raises:
        ValueError: if the image cannot be decoded.
    """
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"OpenCV could not read image: {source}")
        logger.debug(f"Loaded image from path: {source} — shape={img.shape}")
        return img

    elif isinstance(source, bytes):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("OpenCV could not decode raw bytes as image.")
        logger.debug(f"Loaded image from bytes — shape={img.shape}")
        return img

    raise TypeError(f"Unsupported source type: {type(source)}")


def numpy_to_pil(img_bgr: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to a PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_numpy(img_rgb: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Preprocessing pipeline tuned for CAD/engineering drawings from AutoCAD.

    AutoCAD PDFs typically have:
      - Black text (annotation labels, dimension values)
      - Red/blue dimension lines
      - Magenta/pink auxiliary lines

    Strategy:
      1. Isolate black pixels only (suppress all colored pixels)
         → removes red/blue/magenta lines that confuse Tesseract
      2. CLAHE contrast enhancement on the isolated channel
      3. Otsu threshold for clean binary output

    Returns:
        Binary numpy array suitable for Tesseract (black text on white).
    """
    # Step 1: Isolate dark/black pixels
    # A pixel is "black" when R≈G≈B≈dark AND saturation is low
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Low saturation = not colorful (not red/blue/magenta lines)
    low_sat_mask = hsv[:, :, 1] < 60  # saturation < 60 → near-grey or black

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply mask: keep only dark pixels in low-saturation regions
    masked = np.where(low_sat_mask, gray, 255).astype(np.uint8)

    # Step 2: CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked)

    # Step 3: Gaussian blur to smooth noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Step 4: Otsu threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def scale_image(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscale an image using Lanczos interpolation (good for OCR)."""
    h, w = img.shape[:2]
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4
    )


def image_to_bytes(img: np.ndarray, ext: str = ".png") -> bytes:
    """Encode a numpy image array to bytes in the given format."""
    success, buf = cv2.imencode(ext, img)
    if not success:
        raise RuntimeError("Failed to encode image to bytes.")
    return buf.tobytes()


def crop_region(img: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop an image to a bounding box.

    Args:
        img: source image.
        bbox: (x_min, y_min, x_max, y_max) in pixel coordinates.

    Returns:
        Cropped numpy array.
    """
    x_min, y_min, x_max, y_max = bbox
    return img[y_min:y_max, x_min:x_max]
