"""
dicom_utils.py — Utilidades para archivos DICOM
================================================
Carga, normalización, windowing y extracción de metadatos clínicos
de archivos DICOM (.dcm) para mamografías.
"""

from __future__ import annotations
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Carga principal
# ──────────────────────────────────────────────────────────────────

def load_dicom(file_path: str | Path) -> Tuple[np.ndarray, float, Dict]:
    """Carga un archivo DICOM y retorna array numpy, pixel spacing y metadatos.

    Args:
        file_path: Ruta al archivo .dcm

    Returns:
        Tuple:
          - image_array (np.ndarray): Array 2D uint8, listo para PIL
          - pixel_spacing_mm (float): mm por píxel (promedio entre filas y columnas)
          - metadata (dict): Campos clínicos relevantes del header DICOM
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("Instala pydicom: pip install pydicom")

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Archivo DICOM no encontrado: {file_path}")

    ds = pydicom.dcmread(str(file_path), force=True)

    # ── Pixel array ──────────────────────────────────────────────
    pixel_array = ds.pixel_array.astype(np.float32)

    # ── Photometric inversion (algunas mamografías son MONOCHROME1) ──
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if str(photometric).strip() == "MONOCHROME1":
        pixel_array = pixel_array.max() - pixel_array

    # ── Windowing (VOI LUT / Window Center + Width) ───────────────
    wc = getattr(ds, "WindowCenter", None)
    ww = getattr(ds, "WindowWidth",  None)
    if wc is not None and ww is not None:
        if isinstance(wc, pydicom.multival.MultiValue):
            wc, ww = float(wc[0]), float(ww[0])
        else:
            wc, ww = float(wc), float(ww)
        pixel_array = apply_windowing(pixel_array, wc, ww)
    else:
        # Auto-normalize al rango del array
        pixel_array = auto_normalize(pixel_array)

    image_array = pixel_array.astype(np.uint8)

    # ── Pixel spacing ──────────────────────────────────────────────
    pixel_spacing_mm = extract_pixel_spacing(ds)

    # ── Metadatos clínicos ─────────────────────────────────────────
    metadata = extract_metadata(ds)

    logger.info(
        f"DICOM cargado: {file_path.name} | "
        f"Shape: {image_array.shape} | "
        f"Pixel spacing: {pixel_spacing_mm:.4f} mm/px"
    )

    return image_array, pixel_spacing_mm, metadata


def dicom_to_pil(
    array: np.ndarray,
    window_center: Optional[float] = None,
    window_width: Optional[float] = None,
) -> Image.Image:
    """Convierte array numpy 2D → PIL Image RGB.

    Si se proveen window_center y window_width, aplica windowing adicional.
    """
    img = array.copy().astype(np.float32)

    if window_center is not None and window_width is not None:
        img = apply_windowing(img, window_center, window_width)

    # Normalizar a 0-255
    img = auto_normalize(img).astype(np.uint8)

    # Convertir a PIL RGB (3 canales)
    pil_img = Image.fromarray(img, mode="L").convert("RGB")
    return pil_img


# ──────────────────────────────────────────────────────────────────
# Utilidades internas
# ──────────────────────────────────────────────────────────────────

def apply_windowing(arr: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Aplica windowing VOI LUT estándar DICOM."""
    lo = window_center - window_width / 2.0
    hi = window_center + window_width / 2.0
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo) * 255.0
    return arr


def auto_normalize(arr: np.ndarray) -> np.ndarray:
    """Normaliza array al rango [0, 255]."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn) * 255.0


def extract_pixel_spacing(ds) -> float:
    """Extrae pixel spacing en mm del header DICOM.

    Intenta en orden: PixelSpacing, ImagerPixelSpacing, NominalScannedPixelSpacing.
    Si no se encuentra, retorna el valor por defecto de config.
    """
    from config import DEFAULT_PIXEL_SPACING_MM

    for tag in ["PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"]:
        val = getattr(ds, tag, None)
        if val is not None:
            try:
                # PixelSpacing es [row_spacing, col_spacing]
                spacings = [float(v) for v in val]
                return sum(spacings) / len(spacings)
            except (TypeError, ValueError):
                continue

    logger.warning(
        "No se encontró PixelSpacing en el header DICOM. "
        f"Usando valor por defecto: {DEFAULT_PIXEL_SPACING_MM} mm/px"
    )
    return DEFAULT_PIXEL_SPACING_MM


def extract_metadata(ds) -> Dict:
    """Extrae campos clínicos relevantes del header DICOM."""
    def safe_get(tag: str, default="N/A") -> str:
        val = getattr(ds, tag, default)
        if val is None:
            return "N/A"
        return str(val)

    return {
        "PatientID":              safe_get("PatientID"),
        "PatientAge":             safe_get("PatientAge"),
        "PatientSex":             safe_get("PatientSex"),
        "StudyDate":              safe_get("StudyDate"),
        "Modality":               safe_get("Modality"),
        "BodyPartExamined":       safe_get("BodyPartExamined"),
        "ViewPosition":           safe_get("ViewPosition"),     # MLO / CC
        "ImageLaterality":        safe_get("ImageLaterality"),  # L / R
        "Rows":                   safe_get("Rows"),
        "Columns":                safe_get("Columns"),
        "BitsStored":             safe_get("BitsStored"),
        "PhotometricInterpretation": safe_get("PhotometricInterpretation"),
        "Manufacturer":           safe_get("Manufacturer"),
        "InstitutionName":        safe_get("InstitutionName"),
        "StudyDescription":       safe_get("StudyDescription"),
        "SeriesDescription":      safe_get("SeriesDescription"),
        "SOPInstanceUID":         safe_get("SOPInstanceUID"),
        "PixelSpacing":           safe_get("PixelSpacing"),
        "EstimatedRadiographicMagnificationFactor": safe_get(
            "EstimatedRadiographicMagnificationFactor"
        ),
    }


# ──────────────────────────────────────────────────────────────────
# Carga unificada (DICOM + imágenes estándar)
# ──────────────────────────────────────────────────────────────────

def load_pgm(
    file_path: str | Path,
) -> Tuple[np.ndarray, Dict]:
    """Carga un archivo PGM (Portable GrayMap) — formato original CBIS-DDSM.

    Soporta PGM ASCII (P2) y binario (P5), incluyendo imágenes de 16-bit
    que PIL no puede leer directamente.

    Returns:
        Tuple (array uint8 normalizado, metadata dict)
    """
    file_path = Path(file_path)

    # Intentar con PIL primero (soporta P2/P5 8-bit)
    try:
        arr = np.array(Image.open(file_path), dtype=np.float32)
        meta = {"Filename": file_path.name, "Format": "PGM",
                "BitDepth": "8-bit (PIL)"}
        arr = auto_normalize(arr).astype(np.uint8)
        return arr, meta
    except Exception:
        pass

    # Fallback manual para PGM binario 16-bit (P5 con maxval > 255)
    with open(file_path, "rb") as f:
        raw = f.read()

    lines = raw.split(b"\n")
    idx = 0
    magic = lines[idx].decode().strip()
    if magic not in ("P2", "P5"):
        raise ValueError(f"Formato PGM no reconocido: {magic}")
    idx += 1

    while lines[idx].startswith(b"#"):
        idx += 1

    cols, rows = map(int, lines[idx].decode().split())
    idx += 1
    maxval = int(lines[idx].decode().strip())
    idx += 1

    header_size = sum(len(l) + 1 for l in lines[:idx])

    if magic == "P5":
        dtype = np.uint16 if maxval > 255 else np.uint8
        data = np.frombuffer(raw[header_size:], dtype=dtype)
        data = data.reshape((rows, cols)).astype(np.float32)
    else:  # P2 ASCII
        tokens = b" ".join(lines[idx:]).split()
        data = np.array([int(t) for t in tokens], dtype=np.float32).reshape((rows, cols))

    arr = auto_normalize(data).astype(np.uint8)
    meta = {
        "Filename": file_path.name,
        "Format": "PGM",
        "BitDepth": f"{16 if maxval > 255 else 8}-bit",
        "MaxVal": str(maxval),
        "Size": f"{cols} × {rows} px",
    }
    logger.info(f"PGM cargado: {file_path.name} | {cols}×{rows} | maxval={maxval}")
    return arr, meta


def load_image(
    file_path: str | Path,
    pixel_spacing_fallback: float = 0.07,
) -> Tuple[Image.Image, float, Dict]:
    """Carga cualquier imagen (DICOM, PGM, PNG, JPG, etc.) y retorna PIL, pixel_spacing y metadatos.

    Formatos soportados: .dcm, .pgm, .png, .jpg, .jpeg, .tiff, .bmp, .webp
    Para imágenes no-DICOM, pixel_spacing proviene del valor fallback
    o de la calibración manual en la UI.
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix == ".dcm":
        array, spacing, meta = load_dicom(file_path)
        pil_img = Image.fromarray(array, mode="L").convert("RGB")
        return pil_img, spacing, meta

    elif suffix == ".pgm":
        array, meta = load_pgm(file_path)
        meta["Note"] = "PGM — pixel spacing aproximado"
        meta["Size"] = f"{array.shape[1]} × {array.shape[0]} px"
        pil_img = Image.fromarray(array, mode="L").convert("RGB")
        return pil_img, pixel_spacing_fallback, meta

    else:
        try:
            pil_img = Image.open(file_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"No se puede abrir la imagen '{file_path.name}': {e}")

        meta = {
            "Filename": file_path.name,
            "Format":   suffix.upper().lstrip("."),
            "Size":     f"{pil_img.width} × {pil_img.height} px",
            "Note":     "Imagen estándar — pixel spacing es aproximado",
        }
        return pil_img, pixel_spacing_fallback, meta
