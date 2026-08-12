"""
detector.py — Motor de Inferencia y Diagnóstico Visual
=======================================================
Orquesta:
  1. Carga del modelo activo (EfficientNet / ConvNeXt / ViT / Faster R-CNN / DETR)
  2. Inferencia sobre la imagen
  3. Dibujo de Bounding Boxes con medidas superpuestas
  4. Generación de Grad-CAM / heatmap de atención
  5. Superposición diagnóstica final

"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from config import (
    MODELS_DIR,
    PRETRAINED_MODELS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_PIXEL_SPACING_MM,
    BBOX_COLORS,
    BBOX_THICKNESS,
    HEATMAP_COLORMAP,
    DEVICE,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Carga de modelos
# ──────────────────────────────────────────────────────────────────

_model_cache: Dict[str, nn.Module] = {}   # caché en memoria


def load_model(
    model_id: str,
    device: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[nn.Module, Dict]:
    """Carga un modelo por su ID desde el caché o disco.

    Args:
        model_id: Clave en config.PRETRAINED_MODELS
        device: 'cuda', 'cpu' o None (auto-detecta)
        force_reload: Si True, ignora el caché en memoria
    Returns:
        Tuple (model: nn.Module, meta: dict)
    """
    if model_id not in PRETRAINED_MODELS:
        raise ValueError(f"Modelo desconocido: '{model_id}'")

    meta   = PRETRAINED_MODELS[model_id]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_id in _model_cache and not force_reload:
        logger.debug(f"Usando modelo en caché: {model_id}")
        return _model_cache[model_id].to(device), meta

    weights_path = MODELS_DIR / meta["local_filename"]
    arch = meta["architecture"].lower()

    # ── Construir arquitectura ────────────────────────────────────
    if "efficientnet" in arch:
        model = _build_efficientnet(meta, weights_path, device)
    elif "convnext" in arch:
        model = _build_convnext(meta, weights_path, device)
    elif "vit" in arch:
        model = _build_vit(meta, weights_path, device)
    elif "faster r-cnn" in arch or "fasterrcnn" in arch:
        model = _build_fasterrcnn(meta, weights_path, device)
    elif "detr" in arch:
        model = _build_detr(meta, weights_path, device)
    else:
        raise ValueError(f"Arquitectura no soportada: {meta['architecture']}")

    model = model.to(device)
    model.eval()
    _model_cache[model_id] = model
    logger.info(f"Modelo cargado: {meta['name']} → {device}")
    return model, meta


def _build_efficientnet(meta, weights_path, device):
    import timm
    timm_id   = meta.get("timm_id", "efficientnet_b4")
    n_classes = len(meta.get("classes", ["Benigno", "Maligno"]))
    use_pretrained = not weights_path.exists()  # si no hay pesos locales, usar ImageNet
    model     = timm.create_model(timm_id, pretrained=use_pretrained, num_classes=n_classes)
    _try_load_weights(model, weights_path, device)
    return model


def _build_convnext(meta, weights_path, device):
    import timm
    timm_id   = meta.get("timm_id", "convnext_small")
    n_classes = len(meta.get("classes", ["Benigno", "Maligno"]))
    use_pretrained = not weights_path.exists()
    model     = timm.create_model(timm_id, pretrained=use_pretrained, num_classes=n_classes)
    _try_load_weights(model, weights_path, device)
    return model


def _build_vit(meta, weights_path, device):
    import timm
    timm_id   = meta.get("timm_id", "vit_base_patch16_224")
    n_classes = len(meta.get("classes", ["Benigno", "Maligno"]))
    use_pretrained = not weights_path.exists()
    model     = timm.create_model(timm_id, pretrained=use_pretrained, num_classes=n_classes)
    _try_load_weights(model, weights_path, device)
    return model


def _build_fasterrcnn(meta, weights_path, device):
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    n_classes = len(meta.get("classes", ["__background__", "mass", "calcification"]))
    model     = fasterrcnn_resnet50_fpn_v2(weights=None, num_classes=n_classes)

    # Reemplazar el clasificador final
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

    _try_load_weights(model, weights_path, device)
    return model


def _build_detr(meta, weights_path, device):
    """Carga DETR desde HuggingFace transformers."""
    try:
        from transformers import DetrForObjectDetection, DetrConfig
        n_classes = len(meta.get("classes", ["__background__", "mass", "calcification"]))
        config    = DetrConfig(num_labels=n_classes)
        hf_repo   = meta.get("hf_repo", "facebook/detr-resnet-50")

        if weights_path.exists():
            model = DetrForObjectDetection(config)
            _try_load_weights(model, weights_path, device)
        else:
            model = DetrForObjectDetection.from_pretrained(hf_repo)
        return model
    except ImportError:
        raise ImportError("Instala: pip install transformers")


def _try_load_weights(model: nn.Module, weights_path: Path, device: str) -> None:
    """Intenta cargar pesos desde archivo — no falla si no existe."""
    if not weights_path.exists():
        logger.warning(
            f"Pesos no encontrados: {weights_path}. "
            "Usando inicialización por defecto. Descarga el modelo primero."
        )
        return
    try:
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict):
            state = ckpt.get("model_state_dict", ckpt)
        else:
            state = ckpt
        model.load_state_dict(state, strict=False)
        logger.info(f"Pesos cargados desde {weights_path}")
    except Exception as e:
        logger.warning(f"Error cargando pesos: {e}. Usando inicialización aleatoria.")


# ──────────────────────────────────────────────────────────────────
# Inferencia principal
# ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    image: Image.Image,
    model: nn.Module,
    model_meta: Dict,
    task,  # instancia de BaseTask
    pixel_spacing: float = DEFAULT_PIXEL_SPACING_MM,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Ejecuta la inferencia completa sobre una imagen PIL.

    Args:
        image: PIL Image RGB
        model: Modelo nn.Module ya cargado
        model_meta: Metadatos del modelo (de config.PRETRAINED_MODELS)
        task: Instancia de BaseTask (para preprocess/postprocess)
        pixel_spacing: mm por pixel (de DICOM o calibración manual)
        confidence_threshold: Score mínimo para aceptar detecciones

    Returns:
        Dict con resultado completo del diagnóstico
    """
    start_t = time.time()
    model.eval()

    # ── Preprocesamiento (delegado a la tarea) ────────────────────
    input_size = model_meta.get("input_size", (224, 224))
    tensor = task.preprocess(image, input_size=input_size).to(device)

    # ── Forward pass ──────────────────────────────────────────────
    arch = model_meta["architecture"].lower()
    if "detr" in arch:
        from transformers import DetrImageProcessor
        processor = DetrImageProcessor.from_pretrained(
            model_meta.get("hf_repo", "facebook/detr-resnet-50")
        )
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        raw_output = model(**inputs)
        # Convertir a formato estándar
        results = processor.post_process_object_detection(
            raw_output, threshold=confidence_threshold,
            target_sizes=[(image.height, image.width)]
        )
    elif "faster r-cnn" in arch or "fasterrcnn" in arch:
        raw_output = model([tensor.squeeze(0)])
    else:
        raw_output = model(tensor)

    elapsed = time.time() - start_t

    # ── Postprocesamiento (delegado a la tarea) ───────────────────
    report = task.postprocess(
        raw_output=raw_output,
        original_image=image,
        pixel_spacing=pixel_spacing,
        confidence_threshold=confidence_threshold,
    )
    report["inference_time_s"] = round(elapsed, 3)
    report["model_id"]         = model_meta.get("name", "")
    report["pixel_spacing_mm"] = pixel_spacing

    return report


# ──────────────────────────────────────────────────────────────────
# Visualización — Bounding Boxes con medidas
# ──────────────────────────────────────────────────────────────────

def draw_bounding_boxes(
    image: Image.Image,
    detections: List[Dict],
    show_measurements: bool = True,
) -> Image.Image:
    """Dibuja bounding boxes con clase, confianza y diámetro sobre la imagen.

    Args:
        image: PIL Image RGB original
        detections: Lista de dicts con bbox_px, class, confidence, diameter_mm
        show_measurements: Si True, muestra diámetro y área en mm

    Returns:
        PIL Image con anotaciones superpuestas
    """
    img_cv = np.array(image.copy())[:, :, ::-1].copy()  # PIL → OpenCV BGR

    for det in detections:
        x1, y1, x2, y2 = det["bbox_px"]
        cls_name  = det.get("class", "lesion")
        conf      = det.get("confidence", 0)
        diam_mm   = det.get("diameter_mm", 0)
        area_mm2  = det.get("area_mm2", 0)

        color = BBOX_COLORS.get(cls_name, BBOX_COLORS["default"])
        color_bgr = (color[2], color[1], color[0])

        # ── Rectángulo ────────────────────────────────────────────
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, BBOX_THICKNESS)

        # ── Etiqueta superior ─────────────────────────────────────
        label = f"{cls_name.upper()} {conf:.1f}%"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        y_label = max(y1 - 10, th + 5)
        cv2.rectangle(img_cv,
                      (x1, y_label - th - baseline),
                      (x1 + tw, y_label + baseline),
                      color_bgr, -1)
        cv2.putText(img_cv, label,
                    (x1, y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # ── Medidas superpuestas ──────────────────────────────────
        if show_measurements:
            measure_label = f"Ø {diam_mm:.1f}mm | {area_mm2:.1f}mm²"
            (mw, mh), _ = cv2.getTextSize(measure_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cx = det.get("center_x_px", (x1 + x2) // 2)
            cy = det.get("center_y_px", (y1 + y2) // 2)
            # Fondo semi-transparente
            overlay = img_cv.copy()
            cv2.rectangle(overlay,
                          (cx - mw // 2 - 5, cy - mh - 5),
                          (cx + mw // 2 + 5, cy + 5),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, img_cv, 0.4, 0, img_cv)
            cv2.putText(img_cv, measure_label,
                        (cx - mw // 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 100), 1, cv2.LINE_AA)

        # ── Línea de diámetro horizontal ──────────────────────────
        if show_measurements and diam_mm > 0:
            mid_y = (y1 + y2) // 2
            cv2.line(img_cv, (x1, mid_y), (x2, mid_y), color_bgr, 1)
            # Terminadores
            for px in [x1, x2]:
                cv2.line(img_cv, (px, mid_y - 5), (px, mid_y + 5), color_bgr, 2)

    # Convertir de vuelta a PIL
    return Image.fromarray(img_cv[:, :, ::-1])


# ──────────────────────────────────────────────────────────────────
# Grad-CAM / Heatmap de atención
# ──────────────────────────────────────────────────────────────────

def compute_gradcam(
    image: Image.Image,
    model: nn.Module,
    model_meta: Dict,
    task,
    device: str = "cpu",
    target_class: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Genera mapa de activación Grad-CAM.

    Args:
        image: PIL Image RGB
        model: Modelo cargado
        model_meta: Metadatos del modelo
        task: Instancia de BaseTask
        device: Dispositivo
        target_class: Clase objetivo (None = clase predicha)

    Returns:
        numpy array [H, W] de valores 0-1, o None si no disponible
    """
    try:
        from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        logger.warning(
            "pytorch-grad-cam no instalado. "
            "Grad-CAM no disponible. Instala: pip install grad-cam"
        )
        return None

    arch = model_meta["architecture"].lower()

    # ── Seleccionar capa objetivo ──────────────────────────────────
    target_layer = _get_gradcam_layer(model, arch)
    if target_layer is None:
        return None

    # ── Preparar tensor ───────────────────────────────────────────
    input_size = model_meta.get("input_size", (224, 224))
    tensor = task.preprocess(image, input_size=input_size).to(device)

    targets = [ClassifierOutputTarget(target_class)] if target_class is not None else None

    try:
        with GradCAMPlusPlus(model=model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
        return grayscale_cam[0]  # [H, W], valores 0-1
    except Exception as e:
        logger.warning(f"Error generando Grad-CAM: {e}")
        return None


def _get_gradcam_layer(model: nn.Module, arch: str) -> Optional[nn.Module]:
    """Retorna la capa objetivo para Grad-CAM según la arquitectura."""
    try:
        if "efficientnet" in arch:
            return model.blocks[-1]
        elif "convnext" in arch:
            return model.stages[-1].blocks[-1]
        elif "vit" in arch:
            return model.blocks[-1].norm1
        elif "fasterrcnn" in arch or "faster r-cnn" in arch:
            return model.backbone.body.layer4[-1]
        else:
            # Intentar obtener la última capa convolucional genéricamente
            layers = [(name, m) for name, m in model.named_modules()
                      if isinstance(m, nn.Conv2d)]
            if layers:
                return layers[-1][1]
    except (AttributeError, IndexError) as e:
        logger.warning(f"No se pudo obtener capa Grad-CAM: {e}")
    return None


def overlay_heatmap(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_PLASMA,
) -> Image.Image:
    """Superpone el mapa Grad-CAM coloreado sobre la imagen original.

    Args:
        image: PIL Image RGB original
        cam: Mapa de activación [H, W] en rango [0, 1]
        alpha: Opacidad del heatmap (0=invisible, 1=sólido)
        colormap: cv2 colormap (PLASMA, JET, VIRIDIS, etc.)
    Returns:
        PIL Image con heatmap superpuesto
    """
    # Redimensionar CAM a tamaño de imagen
    h, w = image.size[1], image.size[0]
    cam_resized = cv2.resize(cam, (w, h))

    # Convertir a mapa de color
    cam_uint8  = (cam_resized * 255).astype(np.uint8)
    cam_color  = cv2.applyColorMap(cam_uint8, colormap)
    cam_color  = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

    # Mezclar con imagen original
    img_array  = np.array(image)
    overlay    = (alpha * cam_color + (1 - alpha) * img_array).astype(np.uint8)

    return Image.fromarray(overlay)


# ──────────────────────────────────────────────────────────────────
# Pipeline completo de diagnóstico
# ──────────────────────────────────────────────────────────────────

def full_diagnostic_pipeline(
    image: Image.Image,
    model_id: str,
    task,
    pixel_spacing: float = DEFAULT_PIXEL_SPACING_MM,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    generate_gradcam: bool = True,
    device: Optional[str] = None,
) -> Tuple[Image.Image, Image.Image, Dict]:
    """Pipeline completo: inferencia + bboxes + Grad-CAM.

    Returns:
        Tuple:
          - annotated_image: PIL Image con bounding boxes y medidas
          - heatmap_image:   PIL Image con Grad-CAM overlay
          - report:          Dict con reporte completo
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Cargar modelo
    model, meta = load_model(model_id, device=device)

    # 2. Inferencia
    report = run_inference(
        image, model, meta, task,
        pixel_spacing=pixel_spacing,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    # 3. Dibujar bounding boxes
    annotated = draw_bounding_boxes(image, report.get("detections", []))

    # 4. Grad-CAM (solo para clasificadores)
    heatmap_img = image.copy()
    if generate_gradcam:
        arch = meta["architecture"].lower()
        if not any(x in arch for x in ["faster r-cnn", "fasterrcnn", "detr"]):
            cam = compute_gradcam(image, model, meta, task, device=device)
            if cam is not None:
                heatmap_img = overlay_heatmap(image, cam)

    return annotated, heatmap_img, report
