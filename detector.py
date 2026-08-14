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

_model_cache: Dict[str, nn.Module] = {}   # caché modelos torch
_yolo_cache:  Dict[str, Any] = {}          # caché modelos YOLO


def load_yolo_model(model_id: str, force_reload: bool = False) -> Tuple[Any, Dict]:
    """Carga un modelo YOLOv8 via Ultralytics."""
    meta = PRETRAINED_MODELS[model_id]
    if model_id in _yolo_cache and not force_reload:
        return _yolo_cache[model_id], meta

    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Instala ultralytics: pip install ultralytics")

    weights_path = MODELS_DIR / meta["local_filename"]
    variant = meta.get("yolo_variant", "yolov8m")

    if weights_path.exists():
        logger.info(f"Cargando YOLO desde disco: {weights_path}")
        model = YOLO(str(weights_path))
        meta["is_coco_fallback"] = False
    else:
        logger.warning(
            f"Pesos YOLO no encontrados: {weights_path}. "
            f"Cargando pesos COCO base ({variant}) como fallback."
        )
        model = YOLO(f"{variant}.pt")  # descarga automáticamente de Ultralytics
        meta["is_coco_fallback"] = True

    _yolo_cache[model_id] = model
    logger.info(f"YOLO cargado: {meta['name']}")
    return model, meta


def load_model(
    model_id: str,
    device: Optional[str] = None,
    force_reload: bool = False,
) -> Tuple[Any, Dict]:
    """Carga un modelo por su ID desde el caché o disco.
    Soporta modelos torch (timm/HuggingFace) y YOLOv8 (Ultralytics).
    """
    if model_id not in PRETRAINED_MODELS:
        raise ValueError(f"Modelo desconocido: '{model_id}'")

    meta      = PRETRAINED_MODELS[model_id]
    task_type = meta.get("task_type", "classification")

    # ── YOLOv8 — ruta separada ────────────────────────────────────
    if task_type == "detection_yolo":
        return load_yolo_model(model_id, force_reload=force_reload)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if model_id in _model_cache and not force_reload:
        logger.debug(f"Usando modelo en caché: {model_id}")
        return _model_cache[model_id].to(device), meta

    weights_path = MODELS_DIR / meta["local_filename"]
    arch = meta["architecture"].lower()

    # ── Construir arquitectura ────────────────────────────────────
    if task_type == "vlm_classification" or "medgemma" in arch:
        model = _build_vlm(meta, weights_path, device)
        # Los VLM con device_map='auto' se manejan a sí mismos
        _model_cache[model_id] = model
        logger.info(f"VLM Modelo cargado: {meta['name']}")
        return model, meta
        
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


def _build_vlm(meta, weights_path, device):
    """Carga un VLM (Vision-Language Model) usando transformers y bitsandbytes."""
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError:
        import subprocess, sys
        logger.warning("Faltan librerías VLM. Instalando automáticamente (transformers, peft, accelerate, bitsandbytes)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers", "peft", "accelerate", "bitsandbytes"])
        from transformers import AutoProcessor, AutoModelForImageTextToText

    import torch
    hf_repo = meta.get("hf_repo", "ArnauMuns/medgemma-masses-cbis-ddsm")
    
    # Intentar cargar en 4-bit para evitar OOM en GPUs pequeñas (ej. Colab T4)
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            hf_repo,
            quantization_config=bnb_config,
            device_map="auto"
        )
    except (ImportError, ValueError) as e:
        logger.warning(f"BitsAndBytes falló o no está instalado ({e}). Usando fp16.")
        model = AutoModelForImageTextToText.from_pretrained(
            hf_repo,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
    processor = AutoProcessor.from_pretrained(hf_repo)
    
    # Adjuntamos el processor al modelo para tenerlo disponible durante inferencia
    model.processor = processor
    model.eval()
    return model


def _build_efficientnet(meta, weights_path, device):
    import timm
    timm_id   = meta.get("timm_id", "efficientnet_b4")
    n_classes = len(meta.get("classes", ["Benigno", "Maligno"]))
    use_pretrained = not weights_path.exists()  # si no hay pesos locales, usar ImageNet
    
    if use_pretrained:
        meta["is_untrained"] = True
        
    model     = timm.create_model(timm_id, pretrained=use_pretrained, num_classes=n_classes)
    _try_load_weights(model, weights_path, device)
    return model


def _build_convnext(meta, weights_path, device):
    import timm
    timm_id   = meta.get("timm_id", "convnext_small")
    n_classes = len(meta.get("classes", ["Benigno", "Maligno"]))
    use_pretrained = not weights_path.exists()
    
    if use_pretrained:
        meta["is_untrained"] = True
        
    model     = timm.create_model(timm_id, pretrained=use_pretrained, num_classes=n_classes)
    _try_load_weights(model, weights_path, device)
    return model


def _build_vit(meta, weights_path, device):
    import timm
    timm_id   = meta.get("timm_id", "vit_base_patch16_224")
    n_classes = len(meta.get("classes", ["Benigno", "Maligno"]))
    use_pretrained = not weights_path.exists()
    
    if use_pretrained:
        meta["is_untrained"] = True
        
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

    if not weights_path.exists():
        meta["is_untrained"] = True

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
def run_vlm_inference(
    image: Image.Image,
    model: nn.Module,
    model_meta: Dict,
    task,
    pixel_spacing: float = DEFAULT_PIXEL_SPACING_MM,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Ejecuta inferencia con un modelo VLM (Vision-Language Model)."""
    import time
    start_t = time.time()
    
    prompt_text = "Analyze this mammogram for masses and predict malignancy. Output your response clearly."
    
    processor = getattr(model, "processor", None)
    if not processor:
        raise ValueError("El modelo VLM no tiene 'processor' adjunto.")
        
    # Forzar el token <image> que exige la arquitectura PaliGemma/MedGemma
    # ignorando apply_chat_template porque a veces elimina el token visual.
    final_prompt = f"<image> {prompt_text}"

    inputs = processor(text=final_prompt, images=image, return_tensors="pt")
    # Acomodar tipos para int4/fp16
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)
    
    outputs = model.generate(**inputs, max_new_tokens=100)
    generated_text = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Parsear texto para extraer la predicción
    text_lower = generated_text.lower()
    is_malignant = any(word in text_lower for word in ["malignant", "maligno", "cancer", "tumor", "positive"])
    is_benign = any(word in text_lower for word in ["benign", "benigno", "normal", "negative"])
    
    if is_malignant and not is_benign:
        pred_class = "Maligno"
        conf = 95.0
        birads = 5
    elif is_benign and not is_malignant:
        pred_class = "Benigno"
        conf = 90.0
        birads = 2
    elif is_malignant and is_benign:
        pred_class = "Sospechoso"
        conf = 70.0
        birads = 4
    else:
        pred_class = "Desconocido (Ver reporte)"
        conf = 50.0
        birads = 0
        
    # Extraer bounding boxes (estilo PaliGemma <loc0000>)
    detections = []
    import re
    loc_matches = re.findall(r'<loc(\d+)><loc(\d+)><loc(\d+)><loc(\d+)>', generated_text)
    w, h = image.size
    for idx, (y1, x1, y2, x2) in enumerate(loc_matches):
        y1_px = int((int(y1) / 1024.0) * h)
        x1_px = int((int(x1) / 1024.0) * w)
        y2_px = int((int(y2) / 1024.0) * h)
        x2_px = int((int(x2) / 1024.0) * w)
        
        diameter_mm = max(x2_px - x1_px, y2_px - y1_px) * pixel_spacing
        area_mm2 = (x2_px - x1_px) * (y2_px - y1_px) * (pixel_spacing ** 2)
        
        detections.append({
            "id": idx + 1,
            "class": "Masa Detectada (VLM)",
            "confidence": 100.0,
            "bbox_px": [x1_px, y1_px, x2_px, y2_px],
            "width_px": x2_px - x1_px,
            "height_px": y2_px - y1_px,
            "diameter_mm": round(diameter_mm, 2),
            "area_mm2": round(area_mm2, 2),
            "center_x_px": (x1_px + x2_px) // 2,
            "center_y_px": (y1_px + y2_px) // 2,
        })
    
    inference_time = time.time() - start_t
    
    try:
        from tasks.breast_cancer import BIRADS_CATEGORIES
    except ImportError:
        BIRADS_CATEGORIES = {}
    
    report_text = f"╔══════════════════════════════════════════════╗\n"
    report_text += f"║     REPORTE DIAGNÓSTICO VLM — MedGemma       ║\n"
    report_text += f"╚══════════════════════════════════════════════╝\n\n"
    report_text += f"🔬 ANÁLISIS DEL MODELO (TEXTO DIRECTO):\n"
    report_text += f"   {generated_text.strip()}\n\n"
    report_text += f"📋 CLASIFICACIÓN INTERPRETADA: {pred_class} (Confianza estimada: {conf}%)\n"
    
    if birads > 0:
        birads_info = BIRADS_CATEGORIES.get(birads, {})
        report_text += f"\n📋 BIRADS ESTIMADO: {birads_info.get('label', f'BIRADS {birads}')}\n   {birads_info.get('meaning', '')}\n"
    
    if detections:
        report_text += f"\n🎯 LESIONES LOCALIZADAS POR VLM: {len(detections)}\n"
        
    return {
        "classification": {
            "predicted_class": pred_class,
            "confidence": conf,
            "probabilities": {"Maligno": conf if is_malignant else (100-conf), "Benigno": conf if is_benign else (100-conf)},
        },
        "birads": birads,
        "birads_info": BIRADS_CATEGORIES.get(birads, {}),
        "detections": detections,
        "report_text": report_text,
        "inference_time_ms": inference_time * 1000,
    }



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
# Inferencia YOLOv8
# ──────────────────────────────────────────────────────────────────

def run_yolo_inference(
    image: Image.Image,
    yolo_model: Any,
    model_meta: Dict,
    pixel_spacing: float = DEFAULT_PIXEL_SPACING_MM,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Ejecuta inferencia YOLOv8 y devuelve reporte estructurado compatible
    con el resto del pipeline (misma estructura que run_inference)."""
    import numpy as np

    start_t = time.time()
    classes = model_meta.get("classes", ["mass", "calcification"])

    # YOLOv8 acepta PIL directamente
    results = yolo_model.predict(
        source=image,
        conf=confidence_threshold,
        device=device,
        verbose=False,
    )
    elapsed = time.time() - start_t

    detections = []
    for r in results:
        boxes  = r.boxes
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            score = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = classes[cls_id] if cls_id < len(classes) else f"cls_{cls_id}"

            w_px = x2 - x1
            h_px = y2 - y1
            diameter_mm = max(w_px, h_px) * pixel_spacing
            area_mm2    = w_px * h_px * (pixel_spacing ** 2)

            detections.append({
                "id":           len(detections) + 1,
                "class":        cls_name,
                "confidence":   round(score * 100, 2),
                "bbox_px":      [round(x1), round(y1), round(x2), round(y2)],
                "width_px":     round(w_px),
                "height_px":    round(h_px),
                "diameter_mm":  round(diameter_mm, 2),
                "area_mm2":     round(area_mm2, 2),
                "center_x_px":  round((x1 + x2) / 2),
                "center_y_px":  round((y1 + y2) / 2),
            })

    # Estimar BIRADS desde máxima confianza de detección
    from config import BIRADS_CATEGORIES
    if detections:
        max_conf  = max(d["confidence"] for d in detections) / 100
        # Lesiones malignas tienen BIRADS más alto
        malignant = any("malignant" in d["class"] or "maligno" in d["class"] for d in detections)
        birads_prob = max_conf * (1.2 if malignant else 0.8)
        birads_prob = min(birads_prob, 1.0)
    else:
        birads_prob = 0.01

    birads_cat = _estimate_birads(birads_prob)
    birads_info = BIRADS_CATEGORIES.get(birads_cat, {})

    # Resumen de clasificación (derivado de detecciones)
    if detections:
        top = max(detections, key=lambda d: d["confidence"])
        pred_class = "Maligno" if "malignant" in top["class"] else "Sospechoso"
    else:
        pred_class = "Sin hallazgos"

    report = {
        "task":           "breast_cancer",
        "detections":     detections,
        "classification": {
            "predicted_class": pred_class,
            "confidence":      round(birads_prob * 100, 2),
            "probabilities":   {
                "Sin hallazgos": round((1 - birads_prob) * 100, 2),
                "Con lesión":    round(birads_prob * 100, 2),
            },
        },
        "birads":          birads_cat,
        "birads_info":     birads_info,
        "inference_time_s": round(elapsed, 3),
        "model_id":        model_meta.get("name", ""),
        "pixel_spacing_mm": pixel_spacing,
        "report_text":     _build_yolo_report_text(detections, birads_cat, birads_info, model_meta),
        "metadata":        {},
    }
    logger.info(
        f"YOLO inference: {len(detections)} detecciones | "
        f"BIRADS {birads_cat} | {elapsed:.3f}s"
    )
    return report


def _estimate_birads(prob: float) -> int:
    if prob < 0.02:  return 1
    if prob < 0.10:  return 2
    if prob < 0.30:  return 3
    if prob < 0.60:  return 4
    if prob < 0.95:  return 5
    return 6


def _build_yolo_report_text(detections, birads_cat, birads_info, model_meta) -> str:
    lines = [
        "╔══════════════════════════════════════════════╗",
        "║     REPORTE DIAGNÓSTICO — MammoAI  (YOLO)   ║",
        "╚══════════════════════════════════════════════╝",
        "",
        f"🤖 Modelo: {model_meta.get('name', 'YOLOv8')}",
        f"📋 BIRADS ESTIMADO: {birads_info.get('label', f'BIRADS {birads_cat}')}",
        f"   {birads_info.get('meaning', '')}",
        "",
    ]
    if model_meta.get("is_coco_fallback"):
        lines += [
            "⚠️ ATENCIÓN: Se están usando los pesos base (COCO) de YOLOv8.",
            "   El modelo NO HA SIDO ENTRENADO en mamografías, por lo que NO detectará",
            "   masas ni calcificaciones hasta que se realice el Fine-Tuning en la",
            "   pestaña de Entrenamiento.",
            "",
        ]

    if detections:
        lines.append(f"🎯 LESIONES DETECTADAS: {len(detections)}")
        for d in detections:
            lines += [
                "",
                f"   Lesión #{d['id']} — {d['class'].upper()}",
                f"   ├─ Confianza:     {d['confidence']:.1f}%",
                f"   ├─ Diámetro máx:  {d['diameter_mm']:.2f} mm",
                f"   ├─ Área:          {d['area_mm2']:.2f} mm²",
                f"   ├─ Posición:      x={d['center_x_px']}px, y={d['center_y_px']}px",
                f"   └─ BBox (px):     {d['bbox_px']}",
            ]
    else:
        lines.append("✅ YOLOv8 no detectó lesiones sospechosas en esta imagen.")
    lines += [
        "",
        "─" * 48,
        "⚠️  AVISO: Este reporte es una herramienta de apoyo.",
        "   No reemplaza el diagnóstico de un radiólogo certificado.",
        "─" * 48,
    ]
    return "\n".join(lines)



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
        if "vit" in arch:
            return model.blocks[-1].norm1
        elif "fasterrcnn" in arch or "faster r-cnn" in arch:
            return model.backbone.body.layer4[-1]
        
        # Para redes CNN (EfficientNet, ConvNeXt, etc), usar la última capa convolucional
        layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
        if layers:
            # Obtener la última Conv2d
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
    task_type   = meta.get("task_type", "classification")

    # 2. Inferencia — YOLO o modelo Torch
    if task_type == "detection_yolo":
        report      = run_yolo_inference(
            image, model, meta,
            pixel_spacing=pixel_spacing,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        # YOLO ya genera sus propias anotaciones con colores; también usamos draw_bounding_boxes
        annotated   = draw_bounding_boxes(image, report.get("detections", []))
        heatmap_img = image.copy()   # YOLO no usa Grad-CAM
        return annotated, heatmap_img, report

    arch = meta.get("architecture", "").lower()

    # ── VLM (MedGemma, PaliGemma) ──
    if task_type == "vlm_classification" or "medgemma" in arch:
        report = run_vlm_inference(
            image, model, meta, task,
            pixel_spacing=pixel_spacing,
            device=device,
        )
        annotated = draw_bounding_boxes(image, report.get("detections", []))
        # VLM no soporta Grad-CAM por ahora
        heatmap_img = image.copy()
        return annotated, heatmap_img, report

    # ── Modelos Torch (clasificadores / detectores Faster R-CNN / DETR) ──
    report = run_inference(
        image, model, meta, task,
        pixel_spacing=pixel_spacing,
        confidence_threshold=confidence_threshold,
        device=device,
    )

    # 4. Grad-CAM y WSOL (Weakly Supervised Object Localization)
    heatmap_img = image.copy()
    arch = meta["architecture"].lower()
    is_classifier = not any(x in arch for x in ["faster r-cnn", "fasterrcnn", "detr", "yolo"])
    
    if is_classifier:
        cam = compute_gradcam(image, model, meta, task, device=device)
        if cam is not None:
            if generate_gradcam:
                heatmap_img = overlay_heatmap(image, cam)
            
            # ── WSOL: Extraer BBox desde el mapa de calor ──
            # Si el clasificador detectó malignidad pero no hay boxes, usamos el Grad-CAM
            clf = report.get("classification", {})
            pred_class = clf.get("predicted_class")
            conf = clf.get("confidence", 0)
            
            if not report.get("detections") and (pred_class in ["Maligno", "Sospechoso"] or conf > 50.0):
                wsol_det = _extract_bbox_from_cam(cam, image.size, pixel_spacing, conf)
                if wsol_det:
                    report["detections"] = [wsol_det]
                    report["report_text"] = report.get("report_text", "") + \
                        "\n💡 Nota: Se utilizó el mapa de atención (Grad-CAM) para sugerir la ubicación de la lesión."

    # 3. Dibujar bounding boxes (ahora incluye las de WSOL si se generaron)
    annotated = draw_bounding_boxes(image, report.get("detections", []))

    return annotated, heatmap_img, report

def _extract_bbox_from_cam(cam: np.ndarray, img_size: Tuple[int, int], pixel_spacing: float, conf: float) -> Optional[Dict]:
    """Extrae un bounding box del mapa de atención Grad-CAM (WSOL)."""
    import cv2
    import numpy as np
    
    # Binarizar el heatmap (quedarse con el 30% más caliente)
    threshold = np.max(cam) * 0.7
    mask = (cam >= threshold).astype(np.uint8) * 255
    
    # Redimensionar la máscara al tamaño de la imagen original
    w, h = img_size
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Quedarse con el contorno más grande
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest_contour)
    
    # Si la caja es muy pequeña (ruido), ignorarla
    if cw < 5 or ch < 5:
        return None
        
    x2, y2 = x + cw, y + ch
    diameter_mm = max(cw, ch) * pixel_spacing
    area_mm2 = cw * ch * (pixel_spacing ** 2)
    
    return {
        "id": 1,
        "class": "Maligno (Sugerido por CAM)",
        "confidence": conf,
        "bbox_px": [x, y, x2, y2],
        "width_px": cw,
        "height_px": ch,
        "diameter_mm": round(diameter_mm, 2),
        "area_mm2": round(area_mm2, 2),
        "center_x_px": x + cw // 2,
        "center_y_px": y + ch // 2,
    }

