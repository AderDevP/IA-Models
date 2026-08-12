"""
config.py — MammoAI Global Configuration
=========================================
Configuración centralizada del proyecto. Incluye:
 - Catálogo de modelos preentrenados con metadatos
 - Rutas de directorios del proyecto
 - Registro de tareas médicas (escalable)
 - Paletas de colores, parámetros BIRADS, etc.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────
# Rutas base del proyecto
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent.resolve()
MODELS_DIR    = BASE_DIR / "models"
DATASETS_DIR  = BASE_DIR / "datasets"
EXPORTS_DIR   = BASE_DIR / "exports"
LOGS_DIR      = BASE_DIR / "logs"
REPORTS_DIR   = BASE_DIR / "reports"

for _d in [MODELS_DIR, DATASETS_DIR, EXPORTS_DIR, LOGS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Repositorio Git (HTTPS)
# ─────────────────────────────────────────────
GIT_REPO_URL  = "https://github.com/AderDevP/IA-Models"
GIT_BRANCH    = "main"
GIT_USER_NAME = "AderDevP"
GIT_USER_EMAIL = "aderdevp@users.noreply.github.com"

# ─────────────────────────────────────────────
# Catálogo de modelos preentrenados disponibles
# ─────────────────────────────────────────────
# recommended=True  → se muestra con badge ⭐ en la UI
# task_type         → "classification" | "detection_yolo" | "detection"
PRETRAINED_MODELS = {

    # ══ CLASIFICADORES ══════════════════════════════════════════════

    "mammoscreen_efficientnetv2": {
        "name": "MammoScreen EfficientNetV2",
        "architecture": "EfficientNetV2-M",
        "task_type": "classification",
        "source": "huggingface",
        "hf_repo": "ianpan/mammoscreen",
        "pretrained_on": "CBIS-DDSM + RSNA Screening Mammography",
        "input_size": (512, 512),
        "description": "Modelo de producción entrenado en CBIS-DDSM + RSNA.",
        "classes": ["Negativo", "Positivo"],
        "local_filename": "mammoscreen_effv2.pth",
        "recommended": False,
    },
    "efficientnet_b4_cbis": {
        "name": "EfficientNet-B4 (CBIS-DDSM)",
        "architecture": "EfficientNet-B4",
        "task_type": "classification",
        "source": "timm",
        "timm_id": "efficientnet_b4",
        "hf_repo": None,
        "pretrained_on": "ImageNet (fine-tune requerido en CBIS-DDSM)",
        "input_size": (224, 224),
        "description": "EfficientNet-B4 con pesos ImageNet. El mejor detector general según pruebas empíricas.",
        "classes": ["Benigno", "Maligno"],
        "local_filename": "efficientnet_b4_cbis.pth",
        "recommended": True,
        "recommended_reason": "El mejor desempeño empírico en detección de masas",
    },

    # ══ DETECCIÓN CON YOLO ══════════════════════════════════════════
    # YOLOv8 — localización y señalización de lesiones en imagen

    "yolov8m_mammo": {
        "name": "YOLOv8-M Mamografía",
        "architecture": "YOLOv8-Medium",
        "task_type": "detection_yolo",
        "source": "ultralytics",
        "yolo_variant": "yolov8m",
        "hf_repo": "Ultralytics/assets",
        "pretrained_on": "COCO + fine-tune CBIS-DDSM",
        "input_size": (640, 640),
        "description": "YOLOv8-M para detección y localización precisa de masas y calcificaciones.",
        "classes": ["mass", "calcification", "mass_malignant", "calc_malignant"],
        "local_filename": "yolov8m_mammo.pt",
        "recommended": False,
    },
    "yolov8n_mammo": {
        "name": "YOLOv8-N Mamografía (rápido)",
        "architecture": "YOLOv8-Nano",
        "task_type": "detection_yolo",
        "source": "ultralytics",
        "yolo_variant": "yolov8n",
        "hf_repo": "Ultralytics/assets",
        "pretrained_on": "COCO (base — fine-tune requerido)",
        "input_size": (640, 640),
        "description": "YOLOv8-Nano, ultra-rápido. Ideal para pruebas o hardware limitado. Fine-tune necesario.",
        "classes": ["mass", "calcification", "mass_malignant", "calc_malignant"],
        "local_filename": "yolov8n_mammo.pt",
        "recommended": False,
    },
}

# Modelo recomendado por defecto
DEFAULT_MODEL_ID        = "efficientnet_b4_cbis"
DEFAULT_DETECTOR_ID     = "yolov8m_mammo"
RECOMMENDED_CLASSIFIER  = "efficientnet_b4_cbis"
RECOMMENDED_DETECTOR    = "yolov8m_mammo"

# ─────────────────────────────────────────────
# Registro de Tareas Médicas (ESCALABLE)
# ─────────────────────────────────────────────
# Cada tarea tiene: nombre, descripción, módulo Python, modelos compatibles
# Para agregar una nueva tarea (ej. pulmón), simplemente añadir aquí
# y crear tasks/lung_nodule/__init__.py
MEDICAL_TASKS = {
    "breast_cancer": {
        "name": "🎗️ Cáncer de Mama",
        "description": "Detección y localización de masas y calcificaciones en mamografías.",
        "module": "tasks.breast_cancer",
        "compatible_models": list(PRETRAINED_MODELS.keys()),
        "supported_formats": [".dcm", ".pgm", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"],
        "dataset": "CBIS-DDSM",
        "hf_dataset": "CBIS-DDSM",
        "enabled": True,
    },
    # ── Tareas futuras (placeholder) ──────────
    "lung_nodule": {
        "name": "🫁 Nódulos Pulmonares",
        "description": "Detección de nódulos en tomografías de tórax (LUNA16).",
        "module": "tasks.lung_nodule",
        "compatible_models": [],
        "supported_formats": [".dcm"],
        "dataset": "LUNA16",
        "enabled": False,
    },
    "skin_lesion": {
        "name": "🔬 Lesiones Dermatológicas",
        "description": "Clasificación de lesiones en dermoscopía (HAM10000).",
        "module": "tasks.skin_lesion",
        "compatible_models": [],
        "supported_formats": [".jpg", ".png"],
        "dataset": "HAM10000",
        "enabled": False,
    },
    "retinopathy": {
        "name": "👁️ Retinopatía Diabética",
        "description": "Gradación de retinopatía en imágenes de fondo de ojo.",
        "module": "tasks.retinopathy",
        "compatible_models": [],
        "supported_formats": [".jpg", ".png"],
        "dataset": "APTOS2019",
        "enabled": False,
    },
}

# ─────────────────────────────────────────────
# Parámetros BIRADS
# ─────────────────────────────────────────────
BIRADS_CATEGORIES = {
    0: {"label": "BIRADS 0", "color": "#808080", "meaning": "Evaluación incompleta — estudio adicional recomendado"},
    1: {"label": "BIRADS 1", "color": "#2ECC71", "meaning": "Negativo — sin hallazgos"},
    2: {"label": "BIRADS 2", "color": "#27AE60", "meaning": "Benigno — sin malignidad"},
    3: {"label": "BIRADS 3", "color": "#F39C12", "meaning": "Probablemente benigno — seguimiento en 6 meses"},
    4: {"label": "BIRADS 4", "color": "#E67E22", "meaning": "Sospechoso — biopsia recomendada"},
    5: {"label": "BIRADS 5", "color": "#E74C3C", "meaning": "Altamente sugestivo de malignidad"},
    6: {"label": "BIRADS 6", "color": "#8E44AD", "meaning": "Malignidad conocida — biopsia confirmada"},
}

# ─────────────────────────────────────────────
# Parámetros de entrenamiento por defecto
# ─────────────────────────────────────────────
TRAIN_DEFAULTS = {
    "epochs": 20,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "patience": 5,       # early stopping
    "val_split": 0.2,
    "test_split": 0.1,
    "num_workers": 2,
    "seed": 42,
    "mixed_precision": True,
    "gradient_clip": 1.0,
}

# ─────────────────────────────────────────────
# Paletas de color para visualización
# ─────────────────────────────────────────────
HEATMAP_COLORMAP   = "plasma"    # para Grad-CAM
BBOX_COLORS = {
    "mass":          (255, 80,  80),   # rojo
    "calcification": (255, 200, 50),   # amarillo
    "default":       (0,   200, 255),  # cyan
}
BBOX_THICKNESS     = 3
FONT_SCALE         = 0.7

# ─────────────────────────────────────────────
# Grad-CAM — capas objetivo por arquitectura
# ─────────────────────────────────────────────
GRADCAM_TARGET_LAYERS = {
    "efficientnet_b4":    "blocks[-1]",
    "convnext_small":     "stages[-1].blocks[-1]",
    "vit_base_patch16":   "blocks[-1].norm1",
    "fasterrcnn":         "backbone.body.layer4[-1]",
    "detr":               "backbone[-1]",
}

# ─────────────────────────────────────────────
# UI — configuración Gradio
# ─────────────────────────────────────────────
APP_TITLE       = "MammoAI — Diagnóstico Inteligente de Mamografías"
APP_DESCRIPTION = "Sistema open source de IA para detección, localización y análisis cuantitativo de cáncer de mama."
APP_VERSION     = "1.0.0"
APP_THEME       = "soft"   # tema Gradio
SHARE_GRADIO    = True     # para Colab: genera URL pública

# ─────────────────────────────────────────────
# CBIS-DDSM — configuración del dataset
# ─────────────────────────────────────────────
CBIS_DDSM_HF_DATASET = "matthieulel/cbis-ddsm"
CBIS_DDSM_MAX_GB     = 10   # límite de descarga en GB
CBIS_DDSM_SPLITS     = ["train", "validation", "test"]

# Normalización estándar para mamografías (ImageNet stats)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Inferencia
# ─────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_PIXEL_SPACING_MM     = 0.07  # ~70 µm — resolución típica mamografía digital
MAX_DETECTIONS               = 20

# Dispositivo (se sobreescribe en runtime si hay GPU)
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"
