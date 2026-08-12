"""
tasks/breast_cancer/__init__.py — Tarea: Detección de Cáncer de Mama
=====================================================================
Implementa BaseTask para mamografías.
Se registra automáticamente en el TaskRegistry al ser importado.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from PIL import Image
import torch
import torchvision.transforms as T

from core.base_task import BaseTask
from core.registry  import TaskRegistry
from config import (
    PRETRAINED_MODELS,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    BIRADS_CATEGORIES,
)


class BreastCancerTask(BaseTask):
    """Tarea de detección, localización y clasificación de cáncer de mama."""

    @property
    def task_id(self) -> str:
        return "breast_cancer"

    @property
    def task_name(self) -> str:
        return "🎗️ Cáncer de Mama"

    @property
    def task_description(self) -> str:
        return (
            "Detección y localización de masas y calcificaciones en mamografías. "
            "Soporta archivos DICOM y formatos estándar (PNG, JPG). "
            "Dataset de referencia: CBIS-DDSM."
        )

    @property
    def supported_formats(self) -> List[str]:
        return [".dcm", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]

    # ──────────────────────────────────────────
    # Pipeline de inferencia
    # ──────────────────────────────────────────

    def preprocess(
        self,
        image: Image.Image,
        input_size: tuple = (224, 224),
        **kwargs,
    ) -> torch.Tensor:
        """PIL RGB → tensor normalizado [1, C, H, W]."""
        transform = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ])
        return transform(image).unsqueeze(0)  # [1, 3, H, W]

    def postprocess(
        self,
        raw_output: Any,
        original_image: Image.Image,
        metadata: Optional[Dict] = None,
        pixel_spacing: float = 0.07,
        confidence_threshold: float = 0.35,
    ) -> Dict[str, Any]:
        """Convierte salida del modelo → reporte estructurado.

        Maneja tanto salidas de clasificación (tensor [B, C]) como
        de detección (lista de dicts con boxes, scores, labels).
        """
        metadata = metadata or {}
        report = {
            "task":           self.task_id,
            "detections":     [],
            "classification": {},
            "birads":         None,
            "birads_info":    {},
            "report_text":    "",
            "metadata":       metadata,
        }

        # ── Clasificador (tensor de logits) ────────────────────────
        if isinstance(raw_output, torch.Tensor):
            probs = torch.softmax(raw_output, dim=-1)[0]
            classes = ["Benigno", "Maligno"]
            pred_idx  = probs.argmax().item()
            pred_class = classes[pred_idx] if pred_idx < len(classes) else "Desconocido"
            confidence = probs[pred_idx].item()

            report["classification"] = {
                "predicted_class":  pred_class,
                "confidence":       round(confidence * 100, 2),
                "probabilities":    {c: round(p.item() * 100, 2)
                                     for c, p in zip(classes, probs)},
            }

            # Estimar BIRADS desde confianza de malignidad
            malignancy_prob = probs[1].item() if len(probs) > 1 else 0
            birads_cat = self._estimate_birads_from_prob(malignancy_prob)
            report["birads"]      = birads_cat
            report["birads_info"] = BIRADS_CATEGORIES.get(birads_cat, {})

        # ── Detector (lista de dicts: boxes, scores, labels) ────────
        elif isinstance(raw_output, list) and len(raw_output) > 0:
            det = raw_output[0]  # batch[0]
            boxes   = det.get("boxes",   [])
            scores  = det.get("scores",  [])
            labels  = det.get("labels",  [])
            classes = ["__background__", "mass", "calcification"]

            for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                if score < confidence_threshold:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                w_px = x2 - x1
                h_px = y2 - y1
                diameter_px  = max(w_px, h_px)
                diameter_mm  = diameter_px * pixel_spacing
                area_mm2     = w_px * h_px * (pixel_spacing ** 2)
                class_name   = classes[int(label)] if int(label) < len(classes) else "lesion"

                report["detections"].append({
                    "id":           i + 1,
                    "class":        class_name,
                    "confidence":   round(float(score) * 100, 2),
                    "bbox_px":      [round(x1), round(y1), round(x2), round(y2)],
                    "width_px":     round(w_px),
                    "height_px":    round(h_px),
                    "diameter_mm":  round(diameter_mm, 2),
                    "area_mm2":     round(area_mm2, 2),
                    "center_x_px":  round((x1 + x2) / 2),
                    "center_y_px":  round((y1 + y2) / 2),
                })

            # Estimar BIRADS desde detecciones
            if report["detections"]:
                max_conf = max(d["confidence"] for d in report["detections"]) / 100
                birads_cat = self._estimate_birads_from_prob(max_conf)
            else:
                birads_cat = 1
            report["birads"]      = birads_cat
            report["birads_info"] = BIRADS_CATEGORIES.get(birads_cat, {})

        # ── Texto del reporte ──────────────────────────────────────
        report["report_text"] = self._build_report_text(report)
        return report

    # ──────────────────────────────────────────
    # Metadatos
    # ──────────────────────────────────────────

    def get_supported_models(self) -> List[str]:
        return list(PRETRAINED_MODELS.keys())

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name":        "CBIS-DDSM",
            "full_name":   "Curated Breast Imaging Subset of DDSM",
            "source":      "The Cancer Imaging Archive (TCIA)",
            "hf_dataset":  "matthieulel/cbis-ddsm",
            "classes":     ["mass_benign", "mass_malignant", "calc_benign", "calc_malignant"],
            "modality":    "Mamografía digital",
            "size_approx": "~163 GB completo / ~10 GB subconjunto",
            "url":         "https://www.cancerimagingarchive.net/collection/cbis-ddsm/",
        }

    # ──────────────────────────────────────────
    # Augmentaciones de entrenamiento
    # ──────────────────────────────────────────

    def get_train_transforms(self, split: str = "train"):
        if split == "train":
            return T.Compose([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.3),
                T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                T.ToTensor(),
                T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ])
        else:
            return T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
            ])

    def get_loss_fn(self):
        import torch.nn as nn
        # Weighted BCE para dataset desbalanceado (más sanos que enfermos)
        return nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.5]))

    def compute_metrics(self, preds, targets) -> Dict[str, float]:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score,
        )
        import numpy as np
        preds_np   = np.array(preds)
        targets_np = np.array(targets)
        return {
            "accuracy":  round(accuracy_score(targets_np, preds_np), 4),
            "precision": round(precision_score(targets_np, preds_np, zero_division=0), 4),
            "recall":    round(recall_score(targets_np, preds_np, zero_division=0), 4),
            "f1":        round(f1_score(targets_np, preds_np, zero_division=0), 4),
        }

    # ──────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────

    @staticmethod
    def _estimate_birads_from_prob(malignancy_prob: float) -> int:
        """Heurística simple de BIRADS basada en probabilidad de malignidad.
        Esta es una estimación — no reemplaza criterios radiológicos formales.
        """
        if malignancy_prob < 0.02:  return 1
        if malignancy_prob < 0.10:  return 2
        if malignancy_prob < 0.30:  return 3
        if malignancy_prob < 0.60:  return 4
        if malignancy_prob < 0.95:  return 5
        return 6

    @staticmethod
    def _build_report_text(report: Dict) -> str:
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║     REPORTE DIAGNÓSTICO — MammoAI            ║",
            "╚══════════════════════════════════════════════╝",
            "",
        ]

        clf = report.get("classification", {})
        if clf:
            lines += [
                f"🔬 CLASIFICACIÓN: {clf.get('predicted_class', 'N/A')}",
                f"   Confianza: {clf.get('confidence', 0):.1f}%",
                f"   Probabilidades: {clf.get('probabilities', {})}",
                "",
            ]

        birads = report.get("birads")
        birads_info = report.get("birads_info", {})
        if birads is not None:
            lines += [
                f"📋 BIRADS ESTIMADO: {birads_info.get('label', f'BIRADS {birads}')}",
                f"   {birads_info.get('meaning', '')}",
                "",
            ]

        dets = report.get("detections", [])
        if dets:
            lines.append(f"🎯 LESIONES DETECTADAS: {len(dets)}")
            for d in dets:
                lines += [
                    f"",
                    f"   Lesión #{d['id']} — {d['class'].upper()}",
                    f"   ├─ Confianza:     {d['confidence']:.1f}%",
                    f"   ├─ Diámetro máx:  {d['diameter_mm']:.2f} mm",
                    f"   ├─ Área:          {d['area_mm2']:.2f} mm²",
                    f"   ├─ Posición:      x={d['center_x_px']}px, y={d['center_y_px']}px",
                    f"   └─ BBox (px):     {d['bbox_px']}",
                ]
        else:
            lines.append("✅ No se detectaron lesiones sospechosas.")

        lines += [
            "",
            "─" * 48,
            "⚠️  AVISO: Este reporte es una herramienta de apoyo.",
            "   No reemplaza el diagnóstico de un radiólogo certificado.",
            "─" * 48,
        ]
        return "\n".join(lines)


# ── Auto-registro al importar ──────────────────────────────────────
TaskRegistry.register_task("breast_cancer", BreastCancerTask)
