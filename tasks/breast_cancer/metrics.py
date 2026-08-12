"""tasks/breast_cancer/metrics.py — Métricas de Evaluación para Mamografías"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Calcula métricas completas de clasificación binaria.

    Args:
        y_true:  Etiquetas verdaderas (0=Benigno, 1=Maligno)
        y_pred:  Predicciones del modelo
        y_proba: Probabilidades de la clase positiva (para AUC-ROC)

    Returns:
        Dict con accuracy, precision, recall, f1, specificity, AUC-ROC
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)

    metrics = {
        "accuracy":    round(float(accuracy_score(y_true_np, y_pred_np)), 4),
        "precision":   round(float(precision_score(y_true_np, y_pred_np, zero_division=0)), 4),
        "recall":      round(float(recall_score(y_true_np, y_pred_np, zero_division=0)), 4),
        "f1":          round(float(f1_score(y_true_np, y_pred_np, zero_division=0)), 4),
    }

    # Especificidad (TN / (TN + FP))
    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np, labels=[0, 1]).ravel()
    metrics["specificity"] = round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4)
    metrics["sensitivity"] = metrics["recall"]  # alias clínico

    # AUC-ROC (requiere probabilidades)
    if y_proba is not None:
        try:
            metrics["auc_roc"] = round(float(roc_auc_score(y_true_np, y_proba)), 4)
        except ValueError:
            metrics["auc_roc"] = 0.0

    return metrics


def compute_detection_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Calcula mAP@0.5 y métricas de detección.

    Args:
        predictions:   Lista de dicts con 'bbox' y 'score'
        ground_truths: Lista de dicts con 'bbox'
        iou_threshold: Umbral IoU para considerar TP

    Returns:
        Dict con precision, recall, mAP@0.5
    """
    if not predictions or not ground_truths:
        return {"precision": 0.0, "recall": 0.0, "mAP50": 0.0}

    tp = 0
    fp = 0
    fn = len(ground_truths)

    matched_gt = set()
    # Ordenar por score descendente
    preds_sorted = sorted(predictions, key=lambda x: x.get("score", 0), reverse=True)

    for pred in preds_sorted:
        best_iou = 0.0
        best_gt_idx = -1
        for i, gt in enumerate(ground_truths):
            if i in matched_gt:
                continue
            iou = _compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            tp += 1
            fn -= 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "mAP50":     round((precision + recall) / 2, 4),   # aproximación simple
        "tp":        tp, "fp": fp, "fn": fn,
    }


def _compute_iou(box1: List[float], box2: List[float]) -> float:
    """Calcula Intersection over Union entre dos bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
