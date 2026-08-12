"""
core/base_task.py — Clase Base Abstracta para Tareas Médicas
=============================================================
Toda nueva tarea médica (cáncer de mama, nódulos pulmonares,
lesiones de piel, etc.) debe heredar de BaseTask e implementar
los métodos abstractos definidos aquí.

Esto garantiza una interfaz uniforme para el Dashboard y el motor
de inferencia, independientemente del tipo de tarea.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image


class BaseTask(ABC):
    """Interfaz uniforme para cualquier tarea de análisis médico.

    Subclases deben implementar:
      - task_id, task_name, task_description (properties)
      - preprocess(image) → tensor
      - postprocess(raw_output, metadata) → dict
      - get_supported_models() → list[str]
      - get_dataset_info() → dict
    """

    # ──────────────────────────────────────────
    # Identidad de la tarea
    # ──────────────────────────────────────────
    @property
    @abstractmethod
    def task_id(self) -> str:
        """Identificador único, ej. 'breast_cancer'."""
        ...

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Nombre legible, ej. '🎗️ Cáncer de Mama'."""
        ...

    @property
    @abstractmethod
    def task_description(self) -> str:
        """Descripción breve de la tarea médica."""
        ...

    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """Formatos de imagen soportados, ej. ['.dcm', '.png']."""
        ...

    # ──────────────────────────────────────────
    # Pipeline de inferencia
    # ──────────────────────────────────────────
    @abstractmethod
    def preprocess(self, image: Image.Image, **kwargs) -> Any:
        """Convierte PIL Image → tensor listo para el modelo.

        Args:
            image: Imagen PIL en modo RGB
            **kwargs: Parámetros opcionales (pixel_spacing, etc.)
        Returns:
            Tensor PyTorch listo para inferencia
        """
        ...

    @abstractmethod
    def postprocess(
        self,
        raw_output: Any,
        original_image: Image.Image,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Convierte salida del modelo → reporte estructurado.

        Returns:
            dict con claves estándar:
              - 'detections': list[dict] — bounding boxes y scores
              - 'classification': dict — clase y confianza
              - 'birads': int — categoría BIRADS estimada
              - 'report_text': str — texto del informe
        """
        ...

    # ──────────────────────────────────────────
    # Metadatos de modelos y dataset
    # ──────────────────────────────────────────
    @abstractmethod
    def get_supported_models(self) -> List[str]:
        """IDs de modelos compatibles con esta tarea (de config.PRETRAINED_MODELS)."""
        ...

    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """Información sobre el dataset de entrenamiento/validación."""
        ...

    # ──────────────────────────────────────────
    # Hooks de entrenamiento (opcionales)
    # ──────────────────────────────────────────
    def get_train_transforms(self, split: str = "train"):
        """Augmentaciones para entrenamiento (override en subclases)."""
        return None

    def get_loss_fn(self):
        """Función de pérdida (override en subclases)."""
        return None

    def compute_metrics(self, preds: Any, targets: Any) -> Dict[str, float]:
        """Calcula métricas de evaluación (override en subclases)."""
        return {}

    # ──────────────────────────────────────────
    # Utilidades comunes
    # ──────────────────────────────────────────
    def validate_image(self, file_path: Path) -> bool:
        """Verifica que el archivo es un formato soportado."""
        return file_path.suffix.lower() in self.supported_formats

    def __repr__(self) -> str:
        return f"<Task: {self.task_id} — {self.task_name}>"
