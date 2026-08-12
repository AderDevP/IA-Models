"""
core/base_model.py — Clase Base Abstracta para Modelos de IA
=============================================================
Toda integración de modelo (EfficientNet, ViT, ConvNeXt, Faster R-CNN,
DETR, etc.) debe heredar de BaseModel e implementar los métodos
abstractos definidos aquí.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class BaseModel(ABC):
    """Interfaz genérica para modelos de IA médica.

    Subclases implementan:
      - build() → nn.Module
      - forward(x) → raw predictions
      - load_weights(path) / save_weights(path)
      - get_gradcam_target_layer() → nn.Module
      - export(format, output_path) — ONNX / SafeTensors / .pth
    """

    def __init__(self, model_id: str, device: str = "cpu"):
        self.model_id = model_id
        self.device   = torch.device(device)
        self._model: Optional[nn.Module] = None
        self._is_loaded: bool = False

    # ──────────────────────────────────────────
    # Construcción y carga
    # ──────────────────────────────────────────
    @abstractmethod
    def build(self, num_classes: int = 2, pretrained: bool = True) -> nn.Module:
        """Construye la arquitectura del modelo."""
        ...

    def load_weights(self, weights_path: Path) -> None:
        """Carga pesos desde archivo .pth / .pt."""
        if self._model is None:
            raise RuntimeError("Llama a build() antes de load_weights().")
        state = torch.load(weights_path, map_location=self.device)
        # Soporta checkpoints con clave 'model_state_dict' o directo
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self._model.load_state_dict(state, strict=False)
        self._is_loaded = True

    def save_weights(self, output_path: Path, extra_meta: Optional[Dict] = None) -> Path:
        """Guarda pesos + metadatos en .pth."""
        payload = {
            "model_state_dict": self._model.state_dict(),
            "model_id":         self.model_id,
            "meta":             extra_meta or {},
        }
        torch.save(payload, output_path)
        return output_path

    # ──────────────────────────────────────────
    # Inferencia
    # ──────────────────────────────────────────
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Any:
        """Pasa el tensor por el modelo y retorna salida cruda."""
        ...

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Any:
        """Versión no-grad de forward — usar en inferencia."""
        self._model.eval()
        return self.forward(x.to(self.device))

    # ──────────────────────────────────────────
    # Grad-CAM
    # ──────────────────────────────────────────
    @abstractmethod
    def get_gradcam_target_layer(self) -> nn.Module:
        """Retorna la capa objetivo para Grad-CAM."""
        ...

    # ──────────────────────────────────────────
    # Exportación
    # ──────────────────────────────────────────
    def export_onnx(self, output_path: Path, input_shape: Tuple = (1, 3, 224, 224)) -> Path:
        """Exporta el modelo a formato ONNX."""
        import torch.onnx
        dummy = torch.randn(*input_shape).to(self.device)
        self._model.eval()
        torch.onnx.export(
            self._model, dummy, str(output_path),
            opset_version=17,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
        return output_path

    def export_safetensors(self, output_path: Path) -> Path:
        """Exporta pesos en formato SafeTensors."""
        try:
            from safetensors.torch import save_file
            tensors = {k: v.contiguous() for k, v in self._model.state_dict().items()}
            save_file(tensors, str(output_path))
        except ImportError:
            raise ImportError("Instala: pip install safetensors")
        return output_path

    # ──────────────────────────────────────────
    # Propiedades
    # ──────────────────────────────────────────
    @property
    def model(self) -> Optional[nn.Module]:
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def to(self, device: str) -> "BaseModel":
        self.device = torch.device(device)
        if self._model is not None:
            self._model = self._model.to(self.device)
        return self

    def __repr__(self) -> str:
        status = "cargado" if self._is_loaded else "sin pesos"
        return f"<Model: {self.model_id} [{status}] on {self.device}>"
