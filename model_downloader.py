"""
model_downloader.py — Descarga Automática de Modelos Preentrenados
===================================================================
Descarga modelos desde HuggingFace Hub, timm y torchvision.
Sin YOLO — arquitecturas modernas: EfficientNet, ConvNeXt, ViT,
Faster R-CNN, DETR.

Uso:
    from model_downloader import ModelDownloader
    dl = ModelDownloader()
    dl.download("efficientnet_b4_cbis")
    dl.download_all()
"""

from __future__ import annotations
import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

from config import MODELS_DIR, PRETRAINED_MODELS

logger = logging.getLogger(__name__)


class ModelDownloader:
    """Gestiona la descarga e instalación de modelos preentrenados."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────
    # API pública
    # ──────────────────────────────────────────

    def list_available(self) -> Dict[str, Dict]:
        """Retorna el catálogo de modelos disponibles para descargar."""
        return PRETRAINED_MODELS

    def list_installed(self) -> List[str]:
        """Retorna IDs de modelos ya descargados en models_dir."""
        installed = []
        for model_id, meta in PRETRAINED_MODELS.items():
            path = self.models_dir / meta["local_filename"]
            if path.exists():
                installed.append(model_id)
        return installed

    def is_installed(self, model_id: str) -> bool:
        meta = PRETRAINED_MODELS.get(model_id)
        if not meta:
            return False
        return (self.models_dir / meta["local_filename"]).exists()

    def get_model_path(self, model_id: str) -> Optional[Path]:
        meta = PRETRAINED_MODELS.get(model_id)
        if not meta:
            return None
        path = self.models_dir / meta["local_filename"]
        return path if path.exists() else None

    def download(
        self,
        model_id: str,
        progress_callback: Optional[Callable[[str], None]] = None,
        force: bool = False,
    ) -> Path:
        """Descarga un modelo por su ID.

        Args:
            model_id: Clave en config.PRETRAINED_MODELS
            progress_callback: Función que recibe strings de progreso (para Gradio)
            force: Si True, re-descarga aunque ya esté instalado
        Returns:
            Path al archivo descargado
        """
        if model_id not in PRETRAINED_MODELS:
            raise ValueError(f"Modelo desconocido: '{model_id}'. "
                             f"Disponibles: {list(PRETRAINED_MODELS.keys())}")

        meta = PRETRAINED_MODELS[model_id]
        target = self.models_dir / meta["local_filename"]

        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        if target.exists() and not force:
            log(f"✅ '{meta['name']}' ya está instalado en {target}")
            return target

        log(f"⬇️  Descargando '{meta['name']}' desde {meta['source']}...")

        source = meta["source"]

        if source == "huggingface":
            return self._download_huggingface(model_id, meta, target, log)
        elif source == "timm":
            return self._download_timm(model_id, meta, target, log)
        elif source == "torchvision":
            return self._download_torchvision(model_id, meta, target, log)
        elif source == "custom":
            return self._create_custom_placeholder(model_id, meta, target, log)
        else:
            raise ValueError(f"Fuente desconocida: {source}")

    def download_all(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[Path]:
        """Descarga todos los modelos del catálogo que no estén instalados."""
        paths = []
        for model_id in PRETRAINED_MODELS:
            try:
                p = self.download(model_id, progress_callback)
                paths.append(p)
            except Exception as e:
                logger.error(f"Error descargando '{model_id}': {e}")
                if progress_callback:
                    progress_callback(f"❌ Error en '{model_id}': {e}")
        return paths

    def delete(self, model_id: str) -> bool:
        """Elimina un modelo instalado."""
        path = self.get_model_path(model_id)
        if path and path.exists():
            path.unlink()
            logger.info(f"🗑️  Modelo '{model_id}' eliminado.")
            return True
        return False

    # ──────────────────────────────────────────
    # Descargadores por fuente
    # ──────────────────────────────────────────

    def _download_huggingface(self, model_id, meta, target, log) -> Path:
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
        except ImportError:
            raise ImportError("Instala: pip install huggingface_hub")

        hf_repo = meta["hf_repo"]
        log(f"   HuggingFace repo: {hf_repo}")

        try:
            # Intentar descargar archivo de pesos específico
            candidates = [
                "pytorch_model.bin",
                "model.safetensors",
                "model.pth",
                "weights/best.pt",
            ]
            downloaded = None
            for filename in candidates:
                try:
                    downloaded = hf_hub_download(
                        repo_id=hf_repo,
                        filename=filename,
                        cache_dir=str(self.models_dir / "_hf_cache"),
                    )
                    break
                except Exception:
                    continue

            if downloaded:
                shutil.copy2(downloaded, target)
                log(f"✅ Guardado en {target}")
                return target
            else:
                # Si no se encuentra archivo individual, guardar el repo como snapshot
                # y crear un estado de "backbone preentrenado"
                log(f"⚠️  No se encontró archivo de pesos directo en {hf_repo}.")
                log(f"   Guardando backbone pretrain ImageNet como inicialización...")
                return self._save_timm_backbone(meta, target, log)

        except Exception as e:
            log(f"⚠️  Error HuggingFace: {e}. Descargando backbone base...")
            return self._save_timm_backbone(meta, target, log)

    def _download_timm(self, model_id, meta, target, log) -> Path:
        try:
            import timm
        except ImportError:
            raise ImportError("Instala: pip install timm")

        timm_id = meta.get("timm_id", "efficientnet_b4")
        log(f"   timm model: {timm_id} (pretrained=True)")

        model = timm.create_model(timm_id, pretrained=True, num_classes=2)

        payload = {
            "model_state_dict": model.state_dict(),
            "model_id":         model_id,
            "timm_id":          timm_id,
            "source":           "timm",
            "pretrained_on":    meta.get("pretrained_on", "ImageNet"),
            "num_classes":      2,
            "architecture":     meta.get("architecture"),
            "classes":          meta.get("classes", ["Benigno", "Maligno"]),
        }
        torch.save(payload, target)
        log(f"✅ '{meta['name']}' guardado en {target} "
            f"({target.stat().st_size / 1e6:.1f} MB)")
        return target

    def _download_torchvision(self, model_id, meta, target, log) -> Path:
        import torchvision.models as tv_models

        tv_id = meta.get("tv_model", "fasterrcnn_resnet50_fpn_v2")
        log(f"   torchvision model: {tv_id} (pretrained COCO)")

        # Cargar modelo con pesos COCO
        weights_attr = None
        tv_weights_map = {
            "fasterrcnn_resnet50_fpn_v2": "FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT",
        }
        weights_str = tv_weights_map.get(tv_id)

        if weights_str:
            from torchvision.models.detection import (
                FasterRCNN_ResNet50_FPN_V2_Weights,
                fasterrcnn_resnet50_fpn_v2,
            )
            model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            )
        else:
            fn = getattr(tv_models, tv_id, None)
            if fn is None:
                raise ValueError(f"Modelo torchvision no encontrado: {tv_id}")
            model = fn(pretrained=True)

        payload = {
            "model_state_dict": model.state_dict(),
            "model_id":         model_id,
            "tv_model":         tv_id,
            "source":           "torchvision",
            "pretrained_on":    meta.get("pretrained_on", "COCO"),
            "architecture":     meta.get("architecture"),
            "classes":          meta.get("classes"),
        }
        torch.save(payload, target)
        log(f"✅ '{meta['name']}' guardado en {target} "
            f"({target.stat().st_size / 1e6:.1f} MB)")
        return target

    def _create_custom_placeholder(self, model_id, meta, target, log) -> Path:
        """Para modelos 'custom' crea un checkpoint placeholder + descarga backbone."""
        log(f"   Modelo custom — inicializando con backbone EfficientNet-B4...")
        meta_copy = dict(meta)
        meta_copy["timm_id"] = "efficientnet_b4"
        return self._download_timm(model_id, meta_copy, target, log)

    def _save_timm_backbone(self, meta, target, log) -> Path:
        """Fallback: descarga el backbone timm correspondiente."""
        try:
            import timm
            timm_id = meta.get("timm_id", "efficientnet_b4")
            model = timm.create_model(timm_id, pretrained=True, num_classes=2)
            torch.save({
                "model_state_dict": model.state_dict(),
                "source": "timm_fallback",
                "timm_id": timm_id,
            }, target)
            log(f"✅ Backbone {timm_id} guardado en {target}")
            return target
        except Exception as e:
            # Último recurso: crear checkpoint vacío
            log(f"⚠️  Fallback backbone falló: {e}. Guardando placeholder vacío.")
            torch.save({"model_state_dict": {}, "source": "placeholder"}, target)
            return target

    # ──────────────────────────────────────────
    # Reporte de estado
    # ──────────────────────────────────────────

    def status_report(self) -> str:
        """Genera un reporte legible del estado de instalación."""
        lines = ["=" * 55, "  MammoAI — Estado de Modelos", "=" * 55]
        for model_id, meta in PRETRAINED_MODELS.items():
            installed = self.is_installed(model_id)
            status = "✅ Instalado" if installed else "⬜ No instalado"
            size_str = ""
            if installed:
                p = self.models_dir / meta["local_filename"]
                size_str = f"  ({p.stat().st_size / 1e6:.1f} MB)"
            lines.append(
                f"  {status:<16} | {meta['name'][:35]:<35}{size_str}"
            )
        lines.append("=" * 55)
        return "\n".join(lines)
