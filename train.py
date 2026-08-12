"""
train.py — Motor de Entrenamiento / Fine-Tuning
================================================
Entrena o hace fine-tuning de modelos para tareas médicas.
Soporta:
  - Clasificación binaria: EfficientNet-B4, ConvNeXt-Small, ViT-Base
  - Detección: Faster R-CNN (próximamente)
  - Dataset CBIS-DDSM (~10 GB) o datasets propios
  - Métricas en tiempo real para el Dashboard Gradio
  - Exportación automática a .pth / .onnx / SafeTensors
"""

from __future__ import annotations
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast

from config import (
    MODELS_DIR,
    LOGS_DIR,
    TRAIN_DEFAULTS,
    PRETRAINED_MODELS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Clase principal de entrenamiento
# ──────────────────────────────────────────────────────────────────

class Trainer:
    """Entrena o hace fine-tuning de un modelo de clasificación/detección médica.

    Uso:
        trainer = Trainer(model_id="efficientnet_b4_cbis", task=breast_cancer_task)
        for metrics in trainer.train_generator(data_dir="datasets/cbis_ddsm"):
            print(metrics)
    """

    def __init__(
        self,
        model_id: str,
        task,
        device: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        self.model_id   = model_id
        self.task       = task
        self.device     = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_dir = output_dir or MODELS_DIR
        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "precision":  [], "recall":   [], "f1": [],
        }
        self._stop_requested = False
        self.best_val_acc    = 0.0
        self.best_model_path: Optional[Path] = None

    # ──────────────────────────────────────────
    # Generador para streaming de métricas a Gradio
    # ──────────────────────────────────────────

    def train_generator(
        self,
        data_source: str | Path,
        use_cbis_ddsm: bool = False,
        epochs: int = TRAIN_DEFAULTS["epochs"],
        batch_size: int = TRAIN_DEFAULTS["batch_size"],
        learning_rate: float = TRAIN_DEFAULTS["learning_rate"],
        weight_decay: float = TRAIN_DEFAULTS["weight_decay"],
        patience: int = TRAIN_DEFAULTS["patience"],
        val_split: float = TRAIN_DEFAULTS["val_split"],
        mixed_precision: bool = TRAIN_DEFAULTS["mixed_precision"],
        freeze_backbone: bool = True,
        freeze_until_epoch: int = 3,
    ) -> Generator[Dict, None, None]:
        """Ejecuta el entrenamiento y emite métricas por epoch.

        Yields:
            Dict con métricas del epoch actual para mostrar en Gradio.
        """
        self._stop_requested = False
        yield {"status": f"🔧 Preparando entrenamiento en {self.device}..."}

        # ── Cargar modelo ─────────────────────────────────────────
        try:
            model, meta = self._build_model()
            model = model.to(self.device)
            yield {"status": f"✅ Modelo {meta['name']} listo."}
        except Exception as e:
            yield {"status": f"❌ Error cargando modelo: {e}"}
            return

        # ── Cargar dataset ────────────────────────────────────────
        try:
            dataloaders = self._prepare_data(
                data_source, use_cbis_ddsm, batch_size, val_split
            )
            n_train = len(dataloaders["train"].dataset)
            n_val   = len(dataloaders["val"].dataset)
            yield {"status": f"📦 Dataset: {n_train} train, {n_val} val."}
        except Exception as e:
            yield {"status": f"❌ Error en dataset: {e}"}
            return

        # ── Optimizador y scheduler ───────────────────────────────
        if freeze_backbone:
            self._freeze_backbone(model, meta)
            yield {"status": f"🧊 Backbone congelado (primeras {freeze_until_epoch} épocas)."}

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = self.task.get_loss_fn() or nn.CrossEntropyLoss()
        criterion = criterion.to(self.device)
        scaler    = GradScaler(enabled=mixed_precision and self.device.type == "cuda")

        # ── Loop de entrenamiento ─────────────────────────────────
        no_improve_count = 0
        log_path = LOGS_DIR / f"train_{self.model_id}_{int(time.time())}.jsonl"

        for epoch in range(1, epochs + 1):
            if self._stop_requested:
                yield {"status": "⏹️  Entrenamiento detenido por el usuario."}
                break

            # Descongelar backbone después de freeze_until_epoch
            if freeze_backbone and epoch == freeze_until_epoch + 1:
                self._unfreeze_all(model)
                optimizer.add_param_group({
                    "params": [p for p in model.parameters() if not p.requires_grad],
                    "lr": learning_rate * 0.1,
                })
                yield {"status": "🔓 Backbone descongelado — fine-tuning completo."}

            # ── Train epoch ───────────────────────────────────────
            train_metrics = self._train_epoch(
                model, dataloaders["train"], optimizer, criterion, scaler, mixed_precision
            )

            # ── Validation epoch ──────────────────────────────────
            val_metrics = self._val_epoch(model, dataloaders["val"], criterion)

            # ── Actualizar historial ──────────────────────────────
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            scheduler.step()

            # ── Guardar mejor modelo ──────────────────────────────
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.best_model_path = self._save_checkpoint(model, epoch, val_metrics)
                no_improve_count = 0
                improved = "✨ Nuevo mejor modelo!"
            else:
                no_improve_count += 1
                improved = f"(sin mejora: {no_improve_count}/{patience})"

            # ── Emitir métricas a Gradio ──────────────────────────
            epoch_result = {
                "epoch":       epoch,
                "epochs":      epochs,
                "train_loss":  round(train_metrics["loss"], 4),
                "val_loss":    round(val_metrics["loss"], 4),
                "train_acc":   round(train_metrics["accuracy"] * 100, 2),
                "val_acc":     round(val_metrics["accuracy"] * 100, 2),
                "lr":          round(scheduler.get_last_lr()[0], 6),
                "improved":    improved,
                "history":     dict(self.history),
                "status": (
                    f"Epoch {epoch}/{epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} → val: {val_metrics['loss']:.4f} | "
                    f"Acc: {train_metrics['accuracy']*100:.1f}% → val: {val_metrics['accuracy']*100:.1f}% | "
                    f"{improved}"
                ),
            }

            # Log a disco
            with open(log_path, "a") as f:
                f.write(json.dumps(epoch_result) + "\n")

            yield epoch_result

            # ── Early stopping ────────────────────────────────────
            if no_improve_count >= patience:
                yield {"status": f"⏹️  Early stopping — sin mejora en {patience} épocas."}
                break

        yield {
            "status": (
                f"✅ Entrenamiento completado. "
                f"Mejor val_acc: {self.best_val_acc*100:.2f}% "
                f"→ Modelo: {self.best_model_path}"
            ),
            "best_model_path": str(self.best_model_path) if self.best_model_path else None,
            "history": dict(self.history),
        }

    def stop(self) -> None:
        """Detiene el entrenamiento en el siguiente epoch."""
        self._stop_requested = True

    # ──────────────────────────────────────────
    # Helpers de entrenamiento
    # ──────────────────────────────────────────

    def _train_epoch(self, model, loader, optimizer, criterion, scaler, mixed_precision) -> Dict:
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, labels in loader:
            imgs   = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=mixed_precision and self.device.type == "cuda"):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN_DEFAULTS["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)

        return {"loss": total_loss / total, "accuracy": correct / total}

    @torch.no_grad()
    def _val_epoch(self, model, loader, criterion) -> Dict:
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        for imgs, labels in loader:
            imgs   = imgs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = model(imgs)
            loss    = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += imgs.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = {"loss": total_loss / total, "accuracy": correct / total}
        extra   = self.task.compute_metrics(all_preds, all_labels)
        metrics.update(extra)
        return metrics

    # ──────────────────────────────────────────
    # Dataset
    # ──────────────────────────────────────────

    def _prepare_data(self, data_source, use_cbis_ddsm, batch_size, val_split):
        from tasks.breast_cancer.dataset import (
            CBISDDSMDataset, CustomMammographyDataset, get_dataloaders
        )

        if use_cbis_ddsm:
            dataset = CBISDDSMDataset(
                split="train",
                transform=self.task.get_train_transforms("train"),
            )
        else:
            dataset = CustomMammographyDataset(
                data_dir=data_source,
                transform=self.task.get_train_transforms("train"),
            )

        return get_dataloaders(
            dataset,
            val_split=val_split,
            batch_size=batch_size,
            num_workers=TRAIN_DEFAULTS["num_workers"],
            seed=TRAIN_DEFAULTS["seed"],
        )

    # ──────────────────────────────────────────
    # Modelo
    # ──────────────────────────────────────────

    def _build_model(self) -> Tuple[nn.Module, Dict]:
        from detector import load_model
        return load_model(self.model_id, device=str(self.device))

    def _freeze_backbone(self, model: nn.Module, meta: Dict) -> None:
        """Congela todas las capas excepto el clasificador final."""
        arch = meta["architecture"].lower()
        frozen = 0
        for name, param in model.named_parameters():
            if "classifier" not in name and "head" not in name and "fc" not in name:
                param.requires_grad = False
                frozen += 1
        logger.info(f"Congeladas {frozen} capas del backbone.")

    def _unfreeze_all(self, model: nn.Module) -> None:
        for param in model.parameters():
            param.requires_grad = True

    def _save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict) -> Path:
        fname = self.output_dir / f"{self.model_id}_epoch{epoch:03d}_acc{metrics['accuracy']:.4f}.pth"
        torch.save({
            "epoch":             epoch,
            "model_state_dict":  model.state_dict(),
            "model_id":          self.model_id,
            "val_accuracy":      metrics["accuracy"],
            "metrics":           metrics,
        }, fname)
        logger.info(f"Checkpoint guardado: {fname}")
        return fname

    # ──────────────────────────────────────────
    # Exportación
    # ──────────────────────────────────────────

    def export_model(
        self,
        output_path: Optional[Path] = None,
        format: str = "pth",   # "pth", "onnx", "safetensors"
    ) -> Path:
        """Exporta el mejor modelo al formato especificado."""
        if not self.best_model_path or not self.best_model_path.exists():
            raise RuntimeError("No hay checkpoint de mejor modelo. Entrena primero.")

        model, meta = self._build_model()
        checkpoint  = torch.load(self.best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        model.eval()

        out_path = output_path or self.output_dir / f"{self.model_id}_export.{format}"

        if format == "pth":
            torch.save(checkpoint, out_path)
        elif format == "onnx":
            input_size = meta.get("input_size", (224, 224))
            dummy = torch.randn(1, 3, *input_size).to(self.device)
            torch.onnx.export(
                model, dummy, str(out_path),
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
        elif format == "safetensors":
            from safetensors.torch import save_file
            tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
            save_file(tensors, str(out_path))
        else:
            raise ValueError(f"Formato no soportado: {format}")

        logger.info(f"Modelo exportado a {out_path}")
        return out_path
