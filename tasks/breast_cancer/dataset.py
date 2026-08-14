"""
tasks/breast_cancer/dataset.py — Dataset CBIS-DDSM
===================================================
Carga y prepara subconjuntos del dataset CBIS-DDSM desde HuggingFace.
Límite configurable de ~10 GB para entornos Colab.
"""

from __future__ import annotations
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

from config import DATASETS_DIR, CBIS_DDSM_HF_DATASET, CBIS_DDSM_MAX_GB, TRAIN_DEFAULTS

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Dataset CBIS-DDSM desde HuggingFace
# ──────────────────────────────────────────────────────────────────

class CBISDDSMDataset(Dataset):
    """Dataset para clasificación binaria (Benigno / Maligno) en CBIS-DDSM.

    Descarga un subconjunto desde HuggingFace Datasets.
    Límite: CBIS_DDSM_MAX_GB (~10 GB).
    """

    CLASS_MAP = {
        "BENIGN":    0,
        "MALIGNANT": 1,
        "BENIGN_WITHOUT_CALLBACK": 0,
    }

    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        cache_dir: Optional[Path] = None,
    ):
        self.split      = split
        self.transform  = transform
        self.cache_dir  = cache_dir or (DATASETS_DIR / "cbis_ddsm")
        self.samples: List[Tuple[Path, int]] = []

        self._load_or_download(max_samples)

    def _load_or_download(self, max_samples: Optional[int]) -> None:
        """Carga desde caché local o descarga desde HuggingFace."""
        cache_split = self.cache_dir / self.split
        if cache_split.exists() and any(cache_split.rglob("*.png")):
            logger.info(f"Usando caché local CBIS-DDSM: {cache_split}")
            self._load_from_cache(cache_split, max_samples)
            return

        logger.info(f"Descargando CBIS-DDSM (split={self.split}) desde Kaggle...")
        self._download_dataset(max_samples)

    def _download_dataset(self, max_samples: Optional[int]) -> None:
        """Descarga el dataset usando kagglehub (awsaf49/cbis-ddsm-breast-cancer-image-dataset)"""
        logger.info("Descargando CBIS-DDSM desde Kaggle...")
        try:
            import kagglehub
        except ImportError:
            import subprocess, sys
            logger.warning("Falta kagglehub. Instalando automáticamente...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
            import kagglehub

        try:
            # Descargar dataset (puede tardar unos minutos si es de 5GB)
            path_str = kagglehub.dataset_download("awsaf49/cbis-ddsm-breast-cancer-image-dataset")
            dataset_path = Path(path_str)
            logger.info(f"Dataset descargado en: {dataset_path}")

            # Crear directorios locales de cache
            save_dir = self.cache_dir / self.split
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / "benign").mkdir(exist_ok=True)
            (save_dir / "malignant").mkdir(exist_ok=True)

            # Buscar todas las imágenes (png, jpg, jpeg) en la carpeta descargada
            img_files = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                img_files.extend(list(dataset_path.rglob(ext)))

            if not img_files:
                logger.error("No se encontraron imágenes en el dataset de Kaggle.")
                self._create_synthetic_cbis_ddsm(100)
                return

            import shutil
            saved = 0
            for img_file in img_files:
                if max_samples and saved >= max_samples:
                    break

                # Determinar clase basada en el nombre del archivo o ruta
                path_lower = str(img_file).lower()
                if "malignant" in path_lower:
                    label_name = "malignant"
                else:
                    label_name = "benign" # por defecto o si dice benign

                dest_path = save_dir / label_name / img_file.name
                
                # Evitar sobreescribir si el archivo ya existe (o si hay nombres duplicados)
                if dest_path.exists():
                    dest_path = save_dir / label_name / f"{saved}_{img_file.name}"

                shutil.copy2(img_file, dest_path)
                saved += 1
                
                if saved % 500 == 0:
                    logger.info(f"  Procesando imágenes Kaggle: {saved} copiadas...")

            logger.info(f"Dataset de Kaggle procesado: {saved} imágenes guardadas.")
            self._load_from_cache(save_dir, max_samples)

        except Exception as e:
            logger.error(f"Error descargando desde Kaggle: {e}")
            logger.warning("Generando dataset sintético como fallback...")
            self._create_synthetic_cbis_ddsm(100)

    def _create_synthetic_cbis_ddsm(self, num_samples: int = 100) -> None:
        """Crea un dataset sintético/demostrativo de mamografías para entrenamiento."""
        import numpy as np
        save_dir = self.cache_dir / self.split
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir / "benign").mkdir(exist_ok=True)
        (save_dir / "malignant").mkdir(exist_ok=True)

        logger.info(f"Generando {num_samples} mamografías sintéticas en {save_dir}...")
        for i in range(num_samples):
            label = 1 if i % 2 == 0 else 0
            label_name = "malignant" if label == 1 else "benign"

            base = np.random.normal(120, 25, (224, 224)).clip(0, 255).astype(np.uint8)
            if label == 1:
                y, x = np.ogrid[:224, :224]
                center_y, center_x = np.random.randint(60, 160), np.random.randint(60, 160)
                mask = ((x - center_x)**2 + (y - center_y)**2) <= 30**2
                base[mask] = np.clip(base[mask].astype(int) + 80, 0, 255).astype(np.uint8)

            img = Image.fromarray(base).convert("RGB")
            img_path = save_dir / label_name / f"synth_{i:04d}.png"
            img.save(img_path)

        self._load_from_cache(save_dir, num_samples)

    def _load_from_cache(self, cache_split: Path, max_samples: Optional[int]) -> None:
        for class_name, label in [("malignant", 1), ("benign", 0)]:
            class_dir = cache_split / class_name
            if not class_dir.exists():
                continue
            imgs = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
            for img_path in imgs:
                self.samples.append((img_path, label))

        random.shuffle(self.samples)
        if max_samples:
            self.samples = self.samples[:max_samples]

        logger.info(
            f"CBIS-DDSM [{self.split}]: {len(self.samples)} muestras "
            f"({sum(1 for _,l in self.samples if l==1)} malignas, "
            f"{sum(1 for _,l in self.samples if l==0)} benignas)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────
# Dataset para imágenes propias (carpetas con_cancer / sin_cancer)
# ──────────────────────────────────────────────────────────────────

class CustomMammographyDataset(Dataset):
    """Dataset desde carpetas personalizadas.

    Estructura esperada:
        data_dir/
            con_cancer/   → imágenes malignas (label=1)
            sin_cancer/   → imágenes benignas (label=0)

    O formato YOLO/COCO para detección (futuro).
    """

    EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".dcm"}

    def __init__(
        self,
        data_dir: str | Path,
        transform: Optional[Callable] = None,
        task: str = "classification",
    ):
        self.data_dir  = Path(data_dir)
        self.transform = transform
        self.task      = task
        self.samples: List[Tuple[Path, int]] = []

        self._scan_directory()

    def _scan_directory(self) -> None:
        # Mapeo flexible de nombres de carpetas
        class_dirs = {
            1: ["con_cancer", "malignant", "malignante", "positive", "positivo", "cancer"],
            0: ["sin_cancer", "benign",    "benigno",    "negative", "negativo", "normal", "sano"],
        }

        found = False
        for label, dir_names in class_dirs.items():
            for dirname in dir_names:
                class_dir = self.data_dir / dirname
                if class_dir.exists():
                    for img_path in class_dir.iterdir():
                        if img_path.suffix.lower() in self.EXTENSIONS:
                            self.samples.append((img_path, label))
                    found = True
                    break

        if not found:
            # Fallback: tratar cada subcarpeta como una clase
            subdirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
            for i, subdir in enumerate(sorted(subdirs)):
                for img_path in subdir.iterdir():
                    if img_path.suffix.lower() in self.EXTENSIONS:
                        self.samples.append((img_path, i))

        random.shuffle(self.samples)
        logger.info(
            f"Dataset personalizado: {len(self.samples)} muestras en {self.data_dir}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        if img_path.suffix.lower() == ".dcm":
            from dicom_utils import load_dicom
            array, _, _ = load_dicom(img_path)
            img = Image.fromarray(array).convert("RGB")
        else:
            img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────
# Factory de DataLoaders
# ──────────────────────────────────────────────────────────────────

def get_dataloaders(
    dataset: Dataset,
    val_split: float = 0.2,
    test_split: float = 0.1,
    batch_size: int = 16,
    num_workers: int = 2,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Divide un dataset en train/val/test y retorna DataLoaders."""
    total = len(dataset)
    n_test = int(total * test_split)
    n_val  = int(total * val_split)
    n_train = total - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return {
        "train": DataLoader(train_ds, shuffle=True,  **loader_kwargs),
        "val":   DataLoader(val_ds,   shuffle=False, **loader_kwargs),
        "test":  DataLoader(test_ds,  shuffle=False, **loader_kwargs),
    }
