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

        logger.info(f"Descargando CBIS-DDSM (split={self.split}) desde HuggingFace...")
        self._download_from_hf(max_samples)

    def _download_from_hf(self, max_samples: Optional[int]) -> None:
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Instala: pip install datasets")

        try:
            ds = load_dataset(
                CBIS_DDSM_HF_DATASET,
                split=self.split,
                cache_dir=str(self.cache_dir / "_hf_raw"),
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Error cargando HF dataset '{CBIS_DDSM_HF_DATASET}': {e}")
            logger.info("Intentando dataset alternativo: 'breast-cancer-cbis-ddsm'...")
            ds = load_dataset(
                "breast-cancer-cbis-ddsm",
                split=self.split,
                cache_dir=str(self.cache_dir / "_hf_raw"),
                trust_remote_code=True,
            )

        # Limitar samples para respetar el límite de ~10 GB
        if max_samples and len(ds) > max_samples:
            indices = random.sample(range(len(ds)), max_samples)
            ds = ds.select(indices)
        elif max_samples is None:
            # Estimar límite por GB (aprox 50KB por imagen PNG procesada)
            estimated_max = (CBIS_DDSM_MAX_GB * 1024 * 1024 * 1024) // (50 * 1024)
            if len(ds) > estimated_max:
                indices = random.sample(range(len(ds)), int(estimated_max))
                ds = ds.select(indices)

        # Guardar imágenes en caché local
        save_dir = self.cache_dir / self.split
        save_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(ds):
            label_raw = str(sample.get("pathology", sample.get("label", "BENIGN"))).upper()
            label = self.CLASS_MAP.get(label_raw, 0)
            label_dir = save_dir / ("malignant" if label == 1 else "benign")
            label_dir.mkdir(exist_ok=True)

            img = sample.get("image", sample.get("img"))
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            img = img.convert("RGB")
            img_path = label_dir / f"{i:06d}.png"
            img.save(img_path)

            if i % 500 == 0:
                logger.info(f"  Guardado {i}/{len(ds)} imágenes...")

        self._load_from_cache(save_dir, max_samples)

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
