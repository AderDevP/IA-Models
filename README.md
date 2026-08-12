# MammoAI 🔬

**Sistema Open Source de Inteligencia Artificial para Detección y Análisis Cuantitativo de Cáncer de Mama en Mamografías**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c)](https://pytorch.org)
[![Gradio](https://img.shields.io/badge/Gradio-4.x-orange)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AderDevP/IA-Models/blob/main/notebook.ipynb)

---

## 🎯 ¿Qué hace MammoAI?

MammoAI es una plataforma de IA médica escalable que permite:

- **Detectar y localizar** masas y calcificaciones en mamografías
- **Cuantificar lesiones**: diámetro en mm, área, coordenadas exactas
- **Estimar categoría BIRADS** (1-6) automáticamente
- **Visualizar** mapas de calor Grad-CAM para interpretabilidad
- **Entrenar modelos personalizados** con tu propio dataset o CBIS-DDSM (~10 GB)
- **Sincronizar** código y modelos con GitHub automáticamente

Todo desde un **Dashboard Gradio** — sin necesidad de escribir código.

---

## 🏗️ Arquitectura

```
cancerdemama/
├── app.py                     # Dashboard principal (Gradio Blocks)
├── config.py                  # Configuración global y catálogo de modelos
├── detector.py                # Motor de inferencia + Grad-CAM + BBoxes
├── train.py                   # Entrenamiento / Fine-tuning
├── model_downloader.py        # Descarga automática de modelos
├── dicom_utils.py             # Carga DICOM con pixel_spacing
├── git_utils.py               # Push automático a GitHub (HTTPS)
├── report_generator.py        # Exportación PDF / JSON
├── notebook.ipynb             # Google Colab — GPU + launch
├── requirements.txt
│
├── core/                      # ⚡ Framework escalable
│   ├── registry.py            # Registro de tareas médicas (plugin system)
│   ├── base_task.py           # Interfaz abstracta para tareas
│   └── base_model.py          # Interfaz abstracta para modelos
│
└── tasks/                     # 🏥 Tareas médicas (extensibles)
    └── breast_cancer/         # ✅ ACTIVA
        ├── __init__.py        # Implementación BreastCancerTask
        └── dataset.py         # Dataset CBIS-DDSM + datasets propios
    # (futuro) lung_nodule/
    # (futuro) skin_lesion/
    # (futuro) retinopathy/
```

### Arquitecturas de IA (sin YOLO)

| Modelo | Arquitectura | Tarea | Fuente |
|--------|-------------|-------|--------|
| EfficientNet-B4 | CNN eficiente | Clasificación | timm / ImageNet |
| ConvNeXt-Small | CNN moderna | Clasificación | timm / ImageNet |
| ViT-Base/16 | Vision Transformer | Clasificación | timm / ImageNet-21k |
| MammoScreen EfficientNetV2 | CNN | Clasificación | HuggingFace (CBIS-DDSM+RSNA) |
| Faster R-CNN ResNet-50 FPN | Two-stage detector | Detección | torchvision (COCO) |
| DETR ResNet-50 | Detection Transformer | Detección | HuggingFace (COCO) |

---

## 🚀 Inicio Rápido (Google Colab)

**La forma más fácil:** [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AderDevP/IA-Models/blob/main/notebook.ipynb)

1. Abre el notebook en Colab
2. Selecciona **GPU**: Entorno de ejecución → Cambiar tipo → T4/A100
3. Configura tu `GITHUB_TOKEN` en 🔑 Secrets
4. Ejecuta todas las celdas en orden
5. Haz clic en el **enlace público de Gradio** generado

---

## 💻 Ejecución Local

```bash
# Clonar repositorio
git clone https://github.com/AderDevP/IA-Models
cd IA-Models

# Instalar dependencias
pip install -r requirements.txt

# Lanzar dashboard
python app.py

# Con enlace público (share)
python app.py --share
```

---

## 🔬 Uso del Dashboard

### Pestaña 1: Diagnóstico

1. Selecciona la **Tarea Médica** y el **Modelo de IA**
2. Sube tu mamografía (`.dcm`, `.png`, `.jpg`)
3. Ajusta la **confianza mínima** y el **pixel spacing** (auto desde DICOM)
4. Haz clic en **"Analizar Mamografía"**
5. Revisa:
   - Imagen anotada con **bounding boxes** y medidas en **mm**
   - **Mapa Grad-CAM** de zonas de atención
   - **Categoría BIRADS** estimada
   - **Tabla** con diámetro, área y coordenadas de cada lesión
6. Exporta el reporte en **PDF** o **JSON**

### Pestaña 2: Entrenamiento

1. Sube tu dataset (`.zip` con carpetas `con_cancer/` y `sin_cancer/`)
   - O activa **"Usar CBIS-DDSM"** para descarga automática
2. Configura hiperparámetros: epochs, learning rate, batch size
3. Haz clic en **"Iniciar Entrenamiento"**
4. Observa las **métricas en tiempo real** (Loss, Accuracy)
5. Exporta el mejor modelo en `.pth`, `.onnx` o `.safetensors`

### Pestaña 3: Git & Modelos

1. Ingresa tu **GitHub Personal Access Token**
2. Clona o actualiza el repositorio `AderDevP/IA-Models`
3. Descarga modelos preentrenados desde el catálogo
4. Haz **push** automático de código y modelos con un clic

---

## 📏 Medición de Lesiones

Las medidas en mm se calculan usando:

```
diámetro_mm = max(width_px, height_px) × pixel_spacing_mm/px
área_mm²    = width_px × height_px × (pixel_spacing_mm/px)²
```

- **Archivos DICOM**: el `pixel_spacing` se extrae automáticamente del header
- **PNG/JPG**: se usa el slider de calibración manual (por defecto: 0.07 mm/px)

---

## 🔌 Añadir Nueva Tarea Médica (Escalabilidad)

Para agregar detección de nódulos pulmonares:

```python
# tasks/lung_nodule/__init__.py

from core.base_task import BaseTask
from core.registry  import TaskRegistry

class LungNoduleTask(BaseTask):
    @property
    def task_id(self): return "lung_nodule"
    # ... implementar métodos abstractos ...

TaskRegistry.register_task("lung_nodule", LungNoduleTask)
```

Luego en `config.py`, habilita la tarea:
```python
MEDICAL_TASKS["lung_nodule"]["enabled"] = True
```

El sistema la detecta automáticamente gracias al auto-discovery.

---

## 📋 Datasets Soportados

| Dataset | Tamaño | Tarea | Descarga |
|---------|--------|-------|---------|
| **CBIS-DDSM** | ~163 GB (10 GB subset) | Clasificación + Detección | Auto (HuggingFace) |
| **Dataset propio** | Cualquier | Clasificación | ZIP upload en Dashboard |

Estructura para dataset propio:
```
mi_dataset.zip
├── con_cancer/    → imágenes malignas
└── sin_cancer/    → imágenes benignas
```

---

## ⚠️ Aviso Médico

> **MammoAI es una herramienta de investigación y apoyo diagnóstico.**
> Los resultados generados por este sistema NO reemplazan la evaluación
> de un radiólogo certificado ni deben usarse como único criterio diagnóstico.
> Siempre consulte a un profesional de la salud cualificado.

---

## 📄 Licencia

MIT License — Ver [LICENSE](LICENSE) para más detalles.

---

## 🙏 Créditos

- **Dataset**: [CBIS-DDSM](https://www.cancerimagingarchive.net/collection/cbis-ddsm/) — The Cancer Imaging Archive
- **MammoScreen**: [ianpan/mammoscreen](https://huggingface.co/ianpan/mammoscreen) — HuggingFace
- **Arquitecturas**: [timm](https://github.com/huggingface/pytorch-image-models), [torchvision](https://pytorch.org/vision/), [transformers](https://huggingface.co/docs/transformers)
- **Grad-CAM**: [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
