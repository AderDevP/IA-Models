"""
rebuild_notebook_v2.py
Rebuilds notebook.ipynb with:
 - ultralytics in Celda 2
 - correct model IDs in Celda 5 (yolov8m_mammo + mammoscreen_efficientnetv2)
"""
import json

with open("notebook.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]

# ── Fix Celda 2: add ultralytics ────────────────────────────────────────────
CELDA2_SOURCE = [
    "# ╔══════════════════════════════════════════════════════════════╗\n",
    "# ║  CELDA 2 — Instalación de dependencias                     ║\n",
    "# ╚══════════════════════════════════════════════════════════════╝\n",
    "import subprocess, sys\n",
    "print('Instalando dependencias...')\n",
    "print('   (Este proceso tarda ~3-5 minutos en la primera ejecucion)')\n",
    "\n",
    "deps = [\n",
    "    'torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118',\n",
    "    'gradio plotly>=5.18.0',\n",
    "    'timm>=0.9.12 opencv-python-headless pillow scikit-image',\n",
    "    'grad-cam>=1.4.8',\n",
    "    'ultralytics>=8.0.0',\n",
    "    'pydicom>=2.4.3',\n",
    "    'huggingface_hub>=0.20.3 datasets>=2.16.0 transformers>=4.37.0',\n",
    "    'scikit-learn numpy',\n",
    "    'onnx safetensors reportlab',\n",
    "    'gitpython tqdm requests',\n",
    "]\n",
    "\n",
    "for dep in deps:\n",
    "    pkg = dep.split()[0].split('>=')[0].split('==')[0]\n",
    "    print(f'  Installing: {pkg}...')\n",
    "    result = subprocess.run(\n",
    "        [sys.executable, '-m', 'pip', 'install', '-q'] + dep.split(),\n",
    "        capture_output=True, text=True\n",
    "    )\n",
    "    if result.returncode != 0:\n",
    "        print(f'  WARNING: {result.stderr[:200]}')\n",
    "\n",
    "print('\\nTodas las dependencias instaladas.')\n",
]

# ── Fix Celda 5: correct model IDs ─────────────────────────────────────────
CELDA5_SOURCE = [
    "# ╔══════════════════════════════════════════════════════════════╗\n",
    "# ║  CELDA 5 — Descarga automatica de modelos preentrenados    ║\n",
    "# ╚══════════════════════════════════════════════════════════════╝\n",
    "import importlib\n",
    "import model_downloader\n",
    "importlib.reload(model_downloader)\n",
    "from model_downloader import ModelDownloader\n",
    "\n",
    "downloader = ModelDownloader()\n",
    "\n",
    "print('Estado actual de modelos:')\n",
    "print(downloader.status_report())\n",
    "\n",
    "# Modelos a descargar automaticamente\n",
    "# Para deteccion/senalizacion de lesiones (YOLOv8) + clasificacion (EfficientNetV2)\n",
    "AUTO_DOWNLOAD_MODELS = [\n",
    "    'mammoscreen_efficientnetv2',   # clasificador recomendado\n",
    "    'efficientnet_b4_cbis',         # clasificador backup\n",
    "    'yolov8m_mammo',                # detector/localizador recomendado\n",
    "]\n",
    "\n",
    "print('\\nDescargando modelos seleccionados...')\n",
    "for model_id in AUTO_DOWNLOAD_MODELS:\n",
    "    if not downloader.is_installed(model_id):\n",
    "        print(f'\\nDescargando: {model_id}...')\n",
    "        try:\n",
    "            path = downloader.download(model_id, progress_callback=print)\n",
    "            print(f'OK {model_id} -> {path}')\n",
    "        except Exception as e:\n",
    "            print(f'ERROR en {model_id}: {e}')\n",
    "    else:\n",
    "        print(f'OK {model_id} ya instalado.')\n",
    "\n",
    "print('\\nEstado final:')\n",
    "print(downloader.status_report())\n",
    "\n",
    "print('\\n=== MODELOS DISPONIBLES Y SU FUNCION ===')\n",
    "print('  mammoscreen_efficientnetv2 -> CLASIFICACION (recomendado)')\n",
    "print('  efficientnet_b4_cbis       -> CLASIFICACION (backup)')\n",
    "print('  yolov8m_mammo              -> DETECCION/SENALIZACION (recomendado)')\n",
    "print('  yolov8n_mammo              -> DETECCION rapida (nano)')\n",
    "print()\n",
    "print('CONSEJO: En el Dashboard, usa el modo DUAL para combinar')\n",
    "print('  clasificador (EfficientNet) + detector (YOLO) al mismo tiempo.')\n",
]

# Identify and replace cells
cell_sources = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "CELDA 2" in src and "dependencias" in src:
        cell["source"] = CELDA2_SOURCE
        print(f"[OK] Replaced Celda 2 at index {i}")
    elif "CELDA 5" in src and "modelos" in src.lower():
        cell["source"] = CELDA5_SOURCE
        print(f"[OK] Replaced Celda 5 at index {i}")

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

# Validate
json.load(open("notebook.ipynb", encoding="utf-8"))
print("notebook.ipynb rebuilt and JSON validated!")
