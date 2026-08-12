"""
app.py — MammoAI Dashboard Principal (Gradio)
=============================================
Dashboard estilo Ultimate RVC — control total desde la UI.
3 pestañas modulares:
  1. 🔬 Diagnóstico — inferencia, bboxes, Grad-CAM, reporte, exportación
  2. 🧪 Entrenamiento — fine-tuning en tiempo real con gráficos interactivos
  3. 🚀 Git & Modelos — gestión, empaquetado y push a GitHub

Escalable: soporte multi-tarea médica mediante TaskRegistry.
"""

from __future__ import annotations
import logging
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import gradio as gr
import plotly.graph_objects as go
from PIL import Image

# ── Setup del logger ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("MammoAI")

# ── Imports del proyecto ──────────────────────────────────────────
from config import (
    APP_TITLE, APP_DESCRIPTION, APP_VERSION,
    PRETRAINED_MODELS, MEDICAL_TASKS, MODELS_DIR,
    DEFAULT_CONFIDENCE_THRESHOLD, DEFAULT_PIXEL_SPACING_MM,
    BIRADS_CATEGORIES, EXPORTS_DIR,
)
from core.registry      import TaskRegistry
from model_downloader   import ModelDownloader
from git_utils          import GitManager
import tasks  # auto-registro de todas las tareas

# ── Inicializar subsistemas ───────────────────────────────────────
downloader  = ModelDownloader()
_trainer_ref: Dict[str, Any] = {}   # {thread_id: Trainer}


# ══════════════════════════════════════════════════════════════════
# HELPERS UI
# ══════════════════════════════════════════════════════════════════

def get_installed_models_list() -> List[str]:
    installed = downloader.list_installed()
    if not installed:
        return list(PRETRAINED_MODELS.keys())   # mostrar todos aunque no instalados
    return installed


def get_enabled_tasks() -> List[Tuple[str, str]]:
    return [(v["name"], k) for k, v in MEDICAL_TASKS.items() if v["enabled"]]


def build_model_dropdown_choices() -> List[str]:
    choices = []
    for mid, meta in PRETRAINED_MODELS.items():
        badge = "⭐ RECOMENDADO — " if meta.get("recommended") else ""
        task_icon = "🎯" if "yolo" in meta.get("task_type", "") else "🧠"
        choices.append(f"{task_icon} {badge}{meta['name']} [{mid}]")
    return choices


def parse_model_id_from_choice(choice: str) -> str:
    """Extrae el model_id del string del dropdown."""
    if "[" in choice and "]" in choice:
        return choice.split("[")[-1].rstrip("]")
    return choice


def parse_task_id_from_choice(choice: str) -> str:
    """Extrae el task_id del dropdown de tareas."""
    for task_name, task_id in get_enabled_tasks():
        if task_name in choice or task_id in choice:
            return task_id
    return "breast_cancer"


# ══════════════════════════════════════════════════════════════════
# TAB 1 — DIAGNÓSTICO
# ══════════════════════════════════════════════════════════════════

def run_diagnosis(
    image_file,
    task_choice: str,
    model_choice: str,
    pixel_spacing: float,
    confidence_threshold: float,
    generate_gradcam: bool,
    dual_mode: bool = False,
    hf_token: str = "",
    progress=gr.Progress(track_tqdm=True),
) -> Tuple:
    """Callback principal de inferencia.
    
    Si dual_mode=True: corre YOLOv8 para señalizar lesiones Y el clasificador
    elegido para el diagnóstico final — ambos resultados se muestran juntos.
    """
    if image_file is None:
        return (
            None, None,
            "❌ Por favor, sube una imagen primero.",
            "<div>—</div>", [], None, None,
        )

    try:
        if hf_token and hf_token.strip():
            import huggingface_hub
            huggingface_hub.login(token=hf_token.strip())

        progress(0.1, desc="Cargando imagen...")

        # ── Cargar imagen ──────────────────────────────────────────
        from dicom_utils import load_image
        img_path = Path(image_file.name) if hasattr(image_file, "name") else Path(image_file)
        pil_image, auto_spacing, dicom_meta = load_image(img_path, pixel_spacing_fallback=pixel_spacing)

        effective_spacing = auto_spacing if abs(auto_spacing - DEFAULT_PIXEL_SPACING_MM) > 1e-5 else pixel_spacing

        task_id  = parse_task_id_from_choice(task_choice)
        model_id = parse_model_id_from_choice(model_choice)

        task = TaskRegistry.load_task(task_id)
        if task is None:
            return (None, None, f"❌ Tarea '{task_id}' no disponible.", "<div>—</div>", [], None, None)

        from detector import full_diagnostic_pipeline, run_yolo_inference, load_yolo_model, draw_bounding_boxes
        from config import PRETRAINED_MODELS, DEFAULT_DETECTOR_ID

        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"

        selected_meta = PRETRAINED_MODELS.get(model_id, {})
        is_yolo       = selected_meta.get("task_type") == "detection_yolo"

        # ══════════════════════════════════════════════════════
        # MODO DUAL: Clasificador (resultado) + YOLO (señalar)
        # ══════════════════════════════════════════════════════
        if dual_mode and not is_yolo:
            progress(0.3, desc="Cargando clasificador...")
            logger.info(f"[DUAL] Clasificador: {model_id}")

            _, _, clf_report = full_diagnostic_pipeline(
                image=pil_image,
                model_id=model_id,
                task=task,
                pixel_spacing=effective_spacing,
                confidence_threshold=confidence_threshold,
                generate_gradcam=generate_gradcam,
                device=device,
            )
            heatmap_source = pil_image  # se sobreescribe abajo si hay gradcam

            progress(0.6, desc="Cargando YOLOv8 para señalización...")
            logger.info(f"[DUAL] Detector YOLO: {DEFAULT_DETECTOR_ID}")

            try:
                yolo_model, yolo_meta = load_yolo_model(DEFAULT_DETECTOR_ID)
                yolo_report = run_yolo_inference(
                    pil_image, yolo_model, yolo_meta,
                    pixel_spacing=effective_spacing,
                    confidence_threshold=confidence_threshold,
                    device=device,
                )
                # Imagen señalizada por YOLO
                annotated = draw_bounding_boxes(pil_image, yolo_report.get("detections", []))
                detections = yolo_report.get("detections", [])
                logger.info(f"[DUAL] YOLO detectó {len(detections)} lesiones")
            except Exception as yolo_err:
                logger.warning(f"[DUAL] YOLO falló ({yolo_err}), usando imagen original")
                annotated  = pil_image.copy()
                detections = []

            # Resultado del clasificador + detecciones de YOLO
            report = clf_report.copy()
            report["detections"] = detections
            if detections:
                # Si YOLO encontró lesiones, actualizar BIRADS combinado
                max_conf   = max(d["confidence"] for d in detections) / 100
                malignant  = any("malignant" in d["class"] for d in detections)
                yolo_prob  = min(max_conf * (1.2 if malignant else 0.8), 1.0)
                clf_prob   = clf_report.get("classification", {}).get("confidence", 50) / 100
                combined   = 0.6 * clf_prob + 0.4 * yolo_prob
                birads_cat = _combined_birads(combined)
                report["birads"]      = birads_cat
                report["birads_info"] = BIRADS_CATEGORIES.get(birads_cat, {})

            # Agregar nota dual al reporte
            clf_info = clf_report.get("classification", {})
            dual_note = (
                f"\n{'─'*48}\n"
                f"🔀 MODO DUAL ACTIVO\n"
                f"   Clasificador ({model_id}):\n"
                f"   → {clf_info.get('predicted_class','N/A')} "
                f"({clf_info.get('confidence',0):.1f}% confianza)\n"
                f"   YOLO ({DEFAULT_DETECTOR_ID}):\n"
                f"   → {len(detections)} lesión(es) señalizada(s)\n"
                f"{'─'*48}"
            )
            report["report_text"] = report.get("report_text", "") + dual_note

        # ══════════════════════════════════════════════════════
        # MODO NORMAL: solo el modelo elegido
        # ══════════════════════════════════════════════════════
        else:
            progress(0.4, desc="Ejecutando inferencia...")
            annotated, heatmap_source, report = full_diagnostic_pipeline(
                image=pil_image,
                model_id=model_id,
                task=task,
                pixel_spacing=effective_spacing,
                confidence_threshold=confidence_threshold,
                generate_gradcam=generate_gradcam,
                device=device,
            )
            detections = report.get("detections", [])

        progress(0.9, desc="Generando reporte...")

        # ── Datos para la UI ───────────────────────────────────────
        report_text    = report.get("report_text", "")
        birads         = report.get("birads", 0)
        birads_info    = BIRADS_CATEGORIES.get(birads, {})
        classification = report.get("classification", {})

        # Tabla de detecciones — incluye diámetro px, mm y área
        det_rows = [
            [
                d["id"],
                d["class"],
                f"{d['confidence']:.1f}%",
                f"{d['width_px']} × {d['height_px']} px",
                f"{d['diameter_mm']:.2f} mm",
                f"{d['area_mm2']:.2f} mm²",
                str(d["bbox_px"]),
            ]
            for d in detections
        ] if detections else [["—", "Sin lesiones detectadas", "—", "—", "—", "—", "—"]]

        # BIRADS + clasificación en HTML
        birads_color = birads_info.get("color", "#808080")
        clf_class    = classification.get("predicted_class", "—")
        clf_conf     = classification.get("confidence", 0)

        birads_html = (
            f'<div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center;">'
            f'<div style="background:{birads_color};color:white;padding:12px 20px;'
            f'border-radius:8px;font-size:18px;font-weight:bold;">'
            f'{birads_info.get("label","BIRADS ?")} — {birads_info.get("meaning","")}'
            f'</div>'
            f'<div style="background:#1e293b;color:#94a3b8;padding:10px 16px;'
            f'border-radius:8px;font-size:14px;border:1px solid #334155;">'
            f'🧠 Clasificación: <strong style="color:#e2e8f0">{clf_class}</strong> '
            f'({clf_conf:.1f}% confianza)'
            f'</div>'
            f'</div>'
        )

        progress(1.0, desc="¡Diagnóstico completado!")

        return (
            annotated,
            pil_image if dual_mode else (locals().get("heatmap_source") or pil_image),
            report_text,
            birads_html,
            det_rows,
            report,
            pil_image,
        )

    except Exception as e:
        logger.exception("Error en diagnóstico:")
        return (
            None, None,
            f"❌ Error durante el diagnóstico:\n{e}",
            "<div>Error</div>", [], None, None,
        )


def _combined_birads(prob: float) -> int:
    """Estima BIRADS desde probabilidad combinada clasificador+YOLO."""
    if prob < 0.02:  return 1
    if prob < 0.10:  return 2
    if prob < 0.30:  return 3
    if prob < 0.60:  return 4
    if prob < 0.95:  return 5
    return 6




def export_report_pdf(report: Optional[Dict], original_image, annotated_image) -> Optional[str]:
    if not report:
        return None
    try:
        from report_generator import export_pdf
        path = export_pdf(
            report=report,
            image=original_image,
            annotated_image=annotated_image,
        )
        return str(path)
    except Exception as e:
        logger.error(f"Error exportando PDF: {e}")
        return None


def export_report_json(report: Optional[Dict]) -> Optional[str]:
    if not report:
        return None
    try:
        from report_generator import export_json
        path = export_json(report=report)
        return str(path)
    except Exception as e:
        logger.error(f"Error exportando JSON: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# TAB 2 — ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════

def start_training(
    data_zip,
    dataset_choice: str,
    task_choice_train: str,
    model_choice_train: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    freeze_backbone: bool,
    mixed_precision: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """Callback de entrenamiento — generator para actualizar UI en tiempo real."""
    
    use_cbis_ddsm = (dataset_choice == "Usar CBIS-DDSM (Automático)")

    task_id  = parse_task_id_from_choice(task_choice_train)
    model_id = parse_model_id_from_choice(model_choice_train)

    task = TaskRegistry.load_task(task_id)
    if task is None:
        yield "❌ Tarea no disponible.", build_empty_plot(), None
        return

    from train import Trainer
    trainer = Trainer(model_id=model_id, task=task)
    _trainer_ref["current"] = trainer

    # Resolver directorio de datos
    if use_cbis_ddsm:
        data_source = "cbis_ddsm"
    elif data_zip is not None:
        data_source = _extract_zip(data_zip)
    else:
        yield "❌ Por favor, sube un dataset o selecciona CBIS-DDSM.", build_empty_plot(), None
        return

    progress(0.02, desc="Iniciando entrenamiento...")

    best_model_path = None
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for result in trainer.train_generator(
        data_source=data_source,
        use_cbis_ddsm=use_cbis_ddsm,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        freeze_backbone=freeze_backbone,
        mixed_precision=mixed_precision,
    ):
        status_text = result.get("status", "")

        if "history" in result:
            history = result["history"]

        if "best_model_path" in result and result["best_model_path"]:
            best_model_path = result["best_model_path"]

        epoch = result.get("epoch", 0)
        total = result.get("epochs", epochs)
        if epoch:
            progress(epoch / total, desc=f"Epoch {epoch}/{total}")

        fig = build_training_plot(history)
        yield status_text, fig, best_model_path


def stop_training():
    trainer = _trainer_ref.get("current")
    if trainer:
        trainer.stop()
        return "⏹️  Solicitando detención..."
    return "No hay entrenamiento activo."


def _extract_zip(zip_file) -> Path:
    """Extrae un archivo .zip de dataset a datasets/custom/."""
    import zipfile
    from config import DATASETS_DIR
    extract_to = DATASETS_DIR / "custom"
    extract_to.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_file.name, "r") as zf:
        zf.extractall(extract_to)
    return extract_to


def build_training_plot(history: Dict) -> go.Figure:
    """Construye figura Plotly con métricas de entrenamiento."""
    fig = go.Figure()
    colors_map = {
        "train_loss": "#ef5350", "val_loss": "#ab47bc",
        "train_acc":  "#42a5f5", "val_acc":  "#26a69a",
    }
    labels_map = {
        "train_loss": "Loss Train", "val_loss": "Loss Val",
        "train_acc":  "Acc Train",  "val_acc":  "Acc Val",
    }
    for key, vals in history.items():
        if not vals:
            continue
        x = list(range(1, len(vals) + 1))
        yaxis = "y2" if "acc" in key else "y"
        fig.add_trace(go.Scatter(
            x=x, y=vals,
            mode="lines+markers",
            name=labels_map.get(key, key),
            line=dict(color=colors_map.get(key, "#999"), width=2.5),
            marker=dict(size=5),
            yaxis=yaxis,
        ))

    fig.update_layout(
        template="plotly_dark",
        title=dict(text="📈 Métricas de Entrenamiento en Tiempo Real", font=dict(size=15)),
        xaxis=dict(title="Epoch", gridcolor="#333"),
        yaxis=dict(title="Loss",  gridcolor="#333", side="left"),
        yaxis2=dict(title="Accuracy", overlaying="y", side="right", range=[0, 1], gridcolor="#333"),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#eeeeee"),
        height=400,
    )
    return fig


def build_empty_plot() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title="📈 Métricas de Entrenamiento",
        plot_bgcolor="#1e1e1e",
        paper_bgcolor="#1e1e1e",
        font=dict(color="#eeeeee"),
        height=400,
    )
    return fig


def export_trained_model(best_model_path: Optional[str], export_format: str) -> Optional[str]:
    if not best_model_path:
        return None
    try:
        trainer = _trainer_ref.get("current")
        if trainer:
            out = trainer.export_model(format=export_format)
            return str(out)
    except Exception as e:
        logger.error(f"Error exportando: {e}")
    return None


# ══════════════════════════════════════════════════════════════════
# TAB 3 — GIT & MODELOS
# ══════════════════════════════════════════════════════════════════

def download_model_action(model_choice: str) -> Generator[str, None, None]:
    model_id = parse_model_id_from_choice(model_choice)
    log_lines = []

    def callback(msg: str):
        log_lines.append(msg)

    try:
        downloader.download(model_id, progress_callback=callback)
        yield "\n".join(log_lines)
    except Exception as e:
        yield f"❌ Error: {e}"


def get_model_status() -> str:
    return downloader.status_report()


def clone_repo_action(token: str, local_dir: str) -> str:
    if not token.strip():
        return "❌ Por favor, introduce tu GitHub Personal Access Token."
    if not local_dir.strip():
        local_dir = "/content/IA-Models"
    gm = GitManager(token=token, local_dir=local_dir)
    output = gm.clone_or_pull()
    return "\n".join(output)


def push_to_github(
    token: str,
    local_dir: str,
    commit_message: str,
    selected_models: List[str],
    push_all_code: bool,
) -> str:
    if not token.strip():
        return "❌ Token de GitHub no proporcionado."

    logs = []
    gm = GitManager(token=token, local_dir=local_dir or "/content/IA-Models")

    def cb(msg):
        logs.append(msg)

    if push_all_code:
        success, output = gm.commit_and_push(
            add_all=True,
            message=commit_message or "feat: update MammoAI",
            progress_callback=cb,
        )
        logs.append(output)
    else:
        for model_name in selected_models:
            model_path = MODELS_DIR / model_name
            if model_path.exists():
                ok, msg = gm.package_and_push_model(
                    model_path=model_path,
                    commit_message=f"feat: add model {model_name}",
                    progress_callback=cb,
                )
                logs.append(msg)

    return "\n".join(str(l) for l in logs)


def list_local_models() -> List[str]:
    exts = {".pth", ".pt", ".onnx", ".safetensors", ".bin"}
    return [f.name for f in MODELS_DIR.iterdir() if f.is_file() and f.suffix in exts]


def get_git_status(token: str, local_dir: str) -> str:
    if not token.strip():
        return "Introduce tu token para ver el estado."
    gm = GitManager(token=token, local_dir=local_dir or "/content/IA-Models")
    st = gm.get_status()
    if not st.get("initialized"):
        return "⚠️ Repositorio no clonado. Haz clic en 'Clonar Repositorio' primero."
    return (
        f"📁 Directorio: {st['local_dir']}\n"
        f"🌿 Rama: {st['branch']}\n"
        f"💾 Último commit: {st['last_commit']}\n"
        f"📋 Cambios staged: {st['staged_changes']}"
    )


# ══════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL DASHBOARD
# ══════════════════════════════════════════════════════════════════

CSS = """
/* ─── Tema oscuro personalizado ─────────────────────── */
:root {
    --primary:    #ad1457;
    --primary-dk: #880e4f;
    --accent:     #e91e63;
    --bg:         #121212;
    --bg-card:    #1e1e1e;
    --bg-card2:   #252525;
    --text:       #f0f0f0;
    --text-muted: #9e9e9e;
    --border:     #333333;
    --success:    #43a047;
    --warning:    #fb8c00;
    --error:      #e53935;
    --radius:     12px;
    --shadow:     0 4px 24px rgba(0,0,0,0.5);
}

body, .gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

/* Header */
.mammo-header {
    background: linear-gradient(135deg, #880e4f 0%, #1a237e 60%, #0d47a1 100%);
    border-radius: var(--radius);
    padding: 28px 36px;
    margin-bottom: 20px;
    box-shadow: var(--shadow);
    text-align: center;
}
.mammo-header h1 {
    font-size: 2.1rem;
    font-weight: 800;
    color: #fff;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.mammo-header p {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    margin: 0;
}

/* Tabs */
.tab-nav button {
    background: var(--bg-card2) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button.selected {
    background: var(--primary) !important;
    color: #fff !important;
    border-color: var(--primary) !important;
}
.tab-nav button:hover:not(.selected) {
    background: var(--bg-card) !important;
    color: var(--text) !important;
}

/* Cards */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: var(--shadow);
}

/* Botones */
button.primary-btn, .gr-button-primary {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(173,20,87,0.35) !important;
}
button.primary-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(173,20,87,0.55) !important;
}

/* Inputs */
input, textarea, select {
    background: var(--bg-card2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Imágenes */
.image-container img {
    border-radius: var(--radius);
    border: 1px solid var(--border);
}

/* Status badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    background: var(--bg-card2);
    color: var(--text-muted);
    border: 1px solid var(--border);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
"""

HEADER_HTML = """
<div class="mammo-header">
  <h1>🔬 MammoAI</h1>
  <p>Sistema Open Source de IA para Detección y Análisis Cuantitativo de Mamografías</p>
  <p style="font-size:0.8rem;margin-top:8px;opacity:0.6;">
    Clasificación: MammoScreen EfficientNetV2 &nbsp;|&nbsp;
    Detección/Señalización: YOLOv8 &nbsp;|&nbsp;
    Formatos: DICOM · PGM · PNG · JPG · TIFF · BMP &nbsp;|&nbsp; v{version}
  </p>
  <p style="font-size:0.75rem;color:#f39c12;margin-top:4px;">
    ⭐ Modelos recomendados marcados en el selector
  </p>
</div>
""".format(version=APP_VERSION)


def build_app() -> gr.Blocks:
    """Construye y retorna el objeto gr.Blocks del dashboard."""

    enabled_tasks = get_enabled_tasks()
    task_choices  = [f"{name} [{tid}]" for name, tid in enabled_tasks]
    model_choices = build_model_dropdown_choices()

    theme = gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="indigo",
        neutral_hue="slate",
    )

    blocks_kwargs = {"title": "MammoAI — Diagnóstico Inteligente"}
    # En Gradio 4/3, theme y css van en gr.Blocks. En Gradio 5/6, van en launch().
    gr_ver = getattr(gr, "__version__", "4.0")
    if gr_ver.startswith(("3", "4")):
        blocks_kwargs["css"] = CSS
        blocks_kwargs["theme"] = theme

    with gr.Blocks(**blocks_kwargs) as demo:

        # ── Header ────────────────────────────────────────────────
        gr.HTML(HEADER_HTML)

        # ── Estados internos ──────────────────────────────────────
        state_report      = gr.State(None)
        state_original    = gr.State(None)
        state_annotated   = gr.State(None)
        state_best_model  = gr.State(None)

        # ══════════════════════════════════════════════════════════
        # PESTAÑA 1 — DIAGNÓSTICO
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🔬 Diagnóstico"):
            with gr.Row():
                # ── Panel izquierdo: entrada ──────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuración de Análisis")
                    
                    with gr.Group():
                        hf_token = gr.Textbox(
                            label="Hugging Face Token (Para modelos privados/MedGemma)",
                            placeholder="hf_xxxxxxxxxxxxxxxxxxx",
                            type="password"
                        )
                    
                    with gr.Group():
                        task_selector = gr.Dropdown(
                            choices=task_choices,
                            value=task_choices[0] if task_choices else None,
                            label="🏥 Tarea Médica",
                            info="Selecciona el tipo de análisis",
                            interactive=True,
                        )
                    model_selector = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        label="🤖 Modelo de IA",
                        info="⭐ = Recomendado  |  🎯 = Detección/Localización (YOLO)  |  🧠 = Clasificación",
                        interactive=True,
                    )
                    image_input = gr.File(
                        label="📁 Subir Mamografía",
                        file_types=[".dcm", ".pgm", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"],
                        type="filepath",
                    )
                    with gr.Accordion("🔧 Parámetros Avanzados", open=False):
                        confidence_slider = gr.Slider(
                            minimum=0.1, maximum=0.95, value=DEFAULT_CONFIDENCE_THRESHOLD,
                            step=0.05, label="Confianza Mínima",
                        )
                        spacing_slider = gr.Slider(
                            minimum=0.01, maximum=0.5, value=DEFAULT_PIXEL_SPACING_MM,
                            step=0.005, label="Pixel Spacing Manual (mm/px)",
                            info="Se sobreescribe automáticamente si el DICOM tiene PixelSpacing",
                        )
                        gradcam_toggle = gr.Checkbox(
                            value=True, label="Generar Grad-CAM / Heatmap de atención"
                        )
                        dual_mode_toggle = gr.Checkbox(
                            value=False,
                            label="🔀 Modo Dual: Clasificador + YOLO señalizador",
                            info="Corre el clasificador elegido PARA el resultado Y YOLOv8 PARA señalizar lesiones en la imagen",
                        )

                    btn_analyze = gr.Button(
                        "🔍 Analizar Mamografía", variant="primary", size="lg"
                    )

                # ── Panel derecho: resultados ─────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("### 🖼️ Visualización Diagnóstica")
                    with gr.Row():
                        img_output  = gr.Image(label="🎯 Detecciones + Medidas", type="pil")
                        img_heatmap = gr.Image(label="🌡️ Grad-CAM Heatmap",    type="pil")

                    html_birads = gr.HTML("<div style='padding:12px;color:#888;'>BIRADS aparecerá aquí.</div>")

                    txt_report  = gr.Textbox(
                        label="📋 Informe Diagnóstico Completo",
                        lines=12, max_lines=20,
                        interactive=False,
                    )

                    gr.Markdown("#### 🎯 Tabla de Lesiones Detectadas")
                    table_detections = gr.Dataframe(
                        headers=["#", "Tipo", "Confianza", "Tamaño (px)", "Diámetro (mm)", "Área (mm²)", "BBox (px)"],
                        interactive=False,
                    )

                    with gr.Row():
                        btn_export_pdf  = gr.Button("📄 Exportar PDF",  variant="secondary")
                        btn_export_json = gr.Button("📦 Exportar JSON", variant="secondary")
                    file_pdf  = gr.File(label="PDF generado",  visible=False)
                    file_json = gr.File(label="JSON generado", visible=False)

            # ── Eventos Diagnóstico ───────────────────────────────
            btn_analyze.click(
                fn=run_diagnosis,
                inputs=[
                    image_input, task_selector, model_selector,
                    spacing_slider, confidence_slider, gradcam_toggle, dual_mode_toggle, hf_token,
                ],
                outputs=[
                    img_output, img_heatmap,
                    txt_report, html_birads,
                    table_detections,
                    state_report, state_original,
                ],
            )
            img_output.change(fn=lambda x: x, inputs=[img_output], outputs=[state_annotated])

            btn_export_pdf.click(
                fn=export_report_pdf,
                inputs=[state_report, state_original, state_annotated],
                outputs=[file_pdf],
            ).then(fn=lambda p: gr.File(value=p, visible=True) if p else gr.File(visible=False),
                   inputs=[file_pdf], outputs=[file_pdf])

            btn_export_json.click(
                fn=export_report_json,
                inputs=[state_report],
                outputs=[file_json],
            ).then(fn=lambda p: gr.File(value=p, visible=True) if p else gr.File(visible=False),
                   inputs=[file_json], outputs=[file_json])

        # ══════════════════════════════════════════════════════════
        # PESTAÑA 2 — ENTRENAMIENTO
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🧪 Entrenamiento"):
            with gr.Row():
                # ── Panel de configuración ────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙️ Configuración del Entrenamiento")

                    task_selector_train = gr.Dropdown(
                        choices=task_choices,
                        value=task_choices[0] if task_choices else None,
                        label="🏥 Tarea Médica",
                        interactive=True,
                    )
                    model_selector_train = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        label="🤖 Arquitectura Base (Backbone)",
                        interactive=True,
                    )

                    gr.Markdown("#### 📦 Dataset")
                    dataset_choice = gr.Radio(
                        choices=["Usar CBIS-DDSM (Automático)", "Subir archivo .zip"],
                        value="Usar CBIS-DDSM (Automático)",
                        label="Selecciona origen de datos",
                    )
                    data_zip = gr.File(
                        label="Sube tu dataset (.zip con carpetas con_cancer / sin_cancer)",
                        file_types=[".zip"],
                        type="filepath",
                        visible=False,
                    )
                    
                    def toggle_zip_upload(choice):
                        return gr.update(visible=(choice == "Subir archivo .zip"))
                    
                    dataset_choice.change(
                        fn=toggle_zip_upload,
                        inputs=dataset_choice,
                        outputs=data_zip
                    )

                    gr.Markdown("#### 🔧 Hiperparámetros")
                    epochs_slider     = gr.Slider(1, 100, value=20, step=1,    label="Epochs")
                    batch_slider      = gr.Slider(4, 64,  value=16, step=4,    label="Batch Size")
                    lr_slider         = gr.Slider(1e-6, 1e-2, value=1e-4, step=1e-6, label="Learning Rate")
                    freeze_backbone   = gr.Checkbox(value=True, label="Congelar backbone (primeras 3 épocas)")
                    mixed_precision   = gr.Checkbox(value=True, label="Mixed Precision (AMP) — más rápido en GPU")

                    with gr.Row():
                        btn_train  = gr.Button("🚀 Iniciar Entrenamiento", variant="primary", size="lg")
                        btn_stop   = gr.Button("⏹️ Detener", variant="stop")

                # ── Panel de métricas ─────────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Métricas en Tiempo Real")

                    txt_train_status = gr.Textbox(
                        label="📟 Log de Entrenamiento",
                        lines=8, max_lines=15,
                        interactive=False,
                    )
                    plot_metrics = gr.Plot(
                        label="Curvas de Loss y Accuracy",
                        value=build_empty_plot(),
                    )

                    gr.Markdown("#### 💾 Exportar Modelo Entrenado")
                    export_format = gr.Radio(
                        choices=["pth", "onnx", "safetensors"],
                        value="pth",
                        label="Formato de Exportación",
                    )
                    btn_export_model = gr.Button("📥 Exportar Mejor Modelo", variant="secondary")
                    file_model_out   = gr.File(label="Modelo exportado", visible=False)

            # ── Eventos Entrenamiento ─────────────────────────────
            btn_train.click(
                fn=start_training,
                inputs=[
                    data_zip, dataset_choice,
                    task_selector_train, model_selector_train,
                    epochs_slider, batch_slider, lr_slider,
                    freeze_backbone, mixed_precision,
                ],
                outputs=[txt_train_status, plot_metrics, state_best_model],
            )
            btn_stop.click(fn=stop_training, outputs=[txt_train_status])

            btn_export_model.click(
                fn=export_trained_model,
                inputs=[state_best_model, export_format],
                outputs=[file_model_out],
            ).then(fn=lambda p: gr.File(value=p, visible=True) if p else gr.File(visible=False),
                   inputs=[file_model_out], outputs=[file_model_out])

        # ══════════════════════════════════════════════════════════
        # PESTAÑA 3 — GIT & MODELOS
        # ══════════════════════════════════════════════════════════
        with gr.Tab("🚀 Git & Modelos"):
            with gr.Row():
                # ── Gestión de modelos ────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### 📦 Gestión de Modelos")

                    gr.Markdown("#### ⬇️ Descargar Modelos Preentrenados")
                    model_dl_selector = gr.Dropdown(
                        choices=model_choices,
                        value=model_choices[0] if model_choices else None,
                        label="Modelo a descargar",
                        interactive=True,
                    )
                    btn_download_model = gr.Button("⬇️ Descargar Modelo", variant="primary")
                    txt_dl_status = gr.Textbox(
                        label="Estado de descarga",
                        lines=4, interactive=False,
                    )
                    btn_model_status = gr.Button("🔄 Ver Estado de Todos los Modelos")
                    txt_model_status = gr.Textbox(
                        label="Estado de instalación",
                        lines=8, interactive=False,
                        value=downloader.status_report(),
                    )

                # ── Git / GitHub ──────────────────────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### 🐙 GitHub — AderDevP/IA-Models")
                    gr.Markdown(
                        "Repositorio: [https://github.com/AderDevP/IA-Models](https://github.com/AderDevP/IA-Models)"
                    )

                    github_token = gr.Textbox(
                        label="🔑 GitHub Personal Access Token",
                        type="password",
                        placeholder="ghp_xxxxxxxxxxxxxxxxxxxx",
                        info="Necesitas permisos 'repo'. Se usa solo en esta sesión.",
                    )
                    local_dir = gr.Textbox(
                        label="📁 Directorio Local del Repositorio",
                        value="/content/IA-Models",
                        info="Ruta donde clonar el repositorio en Colab",
                    )
                    btn_clone = gr.Button("📥 Clonar / Actualizar Repositorio", variant="secondary")

                    gr.Markdown("#### 🚀 Push a GitHub")
                    commit_msg = gr.Textbox(
                        label="Mensaje del Commit",
                        value="feat: update from MammoAI",
                    )
                    push_all_code = gr.Checkbox(
                        value=True,
                        label="Incluir todo el código del proyecto (git add .)",
                    )
                    local_models_list = list_local_models()
                    model_checkboxes = gr.CheckboxGroup(
                        choices=local_models_list,
                        label="O selecciona modelos específicos a subir",
                        visible=len(local_models_list) > 0,
                    )
                    btn_push = gr.Button("🚀 Push a GitHub", variant="primary", size="lg")

                    txt_git_log = gr.Textbox(
                        label="📟 Log de Git",
                        lines=10, interactive=False,
                    )
                    btn_git_status = gr.Button("🔍 Ver Estado del Repositorio")

            # ── Eventos Git ───────────────────────────────────────
            btn_download_model.click(
                fn=lambda choice: "\n".join(list(download_model_action(choice))),
                inputs=[model_dl_selector],
                outputs=[txt_dl_status],
            )
            btn_model_status.click(
                fn=get_model_status,
                outputs=[txt_model_status],
            )
            btn_clone.click(
                fn=clone_repo_action,
                inputs=[github_token, local_dir],
                outputs=[txt_git_log],
            )
            btn_push.click(
                fn=push_to_github,
                inputs=[github_token, local_dir, commit_msg, model_checkboxes, push_all_code],
                outputs=[txt_git_log],
            )
            btn_git_status.click(
                fn=get_git_status,
                inputs=[github_token, local_dir],
                outputs=[txt_git_log],
            )

        # ── Footer ────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center;padding:20px;color:#555;font-size:0.85rem;margin-top:20px;
                    border-top:1px solid #333;">
            MammoAI v1.0.0 &nbsp;|&nbsp; Open Source &nbsp;|&nbsp;
            <a href="https://github.com/AderDevP/IA-Models" target="_blank"
               style="color:#e91e63;">GitHub: AderDevP/IA-Models</a>
            &nbsp;|&nbsp; ⚠️ Solo para investigación — no sustituye diagnóstico médico profesional.
        </div>
        """)

    return demo


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def launch_app(demo: gr.Blocks, **kwargs) -> Any:
    """Lanza el dashboard Gradio reconectando puertos automáticamente si están ocupados."""
    try:
        gr.close_all()
    except Exception:
        pass

    theme = gr.themes.Soft(
        primary_hue="pink",
        secondary_hue="indigo",
        neutral_hue="slate",
    )
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "share": True,
        "show_error": True,
        "quiet": False,
    }
    launch_kwargs.update(kwargs)

    gr_ver = getattr(gr, "__version__", "4.0")
    if not gr_ver.startswith(("3", "4")):
        launch_kwargs["theme"] = theme
        launch_kwargs["css"] = CSS

    try:
        return demo.launch(**launch_kwargs)
    except OSError as e:
        logger.warning(f"Puerto ocupado ({e}). Reintentando asignación de puerto dinámica...")
        launch_kwargs.pop("server_port", None)
        return demo.launch(**launch_kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MammoAI Dashboard")
    parser.add_argument("--share",  action="store_true", default=False,
                        help="Generar enlace público (Gradio share)")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--host",   type=str, default="0.0.0.0")
    parser.add_argument("--colab",  action="store_true", default=False,
                        help="Modo Colab: activa share automáticamente")
    args = parser.parse_args()

    share = args.share or args.colab

    logger.info(f"🚀 Iniciando MammoAI v{APP_VERSION}...")
    logger.info(f"   Modelos disponibles: {len(PRETRAINED_MODELS)}")
    logger.info(f"   Tareas habilitadas:  {len([t for t in MEDICAL_TASKS.values() if t['enabled']])}")

    demo = build_app()
    launch_app(demo, share=share, server_name=args.host, server_port=args.port)

