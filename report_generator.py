"""
report_generator.py — Exportación de Reportes Médicos
======================================================
Genera informes diagnósticos en PDF y JSON a partir
del reporte estructurado producido por el detector.
"""

from __future__ import annotations
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from PIL import Image

from config import REPORTS_DIR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Exportación PDF
# ──────────────────────────────────────────────────────────────────

def export_pdf(
    report: Dict,
    image: Optional[Image.Image] = None,
    annotated_image: Optional[Image.Image] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Genera un informe PDF completo del diagnóstico.

    Args:
        report: Diccionario del reporte (salida de task.postprocess)
        image: Imagen original (opcional, se incluye en el PDF)
        annotated_image: Imagen con anotaciones (bboxes, medidas)
        output_path: Ruta de destino (auto si None)

    Returns:
        Path al archivo PDF generado
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, Image as RLImage,
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER
    except ImportError:
        raise ImportError(
            "Instala reportlab: pip install reportlab"
        )

    import io

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path or (REPORTS_DIR / f"reporte_mammoai_{timestamp}.pdf")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc    = SimpleDocTemplate(str(output_path), pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    # ── Estilos personalizados ────────────────────────────────────
    title_style = ParagraphStyle(
        "MammoTitle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "MammoSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555555"),
        spaceAfter=4,
    )
    section_style = ParagraphStyle(
        "MammoSection",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#ad1457"),
        spaceBefore=12,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "MammoBody",
        parent=styles["Normal"],
        fontSize=10,
        leading=14,
    )

    # ── Encabezado ────────────────────────────────────────────────
    story.append(Paragraph("🔬 MammoAI — Reporte Diagnóstico", title_style))
    story.append(Paragraph(
        f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} | "
        f"Modelo: {report.get('model_id', 'N/A')} | "
        f"Tiempo inferencia: {report.get('inference_time_s', 0):.3f}s",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a237e")))
    story.append(Spacer(1, 0.4*cm))

    # ── Advertencia médica ────────────────────────────────────────
    warning_style = ParagraphStyle(
        "Warning",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#b71c1c"),
        borderColor=colors.HexColor("#b71c1c"),
        borderWidth=1,
        borderPadding=5,
        backColor=colors.HexColor("#ffebee"),
    )
    story.append(Paragraph(
        "⚠️ AVISO IMPORTANTE: Este reporte es generado por un sistema de IA "
        "como herramienta de apoyo diagnóstico. No reemplaza la evaluación "
        "de un radiólogo certificado. Consulte siempre a un profesional de la salud.",
        warning_style,
    ))
    story.append(Spacer(1, 0.5*cm))

    # ── Metadatos del paciente / imagen ──────────────────────────
    meta = report.get("metadata", {})
    if meta:
        story.append(Paragraph("Información del Estudio", section_style))
        meta_data = [
            ["Campo", "Valor"],
            *[(k, str(v)) for k, v in meta.items() if v and v != "N/A"][:12]
        ]
        meta_table = Table(meta_data, colWidths=[5*cm, 11*cm])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), colors.HexColor("#1a237e")),
            ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, 0), 10),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f5f5f5"), colors.white]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 0.4*cm))

    # ── Resultado de clasificación ────────────────────────────────
    clf = report.get("classification", {})
    if clf:
        story.append(Paragraph("Resultado de Clasificación", section_style))
        pred  = clf.get("predicted_class", "N/A")
        conf  = clf.get("confidence", 0)
        color_class = colors.HexColor("#c62828") if "Maligno" in str(pred) else colors.HexColor("#2e7d32")
        clf_style = ParagraphStyle("ClfResult", parent=styles["Normal"],
                                   fontSize=16, textColor=color_class, fontName="Helvetica-Bold")
        story.append(Paragraph(f"{pred} — Confianza: {conf:.1f}%", clf_style))
        probs = clf.get("probabilities", {})
        if probs:
            prob_text = " | ".join(f"{k}: {v:.1f}%" for k, v in probs.items())
            story.append(Paragraph(f"Probabilidades: {prob_text}", body_style))
        story.append(Spacer(1, 0.3*cm))

    # ── BIRADS ────────────────────────────────────────────────────
    birads     = report.get("birads")
    birads_info = report.get("birads_info", {})
    if birads is not None:
        story.append(Paragraph("Categoría BIRADS Estimada", section_style))
        story.append(Paragraph(
            f"{birads_info.get('label', f'BIRADS {birads}')} — "
            f"{birads_info.get('meaning', '')}",
            body_style,
        ))
        story.append(Spacer(1, 0.3*cm))

    # ── Lesiones detectadas ───────────────────────────────────────
    detections = report.get("detections", [])
    if detections:
        story.append(Paragraph(f"Lesiones Detectadas ({len(detections)})", section_style))
        det_data = [
            ["#", "Tipo", "Confianza", "Diámetro", "Área", "Posición (px)"],
            *[
                [
                    str(d["id"]),
                    d.get("class", "N/A").upper(),
                    f"{d.get('confidence', 0):.1f}%",
                    f"{d.get('diameter_mm', 0):.2f} mm",
                    f"{d.get('area_mm2', 0):.2f} mm²",
                    f"({d.get('center_x_px', 0)}, {d.get('center_y_px', 0)})",
                ]
                for d in detections
            ]
        ]
        det_table = Table(det_data, colWidths=[1*cm, 3*cm, 2.5*cm, 2.5*cm, 2.5*cm, 4*cm])
        det_table.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), colors.HexColor("#ad1457")),
            ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#fce4ec"), colors.white]),
            ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(det_table)
        story.append(Spacer(1, 0.4*cm))
    else:
        story.append(Paragraph("Lesiones Detectadas", section_style))
        story.append(Paragraph("✅ No se detectaron lesiones sospechosas.", body_style))
        story.append(Spacer(1, 0.3*cm))

    # ── Imágenes diagnósticas ────────────────────────────────────
    if annotated_image or image:
        story.append(Paragraph("Imágenes Diagnósticas", section_style))
        img_to_show = annotated_image or image
        buf = io.BytesIO()
        img_to_show.save(buf, format="PNG")
        buf.seek(0)
        rl_img = RLImage(buf, width=10*cm, height=10*cm)
        story.append(rl_img)
        story.append(Spacer(1, 0.3*cm))

    # ── Pixel spacing y calibración ───────────────────────────────
    spacing = report.get("pixel_spacing_mm")
    if spacing:
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
        story.append(Paragraph(
            f"Calibración: pixel spacing = {spacing:.4f} mm/px | "
            f"Nota: Las medidas en mm son estimaciones basadas en el pixel spacing.",
            ParagraphStyle("Small", parent=styles["Normal"], fontSize=8,
                           textColor=colors.grey)
        ))

    doc.build(story)
    logger.info(f"PDF generado: {output_path}")
    return output_path


# ──────────────────────────────────────────────────────────────────
# Exportación JSON
# ──────────────────────────────────────────────────────────────────

def export_json(
    report: Dict,
    output_path: Optional[Path] = None,
) -> Path:
    """Exporta el reporte a JSON estructurado.

    Args:
        report: Diccionario del reporte
        output_path: Ruta de destino (auto si None)

    Returns:
        Path al archivo JSON generado
    """
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_path or (REPORTS_DIR / f"reporte_mammoai_{timestamp}.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Añadir timestamp al reporte
    report_out = {
        "generated_at": datetime.now().isoformat(),
        "system":       "MammoAI v1.0.0",
        "disclaimer":   (
            "Este reporte es generado por IA como herramienta de apoyo. "
            "No reemplaza el diagnóstico de un radiólogo certificado."
        ),
        **report,
    }

    # Limpiar campos no serializables
    def make_serializable(obj):
        if hasattr(obj, "tolist"):       return obj.tolist()
        if hasattr(obj, "__dict__"):     return str(obj)
        return str(obj)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2, ensure_ascii=False,
                  default=make_serializable)

    logger.info(f"JSON generado: {output_path}")
    return output_path
