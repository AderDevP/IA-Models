"""
tasks/__init__.py — Paquete de Tareas Médicas (MammoAI)
=======================================================
Al importar este paquete, se auto-descubren y registran todas
las tareas médicas disponibles en subdirectorios.
"""

from core.registry import TaskRegistry

# Auto-descubrir todas las tareas en tasks/
TaskRegistry.autodiscover("tasks")
