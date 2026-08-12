"""
core/registry.py — Registro Central de Tareas y Modelos (MammoAI)
==================================================================
Sistema de plugins escalable. Para agregar una nueva tarea médica
(ej. nódulos pulmonares) simplemente:
  1. Crear tasks/lung_nodule/__init__.py heredando de BaseTask
  2. Registrarla aquí con TaskRegistry.register()
"""

from __future__ import annotations
import importlib
import logging
from typing import Dict, Optional, Type
from core.base_task import BaseTask
from core.base_model import BaseModel

logger = logging.getLogger(__name__)


class TaskRegistry:
    """Registro singleton de tareas médicas."""

    _tasks:  Dict[str, Type[BaseTask]]  = {}
    _models: Dict[str, Type[BaseModel]] = {}

    # ──────────────────────────────────────────
    # Registro de Tareas
    # ──────────────────────────────────────────
    @classmethod
    def register_task(cls, task_id: str, task_cls: Type[BaseTask]) -> None:
        """Registra una tarea médica en el sistema."""
        cls._tasks[task_id] = task_cls
        logger.info(f"[Registry] Tarea registrada: {task_id}")

    @classmethod
    def get_task(cls, task_id: str) -> Optional[Type[BaseTask]]:
        return cls._tasks.get(task_id)

    @classmethod
    def list_tasks(cls) -> Dict[str, Type[BaseTask]]:
        return dict(cls._tasks)

    @classmethod
    def load_task(cls, task_id: str) -> Optional[BaseTask]:
        """Carga e instancia una tarea por su ID.
        Importa dinámicamente el módulo tasks.<task_id>
        """
        if task_id not in cls._tasks:
            try:
                module = importlib.import_module(f"tasks.{task_id}")
                # El módulo debe llamar TaskRegistry.register_task() al importarse
                logger.info(f"[Registry] Módulo tasks.{task_id} importado.")
            except ModuleNotFoundError:
                logger.error(f"[Registry] Tarea '{task_id}' no encontrada.")
                return None

        task_cls = cls._tasks.get(task_id)
        if task_cls:
            return task_cls()
        return None

    # ──────────────────────────────────────────
    # Registro de Modelos
    # ──────────────────────────────────────────
    @classmethod
    def register_model(cls, model_id: str, model_cls: Type[BaseModel]) -> None:
        cls._models[model_id] = model_cls
        logger.info(f"[Registry] Modelo registrado: {model_id}")

    @classmethod
    def get_model_cls(cls, model_id: str) -> Optional[Type[BaseModel]]:
        return cls._models.get(model_id)

    @classmethod
    def list_models(cls) -> Dict[str, Type[BaseModel]]:
        return dict(cls._models)

    # ──────────────────────────────────────────
    # Auto-descubrimiento
    # ──────────────────────────────────────────
    @classmethod
    def autodiscover(cls, tasks_package: str = "tasks") -> None:
        """Importa automáticamente todos los sub-paquetes en tasks/
        para que se registren solos al importarse.
        """
        import pkgutil
        import importlib
        try:
            pkg = importlib.import_module(tasks_package)
        except ModuleNotFoundError:
            logger.warning(f"Paquete '{tasks_package}' no encontrado.")
            return

        pkg_path = getattr(pkg, "__path__", [])
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=pkg_path, prefix=f"{tasks_package}.", onerror=lambda x: None
        ):
            # Solo importar __init__ de cada sub-paquete de primer nivel
            parts = modname.split(".")
            if len(parts) == 2 and ispkg:
                try:
                    importlib.import_module(modname)
                    logger.debug(f"[Autodiscover] Importado: {modname}")
                except Exception as e:
                    logger.warning(f"[Autodiscover] Error en {modname}: {e}")
