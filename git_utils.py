"""
git_utils.py — Integración Git / GitHub (HTTPS)
================================================
Automatiza: autenticación, clone/pull, commit y push
al repositorio https://github.com/AderDevP/IA-Models
vía HTTPS con Personal Access Token (PAT).

No requiere SSH — solo el token de GitHub en texto.
"""

from __future__ import annotations
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────────

REPO_URL    = "https://github.com/AderDevP/IA-Models"
REPO_BRANCH = "main"
GIT_USER    = "AderDevP"
GIT_EMAIL   = "aderdevp@users.noreply.github.com"


# ──────────────────────────────────────────────────────────────────
# Clase principal
# ──────────────────────────────────────────────────────────────────

class GitManager:
    """Gestiona la sincronización del proyecto con GitHub via HTTPS.

    Uso en Gradio:
        gm = GitManager(token="ghp_xxxx", local_dir="/content/IA-Models")
        for line in gm.push(files=["models/efficientnet.pth"], message="feat: add model"):
            print(line)
    """

    def __init__(
        self,
        token: str,
        local_dir: str | Path,
        repo_url: str = REPO_URL,
        branch: str = REPO_BRANCH,
        user_name: str = GIT_USER,
        user_email: str = GIT_EMAIL,
    ):
        self.token     = token.strip()
        self.local_dir = Path(local_dir)
        self.branch    = branch
        self.user_name = user_name
        self.user_email = user_email

        # URL autenticada con token (nunca se loguea el token completo)
        self.repo_url   = repo_url
        self.auth_url   = self._build_auth_url(repo_url, token)

    # ──────────────────────────────────────────
    # Clone / Pull
    # ──────────────────────────────────────────

    def clone_or_pull(
        self,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Clona el repositorio si no existe, o hace pull si ya existe.

        Returns:
            Lista de líneas de salida
        """
        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        output = []

        if (self.local_dir / ".git").exists():
            log(f"📁 Repositorio ya existe — ejecutando git pull...")
            result = self._run(
                ["git", "pull", "origin", self.branch],
                cwd=self.local_dir,
            )
            output.extend(result)
        else:
            log(f"⬇️  Clonando {self.repo_url} → {self.local_dir} ...")
            self.local_dir.parent.mkdir(parents=True, exist_ok=True)
            result = self._run(
                ["git", "clone", "--branch", self.branch,
                 self.auth_url, str(self.local_dir)],
            )
            output.extend(result)

        self._configure_git()
        log("✅ Repositorio sincronizado.")
        return output

    # ──────────────────────────────────────────
    # Commit y Push
    # ──────────────────────────────────────────

    def commit_and_push(
        self,
        files: Optional[List[str | Path]] = None,
        message: str = "feat: update from MammoAI",
        add_all: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, List[str]]:
        """Hace git add, commit y push de los archivos especificados.

        Args:
            files: Lista de rutas a stagear (relativas al repo)
            message: Mensaje del commit
            add_all: Si True, hace 'git add .' (ignora 'files')
            progress_callback: Función para emitir progreso a Gradio

        Returns:
            Tuple (success: bool, output_lines: list[str])
        """
        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        output = []
        self._configure_git()

        # ── git add ────────────────────────────────────────────────
        if add_all:
            result = self._run(["git", "add", "."], cwd=self.local_dir)
        else:
            files = files or []
            for f in files:
                result = self._run(
                    ["git", "add", str(f)],
                    cwd=self.local_dir,
                )
                output.extend(result)

        output.extend(result)
        log(f"📦 Archivos stageados.")

        # ── Verificar si hay cambios ───────────────────────────────
        status = self._run(["git", "status", "--porcelain"], cwd=self.local_dir)
        if not any(line.strip() for line in status):
            log("ℹ️  No hay cambios para commitear.")
            return True, output + ["Sin cambios."]

        # ── git commit ────────────────────────────────────────────
        commit_result = self._run(
            ["git", "commit", "-m", message],
            cwd=self.local_dir,
        )
        output.extend(commit_result)
        log(f"💾 Commit: {message}")

        # ── git push ──────────────────────────────────────────────
        log(f"🚀 Pushing a {self.repo_url} ({self.branch})...")
        push_result = self._run(
            ["git", "push", self.auth_url, self.branch],
            cwd=self.local_dir,
            mask_url=True,   # oculta el token en el log
        )
        output.extend(push_result)

        success = not any("error" in l.lower() or "fatal" in l.lower()
                          for l in push_result)
        if success:
            log("✅ Push exitoso a GitHub!")
        else:
            log("❌ Error en el push. Revisa el token y permisos del repositorio.")

        return success, output

    # ──────────────────────────────────────────
    # Estado del repositorio
    # ──────────────────────────────────────────

    def get_status(self) -> Dict:
        """Retorna el estado actual del repositorio local."""
        if not (self.local_dir / ".git").exists():
            return {"initialized": False, "message": "Repositorio no clonado aún."}

        branch = self._run(["git", "branch", "--show-current"], cwd=self.local_dir)
        last_commit = self._run(
            ["git", "log", "--oneline", "-1"], cwd=self.local_dir
        )
        diff_stat = self._run(
            ["git", "diff", "--stat", "--cached"], cwd=self.local_dir
        )

        return {
            "initialized":   True,
            "local_dir":     str(self.local_dir),
            "branch":        branch[0].strip() if branch else "unknown",
            "last_commit":   last_commit[0].strip() if last_commit else "N/A",
            "staged_changes": "\n".join(diff_stat).strip() or "Ninguno",
        }

    def list_remote_models(self) -> List[str]:
        """Lista archivos de modelos en el repositorio remoto (models/ dir)."""
        models_dir = self.local_dir / "models"
        if not models_dir.exists():
            return []
        exts = {".pth", ".pt", ".onnx", ".safetensors", ".bin"}
        return [
            str(f.relative_to(self.local_dir))
            for f in models_dir.rglob("*")
            if f.is_file() and f.suffix in exts
        ]

    # ──────────────────────────────────────────
    # Empaquetado de modelos
    # ──────────────────────────────────────────

    def package_and_push_model(
        self,
        model_path: str | Path,
        export_format: str = "pth",
        commit_message: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[bool, str]:
        """Copia un modelo al directorio del repo y hace push.

        Args:
            model_path: Ruta al archivo de modelo local
            export_format: "pth", "onnx", o "safetensors"
            commit_message: Mensaje del commit (auto si None)
            progress_callback: Callback de progreso

        Returns:
            Tuple (success, message)
        """
        def log(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        model_path = Path(model_path)
        if not model_path.exists():
            return False, f"Archivo no encontrado: {model_path}"

        # Copiar al directorio models/ del repositorio
        dest_dir = self.local_dir / "models"
        dest_dir.mkdir(exist_ok=True)
        dest_path = dest_dir / model_path.name

        import shutil
        shutil.copy2(model_path, dest_path)
        log(f"📋 Copiado {model_path.name} → {dest_path}")

        msg = commit_message or f"feat: add model {model_path.name}"
        success, output = self.commit_and_push(
            files=[dest_path.relative_to(self.local_dir)],
            message=msg,
            progress_callback=progress_callback,
        )

        status_msg = "\n".join(output)
        return success, status_msg

    # ──────────────────────────────────────────
    # Helpers privados
    # ──────────────────────────────────────────

    def _configure_git(self) -> None:
        """Configura identidad Git local."""
        if not (self.local_dir / ".git").exists():
            return
        self._run(["git", "config", "user.name",  self.user_name],  cwd=self.local_dir)
        self._run(["git", "config", "user.email", self.user_email], cwd=self.local_dir)

    def _build_auth_url(self, repo_url: str, token: str) -> str:
        """Inserta el token en la URL HTTPS para autenticación."""
        # https://TOKEN@github.com/AderDevP/IA-Models
        url = repo_url.replace("https://", f"https://{token}@")
        return url

    def _run(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        mask_url: bool = False,
    ) -> List[str]:
        """Ejecuta un comando git y retorna las líneas de salida."""
        try:
            result = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                capture_output=True,
                text=True,
                timeout=300,
            )
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            output = []
            if stdout:
                output.extend(stdout.splitlines())
            if stderr:
                # Enmascarar token en logs
                safe_stderr = stderr.replace(self.token, "***TOKEN***") if mask_url else stderr
                output.extend(safe_stderr.splitlines())

            return output
        except subprocess.TimeoutExpired:
            return ["❌ Timeout — operación Git tardó más de 5 minutos."]
        except FileNotFoundError:
            return ["❌ Git no encontrado. Instala git en el sistema."]
        except Exception as e:
            return [f"❌ Error: {e}"]


# ──────────────────────────────────────────────────────────────────
# Función de conveniencia para usar desde el Dashboard
# ──────────────────────────────────────────────────────────────────

def quick_push(
    token: str,
    local_dir: str | Path,
    files: Optional[List[str | Path]] = None,
    message: str = "feat: update from MammoAI",
    add_all: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[bool, str]:
    """Función de alto nivel para push rápido desde el Dashboard."""
    gm = GitManager(token=token, local_dir=local_dir)
    success, output = gm.commit_and_push(
        files=files,
        message=message,
        add_all=add_all,
        progress_callback=progress_callback,
    )
    return success, "\n".join(output)


# Importación lazy para evitar error en type hint
from typing import Dict
