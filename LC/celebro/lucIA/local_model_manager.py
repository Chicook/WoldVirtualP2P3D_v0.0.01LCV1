"""
lucIA.local_model_manager - Gestor Autónomo de Descarga de Modelos Locales (Ollama y LM Studio)
================================================================================================

Objetivo:
Garantizar que en cada sesión se verifique y al menos 1 modelo de IA local operativo
se descargue o actualice (vía Ollama pull o LM Studio CLI lms get), priorizando
modelos ligeros y ultrarrápidos para asegurar autonomía sin saturar disco ni memoria.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Compatibilidad de codificación en terminales Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("lucIA.LocalModelManager")


class LocalModelCandidate:
    """Representa un modelo candidato para ejecución y descarga local."""

    def __init__(
        self,
        name: str,
        backend: str,
        size_tag: str,
        priority: int = 1,
        description: str = "",
    ):
        self.name = name
        self.backend = backend.lower()  # 'ollama' o 'lmstudio'
        self.size_tag = size_tag
        self.priority = priority
        self.description = description

    def __repr__(self) -> str:
        return f"<LocalModelCandidate {self.backend}:{self.name} ({self.size_tag})>"


class LocalModelManager:
    """
    Gestor que supervisa modelos locales en Ollama y LM Studio,
    y ejecuta la descarga autónoma de al menos 1 modelo local por sesión.
    """

    REGISTRO_FILE = Path(__file__).parent / "config" / "local_downloads_history.json"

    # Modelos recomendados ligeros, rápidos y operativos
    CANDIDATOS_OLLAMA: List[LocalModelCandidate] = [
        LocalModelCandidate("qwen3:1.7b", "ollama", "1.4GB", 1, "Qwen3 1.7B ligero para razonamiento y diálogo"),
        LocalModelCandidate("qwen2.5:7b", "ollama", "4.7GB", 2, "Qwen2.5 7B equilibrio óptimo en español"),
        LocalModelCandidate("qwen2.5-coder:3b", "ollama", "1.9GB", 3, "Qwen2.5 Coder 3B código y lógica estructurada"),
        LocalModelCandidate("qwen2.5:1.5b", "ollama", "986MB", 4, "Qwen2.5 1.5B ultraligero"),
        LocalModelCandidate("qwen2.5:0.5b", "ollama", "398MB", 5, "Qwen2.5 0.5B micro-modelo rápido"),
        LocalModelCandidate("llama3.2:1b", "ollama", "1.3GB", 6, "Llama 3.2 1B micro ultra-rápido"),
        LocalModelCandidate("gemma2:2b", "ollama", "1.6GB", 7, "Gemma 2 2B fluidez conversacional"),
    ]

    CANDIDATOS_LMSTUDIO: List[LocalModelCandidate] = [
        LocalModelCandidate("qwen2.5-0.5b-instruct", "lmstudio", "531MB", 1, "Qwen2.5 0.5B GGUF para LM Studio"),
        LocalModelCandidate("deepseek-r1-draft-qwen2.5-coder-0.5b", "lmstudio", "531MB", 2, "DeepSeek R1 Draft 0.5B"),
        LocalModelCandidate("deepseek-coder-1.3b-instruct", "lmstudio", "704MB", 3, "DeepSeek Coder 1.3B"),
        LocalModelCandidate("qwen3-zero-coder-reasoning-v2-0.8b-neo-ex", "lmstudio", "790MB", 4, "Qwen3 Zero 0.8B"),
        LocalModelCandidate("kimi-coder-135m", "lmstudio", "145MB", 5, "Kimi Coder 135M micro"),
    ]

    def __init__(self, timeout_descarga_s: int = 180):
        self.timeout_descarga_s = timeout_descarga_s
        self.registro_file = self.REGISTRO_FILE
        self.modelos_ollama_locales: List[str] = []
        self.modelos_lmstudio_locales: List[str] = []
        self.historial: List[Dict[str, Any]] = []
        self._cargar_historial()

    def _cargar_historial(self) -> None:
        try:
            if self.registro_file.exists():
                data = json.loads(self.registro_file.read_text(encoding="utf-8"))
                self.historial = data.get("descargas", [])
        except Exception as e:
            logger.warning(f"No se pudo cargar historial de descargas locales: {e}")
            self.historial = []

    def _guardar_historial(self) -> None:
        try:
            self.registro_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ultima_actualizacion": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_descargas": len(self.historial),
                "descargas": self.historial[-100:],
            }
            self.registro_file.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"No se pudo guardar historial de descargas locales: {e}")

    def detectar_modelos_ollama(self) -> List[str]:
        """Consulta modelos locales disponibles en Ollama (API /api/tags o CLI)."""
        modelos = []
        # Intento 1: API HTTP de Ollama
        try:
            req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for m in data.get("models", []):
                    name = m.get("name") or m.get("model")
                    if name:
                        modelos.append(name)
        except Exception:
            pass

        # Intento 2: CLI de Ollama si la API no respondió o faltó
        if not modelos and shutil.which("ollama"):
            try:
                res = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if res.returncode == 0:
                    for line in res.stdout.strip().splitlines()[1:]:
                        partes = line.split()
                        if partes:
                            modelos.append(partes[0])
            except Exception:
                pass

        self.modelos_ollama_locales = list(dict.fromkeys(modelos))
        return self.modelos_ollama_locales

    def detectar_modelos_lmstudio(self) -> List[str]:
        """Consulta modelos disponibles en LM Studio (API /v1/models o CLI lms)."""
        modelos = []
        # Intento 1: API HTTP local de LM Studio
        try:
            req = urllib.request.Request("http://127.0.0.1:1234/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                for item in data.get("data", []):
                    mid = item.get("id")
                    if mid and "embed" not in mid.lower():
                        modelos.append(mid)
        except Exception:
            pass

        # Intento 2: CLI lms
        if not modelos and shutil.which("lms"):
            try:
                res = subprocess.run(
                    ["lms", "ls"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                if res.returncode == 0:
                    for line in res.stdout.splitlines():
                        line_s = line.strip()
                        if line_s and not line_s.startswith("Server:") and not line_s.startswith("LLM") and not line_s.startswith("No Models"):
                            partes = line_s.split()
                            if partes and not line_s.startswith("text-embedding"):
                                modelos.append(partes[0])
            except Exception:
                pass

        self.modelos_lmstudio_locales = list(dict.fromkeys(modelos))
        return self.modelos_lmstudio_locales

    def descargar_modelo_ollama(self, modelo: str) -> bool:
        """Descarga un modelo usando 'ollama pull <modelo>'."""
        if not shutil.which("ollama"):
            logger.warning("[LocalModelManager] 'ollama' CLI no disponible en PATH.")
            return False

        logger.info(f"⬇️ [Ollama] Iniciando descarga/actualización de modelo local: {modelo}...")
        print(f"⬇️ [LucIA Local]: Descargando modelo local en Ollama: '{modelo}'...")
        try:
            proc = subprocess.run(
                ["ollama", "pull", modelo],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_descarga_s,
                check=False,
            )
            if proc.returncode == 0:
                print(f"✅ [LucIA Local]: Modelo Ollama '{modelo}' descargado y operativo con éxito.")
                self.historial.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "backend": "ollama",
                    "modelo": modelo,
                    "exito": True,
                })
                self._guardar_historial()
                return True
            else:
                logger.warning(f"[Ollama pull] Falló ({proc.returncode}): {proc.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"⚠️ [LucIA Local]: Tiempo de descarga excedido para '{modelo}'.")
        except Exception as e:
            logger.warning(f"Error descargando modelo Ollama {modelo}: {e}")
        return False

    def descargar_modelo_lmstudio(self, modelo: str) -> bool:
        """Descarga un modelo usando 'lms get <modelo> -y'."""
        if not shutil.which("lms"):
            logger.warning("[LocalModelManager] 'lms' CLI no disponible en PATH.")
            return False

        logger.info(f"⬇️ [LM Studio] Iniciando descarga/verificación de modelo local: {modelo}...")
        print(f"⬇️ [LucIA Local]: Verificando/Descargando modelo en LM Studio: '{modelo}'...")
        try:
            proc = subprocess.run(
                ["lms", "get", modelo, "-y"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_descarga_s,
                check=False,
            )
            if proc.returncode == 0:
                print(f"✅ [LucIA Local]: Modelo LM Studio '{modelo}' listo y operativo.")
                self.historial.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "backend": "lmstudio",
                    "modelo": modelo,
                    "exito": True,
                })
                self._guardar_historial()
                return True
            else:
                logger.warning(f"[lms get] Falló ({proc.returncode}): {proc.stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"⚠️ [LucIA Local]: Tiempo de descarga excedido para LM Studio '{modelo}'.")
        except Exception as e:
            logger.warning(f"Error descargando modelo LM Studio {modelo}: {e}")
        return False

    def asegurar_un_modelo_por_sesion(self, async_mode: bool = False) -> Dict[str, Any]:
        """
        En cada sesión:
        1. Detecta modelos instalados en Ollama y LM Studio.
        2. Determina qué modelos de la lista recomendada aún no han sido descargados.
        3. Si hay alguno pendiente, lo descarga (al menos 1 modelo).
        4. Si todos ya están descargados, comprueba y actualiza el primario más ligero.
        """
        def _proceso() -> Dict[str, Any]:
            self.detectar_modelos_ollama()
            self.detectar_modelos_lmstudio()

            descargado = False
            modelo_elegido = None
            backend_elegido = None

            # Prioridad 1: Candidato de Ollama no presente
            for cand in self.CANDIDATOS_OLLAMA:
                ya_instalado = any(
                    cand.name == m or cand.name.split(":")[0] == m.split(":")[0]
                    for m in self.modelos_ollama_locales
                )
                if not ya_instalado:
                    modelo_elegido = cand.name
                    backend_elegido = "ollama"
                    descargado = self.descargar_modelo_ollama(cand.name)
                    if descargado:
                        break

            # Prioridad 2: Si Ollama no requirió o falló, candidato de LM Studio
            if not descargado:
                for cand in self.CANDIDATOS_LMSTUDIO:
                    ya_instalado = any(cand.name in m for m in self.modelos_lmstudio_locales)
                    if not ya_instalado:
                        modelo_elegido = cand.name
                        backend_elegido = "lmstudio"
                        descargado = self.descargar_modelo_lmstudio(cand.name)
                        if descargado:
                            break

            # Prioridad 3: Si todos ya están presentes, refrescar/asegurar el primario ligero de Ollama
            if not descargado and self.CANDIDATOS_OLLAMA:
                primario = self.CANDIDATOS_OLLAMA[0].name
                modelo_elegido = primario
                backend_elegido = "ollama"
                descargado = self.descargar_modelo_ollama(primario)

            # Actualizar listas tras la descarga
            self.detectar_modelos_ollama()
            self.detectar_modelos_lmstudio()

            resumen = {
                "descargado": descargado,
                "modelo": modelo_elegido,
                "backend": backend_elegido,
                "total_ollama": len(self.modelos_ollama_locales),
                "total_lmstudio": len(self.modelos_lmstudio_locales),
                "modelos_ollama": self.modelos_ollama_locales,
                "modelos_lmstudio": self.modelos_lmstudio_locales,
            }
            return resumen

        if async_mode:
            t = threading.Thread(target=_proceso, daemon=True, name="LucIALocalModelDownloader")
            t.start()
            return {"async": True, "status": "Iniciada descarga de modelo local en segundo plano"}
        else:
            return _proceso()


# Singleton global
_MANAGER: Optional[LocalModelManager] = None


def get_local_model_manager() -> LocalModelManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = LocalModelManager()
    return _MANAGER


if __name__ == "__main__":
    mgr = get_local_model_manager()
    print("Detectando modelos locales...")
    ol = mgr.detectar_modelos_ollama()
    print(f"Ollama ({len(ol)}):", ol)
    lm = mgr.detectar_modelos_lmstudio()
    print(f"LM Studio ({len(lm)}):", lm)
    print("\nAsegurando descarga de al menos 1 modelo local por sesión...")
    res = mgr.asegurar_un_modelo_por_sesion(async_mode=False)
    print("Resultado:", res)
