"""
MDSTM - Modulo de Descarga del Sistema de Transduccion de Modelos (LucIA 2026)
==============================================================================
Capacidad interna REAL de LucIA para descargar modelos de IA ligeros de forma
autonoma en LC/modelosIAlocal. Expone la clase GestorDescargaModelos que el
orquestador mainLCSTM importa y usa sin pedir permiso al modelo remoto.

Capacidades:
  - Perfil de hardware (RAM/VRAM/CPU/disco) con persistencia JSON.
  - Catalogo de modelos ligeros (Ollama + GGUF HuggingFace).
  - Descarga autonoma via Ollama (`ollama pull` / /api/pull) o GGUF directo.
  - Interprete de ordenes en lenguaje natural ("descarga el modelo X").
  - Deteccion de intencion de descarga para interceptar antes de OpenRouter.
  - Ejecucion en hilo (no bloquea el turno de dialogo).
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-MDSTM-Descargas"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.MDSTM")

MDSTM_DIR: Final[Path] = Path(__file__).parent.resolve()
MODELOS_DIR: Final[Path] = MDSTM_DIR.parent.resolve()
PERFIL_HW: Final[Path] = MODELOS_DIR / "perfil_hardware.json"
REGISTRO: Final[Path] = MODELOS_DIR / "mdstm_registro.json"
OLLAMA_HOST: Final[str] = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

CATALOGO: Final[List[Dict[str, Any]]] = [
    {"id": "qwen2.5:0.5b", "tag": "qwen2.5:0.5b", "gb": 0.4, "ram": 2.0, "vram": 0.0,
     "gguf": "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
     "desc": "Ultraligero multilingue, ideal sin GPU."},
    {"id": "tinyllama:1.1b", "tag": "tinyllama", "gb": 0.7, "ram": 2.0, "vram": 0.0,
     "gguf": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
     "desc": "Conversacional 1.1B, corre en cualquier CPU."},
    {"id": "qwen2.5:1.5b", "tag": "qwen2.5:1.5b", "gb": 1.0, "ram": 4.0, "vram": 0.0,
     "gguf": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
     "desc": "Equilibrio calidad/velocidad en PCs modestos."},
    {"id": "llama3.2:1b", "tag": "llama3.2:1b", "gb": 1.3, "ram": 4.0, "vram": 0.0,
     "gguf": "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
     "desc": "Instrucciones Meta optimizadas para CPU."},
    {"id": "gemma2:2b", "tag": "gemma2:2b", "gb": 1.6, "ram": 6.0, "vram": 2.0,
     "gguf": "https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
     "desc": "Google eficiente para seguir instrucciones."},
    {"id": "qwen2.5:3b", "tag": "qwen2.5:3b", "gb": 2.0, "ram": 8.0, "vram": 3.0,
     "gguf": "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
     "desc": "Buen programador local con 3B parametros."},
    {"id": "llama3.2:3b", "tag": "llama3.2:3b", "gb": 2.0, "ram": 8.0, "vram": 3.0,
     "gguf": "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
     "desc": "Uso general solido si hay 8GB RAM."},
    {"id": "phi3:mini", "tag": "phi3:mini", "gb": 2.3, "ram": 6.0, "vram": 2.0,
     "gguf": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
     "desc": "Razonamiento compacto de Microsoft."},
]

PATRON_DESCARGA: Final[re.Pattern] = re.compile(
    r"\b(descarg\w*|baj\w*|instal\w*|trae\w*|pon\w*)\b\s+(el\s+)?(modelo\s+)?(.+)?",
    re.IGNORECASE,
)
PALABRAS_MODELO: Final[List[str]] = ["modelo", "modelos", "ia", "qwen", "llama",
                                     "tinyllama", "phi", "gemma", "ollama", "gguf", "local"]

_LOCK: Final[threading.Lock] = threading.Lock()


def _ram_gb() -> Tuple[float, float]:
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return round(vm.total / 1e9, 2), round(vm.available / 1e9, 2)
    except Exception:
        pass
    try:
        out = subprocess.run(["wmic", "computersystem", "get", "totalphysicalmemory"],
                             capture_output=True, text=True, timeout=5)
        nums = "".join(c if c.isdigit() else " " for c in out.stdout).split()
        if nums:
            total = int(nums[0]) / 1e9
            return round(total, 2), round(total * 0.5, 2)
    except Exception:
        pass
    return 8.0, 4.0


def _vram_gb() -> Tuple[float, str]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,name",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            parte = out.stdout.strip().splitlines()[0].split(",")
            return round(float(parte[0].strip()) / 1024.0, 2), parte[1].strip() if len(parte) > 1 else "NVIDIA"
    except Exception:
        pass
    return 0.0, "sin GPU dedicada (CPU)"


_GESTOR: Optional[GestorDescargaModelos] = None


def get_gestor_mdstm() -> GestorDescargaModelos:
    """Singleton: la instancia de descarga autonoma de LucIA."""
    global _GESTOR
    with _LOCK:
        if _GESTOR is None:
            _GESTOR = GestorDescargaModelos()
        return _GESTOR


def ordenar_descarga_lucia(texto_usuario: str) -> Optional[str]:
    """Atajo: interpreta y ejecuta ordenes de descarga. None si no es orden."""
    return get_gestor_mdstm().ejecutar_orden(texto_usuario)


def capacidad_mdstm() -> str:
    """Frase lista para inyectar en el prompt del sistema de LucIA."""
    return get_gestor_mdstm().capacidad_real()


if __name__ == "__main__":
    g = get_gestor_mdstm()
    print("=" * 70)
    print(f"  MDSTM v{__version__} - Descargas autonomas de LucIA")
    print("=" * 70)
    print(" ", g.capacidad_real())
    print("  Recomendados:", ", ".join(m["id"] for m in g.recomendar()[:3]))
    print("  Ollama:", g.ollama_online(), "| Descargados:", len(g.listar_descargados()))
    print("=" * 70)
from __init___GestorDescargaModelos import GestorDescargaModelos  # CLASSPACK
