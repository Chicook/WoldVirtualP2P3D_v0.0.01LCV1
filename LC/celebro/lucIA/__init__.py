"""
lucIA - Sistema de Inteligencia Artificial Distribuida y Redes Neuronales
========================================================================

Versión reducida (<125 MB) optimizada para el entorno de WoldVirtualP2P3D.
Configuración automática de redirección de __pycache__ hacia Celebro/cache.
Persistencia de pesos en IPFS y limpieza completa de caché al cerrar sesión.
"""

import os
import sys
import logging
import random
from pathlib import Path

# 1b. Logging + seed centrales (única configuración global; los submódulos no reconfiguran).
try:
    import numpy as _np
    _np.random.seed(42)
except Exception:
    pass
try:
    random.seed(42)
except Exception:
    pass
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# 1. Configuración de rutas
_LUCIA_DIR = Path(__file__).parent.resolve()
_PARENT_DIR = _LUCIA_DIR.parent.resolve()

# Asegurar que tanto lucIA como su padre están en sys.path
for _p in [str(_LUCIA_DIR), str(_PARENT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 2. Configurar redirección de __pycache__ INMEDIATAMENTE
_CACHE_DIR = _LUCIA_DIR / "Celebro" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
sys.pycache_prefix = str(_CACHE_DIR)
os.environ["PYTHONPYCACHEPREFIX"] = str(_CACHE_DIR)

# 2b. Cargar .env de la raíz del proyecto (parser propio, sin dependencias).
# Busca .env en: raíz proyecto (padre de lucIA) y dir lucIA. No sobreescribe env real.
for _env in [_PARENT_DIR / ".env", _LUCIA_DIR / ".env"]:
    try:
        if _env.exists():
            for _line in _env.read_text(encoding="utf-8-sig").splitlines():
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    except Exception:
        pass

# 3. Exportar gestores y utilidades
from .session_manager import (
    iniciar_sesion,
    cerrar_sesion,
    LucIASession,
    obtener_sesion,
    verificar_limite_espacio,
    vaciar_directorio_cache,
    configurar_redireccion_pycache,
    codificar_versiones_en_pesos,
    codificar_memoria_en_pesos,
    codificar_cache_en_pesos,
    restaurar_pesos_desde_ipfs
)
from .ipfs_manager import IPFSManager, compute_ipfs_cidv0
from .CORE.base import LucIANeuronBase, LucIASystem
from .CORE.voice_engine import VoiceEngine, speak, cancel_speech
from .CORE.text_encoder import TextEncoder
from .llm_connector import HybridLLMConnector
from .Celebro.conversor_pesos import ConversorRespuestaPesos, get_conversor_pesos
from .CORE.auto_refactor import AutoRefactorEngine, get_autorefactor_engine
from .CORE.memory_manager import MemoryManager, get_memory_manager

__version__ = "0.0.1-reduced"
__all__ = [
    'iniciar_sesion',
    'cerrar_sesion',
    'LucIASession',
    'obtener_sesion',
    'IPFSManager',
    'compute_ipfs_cidv0',
    'LucIANeuronBase',
    'LucIASystem',
    'VoiceEngine',
    'speak',
    'cancel_speech',
    'TextEncoder',
    'HybridLLMConnector',
    'ConversorRespuestaPesos',
    'get_conversor_pesos',
    'AutoRefactorEngine',
    'get_autorefactor_engine',
    'MemoryManager',
    'get_memory_manager',
    'iniciar_chat_interactivo',
    'verificar_limite_espacio',
    'vaciar_directorio_cache',
    'configurar_redireccion_pycache',
    'codificar_versiones_en_pesos',
    'codificar_memoria_en_pesos',
    'codificar_cache_en_pesos',
    'restaurar_pesos_desde_ipfs'
]

# [AUTONOMIA LucIA] Ultima auto-inspeccion: 2026-09-06 19:44:31
