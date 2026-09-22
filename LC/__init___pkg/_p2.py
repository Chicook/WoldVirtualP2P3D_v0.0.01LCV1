"""
__init__ - parte 2/2 (version de sesion LucIA).
Parte del subsistema LucIA (reprocesar_respuesta_lucia).
"""
from __future__ import annotations

"""
SBSTM - SubSistema de Sesion y Transito Neuronal (Arquitectura WoldVirtualP2P3D 2026)
====================================================================================
Subpaquete especializado en la orquestacion de sesiones interactivas, sincronizacion
de bloques transaccionales y pasarela conversacional neuronal P2P:
  - Gestion de ciclo de vida de sesion neuronal (SesionNeuronalP2P / SNSBSTNPRB).
  - Enlace dinamico con la cadena criptografica BKSVCB y transductor PSNRCV.
  - Streaming con modelos de lenguaje locales (Ollama: cogito:3b, qwen2.5:7b, etc.).
  - Transduccion de tokens en transacciones sinapticas y minado por turnos.
  - Control de telemetria en tiempo real: GSNR, deriva Muon y entropia de Shannon.
  - Checkpoint y persistencia automatica en PSNRL e IPFS mediante ganchos atexit/signal.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Final, Generator, List, Optional, Tuple, Union

__version__: Final[str] = "2026.3.1"
__author__: Final[str] = "Equipo Celebro WoldVirtualP2P3D"
__status__: Final[str] = "Production / Optimized"
__package_name__: Final[str] = "LC.celebro.CMFG.SBSTM"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.CMFG.SBSTM")

# ─── JERARQUIA DE DIRECTORIOS Y RUTA RAIS ────────────────────────────────────

def reprocesar_respuesta_lucia(texto_bruto: str, contexto_neuronal: Optional[Dict[str, Any]] = None) -> str:
    """
    Conveniencia: pasa el texto crudo de OpenRouter por el pipeline RPLC
    y devuelve la respuesta reformulada en la voz propia de LucIA.
    Si RPLC no está disponible, devuelve el texto original sin modificar.
    """
    try:
        from LC.celebro.CMFG.SBSTM.RPLC import reprocesar_para_lucia
        return reprocesar_para_lucia(texto_bruto, contexto_neuronal)
    except Exception:
        return texto_bruto


# ─── EXPORTACIONES PUBLICAS OFICIALES DEL PAQUETE ────────────────────────────
__all__: Final[List[str]] = [
    "TerminalColors",
    "SBSTMSessionContext",
    "obtener_directorio_sbstm",
    "obtener_estado_subsistema",
    "registrar_gancho_post_turno",
    "verificar_integridad_sbstm",
    "obtener_clase_sesion",
    "crear_sesion_neuronal",
    "ejecutar_orquestador_interactivo",
    "consultar_modelos_disponibles",
    "recopilar_estadisticas_sesion",
    "notificar_turno_completado",
    "verificar_cadena_bloques_activa",
    "obtener_ultimo_hash_bloque",
    "forzar_checkpoint_psnrl",
    "purgar_archivos_temporales",
    "obtener_servicio_iafree",
    "obtener_motor_stylos",
    # RPLC
    "obtener_procesador_rplc",
    "reprocesar_respuesta_lucia",
    # VOICE ENGINE
    "obtener_motor_voz",
    "reproducir_voz_lucia",
    "cancelar_voz_lucia",
    # DSIALCLGRG (IA local ligera / fallback OpenRouter)
    "obtener_gestor_ia_local",
    "consultar_ia_local",
    "consultar_con_fallback",
]


def obtener_motor_voz() -> Any:
    """Retorna la instancia global del motor de voz de LucIA."""
    try:
        from LC.celebro.CMFG.SBSTM.voice_engine import get_voice_engine
        return get_voice_engine()
    except Exception:
        return None


def reproducir_voz_lucia(texto: str, esperar: bool = False, emocion: Optional[float] = None) -> None:
    """Reproduce texto con la voz juvenil oficial de LucIA (es-ES-ElviraNeural)."""
    try:
        from LC.celebro.CMFG.SBSTM.voice_engine import speak
        speak(texto, esperar=esperar, emocion=emocion)
    except Exception:
        pass


def cancelar_voz_lucia() -> None:
    """Interrumpe inmediatamente cualquier emision sonora de LucIA."""
    try:
        from LC.celebro.CMFG.SBSTM.voice_engine import cancel_speech
        cancel_speech()
    except Exception:
        pass


# ─── INTEGRACIÓN DSIALCLGRG (IA local ligera / fallback OpenRouter) ──────────
def obtener_gestor_ia_local() -> Any:
    """Retorna el gestor singleton de IA local (perfil HW + recomendados)."""
    try:
        from LC.celebro.CMFG.SBSTM.DSIALCLGRG import get_gestor_local
        return get_gestor_local()
    except Exception:
        return None


def consultar_ia_local(prompt: str, contexto_neuronal: Optional[Dict[str, Any]] = None) -> str:
    """Consulta el modelo local ligero (Ollama/GGUF) cuando OpenRouter falla."""
    try:
        from LC.celebro.CMFG.SBSTM.DSIALCLGRG import consultar_lucia_local
        texto, _, _ = consultar_lucia_local(prompt, contexto_neuronal)
        return texto
    except Exception:
        return "[DSIALCLGRG] IA local no disponible."


def consultar_con_fallback(prompt: str, contexto_neuronal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """OpenRouter primero; si falla, IA local en LC/modelosIAlocal."""
    try:
        from LC.celebro.CMFG.SBSTM.DSIALCLGRG import consultar_con_fallback as _fb
        return _fb(prompt, contexto_neuronal)
    except Exception:
        return {"texto": "[DSIALCLGRG] Sin backends disponibles.", "modelo": "none", "latencia_ms": 0.0, "fuente": "none"}
