"""
IAFREE.py - SubSistema de Inferencia Gratuita OpenRouter para LucIA (Arquitectura 2026)
========================================================================================
Gestiona el acceso y rotacion automatica de modelos 100% gratuitos de OpenRouter:
  - Registro exhaustivo y actualizado de modelos :free disponibles en la plataforma.
  - Sincronizacion dinamica con la API de OpenRouter (https://openrouter.ai/api/v1/models).
  - Rotacion automatica ante limites de tasa (HTTP 429 Too Many Requests) o 404.
  - Filtro por capacidades: conversacion general, razonamiento, programacion y analisis.
  - Inyeccion del contexto cognitivo de las 50 neuronas activas de Celebro.
  - Modo streaming y modo sincrono con registro de latencia y metricas de consumo $0.00.
  - Integracion transparente con BKSVCB (registro en bloques) y SNSBSTNPRB (sesion).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Final, Generator, List, Optional, Tuple, Union

# ─── VERSION Y METADATOS DEL SUBSISTEMA ──────────────────────────────────────
__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-IAFREE-Subsystem"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.IAFREE")

# ─── RUTAS Y UBICACIONES CRITICAS ───────────────────────────────────────────
PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
ROOT_DIR: Final[Path] = LC_DIR.parent.resolve()
ENV_FILE: Final[Path] = ROOT_DIR / ".env"
CACHE_FILE: Final[Path] = PACKAGE_ROOT / "openrouter_free_cache.json"

# ─── LISTA BASE DE MODELOS GRATUITOS VERIFICADOS (2026) ─────────────────────
# Obtenidos en vivo desde https://openrouter.ai/models?q=free
CATALOGO_MODELOS_GRATUITOS: Final[List[Dict[str, Any]]] = [
    {"id": "openrouter/free", "nombre": "OpenRouter Free Router", "contexto": 200000, "categoria": "general", "descripcion": "Enrutador oficial que auto-selecciona el mejor modelo gratuito."},
    {"id": "qwen/qwen3.8-27b:free", "nombre": "Qwen 3.8 27B Free", "contexto": 262144, "categoria": "razonamiento", "descripcion": "Gran modelo general con 256k tokens de ventana de contexto."},
    {"id": "google/gemma-4-31b-it:free", "nombre": "Google Gemma 4 31B Instruct Free", "contexto": 262144, "categoria": "instruccion", "descripcion": "Seguimiento estricto de directrices tecnicas y formato."},
    {"id": "google/gemma-4-26b-a4b-it:free", "nombre": "Google Gemma 4 26B A4B Free", "contexto": 262144, "categoria": "ligero", "descripcion": "Arquitectura eficiente de atencion para minima latencia."},
    {"id": "nvidia/nemotron-3-super-120b-a12b:free", "nombre": "NVIDIA Nemotron 3 Super 120B Free", "contexto": 262144, "categoria": "razonamiento", "descripcion": "Modelo masivo de logica y resolucion analitica profunda."},
    {"id": "nvidia/nemotron-3.5-lightning:free", "nombre": "NVIDIA Nemotron 3.5 Lightning Free", "contexto": 1000000, "categoria": "gran_contexto", "descripcion": "1 Millon de tokens de contexto con alta velocidad de inferencia."},
    {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "nombre": "NVIDIA Nemotron 3 Ultra 550B Free", "contexto": 1000000, "categoria": "pesado", "descripcion": "Arquitectura MoE de maximo parametro con coste cero."},
    {"id": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "nombre": "NVIDIA Nemotron 3 Nano Reasoning Free", "contexto": 256000, "categoria": "razonamiento", "descripcion": "Razonamiento por pasos y logica formal."},
    {"id": "cohere/north-mini-code:free", "nombre": "Cohere North Mini Code Free", "contexto": 256000, "categoria": "codigo", "descripcion": "Especializado en generacion, refactorizacion y sintaxis de codigo."},
    {"id": "z-ai/glm-5.2:free", "nombre": "Z-AI GLM 5.2 Free", "contexto": 32768, "categoria": "general", "descripcion": "Conversacion rapida y concisa con baja utilizacion de memoria."},
    {"id": "liquid/lfm-2.5-2.6b:free", "nombre": "Liquid LFM 2.5 2.6B Free", "contexto": 65536, "categoria": "eficiente", "descripcion": "Red neuronal de estado liquido con generacion ultrarrapida."},
    {"id": "thinkingmachines/inkling:free", "nombre": "ThinkingMachines Inkling Free", "contexto": 1048576, "categoria": "gran_contexto", "descripcion": "Contexto extendido de 1M tokens para analisis de grandes textos."},
    {"id": "thinkingmachines/inkling-small:free", "nombre": "ThinkingMachines Inkling Small Free", "contexto": 1048576, "categoria": "gran_contexto", "descripcion": "Version compacta optimizada para respuestas inmediatas."},
    {"id": "poolside/laguna-s-2.1:free", "nombre": "Poolside Laguna S 2.1 Free", "contexto": 262144, "categoria": "codigo", "descripcion": "Optimizacion de nivel produccion para arquitectura de software."},
    {"id": "poolside/laguna-xs-2.1:free", "nombre": "Poolside Laguna XS 2.1 Free", "contexto": 262144, "categoria": "codigo", "descripcion": "Asistencia ultraligera para programacion interactiva."},
    {"id": "dots-studio/dots-3-note-preview:free", "nombre": "Dots 3 Note Preview Free", "contexto": 512000, "categoria": "analisis", "descripcion": "512k tokens dedicados a sintesis y notas estructuradas."},
    {"id": "nex-agi/nex-n2.5-mini:free", "nombre": "Nex-AGI N2.5 Mini Free", "contexto": 262144, "categoria": "general", "descripcion": "Respuestas agiles para dialogos de soporte continuo."},
    {"id": "nex-agi/nex-n2.5-pro:free", "nombre": "Nex-AGI N2.5 Pro Free", "contexto": 262144, "categoria": "razonamiento", "descripcion": "Capacidades avanzadas de comprension logica."},
    {"id": "inclusionai/ling-3.0-flash-vl:free", "nombre": "InclusionAI Ling 3.0 Flash VL Free", "contexto": 262144, "categoria": "multimodal", "descripcion": "Comprension multimodal y razonamiento veloz."},
    {"id": "inclusionai/ling-3.0-flash-fin:free", "nombre": "InclusionAI Ling 3.0 Flash Fin Free", "contexto": 262144, "categoria": "analisis", "descripcion": "Especializado en analitica cuantitativa y calculos precisos."},
    {"id": "inclusionai/ling-3.0-flash-sante:free", "nombre": "InclusionAI Ling 3.0 Flash Sante Free", "contexto": 262144, "categoria": "general", "descripcion": "Alta fidelidad y alineacion etica de respuestas."},
]


# ─── GESTOR DE ROTACION Y DISPONIBILIDAD DE MODELOS ─────────────────────────
_INSTANCIA_IAFREE: Optional[ClienteIAFree] = None
_LOCK_SINGLETON: Final[threading.Lock] = threading.Lock()


def get_cliente_iafree() -> ClienteIAFree:
    """Devuelve la instancia unica y compartida del cliente de inferencia gratuita."""
    global _INSTANCIA_IAFREE
    with _LOCK_SINGLETON:
        if _INSTANCIA_IAFREE is None:
            _INSTANCIA_IAFREE = ClienteIAFree()
        return _INSTANCIA_IAFREE


def consultar_lucia_gratis(prompt: str, contexto_neuronal: Optional[Dict[str, Any]] = None) -> str:
    """Punto de acceso universal para obtener respuestas de LucIA a coste cero ($0.00)."""
    cliente = get_cliente_iafree()
    respuesta, _, _ = cliente.generar_respuesta(prompt, contexto_neuronal=contexto_neuronal, stream_en_vivo=True)
    return respuesta


def listar_catalogo_gratis() -> List[Dict[str, Any]]:
    """Devuelve el catalogo de modelos sin costo disponibles para la sesion."""
    return get_cliente_iafree().gestor.listar_modelos()


def actualizar_modelos_en_red() -> int:
    """Fuerza la sincronizacion del catalogo en tiempo real con OpenRouter."""
    return get_cliente_iafree().gestor.actualizar_catalogo_online()


def obtener_auditoria_costo_cero() -> Dict[str, Any]:
    """Genera comprobante de uso 100% libre de costo."""
    return get_cliente_iafree().obtener_metricas_consumo()


# ─── PUNTO DE ENTRADA EN CONSOLA Y TEST DE DIAGNOSTICO ──────────────────────
if __name__ == "__main__":
    print("\n\033[1;36m" + "=" * 74 + "\033[0m")
    print(f"  \033[1;32mSUBSISTEMA DE INFERENCIA GRATUITA -- {__subsystem__} v{__version__}\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m")

    cliente = get_cliente_iafree()
    print(f"  Autenticado en OpenRouter : \033[1;33m{cliente.esta_autenticado()}\033[0m")
    print(f"  Modelos :free precargados : \033[1;35m{len(CATALOGO_MODELOS_GRATUITOS)}\033[0m")
    print("  Actualizando modelos en vivo desde OpenRouter API...")
    total = cliente.gestor.actualizar_catalogo_online()
    print(f"  Total modelos :free listos: \033[1;32m{total}\033[0m\n")

    print("\033[1;37mModelos Gratuitos Principales (Coste $0.00):\033[0m")
    for m in cliente.gestor.listar_modelos()[:8]:
        print(f"  * \033[1;36m{m['id']}\033[0m | {m['contexto']} tok | {m['nombre']}")

    print("\n\033[1;32mRealizando prueba de inferencia a coste cero ($0.00)...\033[0m")
    resp, mod, lat = cliente.generar_respuesta(
        "Hola LucIA, confirma que este subsistema de inferencia es 100% gratuito.",
        contexto_neuronal={"tono_cognitivo": "seguro y analitico", "estado_emocional": 0.45},
        stream_en_vivo=True,
    )
    print(f"\n  Modelo utilizado: \033[1;33m{mod}\033[0m | Latencia: \033[1;32m{lat:.0f}ms\033[0m")
    auditoria = cliente.obtener_metricas_consumo()
    print(f"  Auditoria de costo : \033[1;32m${auditoria['costo_acumulado_usd']:.2f} USD\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m\n")
from IAFREE_GestorModelosGratuitos import GestorModelosGratuitos  # CLASSPACK
from IAFREE_ClienteIAFree import ClienteIAFree  # CLASSPACK
