"""
STYLOS.py - Motor de Estilos Visuales y Renderizado de Terminal (Arquitectura 2026)
=====================================================================================
Subsistema visual premium para la presentacion de respuestas de LucIA y telemetria:
  - Paleta Cyber-Bioluminiscente (Electric Cyan, Violeta Neon, Ambar Solar, Esmeralda).
  - Cajas de dialogo delimitadas con bordes redondeados Unicode (╭─╮╰─╯│) y tarjetas.
  - Formateo inteligente de Markdown: negrita, listas, bloques de codigo syntax-highlight,
    tablas ASCII/Unicode alineadas y citas indentadas.
  - Streaming token a token suave con indicador de cursor pulsante.
  - Banners de estado, divisores y badges de telemetria sinaptica y blockchain.
  - Adaptabilidad de ancho de consola con deteccion automatica o fallback a 80 columnas.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import time
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

# ─── METADATOS DEL SUBSISTEMA ────────────────────────────────────────────────
__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-Terminal-Stylos"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

# ─── PALETA DE COLOR HEX/256 Y ANSI EXTENDIDO ────────────────────────────────
class Glifos:
    """Simbolos tipograficos y de diseno para bordes de tarjetas y telemetria."""
    ESQ_ARR_IZQ: Final[str] = "╭"
    ESQ_ARR_DER: Final[str] = "╮"
    ESQ_ABA_IZQ: Final[str] = "╰"
    ESQ_ABA_DER: Final[str] = "╯"
    LINEA_H: Final[str] = "─"
    LINEA_V: Final[str] = "│"
    CRUZ_IZQ: Final[str] = "├"
    CRUZ_DER: Final[str] = "┤"
    FLECHA: Final[str] = "◈"
    PUNTO_VIVO: Final[str] = "●"
    SPARKLE: Final[str] = "✦"
    BLOQUE_LLENO: Final[str] = "█"
    BLOQUE_MEDIO: Final[str] = "▒"
    BLOQUE_VACIO: Final[str] = "░"
    RAYO: Final[str] = "⚡"
    CEREBRO: Final[str] = "🧠"


# ─── UTILIDADES DE FORMATO Y ANCHO DE PANTALLA ──────────────────────────────
def obtener_ancho_consola(max_limite: int = 90) -> int:
    """Determina el ancho util de la terminal respetando limites para lectura optima."""
    try:
        columnas, _ = shutil.get_terminal_size((80, 24))
        return min(max_limite, max(60, columnas - 4))
    except Exception:
        return 80


def limpiar_codigos_ansi(texto: str) -> str:
    """Remueve secuencias de escape ANSI para calcular longitudes visuales reales."""
    patron_ansi = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return patron_ansi.sub("", texto)


def longitud_visual(texto: str) -> int:
    """Calcula el numero real de caracteres visibles en pantalla."""
    return len(limpiar_codigos_ansi(texto))


# ─── FORMATEADOR Y RENDERIZADOR MARKDOWN DE TERMINAL ─────────────────────────
def formatear_respuesta_lucia(texto: str, modelo: str = "openrouter/free") -> None:
    """Metodo directo para desplegar respuestas con calidad de terminal moderna."""
    EstiloTerminalLucIA.renderizar_respuesta_completa(texto, modelo_id=modelo)


def banner_bienvenida(sesion_id: str, modelo: str) -> None:
    """Despliega el banner estetico inicial de la sesion."""
    EstiloTerminalLucIA.renderizar_encabezado_sesion(sesion_id, modelo)


def badge_turno(turno: int, dt: float, mod: str, der: float, neu: int, cid: str, txs: int) -> None:
    """Emite la telemetria post-turno formateada."""
    EstiloTerminalLucIA.renderizar_barra_telemetria(turno, dt, mod, der, neu, cid, txs)


def panel_ayuda_comandos() -> None:
    """Despliega la ayuda visual de comandos."""
    EstiloTerminalLucIA.renderizar_panel_ayuda()


def tarjeta_diagnostico(titulo: str, datos: Dict[str, Any]) -> None:
    """Despliega una tarjeta estructurada de clave-valor."""
    EstiloTerminalLucIA.renderizar_tarjeta_sistema(titulo, datos)


# ─── DEMOSTRACION EN CONSOLA Y PRUEBA VISUAL ─────────────────────────────────
if __name__ == "__main__":
    banner_bienvenida("LUCIA_DEV_2026", "openrouter/free")

    demo_prompt = "¿Cual es el rol de las 50 neuronas en WoldVirtualP2P3D?"
    EstiloTerminalLucIA.renderizar_prompt_usuario(demo_prompt)

    demo_respuesta = (
        "Las **50 neuronas activas** se distribuyen en cinco familias especializadas:\n"
        "- `ENRN`: 10 neuronas de activacion y entrada recurrente.\n"
        "- `RF_EN`: 10 neuronas de modulacion por refuerzo adaptativo.\n"
        "- `RF_SL`: 10 neuronas residuales supervisadas.\n"
        "- `RNP`: 10 neuronas de plasticidad Hebbiana y memoria asociativa.\n"
        "- `SLRN`: 10 optimizadores de convergencia continua.\n\n"
        "### Resumen de Arquitectura\n"
        "| Familia | Capacidad | Optimizador |\n"
        "|---|---|---|\n"
        "| ENRN | Entrada 4D a 8D | SOAP Curvature |\n"
        "| RNP | Memoria Distribuida | Hebbian Plasticity |\n"
        "| SLRN | Optimizacion Continua | Muon-NS5 Gradient |\n\n"
        "Cada inferencia ajusta los pesos en `PSNRL` y registra la huella en la blockchain."
    )

    EstiloTerminalLucIA.renderizar_respuesta_completa(demo_respuesta, "openrouter/free")

    badge_turno(
        turno=1,
        dt=1420.0,
        mod="openrouter/free",
        der=12.4589,
        neu=50,
        cid="bafkreihq53...",
        txs=1,
    )
from STYLOS_ColoresLucIA import ColoresLucIA  # CLASSPACK
from STYLOS_RenderizadorMarkdown import RenderizadorMarkdown  # CLASSPACK
from STYLOS_EstiloTerminalLucIA import EstiloTerminalLucIA  # CLASSPACK
