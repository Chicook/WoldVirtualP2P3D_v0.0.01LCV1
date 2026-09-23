"""
HRCNTR.py - Monitor y Refactorizador Neural del Sistema (WoldVirtualP2P3D 2026)
================================================================================
Monitorea TODOS los modulos y neuronas, selecciona clases completas,
genera subsistemas importados con modelos free de OpenRouter, expande
clases a 400/450 lineas, y actualiza la raiz del sistema al cerrar.
Ciclo: monitor -> seleccionar -> generar -> version_sesion -> actualizar.
"""
from __future__ import annotations

import ast
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.HRCNTR")

PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
ROOT_DIR: Final[Path] = LC_DIR.parent.resolve()
CHG_DIR: Final[Path] = ROOT_DIR / "CHG"

MIN_LINEAS: Final[int] = 400
MAX_LINEAS: Final[int] = 450

MODULOS_VIVOS: Final[Tuple[str, ...]] = (
    "mainLCSTM.py", "__init__.py", "HRCTRC.py", "HRCTRC_RFCT.py",
    "HRCNTR.py", "BASELUC.py", "INTEGRACIONRF.py", "SNSBSTNPRB.py",
    "BKSVCB.py", "ipfs_manager.py", "PSNRCV.py", "pesos_vivos.py",
    "PURGADOR.py", "STMRFCR.py", "DSIALCLGRG.py", "IAFREE.py",
)

EXCLUIR_DIRS: Final[Tuple[str, ...]] = (
    "__pycache__", ".ipfs", "node_modules", "pycache", ".git",
    "CHG", "node_modules", "Constructor", "ruff_cache",
)

_BARRA_MAX: Final[int] = 40
_LOCK: Final[threading.Lock] = threading.Lock()

_PAT_CLASE = re.compile(r"^class\s+(\w+)")


def _contar_lineas(ruta: Path) -> int:
    try:
        with open(ruta, "r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def barra_progreso(actual: int, total: int, etiqueta: str = "",
                   ancho: int = _BARRA_MAX) -> str:
    total = max(1, total)
    lleno = int(ancho * min(1.0, actual / total))
    barra = "\u2588" * lleno + "\u2591" * (ancho - lleno)
    return f"[{barra}] {100.0*actual/total:5.1f}% ({actual}/{total}) {etiqueta}"


def _escribir_barra(texto: str) -> None:
    try:
        sys.stdout.write("\r  " + texto)
        sys.stdout.flush()
    except Exception:
        pass


_GESTOR_HRCNTR: Optional[GestorHRCNTR] = None


def get_gestor_hrctnr() -> GestorHRCNTR:
    global _GESTOR_HRCNTR
    with _LOCK:
        if _GESTOR_HRCNTR is None:
            _GESTOR_HRCNTR = GestorHRCNTR()
        return _GESTOR_HRCNTR


def ejecutar_hrctnr(overlay: str = "", mostrar_barra: bool = True) -> Dict[str, Any]:
    return get_gestor_hrctnr().ejecutar_refactor(overlay=overlay, mostrar_barra=mostrar_barra)


def estado_hrctnr() -> Dict[str, Any]:
    return get_gestor_hrctnr().estado()


def actualizar_sistema_hrctnr(confirmar: bool = False) -> Dict[str, Any]:
    return get_gestor_hrctnr().actualizar_sistema(confirmar=confirmar)


def ciclo_cierre_hrctnr() -> Dict[str, Any]:
    return get_gestor_hrctnr().ciclo_cierre()


def confirmar_actualizacion_hrctnr() -> str:
    return get_gestor_hrctnr().confirmar_actualizacion()


if __name__ == "__main__":
    g = get_gestor_hrctnr()
    print("=" * 70)
    print(f"  HRCNTR v{__version__} - Monitor y Refactorizador Neural")
    print("=" * 70)
    mon = g.monitor_sistema()
    print(f"  Archivos: {mon['total_archivos']} | Neuronas: {mon['total_neuronas']}"
          f" | Oversized: {mon['oversized']}")
    sel = g.seleccionar_clases()
    print(f"  Clases seleccionadas: {len(sel)}")
    for s in sel[:5]:
        print(f"    - {s['clase']} en {s['modulo_origen']} ({s['lineas_modulo']} lin)")
    print("=" * 70)
from HRCNTR_GestorHRCNTR import GestorHRCNTR  # CLASSPACK
