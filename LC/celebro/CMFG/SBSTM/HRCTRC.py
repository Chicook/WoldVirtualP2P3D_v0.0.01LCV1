"""
HRCTRC.py - Herrero Constructor de Sesion para LucIA (WoldVirtualP2P3D 2026)
==============================================================================
Permisos via codigo sobre LC: copia de trabajo en LC/Constructor durante la
sesion; al finalizar UNIFICA cambios en rutas reales y deja Constructor vacia.
Ciclo: iniciar -> crear/escribir/leer overlay -> finalizar (unifica + purga).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-HRCTRC-Constructor"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.HRCTRC")

PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
CONSTRUCTOR_DIR: Final[Path] = LC_DIR / "Constructor"
MANIFIESTO_NOMBRE: Final[str] = "manifiesto_sesion.json"

# Rutas del sistema que el constructor puede versionar en el overlay.
RUTAS_VERSIONABLES: Final[Tuple[str, ...]] = (
    "mainLCSTM.py",
    "modelosIAlocal",
    "celebro/CMFG/SBSTM",
    "celebro/CMFG",
    "celebro",
)

_LOCK: Final[threading.Lock] = threading.Lock()


def _sha256(ruta: Path) -> str:
    h = hashlib.sha256()
    try:
        with open(ruta, "rb") as fh:
            for bloque in iter(lambda: fh.read(1 << 20), b""):
                h.update(bloque)
        return h.hexdigest()
    except Exception:
        return ""


def _rel_a_lc(ruta: Path) -> Path:
    try:
        return ruta.resolve().relative_to(LC_DIR.resolve())
    except Exception:
        return Path(ruta.name)


def _dentro_de(base: Path, objetivo: Path) -> bool:
    try:
        objetivo.resolve().relative_to(base.resolve())
        return True
    except Exception:
        return False


_GESTOR_HRCTRC: Optional[GestorConstructorSesion] = None


def get_gestor_hrctrc(sesion_id: str = "") -> GestorConstructorSesion:
    """Singleton del herrero constructor de sesion."""
    global _GESTOR_HRCTRC
    with _LOCK:
        if _GESTOR_HRCTRC is None:
            _GESTOR_HRCTRC = GestorConstructorSesion(sesion_id=sesion_id)
        elif sesion_id and _GESTOR_HRCTRC.sesion_id != sesion_id:
            _GESTOR_HRCTRC = GestorConstructorSesion(sesion_id=sesion_id)
        return _GESTOR_HRCTRC


def iniciar_constructor(sesion_id: str = "") -> Dict[str, Any]:
    """Atajo: abre la copia de trabajo de la sesion."""
    return get_gestor_hrctrc(sesion_id).iniciar_sesion()


def unificar_constructor(aplicar: bool = True) -> Dict[str, Any]:
    """Atajo: unifica cambios en rutas reales y vacia Constructor."""
    return get_gestor_hrctrc().finalizar_sesion(aplicar=aplicar)


def ordenar_constructor_lucia(texto_usuario: str) -> Optional[str]:
    """Atajo: interpreta y ejecuta ordenes de construccion. None si no aplica."""
    return get_gestor_hrctrc().ejecutar_orden(texto_usuario)


def capacidad_hrctrc() -> str:
    """Frase factual de capacidad para inyectar en el contexto de LucIA."""
    return (f"SI puedo crear carpetas y modificar el sistema: modulo HRCTRC v{__version__} "
            f"operativo sobre {LC_DIR} con copia de trabajo en {CONSTRUCTOR_DIR}.")


if __name__ == "__main__":
    g = get_gestor_hrctrc("demo")
    print("=" * 70)
    print(f"  HRCTRC v{__version__} - Herrero constructor de sesion")
    print("=" * 70)
    print(" ", capacidad_hrctrc())
    r = g.iniciar_sesion()
    print(" ", r["mensaje"], f"({r['archivos_versionados']} archivos)")
    print("  Demo escritura:", g.escribir_overlay("demo_herrero.txt", "hola constructor")["mensaje"])
    print("  Pendientes:", g.listar_overlay())
    print(" ", g.finalizar_sesion()["mensaje"])
    print("=" * 70)
from HRCTRC_GestorConstructorSesion import GestorConstructorSesion  # CLASSPACK
