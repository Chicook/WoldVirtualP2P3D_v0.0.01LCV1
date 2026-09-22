"""
INTEGRACIONRF - parte 2/2 (version de sesion LucIA).
Parte del subsistema LucIA.
"""
from __future__ import annotations

"""
INTEGRACIONRF.py - Integrador del Refactor de Sesion para LucIA (2026)
=======================================================================
Lista todas las rutas originales de LC, tras el refactor y al cerrar sesion
sobreescribe cada archivo con el codigo nuevo (nueva distribucion de
subsistemas), transcribe la actividad de ejecucion a .md, convierte el .md
a pesos neuronales (PSNRCV->PSNRL), los sube a IPFS y actualiza la rama
devopencode del repositorio. Prioridad: rapidez y eficiencia.

Pipeline de cierre (cierre_completo):
  1. aplicar_refactor()   -> HRCTRC.finalizar_sesion() (sobreescribe archivos)
  2. generar_md()         -> bitacora INTEGRACIONRF_<sesion>.md en SBSTM
  3. convertir_a_pesos()  -> PSNRCV procesa el .md y persiste npz/js en PSNRL
  4. subir_a_ipfs()       -> IPFSManager.almacenar_pesos(npz)
  5. actualizar_rama()    -> git add/commit/push origin devopencode
"""

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-INTEGRACIONRF-Cierre"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.INTEGRACIONRF")

PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
ROOT_DIR: Final[Path] = LC_DIR.parent.resolve()
CHG_DIR: Final[Path] = ROOT_DIR / "CHG"

RAMA_OBJETIVO: Final[str] = "devopencode"

def get_integrador(sesion_id: str = "") -> IntegradorRefactor:
    """Singleton del integrador de cierre de sesion."""
    global _INTEGRADOR
    with _LOCK:
        if _INTEGRADOR is None:
            _INTEGRADOR = IntegradorRefactor(sesion_id=sesion_id)
        elif sesion_id and _INTEGRADOR.sesion_id != sesion_id:
            _INTEGRADOR = IntegradorRefactor(sesion_id=sesion_id)
        return _INTEGRADOR


def cierre_integracion() -> Dict[str, Any]:
    """Atajo: pipeline completo de cierre en una llamada."""
    return get_integrador().cierre_completo()


def registrar_actividad(tipo: str, detalle: str) -> None:
    """Atajo: anota un evento en la bitacora viva sin instanciar nada."""
    try:
        get_integrador().registrar(tipo, detalle)
    except Exception:
        pass


if __name__ == "__main__":
    print("=" * 70)
    print(f"  INTEGRACIONRF v{__version__} - Integrador de cierre")
    print("=" * 70)
    rutas = listar_rutas_originales()
    print(f"  Archivos en LC: {len(rutas)}")
    g = get_integrador("demo")
    print(" ", g.informe_para_lucia()[:110])
    print("=" * 70)
