"""HRCTRC_RFCT.py - Refactorizador Version de Sesion (2026).
Divide .py oversized del overlay en paquetes <=450 (shim + _pkg) con test previo.
Tambien divide clases grandes en modulos separados. Barra "version de sesion"."""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple
from LC.celebro.CMFG.SBSTM.CLASSPACK import refactorizar_clases as _refactor_clases

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-HRCTRC-RFCT-Refactor"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.HRCTRC_RFCT")

PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
CONSTRUCTOR_DIR: Final[Path] = LC_DIR / "Constructor"
CHG_DIR: Final[Path] = CELEBRO_DIR.parent.parent / "CHG"

MIN_LINEAS: Final[int] = 400
MAX_LINEAS: Final[int] = 450
MAX_RESUMENES_MODELO: Final[int] = 3
TIMEOUT_MODELO: Final[float] = 25.0

# Archivos que JAMAS se dividen: punto de entrada en ejecucion, paquetes,
# el propio herrero y gestores vivos (un shim aqui rompe el arranque).
EXCLUIR_SIEMPRE: Final[Tuple[str, ...]] = (
    "mainLCSTM.py",
    "__init__.py",
    "HRCTRC.py",
    "HRCTRC_RFCT.py",
    "DSIALCLGRG.py",
    "IAFREE.py",
)

# Allowlist: solo se unifican si el test previo del paquete pasa en el overlay.
PERMITIR_CON_TEST: Final[Tuple[str, ...]] = ("PURGADOR.py",)

_PAT_CLASE_DEF: Final[re.Pattern] = re.compile(r"^(class |def |[A-Z][A-Z0-9_]*\s*[:=])")

_LOCK: Final[threading.Lock] = threading.Lock()

def barra_progreso(actual: int, total: int, etiqueta: str = "", ancho: int = 34) -> str:
    """Renderiza una barra █/░ para la version de sesion en terminal."""
    total = max(1, total)
    lleno = int(ancho * min(1.0, actual / total))
    barra = "█" * lleno + "░" * (ancho - lleno)
    pct = 100.0 * actual / total
    return f"\r  [{barra}] {pct:5.1f}% ({actual}/{total}) {etiqueta}"


def _escribir_barra(texto: str) -> None:
    try:
        sys.stdout.write(texto)
        sys.stdout.flush()
    except Exception:
        pass


def contar_lineas(ruta: Path) -> int:
    try:
        with open(ruta, "r", encoding="utf-8", errors="replace") as fh:
            return sum(1 for _ in fh)
    except Exception:
        return 0


def escanear_oversized(overlay: Path) -> List[Dict[str, Any]]:
    """Lista los .py del overlay que superan MAX_LINEAS (violan la regla)."""
    hallados: List[Dict[str, Any]] = []
    if not overlay.exists():
        return hallados
    for f in sorted(overlay.rglob("*.py")):
        if "__pycache__" in f.parts or "_pkg" in f.name:
            continue
        if f.name in EXCLUIR_SIEMPRE and f.name not in PERMITIR_CON_TEST:
            continue
        n = contar_lineas(f)
        if n > MAX_LINEAS:
            hallados.append({"rel": str(f.relative_to(overlay)), "lineas": n,
                             "exceso": n - MAX_LINEAS})
    return sorted(hallados, key=lambda d: d["lineas"], reverse=True)


def _partir_bloques(fuente: str) -> Tuple[str, List[str]]:
    """Separa cabecera (imports/docstring) de bloques top-level (class/def)."""
    lineas = fuente.splitlines(keepends=True)
    cabecera_fin = 0
    for i, ln in enumerate(lineas):
        s = ln.strip()
        if s.startswith(("import ", "from ")) or s.startswith('"""') or s.startswith("'''"):
            cabecera_fin = i + 1
            continue
        if s.startswith("#") or not s:
            cabecera_fin = i + 1
            continue
        if _PAT_CLASE_DEF.match(ln):
            break
        cabecera_fin = i + 1
    cabecera = "".join(lineas[:cabecera_fin])
    bloques: List[str] = []
    actual: List[str] = []
    for ln in lineas[cabecera_fin:]:
        if _PAT_CLASE_DEF.match(ln) and actual:
            bloques.append("".join(actual))
            actual = [ln]
        else:
            actual.append(ln)
    if actual:
        bloques.append("".join(actual))
    bloques = [b for b in bloques if b.strip()]
    # Constantes de modulo (MAYUS = ...) pegadas a la cabecera: si se dividen,
    # las funciones de otras partes fallan con NameError en el test previo.
    _pat_const = re.compile(r"^(?:import |from |[A-Z_][A-Z0-9_]*\s*[:=]|#|\s*$)")
    while bloques and all(_pat_const.match(l) for l in bloques[0].splitlines()):
        cabecera += bloques.pop(0)
    return cabecera, bloques


def _empaquetar(bloques: List[str], limite: int = MAX_LINEAS) -> List[List[str]]:
    """Agrupa bloques en partes que no superen el limite de lineas."""
    partes: List[List[str]] = []
    actual: List[str] = []
    n_actual = 0
    for b in bloques:
        n_b = b.count("\n") + 1
        if n_b > limite:  # bloque gigante: corte duro por lineas
            if actual:
                partes.append(actual)
                actual, n_actual = [], 0
            ln = b.splitlines(keepends=True)
            for i in range(0, len(ln), limite):
                partes.append(ln[i:i + limite])
            continue
        if n_actual + n_b > limite and actual:
            partes.append(actual)
            actual, n_actual = [], 0
        actual.append(b)
        n_actual += n_b
    if actual:
        partes.append(actual)
    return partes


def _resumen_con_modelo_local(codigo: str) -> str:
    """Pide al modelo local (Ollama/MDSTM) 1 linea descriptiva; fallback estatico."""
    muestra = codigo[:1500].replace("\n", " ")
    try:
        from LC.celebro.CMFG.SBSTM.DSIALCLGRG import consultar_lucia_local
        texto, mid, _ = consultar_lucia_local(
            f"Describe en UNA linea (max 12 palabras) que hace este codigo Python: {muestra}",
            None, timeout=TIMEOUT_MODELO)
        if texto and mid != "local:reflejo":
            limpio = " ".join(texto.split())[:140]
            return limpio or "Parte del subsistema LucIA."
    except Exception as exc:
        logger.debug("RFCT modelo local no disponible: %s", exc)
    m = re.search(r"^(?:class|def)\s+(\w+)", codigo, re.MULTILINE)
    return f"Parte del subsistema LucIA ({m.group(1) if m else 'bloque'})."


def test_previo_paquete(pkg_init: Path, sonda: str = "") -> bool:
    import sys as _sys
    try:
        import importlib.util
        esp = importlib.util.spec_from_file_location(
            "rfct_test_pkg", pkg_init, submodule_search_locations=[str(pkg_init.parent)])
        mod = importlib.util.module_from_spec(esp)
        _sys.modules["rfct_test_pkg"] = mod  # relativo `from ._pX` lo exige
        try:
            esp.loader.exec_module(mod)  # type: ignore
            if sonda:
                getattr(mod, sonda)()
            return True
        finally:
            _sys.modules.pop("rfct_test_pkg", None)
    except Exception as exc:
        logger.warning("RFCT test previo fallo: %s", exc)
        return False


def refactorizar_overlay(overlay: Path, mostrar_barra: bool = True) -> Dict[str, Any]:
    return RefactorizadorSesion(overlay).ejecutar(mostrar_barra=mostrar_barra)


if __name__ == "__main__":
    print("=" * 70)
    print(f"  HRCTRC_RFCT v{__version__} - Version de sesion (regla 400/450)")
    print("=" * 70)
from HRCTRC_RFCT_RefactorizadorSesion import RefactorizadorSesion  # CLASSPACK
