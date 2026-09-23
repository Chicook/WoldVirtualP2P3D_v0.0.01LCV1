"""CLASSPACK.py - Extrae clases grandes en modulos separados."""

import importlib.util
import re
import sys
from pathlib import Path
from typing import Any, Dict, Final, List, Tuple

__version__: str = "2026.3.1"
MIN_CLASE: int = 30
EXCLUIR: Final[Tuple[str, ...]] = (
    "mainLCSTM.py", "__init__.py", "HRCTRC.py", "HRCTRC_RFCT.py",
    "HRCNTR.py", "DSIALCLGRG.py", "IAFREE.py", "BASELUC.py",
    "PRTLUC.py", "CMDLUC.py", "STMRFCR.py", "PURGADOR.py",
    "ipfs_manager.py", "neural_math.py", "pesos_vivos.py", "PSNRCV.py",
)


def escanear_clases(overlay: Path, min_body: int = MIN_CLASE) -> List[Dict[str, Any]]:
    if not overlay.exists():
        return []
    resultados: List[Dict[str, Any]] = []
    for f in sorted(overlay.rglob("*.py")):
        if "__pycache__" in f.parts or "_pkg" in f.name:
            continue
        if f.name in EXCLUIR:
            continue
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if _es_modulo_clase(source):
            continue
        for nombre, inicio, fin, nlineas in _clases_body(source):
            if nlineas >= min_body:
                resultados.append({"archivo": str(f.relative_to(overlay)),
                                   "clase": nombre, "lineas": nlineas,
                                   "inicio": inicio, "fin": fin})
    return resultados


def _es_modulo_clase(source: str) -> bool:
    has_import = any(l.strip().startswith(("import ", "from ", "__future__")) for l in source.splitlines())
    has_def = any(l.strip().startswith("def ") for l in source.splitlines())
    has_assign = any(re.match(r"^[A-Z_]", l.strip()) for l in source.splitlines() if l.strip())
    has_class = any(l.strip().startswith("class ") for l in source.splitlines())
    return has_class and not has_import and not has_def and not has_assign


def _clases_body(source: str) -> List[Tuple[str, int, int, int]]:
    lineas = source.splitlines()
    resultados: List[Tuple[str, int, int, int]] = []
    i = 0
    while i < len(lineas):
        m = re.match(r"^class\s+(\w+)", lineas[i])
        if m:
            nombre = m.group(1)
            inicio = i
            # Incluir decoradores (@...) inmediatamente superiores a la clase.
            j = inicio - 1
            while j >= 0 and lineas[j].strip().startswith("@"):
                inicio = j
                j -= 1
            indent = len(lineas[i]) - len(lineas[i].lstrip())
            i += 1
            while i < len(lineas):
                ln = lineas[i]
                if ln.strip() and not ln.strip().startswith("#"):
                    ci = len(ln) - len(ln.lstrip())
                    if ci <= indent:
                        break
                i += 1
            resultados.append((nombre, inicio, i, i - inicio))
        else:
            i += 1
    return resultados


def extraer_clase(archivo: Path, clase: str, overlay: Path,
                   min_body: int = MIN_CLASE) -> Dict[str, Any]:
    try:
        source = archivo.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {"exito": False, "error": str(exc)}
    cuerpos = _clases_body(source)
    objetivo = None
    for nombre, inicio, fin, nlineas in cuerpos:
        if nombre == clase and nlineas >= min_body:
            objetivo = (inicio, fin)
            break
    if objetivo is None:
        return {"exito": False, "error": f"Clase {clase} no encontrada o pequena."}
    inicio, fin = objetivo
    lineas_src = source.splitlines(keepends=True)
    cuerpo = "".join(lineas_src[inicio:fin])
    restante = "".join(lineas_src[:inicio]) + "".join(lineas_src[fin:])
    stem = archivo.stem
    destino = archivo.parent / f"{stem}_{clase}.py"
    if destino.exists():
        return {"exito": False, "error": f"{destino.name} ya existe."}
    destino.write_text(cuerpo, encoding="utf-8")
    partes = archivo.parts
    if "LC" in partes:
        pkg = ".".join(partes[partes.index("LC"):-1])
        linea_import = f"\nfrom {pkg}.{stem}_{clase} import {clase}  # CLASSPACK\n"
    else:
        linea_import = f"\nfrom {stem}_{clase} import {clase}  # CLASSPACK\n"
    nuevo = restante.rstrip("\n") + linea_import
    bak = archivo.with_suffix(".py.bak_clase")
    bak.write_text(source, encoding="utf-8")
    try:
        archivo.write_text(nuevo, encoding="utf-8")
        _verify_imports(archivo, destino)
    except Exception as exc:
        archivo.write_text(source, encoding="utf-8")
        destino.unlink(missing_ok=True)
        bak.unlink(missing_ok=True)
        return {"exito": False, "error": f"Test import fallo: {exc}"}
    return {"exito": True, "clase": clase, "modulo": str(destino),
            "lineas_clase": fin - inicio}


def _verify_imports(archivo: Path, destino: Path) -> None:
    padre = str(archivo.parent)
    agregado = False
    if padre not in sys.path:
        sys.path.insert(0, padre)
        agregado = True
    backup = sys.modules.copy()
    try:
        spec = importlib.util.spec_from_file_location(archivo.stem + "_verify", archivo)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[archivo.stem + "_verify"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.modules.clear()
        sys.modules.update(backup)
        if agregado:
            sys.path.remove(padre)
    _try_load(destino)


def _try_load(ruta: Path) -> None:
    padre = str(ruta.parent)
    agregado = False
    if padre not in sys.path:
        sys.path.insert(0, padre)
        agregado = True
    try:
        spec = importlib.util.spec_from_file_location(ruta.stem + "_load", ruta)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[ruta.stem + "_load"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(ruta.stem + "_load", None)
        if agregado:
            sys.path.remove(padre)


def refactorizar_clases(overlay: Path, min_body: int = MIN_CLASE) -> Dict[str, Any]:
    clases = escanear_clases(overlay, min_body)
    extraidas: List[Dict[str, Any]] = []
    fallos: List[Dict[str, Any]] = []
    for c in clases:
        archivo = overlay / c["archivo"]
        rep = extraer_clase(archivo, c["clase"], overlay, min_body)
        if rep.get("exito"):
            extraidas.append(rep)
        else:
            fallos.append({"archivo": c["archivo"], "clase": c["clase"],
                           "error": rep.get("error")})
    return {"exito": not fallos, "extraidas": extraidas, "fallos": fallos,
            "total": len(clases)}
