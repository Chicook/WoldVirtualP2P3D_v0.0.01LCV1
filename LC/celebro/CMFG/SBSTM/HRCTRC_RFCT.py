"""
HRCTRC_RFCT.py - Refactorizador de Version de Sesion para LucIA (2026)
=======================================================================
Divide .py oversized del overlay en paquetes <=450 (shim + _pkg) con test
previo obligatorio para la allowlist. Barra de progreso "version de sesion".
"""
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
    """Test previo: carga el _pkg aislado (importlib) y corre la sonda."""
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


class RefactorizadorSesion:
    """Version de sesion: divide oversized en paquetes <=450 con modelo local."""

    def __init__(self, overlay: Path) -> None:
        self.overlay = overlay
        self._resumenes_usados = 0
        self._lock = threading.Lock()

    def _resumen(self, codigo: str) -> str:
        with self._lock:
            if self._resumenes_usados >= MAX_RESUMENES_MODELO:
                return "Parte del subsistema LucIA."
            self._resumenes_usados += 1
        return _resumen_con_modelo_local(codigo)

    def refactorizar_archivo(self, rel: str) -> Dict[str, Any]:
        """ARCH.py -> ARCH_pkg/ (partes <=450) + shim loader compatible."""
        origen = self.overlay / rel
        if origen.name in EXCLUIR_SIEMPRE and origen.name not in PERMITIR_CON_TEST:
            return {"exito": True, "archivo": rel,
                    "mensaje": "Excluido por seguridad (entry-point/sistema vivo).", "partes": 1}
        stem = origen.stem
        try:
            fuente = origen.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"exito": False, "archivo": rel, "mensaje": str(exc)}
        cabecera, bloques = _partir_bloques(fuente)
        cab = "".join(l for l in cabecera.splitlines(keepends=True)
                      if l.strip() != "from __future__ import annotations")
        tope = max(100, MAX_LINEAS - (cab.count("\n") + 8))  # cabecera + plantilla
        partes = _empaquetar(bloques, limite=tope)
        if len(partes) <= 1 and contar_lineas(origen) <= MAX_LINEAS:
            return {"exito": True, "archivo": rel, "mensaje": "En regla, sin cambios.", "partes": 1}
        pkg = origen.parent / f"{stem}_pkg"
        pkg.mkdir(parents=True, exist_ok=True)
        # Respaldo en CHG/ (no viaja al sistema; INTEGRACIONRF lo documenta).
        try:
            CHG_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origen, CHG_DIR / f"{stem}.py.bak_sesion")
        except Exception:
            pass
        nombres: List[str] = []
        cab = "".join(l for l in cabecera.splitlines(keepends=True)
                      if l.strip() != "from __future__ import annotations")
        for i, grupo in enumerate(partes, 1):
            cuerpo = "".join(grupo) if isinstance(grupo, list) else "".join(grupo)
            resumen = self._resumen(cuerpo)
            contenido = (f'"""\n{stem} - parte {i}/{len(partes)} (version de sesion LucIA).\n'
                         f"{resumen}\n\"\"\"\nfrom __future__ import annotations\n\n"
                         f"{cab}\n{cuerpo}")
            nombre = f"_p{i}.py"
            (pkg / nombre).write_text(contenido, encoding="utf-8")
            nombres.append(nombre)
        inits = "".join(f"from .{n[:-3]} import *  # noqa\n" for n in nombres)
        (pkg / "__init__.py").write_text(
            f'"""\n{stem}_pkg - paquete de sesion (regla 400/450). Re-exporta todo.\n"""\n'
            f"from __future__ import annotations\n\n{inits}\n__all__ = []\n",
            encoding="utf-8")
        test_ok: Optional[bool] = None
        if f"{stem}.py" in PERMITIR_CON_TEST:
            sonda = "estado_psnrl" if stem == "PURGADOR" else ""
            test_ok = test_previo_paquete(pkg / "__init__.py", sonda)
            if not test_ok:  # revierte: el lote no viaja a la arquitectura
                try:
                    origen.write_text(fuente, encoding="utf-8")
                    shutil.rmtree(pkg, ignore_errors=True)
                except Exception:
                    pass
                return {"exito": False, "archivo": rel, "test_ok": False,
                        "mensaje": "Test previo fallo; lote revertido, original intacto."}
        shim = (f'"""\n{stem}.py - SHIM de sesion (regla 400/450). Codigo en {stem}_pkg/.\n'
                f"Generado por HRCTRC_RFCT v{__version__}; compatible 100%.\n\"\"\"\n"
                f"from __future__ import annotations\n\n"
                f"import importlib.util as _ilu\n"
                f"from pathlib import Path as _Path\n"
                f"_pkgdir = _Path(__file__).resolve().parent / \"{stem}_pkg\"\n"
                f"_spec = _ilu.spec_from_file_location(\n"
                f"    \"{stem}_pkg\", _pkgdir / \"__init__.py\",\n"
                f"    submodule_search_locations=[str(_pkgdir)])\n"
                f"_mod = _ilu.module_from_spec(_spec)\n"
                f"import sys as _sys\n"
                f"_sys.modules[\"{stem}_pkg\"] = _mod\n"
                f"_spec.loader.exec_module(_mod)  # noqa\n"
                f"globals().update({{k: v for k, v in vars(_mod).items() if not k.startswith('__')}})\n"
                f"try:\n    __all__ = list(getattr(_mod, '__all__', []))\n"
                f"except Exception:\n    __all__ = []\n")
        origen.write_text(shim, encoding="utf-8")
        with self._lock:
            try:
                man_path = self.overlay / "manifiesto_sesion.json"
                man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
                man.setdefault("refactors", {})[rel] = {
                    "paquete": f"{stem}_pkg/", "partes": nombres,
                    "lineas_antes": fuente.count("\n") + 1,
                    "test_ok": bool(test_ok),
                    "ts": time.time()}
                man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return {"exito": True, "archivo": rel, "mensaje": f"Dividido en {len(nombres)} partes.",
                "partes": len(nombres), "paquete": f"{stem}_pkg/"}

    def ejecutar(self, mostrar_barra: bool = True) -> Dict[str, Any]:
        """Refactoriza todo el overlay con barra 'version de sesion'. Usa modelo local."""
        objetivos = escanear_oversized(self.overlay)
        total = len(objetivos)
        if mostrar_barra:
            _escribir_barra(barra_progreso(0, max(1, total), "version de sesion: analizando") + "\n")
        hechos, fallos = 0, 0
        detalle: List[Dict[str, Any]] = []
        for i, obj in enumerate(objetivos, 1):
            if mostrar_barra:
                _escribir_barra(barra_progreso(i - 1, total, f"refactorizando {obj['rel']}"))
            rep = self.refactorizar_archivo(obj["rel"])
            detalle.append(rep)
            if rep.get("exito"):
                hechos += 1
            else:
                fallos += 1
            if mostrar_barra:
                _escribir_barra(barra_progreso(i, total, f"version de sesion: {obj['rel']}"))
        if mostrar_barra:
            _escribir_barra("\n")
        return {"exito": fallos == 0, "analizados": total, "refactorizados": hechos,
                "fallos": fallos, "detalle": detalle,
                "mensaje": (f"Version de sesion lista: {hechos} archivo(s) refactorizado(s) "
                            f"a regla 400/450 con modelo local."
                            if total else "Version de sesion: todo el codigo ya cumple 400/450.")}

    def es_orden_refactor(self, texto: str) -> bool:
        t = texto.lower()
        return any(k in t for k in ("refactoriz", "version de ses", "versión de ses",
                                    "regla 400", "regla de oro", "divide el codigo",
                                    "divide el código"))

    def ejecutar_orden(self, texto: str) -> Optional[str]:
        if not self.es_orden_refactor(texto):
            return None
        rep = self.ejecutar(mostrar_barra=False)
        return ("Claro que si, ya lo hago yo misma con mi modelo local. " + rep["mensaje"])

    # ── Restauracion y verificacion ───────────────────────────
    def restaurar_archivo(self, rel: str) -> Dict[str, Any]:
        """Deshace el refactor: recupera el .bak_sesion de CHG/ y borra el paquete."""
        origen = self.overlay / rel
        stem = origen.stem
        bak = CHG_DIR / f"{stem}.py.bak_sesion"
        try:
            if not bak.is_file():
                return {"exito": False, "archivo": rel, "mensaje": "Sin respaldo de sesion."}
            shutil.copy2(bak, origen)
            shutil.rmtree(origen.parent / f"{stem}_pkg", ignore_errors=True)
            bak.unlink(missing_ok=True)
            with self._lock:
                try:
                    man_path = self.overlay / "manifiesto_sesion.json"
                    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
                    man.get("refactors", {}).pop(rel, None)
                    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            return {"exito": True, "archivo": rel, "mensaje": f"{rel} restaurado al original."}
        except Exception as exc:
            return {"exito": False, "archivo": rel, "mensaje": str(exc)}

    def verificar_paquetes(self) -> List[Dict[str, Any]]:
        """Comprueba que cada parte _pkg cumpla <=MAX_LINEAS (auditoria de regla)."""
        reporte: List[Dict[str, Any]] = []
        if not self.overlay.exists():
            return reporte
        for pkg in sorted(self.overlay.rglob("*_pkg")):
            if not pkg.is_dir():
                continue
            for f in sorted(pkg.glob("*.py")):
                n = contar_lineas(f)
                reporte.append({"archivo": str(f.relative_to(self.overlay)), "lineas": n,
                                "cumple": n <= MAX_LINEAS})
        return reporte

    def estado_version(self) -> Dict[str, Any]:
        """Foto completa: oversized restantes, paquetes y cumplimiento."""
        overs = escanear_oversized(self.overlay)
        paqs = self.verificar_paquetes()
        return {"overlay": str(self.overlay), "oversized_restantes": overs, "paquetes": paqs,
                "todo_en_regla": not overs and all(p["cumple"] for p in paqs),
                "resumenes_modelo_usados": self._resumenes_usados}

    def informe_para_lucia(self) -> str:
        """Texto factual: que versiono, con que modelo y que cumple la regla."""
        est = self.estado_version()
        n_over = len(est["oversized_restantes"])
        n_paq = len(est["paquetes"])
        base = (f"Soy LucIA y SI trabajo tu codigo directamente (HRCTRC_RFCT v{__version__}). "
                f"Version de sesion en {self.overlay}: {n_paq} parte(s) en regla 400/450. ")
        if n_over:
            noms = ", ".join(f"{o['rel']} ({o['lineas']})" for o in est["oversized_restantes"][:5])
            return base + f"Pendientes de dividir: {noms}. Dime 'refactoriza' y lo divido ahora."
        return base + "Todo el codigo cumple la regla de oro 400/450."

    def ayuda(self) -> str:
        return ("Refactorizo con modelo local: 'refactoriza la version de sesion', "
                "'verifica la regla 400/450', 'restaura el archivo X'.")

    # ── Planificacion y reportes ──────────────────────────────
    def plan_version(self) -> Dict[str, Any]:
        """Plan en seco: que se dividiria, en cuantas partes y coste aprox."""
        objetivos = escanear_oversized(self.overlay)
        plan: List[Dict[str, Any]] = []
        for obj in objetivos:
            try:
                fuente = (self.overlay / obj["rel"]).read_text(encoding="utf-8", errors="replace")
                _, bloques = _partir_bloques(fuente)
                partes = _empaquetar(bloques)
                plan.append({"archivo": obj["rel"], "lineas": obj["lineas"],
                             "partes_previstas": len(partes),
                             "resumenes_modelo": min(len(partes), MAX_RESUMENES_MODELO)})
            except Exception as exc:
                plan.append({"archivo": obj["rel"], "error": str(exc)})
        return {"objetivos": len(plan), "plan": plan,
                "mensaje": (f"Version de sesion: {len(plan)} archivo(s) por dividir."
                            if plan else "Nada que dividir: todo en regla 400/450.")}

    def exportar_reporte(self, destino: str = "") -> Dict[str, Any]:
        """Guarda el estado de la version de sesion en JSON para auditoria."""
        est = self.estado_version()
        est["plan"] = self.plan_version()
        ruta = Path(destino) if destino else self.overlay / "reporte_version_sesion.json"
        try:
            ruta.write_text(json.dumps(est, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"exito": True, "mensaje": f"Reporte de version en {ruta}."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"No pude exportar: {exc}."}

    def limpiar_respaldos(self) -> int:
        """Borra los .bak_sesion de CHG/ tras una unificacion exitosa."""
        try:
            baks = list(CHG_DIR.glob("*.bak_sesion"))
            for bak in baks:
                bak.unlink(missing_ok=True)
            return len(baks)
        except Exception:
            return 0


def get_refactorizador(overlay: Path) -> RefactorizadorSesion:
    return RefactorizadorSesion(overlay)


def refactorizar_overlay(overlay: Path, mostrar_barra: bool = True) -> Dict[str, Any]:
    return RefactorizadorSesion(overlay).ejecutar(mostrar_barra=mostrar_barra)


if __name__ == "__main__":
    print("=" * 70)
    print(f"  HRCTRC_RFCT v{__version__} - Version de sesion (regla 400/450)")
    print("=" * 70)
