"""
STMRFCR.py - Auditor Estructural del Cierre para LucIA (2026)
==============================================================
Solucion a los problemas estructurales del cierre: antes de unificar,
certifica cada lote shim+_pkg del overlay (compila partes, carga el paquete
aislado, ejecuta sonda, verifica constantes UPPER definidas). Si un lote
falla, lo revierte al original para que nunca rompa la arquitectura viva.

Chequeos por lote (rapidos, sin red):
  1. Sintaxis: cada parte compila (compile, sin ejecutar).
  2. Carga: el _pkg se importa aislado via importlib (+sys.modules).
  3. Sonda: ejecuta una funcion de humo si existe (estado_*/verificar_*).
  4. Constantes: nombres UPPER usados estan definidos (AST, sin ejecutar).
  5. Regla: cada parte <= 450 lineas.
El certificado se registra en el manifiesto para el prefiltro HRCTRC.
"""
from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Set, Tuple

__version__: Final[str] = "2026.3.1"
__subsystem__: Final[str] = "LucIA-STMRFCR-Auditor"
__author__: Final[str] = "Equipo WoldVirtualP2P3D"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.SBSTM.STMRFCR")

PACKAGE_ROOT: Final[Path] = Path(__file__).parent.resolve()
CMFG_DIR: Final[Path] = PACKAGE_ROOT.parent.resolve()
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent.resolve()
LC_DIR: Final[Path] = CELEBRO_DIR.parent.resolve()
CONSTRUCTOR_DIR: Final[Path] = LC_DIR / "Constructor"

MAX_LINEAS: Final[int] = 450
_PAT_UPPER: Final[re.Pattern] = re.compile(r"^[A-Z_][A-Z0-9_]*$")

_LOCK: Final[threading.Lock] = threading.Lock()


class _Nombres(ast.NodeVisitor):
    """Recolecta definidos (Store/clase/def/import) y usados (Load)."""

    def __init__(self) -> None:
        self.definidos: Set[str] = set()
        self.usados: Set[str] = set()

    def visit_Name(self, nodo: ast.Name) -> None:
        (self.definidos if isinstance(nodo.ctx, ast.Store) else self.usados).add(nodo.id)

    def visit_FunctionDef(self, nodo: ast.FunctionDef) -> None:
        self.definidos.add(nodo.name)
        for a in list(nodo.args.posonlyargs) + list(nodo.args.args) + list(nodo.args.kwonlyargs):
            self.definidos.add(a.arg)
        if nodo.args.vararg:
            self.definidos.add(nodo.args.vararg.arg)
        if nodo.args.kwarg:
            self.definidos.add(nodo.args.kwarg.arg)
        self.generic_visit(nodo)

    def visit_AsyncFunctionDef(self, nodo: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(nodo)  # type: ignore

    def visit_ClassDef(self, nodo: ast.ClassDef) -> None:
        self.definidos.add(nodo.name)
        self.generic_visit(nodo)

    def visit_Import(self, nodo: ast.Import) -> None:
        for a in nodo.names:
            self.definidos.add((a.asname or a.name).split(".")[0])
        self.generic_visit(nodo)

    def visit_ImportFrom(self, nodo: ast.ImportFrom) -> None:
        for a in nodo.names:
            self.definidos.add(a.asname or a.name)
        self.generic_visit(nodo)


def _nombres_modulo(fuente: str) -> Tuple[Set[str], Set[str], str]:
    """(definidos, usados, error). Vacio si hay SyntaxError."""
    try:
        reco = _Nombres()
        reco.visit(ast.parse(fuente))
        return reco.definidos, reco.usados, ""
    except SyntaxError as exc:
        return set(), set(), f"Sintaxis: {exc}"


def chequear_constantes(partes: List[Path]) -> List[str]:
    """Nombres UPPER usados en el lote pero no definidos en ninguna parte."""
    defs: Set[str] = set()
    usos: Set[str] = set()
    for p in partes:
        try:
            d, u, _ = _nombres_modulo(p.read_text(encoding="utf-8", errors="replace"))
            defs |= d
            usos |= u
        except Exception:
            pass
    ignorar = {"True", "False", "None"}
    return sorted(n for n in usos if _PAT_UPPER.match(n) and n not in defs and n not in ignorar)


def cargar_aislado(pkg_init: Path, sonda: str = "") -> Tuple[bool, str]:
    """Importa el _pkg aislado y corre la sonda. No toca sys.path global."""
    try:
        import importlib.util
        esp = importlib.util.spec_from_file_location(
            "stmrfcr_audit_pkg", pkg_init, submodule_search_locations=[str(pkg_init.parent)])
        mod = importlib.util.module_from_spec(esp)
        sys.modules["stmrfcr_audit_pkg"] = mod
        try:
            esp.loader.exec_module(mod)  # type: ignore
            if sonda:
                getattr(mod, sonda)()
            return True, ""
        finally:
            sys.modules.pop("stmrfcr_audit_pkg", None)
    except Exception as exc:
        return False, str(exc)[:220]


def detectar_sonda(pkg_init: Path) -> str:
    """Elige funcion de humo: prefiere estado_*/verificar_* definidas."""
    try:
        arbol = ast.parse(pkg_init.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return ""
    defs: Set[str] = set()
    for n in ast.walk(arbol):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.add(n.name)
    for cand in sorted(defs):
        if cand.startswith(("estado_", "verificar_")):
            return cand
    return ""


def auditar_lote(pkg_dir: Path) -> Dict[str, Any]:
    """Certifica un _pkg: sintaxis, regla, constantes, carga y sonda."""
    t0 = time.perf_counter()
    partes = sorted(pkg_dir.glob("_p*.py"))
    fallos: List[str] = []
    if not partes:
        return {"exito": False, "paquete": pkg_dir.name, "fallos": ["Sin partes _p*.py"]}
    for p in partes:
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            compile(src, str(p), "exec")
        except SyntaxError as exc:
            fallos.append(f"{p.name}: sintaxis ({exc})")
            continue
        n = src.count("\n") + 1
        if n > MAX_LINEAS:
            fallos.append(f"{p.name}: {n} lineas excede 450")
    if not fallos:
        faltan = chequear_constantes(partes)
        fallos += [f"constante sin definir: {n}" for n in faltan]
    sonda, carga_msg = "", ""
    if not fallos:
        init = pkg_dir / "__init__.py"
        sonda = detectar_sonda(init)
        ok, carga_msg = cargar_aislado(init, sonda)
        if not ok:
            fallos.append(f"carga/sonda: {carga_msg}")
    return {"exito": not fallos, "paquete": pkg_dir.name,
            "partes": len(partes), "sonda": sonda,
            "fallos": fallos, "ms": round((time.perf_counter() - t0) * 1000.0, 1)}


class AuditorCierre:
    """Audita todos los lotes del overlay y revierte los que fallan."""

    def __init__(self, overlay: Path) -> None:
        self.overlay = overlay
        self._lock = threading.Lock()

    def lotes_pendientes(self) -> List[Dict[str, str]]:
        """Shims con marcador cuyo _pkg viaja en el overlay."""
        lotes: List[Dict[str, str]] = []
        if not self.overlay.exists():
            return lotes
        for f in sorted(self.overlay.rglob("*.py")):
            if "_pkg" in f.name or "__pycache__" in f.parts:
                continue
            try:
                if "SHIM de sesion" not in f.read_text(encoding="utf-8", errors="replace"):
                    continue
            except Exception:
                continue
            pkg = f.parent / (f.stem + "_pkg")
            if pkg.is_dir():
                lotes.append({"shim": str(f.relative_to(self.overlay)),
                              "pkg": str(pkg.relative_to(self.overlay))})
        return lotes

    def revertir_lote(self, rel_shim: str) -> Dict[str, Any]:
        """Restaura el original desde CHG/ y borra shim+pkg del overlay."""
        try:
            from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import CHG_DIR
        except Exception:
            CHG_DIR = CONSTRUCTOR_DIR
        shim = self.overlay / rel_shim
        stem = shim.stem
        try:
            bak = CHG_DIR / f"{stem}.py.bak_sesion"
            if bak.is_file():
                shutil.copy2(bak, shim)
            else:
                shim.unlink(missing_ok=True)
            shutil.rmtree(shim.parent / f"{stem}_pkg", ignore_errors=True)
            with self._lock:
                try:
                    man_path = self.overlay / "manifiesto_sesion.json"
                    man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
                    man.get("refactors", {}).pop(rel_shim, None)
                    man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass
            return {"exito": True, "mensaje": f"Lote {stem} revertido al original."}
        except Exception as exc:
            return {"exito": False, "mensaje": str(exc)[:160]}

    def certificar(self) -> Dict[str, Any]:
        """Audita cada lote: certifica los sanos, revierte los rotos."""
        t0 = time.perf_counter()
        certificados, revertidos, detalle = 0, 0, []
        for lote in self.lotes_pendientes():
            rep = auditar_lote(self.overlay / lote["pkg"])
            rep["shim"] = lote["shim"]
            if rep["exito"]:
                certificados += 1
                self._anotar_certificado(lote["shim"], rep)
            else:
                rev = self.revertir_lote(lote["shim"])
                revertidos += 1
                rep["revertido"] = rev.get("mensaje", "")
            detalle.append(rep)
        seg = round(time.perf_counter() - t0, 1)
        return {"exito": True, "lotes": len(detalle), "certificados": certificados,
                "revertidos": revertidos, "detalle": detalle, "segundos": seg,
                "mensaje": (f"Auditoria de cierre: {certificados} certificado(s), "
                            f"{revertidos} revertido(s) en {seg}s.")}

    def _anotar_certificado(self, rel_shim: str, rep: Dict[str, Any]) -> None:
        with self._lock:
            try:
                man_path = self.overlay / "manifiesto_sesion.json"
                man = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
                ref = man.setdefault("refactors", {}).setdefault(rel_shim, {})
                ref["test_ok"] = True
                ref["auditoria"] = {"sonda": rep.get("sonda", ""), "partes": rep.get("partes", 0),
                                    "ms": rep.get("ms", 0.0), "ts": time.time()}
                man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as exc:
                logger.debug("STMRFCR no anoto certificado: %s", exc)

    # ── Auto-reparacion: inyecta constantes faltantes ─────────
    def reparar_lote(self, rel_pkg: str) -> Dict[str, Any]:
        """Busca UPPER sin definir y las trae del .bak de CHG a la _p1."""
        pkg = self.overlay / rel_pkg
        partes = sorted(pkg.glob("_p*.py"))
        if not partes:
            return {"exito": False, "mensaje": "Sin partes que reparar."}
        faltan = chequear_constantes(partes)
        if not faltan:
            return {"exito": True, "mensaje": "Sin constantes faltantes.", "inyectadas": []}
        try:
            from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import CHG_DIR
        except Exception:
            CHG_DIR = CONSTRUCTOR_DIR
        stem = pkg.name[:-4]
        bak = CHG_DIR / f"{stem}.py.bak_sesion"
        if not bak.is_file():
            return {"exito": False, "mensaje": "Sin .bak en CHG para reparar."}
        defs: Dict[str, str] = {}
        for ln in bak.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*(?::\s*[^=\n]+)?=\s*(.+)$", ln.strip())
            if m and m.group(1) in faltan:
                defs[m.group(1)] = ln
        if not defs:
            return {"exito": False, "mensaje": f"Sin definicion para {faltan}."}
        try:
            p1 = partes[0]
            src = p1.read_text(encoding="utf-8", errors="replace")
            bloque = "# Auto-reparado STMRFCR: constantes izadas del original\n" + \
                "\n".join(defs[n] for n in sorted(defs)) + "\n"
            p1.write_text(src.replace("from __future__ import annotations\n",
                                      "from __future__ import annotations\n" + bloque, 1),
                          encoding="utf-8")
            return {"exito": True, "inyectadas": sorted(defs),
                    "mensaje": f"Inyectadas {len(defs)} constante(s) en {partes[0].name}."}
        except Exception as exc:
            return {"exito": False, "mensaje": str(exc)[:160]}

    def certificar_con_reparo(self) -> Dict[str, Any]:
        """Certifica; ante constantes faltantes intenta reparar y re-audita."""
        pri = self.certificar()
        reintentos = 0
        for det in pri.get("detalle", []):
            if det.get("exito") or "constante sin definir" not in " ".join(det.get("fallos", [])):
                continue
            pkg_rel = str(Path(det.get("shim", "")).parent / det.get("paquete", ""))
            rep_rep = self.reparar_lote(pkg_rel)
            if not rep_rep.get("exito"):
                continue
            reintentos += 1
            nuevo = auditar_lote(self.overlay / pkg_rel)
            nuevo["shim"] = det["shim"]
            det.update(nuevo)
            if nuevo.get("exito"):
                self._anotar_certificado(det["shim"], nuevo)
                pri["certificados"] += 1
                pri["revertidos"] = max(0, pri.get("revertidos", 1) - 1)
            else:
                self.revertir_lote(det["shim"])
        pri["reparos"] = reintentos
        pri["mensaje"] += f" Reparos: {reintentos}."
        return pri

    # ── Auditoria post-cierre del sistema vivo ────────────────
    def verificar_sistema_vivo(self, raiz: Path = LC_DIR) -> Dict[str, Any]:
        """Tras el cierre: ningun .py de LC puede ser shim sin su _pkg."""
        huerfanos: List[str] = []
        try:
            candidatos = list(raiz.rglob("*.py"))
        except Exception:
            candidatos = []
        for f in candidatos:
            if any(x in f.parts for x in ("__pycache__", "Constructor", "CHG", "pycache")):
                continue
            if "_pkg" in f.name:
                continue
            try:
                if "SHIM de sesion" not in f.read_text(encoding="utf-8", errors="replace"):
                    continue
            except Exception:
                continue
            pkg = f.parent / (f.stem + "_pkg")
            if not (pkg / "__init__.py").is_file():
                try:
                    huerfanos.append(str(f.relative_to(raiz)))
                except Exception:
                    huerfanos.append(f.name)
        return {"exito": not huerfanos, "huerfanos": huerfanos,
                "revisados": len(candidatos),
                "mensaje": ("Sistema vivo sano." if not huerfanos
                            else f"{len(huerfanos)} shim(s) huerfano(s): {huerfanos[:5]}.")}

    def exportar_certificado_json(self, destino: str = "") -> Dict[str, Any]:
        """Guarda el ultimo certificado para auditoria externa."""
        rep = self.certificar()
        ruta = Path(destino) if destino else self.overlay / "certificado_cierre.json"
        try:
            ruta.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
            return {"exito": True, "mensaje": f"Certificado en {ruta}."}
        except Exception as exc:
            return {"exito": False, "mensaje": str(exc)[:160]}

    def estado(self) -> Dict[str, Any]:
        """Foto: lotes pendientes y sanidad del sistema vivo."""
        return {"version": __version__, "overlay": str(self.overlay),
                "lotes": len(self.lotes_pendientes()),
                "sistema_vivo": self.verificar_sistema_vivo()}

    def ayuda(self) -> str:
        return ("Audito el cierre: 'audita el cierre', 'certifica los lotes', "
                "'repara el lote X', 'verifica el sistema vivo'.")

    def es_orden_auditoria(self, texto: str) -> bool:
        t = texto.lower()
        return any(k in t for k in ("audita", "auditor", "certifica", "cierre estructural",
                                    "problemas del cierre", "stmrfcr"))

    def ejecutar_orden(self, texto: str) -> Optional[str]:
        if not self.es_orden_auditoria(texto):
            return None
        rep = self.certificar()
        return ("Listo, audite yo misma el cierre. " + rep["mensaje"])

    def informe_para_lucia(self) -> str:
        n = len(self.lotes_pendientes())
        return (f"Audito el cierre yo misma (STMRFCR v{__version__}): {n} lote(s) pendientes. "
                "Certifico sintaxis, constantes, carga y sonda; revierto lo roto.")


_AUDITOR: Optional[AuditorCierre] = None


def get_auditor(overlay: Path) -> AuditorCierre:
    """Singleton del auditor estructural de cierre."""
    global _AUDITOR
    with _LOCK:
        if _AUDITOR is None or _AUDITOR.overlay != overlay:
            _AUDITOR = AuditorCierre(overlay)
        return _AUDITOR


def auditar_cierre(overlay: Path) -> Dict[str, Any]:
    """Atajo: certifica lotes y revierte rotos en una llamada."""
    return AuditorCierre(overlay).certificar()


if __name__ == "__main__":
    print("=" * 70)
    print(f"  STMRFCR v{__version__} - Auditor estructural del cierre")
    print("=" * 70)
