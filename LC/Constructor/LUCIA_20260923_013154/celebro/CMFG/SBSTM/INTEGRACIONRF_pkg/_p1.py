"""
INTEGRACIONRF - parte 1/2 (version de sesion LucIA).
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

EXCLUIR_DIRS: Final[Tuple[str, ...]] = ("__pycache__", "Constructor", "pycache",
                                        ".git", "CHG", "node_modules", ".ruff_cache")
TIMEOUT_GIT: Final[float] = 60.0

_LOCK: Final[threading.Lock] = threading.Lock()


def _rel(lc_file: Path) -> str:
    try:
        return str(lc_file.resolve().relative_to(LC_DIR.resolve()))
    except Exception:
        return lc_file.name


def listar_rutas_originales(raiz: Path = LC_DIR,
                            extensiones: Tuple[str, ...] = (".py",)) -> List[Dict[str, Any]]:
    """Inventario rapido de archivos originales de LC (ruta, lineas, bytes)."""
    out: List[Dict[str, Any]] = []
    pila: List[Path] = [raiz]
    while pila:
        base = pila.pop()
        try:
            with os.scandir(base) as it:
                entradas = list(it)
        except Exception:
            continue
        for e in entradas:
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name not in EXCLUIR_DIRS:
                        pila.append(Path(e.path))
                elif e.is_file(follow_symlinks=False) and e.name.endswith(extensiones):
                    st = e.stat()
                    out.append({"ruta": _rel(Path(e.path)), "bytes": st.st_size,
                                "mtime": st.st_mtime})
            except Exception:
                pass
    return sorted(out, key=lambda d: d["ruta"])


def contar_lineas_rapido(ruta: Path, limite: int = 200000) -> int:
    """Cuenta lineas sin cargar el archivo entero (eficiente)."""
    n, total = 0, 0
    try:
        with open(ruta, "rb") as fh:
            while True:
                bloque = fh.read(65536)
                if not bloque:
                    break
                total += len(bloque)
                if total > limite:
                    return -1
                n += bloque.count(b"\n")
        return n + 1
    except Exception:
        return -1


def diff_rutas(antes: List[Dict[str, Any]],
               despues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compara inventarios: nuevos, eliminados y modificados (por mtime/bytes)."""
    a = {d["ruta"]: d for d in antes}
    b = {d["ruta"]: d for d in despues}
    nuevos = sorted(set(b) - set(a))
    eliminados = sorted(set(a) - set(b))
    modificados = sorted(r for r in set(a) & set(b)
                         if (a[r]["bytes"], a[r]["mtime"]) != (b[r]["bytes"], b[r]["mtime"]))
    return {"nuevos": nuevos, "eliminados": eliminados, "modificados": modificados,
            "total_antes": len(a), "total_despues": len(b)}


class IntegradorRefactor:
    """Bitacora .md -> pesos neuronales -> IPFS -> rama devopencode."""

    def __init__(self, sesion_id: str = "") -> None:
        self.sesion_id = sesion_id or f"RF_{time.strftime('%Y%m%d_%H%M%S')}"
        self._eventos: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        CHG_DIR.mkdir(parents=True, exist_ok=True)
        self.md_path = CHG_DIR / f"INTEGRACIONRF_{self.sesion_id}.md"
        self.registrar("sesion_iniciada", f"Integrador {__version__} activo.")

    # ── Bitacora en vivo ──────────────────────────────────────
    def registrar(self, tipo: str, detalle: str) -> None:
        with self._lock:
            self._eventos.append({"ts": time.time(), "tipo": tipo, "detalle": detalle})

    def eventos(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._eventos)

    # ── Paso 1: sobreescribir archivos con el refactor ────────
    def aplicar_refactor(self) -> Dict[str, Any]:
        """Sobreescribe cada archivo con el codigo nuevo del overlay HRCTRC."""
        try:
            from LC.celebro.CMFG.SBSTM.HRCTRC import get_gestor_hrctrc
            rep = get_gestor_hrctrc().finalizar_sesion(aplicar=True)
            estado = rep.get("estado_refactor", "desconocido")
            self.registrar("refactor_" + estado,
                           f"{rep.get('aplicados', 0)} aplicados; "
                           f"{len(rep.get('omitidos', []))} omitidos; {rep.get('mensaje', '')}")
            return {"exito": rep.get("exito", False), **rep}
        except Exception as exc:
            self.registrar("refactor_error", str(exc))
            return {"exito": False, "mensaje": f"HRCTRC no disponible: {exc}"}

    def verificar_distribucion(self) -> Dict[str, Any]:
        """Comprueba que lo unificado existe en su ruta real con tamano > 0."""
        ok, mal = 0, []
        for item in listar_rutas_originales():
            p = LC_DIR / item["ruta"]
            if p.is_file() and p.stat().st_size > 0:
                ok += 1
            else:
                mal.append(item["ruta"])
        self.registrar("distribucion_verificada", f"{ok} archivos OK, {len(mal)} mal.")
        return {"exito": not mal, "archivos_ok": ok, "fallos": mal}

    # ── Paso 2: transcribir actividad a .md ───────────────────
    def generar_md(self, destino: Optional[Path] = None) -> Dict[str, Any]:
        """Vuelca toda la actividad de ejecucion al .md en CHG/."""
        dest = destino or self.md_path
        evs = self.eventos()
        lineas = [f"# Bitacora INTEGRACIONRF - sesion {self.sesion_id}",
                  f"_Generado: {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                  f"Eventos: {len(evs)} | {__subsystem__} v{__version__}_", "",
                  "## Actividad de ejecucion", ""]
        for e in evs:
            hora = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
            lineas.append(f"- `{hora}` **{e['tipo']}**: {e['detalle']}")
        lineas += ["", "## Monitoreo del sistema", ""]
        turnos = [e for e in evs if e["tipo"] == "turno"]
        lineas.append(f"- Turnos de sesion: {len(turnos)}")
        for t in turnos[-20:]:
            hora = time.strftime("%H:%M:%S", time.localtime(t["ts"]))
            lineas.append(f"  - `{hora}` {t['detalle']}")
        try:
            dif = self.snapshot_fin_dif()
            lineas.append(f"- Refactor: +{len(dif['nuevos'])}/-{len(dif['eliminados'])}/"
                          f"~{len(dif['modificados'])} (nuevos/eliminados/modificados)")
        except Exception:
            pass
        lineas += ["", "## Distribucion de subsistemas tras refactor", ""]
        for item in listar_rutas_originales():
            n = contar_lineas_rapido(LC_DIR / item["ruta"])
            marca = "OK" if 0 <= n <= 450 else ("EXCEDE" if n > 450 else "?")
            lineas.append(f"- `{item['ruta']}` ({n} lin) [{marca}]")
        lineas.append("")
        try:
            dest.write_text("\n".join(lineas), encoding="utf-8")
            self.registrar("md_generado", f"{dest.name} ({dest.stat().st_size} B).")
            return {"exito": True, "md": str(dest), "eventos": len(evs)}
        except Exception as exc:
            return {"exito": False, "mensaje": str(exc)}

    # ── Paso 3: .md -> pesos neuronales (PSNRL) ───────────────
    def convertir_a_pesos(self, md_path: Optional[Path] = None) -> Dict[str, Any]:
        """El .md se asimila en las 50 neuronas y persiste npz/js en PSNRL."""
        md = md_path or self.md_path
        try:
            texto = Path(md).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"exito": False, "mensaje": f"MD ilegible: {exc}"}
        try:
            from LC.celebro.CMFG.PSNRCV import get_conversor_pesos
            conv = get_conversor_pesos()
            chunks = [texto[i:i + 1500] for i in range(0, len(texto), 1500)][:12]
            for ch in chunks:  # rapido: max 12 pasadas, sin red
                try:
                    conv.procesar_consulta_a_pesos(ch)
                except Exception:
                    pass
            npz, js = conv.persistir_pesos_en_psnrl(etiqueta=f"integrar_rf_{self.sesion_id}")
            self.registrar("pesos_generados", f"{npz.name} + {js.name}.")
            return {"exito": True, "npz": str(npz), "json": str(js)}
        except Exception as exc:
            self.registrar("pesos_error", str(exc))
            return {"exito": False, "mensaje": f"PSNRCV fallo: {exc}"}

    # ── Paso 4: subir pesos a IPFS y borrar el .md ─────────────
    def subir_a_ipfs(self, npz_path: str = "") -> Dict[str, Any]:
        """Sube el npz a IPFS; con CID confirmado borra el .md de CHG/."""
        try:
            from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
            if not npz_path:
                return {"exito": False, "mensaje": "Sin npz que subir."}
            rep = get_ipfs_manager().almacenar_pesos(
                npz_path, nombre_modelo=f"integrar_rf_{self.sesion_id}",
                eliminar_local=False,
                metadatos={"sesion": self.sesion_id, "tipo": "bitacora_refactor"})
            cid = rep.get("cid", "")
            self.registrar("ipfs_subido", f"CID {cid or '?'} via {rep.get('nodo', '?')}.")
            if cid:  # solo con CID se borra el .md; si falla, se conserva
                try:
                    self.md_path.unlink(missing_ok=True)
                    self.registrar("md_borrado", f"{self.md_path.name} tras CID {cid[:16]}.")
                except Exception:
                    pass
            return {"exito": True, **rep}
        except Exception as exc:
            self.registrar("ipfs_error", str(exc))
            return {"exito": False, "mensaje": f"IPFS fallo: {exc}"}

    # ── Paso 5: rama devopencode actualizada ──────────────────
    def actualizar_rama(self, rama: str = RAMA_OBJETIVO) -> Dict[str, Any]:
        """git add+commit+push a devopencode tras actualizar el sistema."""
        def _git(*args: str) -> Tuple[int, str]:
            try:
                r = subprocess.run(["git", "-C", str(ROOT_DIR), *args],
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=TIMEOUT_GIT)
                return r.returncode, (r.stdout + r.stderr)[-500:]
            except Exception as exc:
                return 99, str(exc)[:200]
        rc, out = _git("add", "LC")
        if rc != 0:
            return {"exito": False, "mensaje": f"git add fallo: {out}"}
        rc, out = _git("diff", "--cached", "--quiet")
        if rc == 0:
            self.registrar("rama_sin_cambios", f"{rama} ya actualizada.")
            return {"exito": True, "mensaje": f"{rama} sin cambios que subir."}
        rc, out = _git("commit", "-m", f"INTEGRACIONRF {self.sesion_id}: refactor unificado + bitacora")
        if rc != 0:
            return {"exito": False, "mensaje": f"git commit fallo: {out}"}
        rc, out = _git("push", "origin", rama)
        ok = rc == 0
        self.registrar("rama_actualizada" if ok else "rama_error", f"push {rama}: {out[-120:]}")
        return {"exito": ok, "mensaje": f"push origin {rama}: {'OK' if ok else out[-200:]}"}

    # ── Pipeline completo de cierre ───────────────────────────
    def cierre_completo(self) -> Dict[str, Any]:
        """Unificar -> verificar -> .md -> pesos -> IPFS (borra .md) -> devopencode."""
        t0 = time.perf_counter()
        reporte: Dict[str, Any] = {"sesion": self.sesion_id}
        reporte["refactor"] = self.aplicar_refactor()
        reporte["distribucion"] = self.verificar_distribucion()
        reporte["respaldos"] = self.documentar_respaldos_chg()
        reporte["md"] = self.generar_md()
        reporte["pesos"] = self.convertir_a_pesos()
        npz = reporte["pesos"].get("npz", "") if reporte["pesos"].get("exito") else ""
        reporte["ipfs"] = self.subir_a_ipfs(npz)
        reporte["md_borrado"] = not self.md_path.exists()
        reporte["rama"] = self.actualizar_rama()
        reporte["segundos"] = round(time.perf_counter() - t0, 1)
        reporte["exito"] = all(reporte[k].get("exito") for k in ("md", "pesos"))
        self.registrar("cierre_completo",
                       f"exito={reporte['exito']} en {reporte['segundos']}s.")
        return reporte

    def es_orden_integracion(self, texto: str) -> bool:
        t = texto.lower()
        return any(k in t for k in ("integrar", "bitacora", "bitácora", "sube a ipfs",
                                    "cierre completo", "devopencode"))

    def ejecutar_orden(self, texto: str) -> Optional[str]:
        if not self.es_orden_integracion(texto):
            return None
        t = texto.lower()
        if "rama" in t or "devopencode" in t:
            rep = self.actualizar_rama()
            return "Hecho: " + rep["mensaje"]
        if "bitacora" in t or "bitácora" in t:
            rep = self.generar_md()
            return ("Bitacora generada: " + rep.get("md", "") if rep.get("exito")
                    else "No pude generar la bitacora: " + rep.get("mensaje", ""))
        rep = self.cierre_completo()
        estado_refactor = rep["refactor"].get("estado_refactor", "desconocido")
        return (f"Cierre completo en {rep['segundos']}s: refactor "
                f"{estado_refactor}, "
                f"IPFS {rep['ipfs'].get('cid', 'sin CID')}, rama: {rep['rama'].get('mensaje', '')}.")

    def informe_para_lucia(self) -> str:
        n = len(listar_rutas_originales())
        return (f"INTEGRACIONRF v{__version__}: inventario de {n} archivos. "
                "El cierre aplica solo cambios con prueba previa aprobada; "
                "los refactors omitidos se informan como parciales.")

    # ── Snapshots, inventario e higiene ───────────────────────
    def snapshot_inicio(self) -> Dict[str, Any]:
        """Guarda el inventario de rutas originales al abrir la sesion."""
        with self._lock:
            self._snapshot = listar_rutas_originales()
            n = len(self._snapshot)
        self.registrar("snapshot_inicio", f"{n} rutas originales registradas.")
        return {"rutas": n}

    def snapshot_fin_dif(self) -> Dict[str, Any]:
        """Compara el estado final contra el snapshot: que cambio el refactor."""
        antes = getattr(self, "_snapshot", None) or []
        dif = diff_rutas(antes, listar_rutas_originales())
        self.registrar("snapshot_dif",
                       f"+{len(dif['nuevos'])}/-{len(dif['eliminados'])}/~{len(dif['modificados'])}.")
        return dif

    def exportar_inventario_json(self, destino: str = "") -> Dict[str, Any]:
        """Guarda el inventario completo de LC en JSON para auditoria."""
        inv = listar_rutas_originales()
        for item in inv:
            item["lineas"] = contar_lineas_rapido(LC_DIR / item["ruta"])
        ruta = Path(destino) if destino else PACKAGE_ROOT / f"inventario_{self.sesion_id}.json"
        try:
            ruta.write_text(json.dumps({"sesion": self.sesion_id, "total": len(inv),
                                        "archivos": inv}, indent=1, ensure_ascii=False),
                            encoding="utf-8")
            self.registrar("inventario_exportado", f"{len(inv)} rutas en {ruta.name}.")
            return {"exito": True, "json": str(ruta), "total": len(inv)}
        except Exception as exc:
            return {"exito": False, "mensaje": str(exc)}

    def limpiar_mds_antiguos(self, conservar: int = 3) -> int:
        """Borra bitacoras .md de sesiones viejas en CHG; conserva recientes."""
        try:
            mds = sorted(CHG_DIR.glob("INTEGRACIONRF_*.md"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
            n = 0
            for viejo in mds[conservar:]:
                viejo.unlink(missing_ok=True)
                n += 1
            return n
        except Exception:
            return 0

    # ── Respaldos CHG: .bak -> .md explicativo -> pesos ──────
    def documentar_respaldos_chg(self) -> Dict[str, Any]:
        """Cada .bak_sesion de CHG/ se explica en .md y ese .md va a pesos."""
        import ast as _ast
        baks = sorted(CHG_DIR.glob("*.bak_sesion"))
        docs: List[Dict[str, Any]] = []
        for bak in baks:
            try:
                src = bak.read_text(encoding="utf-8", errors="replace")
                arbol = _ast.parse(src)
                doc = (_ast.get_docstring(arbol) or "").splitlines()
                desc = (doc[0].strip() if doc else "Sin descripcion.")[:160]
                clases = [n.name for n in arbol.body if isinstance(n, _ast.ClassDef)]
                funcs = [n.name for n in arbol.body if isinstance(n, _ast.FunctionDef)]
                lin = src.count("\n") + 1
                md = [f"# Codigo documentado: {bak.stem}", "",
                      f"_Respaldo pre-refactor | {lin} lineas | sesion {self.sesion_id}_", "",
                      "## Que hace", "", desc, "",
                      "## Clases"] + [f"- `{c}`" for c in clases] + ["", "## Funciones"] + \
                    [f"- `{f}()`" for f in funcs] + ["", "## Como funciona",
                      "Version original del archivo antes de dividirse a regla 400/450. "
                      "Su logica vive ahora en el paquete `_pkg` hermano; este texto preserva "
                      "el conocimiento para la red neuronal.", ""]
                md_path = CHG_DIR / f"DOC_{bak.stem}.md"
                md_path.write_text("\n".join(md), encoding="utf-8")
                rep = self.convertir_a_pesos(md_path)
                docs.append({"bak": bak.name, "md": md_path.name,
                             "pesos_ok": rep.get("exito", False)})
                self.registrar("respaldo_documentado",
                               f"{bak.name} -> {md_path.name} -> pesos={rep.get('exito')}.")
            except Exception as exc:
                docs.append({"bak": bak.name, "error": str(exc)[:120]})
        return {"exito": True, "documentados": docs, "total": len(docs)}

    def ayuda(self) -> str:
        return ("'integra el refactor', 'genera la bitacora', 'sube a ipfs', "
                "'actualiza la rama devopencode', 'cierre completo'.")

    def estado(self) -> Dict[str, Any]:
        """Foto rapida: eventos, md, snapshot y rama objetivo."""
        return {"version": __version__, "sesion": self.sesion_id, "eventos": len(self.eventos()),
                "md_existe": self.md_path.exists(), "rama": RAMA_OBJETIVO,
                "lc_archivos": len(listar_rutas_originales())}

    def resumen_cierre_txt(self, reporte: Dict[str, Any]) -> str:
        """Resumen de una linea del pipeline para consola y voz."""
        r = reporte
        refactor = r.get("refactor", {})
        estado_refactor = refactor.get("estado_refactor")
        if not estado_refactor:
            estado_refactor = "aplicado" if refactor.get("exito") else "fallido"
        omitidos = len(refactor.get("omitidos", []))
        detalle_omitidos = f" ({omitidos} omitidos)" if omitidos else ""
        return (f"Cierre {self.sesion_id} en {r.get('segundos', 0)}s: "
                f"refactor {estado_refactor}{detalle_omitidos} | "
                f"md {'OK' if r.get('md', {}).get('exito') else 'FALLO'} | "
                f"pesos {'OK' if r.get('pesos', {}).get('exito') else 'FALLO'} | "
                f"IPFS {r.get('ipfs', {}).get('cid', 'sin CID')}.")


_INTEGRADOR: Optional[IntegradorRefactor] = None


