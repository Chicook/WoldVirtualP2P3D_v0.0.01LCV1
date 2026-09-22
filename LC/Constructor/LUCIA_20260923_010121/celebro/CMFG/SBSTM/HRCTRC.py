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


class GestorConstructorSesion:
    """Herrero con permisos de codigo: overlay en Constructor + unificacion."""

    def __init__(self, sesion_id: str = "") -> None:
        self.sesion_id: str = sesion_id or f"sesion_{time.strftime('%Y%m%d_%H%M%S')}"
        self.overlay: Path = CONSTRUCTOR_DIR / self.sesion_id
        self.manifiesto: Dict[str, Any] = {"sesion": self.sesion_id, "archivos": {}}
        self._lock = threading.Lock()
        self._activa: bool = False
        CONSTRUCTOR_DIR.mkdir(parents=True, exist_ok=True)

    # ── Ciclo de vida ─────────────────────────────────────────
    def iniciar_sesion(self, incluir: Optional[List[str]] = None) -> Dict[str, Any]:
        """Crea Constructor/sesion_<id>/ y despliega la copia de trabajo."""
        with self._lock:
            CONSTRUCTOR_DIR.mkdir(parents=True, exist_ok=True)
            self.overlay.mkdir(parents=True, exist_ok=True)
            rutas = incluir or list(RUTAS_VERSIONABLES)
            copiados, omitidos = 0, 0
            archivos: Dict[str, str] = {}
            for rel in rutas:
                origen = LC_DIR / rel
                if not origen.exists():
                    omitidos += 1
                    continue
                destino = self.overlay / rel
                try:
                    if origen.is_dir():
                        shutil.copytree(origen, destino, dirs_exist_ok=True,
                                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
                    else:
                        destino.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(origen, destino)
                    copiados += 1
                except Exception as exc:
                    logger.warning("HRCTRC no pudo versionar %s: %s", rel, exc)
                    omitidos += 1
            for f in sorted(self.overlay.rglob("*")):
                if f.is_file() and f.name != MANIFIESTO_NOMBRE:
                    archivos[str(f.relative_to(self.overlay))] = _sha256(f)
            self.manifiesto = {"sesion": self.sesion_id, "inicio": time.time(),
                               "archivos": archivos, "copiados": copiados, "omitidos": omitidos}
            self._guardar_manifiesto()
            self._activa = True
            return {"exito": True, "overlay": str(self.overlay),
                    "archivos_versionados": len(archivos),
                    "mensaje": f"Constructor listo: copia de trabajo en {self.overlay}."}

    def _guardar_manifiesto(self) -> None:
        try:
            (self.overlay / MANIFIESTO_NOMBRE).write_text(
                json.dumps(self.manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("HRCTRC manifiesto no guardado: %s", exc)

    def esta_activa(self) -> bool:
        return self._activa and self.overlay.exists()

    # ── Permisos via codigo: carpetas y archivos ──────────────
    def crear_carpeta(self, ruta_rel: str, tambien_en_lc: bool = False) -> Dict[str, Any]:
        """Crea carpetas en el overlay; con tambien_en_lc=True, tambien en LC real."""
        rel = Path(ruta_rel)
        if rel.is_absolute() or ".." in rel.parts:
            return {"exito": False, "mensaje": "Ruta no permitida (usa relativa sin '..')."}
        try:
            (self.overlay / rel).mkdir(parents=True, exist_ok=True)
            if tambien_en_lc:
                (LC_DIR / rel).mkdir(parents=True, exist_ok=True)
            return {"exito": True, "mensaje": f"Carpeta creada: {self.overlay / rel}."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"No pude crear {ruta_rel}: {exc}."}

    def escribir_overlay(self, ruta_rel: str, contenido: str) -> Dict[str, Any]:
        """LucIA modifica la copia de sesion; el sistema vivo queda intacto."""
        rel = Path(ruta_rel)
        if rel.is_absolute() or ".." in rel.parts:
            return {"exito": False, "mensaje": "Ruta no permitida."}
        try:
            destino = self.overlay / rel
            destino.parent.mkdir(parents=True, exist_ok=True)
            destino.write_text(contenido, encoding="utf-8")
            return {"exito": True, "mensaje": f"Escrito en overlay: {rel}."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"Error escribiendo {ruta_rel}: {exc}."}

    def leer(self, ruta_rel: str) -> Tuple[bool, str]:
        """Lee overlay primero (sombra); si no existe, lee el sistema real."""
        rel = Path(ruta_rel)
        for base in (self.overlay, LC_DIR):
            try:
                cand = base / rel
                if _dentro_de(base, cand) and cand.is_file():
                    return True, cand.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        return False, ""

    def listar_overlay(self) -> List[str]:
        """Archivos modificados/nuevos respecto al manifiesto inicial."""
        actual: Dict[str, str] = {}
        for f in sorted(self.overlay.rglob("*")):
            if f.is_file() and f.name != MANIFIESTO_NOMBRE:
                actual[str(f.relative_to(self.overlay))] = _sha256(f)
        base = self.manifiesto.get("archivos", {})
        return sorted(p for p, h in actual.items() if base.get(p) != h)

    # ── Unificacion y purga ───────────────────────────────────
    def vista_previa_unificacion(self) -> List[Dict[str, str]]:
        """Que cambiaria en LC sin aplicarlo: nuevos / modificados."""
        previo: List[Dict[str, str]] = []
        for rel in self.listar_overlay():
            real = LC_DIR / rel
            previo.append({"archivo": rel,
                           "estado": "nuevo" if not real.exists() else "modificado"})
        return previo

    def _refactor_con_test_ok(self, rel: str) -> bool:
        try:
            from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import PERMITIR_CON_TEST
            if Path(rel).name not in PERMITIR_CON_TEST:
                return False
            man = json.loads((self.overlay / "manifiesto_sesion.json").read_text(encoding="utf-8"))
            return bool(man.get("refactors", {}).get(rel, {}).get("test_ok", False))
        except Exception:
            return False

    def finalizar_sesion(self, aplicar: bool = True) -> Dict[str, Any]:
        """Unifica cambios en rutas reales y deja Constructor VACIA."""
        with self._lock:
            if not self.overlay.exists():
                self._activa = False
                return {"exito": True, "aplicados": 0, "mensaje": "Sin overlay; nada que unificar."}
            cambios = self.listar_overlay()
            # Prefiltro anti-shim huerfano (separadores normalizados): si un .py con
            # marcador SHIM viaja sin su paquete _pkg, se descarta el shim, el
            # paquete y el .bak para no tocar los archivos reales del sistema.
            norm = [c.replace("\\", "/") for c in cambios]
            descartar = set()
            omitidos: List[Dict[str, Any]] = []
            for c, cn in zip(cambios, norm):
                if not c.endswith(".py"):
                    continue
                try:
                    if "SHIM de sesion" not in (self.overlay / c).read_text(
                            encoding="utf-8", errors="replace"):
                        continue
                except Exception:
                    continue
                stem = Path(cn).stem
                pkg_pref = str(Path(cn).parent / (stem + "_pkg")).replace("\\", "/") + "/"
                if stem == "__init__":
                    pkg_pref = str(Path(cn).parent / "__init___pkg").replace("\\", "/") + "/"
                lote_pkg = [x for x in norm if x.startswith(pkg_pref)]
                if not lote_pkg:
                    descartar.add(cn)
                    omitidos.append({"archivo": c, "motivo": "shim_huerfano"})
                    logger.warning("HRCTRC omite shim huerfano: %s", c)
                elif self._refactor_con_test_ok(c):
                    logger.warning("HRCTRC acepta refactor con test_ok: %s (+%d partes)",
                                   c, len(lote_pkg))
                else:
                    logger.warning("HRCTRC omite refactor de sistema vivo: %s (+%d partes)",
                                   c, len(lote_pkg))
                    omitidos.append({
                        "archivo": c,
                        "partes": len(lote_pkg),
                        "motivo": "sistema_vivo_sin_test_previo_aprobado",
                    })
                    descartar.add(cn)
                    descartar.update(lote_pkg)
            cambios = [c for c, cn in zip(cambios, norm) if cn not in descartar]
            aplicados, errores = 0, 0
            if aplicar:
                for rel in cambios:
                    try:
                        origen, destino = self.overlay / rel, LC_DIR / rel
                        if not _dentro_de(LC_DIR, destino):
                            errores += 1
                            continue
                        destino.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(origen, destino)
                        aplicados += 1
                    except Exception as exc:
                        logger.warning("HRCTRC no unifico %s: %s", rel, exc)
                        errores += 1
            try:
                shutil.rmtree(self.overlay, ignore_errors=True)
                for resto in CONSTRUCTOR_DIR.iterdir():
                    if resto.is_dir():
                        shutil.rmtree(resto, ignore_errors=True)
                    else:
                        resto.unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("HRCTRC purga incompleta: %s", exc)
            self._activa = False
            if errores:
                estado_refactor = "fallido"
            elif omitidos:
                estado_refactor = "parcial_omitido"
            elif aplicados:
                estado_refactor = "aplicado"
            else:
                estado_refactor = "sin_cambios"
            return {"exito": errores == 0, "aplicados": aplicados, "errores": errores,
                    "omitidos": omitidos, "estado_refactor": estado_refactor,
                    "mensaje": f"Sesion unificada: {aplicados} archivo(s) en su ruta real. "
                               "Constructor vacia hasta la proxima sesion."}

    def descartar_sesion(self) -> Dict[str, Any]:
        """Cierra sin aplicar: purga Constructor y deja el sistema intacto."""
        return self.finalizar_sesion(aplicar=False)

    def estado(self) -> Dict[str, Any]:
        return {"version": __version__, "sesion": self.sesion_id,
                "activa": self.esta_activa(), "overlay": str(self.overlay),
                "pendientes": self.listar_overlay() if self.esta_activa() else [],
                "constructor_vacia": not any(CONSTRUCTOR_DIR.iterdir()) if CONSTRUCTOR_DIR.exists() else True,
                "lc": str(LC_DIR)}

    # ── Modelo de sesion: descargar, usar y versionar ─────────────────────────
    def descargar_modelo_sesion(self, modelo_id: str = "") -> Dict[str, Any]:
        """Descarga un modelo ligero y lo deja disponible para la copia de sesion."""
        try:
            from LC.modelosIAlocal.MDSTM import get_gestor_mdstm
            gestor = get_gestor_mdstm()
            if modelo_id:
                return gestor.descargar(modelo_id)
            reps = gestor.descargar_recomendados(limite=1)
            return reps[0] if reps else {"exito": False, "mensaje": "Sin recomendados."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"MDSTM no disponible: {exc}."}

    def responder_con_modelo_sesion(self, prompt: str, modelo_tag: str = "") -> Dict[str, Any]:
        """Usa el modelo local descargado para responder DENTRO de la sesion."""
        try:
            from LC.modelosIAlocal.MDSTM import get_gestor_mdstm
            from LC.celebro.CMFG.SBSTM.DSIALCLGRG import consultar_lucia_local
            texto, mid, lat = consultar_lucia_local(prompt, None)
            self._anotar_uso_modelo(mid, prompt, texto)
            return {"exito": True, "texto": texto, "modelo": mid, "latencia_ms": lat}
        except Exception as exc:
            return {"exito": False, "texto": "", "modelo": "error", "mensaje": str(exc)}

    def _anotar_uso_modelo(self, modelo: str, prompt: str, respuesta: str) -> None:
        try:
            bitacora = self.overlay / "bitacora_modelo_sesion.jsonl"
            bitacora.parent.mkdir(parents=True, exist_ok=True)
            with open(bitacora, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "modelo": modelo,
                                     "prompt": prompt[:300], "respuesta": respuesta[:500]},
                                    ensure_ascii=False) + "\n")
        except Exception:
            pass

    def ejecutar_overlay(self, script_rel: str = "mainLCSTM.py",
                         args: Optional[List[str]] = None, timeout: float = 120.0) -> Dict[str, Any]:
        """Ejecuta la version de sesion (copia en Constructor) como subproceso."""
        import subprocess as _sp
        script = self.overlay / script_rel
        if not script.is_file():
            return {"exito": False, "mensaje": f"No existe en overlay: {script_rel}."}
        try:
            r = _sp.run(["python", str(script)] + list(args or []),
                        capture_output=True, text=True, encoding="utf-8",
                        errors="replace", timeout=timeout)
            return {"exito": r.returncode == 0, "rc": r.returncode,
                    "salida": (r.stdout or "")[-2000:], "error": (r.stderr or "")[-1000:],
                    "mensaje": f"Overlay ejecutado con rc={r.returncode}."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"Fallo ejecutando overlay: {exc}."}

    def resumen_sesion(self) -> str:
        """Resumen factual para que LucIA informe lo construido en la sesion."""
        pendientes = self.listar_overlay() if self.esta_activa() else []
        return (f"Sesion {self.sesion_id}: {len(pendientes)} archivo(s) en construccion "
                f"({', '.join(pendientes[:5])}{'...' if len(pendientes) > 5 else ''}). "
                "Al cerrar se unifican en su ruta real y Constructor queda vacia.")

    # ── Lenguaje natural ──────────────────────────────────────
    def es_orden_constructor(self, texto: str) -> bool:
        t = texto.lower()
        verbos = ("crea", "genera", "construye", "fabrica", "levanta", "abre")
        objetos = ("carpeta", "directorio", "constructor", "copia", "version",
                   "overlay", "sesion", "unifica", "unificar", "sesión")
        return any(v in t for v in verbos) and any(o in t for o in objetos)

    def ejecutar_orden(self, texto: str) -> Optional[str]:
        """Ejecuta ordenes de construccion sin pasar por el modelo remoto."""
        if not self.es_orden_constructor(texto):
            return None
        t = texto.lower()
        if "unific" in t:
            rep = self.finalizar_sesion(aplicar=True)
            return "Hecho: " + rep["mensaje"]
        if "descart" in t or "cancela" in t:
            rep = self.descartar_sesion()
            return "Hecho: sesion descartada, " + rep["mensaje"]
        if "carpeta" in t or "directorio" in t:
            m = re.search(r"(?:carpeta|directorio)\s+(?:llamada\s+)?[\"']?([\w\-./ ]+?)[\"']?\s*$",
                          texto, re.IGNORECASE)
            nombre = (m.group(1).strip() if m else "Constructor") or "Constructor"
            rep = self.crear_carpeta(nombre, tambien_en_lc=("constructor" in t))
            return ("Claro que si, SI puedo crear carpetas. " if rep["exito"] else "") + rep["mensaje"]
        if "constructor" in t or "copia" in t or "overlay" in t or "sesion" in t:
            rep = self.iniciar_sesion()
            return ("Por supuesto: " if rep["exito"] else "") + rep["mensaje"] + " " + self.resumen_sesion()
        return None

    def informe_para_lucia(self) -> str:
        est = self.estado()
        return (f"Soy LucIA y SI puedo crear carpetas y modificar el sistema (HRCTRC v{__version__}). "
                f"Constructor activa: {est['activa']}. Pendientes: {len(est['pendientes'])}. "
                "Trabajo sobre una copia en LC/Constructor y al cerrar unifico cada archivo "
                "en su ruta real, dejando Constructor vacia.")

    # ── Diffs y parches ───────────────────────────────────────
    def dif_detallado(self, max_lineas: int = 120) -> List[Dict[str, Any]]:
        """Diff unificado overlay vs LC real para cada pendiente (auditoria)."""
        import difflib
        diffs: List[Dict[str, Any]] = []
        for rel in self.listar_overlay():
            try:
                real = LC_DIR / rel
                txt_nuevo = (self.overlay / rel).read_text(encoding="utf-8", errors="replace").splitlines()
                txt_viejo = real.read_text(encoding="utf-8", errors="replace").splitlines() if real.is_file() else []
                uni = list(difflib.unified_diff(txt_viejo, txt_nuevo, str(real), str(self.overlay / rel)))
                diffs.append({"archivo": rel, "lineas": len(uni),
                              "vista": "\n".join(uni[:max_lineas])})
            except Exception as exc:
                diffs.append({"archivo": rel, "lineas": 0, "vista": f"Sin diff: {exc}"})
        return diffs

    def exportar_parche(self, destino: str = "") -> Dict[str, Any]:
        """Guarda los diffs pendientes en un .patch para revision humana."""
        diffs = self.dif_detallado()
        if not diffs:
            return {"exito": False, "mensaje": "Sin cambios pendientes que exportar."}
        ruta = Path(destino) if destino else CONSTRUCTOR_DIR / f"parche_{self.sesion_id}.patch"
        try:
            with open(ruta, "w", encoding="utf-8") as fh:
                for d in diffs:
                    fh.write(f"### {d['archivo']} ({d['lineas']} lineas)\n{d['vista']}\n")
            return {"exito": True, "mensaje": f"Parche exportado: {ruta}."}
        except Exception as exc:
            return {"exito": False, "mensaje": f"No pude exportar: {exc}."}


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
