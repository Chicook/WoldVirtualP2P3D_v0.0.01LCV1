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


class GestorHRCNTR:
    """Monitor, refactorizador y actualizador neural del sistema."""

    def __init__(self) -> None:
        self.sesion_id: str = f"hrctnr_{time.strftime('%Y%m%d_%H%M%S')}"
        self._inventario: List[Dict[str, Any]] = []
        self._seleccionados: List[Dict[str, Any]] = []
        self._generados: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._actualizado = False
        CHG_DIR.mkdir(parents=True, exist_ok=True)

    def monitor_sistema(self) -> Dict[str, Any]:
        escaneo, neuronas = [], []
        for raiz, dirs, archivos in os.walk(LC_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUIR_DIRS]
            for arch in sorted(archivos):
                if not arch.endswith(".py") or arch == "conftest.py":
                    continue
                ruta = Path(raiz) / arch
                rel = str(ruta.relative_to(LC_DIR))
                n_lineas = _contar_lineas(ruta)
                info: Dict[str, Any] = {
                    "ruta": rel, "absoluta": ruta, "lineas": n_lineas,
                    "sobreescrito": n_lineas > MAX_LINEAS,
                    "clases": [], "funciones": [], "imports": [],
                }
                try:
                    arbol = ast.parse(ruta.read_text(encoding="utf-8", errors="replace"))
                    for nodo in ast.iter_child_nodes(arbol):
                        if isinstance(nodo, ast.ClassDef):
                            info["clases"].append({
                                "nombre": nodo.name,
                                "lineas": nodo.end_lineno - nodo.lineno + 1
                                if hasattr(nodo, 'end_lineno') else 0,
                            })
                        elif isinstance(nodo, ast.FunctionDef):
                            info["funciones"].append(nodo.name)
                        elif isinstance(nodo, (ast.ImportFrom, ast.Import)):
                            info["imports"].append(
                                "from ... import ..."
                                if isinstance(nodo, ast.ImportFrom)
                                else "import ...")
                except Exception:
                    pass
                if any(p in rel for p in ("red_neuronal", "RF_EN", "RF_SL", "ENRN")):
                    neuronas.append(info)
                escaneo.append(info)
        self._inventario = escaneo
        oversized = [e for e in escaneo if e["sobreescrito"]]
        return {"total_archivos": len(escaneo), "total_neuronas": len(neuronas),
                "oversized": len(oversized), "neuronas": neuronas,
                "archivos": escaneo, "sesion": self.sesion_id}

    def listar_modulos(self) -> List[Dict[str, Any]]:
        if not self._inventario:
            self.monitor_sistema()
        return [{"ruta": e["ruta"], "lineas": e["lineas"],
                 "clases": [c["nombre"] for c in e["clases"]],
                 "sobreescrito": e["sobreescrito"]} for e in self._inventario]

    def listar_neuronas(self) -> List[Dict[str, Any]]:
        if not self._inventario:
            self.monitor_sistema()
        return self._inventario

    def es_oversized(self, ruta_rel: str) -> bool:
        for e in self._inventario:
            if e["ruta"] == ruta_rel:
                return e["sobreescrito"]
        return _contar_lineas(LC_DIR / ruta_rel) > MAX_LINEAS

    def seleccionar_clases(self) -> List[Dict[str, Any]]:
        if not self._inventario:
            self.monitor_sistema()
        sel = []
        for e in self._inventario:
            if e["sobreescrito"] and e["clases"]:
                for cls in e["clases"]:
                    if any(p in e["ruta"] for p in MODULOS_VIVOS):
                        continue
                    sel.append({"modulo_origen": e["ruta"], "clase": cls["nombre"],
                                "lineas_clase": cls["lineas"], "lineas_modulo": e["lineas"],
                                "ruta_absoluta": e["absoluta"]})
        self._seleccionados = sel
        return sel

    def _llamar_modelo_free(self, prompt: str) -> str:
        try:
            from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree
            cliente = get_cliente_iafree()
            if cliente and cliente.esta_autenticado():
                modelos = cliente.gestor.listar_modelos()
                free_models = [m for m in modelos if "free" in m.get("id", "").lower()]
                if not free_models:
                    free_models = modelos[:2]
                modelo_id = free_models[0]["id"] if free_models else modelos[0]["id"]
                resp = cliente.generar_respuesta(
                    prompt=prompt, contexto_neuronal=None,
                    max_reintentos=1, stream_en_vivo=False,
                )
                return resp[0] or ""
        except Exception as exc:
            logger.debug("HRCNTR modelo free no disponible: %s", exc)
        return ""

    def generar_subsistema(self, modulo_origen: str, clase: str) -> Dict[str, Any]:
        ruta_orig = LC_DIR / modulo_origen
        if not ruta_orig.exists():
            return {"exito": False, "mensaje": f"No existe: {modulo_origen}"}
        try:
            codigo_orig = ruta_orig.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return {"exito": False, "mensaje": str(exc)}
        prompt = (f"Expande la clase {clase} del siguiente codigo Python a "
                  f"entre {MIN_LINEAS} y {MAX_LINEAS} lineas. Mantiene la misma "
                  f"funcionalidad pero con documentacion, metodos auxiliares, "
                  f"validaciones y ejemplos de uso. El codigo debe ser ejecutable "
                  f"y completo. Incluye docstring de clase y cada metodo.\n\n"
                  f"CODIGO ORIGINAL:\n{codigo_orig[:3000]}")
        resultado = self._llamar_modelo_free(prompt)
        nombre_base = Path(modulo_origen).stem
        destino = (PACKAGE_ROOT / f"{nombre_base}_hrctnr"
                   if resultado else PACKAGE_ROOT / f"{nombre_base}_expandido.py")
        dl = chr(34) * 3
        if resultado:
            destino.mkdir(parents=True, exist_ok=True)
            init_text = (f"{dl}\n{nombre_base}_hrctnr - subsistema expandido.\n"
                         f"Generado por HRCNTR v{__version__} con modelo free OpenRouter.\n"
                         f"Importa desde {modulo_origen}.\n"
                         f"{dl}\n"
                         "from __future__ import annotations\n\n"
                         f"from {modulo_origen} import *  # noqa\n")
            (destino / "__init__.py").write_text(init_text, encoding="utf-8")
            main_file = destino / f"{nombre_base}_expandido.py"
            contenido = (f"{dl}\n{clase} - version expandida "
                         f"({MIN_LINEAS}-{MAX_LINEAS} lineas).\n"
                         "Modelo free OpenRouter. Subsistema HRCNTR.\n"
                         f"{dl}\n"
                         "from __future__ import annotations\n\n"
                         "import importlib\nimport sys as _sys\n"
                         f"_mod_orig = importlib.import_module('{modulo_origen}')\n"
                         f"{clase} = getattr(_mod_orig, '{clase}', None)\n\n"
                         f"{resultado}")
            main_file.write_text(contenido, encoding="utf-8")
        else:
            destino.write_text(codigo_orig, encoding="utf-8")
            main_file = destino
        n_lines = main_file.read_text(encoding="utf-8", errors="replace").count("\n") + 1
        with self._lock:
            self._generados.append({"modulo_origen": modulo_origen, "clase": clase,
                                    "subsistema": str(destino), "archivo": str(main_file),
                                    "lineas": n_lines})
        tipo = "generado" if resultado else "copiado (fallback)"
        return {"exito": True, "subsistema": str(destino), "archivo": str(main_file),
                "lineas": n_lines, "mensaje": f"Subsistema {tipo}: {destino} ({n_lines} lin)"}

    def ejecutar_refactor(self, overlay: str = "", mostrar_barra: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if mostrar_barra:
            _escribir_barra(barra_progreso(0, 5, "refactorizando: monitoreando") + "\n")
        monitor = self.monitor_sistema()
        if mostrar_barra:
            _escribir_barra(barra_progreso(1, 5, "monitoreando: OK") + "\n")
        seleccion = self.seleccionar_clases()
        if mostrar_barra:
            _escribir_barra(barra_progreso(2, 5, f"seleccionadas: {len(seleccion)} clases") + "\n")
        if not seleccion:
            if mostrar_barra:
                _escribir_barra(barra_progreso(5, 5, "todo en regla") + "\n")
                print()
            return {"exito": True, "mensaje": "Todo el codigo cumple 400/450. Sin refactorizar.",
                    "generados": 0, "sesion": self.sesion_id}
        total = len(seleccion)
        hechos, fallos = 0, 0
        detalle: List[Dict[str, Any]] = []
        for i, obj in enumerate(seleccion, 1):
            if mostrar_barra:
                _escribir_barra(barra_progreso(i, total, f"generando: {obj['clase']}") + "\n")
            rep = self.generar_subsistema(obj["modulo_origen"], obj["clase"])
            detalle.append(rep)
            if rep.get("exito"):
                hechos += 1
            else:
                fallos += 1
            time.sleep(0.05)
        if mostrar_barra:
            _escribir_barra(barra_progreso(total, total, "refactorizacion completa") + "\n")
            print()
        segundos = round(time.perf_counter() - t0, 2)
        return {"exito": fallos == 0, "sesion": self.sesion_id, "monitoreo": monitor,
                "seleccionados": len(seleccion), "generados": hechos, "fallos": fallos,
                "detalle": detalle, "segundos": segundos,
                "mensaje": (f"Version de sesion HRCNTR lista: {hechos}/{total} subsistemas "
                            f"generados con modelos free en {segundos}s."
                            if total else "Sin clases para refactorizar.")}

    def _backup_pre_update(self) -> Dict[str, Any]:
        CHG_DIR.mkdir(parents=True, exist_ok=True)
        respaldos = []
        for item in self._inventario:
            ruta = item["absoluta"]
            if ruta.is_file() and not any(p in item["ruta"] for p in MODULOS_VIVOS):
                try:
                    shutil.copy2(ruta, CHG_DIR / f"{Path(item['ruta']).stem}.py.bak_hrctnr")
                    respaldos.append(item["ruta"])
                except Exception:
                    pass
        return {"respaldados": len(respaldos), "bak_dir": str(CHG_DIR)}

    def actualizar_sistema(self, confirmar: bool = False) -> Dict[str, Any]:
        if not confirmar:
            return {"exito": False,
                    "mensaje": "Confirmacion requerida: establece confirmar=True.",
                    "accion": "Llama con confirmar=True."}
        with self._lock:
            if self._actualizado:
                return {"exito": True, "mensaje": "Sistema ya actualizado."}
            backup = self._backup_pre_update()
            aplicados = errores = 0
            for item in self._inventario:
                ruta = item["absoluta"]
                if not ruta.is_file() or any(p in item["ruta"] for p in MODULOS_VIVOS):
                    continue
                try:
                    gen = next((g for g in self._generados
                                if Path(item["ruta"]).stem in g["modulo_origen"]), None)
                    if gen and Path(gen["archivo"]).exists():
                        ruta.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(Path(gen["archivo"]), ruta)
                        aplicados += 1
                except Exception:
                    errores += 1
            self._actualizado = True
            return {"exito": errores == 0, "aplicados": aplicados, "errores": errores,
                    "backup": backup,
                    "mensaje": f"Sistema actualizado: {aplicados} archivos, "
                               f"{errores} errores, {backup['respaldados']} respaldos en CHG/."}

    def confirmar_actualizacion(self) -> str:
        n = len([e for e in self._inventario
                 if e["sobreescrito"] and not any(p in e["ruta"] for p in MODULOS_VIVOS)])
        if n == 0:
            return "Sistema en regla. No requiere actualizacion."
        return ("╔══════════════════════════════════════════════════╗\n"
                "║     CONFIRMAR ACTUALIZACIÓN DEL SISTEMA          ║\n"
                "╠══════════════════════════════════════════════════╣\n"
                f"║  {n} módulos refactorizados listos para aplicar.  ║\n"
                "║  Respaldos en CHG/ (bak_hrctnr).                ║\n"
                "║  [S] Sí, actualizar todo   [N] No, continuar    ║\n"
                "╚══════════════════════════════════════════════════╝")

    def ciclo_cierre(self) -> Dict[str, Any]:
        reporte = {"sesion": self.sesion_id}
        reporte["refactor"] = self.ejecutar_refactor(mostrar_barra=False)
        reporte["confirmacion"] = self.confirmar_actualizacion()
        if "refactorizados listos" in reporte["confirmacion"]:
            reporte["actualizacion"] = self.actualizar_sistema(confirmar=True)
        else:
            reporte["actualizacion"] = {"exito": True, "mensaje": "Sin cambios que aplicar."}
        reporte["exito"] = True
        return reporte

    def listar_subsistemas(self) -> List[Dict[str, Any]]:
        """Devuelve los subsistemas generados en esta sesion."""
        return list(self._generados)

    def resumen_sesion(self) -> str:
        """Resumen factual de la sesion HRCNTR."""
        n = len(self._generados)
        if n == 0:
            return f"Sesion {self.sesion_id}: sin subsistemas generados."
        nombres = [g["clase"] for g in self._generados]
        return (f"Sesion {self.sesion_id}: {n} subsistema(s) generados "
                f"({', '.join(nombres[:3])}"
                f"{'...' if len(nombres) > 3 else ''}).")

    def estado(self) -> Dict[str, Any]:
        return {"version": __version__, "sesion": self.sesion_id,
                "activa": False, "actualizado": self._actualizado,
                "monitoreo": {"archivos": len(self._inventario),
                               "neuronas": len([e for e in self._inventario
                                                   if "red_neuronal" in e["ruta"]
                                                   or "RF_" in e["ruta"]]),
                               "oversized": len([e for e in self._inventario
                                                  if e["sobreescrito"]])},
                "seleccionados": len(self._seleccionados),
                "generados": len(self._generados),
                "subsistemas": [g["clase"] for g in self._generados]}


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
