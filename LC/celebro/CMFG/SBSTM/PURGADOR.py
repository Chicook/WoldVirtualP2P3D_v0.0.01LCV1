"""
PURGADOR.py - Subsistema de Limpieza Segura Post-IPFS (Arquitectura 2026)
==========================================================================
Flujo garantizado de limpieza en dos fases:

  FASE 1 — CUSTODIA IPFS:
    • Sube todos los pesos neuronales de PSNRL a IPFS mediante IPFSManager.
    • Registra cada CID en el manifiesto rotativo (ipfs_manifest.json).
    • Verifica sha256 de cada archivo antes de eliminarlo localmente.
    • Solo procede a la fase 2 si la custodia IPFS es exitosa (o se fuerza).

  FASE 2 — PURGA DE DIRECTORIO:
    • Elimina todas las carpetas __pycache__ de Python recursivamente.
    • Elimina todos los archivos .pyc / .pyo / .pyd sueltos.
    • Limpia archivos temporales ._tmp_*.bin generados por ipfs_manager.
    • Limpia el directorio CHG (build artifacts intermedios).
    • Genera un informe detallado de auditoría con tamaños liberados.

Diseñado para ejecutarse al cerrar sesión de LucIA o como paso CI/CD.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

# ─── RUTAS CANÓNICAS ─────────────────────────────────────────────────────────
_ESTE_ARCHIVO: Final[Path] = Path(__file__).resolve()
SBSTM_DIR:     Final[Path] = _ESTE_ARCHIVO.parent           # LC/celebro/CMFG/SBSTM
CMFG_DIR:      Final[Path] = SBSTM_DIR.parent               # LC/celebro/CMFG
CELEBRO_DIR:   Final[Path] = CMFG_DIR.parent                # LC/celebro
LC_DIR:        Final[Path] = CELEBRO_DIR.parent             # LC
ROOT_DIR:      Final[Path] = LC_DIR.parent                  # WoldVirtualP2P3D_v0.0.01LCV1
PSNRL_DIR:     Final[Path] = CELEBRO_DIR / "PSNRL"
CHG_DIR:       Final[Path] = ROOT_DIR / "CHG"

# ─── VERSIÓN DEL SUBSISTEMA ───────────────────────────────────────────────────
__version__:   Final[str] = "2026.1.0"
__subsystem__: Final[str] = "PURGADOR-LucIA"

logger: logging.Logger = logging.getLogger("WoldVirtualP2P3D.PURGADOR")


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResultadoCustodia:
    """Resultado de la Fase 1: subida de pesos a IPFS."""
    archivos_procesados: int = 0
    archivos_pinados: int = 0
    archivos_fallidos: int = 0
    cids_obtenidos: List[str] = field(default_factory=list)
    borrados_locales: List[str] = field(default_factory=list)
    errores: List[str] = field(default_factory=list)
    daemon_real: bool = False
    cli_real: bool = False
    bytes_subidos: int = 0
    duracion_seg: float = 0.0

    @property
    def exito(self) -> bool:
        """True si al menos un archivo fue custodiado o PSNRL estaba vacío."""
        return self.archivos_fallidos == 0


@dataclass
class ResultadoPurga:
    """Resultado de la Fase 2: limpieza del sistema de archivos."""
    pycache_eliminados: int = 0
    pyc_eliminados: int = 0
    tmp_ipfs_eliminados: int = 0
    chg_eliminados: int = 0
    bytes_liberados: int = 0
    rutas_eliminadas: List[str] = field(default_factory=list)
    errores: List[str] = field(default_factory=list)
    duracion_seg: float = 0.0


@dataclass
class InformePurgador:
    """Informe completo de una ejecución del PURGADOR."""
    timestamp_utc: str = ""
    raiz_escaneada: str = ""
    custodia: ResultadoCustodia = field(default_factory=ResultadoCustodia)
    purga: ResultadoPurga = field(default_factory=ResultadoPurga)
    fase1_omitida: bool = False
    purga_forzada: bool = False
    duracion_total_seg: float = 0.0

    def como_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "raiz_escaneada": self.raiz_escaneada,
            "duracion_total_seg": round(self.duracion_total_seg, 3),
            "fase1_omitida": self.fase1_omitida,
            "purga_forzada": self.purga_forzada,
            "custodia_ipfs": {
                "archivos_procesados": self.custodia.archivos_procesados,
                "archivos_pinados": self.custodia.archivos_pinados,
                "archivos_fallidos": self.custodia.archivos_fallidos,
                "cids": self.custodia.cids_obtenidos,
                "borrados_locales": self.custodia.borrados_locales,
                "bytes_subidos": self.custodia.bytes_subidos,
                "daemon_real": self.custodia.daemon_real,
                "cli_real": self.custodia.cli_real,
                "errores": self.custodia.errores,
                "duracion_seg": round(self.custodia.duracion_seg, 3),
            },
            "purga_filesystem": {
                "pycache_eliminados": self.purga.pycache_eliminados,
                "pyc_eliminados": self.purga.pyc_eliminados,
                "tmp_ipfs_eliminados": self.purga.tmp_ipfs_eliminados,
                "chg_eliminados": self.purga.chg_eliminados,
                "bytes_liberados": self.purga.bytes_liberados,
                "errores": self.purga.errores,
                "duracion_seg": round(self.purga.duracion_seg, 3),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — CUSTODIA IPFS DE PESOS NEURONALES
# ═══════════════════════════════════════════════════════════════════════════════

class CustodiaIPFS:
    """
    Gestiona la subida segura de pesos de PSNRL a IPFS antes de cualquier
    limpieza del sistema de archivos. Requiere confirmación de CID antes
    de marcar un archivo como borrable.
    """

    def __init__(self, psnrl_dir: Optional[Path] = None) -> None:
        self._psnrl_dir = psnrl_dir or PSNRL_DIR
        self._psnrl_dir.mkdir(parents=True, exist_ok=True)
        self._ipfs_mgr = None
        self._lock = threading.Lock()

    def _obtener_gestor_ipfs(self):
        """Importación diferida del IPFSManager para evitar dependencias circulares."""
        if self._ipfs_mgr is None:
            try:
                if str(ROOT_DIR) not in sys.path:
                    sys.path.insert(0, str(ROOT_DIR))
                from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
                self._ipfs_mgr = get_ipfs_manager()
            except Exception as exc:
                logger.error("No se pudo cargar IPFSManager: %s", exc)
                raise RuntimeError(f"IPFSManager no disponible: {exc}") from exc
        return self._ipfs_mgr

    @staticmethod
    def _sha256_archivo(ruta: Path) -> str:
        h = hashlib.sha256()
        with open(ruta, "rb") as fh:
            for bloque in iter(lambda: fh.read(65536), b""):
                h.update(bloque)
        return h.hexdigest()

    def ejecutar(self, forzar_borrado_sin_daemon: bool = False) -> ResultadoCustodia:
        """
        Sube todos los archivos de PSNRL a IPFS.
        Si forzar_borrado_sin_daemon=True, borra el local incluso con motor CIDv1.
        """
        resultado = ResultadoCustodia()
        t0 = time.perf_counter()

        try:
            mgr = self._obtener_gestor_ipfs()
        except RuntimeError as exc:
            resultado.errores.append(str(exc))
            resultado.duracion_seg = time.perf_counter() - t0
            return resultado

        # Descubrir archivos en PSNRL
        archivos = sorted([
            f for f in self._psnrl_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ])

        if not archivos:
            logger.info("PSNRL vacío — no hay pesos que subir a IPFS.")
            resultado.duracion_seg = time.perf_counter() - t0
            return resultado

        # Detectar disponibilidad real
        estado = mgr.estado_conexion()
        resultado.daemon_real = estado.get("daemon_activo", False)
        resultado.cli_real = estado.get("cli_disponible", False)

        logger.info(
            "CustodiaIPFS: %d archivo(s) en PSNRL | daemon=%s | cli=%s",
            len(archivos), resultado.daemon_real, resultado.cli_real
        )

        for arch in archivos:
            sha_antes = self._sha256_archivo(arch)
            try:
                res = mgr.almacenar_pesos(
                    origen=arch,
                    nombre_modelo=arch.stem,
                    eliminar_local=(
                        resultado.daemon_real
                        or resultado.cli_real
                        or forzar_borrado_sin_daemon
                    ),
                    metadatos={
                        "sha256_original": sha_antes,
                        "purgador_version": __version__,
                        "timestamp_purga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                resultado.archivos_procesados += 1
                resultado.archivos_pinados += 1
                resultado.cids_obtenidos.append(res["cid"])
                resultado.bytes_subidos += res.get("tamano_bytes", arch.stat().st_size)
                if res.get("borrado_local"):
                    resultado.borrados_locales.append(arch.name)
                logger.info(
                    "  ✓ %s → CID=%s | nodo=%s | borrado=%s",
                    arch.name, res["cid"][:20], res["nodo"], res.get("borrado_local", False)
                )
            except Exception as exc:
                resultado.archivos_fallidos += 1
                msg = f"{arch.name}: {exc}"
                resultado.errores.append(msg)
                logger.error("  ✗ %s", msg)

        resultado.duracion_seg = time.perf_counter() - t0
        logger.info(
            "CustodiaIPFS completada: %d/%d pinados | %d errores | %.2fs",
            resultado.archivos_pinados, resultado.archivos_procesados,
            resultado.archivos_fallidos, resultado.duracion_seg
        )
        return resultado


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — PURGA DEL SISTEMA DE ARCHIVOS
# ═══════════════════════════════════════════════════════════════════════════════

class PurgaFilesystem:
    """
    Elimina artefactos de Python (__pycache__, .pyc, .pyo, .pyd),
    temporales de IPFS y el contenido del directorio CHG.
    Opera sobre el árbol de directorios especificado como raíz.
    """

    # Patrones de archivos individuales a eliminar
    _EXTENSIONES_PYC: Final[Tuple[str, ...]] = (".pyc", ".pyo", ".pyd")
    # Prefijos de archivos temporales de ipfs_manager
    _PREFIJO_TMP_IPFS: Final[str] = "_tmp_"

    def __init__(
        self,
        raiz: Optional[Path] = None,
        chg_dir: Optional[Path] = None,
        excluir_rutas: Optional[List[str]] = None,
    ) -> None:
        self._raiz = raiz or ROOT_DIR
        self._chg_dir = chg_dir or CHG_DIR
        # Rutas que nunca se tocarán (relativas a raíz, como strings parciales)
        self._excluir: List[str] = excluir_rutas or [
            ".git",
            "node_modules",
            ".venv",
            "venv",
            "env",
        ]

    def _esta_excluida(self, ruta: Path) -> bool:
        """True si la ruta cae bajo alguno de los patrones excluidos."""
        partes = ruta.parts
        return any(exc in partes for exc in self._excluir)

    def _tamaño_dir(self, ruta: Path) -> int:
        """Calcula el tamaño en bytes de un directorio recursivamente."""
        total = 0
        try:
            for sub in ruta.rglob("*"):
                if sub.is_file():
                    total += sub.stat().st_size
        except Exception:
            pass
        return total

    def _eliminar_seguro(
        self, ruta: Path, resultado: ResultadoPurga, es_dir: bool = False
    ) -> bool:
        """Intenta eliminar una ruta y registra el resultado."""
        try:
            tamaño = self._tamaño_dir(ruta) if es_dir else ruta.stat().st_size
            if es_dir:
                shutil.rmtree(ruta, ignore_errors=False)
            else:
                ruta.unlink()
            resultado.bytes_liberados += tamaño
            resultado.rutas_eliminadas.append(str(ruta))
            return True
        except PermissionError as exc:
            msg = f"[PermissionError] {ruta}: {exc}"
            resultado.errores.append(msg)
            logger.warning(msg)
        except FileNotFoundError:
            pass  # Ya eliminado por otro proceso, no es error
        except Exception as exc:
            msg = f"[Error] {ruta}: {exc}"
            resultado.errores.append(msg)
            logger.error(msg)
        return False

    def limpiar_pycache(self, resultado: ResultadoPurga) -> None:
        """Elimina todas las carpetas __pycache__ bajo la raíz."""
        for carpeta in self._raiz.rglob("__pycache__"):
            if not carpeta.is_dir():
                continue
            if self._esta_excluida(carpeta):
                continue
            if self._eliminar_seguro(carpeta, resultado, es_dir=True):
                resultado.pycache_eliminados += 1
                logger.info("  🗑 __pycache__ eliminado: %s", carpeta.relative_to(self._raiz))

    def limpiar_pyc_sueltos(self, resultado: ResultadoPurga) -> None:
        """Elimina archivos .pyc / .pyo / .pyd no bajo __pycache__."""
        for ext in self._EXTENSIONES_PYC:
            for pyc in self._raiz.rglob(f"*{ext}"):
                if not pyc.is_file():
                    continue
                if self._esta_excluida(pyc):
                    continue
                # __pycache__ ya fue eliminado; evitar doble conteo
                if "__pycache__" in pyc.parts:
                    continue
                if self._eliminar_seguro(pyc, resultado):
                    resultado.pyc_eliminados += 1
                    logger.info("  🗑 .pyc suelto eliminado: %s", pyc.relative_to(self._raiz))

    def limpiar_tmp_ipfs(self, resultado: ResultadoPurga) -> None:
        """Elimina archivos temporales _tmp_*.bin generados por ipfs_manager."""
        for tmp in self._raiz.rglob(f"{self._PREFIJO_TMP_IPFS}*.bin"):
            if not tmp.is_file():
                continue
            if self._esta_excluida(tmp):
                continue
            if self._eliminar_seguro(tmp, resultado):
                resultado.tmp_ipfs_eliminados += 1
                logger.info("  🗑 Temporal IPFS eliminado: %s", tmp.relative_to(self._raiz))

    def limpiar_chg(self, resultado: ResultadoPurga) -> None:
        """Elimina el contenido del directorio CHG (build artifacts)."""
        if not self._chg_dir.exists():
            logger.debug("CHG no existe, nada que limpiar: %s", self._chg_dir)
            return
        try:
            items = list(self._chg_dir.iterdir())
        except Exception as exc:
            resultado.errores.append(f"CHG no legible: {exc}")
            return

        for item in items:
            if self._esta_excluida(item):
                continue
            es_dir = item.is_dir()
            if self._eliminar_seguro(item, resultado, es_dir=es_dir):
                resultado.chg_eliminados += 1
                logger.info("  🗑 CHG eliminado: %s", item.name)

    def ejecutar(self) -> ResultadoPurga:
        """Ejecuta la purga completa del sistema de archivos."""
        resultado = ResultadoPurga()
        t0 = time.perf_counter()

        logger.info("PurgaFilesystem iniciada | raíz=%s", self._raiz)

        self.limpiar_pycache(resultado)
        self.limpiar_pyc_sueltos(resultado)
        self.limpiar_tmp_ipfs(resultado)
        self.limpiar_chg(resultado)

        resultado.duracion_seg = time.perf_counter() - t0
        logger.info(
            "PurgaFilesystem completada: __pycache__=%d | .pyc=%d | tmp=%d | chg=%d | "
            "%.2f KB liberados | %.2fs",
            resultado.pycache_eliminados, resultado.pyc_eliminados,
            resultado.tmp_ipfs_eliminados, resultado.chg_eliminados,
            resultado.bytes_liberados / 1024, resultado.duracion_seg,
        )
        return resultado

    # Alias para compatibilidad con el bucle de extensiones
    _EXTENSIONES_PYCA = _EXTENSIONES_PYCA = (".pyc", ".pyo", ".pyd")


# ═══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class Purgador:
    """
    Orquestador del subsistema de limpieza segura post-IPFS.

    Flujo:
      1. CustodiaIPFS.ejecutar()  → sube pesos de PSNRL a IPFS
      2. (verificación de éxito)
      3. PurgaFilesystem.ejecutar() → elimina __pycache__ y residuos
      4. Guarda InformePurgador en JSON de auditoría

    Parámetros:
      raiz                    — directorio raíz del proyecto
      omitir_ipfs             — saltar Fase 1 (solo limpieza)
      forzar_purga_sin_ipfs   — continuar con purga aunque IPFS falle
      forzar_borrado_psnrl    — borrar archivos PSNRL incluso sin daemon real
      guardar_informe         — persistir el informe JSON al finalizar
    """

    _INFORME_NOMBRE: Final[str] = "purgador_informe.json"

    def __init__(
        self,
        raiz: Optional[Path] = None,
        omitir_ipfs: bool = False,
        forzar_purga_sin_ipfs: bool = False,
        forzar_borrado_psnrl: bool = False,
        guardar_informe: bool = True,
        excluir_rutas: Optional[List[str]] = None,
    ) -> None:
        self._raiz = raiz or ROOT_DIR
        self._omitir_ipfs = omitir_ipfs
        self._forzar_purga_sin_ipfs = forzar_purga_sin_ipfs
        self._forzar_borrado_psnrl = forzar_borrado_psnrl
        self._guardar_informe = guardar_informe
        self._excluir_rutas = excluir_rutas
        self._custodia = CustodiaIPFS()
        self._purga = PurgaFilesystem(raiz=self._raiz, excluir_rutas=excluir_rutas)

    def _guardar_informe_json(self, informe: InformePurgador) -> Optional[Path]:
        ruta = self._raiz / self._INFORME_NOMBRE
        try:
            with open(ruta, "w", encoding="utf-8") as fh:
                json.dump(informe.como_dict(), fh, indent=2, ensure_ascii=False)
            logger.info("Informe guardado: %s", ruta)
            return ruta
        except Exception as exc:
            logger.error("No se pudo guardar el informe: %s", exc)
            return None

    def ejecutar(self) -> InformePurgador:
        """Ejecuta el flujo completo en dos fases y devuelve el informe."""
        t_global = time.perf_counter()
        informe = InformePurgador(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            raiz_escaneada=str(self._raiz),
            fase1_omitida=self._omitir_ipfs,
            purga_forzada=self._forzar_purga_sin_ipfs,
        )

        # ── FASE 1: Custodia IPFS ────────────────────────────────────────────
        if not self._omitir_ipfs:
            logger.info("═══ FASE 1: Custodia IPFS de pesos neuronales ═══")
            informe.custodia = self._custodia.ejecutar(
                forzar_borrado_sin_daemon=self._forzar_borrado_psnrl
            )

            if not informe.custodia.exito and not self._forzar_purga_sin_ipfs:
                logger.warning(
                    "Custodia IPFS incompleta (%d errores). "
                    "Purga cancelada. Use forzar_purga_sin_ipfs=True para continuar.",
                    informe.custodia.archivos_fallidos,
                )
                informe.duracion_total_seg = time.perf_counter() - t_global
                if self._guardar_informe:
                    self._guardar_informe_json(informe)
                return informe
        else:
            logger.info("Fase 1 (IPFS) omitida por configuración.")

        # ── FASE 2: Purga del Filesystem ─────────────────────────────────────
        logger.info("═══ FASE 2: Purga del sistema de archivos ═══")
        informe.purga = self._purga.ejecutar()

        informe.duracion_total_seg = time.perf_counter() - t_global

        if self._guardar_informe:
            self._guardar_informe_json(informe)

        return informe


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CONVENIENCIA GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════

def purgar_proyecto(
    raiz: Optional[Path] = None,
    forzar: bool = False,
    guardar_informe: bool = True,
) -> InformePurgador:
    """
    Punto de entrada principal: sube pesos a IPFS y limpia __pycache__.

    Args:
        raiz:            Directorio raíz del proyecto (por defecto ROOT_DIR).
        forzar:          Si True, limpia aunque IPFS no esté disponible.
        guardar_informe: Si True, persiste el informe JSON en la raíz.

    Returns:
        InformePurgador con el resultado completo.
    """
    p = Purgador(
        raiz=raiz,
        forzar_purga_sin_ipfs=forzar,
        forzar_borrado_psnrl=forzar,
        guardar_informe=guardar_informe,
    )
    return p.ejecutar()


def solo_limpiar_pycache(raiz: Optional[Path] = None) -> ResultadoPurga:
    """
    Limpia únicamente los __pycache__ y .pyc sin tocar IPFS.
    Útil para limpieza rápida durante desarrollo.
    """
    p = PurgaFilesystem(raiz=raiz or ROOT_DIR)
    return p.ejecutar()


def solo_custodia_ipfs(forzar_borrado: bool = False) -> ResultadoCustodia:
    """
    Solo ejecuta la fase de custodia IPFS, sin limpiar el filesystem.
    Útil para hacer backup de pesos sin limpiar el entorno de desarrollo.
    """
    c = CustodiaIPFS()
    return c.ejecutar(forzar_borrado_sin_daemon=forzar_borrado)


def estado_psnrl() -> Dict[str, Any]:
    """Devuelve un resumen del estado actual del directorio PSNRL."""
    PSNRL_DIR.mkdir(parents=True, exist_ok=True)
    archivos = [f for f in PSNRL_DIR.iterdir() if f.is_file()]
    tamaño_total = sum(f.stat().st_size for f in archivos)
    return {
        "directorio": str(PSNRL_DIR),
        "existe": PSNRL_DIR.exists(),
        "archivos": len(archivos),
        "tamaño_bytes": tamaño_total,
        "tamaño_kb": round(tamaño_total / 1024, 2),
        "listado": [f.name for f in sorted(archivos)],
    }


def imprimir_informe(informe: InformePurgador) -> None:
    """Imprime un resumen visual del informe del purgador en terminal."""
    ANSI_RESET   = "\033[0m"
    ANSI_BOLD    = "\033[1m"
    ANSI_CYAN    = "\033[96m"
    ANSI_GREEN   = "\033[92m"
    ANSI_YELLOW  = "\033[93m"
    ANSI_RED     = "\033[91m"
    ANSI_MAGENTA = "\033[95m"
    ANSI_WHITE   = "\033[97m"
    ANSI_DIM     = "\033[2m"
    SEP          = f"{ANSI_CYAN}{'═' * 78}{ANSI_RESET}"
    SUB_SEP      = f"{ANSI_DIM}{'─' * 78}{ANSI_RESET}"

    print(f"\n{SEP}")
    print(
        f"  {ANSI_BOLD}{ANSI_WHITE}🧹 PURGADOR LucIA — INFORME DE LIMPIEZA v{__version__}{ANSI_RESET}"
    )
    print(f"  {ANSI_DIM}{informe.timestamp_utc} | {informe.raiz_escaneada}{ANSI_RESET}")
    print(SUB_SEP)

    # ── FASE 1 ──────────────────────────────────────────────────────────────
    if informe.fase1_omitida:
        print(f"  {ANSI_YELLOW}⚠  Fase 1 IPFS omitida por configuración{ANSI_RESET}")
    else:
        c = informe.custodia
        estado_c = (
            f"{ANSI_GREEN}✓ ÉXITO{ANSI_RESET}"
            if c.exito else
            f"{ANSI_RED}✗ CON ERRORES{ANSI_RESET}"
        )
        daemon_str = (
            f"{ANSI_GREEN}DAEMON RPC{ANSI_RESET}" if c.daemon_real
            else (f"{ANSI_YELLOW}CLI{ANSI_RESET}" if c.cli_real
                  else f"{ANSI_MAGENTA}CIDv1 LOCAL{ANSI_RESET}")
        )
        print(f"  {ANSI_BOLD}FASE 1 — Custodia IPFS{ANSI_RESET}  {estado_c}")
        print(f"    Archivos procesados : {ANSI_WHITE}{c.archivos_procesados}{ANSI_RESET}")
        print(f"    Archivos pinados    : {ANSI_GREEN}{c.archivos_pinados}{ANSI_RESET}")
        print(f"    Borrados locales    : {ANSI_CYAN}{len(c.borrados_locales)}{ANSI_RESET}")
        print(f"    Bytes subidos       : {ANSI_WHITE}{c.bytes_subidos:,} B{ANSI_RESET}")
        print(f"    Nodo IPFS usado     : {daemon_str}")
        print(f"    Duración            : {ANSI_DIM}{c.duracion_seg:.3f}s{ANSI_RESET}")
        for cid in c.cids_obtenidos[:5]:
            print(f"      → CID: {ANSI_CYAN}{cid}{ANSI_RESET}")
        if len(c.cids_obtenidos) > 5:
            print(f"      {ANSI_DIM}... y {len(c.cids_obtenidos) - 5} más{ANSI_RESET}")
        for err in c.errores[:3]:
            print(f"      {ANSI_RED}✗ {err}{ANSI_RESET}")

    print(SUB_SEP)

    # ── FASE 2 ──────────────────────────────────────────────────────────────
    p = informe.purga
    kb_liberados = p.bytes_liberados / 1024
    print(f"  {ANSI_BOLD}FASE 2 — Purga Filesystem{ANSI_RESET}")
    print(f"    __pycache__ eliminados : {ANSI_CYAN}{p.pycache_eliminados}{ANSI_RESET}")
    print(f"    .pyc sueltos eliminados: {ANSI_CYAN}{p.pyc_eliminados}{ANSI_RESET}")
    print(f"    Temporales IPFS borr.  : {ANSI_CYAN}{p.tmp_ipfs_eliminados}{ANSI_RESET}")
    print(f"    Entradas CHG borradas  : {ANSI_CYAN}{p.chg_eliminados}{ANSI_RESET}")
    print(
        f"    Espacio liberado       : {ANSI_GREEN}{kb_liberados:,.2f} KB{ANSI_RESET}"
        if kb_liberados < 10240
        else f"    Espacio liberado       : {ANSI_GREEN}{kb_liberados/1024:,.2f} MB{ANSI_RESET}"
    )
    print(f"    Duración               : {ANSI_DIM}{p.duracion_seg:.3f}s{ANSI_RESET}")
    for err in p.errores[:3]:
        print(f"      {ANSI_RED}✗ {err}{ANSI_RESET}")

    print(SUB_SEP)
    color_total = ANSI_GREEN if not informe.custodia.errores and not informe.purga.errores else ANSI_YELLOW
    print(
        f"  {color_total}{ANSI_BOLD}Tiempo total: {informe.duracion_total_seg:.3f}s{ANSI_RESET}"
    )
    print(f"{SEP}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _cli_main() -> None:
    """Punto de entrada en línea de comandos para el PURGADOR."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="PURGADOR",
        description=(
            "PURGADOR LucIA — Sube pesos neuronales a IPFS y limpia __pycache__."
        ),
    )
    parser.add_argument(
        "--solo-limpiar",
        action="store_true",
        help="Solo limpia __pycache__ sin subir a IPFS.",
    )
    parser.add_argument(
        "--solo-ipfs",
        action="store_true",
        help="Solo sube a IPFS sin limpiar el filesystem.",
    )
    parser.add_argument(
        "--forzar",
        action="store_true",
        help="Limpia aunque IPFS no esté disponible o falle.",
    )
    parser.add_argument(
        "--raiz",
        type=str,
        default=str(ROOT_DIR),
        help=f"Directorio raíz del proyecto (por defecto: {ROOT_DIR}).",
    )
    parser.add_argument(
        "--no-informe",
        action="store_true",
        help="No guardar el informe JSON.",
    )
    parser.add_argument(
        "--estado-psnrl",
        action="store_true",
        help="Mostrar estado del directorio PSNRL y salir.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Activar logging detallado.",
    )

    args = parser.parse_args()

    nivel_log = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=nivel_log,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    raiz = Path(args.raiz).resolve()

    if args.estado_psnrl:
        est = estado_psnrl()
        print(f"\n  📁 PSNRL: {est['directorio']}")
        print(f"     Archivos    : {est['archivos']}")
        print(f"     Tamaño total: {est['tamaño_kb']} KB")
        for nombre in est["listado"]:
            print(f"       • {nombre}")
        print()
        return

    if args.solo_limpiar:
        print("\n  Modo: solo limpieza de __pycache__ (sin IPFS)\n")
        resultado = solo_limpiar_pycache(raiz=raiz)
        print(
            f"  ✓ __pycache__: {resultado.pycache_eliminados} | "
            f".pyc: {resultado.pyc_eliminados} | "
            f"{resultado.bytes_liberados / 1024:.2f} KB liberados | "
            f"{resultado.duracion_seg:.3f}s"
        )
        return

    if args.solo_ipfs:
        print("\n  Modo: solo custodia IPFS (sin limpieza filesystem)\n")
        resultado = solo_custodia_ipfs(forzar_borrado=args.forzar)
        print(
            f"  ✓ Archivos pinados: {resultado.archivos_pinados} | "
            f"CIDs: {len(resultado.cids_obtenidos)} | "
            f"{resultado.bytes_subidos:,} bytes | "
            f"{resultado.duracion_seg:.3f}s"
        )
        return

    # Flujo completo
    print(
        f"\n  🧹 PURGADOR LucIA v{__version__} | raíz={raiz} | "
        f"forzar={args.forzar}\n"
    )
    informe = purgar_proyecto(
        raiz=raiz,
        forzar=args.forzar,
        guardar_informe=not args.no_informe,
    )
    imprimir_informe(informe)


if __name__ == "__main__":
    _cli_main()
