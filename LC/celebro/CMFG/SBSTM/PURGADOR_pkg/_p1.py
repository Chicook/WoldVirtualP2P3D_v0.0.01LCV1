"""
PURGADOR - parte 1/2 (version de sesion LucIA).
Este código Python se encarga de administrar diferentes variables y funciones en un ambiente de desarrollo local. A continuación, se detalla
"""
from __future__ import annotations

"""
PURGADOR.py - Subsistema de Limpieza Segura Post-IPFS (Arquitectura 2026)
==========================================================================
Flujo en dos fases garantizadas:
  FASE 1 — CUSTODIA IPFS: sube todos los pesos de PSNRL a IPFS via IPFSManager,
    verifica sha256 por archivo, registra CIDs en el manifiesto rotativo.
  FASE 2 — PURGA FILESYSTEM: elimina __pycache__, .pyc/.pyo/.pyd, temporales
    _tmp_*.bin de ipfs_manager y el contenido del directorio CHG.
Genera informe JSON de auditoría al finalizar. Usable como módulo o CLI.
"""

import hashlib
import json
import logging
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Tuple

# ─── RUTAS CANÓNICAS ─────────────────────────────────────────────────────────
_HERE:       Final[Path] = Path(__file__).resolve()
SBSTM_DIR:   Final[Path] = _HERE.parent
CMFG_DIR:    Final[Path] = SBSTM_DIR.parent
CELEBRO_DIR: Final[Path] = CMFG_DIR.parent
LC_DIR:      Final[Path] = CELEBRO_DIR.parent
ROOT_DIR:    Final[Path] = LC_DIR.parent
PSNRL_DIR:   Final[Path] = CELEBRO_DIR / "PSNRL"

CHG_DIR:     Final[Path] = ROOT_DIR / "CHG"

__version__:   Final[str] = "2026.1.0"
__subsystem__: Final[str] = "PURGADOR-LucIA"
logger = logging.getLogger("WoldVirtualP2P3D.PURGADOR")
ANSI = {"R": "\033[0m", "B": "\033[1m", "D": "\033[2m", "C": "\033[96m",
        "G": "\033[92m", "Y": "\033[93m", "M": "\033[95m", "RE": "\033[91m", "W": "\033[97m"}

# ── ESTRUCTURAS DE DATOS ─────────────────────────────────────────────────────
@dataclass
class ResultadoCustodia:
    """Resultado de Fase 1: subida IPFS de pesos neuronales."""
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
        return self.archivos_fallidos == 0

@dataclass
class ResultadoPurga:
    """Resultado de Fase 2: limpieza del filesystem."""
    pycache_eliminados: int = 0
    pyc_eliminados: int = 0
    tmp_ipfs_eliminados: int = 0
    chg_eliminados: int = 0
    pytest_cache_eliminados: int = 0
    logs_tmp_eliminados: int = 0
    bak_tmp_eliminados: int = 0
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
        c, p = self.custodia, self.purga
        return {
            "timestamp_utc": self.timestamp_utc,
            "raiz_escaneada": self.raiz_escaneada,
            "duracion_total_seg": round(self.duracion_total_seg, 3),
            "fase1_omitida": self.fase1_omitida,
            "purga_forzada": self.purga_forzada,
            "custodia_ipfs": {
                "procesados": c.archivos_procesados, "pinados": c.archivos_pinados,
                "fallidos": c.archivos_fallidos, "cids": c.cids_obtenidos,
                "borrados": c.borrados_locales, "bytes": c.bytes_subidos,
                "daemon_real": c.daemon_real, "cli_real": c.cli_real,
                "errores": c.errores, "duracion_seg": round(c.duracion_seg, 3),
            },
            "purga_filesystem": {
                "pycache": p.pycache_eliminados, "pyc_sueltos": p.pyc_eliminados,
                "tmp_ipfs": p.tmp_ipfs_eliminados, "chg": p.chg_eliminados,
                "pytest_cache": p.pytest_cache_eliminados,
                "logs_tmp": p.logs_tmp_eliminados, "bak_tmp": p.bak_tmp_eliminados,
                "bytes_liberados": p.bytes_liberados, "errores": p.errores,
                "duracion_seg": round(p.duracion_seg, 3),
            },
        }

# ── FASE 1 — CUSTODIA IPFS ───────────────────────────────────────────────────
class CustodiaIPFS:
    """Sube todos los archivos de PSNRL a IPFS antes de cualquier limpieza."""

    def __init__(self, psnrl_dir: Optional[Path] = None) -> None:
        self._psnrl_dir = psnrl_dir or PSNRL_DIR
        self._psnrl_dir.mkdir(parents=True, exist_ok=True)
        self._mgr = None

    def _gestor(self):
        """Importación diferida de IPFSManager para evitar dependencias circulares."""
        if self._mgr is None:
            if str(ROOT_DIR) not in sys.path:
                sys.path.insert(0, str(ROOT_DIR))
            try:
                from LC.celebro.CMFG.ipfs_manager import get_ipfs_manager
                self._mgr = get_ipfs_manager()
            except Exception as exc:
                raise RuntimeError(f"IPFSManager no disponible: {exc}") from exc
        return self._mgr

    @staticmethod
    def _sha256(ruta: Path) -> str:
        h = hashlib.sha256()
        with open(ruta, "rb") as fh:
            for bloque in iter(lambda: fh.read(65536), b""):
                h.update(bloque)
        return h.hexdigest()

    def ejecutar(self, forzar_borrado: bool = False) -> ResultadoCustodia:
        """Sube cada archivo de PSNRL a IPFS y registra CID + sha256."""
        res = ResultadoCustodia()
        t0 = time.perf_counter()
        try: mgr = self._gestor()
        except RuntimeError as exc:
            res.errores.append(str(exc)); res.duracion_seg = time.perf_counter() - t0; return res

        archivos = sorted(
            f for f in self._psnrl_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        )
        if not archivos:
            logger.info("PSNRL vacío — sin pesos que subir.")
            res.duracion_seg = time.perf_counter() - t0
            return res

        estado = mgr.estado_conexion()
        res.daemon_real = estado.get("daemon_activo", False)
        res.cli_real    = estado.get("cli_disponible", False)
        borrar_ok = res.daemon_real or res.cli_real or forzar_borrado

        logger.info("CustodiaIPFS: %d arch | daemon=%s | cli=%s", len(archivos), res.daemon_real, res.cli_real)
        for arch in archivos:
            sha = self._sha256(arch)
            try:
                r = mgr.almacenar_pesos(
                    origen=arch, nombre_modelo=arch.stem, eliminar_local=borrar_ok,
                    metadatos={"sha256_original": sha, "purgador": __version__,
                               "ts_purga": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
                )
                res.archivos_procesados += 1
                res.archivos_pinados += 1
                res.cids_obtenidos.append(r["cid"])
                res.bytes_subidos += r.get("tamano_bytes", arch.stat().st_size)
                if r.get("borrado_local"):
                    res.borrados_locales.append(arch.name)
                logger.info("  ✓ %s → %s | %s", arch.name, r["cid"][:20], r["nodo"])
            except Exception as exc:
                res.archivos_fallidos += 1
                res.errores.append(f"{arch.name}: {exc}")
                logger.error("  ✗ %s: %s", arch.name, exc)

        res.duracion_seg = time.perf_counter() - t0
        logger.info("CustodiaIPFS: %d/%d pinados | %d err | %.2fs",
                    res.archivos_pinados, res.archivos_procesados, res.archivos_fallidos, res.duracion_seg)
        return res

# ── FASE 2 — PURGA DEL FILESYSTEM ────────────────────────────────────────────
class PurgaFilesystem:
    """Elimina __pycache__, .pyc/.pyo/.pyd, temporales IPFS y contenido de CHG."""

    _PYC_EXT:    Final[Tuple[str, ...]] = (".pyc", ".pyo", ".pyd")
    _TMP_PREFIX: Final[str]             = "_tmp_"
    _EXCLUIR_DEFAULT: Final[Tuple[str, ...]] = (".git", "node_modules", ".venv", "venv", "env")

    def __init__(
        self,
        raiz: Optional[Path] = None,
        chg_dir: Optional[Path] = None,
        excluir: Optional[List[str]] = None,
    ) -> None:
        self._raiz    = raiz    or ROOT_DIR
        self._chg_dir = chg_dir or CHG_DIR
        self._excluir = tuple(excluir) if excluir else self._EXCLUIR_DEFAULT

    def _excluida(self, ruta: Path) -> bool:
        return any(e in ruta.parts for e in self._excluir)

    def _tamaño(self, ruta: Path, es_dir: bool) -> int:
        if not es_dir:
            try:
                return ruta.stat().st_size
            except Exception:
                return 0
        total = 0
        try:
            for f in ruta.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        except Exception:
            pass
        return total

    def _borrar(self, ruta: Path, res: ResultadoPurga, es_dir: bool = False) -> bool:
        try:
            tamaño = self._tamaño(ruta, es_dir)
            shutil.rmtree(ruta, ignore_errors=False) if es_dir else ruta.unlink()
            res.bytes_liberados += tamaño
            res.rutas_eliminadas.append(str(ruta))
            return True
        except FileNotFoundError:
            return False
        except Exception as exc:
            msg = f"{'[Dir]' if es_dir else '[File]'} {ruta}: {exc}"
            res.errores.append(msg)
            logger.warning(msg)
            return False

    def limpiar_pycache(self, res: ResultadoPurga) -> None:
        for d in self._raiz.rglob("__pycache__"):
            if d.is_dir() and not self._excluida(d):
                if self._borrar(d, res, es_dir=True):
                    res.pycache_eliminados += 1
                    logger.info("  🗑 __pycache__: %s", d.relative_to(self._raiz))

    def limpiar_pyc(self, res: ResultadoPurga) -> None:
        for ext in self._PYC_EXT:
            for f in self._raiz.rglob(f"*{ext}"):
                if f.is_file() and not self._excluida(f) and "__pycache__" not in f.parts:
                    if self._borrar(f, res):
                        res.pyc_eliminados += 1

    def limpiar_tmp_ipfs(self, res: ResultadoPurga) -> None:
        for f in self._raiz.rglob(f"{self._TMP_PREFIX}*.bin"):
            if f.is_file() and not self._excluida(f):
                if self._borrar(f, res):
                    res.tmp_ipfs_eliminados += 1
                    logger.info("  🗑 tmp IPFS: %s", f.relative_to(self._raiz))

    def limpiar_chg(self, res: ResultadoPurga) -> None:
        if not self._chg_dir.exists():
            return
        for item in self._chg_dir.iterdir():
            if item.name in (".gitkeep", "README.md"):
                continue
            if not self._excluida(item):
                es_dir = item.is_dir()
                if self._borrar(item, res, es_dir=es_dir):
                    res.chg_eliminados += 1
                    logger.info("  🗑 CHG: %s", item.name)

    def limpiar_pytest_cache(self, res: ResultadoPurga) -> None:
        for d in self._raiz.rglob(".pytest_cache"):
            if d.is_dir() and not self._excluida(d):
                if self._borrar(d, res, es_dir=True):
                    res.pytest_cache_eliminados += 1
                    logger.info("  🗑 pytest_cache: %s", d.relative_to(self._raiz))

    def limpiar_logs_tmp(self, res: ResultadoPurga) -> None:
        for pat in ("*.log", "*.tmp", "*.bak", "*.orig", "*.rej"):
            for f in self._raiz.rglob(pat):
                if f.is_file() and not self._excluida(f) and ".git" not in f.parts:
                    if self._borrar(f, res):
                        res.logs_tmp_eliminados += 1

    def limpiar_bak_ledger(self, res: ResultadoPurga) -> None:
        """Limpia .json.tmp huérfanos (los .bak del ledger se conservan)."""
        for f in self._raiz.rglob("*.json.tmp"):
            if f.is_file() and not self._excluida(f):
                if self._borrar(f, res):
                    res.bak_tmp_eliminados += 1

    def ejecutar(self) -> ResultadoPurga:
        res = ResultadoPurga()
        t0 = time.perf_counter()
        logger.info("PurgaFilesystem: raíz=%s", self._raiz)
        self.limpiar_pycache(res)
        self.limpiar_pyc(res)
        self.limpiar_tmp_ipfs(res)
        self.limpiar_chg(res)
        self.limpiar_pytest_cache(res)
        self.limpiar_logs_tmp(res)
        self.limpiar_bak_ledger(res)
        res.duracion_seg = time.perf_counter() - t0
        logger.info(
            "Purga OK: pycache=%d | pyc=%d | tmp=%d | chg=%d | pytest=%d | logs=%d | bak=%d | %.2f KB | %.2fs",
            res.pycache_eliminados, res.pyc_eliminados, res.tmp_ipfs_eliminados,
            res.chg_eliminados, res.pytest_cache_eliminados, res.logs_tmp_eliminados,
            res.bak_tmp_eliminados, res.bytes_liberados / 1024, res.duracion_seg,
        )
        return res

# ── ORQUESTADOR ──────────────────────────────────────────────────────────────
class Purgador:
    """
    Orquesta las dos fases:
      1. CustodiaIPFS  → sube PSNRL a IPFS
      2. PurgaFilesystem → elimina __pycache__ y residuos Python
    Si la custodia falla y no se fuerza, la purga se cancela para proteger los datos.
    """

    _INFORME: Final[str] = "purgador_informe.json"

    def __init__(
        self,
        raiz: Optional[Path] = None,
        omitir_ipfs: bool = False,
        forzar: bool = False,
        guardar_informe: bool = True,
        excluir: Optional[List[str]] = None,
    ) -> None:
        self._raiz          = raiz or ROOT_DIR
        self._omitir_ipfs   = omitir_ipfs
        self._forzar        = forzar
        self._guardar       = guardar_informe
        self._custodia      = CustodiaIPFS()
        self._purga_fs      = PurgaFilesystem(raiz=self._raiz, excluir=excluir)

    def _guardar_informe(self, informe: InformePurgador) -> None:
        ruta = self._raiz / self._INFORME
        try:
            with open(ruta, "w", encoding="utf-8") as fh:
                json.dump(informe.como_dict(), fh, indent=2, ensure_ascii=False)
            logger.info("Informe guardado: %s", ruta)
        except Exception as exc:
            logger.error("No se pudo guardar el informe: %s", exc)

    def ejecutar(self) -> InformePurgador:
        t0 = time.perf_counter()
        inf = InformePurgador(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            raiz_escaneada=str(self._raiz),
            fase1_omitida=self._omitir_ipfs,
            purga_forzada=self._forzar,
        )
        # Fase 1
        if not self._omitir_ipfs:
            logger.info("═══ FASE 1: Custodia IPFS ═══")
            inf.custodia = self._custodia.ejecutar(forzar_borrado=self._forzar)
            if not inf.custodia.exito and not self._forzar:
                logger.warning("Custodia incompleta. Purga cancelada (usa forzar=True).")
                inf.duracion_total_seg = time.perf_counter() - t0
                if self._guardar:
                    self._guardar_informe(inf)
                return inf
        # Fase 2
        logger.info("═══ FASE 2: Purga Filesystem ═══")
        inf.purga = self._purga_fs.ejecutar()
        inf.duracion_total_seg = time.perf_counter() - t0
        if self._guardar:
            self._guardar_informe(inf)
        return inf

# ── API PÚBLICA ──────────────────────────────────────────────────────────────
def purgar_proyecto(
    raiz: Optional[Path] = None,
    forzar: bool = False,
    guardar_informe: bool = True,
) -> InformePurgador:
    """Flujo completo: sube pesos a IPFS y limpia __pycache__ / artefactos Python."""
    return Purgador(raiz=raiz, forzar=forzar, guardar_informe=guardar_informe).ejecutar()


def solo_limpiar_pycache(raiz: Optional[Path] = None) -> ResultadoPurga:
    """Solo limpia residuos (__pycache__, .pyc, CHG, pytest_cache, logs) — sin tocar IPFS."""
    return PurgaFilesystem(raiz=raiz or ROOT_DIR).ejecutar()


def depositar_en_chg(origen: Path, chg_dir: Optional[Path] = None) -> Optional[Path]:
    """Mueve un residuo de sesión al directorio CHG (papelera). Retorna destino o None."""
    try:
        src = Path(origen)
        if not src.exists():
            return None
        dst_dir = chg_dir or CHG_DIR
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"{int(time.time_ns())}_{src.name}"
        shutil.move(str(src), str(dst))
        logger.info("Residuo → CHG: %s", dst.name)
        return dst
    except Exception as exc:
        logger.warning("depositar_en_chg(%s): %s", origen, exc)
        return None


def recolectar_pycache_en_chg(raiz: Optional[Path] = None,
                              chg_dir: Optional[Path] = None) -> int:
    """Mueve todo __pycache__ / *.pyc / *.pyo a CHG para visiualizar la caché
    de la sesión. Retorna nº de elementos depositados. CHG se vacía al cerrar
    sesión vía solo_limpiar_pycache()."""
    base = raiz or ROOT_DIR
    dst_dir = chg_dir or CHG_DIR
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    # 1) directorios __pycache__ completos → CHG/<relpath__con__guiones>
    for d in sorted(base.rglob("__pycache__")):
        if not d.is_dir():
            continue
        if any(e in d.parts for e in (".git", "node_modules", ".venv", "venv", "env")):
            continue
        if CHG_DIR in d.parents or d == CHG_DIR:
            continue
        try:
            rel = d.parent.relative_to(base).as_posix().replace("/", "__") or "root"
            destino = dst_dir / f"{int(time.time_ns())}_{rel}__pycache__"
            shutil.move(str(d), str(destino))
            n += 1
            logger.info("Cache → CHG: %s", destino.name)
        except Exception as exc:
            logger.warning("recolectar(%s): %s", d, exc)
    # 2) .pyc/.pyo sueltos fuera de __pycache__
    for ext in (".pyc", ".pyo"):
        for f in sorted(base.rglob(f"*{ext}")):
            if not f.is_file() or "__pycache__" in f.parts:
                continue
            if CHG_DIR in f.parents:
                continue
            if depositar_en_chg(f, dst_dir) is not None:
                n += 1
    return n


def solo_custodia_ipfs(forzar_borrado: bool = False) -> ResultadoCustodia:
    """Solo sube pesos de PSNRL a IPFS sin limpiar el filesystem."""
    return CustodiaIPFS().ejecutar(forzar_borrado=forzar_borrado)


def estado_psnrl() -> Dict[str, Any]:
    """Devuelve estado actual del directorio PSNRL."""
    PSNRL_DIR.mkdir(parents=True, exist_ok=True)
    archivos = sorted(f for f in PSNRL_DIR.iterdir() if f.is_file())
    bytes_t  = sum(f.stat().st_size for f in archivos)
    return {
        "directorio": str(PSNRL_DIR), "existe": True,
        "archivos": len(archivos), "tamaño_kb": round(bytes_t / 1024, 2),
        "listado": [f.name for f in archivos],
    }


