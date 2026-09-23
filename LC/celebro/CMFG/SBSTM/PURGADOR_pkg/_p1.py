"""
PURGADOR - parte 1/2 (version de sesion LucIA).
Este código Python está generando y procesando archivos y datos. Se crea una estructura de datos (`ResultadoCustodia`) que mantiene ciertos 
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
# Esta parte vive un nivel por debajo de SBSTM al estar dentro de PURGADOR_pkg.
SBSTM_DIR:   Final[Path] = _HERE.parent.parent
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
from _p1_CustodiaIPFS import CustodiaIPFS  # CLASSPACK
from _p1_PurgaFilesystem import PurgaFilesystem  # CLASSPACK
from _p1_Purgador import Purgador  # CLASSPACK
