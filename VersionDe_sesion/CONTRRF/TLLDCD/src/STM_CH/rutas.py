"""Rutas centralizadas de artefactos runtime -> STM_CH.

Todo lo que el sistema crea en ejecucion (pycache, pesos PSNRL,
ledger, cache IAFREE) vive bajo STM_CH/ y se borra al cerrar.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

STM_CH_DIR: Path = Path(__file__).resolve().parent
PYCACHE_DIR: Path = STM_CH_DIR / "pycache"
PSNRL_DIR: Path = STM_CH_DIR / "PSNRL"
LEDGER_PATH: Path = STM_CH_DIR / "blockchain_ledger.json"
CACHE_IAFREE: Path = STM_CH_DIR / "openrouter_free_cache.json"


def inicializar() -> None:
    """Crea STM_CH/PSNRL y redirige __pycache__ a STM_CH/pycache."""
    PSNRL_DIR.mkdir(parents=True, exist_ok=True)
    PYCACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        sys.pycache_prefix = str(PYCACHE_DIR)
    except Exception:
        pass


def limpiar() -> dict:
    """Borra todo el contenido generado en STM_CH/. Devuelve resumen."""
    resumen = {"borrados": 0, "errores": []}
    for objetivo in (PYCACHE_DIR, PSNRL_DIR, STM_CH_DIR / "__pycache__",
                       STM_CH_DIR.parent / "__pycache__"):
        if objetivo.exists():
            for hijo in list(objetivo.iterdir()):
                try:
                    if hijo.is_dir() and not hijo.is_symlink():
                        shutil.rmtree(hijo, ignore_errors=True)
                    else:
                        hijo.unlink(missing_ok=True)
                    resumen["borrados"] += 1
                except Exception as e:
                    resumen["errores"].append(f"{hijo}: {e}")
    for fich in (LEDGER_PATH, CACHE_IAFREE):
        try:
            if fich.exists():
                fich.unlink()
                resumen["borrados"] += 1
        except Exception as e:
            resumen["errores"].append(f"{fich}: {e}")
    return resumen
