"""
PURGADOR - parte 2/2 (version de sesion LucIA).
El código imprime un resumen de un informe Purgador utilizando ANSI para el color y formato en terminal.
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

def imprimir_informe(inf: InformePurgador) -> None:
    """Imprime resumen visual con ANSI en terminal."""
    SEP = f"{ANSI['C']}{'═'*78}{ANSI['R']}"
    SUB = f"{ANSI['D']}{'─'*78}{ANSI['R']}"
    print(f"\n{SEP}")
    print(f"  {ANSI['B']}{ANSI['W']}🧹 PURGADOR LucIA v{__version__}{ANSI['R']}")
    print(f"  {ANSI['D']}{inf.timestamp_utc} | {inf.raiz_escaneada}{ANSI['R']}")
    print(SUB)
    c = inf.custodia
    if inf.fase1_omitida:
        print(f"  {ANSI['Y']}⚠ Fase 1 IPFS omitida{ANSI['R']}")
    else:
        ok = f"{ANSI['G']}✓ OK{ANSI['R']}" if c.exito else f"{ANSI['RE']}✗ ERRORES{ANSI['R']}"
        nodo = (f"{ANSI['G']}DAEMON{ANSI['R']}" if c.daemon_real
                else (f"{ANSI['Y']}CLI{ANSI['R']}" if c.cli_real
                      else f"{ANSI['M']}CIDv1 LOCAL{ANSI['R']}"))
        print(f"  {ANSI['B']}FASE 1 — Custodia IPFS{ANSI['R']}  {ok}")
        print(f"    Pinados: {ANSI['G']}{c.archivos_pinados}{ANSI['R']} | "
              f"Borrados: {ANSI['C']}{len(c.borrados_locales)}{ANSI['R']} | "
              f"Bytes: {ANSI['W']}{c.bytes_subidos:,}{ANSI['R']} | Nodo: {nodo} | "
              f"{ANSI['D']}{c.duracion_seg:.3f}s{ANSI['R']}")
        for cid in c.cids_obtenidos[:4]:
            print(f"    → {ANSI['C']}{cid}{ANSI['R']}")
        for err in c.errores[:2]:
            print(f"    {ANSI['RE']}✗ {err}{ANSI['R']}")
    print(SUB)
    p = inf.purga
    kb = p.bytes_liberados / 1024
    tam = f"{kb:,.2f} KB" if kb < 10240 else f"{kb/1024:,.2f} MB"
    print(f"  {ANSI['B']}FASE 2 — Purga Filesystem{ANSI['R']}")
    print(f"    __pycache__: {ANSI['C']}{p.pycache_eliminados}{ANSI['R']} | "
          f".pyc: {ANSI['C']}{p.pyc_eliminados}{ANSI['R']} | "
          f"tmp: {ANSI['C']}{p.tmp_ipfs_eliminados}{ANSI['R']} | "
          f"CHG: {ANSI['C']}{p.chg_eliminados}{ANSI['R']} | "
          f"pytest: {ANSI['C']}{p.pytest_cache_eliminados}{ANSI['R']} | "
          f"logs: {ANSI['C']}{p.logs_tmp_eliminados}{ANSI['R']} | "
          f"{ANSI['G']}{tam}{ANSI['R']} | "
          f"{ANSI['D']}{p.duracion_seg:.3f}s{ANSI['R']}")
    for err in p.errores[:2]:
        print(f"    {ANSI['RE']}✗ {err}{ANSI['R']}")
    print(SUB)
    col = ANSI['G'] if not c.errores and not p.errores else ANSI['Y']
    print(f"  {col}{ANSI['B']}Total: {inf.duracion_total_seg:.3f}s{ANSI['R']}")
    print(f"{SEP}\n")


# ── CLI ──────────────────────────────────────────────────────────────────────
def _cli_main() -> None:
    import argparse
    pa = argparse.ArgumentParser(prog="PURGADOR",
        description="PURGADOR LucIA: sube pesos a IPFS y limpia __pycache__.")
    pa.add_argument("--solo-limpiar",  action="store_true", help="Solo __pycache__, sin IPFS.")
    pa.add_argument("--solo-ipfs",     action="store_true", help="Solo IPFS, sin limpieza.")
    pa.add_argument("--forzar",        action="store_true", help="Continuar aunque IPFS falle.")
    pa.add_argument("--raiz",          type=str, default=str(ROOT_DIR), help="Raíz del proyecto.")
    pa.add_argument("--no-informe",    action="store_true", help="No guardar informe JSON.")
    pa.add_argument("--estado-psnrl",  action="store_true", help="Mostrar estado PSNRL y salir.")
    pa.add_argument("--verbose",       action="store_true", help="Logging detallado.")
    args = pa.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    raiz = Path(args.raiz).resolve()
    if args.estado_psnrl:
        est = estado_psnrl()
        print(f"\n  📁 PSNRL: {est['directorio']}")
        print(f"     Archivos: {est['archivos']} | {est['tamaño_kb']} KB")
        for n in est["listado"]: print(f"       • {n}")
        return
    if args.solo_limpiar:
        r = solo_limpiar_pycache(raiz=raiz)
        print(f"  ✓ pycache={r.pycache_eliminados} pyc={r.pyc_eliminados} "
              f"chg={r.chg_eliminados} {r.bytes_liberados/1024:.2f} KB {r.duracion_seg:.3f}s")
        return
    if args.solo_ipfs:
        r = solo_custodia_ipfs(forzar_borrado=args.forzar)
        print(f"  ✓ pinados={r.archivos_pinados} cids={len(r.cids_obtenidos)} "
              f"bytes={r.bytes_subidos:,} {r.duracion_seg:.3f}s")
        return
    imprimir_informe(purgar_proyecto(raiz=raiz, forzar=args.forzar, guardar_informe=not args.no_informe))

if __name__ == "__main__":
    _cli_main()