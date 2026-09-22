"""
ipfs_manager - parte 2/2 (version de sesion LucIA).
Parte del subsistema LucIA.
"""
from __future__ import annotations

"""
ipfs_manager.py - Gestor IPFS Real para Pesos Neuronales de Celebro (Arquitectura 2026)
========================================================================================
Conexion real a IPFS via Kubo HTTP RPC API v0 y CLI:
  - Auto-discovery de daemon Kubo en puertos 5001/5002/5003.
  - IPFS CLI como fallback real con binario local IPFSonl o PATH del sistema.
  - Motor puro Python CIDv1 (dag-raw + sha2-256 + base32lower) para operacion offline.
  - Subida en batch de todos los pesos de PSNRL al cerrar sesion.
  - Borrado garantizado del archivo local tras confirmacion de pin exitoso.
  - Manifiesto rotativo TTL 50 entradas (payload_b64 solo en las 10 mas recientes).
  - Compatibilidad directa con pesos_vivos.py / PSNRCV.py.
"""
import os, sys, json, time, base64, hashlib, logging, subprocess
import urllib.request, urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("WoldVirtualP2P3D.IPFS")

_KUBO_PORTS = [5001, 5002, 5003]
_KUBO_TIMEOUT_FAST = 1.5
_KUBO_TIMEOUT_UP = 30
_MULTICODEC_RAW = 0x55
_MULTIHASH_SHA256 = (0x12, 0x20)
_BASE32_PREFIX = "b"

_LOCAL_IPFS_BIN = Path(__file__).parent / "IPFSonl" / "ipfs.exe"
_LOCAL_IPFS_ENV = {**os.environ, "IPFS_PATH": str(Path(__file__).parent / "IPFSonl" / ".ipfs")}



def cerrar_sesion_y_subir_psnrl(forzar: bool = False) -> Dict[str, Any]:
    """Hook de cierre: sube todos los pesos de PSNRL a IPFS y los borra."""
    return get_ipfs_manager().subir_y_limpiar_psnrl(forzar_borrado_sin_daemon=forzar)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    print("\n=== IPFS MANAGER 2026 - TEST EN VIVO ===\n")
    print(f"  Binario Kubo local : {_LOCAL_IPFS_BIN} | existe={_LOCAL_IPFS_BIN.exists()}")

    try:
        from LC.celebro.CMFG.IPFSonl.ipfs_launcher import arrancar_daemon
        print("  Arrancando daemon IPFS local (Kubo)...")
        daemon_ok = arrancar_daemon(esperar_segundos=20)
        print(f"  Daemon arrancado: {daemon_ok}")
    except Exception as ex:
        print(f"  Launcher no disponible ({ex}). Continuando sin daemon.")

    mgr = get_ipfs_manager()
    estado = mgr.estado_conexion()
    print(f"  Daemon IPFS activo  : {estado['daemon_activo']} ({estado['daemon_url']})")
    print(f"  CLI kubo disponible : {estado['cli_disponible']}")
    print(f"  Motor autonomo CIDv1: {estado['motor_autonomo']}\n")

    psnrl = Path(__file__).parent.parent / "PSNRL"
    archivos_psnrl = list(psnrl.glob("*.*")) if psnrl.exists() else []
    print(f"  Archivos en PSNRL: {len(archivos_psnrl)}")

    if archivos_psnrl:
        print("  Subiendo + limpiando PSNRL hacia IPFS...")
        resumen = cerrar_sesion_y_subir_psnrl()
        print(f"  Procesados : {resumen['archivos_procesados']}")
        print(f"  CIDs subidos: {len(resumen['cids'])}")
        print(f"  Borrados   : {len(resumen['borrados'])}")
        print(f"  Errores    : {len(resumen['errores'])}")
        for cid in resumen["cids"][:5]:
            print(f"    {cid}")
    else:
        print("  PSNRL vacia. Demo con payload ficticio (subida CLI real si daemon activo)...")
        payload = b'{"demo":"pesos_celebro_2026","neuronas":50}' * 100
        res = mgr.almacenar_pesos(payload, nombre_modelo="demo_pesos", eliminar_local=False)
        print(f"  Demo CID   : {res['cid']}")
        print(f"  Nodo usado : {res['nodo']} | subida_real={res['subida_real']}")
        if res['subida_real']:
            print(f"  Ver en gateway: https://ipfs.io/ipfs/{res['cid']}")
        rec = mgr.descargar_bytes(res["cid"])
        print(f"  Bytes rec  : {len(rec) if rec else 0}")

    print(f"\n  Total en manifiesto: {mgr.manifest.get('total_stored', 0)} entradas")
    print("\n=== TEST FINALIZADO ===\n")