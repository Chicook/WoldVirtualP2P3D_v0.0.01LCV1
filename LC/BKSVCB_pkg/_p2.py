"""
BKSVCB - parte 2/2 (version de sesion LucIA).
Parte del subsistema LucIA (_hook_salida_sistema).
"""
from __future__ import annotations

"""
BKSVCB.py - Blockchain Server & Consenso Neuronal de Celebro (Arquitectura 2026)
================================================================================
Servidor de bloques inmutable y libro mayor distribuido para WoldVirtualP2P3D:
  - Transforma blockchain_ledger.json en pesos neuronales nada mas arrancar.
  - Mantiene activa la sincronizacion y actualizacion neuronal durante la sesion.
  - Al cerrar sesion, sube atomicamente el estado y pesos de PSNRL a IPFS.
  - Estructura de bloques criptograficos con hashing doble SHA-256 y Merkle Root.
  - Visualizacion de hashes unicos de cada bloque en terminal en tiempo real.
  - Consenso PoNL (Proof of Neural Learning) y servidor HTTP REST/JSON-RPC.
"""
from __future__ import annotations
import os, sys, time, json, hashlib, logging, threading, atexit
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def _hook_salida_sistema() -> None:
    """Ejecutado al cerrar la sesion de terminal: sube ledger y pesos a IPFS."""
    if _blockchain_instancia:
        _blockchain_instancia.cerrar_sesion_y_subir_ipfs()


atexit.register(_hook_salida_sistema)


if __name__ == "__main__":
    bks = get_blockchain_server()
    print("\n\033[1;36m" + "=" * 74 + "\033[0m")
    print(f"  \033[1;32mBLOCKCHAIN SERVER & CONSENSO NEURAL: {__server_name__}\033[0m")
    print(f"  Neuronas Activas: \033[1;35m{len(bks.conversor.neuronas)}\033[0m | Dificultad: \033[1;37m{bks.dificultad}\033[0m | Bloques: \033[1;37m{len(bks.cadena)}\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m")

    # Muestra los hashes unicos inmediatamente
    bks.mostrar_cadena_hashes_terminal()

    # 1. Transformacion del ledger en pesos neuronales
    print("[+] Transformando blockchain_ledger.json en pesos neuronales (50 neuronas)...")
    res_t = bks.transformar_ledger_a_pesos_neuronales()
    print(f"    Bloques convertidos : \033[1;33m{res_t['total_bloques']}\033[0m")
    print(f"    Pesos en PSNRL      : \033[1;32m{res_t['npz_pesos']}\033[0m (Norma: {res_t['norma_acumulada']})")

    # 2. Arranque del actualizador continuo de la red neuronal durante la sesion
    bks.arrancar_actualizador_red()
    print("    Actualizador neuronal: \033[1;32mACTIVO EN SEGUNDO PLANO DURANTE LA SESION\033[0m")

    # 3. Transaccion y minado en vivo
    print("\n[+] Nueva transaccion cognitiva en blockchain...")
    res = bks.registrar_aprendizaje_neural(
        prompt="¿Cual es el rol de las 50 neuronas en la blockchain de Celebro?",
        respuesta="Registrar cada gradiente Hebbiano y estado emocional en bloques inmutables con hashes unicos.",
        modelo="qwen2.5:7b",
    )
    print(f"    CID IPFS vinculado   : \033[1;36m{res['ipfs_cid']}\033[0m")

    bloque = bks.minar_transacciones_pendientes()
    if bloque:
        print(f"    \033[1;32mBloque #{bloque.indice} consolidado con exito!\033[0m")
        print(f"    Hash unico del bloque: \033[1;33m{bloque.hash_bloque}\033[0m")
        print(f"    Raiz de Merkle       : \033[1;35m{bloque.merkle_root}\033[0m")

    valida, err = bks.validar_cadena()
    estado_str = "\033[1;32mINTEGRA Y VALIDA\033[0m" if valida else f"\033[1;31mERROR: {err}\033[0m"
    print(f"\n[+] Estado de la cadena: {estado_str}")
    print("    Persistencia IPFS al cierre: \033[1;36mREGISTRADA (atexit hook activo)\033[0m")
    print("\033[1;36m" + "=" * 74 + "\033[0m\n")
