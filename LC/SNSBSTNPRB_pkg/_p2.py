"""
SNSBSTNPRB - parte 2/2 (version de sesion LucIA).
Parte del subsistema LucIA (_manejador_senal).
"""
from __future__ import annotations

"""
SNSBSTNPRB.PY - SubSistema de Sesion Neuronal P2P de Celebro (Arquitectura 2026)
==================================================================================
Orquestador de sesion completa para WoldVirtualP2P3D:
  - Arranca la blockchain (BKSVCB) y despliega hashes unicos en terminal.
  - Transforma el ledger existente en pesos neuronales sobre las 50 sinapsis.
  - Mantiene conversacion real con modelos Ollama locales (cogito:3b, qwen2.5:7b).
  - Cada respuesta del modelo genera transaccion sinaptica en blockchain.
  - Monitor de pesos vivos (GSNR, deriva Muon, entropia Shannon) por turno.
  - Minado automatico cada 3 turnos; checkpoint PSNRL + IPFS al cerrar sesion.
"""
from __future__ import annotations
import os, sys, json, time, urllib.request, urllib.error, atexit, signal
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── RUTAS BASE ─────────────────────────────────────────────────────────────

def _manejador_senal(sig, frame) -> None:  # noqa: ARG001
    print(C.c(C.YL, "\n  [SIGINT] Cerrando sesion neural..."))
    sys.exit(0)


signal.signal(signal.SIGINT, _manejador_senal)


# ─── PUNTO DE ENTRADA ────────────────────────────────────────────────────────
if __name__ == "__main__":
    sesion = SesionNeuronalP2P()
    sesion.arrancar()
    sesion.bucle_conversacion()