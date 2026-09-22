"""
SNSBSTNPRB - parte 1/2 (version de sesion LucIA).
Parte del subsistema LucIA (C).
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
import os, sys, json, time, urllib.request, urllib.error, atexit, signal
from pathlib import Path
from typing import Any, Dict, List, Optional

# ─── RUTAS BASE ─────────────────────────────────────────────────────────────
SBSTM_DIR   = Path(__file__).resolve().parent
CMFG_DIR    = SBSTM_DIR.parent
CELEBRO_DIR = CMFG_DIR.parent
ROOT_DIR    = CELEBRO_DIR.parent.parent

PSNRL_DIR   = CELEBRO_DIR / "PSNRL"
PSNRL_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING))
logger = logging.getLogger("WoldVirtualP2P3D.SNSBSTNPRB")

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ─── PALETA ANSI PREMIUM ─────────────────────────────────────────────────────
class C:
    R = "\033[0m";  B = "\033[1m";  DIM = "\033[2m"
    CY = "\033[96m"; BL = "\033[94m"; GR = "\033[92m"
    YL = "\033[93m"; MG = "\033[95m"; RD = "\033[91m"
    WH = "\033[97m"

    @staticmethod
    def c(color: str, txt: str) -> str:
        return f"{color}{txt}{C.R}"


# ─── CARGA DIFERIDA DEL NUCLEO ───────────────────────────────────────────────
def _importar_nucleo():
    """Carga los modulos core de Celebro; aborta con mensaje claro si faltan."""
    try:
        from LC.celebro.BKSVCB import get_blockchain_server, iniciar_servidor_blockchain
        from LC.celebro.CMFG.PSNRCV import get_conversor_pesos
        from LC.celebro.CMFG.pesos_vivos import get_gestor_pesos_vivos
        return get_blockchain_server, iniciar_servidor_blockchain, get_conversor_pesos, get_gestor_pesos_vivos
    except ImportError as e:
        print(C.c(C.RD, f"[ERROR] Dependencia no encontrada: {e}"))
        sys.exit(1)


# ─── CLIENTE OLLAMA ──────────────────────────────────────────────────────────
OLLAMA_URL  = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
IA_CFG_PATH = ROOT_DIR / "LC" / "modelosIAlocal" / "IAlocal.json"


def _cargar_config_ia() -> Dict[str, Any]:
    """Lee IAlocal.json y normaliza el esquema nuevo al contrato legacy."""
    defaults: Dict[str, Any] = {
        "schema_version": 1,
        "default_model": "qwen2.5:0.5b",
        "temperature": 0.7,
        "max_tokens": 2048,
        "models_available": [],
    }
    if IA_CFG_PATH.exists():
        try:
            with open(IA_CFG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
            if not isinstance(config, dict):
                return defaults
            normalized = {**defaults, **config}
            backends = config.get("backends", [])
            ollama = next((item for item in backends if item.get("type") == "ollama"), None)
            if isinstance(ollama, dict) and ollama.get("base_url"):
                normalized["ollama_url"] = str(ollama["base_url"]).rstrip("/")
            return normalized
        except Exception:
            pass
    return defaults


def _ollama_disponible() -> bool:
    """Comprueba si Ollama esta corriendo."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _listar_modelos_ollama() -> List[str]:
    """Retorna modelos disponibles en Ollama."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def consultar_ollama_stream(prompt: str, modelo: str,
                            temperatura: float = 0.7, max_tokens: int = 2048) -> str:
    """Envia prompt a Ollama con streaming real; imprime tokens en vivo."""
    payload = json.dumps({
        "model": modelo, "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperatura, "num_predict": max_tokens},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    tokens: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(f"\n{C.c(C.GR, chr(9654))} ", end="", flush=True)
            for linea in resp:
                try:
                    chunk = json.loads(linea.decode("utf-8"))
                    tok = chunk.get("response", "")
                    if tok:
                        print(tok, end="", flush=True)
                        tokens.append(tok)
                    if chunk.get("done", False):
                        break
                except Exception:
                    continue
            print()
    except urllib.error.URLError as e:
        print(C.c(C.RD, f"\n[Ollama] Error: {e}"))
    return "".join(tokens)


# ─── TELEMETRIA COMPACTA DE PESOS ────────────────────────────────────────────
def _mini_pesos(turno: int, gsnr: float, deriva: float,
                neuronas: int, delta: float, tono: str) -> str:
    """Genera linea compacta de telemetria sinaptica para terminal."""
    BL = 12
    ratio  = min(1.0, delta / max(0.01, deriva + 1e-9))
    llenos = int(round(ratio * BL))
    barra  = C.c(C.CY, "\u2588" * llenos) + C.c(C.DIM, "\u2591" * (BL - llenos))
    ec     = C.GR if gsnr > 1.5 else (C.YL if gsnr > 0.8 else C.RD)
    return (
        f"  {C.c(C.DIM, f'[T#{turno:04d}]')} "
        f"GSNR:{C.c(ec, f'{gsnr:.2f}')} "
        f"Delta:{C.c(C.MG, f'{delta:.5f}')} "
        f"Deriva:{C.c(C.YL, f'{deriva:.4f}')} "
        f"N:{C.c(C.WH, str(neuronas))} "
        f"{barra} {C.c(C.BL, tono[:22])}"
    )


# ─── ORQUESTADOR PRINCIPAL ───────────────────────────────────────────────────
