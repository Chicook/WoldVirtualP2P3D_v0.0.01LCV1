"""
BASELUC.py - Base compartida del orquestador LucIA (2026)
Rutas, imports resilientes con flags, logging y GestorEntornoSeguro.
Los mixins PRTLUC/TRNLUC/CMDLUC y mainLCSTM importan desde aqui.
"""
from __future__ import annotations

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union

# ─── JERARQUIA DE DIRECTORIOS Y CONFIGURACION DE ENTORNO ─────────────────────
CURRENT_FILE: Final[Path] = Path(__file__).resolve()
LC_DIR: Final[Path] = CURRENT_FILE.parent
ROOT_DIR: Final[Path] = LC_DIR.parent
ENV_FILE: Final[Path] = ROOT_DIR / ".env"
CELEBRO_DIR: Final[Path] = LC_DIR / "celebro"
PSNRL_DIR: Final[Path] = CELEBRO_DIR / "PSNRL"
CMFG_DIR: Final[Path] = CELEBRO_DIR / "CMFG"
SBSTM_DIR: Final[Path] = CMFG_DIR / "SBSTM"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ─── CACHE PYTHON → CHG (papelera de sesión, se vacía al cerrar) ─────────────
# Todo __pycache__ generado durante la sesión aparece en CHG/ y se borra
# en cerrar_sistema() vía PURGADOR.solo_limpiar_pycache().
_CHG_DIR = ROOT_DIR / "CHG"
_CHG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_CHG_DIR / "pycache"))
sys.dont_write_bytecode = False

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ─── IMPORTACION DEL SUBSISTEMA DE INFERENCIA GRATUITA Y ESTILOS ─────────────
try:
    from LC.celebro.CMFG.SBSTM.IAFREE import ClienteIAFree, get_cliente_iafree
except Exception:
    get_cliente_iafree = None
    ClienteIAFree = None

try:
    from LC.celebro.CMFG.SBSTM.STYLOS import (
        ColoresLucIA,
        EstiloTerminalLucIA,
        Glifos,
        badge_turno,
        banner_bienvenida,
        formatear_respuesta_lucia,
        panel_ayuda_comandos,
    )
except Exception:
    EstiloTerminalLucIA = None

try:
    from LC.celebro.CMFG.SBSTM.RPLC import (
        get_procesador_rplc,
        reprocesar_con_metricas,
    )
    _RPLC_DISPONIBLE = True
except Exception:
    _RPLC_DISPONIBLE = False
    get_procesador_rplc = None  # type: ignore
    reprocesar_con_metricas = None  # type: ignore

try:
    from LC.celebro.CMFG.SBSTM.voice_engine import (
        cancel_speech as _cancelar_voz,
        configurar_prosodia_juvenil as _prosodia_voz,
        speak as _hablar_voz,
    )
    _VOZ_DISPONIBLE = True
except Exception:
    _VOZ_DISPONIBLE = False
    _hablar_voz = None  # type: ignore
    _cancelar_voz = None  # type: ignore
    _prosodia_voz = None  # type: ignore

# ─── IMPORTACION DSIALCLGRG (IA local ligera autonoma / fallback OpenRouter) ──
try:
    from LC.celebro.CMFG.SBSTM.DSIALCLGRG import (
        consultar_lucia_local,
        descargar_modelo as _descargar_ia_local,
        descargar_todos_recomendados as _descargar_recomendados,
        estado_dsialclgrg as _estado_ia_local,
        perfilar_hardware as _perfilar_hw_local,
        recomendar_modelos as _recomendar_ia_local,
    )
    _IALOCAL_DISPONIBLE = True
except Exception:
    _IALOCAL_DISPONIBLE = False
    consultar_lucia_local = None  # type: ignore
    _descargar_ia_local = None  # type: ignore
    _descargar_recomendados = None  # type: ignore
    _estado_ia_local = None  # type: ignore
    _perfilar_hw_local = None  # type: ignore
    _recomendar_ia_local = None  # type: ignore

try:
    from LC.celebro.CMFG.SBSTM.ROTACIONIA import RotadorIA, get_rotador_ia
    _ROTACIONIA_DISPONIBLE = True
except Exception:
    RotadorIA = None  # type: ignore
    get_rotador_ia = None  # type: ignore
    _ROTACIONIA_DISPONIBLE = False

try:
    from LC.celebro.CMFG.SBSTM.NEUROSINTESIS import (
        SintetizadorNeuronalLucIA,
        get_sintetizador_lucia,
    )
    _NEUROSINTESIS_DISPONIBLE = True
except Exception:
    SintetizadorNeuronalLucIA = None  # type: ignore
    get_sintetizador_lucia = None  # type: ignore
    _NEUROSINTESIS_DISPONIBLE = False

# ─── IMPORTACION MDSTM (descarga autonoma real en LC/modelosIAlocal) ──────
try:
    from LC.modelosIAlocal.MDSTM import (
        GestorDescargaModelos,
        get_gestor_mdstm,
        ordenar_descarga_lucia,
    )
    _MDSTM_DISPONIBLE = True
except Exception:
    _MDSTM_DISPONIBLE = False
    GestorDescargaModelos = None  # type: ignore
    get_gestor_mdstm = None  # type: ignore
    ordenar_descarga_lucia = None  # type: ignore

# ─── IMPORTACION HRCTRC (herrero constructor: overlay + unificacion) ──────
try:
    from LC.celebro.CMFG.SBSTM.HRCTRC import (
        GestorConstructorSesion,
        get_gestor_hrctrc,
        ordenar_constructor_lucia,
    )
    _HRCTRC_DISPONIBLE = True
except Exception:
    _HRCTRC_DISPONIBLE = False
    GestorConstructorSesion = None  # type: ignore
    get_gestor_hrctrc = None  # type: ignore
    ordenar_constructor_lucia = None  # type: ignore

# ─── IMPORTACION HRCTRC_RFCT (version de sesion 400/450 + modelo local) ──
try:
    from LC.celebro.CMFG.SBSTM.HRCTRC_RFCT import (
        RefactorizadorSesion,
        refactorizar_overlay,
    )
    _HRCTRC_RFCT_DISPONIBLE = True
except Exception:
    _HRCTRC_RFCT_DISPONIBLE = False
    RefactorizadorSesion = None  # type: ignore
    refactorizar_overlay = None  # type: ignore

# ─── IMPORTACION HRCNTR (monitor y refactorizador neural) ──
try:
    from LC.celebro.CMFG.SBSTM.HRCNTR import (
        GestorHRCNTR,
        get_gestor_hrctnr,
        ejecutar_hrctnr,
        estado_hrctnr,
        actualizar_sistema_hrctnr,
    )
    _HRCNTR_DISPONIBLE = True
except Exception:
    _HRCNTR_DISPONIBLE = False
    GestorHRCNTR = None  # type: ignore
    get_gestor_hrctnr = None  # type: ignore
    ejecutar_hrctnr = None  # type: ignore
    estado_hrctnr = None  # type: ignore
    actualizar_sistema_hrctnr = None  # type: ignore

# ─── IMPORTACION INTEGRACIONRF (cierre: refactor->md->pesos->IPFS->rama) ──
try:
    from LC.celebro.CMFG.SBSTM.INTEGRACIONRF import (
        IntegradorRefactor,
        get_integrador,
        registrar_actividad,
    )
    _INTEGRACIONRF_DISPONIBLE = True
except Exception:
    _INTEGRACIONRF_DISPONIBLE = False
    IntegradorRefactor = None  # type: ignore
    get_integrador = None  # type: ignore
    registrar_actividad = None  # type: ignore

LOG_LEVEL = os.getenv("LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.WARNING))
for _log_name in ("", "WoldVirtualP2P3D", "LC", "urllib3", "ENRN", "SLRN", "RNP", "httpx"):
    logging.getLogger(_log_name).setLevel(getattr(logging, LOG_LEVEL, logging.WARNING))
logger = logging.getLogger(__name__)

# ─── GESTOR SEGURO DE VARIABLES DE ENTORNO (.ENV) ───────────────────────────
class GestorEntornoSeguro:
    """Carga y gestiona de forma aislada credenciales y rutas desde .env."""

    @staticmethod
    def cargar_variables(ruta_env: Path = ENV_FILE) -> Dict[str, str]:
        variables: Dict[str, str] = {}
        if not ruta_env.exists():
            return variables

        try:
            contenido = ruta_env.read_text(encoding="utf-8-sig", errors="replace")
            for linea in contenido.splitlines():
                linea_limpia = linea.strip()
                if not linea_limpia or linea_limpia.startswith("#"):
                    continue
                if "=" in linea_limpia:
                    clave, _, valor = linea_limpia.partition("=")
                    clave = clave.strip().lstrip("\ufeff")
                    valor = valor.strip().strip("'\"")
                    variables[clave] = valor
                    os.environ[clave] = valor
        except Exception as err:
            sys.stderr.write(f"[mainLCSTM] Advertencia leyendo .env: {err}\n")
        return variables

    @classmethod
    def obtener_openrouter_key(cls) -> str:
        vars_env = cls.cargar_variables()
        return vars_env.get("OPENROUTER_API_KEY", os.getenv("OPENROUTER_API_KEY", "")).strip()
