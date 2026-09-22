"""
__init__.py — Paquete de Tests WoldVirtualP2P3D LucIA (v0.0.01LCV1)
====================================================================
Directorio : LC/celebro/test/
Ejecución  : pytest LC/celebro/test/ -v --timeout=60
Cobertura  : pytest LC/celebro/test/ --cov=LC --cov-report=term-missing

Módulos cubiertos
─────────────────
  test_blockchain.py   → BKSVCB, BloqueNeuronal, PoNL, Merkle Root
  test_psnrcv.py       → ConversorRespuestaPesos, motor Muon NS-5, GSNR
  test_iafree.py       → GestorModelosGratuitos, ClienteIAFree, rotación 429
  test_neuronas.py     → ENRN × 10, RF_SL × 10, RF_EN × 10, RNP × 10, SLRN × 10
  test_ipfs.py         → IPFSManager, CIDv1, TTL manifiesto, upload batch
  test_stylos.py       → ColoresLucIA, Glifos, badges, banners, Markdown
  test_rplc.py         → AnalizadorTextualNeuronal, filtro voz ajena, re-procesado
  test_pesos_vivos.py  → GestorPesosVivos, telemetría ANSI, checkpoint seguro
  test_integracion.py  → Pipeline completo prompt → neuronal → blockchain → IPFS
  test_voz.py          → sanitización TTS, modulador expresivo, cancelación MCI

Convenciones
────────────
  • Cada test comienza con `test_` y tiene docstring descriptivo.
  • Mocks externos se definen en conftest.py.
  • Asserts críticos incluyen mensajes descriptivos (assert X, "motivo").
  • Fixtures de tmp_path garantizan aislamiento entre tests.
  • Marcadores pytest disponibles:
      @pytest.mark.slow    → test que tarda > 2 s
      @pytest.mark.net     → requiere conexión de red real
      @pytest.mark.voice   → requiere edge-tts instalado

Exclusiones de pytest-cov
─────────────────────────
  • LC/celebro/CMFG/IPFSonl/  (binario ipfs.exe, no Python)
  • LC/celebro/PSNRL/          (directorio de datos, no código)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# ─── RUTAS CANÓNICAS DEL PAQUETE ─────────────────────────────────────────────
TEST_DIR: Path    = Path(__file__).resolve().parent
CELEBRO_DIR: Path = TEST_DIR.parent
LC_DIR: Path      = CELEBRO_DIR.parent
ROOT_DIR: Path    = LC_DIR.parent

# Garantizar importabilidad desde cualquier CWD
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# ─── VERSIÓN DE LA SUITE ──────────────────────────────────────────────────────
__version__: str       = "0.1.0"
__suite_name__: str    = "LucIA-TestSuite"
__total_modulos__: int = 10
__total_familias__: int = 5
__total_neuronas__: int = 50

# ─── CATÁLOGO DE MÓDULOS TESTEABLES ──────────────────────────────────────────
MODULOS_CORE: Dict[str, str] = {
    "BKSVCB"       : "LC.celebro.BKSVCB",
    "PSNRCV"       : "LC.celebro.CMFG.PSNRCV",
    "IAFREE"       : "LC.celebro.CMFG.SBSTM.IAFREE",
    "IPFS_MGR"     : "LC.celebro.CMFG.ipfs_manager",
    "PESOS_VIVOS"  : "LC.celebro.CMFG.pesos_vivos",
    "STYLOS"       : "LC.celebro.CMFG.SBSTM.STYLOS",
    "RPLC"         : "LC.celebro.CMFG.SBSTM.RPLC",
    "VOICE_ENGINE" : "LC.celebro.CMFG.SBSTM.voice_engine",
    "SNSBSTNPRB"   : "LC.celebro.CMFG.SBSTM.SNSBSTNPRB",
    "CELEBRO_INIT" : "LC.celebro",
}

MODULOS_NEURONAS: Dict[str, str] = {
    "ENRN"  : "LC.celebro.red_neuronal.ENRN",
    "RF_SL" : "LC.celebro.red_neuronal.RF_SL",
    "RF_EN" : "LC.celebro.red_neuronal.RF_EN",
    "RNP"   : "LC.celebro.red_neuronal.RNP",
    "SLRN"  : "LC.celebro.red_neuronal.SLRN",
}

NEURONAS_POR_FAMILIA: Dict[str, List[str]] = {
    "ENRN" : [f"LC.celebro.red_neuronal.ENRN.EN{i}_RN" for i in range(1, 11)],
    "RF_SL": [f"LC.celebro.red_neuronal.RF_SL.RFSL1.RF_SL1_{i}" for i in range(1, 11)],
    "RF_EN": [f"LC.celebro.red_neuronal.RF_EN.RFEN1_RN_{i}" for i in range(1, 11)],
    "RNP"  : [f"LC.celebro.red_neuronal.RNP.RN{i}" for i in range(1, 11)],
    "SLRN" : [f"LC.celebro.red_neuronal.SLRN.SL{i}" for i in range(1, 11)],
}


# ─── UTILIDADES DE INTROSPECCIÓN PARA TESTS ───────────────────────────────────
def verificar_importacion(modulo: str) -> Tuple[bool, str]:
    """
    Intenta importar un módulo y devuelve (éxito, mensaje).

    Parameters
    ----------
    modulo : str
        Nombre completo del módulo Python (e.g. 'LC.celebro.BKSVCB').

    Returns
    -------
    Tuple[bool, str]
        (True, "") si la importación fue exitosa.
        (False, mensaje_de_error) si falló.
    """
    try:
        importlib.import_module(modulo)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def listar_modulos_disponibles() -> Dict[str, bool]:
    """
    Devuelve un mapa {nombre_modulo: disponible} para todos los módulos
    core y neuronales definidos en este paquete de tests.

    Útil para generar informes de disponibilidad antes de ejecutar la suite.
    """
    resultado: Dict[str, bool] = {}
    for nombre, ruta in {**MODULOS_CORE, **MODULOS_NEURONAS}.items():
        ok, _ = verificar_importacion(ruta)
        resultado[nombre] = ok
    return resultado


def listar_neuronas_disponibles() -> Dict[str, Dict[str, bool]]:
    """
    Verifica la importabilidad de cada una de las 50 neuronas individuales.

    Returns
    -------
    Dict[str, Dict[str, bool]]
        {familia: {modulo_neurona: disponible}}
    """
    reporte: Dict[str, Dict[str, bool]] = {}
    for familia, modulos in NEURONAS_POR_FAMILIA.items():
        reporte[familia] = {}
        for mod in modulos:
            ok, _ = verificar_importacion(mod)
            reporte[familia][mod] = ok
    return reporte


def contar_neuronas_activas() -> int:
    """Cuenta cuántas de las 50 neuronas se pueden importar correctamente."""
    total = 0
    for familia_mods in NEURONAS_POR_FAMILIA.values():
        for mod in familia_mods:
            ok, _ = verificar_importacion(mod)
            if ok:
                total += 1
    return total


def resumen_suite() -> str:
    """
    Genera un resumen legible de la suite de tests.

    Returns
    -------
    str
        Texto multilínea con el estado de módulos y neuronas.
    """
    lineas = [
        f"{'─' * 64}",
        f"  LucIA TestSuite v{__version__}",
        f"  Neuronas activas: {contar_neuronas_activas()} / {__total_neuronas__}",
        f"{'─' * 64}",
    ]
    modulos = listar_modulos_disponibles()
    for nombre, disponible in modulos.items():
        estado = "✓" if disponible else "✗"
        lineas.append(f"  [{estado}] {nombre}")
    lineas.append(f"{'─' * 64}")
    return "\n".join(lineas)


# ─── CONSTANTES REUTILIZABLES EN TESTS ───────────────────────────────────────
PROMPT_CORTO: str  = "¿Qué es una red neuronal?"
PROMPT_LARGO: str  = (
    "Explica en detalle el funcionamiento de la ortogonalización polar "
    "Newton-Schulz de grado 5 aplicada a tensores sinápticos en una "
    "arquitectura distribuida con consenso Proof of Neural Learning."
)
PROMPT_TECNICO: str = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
PROMPT_VACIO: str   = ""
PROMPT_UNICODE: str = "¿Cómo están las métricas de 派生 y αβγ?"

HASH_GENESIS: str    = "0" * 64
DIFICULTAD_TEST: int = 1   # Dificultad baja para tests rápidos
MAX_ITER_TEST: int   = 5000

# ─── EXPORTACIONES PÚBLICAS ───────────────────────────────────────────────────
__all__: List[str] = [
    "__version__",
    "__suite_name__",
    "__total_neuronas__",
    "TEST_DIR",
    "CELEBRO_DIR",
    "ROOT_DIR",
    "MODULOS_CORE",
    "MODULOS_NEURONAS",
    "NEURONAS_POR_FAMILIA",
    "verificar_importacion",
    "listar_modulos_disponibles",
    "listar_neuronas_disponibles",
    "contar_neuronas_activas",
    "resumen_suite",
    "PROMPT_CORTO",
    "PROMPT_LARGO",
    "PROMPT_TECNICO",
    "PROMPT_VACIO",
    "PROMPT_UNICODE",
    "HASH_GENESIS",
    "DIFICULTAD_TEST",
    "MAX_ITER_TEST",
]
