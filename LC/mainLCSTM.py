"""
mainLCSTM.py - Orquestador Central del Sistema Cognitivo LucIA (2026)
Clase delgada: compone PRTLUC + TRNLUC + CMDLUC. Coordinacion maestro.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from LC.celebro.CMFG.SBSTM.BASELUC import EstiloTerminalLucIA, get_cliente_iafree
from LC.celebro.CMFG.SBSTM.CMDLUC import CMDLUCMixin
from LC.celebro.CMFG.SBSTM.PRTLUC import PRTLUCMixin
from LC.celebro.CMFG.SBSTM.TRNLUC import TRNLUCMixin


class OrquestadorSistemaLucIA(PRTLUCMixin, TRNLUCMixin, CMDLUCMixin):
    """
    Orquestador maestro que integra BKSVCB, SNSBSTNPRB, IAFREE, STYLOS y las 50 neuronas.
    """

# ─── GESTOR DE CONTEXTO PARA INTEGRACIONES EXTERNAS ─────────────────────────
class ContextoOrquestadorLucIA:
    """Gestor de contexto para pruebas, evaluacion o invocacion programatica."""

    def __init__(self) -> None:
        self.orquestador = OrquestadorSistemaLucIA()

    def __enter__(self) -> OrquestadorSistemaLucIA:
        if not self.orquestador.inicializar_subsistemas():
            raise RuntimeError("Fallo durante la inicializacion de subsistemas en LucIA")
        return self.orquestador

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        self.orquestador.cerrar_sistema()
        return False


# ─── SERVICIOS DE CONSULTA Y DIAGNOSTICO PROGRAMATICO ───────────────────────
def obtener_diagnostico_orquestador() -> Dict[str, Any]:
    """Genera un reporte estructural para validar la salud de todo el stack."""
    from LC.celebro.CMFG.SBSTM.IAFREE import get_cliente_iafree
    cliente_free = get_cliente_iafree()
    return {
        "orquestador_version": "2026.3.1",
        "iafree_activo": cliente_free.esta_autenticado(),
        "modelos_gratuitos_total": len(cliente_free.gestor.listar_modelos()),
        "modelo_predeterminado": cliente_free.gestor.obtener_modelo_activo()["id"],
        "costo_acumulado": cliente_free.obtener_metricas_consumo()["costo_acumulado_usd"],
        "timestamp": time.time(),
    }


# ─── FUNCION PRINCIPAL DE ENTRADA AL SISTEMA ─────────────────────────────────
def main() -> int:
    """Punto de arranque del orquestador unificado mainLCSTM."""
    orquestador = OrquestadorSistemaLucIA()
    if not orquestador.inicializar_subsistemas():
        sys.stderr.write("[mainLCSTM] Fallo durante la inicializacion de subsistemas.\n")
        return 1

    try:
        orquestador.ejecutar_bucle_interactivo()
        return 0
    except Exception as err:
        sys.stderr.write(f"[mainLCSTM] Error no controlado: {err}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())