"""
TRNLUC.py - Turno de dialogo de LucIA (2026)
Pipeline por turno: fases locales, IAFREE, cierre, reflejo y telemetria.
Mixin de OrquestadorSistemaLucIA; importa desde BASELUC.
"""
from __future__ import annotations

from LC.celebro.CMFG.SBSTM.BASELUC import (
    Any, Dict, Optional, time, EstiloTerminalLucIA, _RPLC_DISPONIBLE,
    _VOZ_DISPONIBLE, _hablar_voz, badge_turno, formatear_respuesta_lucia,
    reprocesar_con_metricas,
)
from LC.celebro.CMFG.SBSTM.TRNLUC_TRNLUCMixin import TRNLUCMixin  # CLASSPACK
