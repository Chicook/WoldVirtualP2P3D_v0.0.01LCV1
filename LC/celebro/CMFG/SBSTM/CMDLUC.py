"""
CMDLUC.py - Consola y cierre de LucIA (2026)
Bucle interactivo, comandos _cmd_*, estado y cerrar_sistema.
Mixin de OrquestadorSistemaLucIA; importa desde BASELUC.
"""
from __future__ import annotations

from LC.celebro.CMFG.SBSTM.BASELUC import (
    Any, Dict, List, Optional, EstiloTerminalLucIA, _IALOCAL_DISPONIBLE,
    _estado_ia_local, badge_turno, panel_ayuda_comandos,
)
from CMDLUC_CMDLUCMixin import CMDLUCMixin  # CLASSPACK
