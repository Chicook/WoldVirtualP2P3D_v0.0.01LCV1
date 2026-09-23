"""
PRTLUC.py - Arranque del orquestador LucIA (2026)
__init__, inicializar_subsistemas (pasos 0-8) y helpers de IA local.
Mixin de OrquestadorSistemaLucIA; importa desde BASELUC.
"""
from __future__ import annotations

from LC.celebro.CMFG.SBSTM.BASELUC import (
    Any, Dict, List, Optional, Tuple, atexit, logger, os, threading, time,
    EstiloTerminalLucIA, GestorEntornoSeguro, RefactorizadorSesion,
    _HRCTRC_DISPONIBLE, _HRCTRC_RFCT_DISPONIBLE, _IALOCAL_DISPONIBLE,
    _INTEGRACIONRF_DISPONIBLE, _MDSTM_DISPONIBLE, _descargar_recomendados,
    _perfilar_hw_local, _recomendar_ia_local, banner_bienvenida,
    consultar_lucia_local, get_cliente_iafree, get_gestor_hrctrc,
    get_gestor_mdstm, get_integrador, refactorizar_overlay,
    _ROTACIONIA_DISPONIBLE, get_rotador_ia,
    _NEUROSINTESIS_DISPONIBLE, get_sintetizador_lucia,
    _APRENDIZAJE_SEGURO_DISPONIBLE, get_aprendizaje_seguro,
)
from PRTLUC_PRTLUCMixin import PRTLUCMixin  # CLASSPACK
