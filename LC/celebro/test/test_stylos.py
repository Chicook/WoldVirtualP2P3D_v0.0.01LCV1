"""
test_stylos.py — Suite exhaustiva de pruebas para el subsistema STYLOS de LucIA
=============================================================================
Cubre el motor de estilos visuales y renderizado de terminal (Arquitectura 2026):
  1. Constantes de color ANSI y paleta cyber-bioluminiscente (ColoresLucIA).
  2. Glifos y caracteres estructurales Unicode (Glifos).
  3. Utilidades de cálculo y saneamiento (limpiar_codigos_ansi, longitud_visual, obtener_ancho_consola).
  4. Renderizador Markdown: sintaxis inline (bold, italic, code), encabezados y tablas estructuradas.
  5. Tarjetas y componentes de UI en terminal (EstiloTerminalLucIA):
     - Encabezados de sesión y bienvenida (banner_bienvenida).
     - Tarjetas de prompt de usuario.
     - Streaming token a token y apertura/cierre de respuesta.
     - Renderizado de respuesta completa y paneles de ayuda.
     - Badges de telemetría sináptica/blockchain y notificaciones de bloque PoNL.
  6. Casos borde de formato (tablas asimétricas, markdown anidado, textos multilínea).
  7. Métodos de conveniencia y atajos globales del subsistema.
"""
from __future__ import annotations

import io
import re
import sys
from typing import Any, Dict, List
import pytest

from LC.celebro.CMFG.SBSTM.STYLOS import (
    ColoresLucIA,
    EstiloTerminalLucIA,
    Glifos,
    RenderizadorMarkdown,
    badge_turno,
    banner_bienvenida,
    formatear_respuesta_lucia,
    limpiar_codigos_ansi,
    longitud_visual,
    obtener_ancho_consola,
    panel_ayuda_comandos,
    tarjeta_diagnostico,
)


# ============================================================================
# 1. CONSTANTES DE COLOR Y PALETA CYBER-BIOLUMINISCENTE
# ============================================================================

from test_stylos_TestColoresLucIA import TestColoresLucIA  # CLASSPACK
from test_stylos_TestGlifos import TestGlifos  # CLASSPACK
from test_stylos_TestUtilidadesVisuales import TestUtilidadesVisuales  # CLASSPACK
from test_stylos_TestRenderizadorMarkdown import TestRenderizadorMarkdown  # CLASSPACK
from test_stylos_TestEstiloTerminalLucIA import TestEstiloTerminalLucIA  # CLASSPACK
from test_stylos_TestMetodosGlobalesConveniencia import TestMetodosGlobalesConveniencia  # CLASSPACK
