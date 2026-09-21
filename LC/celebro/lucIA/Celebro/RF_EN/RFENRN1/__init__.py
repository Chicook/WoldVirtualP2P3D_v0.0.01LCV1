"""
RFENRN1 - Redes de Refuerzo Especializadas para LucIA
=====================================================

Este paquete contiene neuronas de refuerzo especializadas para la inteligencia artificial LucIA.
Implementa los algoritmos de refuerzo más avanzados de 2025 con optimización de pesos.

Características:
- Algoritmos de refuerzo modernos (Q-Learning, Policy Gradient, Actor-Critic, etc.)
- Optimización de pesos específica para cada algoritmo
- Monitoreo de convergencia y estabilidad
- Adaptación automática de hiperparámetros

Autor: LucIA Development Team
Versión: 1.0.0
Fecha: 2025
"""

from .RFEN1_RN_10 import NeuronaRefuerzoIMPALA
from .RFEN1_RN_9 import NeuronaRefuerzoRainbowDQN
from .RFEN1_RN_8 import NeuronaRefuerzoTD3
from .RFEN1_RN_7 import NeuronaRefuerzoSAC
from .RFEN1_RN_6 import NeuronaRefuerzoPPO
from .RFEN1_RN_5 import NeuronaRefuerzoA3C
from .RFEN1_RN_4 import NeuronaRefuerzoDQN
from .RFEN1_RN_3 import NeuronaRefuerzoActorCritic
from .RFEN1_RN_2 import NeuronaRefuerzoPolicyGradient
from .RFEN1_RN_1 import NeuronaRefuerzoQLearning
import logging

# Importar las clases base y funciones de utilidad desde base.py
from .base import (
    NeuronaRefuerzoBase,
    LUCIA_RL_CONFIG,
    inicializar_pesos_he,
    inicializar_pesos_xavier,
    inicializar_pesos_lecun,
    inicializar_pesos_ortogonal,
    inicializar_pesos_espectral,
    calcular_recompensa_descontada,
    normalizar_recompensas,
    calcular_gae,
    aplicar_clip_gradientes,
    crear_buffer_experiencia,
    muestrear_buffer
)

# Logging centralizado en lucIA/__init__.py; aquí solo logger local.
logger = logging.getLogger('RFENRN1')
logger.debug("Paquete RFENRN1 inicializado correctamente para LucIA Reinforcement Learning")


# ============================================================================
# IMPORTAR NEURONAS DE REFUERZO (después de definir clases base)
# ============================================================================

# Importar todas las neuronas de refuerzo

__all__ = [
    'NeuronaRefuerzoBase',
    'NeuronaRefuerzoQLearning',
    'NeuronaRefuerzoPolicyGradient',
    'NeuronaRefuerzoActorCritic',
    'NeuronaRefuerzoDQN',
    'NeuronaRefuerzoA3C',
    'NeuronaRefuerzoPPO',
    'NeuronaRefuerzoSAC',
    'NeuronaRefuerzoTD3',
    'NeuronaRefuerzoRainbowDQN',
    'NeuronaRefuerzoIMPALA',
    'LUCIA_RL_CONFIG',
    'inicializar_pesos_he',
    'inicializar_pesos_xavier',
    'inicializar_pesos_lecun',
    'inicializar_pesos_ortogonal',
    'inicializar_pesos_espectral',
    'calcular_recompensa_descontada',
    'normalizar_recompensas',
    'calcular_gae',
    'aplicar_clip_gradientes',
    'crear_buffer_experiencia',
    'muestrear_buffer'
]
