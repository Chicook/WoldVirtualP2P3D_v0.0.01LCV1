"""
Red Neuronal de Refuerzo RFEN1_RN_10 - Sistema 3D y Avatares
Especializado en entornos virtuales, avatares 3D y optimización de pesos para 3D
"""

from .RF_RFEN1_RN_10_1 import *
from .RF_RFEN1_RN_10_2 import *
from .RF_RFEN1_RN_10_3 import *
from .RF_RFEN1_RN_10_4 import *
from .RF_RFEN1_RN_10_5 import *
from .RF_RFEN1_RN_10_6 import *
from .RF_RFEN1_RN_10_7 import *
from .RF_RFEN1_RN_10_8 import *
from .RF_RFEN1_RN_10_9 import *
from .RF_RFEN1_RN_10_10 import *

__version__ = "1.0.0"

# Configuración del módulo
MODULE_CONFIG = {
    'module_name': 'RF_RFENRN1_10',
    'description': 'Sistema de Refuerzo Neural para Avatares y Entornos 3D',
    'version': __version__,
    'components': [
        'Avatar3DGenerator',
        'VirtualEnvironmentRenderer',
        'Physics3DNeuron',
        'WeightOptimization3D',
        'Networking3DNeuron',
        'Animation3DController',
        'CollisionDetectionNeuron',
        'RaycastingNeuron',
        'ShadingLightingNeuron',
        'PerformanceOptimization3D'
    ]
}

__all__ = [
    "Avatar3DGenerator",
    "VirtualEnvironmentRenderer",
    "Physics3DNeuron",
    "WeightOptimization3D",
    "Networking3DNeuron",
    "Animation3DController",
    "CollisionDetectionNeuron",
    "RaycastingNeuron",
    "ShadingLightingNeuron",
    "PerformanceOptimization3D"
]
