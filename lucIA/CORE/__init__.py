"""
lucIA.CORE - Núcleo base de la Inteligencia Artificial LucIA
"""

from .base import LucIANeuronBase, LucIASystem
from .initializers import he_initialization, xavier_initialization, lecun_initialization
from .utils import check_convergence, normalize_data

__all__ = [
    'LucIANeuronBase',
    'LucIASystem',
    'he_initialization',
    'xavier_initialization',
    'lecun_initialization',
    'check_convergence',
    'normalize_data'
]
