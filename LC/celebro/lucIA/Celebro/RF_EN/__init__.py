"""
RF_EN - Reinforcement Learning Neurons Package
==============================================

This package contains reinforcement learning neuron implementations for LucIA.

Autor: LucIA Development Team
Versión: 1.0.0
"""

__version__ = "1.0.0"

# Exportar submódulos de neuronas de refuerzo de forma limpia
_submodulos = [
    'RFENRN1', 'RFENRN2', 'RFENRN3', 'RFENRN4', 'RFENRN5',
    'RFENRN6', 'RFENRN7', 'RFENRN8', 'RFENRN9', 'RFENRN10'
]
__all__ = []

for _nombre in _submodulos:
    try:
        _mod = __import__(f"{__name__}.{_nombre}", fromlist=[_nombre])
        globals()[_nombre] = _mod
        __all__.append(_nombre)
    except (ImportError, ModuleNotFoundError):
        # Módulos con dependencias opcionales pesadas (torch, scipy) operan en modo ligero
        pass
