"""
RF_EN - Reinforcement Learning Neurons Package
==============================================

This package contains reinforcement learning neuron implementations for LucIA.

Autor: LucIA Development Team
Versión: 1.0.0
"""

__version__ = "1.0.0"

# Exportar submódulos de neuronas de refuerzo
try:
    from . import RFENRN1
    from . import RFENRN2
    from . import RFENRN3
    from . import RFENRN4
    from . import RFENRN5
    from . import RFENRN6
    from . import RFENRN7
    from . import RFENRN8
    from . import RFENRN9
    from . import RFENRN10

    __all__ = [
        'RFENRN1',
        'RFENRN2',
        'RFENRN3',
        'RFENRN4',
        'RFENRN5',
        'RFENRN6',
        'RFENRN7',
        'RFENRN8',
        'RFENRN9',
        'RFENRN10'
    ]
except ImportError as e:
    import warnings
    warnings.warn(f"Error al importar módulos RF_EN: {e}")
    __all__ = []
