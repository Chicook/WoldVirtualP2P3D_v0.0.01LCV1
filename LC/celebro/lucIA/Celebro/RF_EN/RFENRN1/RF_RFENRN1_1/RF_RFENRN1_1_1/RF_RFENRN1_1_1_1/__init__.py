"""
RF_RFENRN1_1_1_1 - Neuronas de Refuerzo para Simulación OpenSim
=============================================================

Neuronas especializadas en:
- Bucle de simulación OpenSimulator
- Integración monolítica sin protocolo de red
- Métricas de rendimiento (FPS, memoria, CPU)
- Actuador de IA basado en Calidad Operacional
- Optimización de nivel de detalle (LOD)
"""

__version__ = "0.1.0"

# Importar todas las clases - try/except para imports relativos y absolutos
try:
    # Intentar imports relativos (cuando se importa como paquete)
    from .RF_RF_RF_RFEN1_RN_1_1_1_1 import SimulationLoopManager
    from .RF_RF_RF_RFEN1_RN_1_1_1_2 import MetaverseMetricsCollector
    from .RF_RF_RF_RFEN1_RN_1_1_1_3 import PerformanceActuator
    from .RF_RF_RF_RFEN1_RN_1_1_1_4 import LODOptimizer
    from .RF_RF_RF_RFEN1_RN_1_1_1_5 import MemoryManager
    from .RF_RF_RF_RFEN1_RN_1_1_1_6 import CPUMonitor
    from .RF_RF_RF_RFEN1_RN_1_1_1_7 import FPSController
    from .RF_RF_RF_RFEN1_RN_1_1_1_8 import PrimCounter
    from .RF_RF_RF_RFEN1_RN_1_1_1_9 import ScriptAnalyzer
    from .RF_RF_RF_RFEN1_RN_1_1_1_10 import QODecisionMaker
except (ImportError, ValueError):
    # Fallback a imports absolutos (cuando se ejecuta directamente)
    import sys
    import os
    # Agregar el directorio actual al path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from RF_RF_RF_RFEN1_RN_1_1_1_1 import SimulationLoopManager
    from RF_RF_RF_RFEN1_RN_1_1_1_2 import MetaverseMetricsCollector
    from RF_RF_RF_RFEN1_RN_1_1_1_3 import PerformanceActuator
    from RF_RF_RF_RFEN1_RN_1_1_1_4 import LODOptimizer
    from RF_RF_RF_RFEN1_RN_1_1_1_5 import MemoryManager
    from RF_RF_RF_RFEN1_RN_1_1_1_6 import CPUMonitor
    from RF_RF_RF_RFEN1_RN_1_1_1_7 import FPSController
    from RF_RF_RF_RFEN1_RN_1_1_1_8 import PrimCounter
    from RF_RF_RF_RFEN1_RN_1_1_1_9 import ScriptAnalyzer
    from RF_RF_RF_RFEN1_RN_1_1_1_10 import QODecisionMaker

__all__ = [
    'SimulationLoopManager',
    'MetaverseMetricsCollector',
    'PerformanceActuator',
    'LODOptimizer',
    'MemoryManager',
    'CPUMonitor',
    'FPSController',
    'PrimCounter',
    'ScriptAnalyzer',
    'QODecisionMaker'
]

# Configuración del módulo
MODULE_CONFIG = {
    'module_name': 'RF_RFENRN1_1_1_1',
    'description': 'Sistema de Simulación OpenSim con Optimización basada en QO',
    'version': __version__,
    'components': [
        'SimulationLoopManager',
        'MetaverseMetricsCollector',
        'PerformanceActuator',
        'LODOptimizer',
        'MemoryManager',
        'CPUMonitor',
        'FPSController',
        'PrimCounter',
        'ScriptAnalyzer',
        'QODecisionMaker'
    ]
}
