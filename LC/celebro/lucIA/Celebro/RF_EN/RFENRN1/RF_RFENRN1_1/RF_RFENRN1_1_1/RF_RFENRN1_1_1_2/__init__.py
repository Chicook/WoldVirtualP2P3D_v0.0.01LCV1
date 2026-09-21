"""
RF_RFENRN1_1_1_2 - Neuronas de Refuerzo para Optimización de Latencia y P2P
=====================================================================

Neuronas especializadas en:
- Optimización matemática de latencia en redes neuronales
- Aceleración de cálculos y pesos
- Red P2P descentralizada
- DHT (Distributed Hash Table)
- NAT Traversal
- Optimización de ancho de banda
- Priorización de peers
"""

__version__ = "0.1.0"

# Importar todas las clases - try/except para imports relativos y absolutos
try:
    # Intentar imports relativos (cuando se importa como paquete)
    from .RF_RF_RF_RFEN1_RN_1_1_2_1 import LatencyOptimizationNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_2 import WeightAccelerationNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_3 import MathematicalAccelerationNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_4 import DHTImplementationNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_5 import NATTraversalNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_6 import P2PNetworkOptimizerNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_7 import PeerPrioritizationNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_8 import DistributedStorageNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_9 import NetworkLatencyPredictorNeuron
    from .RF_RF_RF_RFEN1_RN_1_1_2_10 import BandwidthAllocatorNeuron
except (ImportError, ValueError):
    # Fallback a imports absolutos (cuando se ejecuta directamente)
    import sys
    import os
    # Agregar el directorio actual al path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    from RF_RF_RF_RFEN1_RN_1_1_2_1 import LatencyOptimizationNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_2 import WeightAccelerationNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_3 import MathematicalAccelerationNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_4 import DHTImplementationNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_5 import NATTraversalNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_6 import P2PNetworkOptimizerNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_7 import PeerPrioritizationNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_8 import DistributedStorageNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_9 import NetworkLatencyPredictorNeuron
    from RF_RF_RF_RFEN1_RN_1_1_2_10 import BandwidthAllocatorNeuron

__all__ = [
    'LatencyOptimizationNeuron',
    'WeightAccelerationNeuron',
    'MathematicalAccelerationNeuron',
    'DHTImplementationNeuron',
    'NATTraversalNeuron',
    'P2PNetworkOptimizerNeuron',
    'PeerPrioritizationNeuron',
    'DistributedStorageNeuron',
    'NetworkLatencyPredictorNeuron',
    'BandwidthAllocatorNeuron'
]

# Configuración del módulo
MODULE_CONFIG = {
    'module_name': 'RF_RFENRN1_1_1_2',
    'description': 'Sistema de Optimización de Latencia y Red P2P',
    'version': __version__,
    'components': [
        'LatencyOptimizationNeuron',
        'WeightAccelerationNeuron',
        'MathematicalAccelerationNeuron',
        'DHTImplementationNeuron',
        'NATTraversalNeuron',
        'P2PNetworkOptimizerNeuron',
        'PeerPrioritizationNeuron',
        'DistributedStorageNeuron',
        'NetworkLatencyPredictorNeuron',
        'BandwidthAllocatorNeuron'
    ]
}
