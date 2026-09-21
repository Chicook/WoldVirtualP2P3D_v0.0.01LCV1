"""
RF_RFENRN1_4 - Sistema Avanzado de Optimización de Pesos Neuronales
==================================================================

Módulo especializado en algoritmos avanzados de mejora y optimización de pesos
para redes neuronales de aprendizaje por refuerzo. Implementa técnicas desde
métodos clásicos hasta algoritmos de vanguardia de 2025.

Arquitectura Modular:
- RF_RFENRN1_4_1: Optimizadores Clásicos (SGD, Adam, RMSprop, Adagrad, Adadelta)
- RF_RFENRN1_4_2: Optimizadores Adaptativos Modernos (Lion, AdaBelief, RAdam, AdamW)
- RF_RFENRN1_4_3: Algoritmos de Segunda Orden (L-BFGS, Newton, Shampoo)
- RF_RFENRN1_4_4: Técnicas de Regularización Avanzada (Dropout Adaptativo, Weight Decay)
- RF_RFENRN1_4_5: Meta-Optimización y AutoML para Pesos (Bayesian, NAS)
- RF_RFENRN1_4_6: Algoritmos Evolutivos (PSO, GA, DE)
- RF_RFENRN1_4_7: Técnicas de Pruning y Sparse Optimization (Lottery Ticket, Knowledge Distillation)
- RF_RFENRN1_4_8: Análisis de Convergencia y Estabilidad (Gradient Analysis, Hessian)
- RF_RFENRN1_4_9: Optimización Distribuida y Paralela (Federated Learning, AllReduce)
- RF_RFENRN1_4_10: Técnicas Avanzadas de Optimización (Quantum, Physics-Inspired, Multi-Objective)

Características Principales:
- Más de 50 algoritmos de optimización implementados
- Soporte para optimización multi-objetivo y distribuida
- Integración con PyTorch, TensorFlow y frameworks nativos
- Análisis automático de convergencia y estabilidad
- Técnicas de pruning y compresión de modelos
- Optimización cuántica y basada en física
- Sistema de logging y métricas avanzadas
- Configuración adaptativa basada en arquitectura

Autor: LucIA Development Team
Versión: 4.0.0
Fecha: Enero 2025
"""

from .RF_RFENRN1_4_1 import ClassicalOptimizerManager
from .RF_RFENRN1_4_2 import AdaptiveOptimizerManager
from .RF_RFENRN1_4_3 import SecondOrderOptimizerManager
from .RF_RFENRN1_4_4 import RegularizationManager
from .RF_RFENRN1_4_5 import MetaOptimizationManager
from .RF_RFENRN1_4_6 import EvolutionaryOptimizerManager
from .RF_RFENRN1_4_7 import PruningOptimizerManager
from .RF_RFENRN1_4_8 import ConvergenceOptimizerManager
from .RF_RFENRN1_4_9 import DistributedOptimizerManager
from .RF_RFENRN1_4_10 import AdvancedOptimizationManager

__version__ = "4.0.0"
__author__ = "LucIA Development Team"
__all__ = [
    "ClassicalOptimizerManager",
    "AdaptiveOptimizerManager",
    "SecondOrderOptimizerManager",
    "RegularizationManager",
    "MetaOptimizationManager",
    "EvolutionaryOptimizerManager",
    "PruningOptimizerManager",
    "ConvergenceOptimizerManager",
    "DistributedOptimizerManager",
    "AdvancedOptimizationManager"
]

# Configuración del módulo
MODULE_CONFIG = {
    'module_name': 'RF_RFENRN1_4',
    'description': 'Sistema Avanzado de Optimización de Pesos Neuronales',
    'version': __version__,
    'components': [
        'Classical Optimizer Manager',
        'Adaptive Optimizer Manager',
        'Second Order Optimizer Manager',
        'Regularization Manager',
        'Meta Optimization Manager',
        'Evolutionary Optimizer Manager',
        'Pruning Optimizer Manager',
        'Convergence Optimizer Manager',
        'Distributed Optimizer Manager',
        'Advanced Optimization Manager'
    ]
}


def get_module_info():
    """Retorna información del módulo RF_RFENRN1_4"""
    return {
        "name": "RF_RFENRN1_4",
        "version": __version__,
        "author": __author__,
        "description": "Sistema Avanzado de Optimización de Pesos Neuronales",
        "algorithms_count": 50,
        "supported_frameworks": ["PyTorch", "TensorFlow", "JAX", "NumPy"],
        "features": [
            "Optimización Clásica",
            "Optimización Adaptativa",
            "Segunda Orden",
            "Regularización Avanzada",
            "Meta-Optimización",
            "Algoritmos Evolutivos",
            "Pruning Inteligente",
            "Análisis de Convergencia",
            "Optimización Distribuida",
            "Técnicas Avanzadas"
        ]
    }
