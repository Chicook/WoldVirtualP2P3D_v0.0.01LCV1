"""
Red Neuronal de Refuerzo RFEN1_RN_9 - Sistema de Mejora Automática de Código
Especializado en análisis de código, refactorización y generación de tests unitarios
"""

from .RF_RFEN1_RN_9_1 import *
from .RF_RFEN1_RN_9_2 import *
from .RF_RFEN1_RN_9_3 import *
from .RF_RFEN1_RN_9_4 import *
from .RF_RFEN1_RN_9_5 import *
from .RF_RFEN1_RN_9_6 import *
from .RF_RFEN1_RN_9_7 import *
from .RF_RFEN1_RN_9_8 import *
from .RF_RFEN1_RN_9_9 import *
from .RF_RFEN1_RN_9_10 import *

__version__ = "1.0.0"
__author__ = "WoldVirtual3D Skin Team"

# Configuración del módulo
MODULE_CONFIG = {
    'module_name': 'RF_RFENRN1_9',
    'description': 'Sistema de Mejora Automática de Código con Análisis Neural',
    'version': __version__,
    'components': [
        'Code Analysis Neuron',
        'Code Refactoring Neuron',
        'Test Generation Neuron',
        'Code Optimization Neuron',
        'Pattern Matching Neuron',
        'Complexity Analyzer Neuron',
        'Security Analyzer Neuron',
        'Performance Analyzer Neuron',
        'Documentation Neuron',
        'Code Quality Neuron'
    ]
}

__all__ = [
    "CodeAnalysisNeuron",
    "CodeRefactoringNeuron",
    "TestGenerationNeuron",
    "CodeOptimizationNeuron",
    "PatternMatchingNeuron",
    "ComplexityAnalyzerNeuron",
    "SecurityAnalyzerNeuron",
    "PerformanceAnalyzerNeuron",
    "DocumentationNeuron",
    "CodeQualityNeuron",
    "MODULE_CONFIG"
]
