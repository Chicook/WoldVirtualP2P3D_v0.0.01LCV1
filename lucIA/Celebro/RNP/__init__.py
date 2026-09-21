"""
RNP/__init__.py - Calibración interna de Celebro: EXACTAMENTE RN11, RN12, RN13 y RN14
=====================================================================================

RNP son cuatro nodos y ningún otro:
- RN11: peso neuronal (OptimizadorIntegradoPesoNeuronal).
- RN12: ajuste dinámico (ModuloAjusteDinamico).
- RN13: controlador de gradientes (ControladorGradientes).
- RN14: puertas de atención por neurona (PuertasAtencion, no computa: modula).

NOTA DE LÍMITES (no repetir la confusión): LAMB y RAdam viven en SLRN
(SL11 y SL12), NUNCA en RNP. Los campos lamb_*/radam_* de
NeuralWeightOptimizationConfig son hiperparámetros genéricos de referencia,
no instancias: las instancias reales están en lucIA.Celebro.SLRN.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time

# Configuración del logger
logger = logging.getLogger(__name__)

# Configuración del módulo RNP
RNP_CONFIG = {
    "module_name": "RNP",
    "version": "2025.1.0",
    "description": "Calibración interna de Celebro: RN11 (peso), RN12 (ajuste), RN13 (gradientes), RN14 (puertas)",
    "author": "LucIA Celebro",
    "created": "2025",
    "nodos": [
        "RN11_OptimizadorPesoNeuronal (peso)",
        "RN12_ModuloAjusteDinamico (ajuste)",
        "RN13_ControladorGradientes (gradientes)",
        "RN14_PuertasAtencion (puertas, no computa)"
    ],
    "libraries": [
        "TensorFlow",
        "PyTorch",
        "NumPy",
        "Pandas",
        "Matplotlib",
        "Seaborn",
        "Optuna",
        "Keras",
        "JAX",
        "TensorFlow Text 3.0"
    ],
    "features": [
        "Optimización de pesos neuronales principales",
        "Salida de resultados en consola",
        "Algoritmos avanzados 2025+",
        "Integración con librerías Python",
        "Sistema de evaluación integral",
        "Análisis comparativo",
        "Recomendaciones automáticas"
    ]
}

@dataclass
class NeuralWeightOptimizationConfig:
    """Configuración para algoritmos de optimización de pesos neuronales"""
    # Parámetros generales
    learning_rate: float = 0.0003
    max_iterations: int = 1500
    batch_size: int = 32
    weight_decay: float = 0.01
    random_state: int = 42
    
    # Parámetros específicos de AdamW
    adamw_beta1: float = 0.8
    adamw_beta2: float = 0.95
    adamw_epsilon: float = 1e-8
    
    # Parámetros específicos de RAdam
    radam_beta1: float = 0.9
    radam_beta2: float = 0.999
    radam_epsilon: float = 1e-8
    
    # Parámetros específicos de Lookahead
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    
    # Parámetros específicos de Nadam
    nadam_beta1: float = 0.9
    nadam_beta2: float = 0.999
    nadam_epsilon: float = 1e-8
    
    # Parámetros específicos de LAMB
    lamb_beta1: float = 0.9
    lamb_beta2: float = 0.999
    lamb_epsilon: float = 1e-8
    
    # Parámetros específicos de AdaBelief
    adabelief_beta1: float = 0.9
    adabelief_beta2: float = 0.999
    adabelief_epsilon: float = 1e-8
    
    # Parámetros específicos de Lion
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    lion_epsilon: float = 1e-8
    
    # Parámetros específicos de SAM
    sam_rho: float = 0.05
    sam_adaptive: bool = True
    
    # Parámetros específicos de SWATS
    swats_switch_iter: int = 100
    swats_switch_threshold: float = 0.1
    
    def __post_init__(self):
        pass

@dataclass
class NeuralWeightOptimizationMetrics:
    """Métricas para algoritmos de optimización de pesos neuronales"""
    algorithm_name: str
    initial_loss: float
    final_loss: float
    convergence_iterations: int
    adamw_weight_decay_efficiency: float
    radam_rectification_stability: float
    lookahead_convergence_speed: float
    nadam_nesterov_acceleration: float
    lamb_layer_wise_adaptation: float
    adabelief_belief_correction: float
    lion_momentum_efficiency: float
    sam_sharpness_awareness: float
    swats_switching_efficiency: float
    neural_weight_integration_score: float
    overall_score: float
    optimization_time: float
    timestamp: str

class BaseNeuralWeightOptimizer(ABC):
    """Clase base para optimizadores de pesos neuronales"""
    
    def __init__(self, config: NeuralWeightOptimizationConfig):
        self.config = config
        self.optimizer = None
        self.history = []
        self.metrics = {}
        
    @abstractmethod
    def create_optimizer(self, model: Any) -> Any:
        """Crea el optimizador específico"""
        pass
    
    @abstractmethod
    def optimize_weights(self, model: Any, data_loader: Any, criterion: Any = None) -> 'NeuralWeightOptimizationResult':
        """Optimiza los pesos del modelo"""
        pass
    
    def _evaluate_model(self, model: Any, data_loader: Any, criterion: Any) -> Dict[str, float]:
        """Evalúa el modelo"""
        try:
            total_loss = 0.0
            total_samples = 0
            correct_predictions = 0
            
            for data, target in data_loader:
                if hasattr(model, 'predict'):
                    predictions = model.predict(data)
                    if hasattr(criterion, '__call__'):
                        loss = criterion(predictions, target)
                        total_loss += loss
                    total_samples += len(target)
                    if hasattr(predictions, 'argmax'):
                        correct_predictions += (predictions.argmax(axis=1) == target).sum()
                else:
                    # Para modelos de scikit-learn
                    predictions = model.predict(data)
                    total_samples += len(target)
                    correct_predictions += (predictions == target).sum()
            
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
            
            return {
                'loss': avg_loss,
                'accuracy': accuracy,
                'samples': total_samples
            }
            
        except Exception as e:
            logger.error(f"Error evaluando modelo: {e}")
            return {'loss': 0.0, 'accuracy': 0.0, 'samples': 0}
    
    def _check_convergence(self, loss_history: List[float], patience: int = 10) -> bool:
        """Verifica convergencia usando el CORE"""
        from lucIA.CORE.utils import check_convergence
        return check_convergence(loss_history, patience)

@dataclass
class NeuralWeightOptimizationResult:
    """Resultado de optimización de pesos neuronales"""
    success: bool
    optimized_model: Optional[Any]
    metrics: Optional[NeuralWeightOptimizationMetrics]
    optimization_history: List[float]
    best_weights: Dict[str, Any]
    theoretical_analysis: Dict[str, Any]
    performance_analysis: Dict[str, Any]
    recommendations: List[str]
    error_message: Optional[str]

class NeuralWeightOptimizationSystem:
    """Sistema de gestión de algoritmos de optimización de pesos neuronales"""
    
    def __init__(self, config: Optional[NeuralWeightOptimizationConfig] = None):
        self.config = config or NeuralWeightOptimizationConfig()
        self.optimizers = {}
        self.results = {}
        
    def register_optimizer(self, name: str, optimizer: BaseNeuralWeightOptimizer):
        """Registra un optimizador"""
        self.optimizers[name] = optimizer
        
    def get_optimizer(self, name: str) -> Optional[BaseNeuralWeightOptimizer]:
        """Obtiene un optimizador"""
        return self.optimizers.get(name)
    
    def list_optimizers(self) -> List[str]:
        """Lista todos los optimizadores disponibles"""
        return list(self.optimizers.keys())
    
    def analyze_performance(self, model: Any, data_loader: Any, criterion: Any = None) -> Dict[str, NeuralWeightOptimizationResult]:
        """Analiza el rendimiento de todos los optimizadores"""
        results = {}
        
        for name, optimizer in self.optimizers.items():
            try:
                result = optimizer.optimize_weights(model, data_loader, criterion)
                results[name] = result
            except Exception as e:
                logger.error(f"Error analizando {name}: {e}")
                results[name] = NeuralWeightOptimizationResult(
                    success=False, optimized_model=None, metrics=None,
                    optimization_history=[], best_weights={},
                    theoretical_analysis={}, performance_analysis={},
                    recommendations=[], error_message=str(e)
                )
        
        return results

def create_integrated_neural_weight_optimizer(config: Optional[NeuralWeightOptimizationConfig] = None) -> NeuralWeightOptimizationSystem:
    """Crea un sistema integrado de optimización de pesos neuronales"""
    return NeuralWeightOptimizationSystem(config)

def print_neural_weight_optimization_results(results: Dict[str, NeuralWeightOptimizationResult]):
    """Imprime los resultados de optimización de pesos neuronales en consola"""
    print("\n" + "="*80)
    print("RESULTADOS DE OPTIMIZACIÓN DE PESOS NEURONALES - RED NEURONAL PRINCIPAL")
    print("="*80)
    
    for name, result in results.items():
        if result.success and result.metrics:
            print(f"\n🧠 {name.upper()}:")
            print(f"   Score General: {result.metrics.overall_score:.4f}")
            print(f"   Pérdida Inicial: {result.metrics.initial_loss:.4f}")
            print(f"   Pérdida Final: {result.metrics.final_loss:.4f}")
            print(f"   Iteraciones: {result.metrics.convergence_iterations}")
            print(f"   Tiempo: {result.metrics.optimization_time:.2f}s")
            
            # Mostrar pesos específicos del algoritmo
            if result.optimized_model and hasattr(result.optimized_model, 'get_weights'):
                print(f"   Pesos Optimizados:")
                weights = result.optimized_model.get_weights()
                for i, weight in enumerate(weights):
                    print(f"     Capa {i}: {weight.shape} - Min: {weight.min():.4f}, Max: {weight.max():.4f}")
            
            # Mostrar recomendaciones
            if result.recommendations:
                print(f"   Recomendaciones:")
                for rec in result.recommendations:
                    print(f"     • {rec}")
        else:
            print(f"\n❌ {name.upper()}: Error - {result.error_message}")
    
    print("\n" + "="*80)
    print("ANÁLISIS COMPLETO DE PESOS NEURONALES FINALIZADO")
    print("="*80)

logger.debug("RNP/__init__.py - Configuración de Optimización de Pesos Neuronales 2025+ cargada exitosamente")
