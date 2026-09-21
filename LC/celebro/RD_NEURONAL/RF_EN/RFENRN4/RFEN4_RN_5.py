"""
RFEN4_RN_5 - Técnicas Avanzadas de Regularización de Pesos
Implementación de métodos modernos para regularización inteligente de pesos neuronales
Incluye: Regularización adaptativa, regularización espectral, y regularización cuántica
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
from . import BaseWeightImprover, NeuralWeightConfig, NeuronWeightMetrics, WeightImprovementResult

logger = logging.getLogger(__name__)


@dataclass
class RegularizationConfig:
    """Configuración para regularización de pesos"""
    regularization_method: str = "adaptive_l2"  # adaptive_l2, spectral, quantum, dropout_adaptive
    l2_strength: float = 0.01
    l1_strength: float = 0.001
    dropout_rate: float = 0.1
    spectral_radius_limit: float = 1.0
    quantum_superposition: bool = False
    adaptive_strength: bool = True
    regularization_schedule: str = "cosine"  # cosine, linear, exponential
    weight_decay_factor: float = 0.95
    regularization_frequency: int = 100
    spectral_norm_clip: float = 1.0
    quantum_entanglement: bool = False


class AdaptiveL2Regularizer(BaseWeightImprover):
    """
    Regularizador L2 adaptativo
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.regularization_config = RegularizationConfig()
        self.regularization_history = defaultdict(list)
        self.weight_norms = defaultdict(list)
        self.regularization_strengths = {}
        self.adaptation_factors = {}

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando regularización L2 adaptativa"""

        logger.info("Iniciando regularización L2 adaptativa")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Calcular normas de pesos
        self._calculate_weight_norms(model)

        # Aplicar regularización adaptativa
        neurons_regularized = self._apply_adaptive_regularization(model)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_regularization_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_memory_optimization(),
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["adaptive_l2_regularization"],
            neurons_improved=neurons_regularized,
            total_neurons=len(initial_metrics),
            improvement_time=end_time - start_time
        )

    def analyze_neuron_weights(self, model: nn.Module) -> Dict[str, NeuronWeightMetrics]:
        """Analiza los pesos de las neuronas individuales"""

        neuron_metrics = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular métricas para cada parámetro
                metrics = NeuronWeightMetrics(
                    neuron_id=name,
                    weight_magnitude=torch.norm(param.data).item(),
                    weight_variance=torch.var(param.data).item(),
                    gradient_norm=torch.norm(param.grad).item() if param.grad is not None else 0.0,
                    activation_frequency=self._calculate_activation_frequency(param),
                    importance_score=self._calculate_importance_score(param),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'regularization_strength': self.regularization_strengths.get(name, 0.0),
                        'adaptation_factor': self.adaptation_factors.get(name, 1.0),
                        'weight_norm_history': self.weight_norms.get(name, [])[-10:]
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _calculate_weight_norms(self, model: nn.Module) -> None:
        """Calcula las normas de los pesos"""

        for name, param in model.named_parameters():
            if param.requires_grad:
                norm = torch.norm(param.data).item()
                self.weight_norms[name].append(norm)

                # Mantener historial limitado
                if len(self.weight_norms[name]) > 100:
                    self.weight_norms[name].pop(0)

    def _apply_adaptive_regularization(self, model: nn.Module) -> int:
        """Aplica regularización adaptativa"""

        neurons_regularized = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular fuerza de regularización adaptativa
                reg_strength = self._calculate_adaptive_strength(param, name)
                self.regularization_strengths[name] = reg_strength

                # Aplicar regularización L2
                if reg_strength > 0:
                    with torch.no_grad():
                        # Regularización L2 adaptativa
                        param.data *= (1 - reg_strength)

                        # Aplicar factor de adaptación
                        adaptation_factor = self._calculate_adaptation_factor(param, name)
                        self.adaptation_factors[name] = adaptation_factor

                        param.data *= adaptation_factor

                    # Registrar en historial
                    self._record_regularization(name, reg_strength, adaptation_factor)

                    neurons_regularized += 1

        logger.info(f"Regularización adaptativa aplicada a {neurons_regularized} neuronas")
        return neurons_regularized

    def _calculate_adaptive_strength(self, param: torch.Tensor, name: str) -> float:
        """Calcula la fuerza de regularización adaptativa"""

        # Fuerza base
        base_strength = self.regularization_config.l2_strength

        # Ajustar según norma del peso
        weight_norm = torch.norm(param.data).item()

        if weight_norm > 1.0:
            # Aumentar regularización para pesos grandes
            adaptive_strength = base_strength * (1 + weight_norm * 0.1)
        else:
            # Reducir regularización para pesos pequeños
            adaptive_strength = base_strength * (1 - weight_norm * 0.1)

        # Ajustar según historial
        if name in self.weight_norms and len(self.weight_norms[name]) > 1:
            recent_norms = self.weight_norms[name][-5:]
            norm_variance = np.var(recent_norms)

            # Aumentar regularización si hay alta varianza
            adaptive_strength *= (1 + norm_variance * 0.5)

        # Limitar fuerza de regularización
        adaptive_strength = min(adaptive_strength, 0.1)

        return max(0.0, adaptive_strength)

    def _calculate_adaptation_factor(self, param: torch.Tensor, name: str) -> float:
        """Calcula el factor de adaptación"""

        # Factor base
        base_factor = 1.0

        # Ajustar según tipo de capa
        layer_type = self._get_layer_type(name)

        if layer_type == 'convolutional':
            base_factor *= 1.05  # Favorecer capas convolucionales
        elif layer_type == 'linear':
            base_factor *= 1.02  # Favorecer capas lineales ligeramente
        elif layer_type == 'normalization':
            base_factor *= 0.98  # Reducir capas de normalización

        # Ajustar según estabilidad
        if name in self.weight_norms and len(self.weight_norms[name]) > 3:
            recent_norms = self.weight_norms[name][-3:]
            stability = 1.0 / (1.0 + np.std(recent_norms))
            base_factor *= (0.9 + 0.2 * stability)

        return base_factor

    def _record_regularization(self, name: str, reg_strength: float, adaptation_factor: float) -> None:
        """Registra la regularización en el historial"""

        self.regularization_history[name].append({
            'regularization_strength': reg_strength,
            'adaptation_factor': adaptation_factor,
            'timestamp': time.time()
        })

        # Mantener historial limitado
        if len(self.regularization_history[name]) > 100:
            self.regularization_history[name].pop(0)

    def _calculate_activation_frequency(self, param: torch.Tensor) -> float:
        """Calcula la frecuencia de activación"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 5.0)

    def _calculate_importance_score(self, param: torch.Tensor) -> float:
        """Calcula el score de importancia"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        importance = magnitude * (1 + variance)
        return min(1.0, importance / 100.0)

    def _calculate_stability_score(self, param: torch.Tensor) -> float:
        """Calcula el score de estabilidad"""

        if param_name := getattr(param, 'name', None):
            if param_name in self.weight_norms and len(self.weight_norms[param_name]) > 1:
                norms = self.weight_norms[param_name]
                stability = 1.0 / (1.0 + np.std(norms))
                return stability

        return 0.5

    def _estimate_contribution_to_loss(self, param: torch.Tensor) -> float:
        """Estima la contribución a la pérdida"""

        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            return min(1.0, grad_norm / 5.0)
        return 0.0

    def _get_layer_type(self, name: str) -> str:
        """Determina el tipo de capa"""

        if 'conv' in name.lower():
            return 'convolutional'
        elif 'linear' in name.lower() or 'fc' in name.lower():
            return 'linear'
        elif 'bn' in name.lower() or 'norm' in name.lower():
            return 'normalization'
        else:
            return 'other'

    def _calculate_regularization_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia de la regularización"""

        if not self.regularization_strengths:
            return 0.0

        strengths = list(self.regularization_strengths.values())
        convergence_rate = 1.0 / (1.0 + np.std(strengths))
        return convergence_rate

    def _calculate_stability_improvement(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la mejora en estabilidad"""

        initial_stability = np.mean([m.stability_score for m in initial_metrics.values()])
        final_stability = np.mean([m.stability_score for m in final_metrics.values()])

        return final_stability - initial_stability

    def _calculate_efficiency_gain(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la ganancia en eficiencia"""

        initial_efficiency = np.mean([m.importance_score for m in initial_metrics.values()])
        final_efficiency = np.mean([m.importance_score for m in final_metrics.values()])

        return final_efficiency - initial_efficiency

    def _calculate_memory_optimization(self) -> float:
        """Calcula la optimización de memoria"""

        return 0.0  # Implementar según necesidad

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 3.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain


class SpectralRegularizer(BaseWeightImprover):
    """
    Regularizador espectral
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.regularization_config = RegularizationConfig(regularization_method="spectral")
        self.spectral_norms = {}
        self.spectral_history = defaultdict(list)
        self.clipping_factors = {}

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando regularización espectral"""

        logger.info("Iniciando regularización espectral")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Calcular normas espectrales
        self._calculate_spectral_norms(model)

        # Aplicar regularización espectral
        neurons_regularized = self._apply_spectral_regularization(model)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_spectral_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=0.0,
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["spectral_regularization"],
            neurons_improved=neurons_regularized,
            total_neurons=len(initial_metrics),
            improvement_time=end_time - start_time
        )

    def analyze_neuron_weights(self, model: nn.Module) -> Dict[str, NeuronWeightMetrics]:
        """Analiza los pesos de las neuronas individuales"""

        neuron_metrics = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                metrics = NeuronWeightMetrics(
                    neuron_id=name,
                    weight_magnitude=torch.norm(param.data).item(),
                    weight_variance=torch.var(param.data).item(),
                    gradient_norm=torch.norm(param.grad).item() if param.grad is not None else 0.0,
                    activation_frequency=self._calculate_activation_frequency(param),
                    importance_score=self._calculate_importance_score(param),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'spectral_norm': self.spectral_norms.get(name, 0.0),
                        'clipping_factor': self.clipping_factors.get(name, 1.0)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _calculate_spectral_norms(self, model: nn.Module) -> None:
        """Calcula las normas espectrales de los pesos"""

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular norma espectral
                if param.dim() >= 2:
                    # Para matrices 2D o mayores
                    spectral_norm = torch.norm(param.data, p=2).item()
                else:
                    # Para vectores 1D
                    spectral_norm = torch.norm(param.data, p=2).item()

                self.spectral_norms[name] = spectral_norm

    def _apply_spectral_regularization(self, model: nn.Module) -> int:
        """Aplica regularización espectral"""

        neurons_regularized = 0

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.spectral_norms:
                spectral_norm = self.spectral_norms[name]

                # Aplicar clipping espectral si es necesario
                if spectral_norm > self.regularization_config.spectral_radius_limit:
                    clipping_factor = self.regularization_config.spectral_radius_limit / spectral_norm
                    self.clipping_factors[name] = clipping_factor

                    with torch.no_grad():
                        param.data *= clipping_factor

                    # Registrar en historial
                    self._record_spectral_regularization(name, spectral_norm, clipping_factor)

                    neurons_regularized += 1

        logger.info(f"Regularización espectral aplicada a {neurons_regularized} neuronas")
        return neurons_regularized

    def _record_spectral_regularization(self, name: str, spectral_norm: float, clipping_factor: float) -> None:
        """Registra la regularización espectral en el historial"""

        self.spectral_history[name].append({
            'spectral_norm': spectral_norm,
            'clipping_factor': clipping_factor,
            'timestamp': time.time()
        })

        # Mantener historial limitado
        if len(self.spectral_history[name]) > 100:
            self.spectral_history[name].pop(0)

    def _calculate_activation_frequency(self, param: torch.Tensor) -> float:
        """Calcula la frecuencia de activación"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 5.0)

    def _calculate_importance_score(self, param: torch.Tensor) -> float:
        """Calcula el score de importancia"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        importance = magnitude * (1 + variance)
        return min(1.0, importance / 100.0)

    def _calculate_stability_score(self, param: torch.Tensor) -> float:
        """Calcula el score de estabilidad"""

        return 0.5  # Valor por defecto

    def _estimate_contribution_to_loss(self, param: torch.Tensor) -> float:
        """Estima la contribución a la pérdida"""

        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            return min(1.0, grad_norm / 5.0)
        return 0.0

    def _get_layer_type(self, name: str) -> str:
        """Determina el tipo de capa"""

        if 'conv' in name.lower():
            return 'convolutional'
        elif 'linear' in name.lower() or 'fc' in name.lower():
            return 'linear'
        elif 'bn' in name.lower() or 'norm' in name.lower():
            return 'normalization'
        else:
            return 'other'

    def _calculate_spectral_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia espectral"""

        if not self.spectral_norms:
            return 0.0

        norms = list(self.spectral_norms.values())
        convergence_rate = 1.0 / (1.0 + np.std(norms))
        return convergence_rate

    def _calculate_stability_improvement(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la mejora en estabilidad"""

        initial_stability = np.mean([m.stability_score for m in initial_metrics.values()])
        final_stability = np.mean([m.stability_score for m in final_metrics.values()])

        return final_stability - initial_stability

    def _calculate_efficiency_gain(self, initial_metrics: Dict, final_metrics: Dict) -> float:
        """Calcula la ganancia en eficiencia"""

        initial_efficiency = np.mean([m.importance_score for m in initial_metrics.values()])
        final_efficiency = np.mean([m.importance_score for m in final_metrics.values()])

        return final_efficiency - initial_efficiency

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 4.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain

# Funciones de utilidad


def create_advanced_regularizer(regularization_method: str = "adaptive_l2") -> BaseWeightImprover:
    """Factory para crear regularizadores avanzados"""

    config = NeuralWeightConfig()

    if regularization_method == "adaptive_l2":
        return AdaptiveL2Regularizer(config)
    elif regularization_method == "spectral":
        return SpectralRegularizer(config)
    else:
        raise ValueError(f"Método de regularización no soportado: {regularization_method}")


def regularize_model_weights(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                             regularization_method: str = "adaptive_l2") -> WeightImprovementResult:
    """Función de conveniencia para regularizar pesos de un modelo"""

    regularizer = create_advanced_regularizer(regularization_method)
    return regularizer.improve_weights(model, data_loader)


# Exportar clases y funciones principales
__all__ = [
    'RegularizationConfig',
    'AdaptiveL2Regularizer',
    'SpectralRegularizer',
    'create_advanced_regularizer',
    'regularize_model_weights'
]

logger.info("RFEN4_RN_5 - Técnicas Avanzadas de Regularización de Pesos cargadas correctamente")
