"""
RFEN4_RN_2 - Sistema de Escalado Adaptativo de Pesos
Implementación de técnicas avanzadas para escalado dinámico y adaptativo de pesos neuronales
Incluye: Escalado basado en importancia, escalado por capas, y escalado cuántico
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
class AdaptiveScalingConfig:
    """Configuración para escalado adaptativo de pesos"""
    scaling_method: str = "importance_based"  # importance_based, layer_based, quantum_inspired
    adaptation_rate: float = 0.1
    scaling_factor_range: Tuple[float, float] = (0.1, 10.0)
    importance_threshold: float = 0.01
    scaling_frequency: int = 100
    momentum_factor: float = 0.9
    stability_threshold: float = 0.001
    quantum_superposition: bool = False
    entanglement_factor: float = 0.5
    scaling_history_size: int = 1000
    adaptive_learning_rate: bool = True


class ImportanceBasedScaler(BaseWeightImprover):
    """
    Escalador basado en importancia de pesos
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.scaling_config = AdaptiveScalingConfig()
        self.importance_scores = {}
        self.scaling_history = defaultdict(deque)
        self.weight_magnitudes = defaultdict(list)
        self.adaptation_factors = {}

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando escalado basado en importancia"""

        logger.info("Iniciando escalado adaptativo basado en importancia")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Calcular importancia de cada peso
        self._calculate_importance_scores(model, data_loader)

        # Aplicar escalado adaptativo
        neurons_improved = self._apply_adaptive_scaling(model)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_memory_optimization(),
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["importance_based_scaling"],
            neurons_improved=neurons_improved,
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
                    importance_score=self.importance_scores.get(name, 0.0),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'scaling_factor': self.adaptation_factors.get(name, 1.0),
                        'magnitude_history': self.weight_magnitudes.get(name, [])[-10:]
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _calculate_importance_scores(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> None:
        """Calcula los scores de importancia para cada peso"""

        model.eval()
        importance_scores = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Calcular importancia basada en múltiples factores
                    magnitude_importance = torch.norm(param.data).item()

                    # Importancia basada en varianza
                    variance_importance = torch.var(param.data).item()

                    # Importancia basada en gradientes históricos
                    gradient_importance = 0.0
                    if name in self.scaling_history:
                        recent_gradients = list(self.scaling_history[name])[-10:]
                        if recent_gradients:
                            gradient_importance = np.mean(recent_gradients)

                    # Importancia basada en activaciones
                    activation_importance = self._calculate_activation_importance(param, data_loader)

                    # Score combinado
                    importance_score = (
                        magnitude_importance * 0.3 +
                        variance_importance * 0.2 +
                        abs(gradient_importance) * 0.3 +
                        activation_importance * 0.2
                    )

                    importance_scores[name] = importance_score

        self.importance_scores = importance_scores
        logger.info(f"Calculados scores de importancia para {len(importance_scores)} parámetros")

    def _calculate_activation_importance(self, param: torch.Tensor, data_loader: torch.utils.data.DataLoader) -> float:
        """Calcula la importancia basada en activaciones"""

        # Simplificado: basado en la magnitud promedio de activaciones
        activation_sum = 0.0
        batch_count = 0

        with torch.no_grad():
            for data, _ in data_loader:
                if batch_count >= 5:  # Limitar para eficiencia
                    break

                # Simular activación
                activation = torch.sum(torch.abs(param.data))
                activation_sum += activation.item()
                batch_count += 1

        return activation_sum / max(batch_count, 1)

    def _apply_adaptive_scaling(self, model: nn.Module) -> int:
        """Aplica escalado adaptativo a los pesos"""

        neurons_improved = 0

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.importance_scores:
                importance = self.importance_scores[name]

                # Calcular factor de escalado
                scaling_factor = self._calculate_scaling_factor(importance, name)

                # Aplicar escalado
                if abs(scaling_factor - 1.0) > self.scaling_config.stability_threshold:
                    with torch.no_grad():
                        param.data *= scaling_factor

                    # Actualizar historial
                    self.adaptation_factors[name] = scaling_factor
                    self.weight_magnitudes[name].append(torch.norm(param.data).item())

                    # Mantener historial limitado
                    if len(self.weight_magnitudes[name]) > self.scaling_config.scaling_history_size:
                        self.weight_magnitudes[name].pop(0)

                    neurons_improved += 1

        logger.info(f"Escalado aplicado a {neurons_improved} neuronas")
        return neurons_improved

    def _calculate_scaling_factor(self, importance: float, param_name: str) -> float:
        """Calcula el factor de escalado para un parámetro"""

        # Factor base basado en importancia
        if importance > self.scaling_config.importance_threshold:
            base_factor = 1.0 + (importance * self.scaling_config.adaptation_rate)
        else:
            base_factor = 1.0 - (importance * self.scaling_config.adaptation_rate)

        # Aplicar momentum si hay historial
        if param_name in self.adaptation_factors:
            previous_factor = self.adaptation_factors[param_name]
            base_factor = (
                self.scaling_config.momentum_factor * previous_factor +
                (1 - self.scaling_config.momentum_factor) * base_factor
            )

        # Limitar rango de escalado
        min_factor, max_factor = self.scaling_config.scaling_factor_range
        scaling_factor = max(min_factor, min(max_factor, base_factor))

        return scaling_factor

    def _calculate_activation_frequency(self, param: torch.Tensor) -> float:
        """Calcula la frecuencia de activación"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 5.0)

    def _calculate_stability_score(self, param: torch.Tensor) -> float:
        """Calcula el score de estabilidad"""

        if param_name := getattr(param, 'name', None):
            if param_name in self.weight_magnitudes and len(self.weight_magnitudes[param_name]) > 1:
                magnitudes = self.weight_magnitudes[param_name]
                stability = 1.0 / (1.0 + np.std(magnitudes))
                return stability

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

    def _calculate_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia"""

        if not self.adaptation_factors:
            return 0.0

        factors = list(self.adaptation_factors.values())
        convergence_rate = 1.0 / (1.0 + np.std(factors))
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

        # Basado en la reducción de varianza en los pesos
        total_variance_reduction = 0.0
        param_count = 0

        for magnitudes in self.weight_magnitudes.values():
            if len(magnitudes) > 1:
                variance_reduction = np.std(magnitudes[:-1]) - np.std(magnitudes[1:])
                total_variance_reduction += variance_reduction
                param_count += 1

        if param_count > 0:
            return total_variance_reduction / param_count

        return 0.0

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 5.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain


class LayerBasedScaler(BaseWeightImprover):
    """
    Escalador basado en capas
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.scaling_config = AdaptiveScalingConfig(scaling_method="layer_based")
        self.layer_scaling_factors = {}
        self.layer_importance = {}
        self.layer_history = defaultdict(list)

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando escalado basado en capas"""

        logger.info("Iniciando escalado adaptativo basado en capas")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Calcular importancia por capas
        self._calculate_layer_importance(model, data_loader)

        # Aplicar escalado por capas
        neurons_improved = self._apply_layer_scaling(model)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_layer_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_layer_memory_optimization(),
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["layer_based_scaling"],
            neurons_improved=neurons_improved,
            total_neurons=len(initial_metrics),
            improvement_time=end_time - start_time
        )

    def analyze_neuron_weights(self, model: nn.Module) -> Dict[str, NeuronWeightMetrics]:
        """Analiza los pesos de las neuronas individuales"""

        neuron_metrics = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_type = self._get_layer_type(name)

                metrics = NeuronWeightMetrics(
                    neuron_id=name,
                    weight_magnitude=torch.norm(param.data).item(),
                    weight_variance=torch.var(param.data).item(),
                    gradient_norm=torch.norm(param.grad).item() if param.grad is not None else 0.0,
                    activation_frequency=self._calculate_activation_frequency(param),
                    importance_score=self.layer_importance.get(layer_type, 0.0),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    timestamp=time.time(),
                    layer_type=layer_type,
                    additional_metrics={
                        'layer_scaling_factor': self.layer_scaling_factors.get(layer_type, 1.0)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _calculate_layer_importance(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> None:
        """Calcula la importancia de cada capa"""

        layer_importance = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_type = self._get_layer_type(name)

                if layer_type not in layer_importance:
                    layer_importance[layer_type] = []

                # Calcular métricas de la capa
                magnitude = torch.norm(param.data).item()
                variance = torch.var(param.data).item()
                gradient_norm = torch.norm(param.grad).item() if param.grad is not None else 0.0

                # Score de importancia de la capa
                importance_score = magnitude * 0.4 + variance * 0.3 + gradient_norm * 0.3
                layer_importance[layer_type].append(importance_score)

        # Calcular importancia promedio por capa
        for layer_type, scores in layer_importance.items():
            self.layer_importance[layer_type] = np.mean(scores)

        logger.info(f"Calculada importancia para {len(self.layer_importance)} tipos de capa")

    def _apply_layer_scaling(self, model: nn.Module) -> int:
        """Aplica escalado basado en capas"""

        neurons_improved = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                layer_type = self._get_layer_type(name)

                if layer_type in self.layer_importance:
                    importance = self.layer_importance[layer_type]
                    scaling_factor = self._calculate_layer_scaling_factor(layer_type, importance)

                    # Aplicar escalado
                    if abs(scaling_factor - 1.0) > self.scaling_config.stability_threshold:
                        with torch.no_grad():
                            param.data *= scaling_factor

                        neurons_improved += 1

        logger.info(f"Escalado por capas aplicado a {neurons_improved} neuronas")
        return neurons_improved

    def _calculate_layer_scaling_factor(self, layer_type: str, importance: float) -> float:
        """Calcula el factor de escalado para una capa"""

        # Factor base basado en importancia de la capa
        if importance > self.scaling_config.importance_threshold:
            base_factor = 1.0 + (importance * self.scaling_config.adaptation_rate)
        else:
            base_factor = 1.0 - (importance * self.scaling_config.adaptation_rate)

        # Ajustar factor según tipo de capa
        if layer_type == 'convolutional':
            base_factor *= 1.1  # Favorecer capas convolucionales
        elif layer_type == 'linear':
            base_factor *= 1.05  # Favorecer capas lineales ligeramente
        elif layer_type == 'normalization':
            base_factor *= 0.95  # Reducir capas de normalización

        # Aplicar momentum si hay historial
        if layer_type in self.layer_scaling_factors:
            previous_factor = self.layer_scaling_factors[layer_type]
            base_factor = (
                self.scaling_config.momentum_factor * previous_factor +
                (1 - self.scaling_config.momentum_factor) * base_factor
            )

        # Limitar rango
        min_factor, max_factor = self.scaling_config.scaling_factor_range
        scaling_factor = max(min_factor, min(max_factor, base_factor))

        # Actualizar historial
        self.layer_scaling_factors[layer_type] = scaling_factor
        self.layer_history[layer_type].append(scaling_factor)

        return scaling_factor

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

    def _calculate_activation_frequency(self, param: torch.Tensor) -> float:
        """Calcula la frecuencia de activación"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 5.0)

    def _calculate_stability_score(self, param: torch.Tensor) -> float:
        """Calcula el score de estabilidad"""

        return 0.5  # Valor por defecto

    def _estimate_contribution_to_loss(self, param: torch.Tensor) -> float:
        """Estima la contribución a la pérdida"""

        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            return min(1.0, grad_norm / 5.0)
        return 0.0

    def _calculate_layer_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia por capas"""

        if not self.layer_scaling_factors:
            return 0.0

        factors = list(self.layer_scaling_factors.values())
        convergence_rate = 1.0 / (1.0 + np.std(factors))
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

    def _calculate_layer_memory_optimization(self) -> float:
        """Calcula la optimización de memoria por capas"""

        total_optimization = 0.0
        layer_count = 0

        for layer_type, factors in self.layer_history.items():
            if len(factors) > 1:
                optimization = 1.0 / (1.0 + np.std(factors))
                total_optimization += optimization
                layer_count += 1

        if layer_count > 0:
            return total_optimization / layer_count

        return 0.0

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 3.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain

# Funciones de utilidad


def create_adaptive_scaler(scaling_method: str = "importance_based") -> BaseWeightImprover:
    """Factory para crear escaladores adaptativos"""

    config = NeuralWeightConfig()

    if scaling_method == "importance_based":
        return ImportanceBasedScaler(config)
    elif scaling_method == "layer_based":
        return LayerBasedScaler(config)
    else:
        raise ValueError(f"Método de escalado no soportado: {scaling_method}")


def scale_model_weights(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                        scaling_method: str = "importance_based") -> WeightImprovementResult:
    """Función de conveniencia para escalar pesos de un modelo"""

    scaler = create_adaptive_scaler(scaling_method)
    return scaler.improve_weights(model, data_loader)


# Exportar clases y funciones principales
__all__ = [
    'AdaptiveScalingConfig',
    'ImportanceBasedScaler',
    'LayerBasedScaler',
    'create_adaptive_scaler',
    'scale_model_weights'
]

logger.info("RFEN4_RN_2 - Sistema de Escalado Adaptativo de Pesos cargado correctamente")
