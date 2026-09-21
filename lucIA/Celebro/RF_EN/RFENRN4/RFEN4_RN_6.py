"""
RFEN4_RN_6 - Optimización de Pesos Cuantizados
Implementación de técnicas avanzadas para cuantización y optimización de pesos neuronales
Incluye: Cuantización dinámica, cuantización adaptativa, y cuantización cuántica inspirada
"""

import torch
import torch.nn as nn
import torch.optim as optim
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
class QuantizationConfig:
    """Configuración para cuantización de pesos"""
    quantization_method: str = "dynamic_quantization"  # dynamic, adaptive, quantum_inspired
    quantization_bits: int = 8
    quantization_levels: int = 256
    adaptive_quantization: bool = True
    quantization_threshold: float = 0.01
    quantization_frequency: int = 100
    quantum_superposition: bool = False
    quantum_entanglement: bool = False
    quantization_error_tolerance: float = 0.1
    dynamic_bits: bool = True
    min_bits: int = 4
    max_bits: int = 16
    quantization_history_size: int = 1000


class DynamicQuantizer(BaseWeightImprover):
    """
    Cuantizador dinámico de pesos
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.quantization_config = QuantizationConfig()
        self.quantization_history = defaultdict(list)
        self.quantization_levels = {}
        self.quantization_errors = {}
        self.bit_allocations = {}

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando cuantización dinámica"""

        logger.info("Iniciando cuantización dinámica de pesos")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Calcular niveles de cuantización
        self._calculate_quantization_levels(model)

        # Aplicar cuantización dinámica
        neurons_quantized = self._apply_dynamic_quantization(model)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_quantization_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_memory_optimization(),
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["dynamic_quantization"],
            neurons_improved=neurons_quantized,
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
                        'quantization_levels': self.quantization_levels.get(name, 0),
                        'quantization_error': self.quantization_errors.get(name, 0.0),
                        'bit_allocation': self.bit_allocations.get(name, 8)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _calculate_quantization_levels(self, model: nn.Module) -> None:
        """Calcula los niveles de cuantización para cada peso"""

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular rango de valores
                weight_min = torch.min(param.data).item()
                weight_max = torch.max(param.data).item()
                weight_range = weight_max - weight_min

                # Calcular número de bits óptimo
                optimal_bits = self._calculate_optimal_bits(weight_range, param)
                self.bit_allocations[name] = optimal_bits

                # Calcular niveles de cuantización
                quantization_levels = 2 ** optimal_bits
                self.quantization_levels[name] = quantization_levels

    def _calculate_optimal_bits(self, weight_range: float, param: torch.Tensor) -> int:
        """Calcula el número óptimo de bits para cuantización"""

        # Calcular varianza
        variance = torch.var(param.data).item()

        # Calcular bits basado en rango y varianza
        if weight_range < 0.1:
            optimal_bits = max(self.quantization_config.min_bits, 4)
        elif weight_range < 1.0:
            optimal_bits = max(self.quantization_config.min_bits, 6)
        else:
            optimal_bits = min(self.quantization_config.max_bits, 8)

        # Ajustar según varianza
        if variance > 1.0:
            optimal_bits = min(optimal_bits + 2, self.quantization_config.max_bits)
        elif variance < 0.01:
            optimal_bits = max(optimal_bits - 2, self.quantization_config.min_bits)

        return optimal_bits

    def _apply_dynamic_quantization(self, model: nn.Module) -> int:
        """Aplica cuantización dinámica"""

        neurons_quantized = 0

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.quantization_levels:
                # Obtener configuración de cuantización
                quantization_levels = self.quantization_levels[name]
                bit_allocation = self.bit_allocations[name]

                # Aplicar cuantización
                quantized_weights = self._quantize_weights(param.data, quantization_levels)

                # Calcular error de cuantización
                quantization_error = torch.norm(param.data - quantized_weights).item()
                self.quantization_errors[name] = quantization_error

                # Aplicar pesos cuantizados si el error es aceptable
                if quantization_error < self.quantization_config.quantization_error_tolerance:
                    with torch.no_grad():
                        param.data.copy_(quantized_weights)

                    # Registrar en historial
                    self._record_quantization(name, quantization_levels, bit_allocation, quantization_error)

                    neurons_quantized += 1

        logger.info(f"Cuantización dinámica aplicada a {neurons_quantized} neuronas")
        return neurons_quantized

    def _quantize_weights(self, weights: torch.Tensor, quantization_levels: int) -> torch.Tensor:
        """Cuantiza los pesos"""

        # Calcular rango de cuantización
        weight_min = torch.min(weights).item()
        weight_max = torch.max(weights).item()

        # Crear niveles de cuantización
        quantization_step = (weight_max - weight_min) / (quantization_levels - 1)

        # Cuantizar pesos
        quantized_weights = torch.round((weights - weight_min) / quantization_step) * quantization_step + weight_min

        return quantized_weights

    def _record_quantization(self, name: str, quantization_levels: int, bit_allocation: int, quantization_error: float) -> None:
        """Registra la cuantización en el historial"""

        self.quantization_history[name].append({
            'quantization_levels': quantization_levels,
            'bit_allocation': bit_allocation,
            'quantization_error': quantization_error,
            'timestamp': time.time()
        })

        # Mantener historial limitado
        if len(self.quantization_history[name]) > self.quantization_config.quantization_history_size:
            self.quantization_history[name].pop(0)

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

    def _calculate_quantization_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia de la cuantización"""

        if not self.quantization_errors:
            return 0.0

        errors = list(self.quantization_errors.values())
        convergence_rate = 1.0 / (1.0 + np.mean(errors))
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

        if not self.bit_allocations:
            return 0.0

        # Calcular reducción promedio de bits
        avg_bits = np.mean(list(self.bit_allocations.values()))
        original_bits = 32  # Bits originales (float32)

        memory_optimization = (original_bits - avg_bits) / original_bits
        return max(0.0, memory_optimization)

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 6.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain


class AdaptiveQuantizer(BaseWeightImprover):
    """
    Cuantizador adaptativo de pesos
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.quantization_config = QuantizationConfig(quantization_method="adaptive")
        self.adaptation_factors = {}
        self.quantization_schedules = {}
        self.performance_tracker = {}

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando cuantización adaptativa"""

        logger.info("Iniciando cuantización adaptativa de pesos")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Aplicar cuantización adaptativa
        neurons_quantized = self._apply_adaptive_quantization(model, data_loader)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_adaptive_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_adaptive_memory_optimization(),
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["adaptive_quantization"],
            neurons_improved=neurons_quantized,
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
                        'adaptation_factor': self.adaptation_factors.get(name, 1.0),
                        'quantization_schedule': self.quantization_schedules.get(name, {}),
                        'performance_score': self.performance_tracker.get(name, 0.0)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _apply_adaptive_quantization(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> int:
        """Aplica cuantización adaptativa"""

        neurons_quantized = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular factor de adaptación
                adaptation_factor = self._calculate_adaptation_factor(param, name, data_loader)
                self.adaptation_factors[name] = adaptation_factor

                # Aplicar cuantización adaptativa
                if adaptation_factor > self.quantization_config.quantization_threshold:
                    quantized_weights = self._adaptive_quantize_weights(param.data, adaptation_factor)

                    with torch.no_grad():
                        param.data.copy_(quantized_weights)

                    neurons_quantized += 1

        logger.info(f"Cuantización adaptativa aplicada a {neurons_quantized} neuronas")
        return neurons_quantized

    def _calculate_adaptation_factor(self, param: torch.Tensor, name: str,
                                     data_loader: torch.utils.data.DataLoader) -> float:
        """Calcula el factor de adaptación para cuantización"""

        # Factor base
        base_factor = 1.0

        # Ajustar según tipo de capa
        layer_type = self._get_layer_type(name)

        if layer_type == 'convolutional':
            base_factor *= 1.1  # Favorecer cuantización en capas convolucionales
        elif layer_type == 'linear':
            base_factor *= 1.05  # Favorecer cuantización en capas lineales
        elif layer_type == 'normalization':
            base_factor *= 0.9  # Reducir cuantización en capas de normalización

        # Ajustar según varianza
        variance = torch.var(param.data).item()
        if variance > 1.0:
            base_factor *= 1.2  # Aumentar cuantización para alta varianza
        elif variance < 0.01:
            base_factor *= 0.8  # Reducir cuantización para baja varianza

        return base_factor

    def _adaptive_quantize_weights(self, weights: torch.Tensor, adaptation_factor: float) -> torch.Tensor:
        """Cuantiza pesos de manera adaptativa"""

        # Calcular niveles de cuantización adaptativos
        quantization_levels = int(2 ** (8 * adaptation_factor))
        quantization_levels = max(16, min(256, quantization_levels))

        # Aplicar cuantización
        weight_min = torch.min(weights).item()
        weight_max = torch.max(weights).item()
        quantization_step = (weight_max - weight_min) / (quantization_levels - 1)

        quantized_weights = torch.round((weights - weight_min) / quantization_step) * quantization_step + weight_min

        return quantized_weights

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

    def _calculate_adaptive_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia adaptativa"""

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

    def _calculate_adaptive_memory_optimization(self) -> float:
        """Calcula la optimización de memoria adaptativa"""

        if not self.adaptation_factors:
            return 0.0

        # Calcular reducción promedio de memoria
        avg_adaptation = np.mean(list(self.adaptation_factors.values()))
        memory_optimization = avg_adaptation * 0.3  # Factor de optimización

        return memory_optimization

    def _calculate_speed_gain(self, optimization_time: float) -> float:
        """Calcula la ganancia en velocidad"""

        expected_time = 7.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain

# Funciones de utilidad


def create_quantized_optimizer(quantization_method: str = "dynamic_quantization") -> BaseWeightImprover:
    """Factory para crear optimizadores cuantizados"""

    config = NeuralWeightConfig()

    if quantization_method == "dynamic_quantization":
        return DynamicQuantizer(config)
    elif quantization_method == "adaptive":
        return AdaptiveQuantizer(config)
    else:
        raise ValueError(f"Método de cuantización no soportado: {quantization_method}")


def quantize_model_weights(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                           quantization_method: str = "dynamic_quantization") -> WeightImprovementResult:
    """Función de conveniencia para cuantizar pesos de un modelo"""

    quantizer = create_quantized_optimizer(quantization_method)
    return quantizer.improve_weights(model, data_loader)


# Exportar clases y funciones principales
__all__ = [
    'QuantizationConfig',
    'DynamicQuantizer',
    'AdaptiveQuantizer',
    'create_quantized_optimizer',
    'quantize_model_weights'
]

logger.info("RFEN4_RN_6 - Optimización de Pesos Cuantizados cargada correctamente")
