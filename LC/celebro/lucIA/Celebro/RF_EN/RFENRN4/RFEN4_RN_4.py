"""
RFEN4_RN_4 - Sistema de Ajuste Dinámico de Pesos
Implementación de técnicas avanzadas para ajuste dinámico y en tiempo real de pesos neuronales
Incluye: Ajuste adaptativo, optimización bayesiana, y ajuste basado en rendimiento
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
class DynamicAdjustmentConfig:
    """Configuración para ajuste dinámico de pesos"""
    adjustment_method: str = "performance_based"  # performance_based, bayesian, adaptive_momentum
    adjustment_rate: float = 0.01
    performance_threshold: float = 0.01
    adjustment_frequency: int = 50
    momentum_factor: float = 0.9
    adaptation_speed: float = 0.1
    stability_window: int = 10
    exploration_rate: float = 0.1
    exploitation_rate: float = 0.9
    bayesian_samples: int = 100
    gaussian_noise_std: float = 0.01
    adjustment_history_size: int = 1000


class PerformanceBasedAdjuster(BaseWeightImprover):
    """
    Ajustador dinámico basado en rendimiento
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.adjustment_config = DynamicAdjustmentConfig()
        self.performance_history = defaultdict(list)
        self.adjustment_history = defaultdict(list)
        self.weight_trajectories = defaultdict(list)
        self.performance_scores = {}
        self.adjustment_vectors = {}

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando ajuste dinámico basado en rendimiento"""

        logger.info("Iniciando ajuste dinámico basado en rendimiento")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Evaluar rendimiento inicial
        initial_performance = self._evaluate_model_performance(model, data_loader)

        # Aplicar ajustes dinámicos
        neurons_adjusted = self._apply_dynamic_adjustments(model, data_loader)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Evaluar rendimiento final
        final_performance = self._evaluate_model_performance(model, data_loader)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_adjustment_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_memory_optimization(),
            accuracy_improvement=final_performance - initial_performance,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["performance_based_dynamic_adjustment"],
            neurons_improved=neurons_adjusted,
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
                    improvement_rate=self._calculate_improvement_rate(name),
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'performance_score': self.performance_scores.get(name, 0.0),
                        'adjustment_vector': self.adjustment_vectors.get(name, None),
                        'trajectory_history': self.weight_trajectories.get(name, [])[-10:]
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _evaluate_model_performance(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> float:
        """Evalúa el rendimiento del modelo"""

        model.eval()
        total_loss = 0.0
        batch_count = 0

        with torch.no_grad():
            for data, target in data_loader:
                if batch_count >= 10:  # Limitar para eficiencia
                    break

                output = model(data)
                loss = nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item()
                batch_count += 1

        return total_loss / max(batch_count, 1)

    def _apply_dynamic_adjustments(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> int:
        """Aplica ajustes dinámicos a los pesos"""

        neurons_adjusted = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular score de rendimiento para esta neurona
                performance_score = self._calculate_neuron_performance_score(param, name, data_loader)
                self.performance_scores[name] = performance_score

                # Determinar si necesita ajuste
                if self._needs_adjustment(name, performance_score):
                    # Calcular vector de ajuste
                    adjustment_vector = self._calculate_adjustment_vector(param, performance_score)
                    self.adjustment_vectors[name] = adjustment_vector

                    # Aplicar ajuste
                    self._apply_weight_adjustment(param, adjustment_vector)

                    # Registrar en historial
                    self._record_adjustment(name, adjustment_vector, performance_score)

                    neurons_adjusted += 1

        logger.info(f"Ajustes dinámicos aplicados a {neurons_adjusted} neuronas")
        return neurons_adjusted

    def _calculate_neuron_performance_score(self, param: torch.Tensor, name: str,
                                            data_loader: torch.utils.data.DataLoader) -> float:
        """Calcula el score de rendimiento para una neurona específica"""

        # Score basado en múltiples factores
        magnitude_score = torch.norm(param.data).item()
        variance_score = torch.var(param.data).item()
        gradient_score = torch.norm(param.grad).item() if param.grad is not None else 0.0

        # Score combinado
        performance_score = (
            magnitude_score * 0.4 +
            variance_score * 0.3 +
            gradient_score * 0.3
        )

        return performance_score

    def _needs_adjustment(self, name: str, performance_score: float) -> bool:
        """Determina si una neurona necesita ajuste"""

        # Verificar si hay historial
        if name in self.performance_history:
            recent_scores = self.performance_history[name][-self.adjustment_config.stability_window:]

            if len(recent_scores) >= 3:
                # Calcular tendencia
                trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

                # Necesita ajuste si la tendencia es negativa o el score es bajo
                return trend < 0 or performance_score < self.adjustment_config.performance_threshold

        return True  # Ajustar si no hay historial

    def _calculate_adjustment_vector(self, param: torch.Tensor, performance_score: float) -> torch.Tensor:
        """Calcula el vector de ajuste para un parámetro"""

        # Vector base basado en gradientes
        if param.grad is not None:
            base_vector = param.grad.clone()
        else:
            base_vector = torch.randn_like(param.data) * self.adjustment_config.gaussian_noise_std

        # Escalar según rendimiento
        if performance_score < self.adjustment_config.performance_threshold:
            # Aumentar ajuste para rendimiento bajo
            adjustment_scale = self.adjustment_config.adjustment_rate * 2.0
        else:
            # Ajuste normal
            adjustment_scale = self.adjustment_config.adjustment_rate

        # Aplicar escala
        adjustment_vector = base_vector * adjustment_scale

        # Añadir exploración
        if random.random() < self.adjustment_config.exploration_rate:
            noise = torch.randn_like(adjustment_vector) * self.adjustment_config.gaussian_noise_std
            adjustment_vector += noise

        return adjustment_vector

    def _apply_weight_adjustment(self, param: torch.Tensor, adjustment_vector: torch.Tensor) -> None:
        """Aplica el ajuste de pesos"""

        with torch.no_grad():
            # Aplicar ajuste con momentum
            if hasattr(self, '_momentum_vectors'):
                if param_name := getattr(param, 'name', None):
                    if param_name in self._momentum_vectors:
                        momentum = self._momentum_vectors[param_name]
                        adjustment_vector = (
                            self.adjustment_config.momentum_factor * momentum +
                            (1 - self.adjustment_config.momentum_factor) * adjustment_vector
                        )
                        self._momentum_vectors[param_name] = adjustment_vector
                    else:
                        self._momentum_vectors[param_name] = adjustment_vector
            else:
                self._momentum_vectors = {getattr(param, 'name', 'unknown'): adjustment_vector}

            # Aplicar ajuste
            param.data += adjustment_vector

    def _record_adjustment(self, name: str, adjustment_vector: torch.Tensor, performance_score: float) -> None:
        """Registra el ajuste en el historial"""

        adjustment_magnitude = torch.norm(adjustment_vector).item()

        # Registrar en historial de rendimiento
        self.performance_history[name].append(performance_score)

        # Registrar en historial de ajustes
        self.adjustment_history[name].append({
            'magnitude': adjustment_magnitude,
            'timestamp': time.time(),
            'performance_score': performance_score
        })

        # Registrar en trayectoria de pesos
        self.weight_trajectories[name].append(performance_score)

        # Mantener historial limitado
        for history in [self.performance_history[name], self.adjustment_history[name], self.weight_trajectories[name]]:
            if len(history) > self.adjustment_config.adjustment_history_size:
                history.pop(0)

    def _calculate_improvement_rate(self, name: str) -> float:
        """Calcula la tasa de mejora para una neurona"""

        if name in self.performance_history and len(self.performance_history[name]) > 1:
            scores = self.performance_history[name]
            improvement_rate = (scores[-1] - scores[0]) / len(scores)
            return improvement_rate

        return 0.0

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

    def _calculate_adjustment_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia de los ajustes"""

        if not self.adjustment_history:
            return 0.0

        convergence_rates = []

        for name, history in self.adjustment_history.items():
            if len(history) > 3:
                magnitudes = [adj['magnitude'] for adj in history[-10:]]
                convergence_rate = 1.0 / (1.0 + np.std(magnitudes))
                convergence_rates.append(convergence_rate)

        return np.mean(convergence_rates) if convergence_rates else 0.0

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

        expected_time = 5.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain


class BayesianWeightAdjuster(BaseWeightImprover):
    """
    Ajustador bayesiano para pesos
    """

    def __init__(self, config: NeuralWeightConfig):
        super().__init__(config)
        self.adjustment_config = DynamicAdjustmentConfig(adjustment_method="bayesian")
        self.bayesian_models = {}
        self.sample_history = defaultdict(list)
        self.acquisition_history = defaultdict(list)

    def improve_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> WeightImprovementResult:
        """Mejora pesos usando ajuste bayesiano"""

        logger.info("Iniciando ajuste bayesiano de pesos")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_neuron_weights(model)

        # Aplicar ajuste bayesiano
        neurons_adjusted = self._apply_bayesian_adjustments(model, data_loader)

        # Analizar pesos finales
        final_metrics = self.analyze_neuron_weights(model)

        # Calcular mejoras
        improvement_score = self.calculate_improvement_score(initial_metrics, final_metrics)

        end_time = time.time()

        return WeightImprovementResult(
            improvement_score=improvement_score,
            convergence_rate=self._calculate_bayesian_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=0.0,
            accuracy_improvement=improvement_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            techniques_applied=["bayesian_weight_adjustment"],
            neurons_improved=neurons_adjusted,
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
                        'bayesian_samples': len(self.sample_history.get(name, [])),
                        'acquisition_score': self._calculate_acquisition_score(name)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _apply_bayesian_adjustments(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> int:
        """Aplica ajustes bayesianos"""

        neurons_adjusted = 0

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Generar muestras bayesianas
                samples = self._generate_bayesian_samples(param, name)

                # Evaluar muestras
                best_sample = self._evaluate_bayesian_samples(param, samples, data_loader)

                # Aplicar mejor muestra
                if best_sample is not None:
                    with torch.no_grad():
                        param.data = best_sample

                    neurons_adjusted += 1

        logger.info(f"Ajustes bayesianos aplicados a {neurons_adjusted} neuronas")
        return neurons_adjusted

    def _generate_bayesian_samples(self, param: torch.Tensor, name: str) -> List[torch.Tensor]:
        """Genera muestras bayesianas para un parámetro"""

        samples = []

        # Generar muestras basadas en distribución gaussiana
        for _ in range(self.adjustment_config.bayesian_samples):
            # Muestra centrada en el peso actual
            sample = param.data + torch.randn_like(param.data) * self.adjustment_config.gaussian_noise_std
            samples.append(sample)

        return samples

    def _evaluate_bayesian_samples(self, param: torch.Tensor, samples: List[torch.Tensor],
                                   data_loader: torch.utils.data.DataLoader) -> Optional[torch.Tensor]:
        """Evalúa las muestras bayesianas"""

        best_sample = None
        best_score = float('inf')

        for sample in samples:
            # Evaluar muestra
            score = self._evaluate_sample(sample, data_loader)

            if score < best_score:
                best_score = score
                best_sample = sample

        return best_sample

    def _evaluate_sample(self, sample: torch.Tensor, data_loader: torch.utils.data.DataLoader) -> float:
        """Evalúa una muestra de pesos"""

        # Simplificado: evaluar basado en magnitud y varianza
        magnitude_score = torch.norm(sample).item()
        variance_score = torch.var(sample).item()

        # Score combinado (menor es mejor)
        score = magnitude_score + variance_score
        return score

    def _calculate_acquisition_score(self, name: str) -> float:
        """Calcula el score de adquisición bayesiana"""

        if name in self.acquisition_history:
            scores = self.acquisition_history[name]
            return np.mean(scores[-10:]) if scores else 0.0

        return 0.0

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

    def _calculate_bayesian_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia bayesiana"""

        if not self.sample_history:
            return 0.0

        convergence_rates = []

        for name, history in self.sample_history.items():
            if len(history) > 3:
                # Calcular convergencia basada en varianza de muestras
                variance = np.var(history[-10:])
                convergence_rate = 1.0 / (1.0 + variance)
                convergence_rates.append(convergence_rate)

        return np.mean(convergence_rates) if convergence_rates else 0.0

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

        expected_time = 8.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain

# Funciones de utilidad


def create_dynamic_adjuster(adjustment_method: str = "performance_based") -> BaseWeightImprover:
    """Factory para crear ajustadores dinámicos"""

    config = NeuralWeightConfig()

    if adjustment_method == "performance_based":
        return PerformanceBasedAdjuster(config)
    elif adjustment_method == "bayesian":
        return BayesianWeightAdjuster(config)
    else:
        raise ValueError(f"Método de ajuste no soportado: {adjustment_method}")


def adjust_model_weights_dynamically(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                     adjustment_method: str = "performance_based") -> WeightImprovementResult:
    """Función de conveniencia para ajustar pesos dinámicamente"""

    adjuster = create_dynamic_adjuster(adjustment_method)
    return adjuster.improve_weights(model, data_loader)


# Exportar clases y funciones principales
__all__ = [
    'DynamicAdjustmentConfig',
    'PerformanceBasedAdjuster',
    'BayesianWeightAdjuster',
    'create_dynamic_adjuster',
    'adjust_model_weights_dynamically'
]

logger.info("RFEN4_RN_4 - Sistema de Ajuste Dinámico de Pesos cargado correctamente")
