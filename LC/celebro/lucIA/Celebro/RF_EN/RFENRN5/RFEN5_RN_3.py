"""
RFEN5_RN_3 - Redes Neuronales Fractales para Optimización de Pesos
Implementación de redes neuronales fractales para optimización de pesos neuronales
Incluye: FractalNet, redes fractales adaptativas, y optimización fractal multi-escala
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
from . import BaseAIWeightOptimizer, AIWeightConfig, AIWeightMetrics, AIWeightOptimizationResult

logger = logging.getLogger(__name__)


@dataclass
class FractalNeuralConfig:
    """Configuración para redes neuronales fractales"""
    fractal_type: str = "fractalnet"  # fractalnet, adaptive_fractal, multi_scale_fractal
    fractal_depth: int = 3
    fractal_width: int = 64
    fractal_scaling: float = 0.5
    fractal_dropout: float = 0.1
    fractal_activation: str = "relu"  # relu, tanh, sigmoid
    fractal_normalization: bool = True
    fractal_residual: bool = True
    fractal_attention: bool = False
    fractal_adaptive_scaling: bool = True
    fractal_multi_scale: bool = True
    fractal_optimization_steps: int = 100
    fractal_convergence_threshold: float = 1e-6


class FractalNeuralOptimizer(BaseAIWeightOptimizer):
    """
    Optimizador de redes neuronales fractales para pesos
    """

    def __init__(self, config: AIWeightConfig):
        super().__init__(config)
        self.fractal_config = FractalNeuralConfig()
        self.fractal_blocks = {}
        self.fractal_weights = {}
        self.fractal_activations = {}
        self.fractal_scaling_factors = {}
        self.fractal_optimization_history = defaultdict(list)

    def optimize_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> AIWeightOptimizationResult:
        """Optimiza pesos usando redes neuronales fractales"""

        logger.info("Iniciando optimización con redes neuronales fractales")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_ai_neuron_weights(model)

        # Crear estructura fractal
        self._create_fractal_structure(model)

        # Optimización fractal
        for step in range(self.fractal_config.fractal_optimization_steps):
            # Aplicar optimización fractal
            self._apply_fractal_optimization(model, data_loader, step)

            # Verificar convergencia
            if self._check_fractal_convergence(step):
                break

        # Analizar pesos finales
        final_metrics = self.analyze_ai_neuron_weights(model)

        # Calcular mejoras
        optimization_score = self.calculate_ai_optimization_score(initial_metrics, final_metrics)

        end_time = time.time()

        return AIWeightOptimizationResult(
            optimization_score=optimization_score,
            convergence_rate=self._calculate_fractal_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_memory_optimization(),
            accuracy_improvement=optimization_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            ai_techniques_applied=["fractal_neural_optimization"],
            neurons_optimized=len(initial_metrics),
            total_neurons=len(initial_metrics),
            optimization_time=end_time - start_time,
            fractal_depth_used=self.fractal_config.fractal_depth
        )

    def analyze_ai_neuron_weights(self, model: nn.Module) -> Dict[str, AIWeightMetrics]:
        """Analiza los pesos de las neuronas usando IA"""

        neuron_metrics = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular métricas para cada parámetro
                metrics = AIWeightMetrics(
                    neuron_id=name,
                    weight_magnitude=torch.norm(param.data).item(),
                    weight_variance=torch.var(param.data).item(),
                    gradient_norm=torch.norm(param.grad).item() if param.grad is not None else 0.0,
                    activation_frequency=self._calculate_activation_frequency(param),
                    importance_score=self._calculate_importance_score(param),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    ai_optimization_score=self._calculate_ai_optimization_score(param),
                    evolutionary_fitness=0.0,
                    swarm_velocity=0.0,
                    fractal_complexity=self._calculate_fractal_complexity(param),
                    reinforcement_reward=0.0,
                    meta_learning_efficiency=0.0,
                    quantum_coherence=0.0,
                    ensemble_diversity=0.0,
                    real_time_ai_score=0.0,
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'fractal_scaling': self.fractal_scaling_factors.get(name, 1.0),
                        'fractal_depth': self._calculate_fractal_depth(param),
                        'fractal_activation': self._calculate_fractal_activation(param)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _create_fractal_structure(self, model: nn.Module) -> None:
        """Crea la estructura fractal para el modelo"""

        self.fractal_blocks = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Crear bloque fractal
                fractal_block = self._create_fractal_block(param, name)
                self.fractal_blocks[name] = fractal_block

                # Inicializar pesos fractales
                self._initialize_fractal_weights(param, name)

    def _create_fractal_block(self, param: torch.Tensor, name: str) -> Dict:
        """Crea un bloque fractal para un parámetro"""

        fractal_block = {
            'depth': self.fractal_config.fractal_depth,
            'width': self.fractal_config.fractal_width,
            'scaling': self.fractal_config.fractal_scaling,
            'activation': self.fractal_config.fractal_activation,
            'normalization': self.fractal_config.fractal_normalization,
            'residual': self.fractal_config.fractal_residual,
            'attention': self.fractal_config.fractal_attention,
            'adaptive_scaling': self.fractal_config.fractal_adaptive_scaling,
            'multi_scale': self.fractal_config.fractal_multi_scale
        }

        return fractal_block

    def _initialize_fractal_weights(self, param: torch.Tensor, name: str) -> None:
        """Inicializa los pesos fractales"""

        # Crear pesos fractales basados en el parámetro original
        fractal_weights = {}

        for depth in range(self.fractal_config.fractal_depth):
            # Crear pesos para cada nivel de profundidad fractal
            depth_weights = param.data.clone()

            # Aplicar escalado fractal
            scaling_factor = self.fractal_config.fractal_scaling ** depth
            depth_weights *= scaling_factor

            fractal_weights[f'depth_{depth}'] = depth_weights

        self.fractal_weights[name] = fractal_weights

        # Inicializar factores de escalado
        self.fractal_scaling_factors[name] = 1.0

    def _apply_fractal_optimization(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, step: int) -> None:
        """Aplica optimización fractal"""

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fractal_blocks:
                # Obtener bloque fractal
                fractal_block = self.fractal_blocks[name]

                # Aplicar optimización fractal
                optimized_weights = self._optimize_fractal_weights(param, fractal_block, step)

                # Actualizar pesos
                with torch.no_grad():
                    param.data.copy_(optimized_weights)

                # Registrar en historial
                self._record_fractal_optimization(name, optimized_weights, step)

    def _optimize_fractal_weights(self, param: torch.Tensor, fractal_block: Dict, step: int) -> torch.Tensor:
        """Optimiza los pesos usando estructura fractal"""

        # Obtener pesos fractales
        fractal_weights = self.fractal_weights.get(param.name if hasattr(param, 'name') else 'unknown', {})

        if not fractal_weights:
            return param.data

        # Aplicar optimización fractal multi-escala
        optimized_weights = torch.zeros_like(param.data)

        for depth in range(fractal_block['depth']):
            depth_key = f'depth_{depth}'
            if depth_key in fractal_weights:
                # Obtener pesos del nivel de profundidad
                depth_weights = fractal_weights[depth_key]

                # Aplicar escalado fractal
                scaling_factor = fractal_block['scaling'] ** depth

                # Aplicar activación fractal
                if fractal_block['activation'] == 'relu':
                    depth_weights = torch.relu(depth_weights)
                elif fractal_block['activation'] == 'tanh':
                    depth_weights = torch.tanh(depth_weights)
                elif fractal_block['activation'] == 'sigmoid':
                    depth_weights = torch.sigmoid(depth_weights)

                # Aplicar normalización fractal
                if fractal_block['normalization']:
                    depth_weights = self._apply_fractal_normalization(depth_weights)

                # Aplicar atención fractal
                if fractal_block['attention']:
                    depth_weights = self._apply_fractal_attention(depth_weights)

                # Aplicar escalado adaptativo
                if fractal_block['adaptive_scaling']:
                    scaling_factor *= self._calculate_adaptive_scaling_factor(depth_weights, step)

                # Sumar contribución fractal
                optimized_weights += depth_weights * scaling_factor

        # Aplicar conexión residual
        if fractal_block['residual']:
            optimized_weights += param.data * 0.1

        return optimized_weights

    def _apply_fractal_normalization(self, weights: torch.Tensor) -> torch.Tensor:
        """Aplica normalización fractal"""

        # Normalización basada en la norma fractal
        weight_norm = torch.norm(weights).item()
        if weight_norm > 0:
            normalized_weights = weights / weight_norm
        else:
            normalized_weights = weights

        return normalized_weights

    def _apply_fractal_attention(self, weights: torch.Tensor) -> torch.Tensor:
        """Aplica atención fractal"""

        # Calcular pesos de atención basados en la magnitud
        attention_weights = torch.sigmoid(torch.abs(weights))

        # Aplicar atención
        attended_weights = weights * attention_weights

        return attended_weights

    def _calculate_adaptive_scaling_factor(self, weights: torch.Tensor, step: int) -> float:
        """Calcula el factor de escalado adaptativo"""

        # Factor base
        base_factor = 1.0

        # Ajustar según la magnitud de los pesos
        weight_magnitude = torch.norm(weights).item()
        if weight_magnitude > 1.0:
            base_factor *= 1.1
        elif weight_magnitude < 0.1:
            base_factor *= 0.9

        # Ajustar según el paso de optimización
        step_factor = 1.0 - (step / self.fractal_config.fractal_optimization_steps) * 0.1

        return base_factor * step_factor

    def _record_fractal_optimization(self, name: str, optimized_weights: torch.Tensor, step: int) -> None:
        """Registra la optimización fractal"""

        self.fractal_optimization_history[name].append({
            'step': step,
            'weight_norm': torch.norm(optimized_weights).item(),
            'weight_variance': torch.var(optimized_weights).item(),
            'timestamp': time.time()
        })

    def _check_fractal_convergence(self, step: int) -> bool:
        """Verifica convergencia fractal"""

        if step < 10:
            return False

        # Verificar convergencia basada en el historial de optimización
        for name, history in self.fractal_optimization_history.items():
            if len(history) >= 5:
                recent_norms = [h['weight_norm'] for h in history[-5:]]
                norm_variance = np.var(recent_norms)

                if norm_variance < self.fractal_config.fractal_convergence_threshold:
                    return True

        return False

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

    def _calculate_ai_optimization_score(self, param: torch.Tensor) -> float:
        """Calcula el score de optimización con IA"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        ai_score = magnitude * 0.6 + variance * 0.4
        return min(1.0, ai_score / 100.0)

    def _calculate_fractal_complexity(self, param: torch.Tensor) -> float:
        """Calcula la complejidad fractal"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        complexity = magnitude * variance
        return min(1.0, complexity / 100.0)

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

    def _calculate_fractal_depth(self, param: torch.Tensor) -> int:
        """Calcula la profundidad fractal"""

        return self.fractal_config.fractal_depth

    def _calculate_fractal_activation(self, param: torch.Tensor) -> float:
        """Calcula la activación fractal"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 10.0)

    def _calculate_fractal_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia fractal"""

        if not self.fractal_optimization_history:
            return 0.0

        convergence_rates = []

        for name, history in self.fractal_optimization_history.items():
            if len(history) > 1:
                # Calcular tasa de convergencia basada en la estabilidad de los pesos
                weight_norms = [h['weight_norm'] for h in history]
                norm_variance = np.var(weight_norms)
                convergence_rate = 1.0 / (1.0 + norm_variance)
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

        expected_time = 6.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain


class AdaptiveFractalOptimizer(BaseAIWeightOptimizer):
    """
    Optimizador de redes fractales adaptativas
    """

    def __init__(self, config: AIWeightConfig):
        super().__init__(config)
        self.fractal_config = FractalNeuralConfig(fractal_type="adaptive_fractal")
        self.adaptive_fractal_blocks = {}
        self.adaptive_scaling_factors = {}
        self.fractal_adaptation_history = defaultdict(list)

    def optimize_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> AIWeightOptimizationResult:
        """Optimiza pesos usando redes fractales adaptativas"""

        logger.info("Iniciando optimización con redes fractales adaptativas")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_ai_neuron_weights(model)

        # Crear estructura fractal adaptativa
        self._create_adaptive_fractal_structure(model)

        # Optimización fractal adaptativa
        for step in range(self.fractal_config.fractal_optimization_steps):
            # Aplicar optimización fractal adaptativa
            self._apply_adaptive_fractal_optimization(model, data_loader, step)

            # Verificar convergencia
            if self._check_adaptive_fractal_convergence(step):
                break

        # Analizar pesos finales
        final_metrics = self.analyze_ai_neuron_weights(model)

        # Calcular mejoras
        optimization_score = self.calculate_ai_optimization_score(initial_metrics, final_metrics)

        end_time = time.time()

        return AIWeightOptimizationResult(
            optimization_score=optimization_score,
            convergence_rate=self._calculate_adaptive_fractal_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=0.0,
            accuracy_improvement=optimization_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            ai_techniques_applied=["adaptive_fractal_optimization"],
            neurons_optimized=len(initial_metrics),
            total_neurons=len(initial_metrics),
            optimization_time=end_time - start_time,
            fractal_depth_used=self.fractal_config.fractal_depth
        )

    def analyze_ai_neuron_weights(self, model: nn.Module) -> Dict[str, AIWeightMetrics]:
        """Analiza los pesos de las neuronas usando IA"""

        neuron_metrics = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                metrics = AIWeightMetrics(
                    neuron_id=name,
                    weight_magnitude=torch.norm(param.data).item(),
                    weight_variance=torch.var(param.data).item(),
                    gradient_norm=torch.norm(param.grad).item() if param.grad is not None else 0.0,
                    activation_frequency=self._calculate_activation_frequency(param),
                    importance_score=self._calculate_importance_score(param),
                    improvement_rate=0.0,
                    stability_score=self._calculate_stability_score(param),
                    contribution_to_loss=self._estimate_contribution_to_loss(param),
                    ai_optimization_score=self._calculate_ai_optimization_score(param),
                    evolutionary_fitness=0.0,
                    swarm_velocity=0.0,
                    fractal_complexity=self._calculate_fractal_complexity(param),
                    reinforcement_reward=0.0,
                    meta_learning_efficiency=0.0,
                    quantum_coherence=0.0,
                    ensemble_diversity=0.0,
                    real_time_ai_score=0.0,
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name)
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _create_adaptive_fractal_structure(self, model: nn.Module) -> None:
        """Crea la estructura fractal adaptativa"""

        self.adaptive_fractal_blocks = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                # Crear bloque fractal adaptativo
                adaptive_fractal_block = self._create_adaptive_fractal_block(param, name)
                self.adaptive_fractal_blocks[name] = adaptive_fractal_block

    def _create_adaptive_fractal_block(self, param: torch.Tensor, name: str) -> Dict:
        """Crea un bloque fractal adaptativo"""

        adaptive_fractal_block = {
            'depth': self.fractal_config.fractal_depth,
            'width': self.fractal_config.fractal_width,
            'scaling': self.fractal_config.fractal_scaling,
            'activation': self.fractal_config.fractal_activation,
            'normalization': self.fractal_config.fractal_normalization,
            'residual': self.fractal_config.fractal_residual,
            'attention': self.fractal_config.fractal_attention,
            'adaptive_scaling': True,  # Siempre activo para fractales adaptativos
            'multi_scale': self.fractal_config.fractal_multi_scale,
            'adaptation_rate': 0.1
        }

        return adaptive_fractal_block

    def _apply_adaptive_fractal_optimization(self, model: nn.Module, data_loader: torch.utils.data.DataLoader, step: int) -> None:
        """Aplica optimización fractal adaptativa"""

        for name, param in model.named_parameters():
            if param.requires_grad and name in self.adaptive_fractal_blocks:
                # Obtener bloque fractal adaptativo
                adaptive_fractal_block = self.adaptive_fractal_blocks[name]

                # Aplicar optimización fractal adaptativa
                optimized_weights = self._optimize_adaptive_fractal_weights(param, adaptive_fractal_block, step)

                # Actualizar pesos
                with torch.no_grad():
                    param.data.copy_(optimized_weights)

                # Registrar en historial
                self._record_adaptive_fractal_optimization(name, optimized_weights, step)

    def _optimize_adaptive_fractal_weights(self, param: torch.Tensor, adaptive_fractal_block: Dict, step: int) -> torch.Tensor:
        """Optimiza los pesos usando estructura fractal adaptativa"""

        # Crear pesos fractales adaptativos
        adaptive_fractal_weights = torch.zeros_like(param.data)

        for depth in range(adaptive_fractal_block['depth']):
            # Crear pesos del nivel de profundidad
            depth_weights = param.data.clone()

            # Aplicar escalado fractal adaptativo
            adaptive_scaling_factor = self._calculate_adaptive_fractal_scaling(depth_weights, depth, step)
            depth_weights *= adaptive_scaling_factor

            # Aplicar activación fractal
            if adaptive_fractal_block['activation'] == 'relu':
                depth_weights = torch.relu(depth_weights)
            elif adaptive_fractal_block['activation'] == 'tanh':
                depth_weights = torch.tanh(depth_weights)
            elif adaptive_fractal_block['activation'] == 'sigmoid':
                depth_weights = torch.sigmoid(depth_weights)

            # Aplicar normalización fractal adaptativa
            if adaptive_fractal_block['normalization']:
                depth_weights = self._apply_adaptive_fractal_normalization(depth_weights)

            # Aplicar atención fractal adaptativa
            if adaptive_fractal_block['attention']:
                depth_weights = self._apply_adaptive_fractal_attention(depth_weights)

            # Sumar contribución fractal adaptativa
            adaptive_fractal_weights += depth_weights

        # Aplicar conexión residual adaptativa
        if adaptive_fractal_block['residual']:
            adaptive_fractal_weights += param.data * 0.1

        return adaptive_fractal_weights

    def _calculate_adaptive_fractal_scaling(self, weights: torch.Tensor, depth: int, step: int) -> float:
        """Calcula el escalado fractal adaptativo"""

        # Factor base
        base_factor = self.fractal_config.fractal_scaling ** depth

        # Ajustar según la magnitud de los pesos
        weight_magnitude = torch.norm(weights).item()
        if weight_magnitude > 1.0:
            base_factor *= 1.1
        elif weight_magnitude < 0.1:
            base_factor *= 0.9

        # Ajustar según el paso de optimización
        step_factor = 1.0 - (step / self.fractal_config.fractal_optimization_steps) * 0.1

        return base_factor * step_factor

    def _apply_adaptive_fractal_normalization(self, weights: torch.Tensor) -> torch.Tensor:
        """Aplica normalización fractal adaptativa"""

        # Normalización adaptativa basada en la norma fractal
        weight_norm = torch.norm(weights).item()
        if weight_norm > 0:
            normalized_weights = weights / weight_norm
        else:
            normalized_weights = weights

        return normalized_weights

    def _apply_adaptive_fractal_attention(self, weights: torch.Tensor) -> torch.Tensor:
        """Aplica atención fractal adaptativa"""

        # Calcular pesos de atención adaptativos basados en la magnitud
        attention_weights = torch.sigmoid(torch.abs(weights))

        # Aplicar atención adaptativa
        attended_weights = weights * attention_weights

        return attended_weights

    def _record_adaptive_fractal_optimization(self, name: str, optimized_weights: torch.Tensor, step: int) -> None:
        """Registra la optimización fractal adaptativa"""

        self.fractal_adaptation_history[name].append({
            'step': step,
            'weight_norm': torch.norm(optimized_weights).item(),
            'weight_variance': torch.var(optimized_weights).item(),
            'timestamp': time.time()
        })

    def _check_adaptive_fractal_convergence(self, step: int) -> bool:
        """Verifica convergencia fractal adaptativa"""

        if step < 10:
            return False

        # Verificar convergencia basada en el historial de adaptación fractal
        for name, history in self.fractal_adaptation_history.items():
            if len(history) >= 5:
                recent_norms = [h['weight_norm'] for h in history[-5:]]
                norm_variance = np.var(recent_norms)

                if norm_variance < self.fractal_config.fractal_convergence_threshold:
                    return True

        return False

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

        return 0.5

    def _estimate_contribution_to_loss(self, param: torch.Tensor) -> float:
        """Estima la contribución a la pérdida"""

        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            return min(1.0, grad_norm / 5.0)
        return 0.0

    def _calculate_ai_optimization_score(self, param: torch.Tensor) -> float:
        """Calcula el score de optimización con IA"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        ai_score = magnitude * 0.6 + variance * 0.4
        return min(1.0, ai_score / 100.0)

    def _calculate_fractal_complexity(self, param: torch.Tensor) -> float:
        """Calcula la complejidad fractal"""

        magnitude = torch.norm(param.data).item()
        variance = torch.var(param.data).item()

        complexity = magnitude * variance
        return min(1.0, complexity / 100.0)

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

    def _calculate_adaptive_fractal_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia fractal adaptativa"""

        if not self.fractal_adaptation_history:
            return 0.0

        convergence_rates = []

        for name, history in self.fractal_adaptation_history.items():
            if len(history) > 1:
                # Calcular tasa de convergencia basada en la estabilidad de los pesos
                weight_norms = [h['weight_norm'] for h in history]
                norm_variance = np.var(weight_norms)
                convergence_rate = 1.0 / (1.0 + norm_variance)
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

        expected_time = 8.0
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain

# Funciones de utilidad


def create_fractal_optimizer(fractal_type: str = "fractalnet") -> BaseAIWeightOptimizer:
    """Factory para crear optimizadores fractales"""

    config = AIWeightConfig()

    if fractal_type == "fractalnet":
        return FractalNeuralOptimizer(config)
    elif fractal_type == "adaptive_fractal":
        return AdaptiveFractalOptimizer(config)
    else:
        raise ValueError(f"Tipo de fractal no soportado: {fractal_type}")


def optimize_weights_with_fractals(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                   fractal_type: str = "fractalnet") -> AIWeightOptimizationResult:
    """Función de conveniencia para optimización fractal de pesos"""

    optimizer = create_fractal_optimizer(fractal_type)
    return optimizer.optimize_weights(model, data_loader)


# Exportar clases y funciones principales
__all__ = [
    'FractalNeuralConfig',
    'FractalNeuralOptimizer',
    'AdaptiveFractalOptimizer',
    'create_fractal_optimizer',
    'optimize_weights_with_fractals'
]

logger.info("RFEN5_RN_3 - Redes Neuronales Fractales para Optimización de Pesos cargada correctamente")
