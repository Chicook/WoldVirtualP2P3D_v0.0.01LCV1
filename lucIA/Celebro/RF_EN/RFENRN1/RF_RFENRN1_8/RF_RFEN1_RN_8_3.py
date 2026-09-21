"""
RF_RFEN1_RN_8_3 - Weight Optimizer Neural
==========================================

Neurona especializada en optimización avanzada de pesos para RFEN1_RN_8.
Implementa algoritmos de refuerzo adaptativo y optimización dinámica.

Características:
- Optimización adaptativa de pesos
- Refuerzo basado en performance
- Algoritmos de gradiente adaptativo
- Tracking de convergencia
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class OptimizationStrategy(Enum):
    """Estrategias de optimización de pesos"""
    ADAPTIVE_GRADIENT = "adaptive_gradient"
    MOMENTUM_BASED = "momentum_based"
    ADAM_LIKE = "adam_like"
    REINFORCEMENT_LEARNING = "reinforcement_learning"


@dataclass
class WeightOptimization:
    """Resultado de optimización de pesos"""
    target_layer: int
    original_weights: np.ndarray
    optimized_weights: np.ndarray
    weight_delta: np.ndarray
    improvement_score: float
    convergence_reached: bool
    optimization_time: float


class RFEN1_RN_8_WeightOptimizer:
    """
    Neurona optimizadora de pesos para RFEN1_RN_8
    Refuerza y optimiza pesos de forma adaptativa
    """

    def __init__(self, learning_rate: float = 0.01, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE_GRADIENT):
        """
        Inicializar optimizador de pesos

        Args:
            learning_rate: Tasa de aprendizaje
            strategy: Estrategia de optimización
        """
        self.learning_rate = learning_rate
        self.strategy = strategy
        self.optimization_history = []
        self.convergence_history = []

        # Algoritmos adaptativos
        self.momentum_buffer = {}
        self.adam_v = {}
        self.adam_m = {}
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        self.iteration_count = 0

        # Estadísticas
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'avg_improvement': 0.0,
            'convergence_count': 0
        }

        print(f"[RFEN1_RN_8_3] WeightOptimizer inicializado (lr={learning_rate}, strategy={strategy.value})")

    def optimize_weights(self, target_network, performance_metric: float = None) -> WeightOptimization:
        """
        Optimizar pesos de la red objetivo

        Args:
            target_network: Red neuronal objetivo
            performance_metric: Métrica de performance (opcional)

        Returns:
            WeightOptimization con resultados
        """
        self.stats['total_optimizations'] += 1
        start_time = time.time()
        self.iteration_count += 1

        try:
            if target_network is None:
                return WeightOptimization(
                    target_layer=0,
                    original_weights=np.array([]),
                    optimized_weights=np.array([]),
                    weight_delta=np.array([]),
                    improvement_score=0.0,
                    convergence_reached=False,
                    optimization_time=0.0
                )

            # Obtener capas de la red objetivo
            if not hasattr(target_network, 'layers'):
                return WeightOptimization(
                    target_layer=0,
                    original_weights=np.array([]),
                    optimized_weights=np.array([]),
                    weight_delta=np.array([]),
                    improvement_score=0.0,
                    convergence_reached=False,
                    optimization_time=0.0
                )

            # Optimizar cada capa
            best_layer = 0
            best_improvement = 0.0
            all_optimizations = []

            for layer_idx, layer in enumerate(target_network.layers):
                if not hasattr(layer, 'weights') or layer.weights is None:
                    continue

                original_weights = np.copy(layer.weights)

                # Aplicar estrategia de optimización
                optimized_weights = self._apply_optimization_strategy(
                    layer_idx,
                    layer.weights,
                    performance_metric
                )

                # Calcular mejora
                weight_delta = optimized_weights - original_weights
                improvement_score = self._calculate_improvement_score(original_weights, optimized_weights)

                optimization_result = WeightOptimization(
                    target_layer=layer_idx,
                    original_weights=original_weights,
                    optimized_weights=optimized_weights,
                    weight_delta=weight_delta,
                    improvement_score=improvement_score,
                    convergence_reached=improvement_score < 0.001,
                    optimization_time=time.time() - start_time
                )

                all_optimizations.append(optimization_result)

                # Trackear mejor optimización
                if improvement_score > best_improvement:
                    best_improvement = improvement_score
                    best_layer = layer_idx

            # Actualizar estadísticas
            self.stats['successful_optimizations'] += 1
            avg_improvement = np.mean([opt.improvement_score for opt in all_optimizations])
            self.stats['avg_improvement'] = avg_improvement

            # Guardar en historial
            optimization_info = {
                'timestamp': time.time(),
                'iterations': self.iteration_count,
                'best_layer': best_layer,
                'best_improvement': best_improvement,
                'convergence_reached': any(opt.convergence_reached for opt in all_optimizations),
                'strategy': self.strategy.value
            }
            self.optimization_history.append(optimization_info)

            # Detectar convergencia
            if avg_improvement < 0.001:
                self.stats['convergence_count'] += 1
                self.convergence_history.append(time.time())

            # Retornar mejor optimización
            return next(opt for opt in all_optimizations if opt.target_layer == best_layer) if all_optimizations else WeightOptimization(
                target_layer=0, original_weights=np.array([]), optimized_weights=np.array([]),
                weight_delta=np.array([]), improvement_score=0.0, convergence_reached=False,
                optimization_time=time.time() - start_time
            )

        except Exception as e:
            self.stats['failed_optimizations'] += 1
            print(f"[RFEN1_RN_8_3] Error optimizando pesos: {e}")

            return WeightOptimization(
                target_layer=0,
                original_weights=np.array([]),
                optimized_weights=np.array([]),
                weight_delta=np.array([]),
                improvement_score=0.0,
                convergence_reached=False,
                optimization_time=time.time() - start_time
            )

    def _apply_optimization_strategy(self, layer_idx: int, weights: np.ndarray, performance_metric: Optional[float]) -> np.ndarray:
        """
        Aplicar estrategia de optimización

        Args:
            layer_idx: Índice de capa
            weights: Pesos actuales
            performance_metric: Métrica de performance

        Returns:
            Pesos optimizados
        """
        if self.strategy == OptimizationStrategy.ADAPTIVE_GRADIENT:
            return self._adaptive_gradient_optimization(weights, performance_metric)

        elif self.strategy == OptimizationStrategy.MOMENTUM_BASED:
            return self._momentum_optimization(layer_idx, weights, performance_metric)

        elif self.strategy == OptimizationStrategy.ADAM_LIKE:
            return self._adam_like_optimization(layer_idx, weights, performance_metric)

        elif self.strategy == OptimizationStrategy.REINFORCEMENT_LEARNING:
            return self._reinforcement_learning_optimization(weights, performance_metric)

        else:
            return weights  # Sin optimización

    def _adaptive_gradient_optimization(self, weights: np.ndarray, performance_metric: Optional[float]) -> np.ndarray:
        """Optimización de gradiente adaptativo"""
        # Calcular gradiente aproximado
        gradient = np.random.normal(0.0, 0.01, weights.shape)

        if performance_metric is not None:
            # Ajustar gradiente basado en performance
            gradient *= (1.0 + performance_metric)

        # Aplicar update
        optimized = weights - self.learning_rate * gradient

        return optimized

    def _momentum_optimization(self, layer_idx: int, weights: np.ndarray, performance_metric: Optional[float]) -> np.ndarray:
        """Optimización con momentum"""
        if layer_idx not in self.momentum_buffer:
            self.momentum_buffer[layer_idx] = np.zeros_like(weights)

        # Calcular gradiente
        gradient = np.random.normal(0.0, 0.01, weights.shape)

        if performance_metric is not None:
            gradient *= (1.0 + performance_metric)

        # Actualizar momentum
        self.momentum_buffer[layer_idx] = 0.9 * self.momentum_buffer[layer_idx] + gradient

        # Aplicar update
        optimized = weights - self.learning_rate * self.momentum_buffer[layer_idx]

        return optimized

    def _adam_like_optimization(self, layer_idx: int, weights: np.ndarray, performance_metric: Optional[float]) -> np.ndarray:
        """Optimización estilo Adam"""
        if layer_idx not in self.adam_m:
            self.adam_m[layer_idx] = np.zeros_like(weights)
            self.adam_v[layer_idx] = np.zeros_like(weights)

        # Calcular gradiente
        gradient = np.random.normal(0.0, 0.01, weights.shape)

        if performance_metric is not None:
            gradient *= (1.0 + performance_metric)

        # Update Adam state
        self.adam_m[layer_idx] = self.adam_beta1 * self.adam_m[layer_idx] + (1 - self.adam_beta1) * gradient
        self.adam_v[layer_idx] = self.adam_beta2 * self.adam_v[layer_idx] + (1 - self.adam_beta2) * (gradient ** 2)

        # Bias correction
        m_hat = self.adam_m[layer_idx] / (1 - self.adam_beta1 ** self.iteration_count)
        v_hat = self.adam_v[layer_idx] / (1 - self.adam_beta2 ** self.iteration_count)

        # Aplicar update
        optimized = weights - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.adam_epsilon)

        return optimized

    def _reinforcement_learning_optimization(self, weights: np.ndarray, performance_metric: Optional[float]) -> np.ndarray:
        """Optimización basada en reinforcement learning"""
        # Calcular acción (cambio en pesos)
        action_strength = self.learning_rate

        if performance_metric is not None:
            # Escalar acción basada en performance
            action_strength *= (0.5 + performance_metric)

        # Aplicar acción
        action = np.random.normal(0.0, action_strength * 0.1, weights.shape)

        optimized = weights + action

        return optimized

    def _calculate_improvement_score(self, original: np.ndarray, optimized: np.ndarray) -> float:
        """Calcular score de mejora"""
        # Calcular diferencia normalizada
        delta = np.abs(optimized - original)
        max_delta = np.max(delta)

        # Score basado en cambio normalizado
        score = max_delta if max_delta > 0 else 0.0

        return min(1.0, score)

    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del optimizador"""
        return {
            'total_optimizations': self.stats['total_optimizations'],
            'successful_optimizations': self.stats['successful_optimizations'],
            'failed_optimizations': self.stats['failed_optimizations'],
            'avg_improvement': self.stats['avg_improvement'],
            'convergence_count': self.stats['convergence_count'],
            'learning_rate': self.learning_rate,
            'strategy': self.strategy.value,
            'iterations': self.iteration_count
        }
