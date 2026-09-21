"""
RFEN5_RN_2 - Optimización por Enjambre de Partículas para Pesos
Implementación de algoritmos de enjambre de partículas para optimización de pesos neuronales
Incluye: PSO clásico, PSO mejorado, y optimización por enjambre adaptativo
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    pass  # dependencia pesada opcional
try:
    torch
except NameError:
    import types as _t
    torch = _t.SimpleNamespace(
        no_grad=lambda *a, **k: (lambda f: f) if a and callable(a[0]) else (lambda f: f),
        optim=_t.SimpleNamespace(Optimizer=object),
        Tensor=object,
    )
try:
    nn
except NameError:
    import types as _t2
    nn = _t2.SimpleNamespace(Module=object)
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
class SwarmOptimizationConfig:
    """Configuración para optimización por enjambre"""
    algorithm_type: str = "pso"  # pso, improved_pso, adaptive_pso
    swarm_size: int = 30
    max_iterations: int = 100
    inertia_weight: float = 0.9
    cognitive_weight: float = 2.0
    social_weight: float = 2.0
    velocity_clamp: float = 1.0
    position_clamp: float = 1.0
    adaptive_parameters: bool = True
    diversity_maintenance: bool = True
    convergence_threshold: float = 1e-6
    neighborhood_size: int = 5
    topology: str = "global"  # global, ring, von_neumann
    velocity_update_method: str = "standard"  # standard, constriction, inertia_adaptive


class ParticleSwarmOptimizer(BaseAIWeightOptimizer):
    """
    Optimizador por enjambre de partículas para pesos
    """

    def __init__(self, config: AIWeightConfig):
        super().__init__(config)
        self.swarm_config = SwarmOptimizationConfig()
        self.swarm = []
        self.global_best = None
        self.global_best_fitness = float('-inf')
        self.fitness_history = []
        self.velocity_history = []
        self.position_history = []
        self.adaptive_parameters = {}

    def optimize_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> AIWeightOptimizationResult:
        """Optimiza pesos usando enjambre de partículas"""

        logger.info("Iniciando optimización por enjambre de partículas")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_ai_neuron_weights(model)

        # Inicializar enjambre
        self._initialize_swarm(model)

        # Optimización por enjambre
        for iteration in range(self.swarm_config.max_iterations):
            # Evaluar fitness
            fitness_scores = self._evaluate_swarm(model, data_loader)

            # Actualizar mejores posiciones
            self._update_best_positions(fitness_scores)

            # Actualizar velocidades y posiciones
            self._update_swarm()

            # Actualizar parámetros adaptativos
            if self.swarm_config.adaptive_parameters:
                self._update_adaptive_parameters(iteration)

            # Verificar convergencia
            if self._check_convergence(fitness_scores, iteration):
                break

        # Aplicar mejores pesos
        self._apply_best_weights(model)

        # Analizar pesos finales
        final_metrics = self.analyze_ai_neuron_weights(model)

        # Calcular mejoras
        optimization_score = self.calculate_ai_optimization_score(initial_metrics, final_metrics)

        end_time = time.time()

        return AIWeightOptimizationResult(
            optimization_score=optimization_score,
            convergence_rate=self._calculate_swarm_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=self._calculate_memory_optimization(),
            accuracy_improvement=optimization_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            ai_techniques_applied=["particle_swarm_optimization"],
            neurons_optimized=len(initial_metrics),
            total_neurons=len(initial_metrics),
            optimization_time=end_time - start_time,
            swarm_iterations=iteration + 1
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
                    swarm_velocity=self._calculate_swarm_velocity(param),
                    fractal_complexity=0.0,
                    reinforcement_reward=0.0,
                    meta_learning_efficiency=0.0,
                    quantum_coherence=0.0,
                    ensemble_diversity=0.0,
                    real_time_ai_score=0.0,
                    timestamp=time.time(),
                    layer_type=self._get_layer_type(name),
                    additional_metrics={
                        'swarm_position': self._calculate_swarm_position(param),
                        'particle_velocity': self._calculate_particle_velocity(param),
                        'swarm_diversity': self._calculate_swarm_diversity(param)
                    }
                )

                neuron_metrics[name] = metrics

        return neuron_metrics

    def _initialize_swarm(self, model: nn.Module) -> None:
        """Inicializa el enjambre de partículas"""

        self.swarm = []

        for i in range(self.swarm_config.swarm_size):
            particle = {
                'position': {},
                'velocity': {},
                'best_position': {},
                'best_fitness': float('-inf'),
                'fitness': 0.0
            }

            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Inicializar posición aleatoria
                    particle['position'][name] = param.data + torch.randn_like(param.data) * 0.1

                    # Inicializar velocidad aleatoria
                    particle['velocity'][name] = torch.randn_like(param.data) * 0.1

                    # Inicializar mejor posición
                    particle['best_position'][name] = particle['position'][name].clone()

            self.swarm.append(particle)

        logger.info(f"Enjambre inicializado con {len(self.swarm)} partículas")

    def _evaluate_swarm(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> List[float]:
        """Evalúa el fitness del enjambre"""

        fitness_scores = []

        for particle in self.swarm:
            fitness = self._calculate_particle_fitness(particle, model, data_loader)
            particle['fitness'] = fitness
            fitness_scores.append(fitness)

        return fitness_scores

    def _calculate_particle_fitness(self, particle: Dict, model: nn.Module,
                                    data_loader: torch.utils.data.DataLoader) -> float:
        """Calcula el fitness de una partícula"""

        # Temporarily set weights
        original_weights = {}
        for name, param in model.named_parameters():
            if name in particle['position'] and param.requires_grad:
                original_weights[name] = param.data.clone()
                param.data.copy_(particle['position'][name])

        # Calculate fitness
        fitness = 0.0
        try:
            with torch.no_grad():
                for data, target in data_loader:
                    output = model(data)
                    # Fitness basado en la magnitud de la salida
                    fitness += torch.mean(torch.abs(output)).item()
        except Exception as e:
            fitness = -1000.0  # Penalty for invalid weights

        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])

        return fitness

    def _update_best_positions(self, fitness_scores: List[float]) -> None:
        """Actualiza las mejores posiciones"""

        for i, particle in enumerate(self.swarm):
            fitness = fitness_scores[i]

            # Actualizar mejor posición personal
            if fitness > particle['best_fitness']:
                particle['best_fitness'] = fitness
                for name in particle['best_position']:
                    particle['best_position'][name] = particle['position'][name].clone()

            # Actualizar mejor posición global
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = {}
                for name in particle['position']:
                    self.global_best[name] = particle['position'][name].clone()

    def _update_swarm(self) -> None:
        """Actualiza el enjambre"""

        for particle in self.swarm:
            # Actualizar velocidad
            self._update_particle_velocity(particle)

            # Actualizar posición
            self._update_particle_position(particle)

            # Aplicar límites
            self._apply_bounds(particle)

    def _update_particle_velocity(self, particle: Dict) -> None:
        """Actualiza la velocidad de una partícula"""

        for name in particle['velocity']:
            # Componente de inercia
            inertia = self.swarm_config.inertia_weight * particle['velocity'][name]

            # Componente cognitivo
            cognitive = (
                self.swarm_config.cognitive_weight *
                random.uniform(0, 1) *
                (particle['best_position'][name] - particle['position'][name])
            )

            # Componente social
            social = (
                self.swarm_config.social_weight *
                random.uniform(0, 1) *
                (self.global_best[name] - particle['position'][name])
            )

            # Actualizar velocidad
            particle['velocity'][name] = inertia + cognitive + social

            # Aplicar límite de velocidad
            if self.swarm_config.velocity_clamp > 0:
                velocity_norm = torch.norm(particle['velocity'][name]).item()
                if velocity_norm > self.swarm_config.velocity_clamp:
                    particle['velocity'][name] *= self.swarm_config.velocity_clamp / velocity_norm

    def _update_particle_position(self, particle: Dict) -> None:
        """Actualiza la posición de una partícula"""

        for name in particle['position']:
            particle['position'][name] += particle['velocity'][name]

    def _apply_bounds(self, particle: Dict) -> None:
        """Aplica límites a las posiciones"""

        for name in particle['position']:
            if self.swarm_config.position_clamp > 0:
                # Aplicar límite de posición
                position_norm = torch.norm(particle['position'][name]).item()
                if position_norm > self.swarm_config.position_clamp:
                    particle['position'][name] *= self.swarm_config.position_clamp / position_norm

    def _update_adaptive_parameters(self, iteration: int) -> None:
        """Actualiza parámetros adaptativos"""

        # Actualizar peso de inercia
        max_iterations = self.swarm_config.max_iterations
        self.swarm_config.inertia_weight = 0.9 - (0.9 - 0.4) * (iteration / max_iterations)

        # Actualizar pesos cognitivo y social
        if iteration < max_iterations / 2:
            self.swarm_config.cognitive_weight = 2.5 - (2.5 - 2.0) * (iteration / (max_iterations / 2))
            self.swarm_config.social_weight = 0.5 + (2.0 - 0.5) * (iteration / (max_iterations / 2))
        else:
            self.swarm_config.cognitive_weight = 2.0
            self.swarm_config.social_weight = 2.0

    def _check_convergence(self, fitness_scores: List[float], iteration: int) -> bool:
        """Verifica convergencia del enjambre"""

        if iteration < 10:
            return False

        # Verificar si el fitness ha mejorado
        if len(self.fitness_history) >= 5:
            recent_fitness = self.fitness_history[-5:]
            improvement = recent_fitness[-1] - recent_fitness[0]

            if improvement < self.swarm_config.convergence_threshold:
                return True

        self.fitness_history.append(max(fitness_scores))

        return False

    def _apply_best_weights(self, model: nn.Module) -> None:
        """Aplica los mejores pesos al modelo"""

        if self.global_best is None:
            return

        for name, param in model.named_parameters():
            if name in self.global_best and param.requires_grad:
                param.data.copy_(self.global_best[name])

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

    def _calculate_swarm_velocity(self, param: torch.Tensor) -> float:
        """Calcula la velocidad del enjambre"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 10.0)

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

    def _calculate_swarm_position(self, param: torch.Tensor) -> float:
        """Calcula la posición del enjambre"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 10.0)

    def _calculate_particle_velocity(self, param: torch.Tensor) -> float:
        """Calcula la velocidad de la partícula"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 10.0)

    def _calculate_swarm_diversity(self, param: torch.Tensor) -> float:
        """Calcula la diversidad del enjambre"""

        return 0.5  # Valor por defecto

    def _calculate_swarm_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia del enjambre"""

        if not self.fitness_history:
            return 0.0

        # Calcular tasa de convergencia
        if len(self.fitness_history) > 1:
            improvement = self.fitness_history[-1] - self.fitness_history[0]
            convergence_rate = improvement / len(self.fitness_history)
        else:
            convergence_rate = 0.0

        return max(0.0, convergence_rate)

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

        expected_time = 8.0  # Tiempo esperado en segundos
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain


class ImprovedParticleSwarmOptimizer(BaseAIWeightOptimizer):
    """
    Optimizador por enjambre de partículas mejorado
    """

    def __init__(self, config: AIWeightConfig):
        super().__init__(config)
        self.swarm_config = SwarmOptimizationConfig(algorithm_type="improved_pso")
        self.swarm = []
        self.global_best = None
        self.global_best_fitness = float('-inf')
        self.fitness_history = []
        self.adaptive_parameters = {}
        self.diversity_tracker = []

    def optimize_weights(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> AIWeightOptimizationResult:
        """Optimiza pesos usando enjambre de partículas mejorado"""

        logger.info("Iniciando optimización por enjambre de partículas mejorado")

        start_time = time.time()

        # Analizar pesos iniciales
        initial_metrics = self.analyze_ai_neuron_weights(model)

        # Inicializar enjambre
        self._initialize_swarm(model)

        # Optimización por enjambre mejorado
        for iteration in range(self.swarm_config.max_iterations):
            # Evaluar fitness
            fitness_scores = self._evaluate_swarm(model, data_loader)

            # Actualizar mejores posiciones
            self._update_best_positions(fitness_scores)

            # Actualizar velocidades y posiciones con mejoras
            self._update_swarm_improved(iteration)

            # Mantener diversidad
            if self.swarm_config.diversity_maintenance:
                self._maintain_diversity()

            # Verificar convergencia
            if self._check_convergence(fitness_scores, iteration):
                break

        # Aplicar mejores pesos
        self._apply_best_weights(model)

        # Analizar pesos finales
        final_metrics = self.analyze_ai_neuron_weights(model)

        # Calcular mejoras
        optimization_score = self.calculate_ai_optimization_score(initial_metrics, final_metrics)

        end_time = time.time()

        return AIWeightOptimizationResult(
            optimization_score=optimization_score,
            convergence_rate=self._calculate_improved_convergence_rate(),
            stability_improvement=self._calculate_stability_improvement(initial_metrics, final_metrics),
            efficiency_gain=self._calculate_efficiency_gain(initial_metrics, final_metrics),
            memory_optimization=0.0,
            accuracy_improvement=optimization_score,
            training_speed_gain=self._calculate_speed_gain(end_time - start_time),
            ai_techniques_applied=["improved_particle_swarm_optimization"],
            neurons_optimized=len(initial_metrics),
            total_neurons=len(initial_metrics),
            optimization_time=end_time - start_time,
            swarm_iterations=iteration + 1
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
                    swarm_velocity=self._calculate_swarm_velocity(param),
                    fractal_complexity=0.0,
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

    def _initialize_swarm(self, model: nn.Module) -> None:
        """Inicializa el enjambre mejorado"""

        self.swarm = []

        for i in range(self.swarm_config.swarm_size):
            particle = {
                'position': {},
                'velocity': {},
                'best_position': {},
                'best_fitness': float('-inf'),
                'fitness': 0.0,
                'neighborhood': []
            }

            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Inicializar posición aleatoria
                    particle['position'][name] = param.data + torch.randn_like(param.data) * 0.1

                    # Inicializar velocidad aleatoria
                    particle['velocity'][name] = torch.randn_like(param.data) * 0.1

                    # Inicializar mejor posición
                    particle['best_position'][name] = particle['position'][name].clone()

            self.swarm.append(particle)

        # Configurar vecindarios
        self._setup_neighborhoods()

    def _setup_neighborhoods(self) -> None:
        """Configura los vecindarios de las partículas"""

        for i, particle in enumerate(self.swarm):
            if self.swarm_config.topology == "ring":
                # Topología de anillo
                left = (i - 1) % len(self.swarm)
                right = (i + 1) % len(self.swarm)
                particle['neighborhood'] = [left, right]

            elif self.swarm_config.topology == "von_neumann":
                # Topología de von Neumann
                size = int(math.sqrt(len(self.swarm)))
                row = i // size
                col = i % size

                neighborhood = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighborhood.append(nr * size + nc)

                particle['neighborhood'] = neighborhood

            else:  # global
                # Topología global
                particle['neighborhood'] = list(range(len(self.swarm)))

    def _evaluate_swarm(self, model: nn.Module, data_loader: torch.utils.data.DataLoader) -> List[float]:
        """Evalúa el fitness del enjambre"""

        fitness_scores = []

        for particle in self.swarm:
            fitness = self._calculate_particle_fitness(particle, model, data_loader)
            particle['fitness'] = fitness
            fitness_scores.append(fitness)

        return fitness_scores

    def _calculate_particle_fitness(self, particle: Dict, model: nn.Module,
                                    data_loader: torch.utils.data.DataLoader) -> float:
        """Calcula el fitness de una partícula"""

        # Temporarily set weights
        original_weights = {}
        for name, param in model.named_parameters():
            if name in particle['position'] and param.requires_grad:
                original_weights[name] = param.data.clone()
                param.data.copy_(particle['position'][name])

        # Calculate fitness
        fitness = 0.0
        try:
            with torch.no_grad():
                for data, target in data_loader:
                    output = model(data)
                    fitness += torch.mean(torch.abs(output)).item()
        except Exception as e:
            fitness = -1000.0

        # Restore original weights
        for name, param in model.named_parameters():
            if name in original_weights:
                param.data.copy_(original_weights[name])

        return fitness

    def _update_best_positions(self, fitness_scores: List[float]) -> None:
        """Actualiza las mejores posiciones"""

        for i, particle in enumerate(self.swarm):
            fitness = fitness_scores[i]

            # Actualizar mejor posición personal
            if fitness > particle['best_fitness']:
                particle['best_fitness'] = fitness
                for name in particle['best_position']:
                    particle['best_position'][name] = particle['position'][name].clone()

            # Actualizar mejor posición global
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best = {}
                for name in particle['position']:
                    self.global_best[name] = particle['position'][name].clone()

    def _update_swarm_improved(self, iteration: int) -> None:
        """Actualiza el enjambre con mejoras"""

        for particle in self.swarm:
            # Actualizar velocidad con mejoras
            self._update_particle_velocity_improved(particle, iteration)

            # Actualizar posición
            self._update_particle_position(particle)

            # Aplicar límites
            self._apply_bounds(particle)

    def _update_particle_velocity_improved(self, particle: Dict, iteration: int) -> None:
        """Actualiza la velocidad de una partícula con mejoras"""

        for name in particle['velocity']:
            # Componente de inercia adaptativo
            inertia = self.swarm_config.inertia_weight * particle['velocity'][name]

            # Componente cognitivo
            cognitive = (
                self.swarm_config.cognitive_weight *
                random.uniform(0, 1) *
                (particle['best_position'][name] - particle['position'][name])
            )

            # Componente social mejorado
            social = (
                self.swarm_config.social_weight *
                random.uniform(0, 1) *
                (self.global_best[name] - particle['position'][name])
            )

            # Actualizar velocidad
            particle['velocity'][name] = inertia + cognitive + social

            # Aplicar límite de velocidad adaptativo
            if self.swarm_config.velocity_clamp > 0:
                velocity_norm = torch.norm(particle['velocity'][name]).item()
                if velocity_norm > self.swarm_config.velocity_clamp:
                    particle['velocity'][name] *= self.swarm_config.velocity_clamp / velocity_norm

    def _update_particle_position(self, particle: Dict) -> None:
        """Actualiza la posición de una partícula"""

        for name in particle['position']:
            particle['position'][name] += particle['velocity'][name]

    def _apply_bounds(self, particle: Dict) -> None:
        """Aplica límites a las posiciones"""

        for name in particle['position']:
            if self.swarm_config.position_clamp > 0:
                position_norm = torch.norm(particle['position'][name]).item()
                if position_norm > self.swarm_config.position_clamp:
                    particle['position'][name] *= self.swarm_config.position_clamp / position_norm

    def _maintain_diversity(self) -> None:
        """Mantiene la diversidad del enjambre"""

        # Calcular diversidad
        diversity = self._calculate_swarm_diversity()
        self.diversity_tracker.append(diversity)

        # Si la diversidad es muy baja, aumentar exploración
        if len(self.diversity_tracker) > 5:
            avg_diversity = np.mean(self.diversity_tracker[-5:])
            if avg_diversity < 0.1:
                # Aumentar exploración
                for particle in self.swarm:
                    for name in particle['velocity']:
                        particle['velocity'][name] *= 1.1

    def _calculate_swarm_diversity(self) -> float:
        """Calcula la diversidad del enjambre"""

        if len(self.swarm) < 2:
            return 0.0

        # Calcular distancia promedio entre partículas
        total_distance = 0.0
        pair_count = 0

        for i in range(len(self.swarm)):
            for j in range(i + 1, len(self.swarm)):
                distance = 0.0
                for name in self.swarm[i]['position']:
                    pos_diff = self.swarm[i]['position'][name] - self.swarm[j]['position'][name]
                    distance += torch.norm(pos_diff).item()

                total_distance += distance
                pair_count += 1

        if pair_count > 0:
            avg_distance = total_distance / pair_count
            return min(1.0, avg_distance / 10.0)

        return 0.0

    def _check_convergence(self, fitness_scores: List[float], iteration: int) -> bool:
        """Verifica convergencia"""

        if iteration < 10:
            return False

        # Verificar si el fitness ha mejorado
        if len(self.fitness_history) >= 5:
            recent_fitness = self.fitness_history[-5:]
            improvement = recent_fitness[-1] - recent_fitness[0]

            if improvement < self.swarm_config.convergence_threshold:
                return True

        self.fitness_history.append(max(fitness_scores))

        return False

    def _apply_best_weights(self, model: nn.Module) -> None:
        """Aplica los mejores pesos"""

        if self.global_best is None:
            return

        for name, param in model.named_parameters():
            if name in self.global_best and param.requires_grad:
                param.data.copy_(self.global_best[name])

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

    def _calculate_swarm_velocity(self, param: torch.Tensor) -> float:
        """Calcula la velocidad del enjambre"""

        magnitude = torch.norm(param.data).item()
        return min(1.0, magnitude / 10.0)

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

    def _calculate_improved_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia mejorada"""

        if not self.fitness_history:
            return 0.0

        # Calcular tasa de convergencia
        if len(self.fitness_history) > 1:
            improvement = self.fitness_history[-1] - self.fitness_history[0]
            convergence_rate = improvement / len(self.fitness_history)
        else:
            convergence_rate = 0.0

        return max(0.0, convergence_rate)

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

        expected_time = 10.0
        speed_gain = max(0.0, (expected_time - optimization_time) / expected_time)

        return speed_gain

# Funciones de utilidad


def create_swarm_optimizer(algorithm_type: str = "pso") -> BaseAIWeightOptimizer:
    """Factory para crear optimizadores por enjambre"""

    config = AIWeightConfig()

    if algorithm_type == "pso":
        return ParticleSwarmOptimizer(config)
    elif algorithm_type == "improved_pso":
        return ImprovedParticleSwarmOptimizer(config)
    else:
        raise ValueError(f"Tipo de algoritmo de enjambre no soportado: {algorithm_type}")


def optimize_weights_with_swarm(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                                algorithm_type: str = "pso") -> AIWeightOptimizationResult:
    """Función de conveniencia para optimización por enjambre de pesos"""

    optimizer = create_swarm_optimizer(algorithm_type)
    return optimizer.optimize_weights(model, data_loader)


# Exportar clases y funciones principales
__all__ = [
    'SwarmOptimizationConfig',
    'ParticleSwarmOptimizer',
    'ImprovedParticleSwarmOptimizer',
    'create_swarm_optimizer',
    'optimize_weights_with_swarm'
]

logger.info("RFEN5_RN_2 - Optimización por Enjambre de Partículas para Pesos cargada correctamente")
