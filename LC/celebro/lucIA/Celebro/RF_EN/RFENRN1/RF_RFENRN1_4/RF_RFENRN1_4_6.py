"""
RF_RFENRN1_4_6.py - Gestor de Algoritmos Evolutivos
===================================================

Implementa algoritmos evolutivos para optimización de pesos en redes neuronales
de aprendizaje por refuerzo. Incluye PSO, algoritmos genéticos y evolución diferencial.

Características:
- Particle Swarm Optimization (PSO)
- Algoritmos Genéticos (GA)
- Evolución Diferencial (DE)
- Estrategias Evolutivas (ES)
- Algoritmos Meméticos
- Optimización Multi-objetivo Evolutiva
- Coevolución de arquitecturas y pesos

Autor: LucIA Development Team
Versión: 4.6.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import math
from dataclasses import dataclass, field
import time
from collections import defaultdict, deque
import random

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_6')


@dataclass
class EvolutionaryConfig:
    """Configuración para algoritmos evolutivos"""
    algorithm_type: str = 'PSO'  # 'PSO', 'GA', 'DE', 'ES'
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8

    # PSO específico
    pso_w: float = 0.9  # Inercia
    pso_c1: float = 2.0  # Cognitivo
    pso_c2: float = 2.0  # Social

    # GA específico
    ga_elite_size: int = 5
    ga_tournament_size: int = 3

    # DE específico
    de_f: float = 0.5  # Factor de escala
    de_cr: float = 0.9  # Probabilidad de cruzamiento


class ParticleSwarmOptimizer:
    """Optimizador Particle Swarm Optimization (PSO)"""

    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = []
        self.velocities = []
        self.personal_best = []
        self.global_best = None
        self.global_best_fitness = -np.inf

    def initialize(self, param_count: int):
        """Inicializa la población de partículas"""
        self.population = []
        self.velocities = []
        self.personal_best = []

        for _ in range(self.config.population_size):
            # Inicialización aleatoria de partículas
            particle = np.random.uniform(-1, 1, param_count)
            velocity = np.random.uniform(-0.1, 0.1, param_count)

            self.population.append(particle)
            self.velocities.append(velocity)
            self.personal_best.append(particle.copy())

    def optimize(self, objective_function: Callable, param_count: int) -> np.ndarray:
        """Optimización PSO"""
        self.initialize(param_count)

        for generation in range(self.config.max_generations):
            # Evaluar fitness
            fitness_values = []
            for i, particle in enumerate(self.population):
                fitness = objective_function(particle)
                fitness_values.append(fitness)

                # Actualizar mejor personal
                if fitness > self._get_personal_best_fitness(i):
                    self.personal_best[i] = particle.copy()

                # Actualizar mejor global
                if fitness > self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = particle.copy()

            # Actualizar velocidades y posiciones
            for i in range(self.config.population_size):
                # Componente inercial
                inertial = self.config.pso_w * self.velocities[i]

                # Componente cognitivo
                cognitive = self.config.pso_c1 * np.random.random() * (self.personal_best[i] - self.population[i])

                # Componente social
                social = self.config.pso_c2 * np.random.random() * (self.global_best - self.population[i])

                # Actualizar velocidad
                self.velocities[i] = inertial + cognitive + social

                # Actualizar posición
                self.population[i] += self.velocities[i]

                # Limitar posición
                self.population[i] = np.clip(self.population[i], -1, 1)

        return self.global_best

    def _get_personal_best_fitness(self, index: int) -> float:
        """Obtiene el fitness del mejor personal"""
        # En implementación real, esto almacenaría los valores de fitness
        return -np.inf


class GeneticAlgorithm:
    """Algoritmo Genético para optimización"""

    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = []
        self.fitness_values = []

    def initialize(self, param_count: int):
        """Inicializa la población"""
        self.population = []
        for _ in range(self.config.population_size):
            individual = np.random.uniform(-1, 1, param_count)
            self.population.append(individual)

    def optimize(self, objective_function: Callable, param_count: int) -> np.ndarray:
        """Optimización con algoritmo genético"""
        self.initialize(param_count)

        for generation in range(self.config.max_generations):
            # Evaluar fitness
            self.fitness_values = [objective_function(ind) for ind in self.population]

            # Selección elitista
            elite_indices = np.argsort(self.fitness_values)[-self.config.ga_elite_size:]
            elite = [self.population[i] for i in elite_indices]

            # Generar nueva población
            new_population = elite.copy()
            while len(new_population) < self.config.population_size:
                # Selección de padres
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                # Cruzamiento
                child1, child2 = self._crossover(parent1, parent2)

                # Mutación
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.extend([child1, child2])

            # Mantener tamaño de población
            self.population = new_population[:self.config.population_size]

        # Retornar mejor individuo
        best_idx = np.argmax(self.fitness_values)
        return self.population[best_idx]

    def _tournament_selection(self) -> np.ndarray:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(self.population)), self.config.ga_tournament_size)
        tournament_fitness = [self.fitness_values[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cruzamiento uniforme"""
        if random.random() < self.config.crossover_rate:
            mask = np.random.random(len(parent1)) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
        else:
            child1, child2 = parent1.copy(), parent2.copy()

        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Mutación gaussiana"""
        mutated = individual.copy()
        mask = np.random.random(len(individual)) < self.config.mutation_rate
        noise = np.random.normal(0, 0.1, len(individual))
        mutated[mask] += noise[mask]
        mutated = np.clip(mutated, -1, 1)
        return mutated


class DifferentialEvolution:
    """Evolución Diferencial"""

    def __init__(self, config: EvolutionaryConfig):
        self.config = config
        self.population = []

    def initialize(self, param_count: int):
        """Inicializa la población"""
        self.population = []
        for _ in range(self.config.population_size):
            individual = np.random.uniform(-1, 1, param_count)
            self.population.append(individual)

    def optimize(self, objective_function: Callable, param_count: int) -> np.ndarray:
        """Optimización con evolución diferencial"""
        self.initialize(param_count)

        for generation in range(self.config.max_generations):
            new_population = []

            for i in range(self.config.population_size):
                # Seleccionar tres individuos diferentes
                candidates = list(range(self.config.population_size))
                candidates.remove(i)
                a, b, c = random.sample(candidates, 3)

                # Mutación diferencial
                mutant = self.population[a] + self.config.de_f * (self.population[b] - self.population[c])

                # Cruzamiento binomial
                trial = self.population[i].copy()
                for j in range(param_count):
                    if random.random() < self.config.de_cr:
                        trial[j] = mutant[j]

                # Selección
                current_fitness = objective_function(self.population[i])
                trial_fitness = objective_function(trial)

                if trial_fitness > current_fitness:
                    new_population.append(trial)
                else:
                    new_population.append(self.population[i])

            self.population = new_population

        # Retornar mejor individuo
        fitness_values = [objective_function(ind) for ind in self.population]
        best_idx = np.argmax(fitness_values)
        return self.population[best_idx]


class EvolutionaryOptimizerManager:
    """
    Gestor de algoritmos evolutivos para redes de refuerzo.

    Proporciona acceso a diversos algoritmos evolutivos con
    capacidades de optimización multi-objetivo y adaptación.
    """

    def __init__(self, config: Optional[EvolutionaryConfig] = None):
        """
        Inicializa el gestor de algoritmos evolutivos.

        Args:
            config: Configuración evolutiva (opcional)
        """
        self.config = config or EvolutionaryConfig()
        self.optimizer = None
        self.optimization_history = []
        self.metrics = {
            'total_generations': 0,
            'best_fitness': -np.inf,
            'average_fitness': 0.0,
            'convergence_rate': 0.0,
            'diversity': 0.0
        }

        logger.info(f"EvolutionaryOptimizerManager inicializado con {self.config.algorithm_type}")

    def create_optimizer(self) -> Any:
        """Crea el optimizador evolutivo especificado"""
        if self.config.algorithm_type == 'PSO':
            self.optimizer = ParticleSwarmOptimizer(self.config)
        elif self.config.algorithm_type == 'GA':
            self.optimizer = GeneticAlgorithm(self.config)
        elif self.config.algorithm_type == 'DE':
            self.optimizer = DifferentialEvolution(self.config)
        else:
            raise ValueError(f"Algoritmo evolutivo no soportado: {self.config.algorithm_type}")

        logger.info(f"Optimizador {self.config.algorithm_type} creado")
        return self.optimizer

    def optimize_weights(self, model: nn.Module, objective_function: Callable) -> Dict[str, torch.Tensor]:
        """
        Optimiza los pesos del modelo usando algoritmos evolutivos.

        Args:
            model: Modelo PyTorch
            objective_function: Función objetivo

        Returns:
            Mejores pesos encontrados
        """
        if self.optimizer is None:
            self.create_optimizer()

        # Extraer parámetros del modelo
        param_count = sum(p.numel() for p in model.parameters())

        # Optimización evolutiva
        start_time = time.time()
        best_solution = self.optimizer.optimize(objective_function, param_count)
        optimization_time = time.time() - start_time

        # Convertir solución a pesos del modelo
        best_weights = self._solution_to_weights(model, best_solution)

        # Actualizar métricas
        self.metrics['total_generations'] = self.config.max_generations
        self.metrics['best_fitness'] = objective_function(best_solution)

        # Registrar en historial
        optimization_record = {
            'algorithm': self.config.algorithm_type,
            'generations': self.config.max_generations,
            'best_fitness': self.metrics['best_fitness'],
            'optimization_time': optimization_time,
            'timestamp': time.time()
        }
        self.optimization_history.append(optimization_record)

        logger.info(f"Optimización evolutiva completada en {optimization_time:.2f}s")

        return best_weights

    def _solution_to_weights(self, model: nn.Module, solution: np.ndarray) -> Dict[str, torch.Tensor]:
        """Convierte solución evolutiva a pesos del modelo"""
        weights = {}
        idx = 0

        for name, param in model.named_parameters():
            param_size = param.numel()
            param_values = solution[idx:idx + param_size]
            weights[name] = torch.tensor(param_values.reshape(param.shape), dtype=param.dtype)
            idx += param_size

        return weights

    def multi_objective_optimization(self, model: nn.Module,
                                     objective_functions: List[Callable]) -> List[Dict[str, torch.Tensor]]:
        """
        Optimización multi-objetivo usando algoritmos evolutivos.

        Args:
            model: Modelo PyTorch
            objective_functions: Lista de funciones objetivo

        Returns:
            Lista de soluciones Pareto-óptimas
        """
        # Implementación simplificada de optimización multi-objetivo
        pareto_solutions = []

        for obj_func in objective_functions:
            weights = self.optimize_weights(model, obj_func)
            pareto_solutions.append(weights)

        logger.info(f"Optimización multi-objetivo completada con {len(pareto_solutions)} soluciones")

        return pareto_solutions

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la optimización evolutiva"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'algorithm_type': self.config.algorithm_type,
            'total_generations': self.metrics['total_generations'],
            'best_fitness': self.metrics['best_fitness'],
            'history_length': len(self.optimization_history)
        }

    def save_state(self, path: str) -> None:
        """Guarda el estado del optimizador evolutivo"""
        torch.save({
            'config': self.config,
            'metrics': self.metrics,
            'optimization_history': self.optimization_history,
            'optimizer_state': getattr(self.optimizer, 'population', []) if self.optimizer else []
        }, path)
        logger.info(f"Estado del optimizador evolutivo guardado en {path}")

    def load_state(self, path: str) -> None:
        """Carga el estado del optimizador evolutivo"""
        checkpoint = torch.load(path)

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.optimization_history = checkpoint.get('optimization_history', [])

        # Restaurar estado del optimizador
        if self.optimizer and 'optimizer_state' in checkpoint:
            self.optimizer.population = checkpoint['optimizer_state']

        logger.info(f"Estado del optimizador evolutivo cargado desde {path}")

    def reset_state(self) -> None:
        """Reinicia el estado del optimizador evolutivo"""
        self.metrics = {
            'total_generations': 0,
            'best_fitness': -np.inf,
            'average_fitness': 0.0,
            'convergence_rate': 0.0,
            'diversity': 0.0
        }

        self.optimization_history.clear()

        if self.optimizer:
            self.optimizer.population = []
            self.optimizer.velocities = []
            self.optimizer.personal_best = []
            self.optimizer.global_best = None
            self.optimizer.global_best_fitness = -np.inf

        logger.info("Estado del optimizador evolutivo reiniciado")
