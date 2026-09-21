try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass  # dependencia pesada opcional
import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import random
import math
from collections import defaultdict, deque
import copy
import time
import threading

# Configuración del logger
logger = logging.getLogger(__name__)

class EvolutionaryWeightOptimizer(ABC):
    """
    Clase base abstracta para optimizadores evolutivos de pesos.
    Define la interfaz común para todas las estrategias de optimización evolutiva.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("EvolutionaryWeightOptimizer base inicializado.")

    @abstractmethod
    def evolutionary_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        """
        Método abstracto para optimizar los pesos usando optimización evolutiva.
        Debe ser implementado por las subclases.
        """
        pass

class GeneticAlgorithmWeightOptimizer(EvolutionaryWeightOptimizer):
    """
    Optimizador de pesos basado en Algoritmos Genéticos.
    Utiliza algoritmos genéticos para optimizar los pesos de la red neuronal.
    """
    def __init__(self, population_size: int = 50, generations: int = 100, 
                 mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 selection_method: str = "tournament", tournament_size: int = 3, config=None):
        super().__init__(config)
        self.population_size = self.config.get('population_size', population_size)
        self.generations = self.config.get('generations', generations)
        self.mutation_rate = self.config.get('mutation_rate', mutation_rate)
        self.crossover_rate = self.config.get('crossover_rate', crossover_rate)
        self.selection_method = self.config.get('selection_method', selection_method)
        self.tournament_size = self.config.get('tournament_size', tournament_size)
        self.population = []
        self.fitness_scores = []
        self.generation_history = []
        logger.info(f"GeneticAlgorithmWeightOptimizer inicializado: pop_size={self.population_size}, generations={self.generations}")

    def _create_individual(self, model: nn.Module) -> Dict:
        """
        Crea un individuo (conjunto de pesos) para la población.
        """
        individual = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Crear pesos aleatorios basados en la forma del parámetro
                individual[name] = torch.randn_like(param.data) * 0.1
        return individual

    def _initialize_population(self, model: nn.Module) -> None:
        """
        Inicializa la población con individuos aleatorios.
        """
        self.population = []
        for _ in range(self.population_size):
            individual = self._create_individual(model)
            self.population.append(individual)
        logger.info(f"Población inicializada con {len(self.population)} individuos.")

    def _evaluate_fitness(self, model: nn.Module, individual: Dict, data_loader=None) -> float:
        """
        Evalúa la aptitud (fitness) de un individuo.
        """
        # Guardar pesos originales
        original_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_weights[name] = param.data.clone()
        
        # Aplicar pesos del individuo
        for name, param in model.named_parameters():
            if param.requires_grad and name in individual:
                param.data = individual[name]
        
        # Evaluar modelo
        if data_loader is None:
            # Evaluación simplificada basada en normas de pesos
            total_norm = sum(torch.norm(param.data).item() for param in model.parameters() if param.requires_grad)
            fitness = 1.0 / (1.0 + total_norm / 1000.0)  # Normalizar
        else:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, targets in data_loader:
                    outputs = model(inputs)
                    loss = F.mse_loss(outputs, targets)
                    total_loss += loss.item()
            fitness = 1.0 / (1.0 + total_loss)  # Convertir pérdida a fitness
        
        # Restaurar pesos originales
        for name, param in model.named_parameters():
            if param.requires_grad and name in original_weights:
                param.data = original_weights[name]
        
        return fitness

    def _tournament_selection(self, tournament_size: int = None) -> Dict:
        """
        Selección por torneo.
        """
        if tournament_size is None:
            tournament_size = self.tournament_size
        
        # Seleccionar individuos aleatorios para el torneo
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_individuals = [self.population[i] for i in tournament_indices]
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        
        # Seleccionar el mejor individuo del torneo
        best_idx = np.argmax(tournament_fitness)
        return tournament_individuals[best_idx]

    def _roulette_selection(self) -> Dict:
        """
        Selección por ruleta.
        """
        # Calcular probabilidades de selección
        total_fitness = sum(self.fitness_scores)
        if total_fitness == 0:
            return random.choice(self.population)
        
        probabilities = [fitness / total_fitness for fitness in self.fitness_scores]
        
        # Selección aleatoria basada en probabilidades
        r = random.random()
        cumulative_probability = 0.0
        
        for i, probability in enumerate(probabilities):
            cumulative_probability += probability
            if r <= cumulative_probability:
                return self.population[i]
        
        return self.population[-1]  # Fallback

    def _select_parents(self) -> Tuple[Dict, Dict]:
        """
        Selecciona dos padres para la reproducción.
        """
        if self.selection_method == "tournament":
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
        elif self.selection_method == "roulette":
            parent1 = self._roulette_selection()
            parent2 = self._roulette_selection()
        else:
            # Selección aleatoria
            parent1 = random.choice(self.population)
            parent2 = random.choice(self.population)
        
        return parent1, parent2

    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Cruce entre dos padres para producir descendencia.
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        child1 = {}
        child2 = {}
        
        for name in parent1:
            if random.random() < 0.5:
                # Cruce uniforme
                child1[name] = parent1[name].clone()
                child2[name] = parent2[name].clone()
            else:
                # Cruce de punto único
                child1[name] = parent2[name].clone()
                child2[name] = parent1[name].clone()
        
        return child1, child2

    def _mutate(self, individual: Dict) -> Dict:
        """
        Mutación de un individuo.
        """
        mutated_individual = {}
        
        for name, weights in individual.items():
            if random.random() < self.mutation_rate:
                # Mutación gaussiana
                mutation_noise = torch.randn_like(weights) * 0.1
                mutated_individual[name] = weights + mutation_noise
            else:
                mutated_individual[name] = weights.clone()
        
        return mutated_individual

    def _evolve_generation(self, model: nn.Module, data_loader=None) -> None:
        """
        Evoluciona una generación.
        """
        new_population = []
        
        # Evaluar fitness de la población actual
        self.fitness_scores = []
        for individual in self.population:
            fitness = self._evaluate_fitness(model, individual, data_loader)
            self.fitness_scores.append(fitness)
        
        # Guardar estadísticas de la generación
        generation_stats = {
            'generation': len(self.generation_history),
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': np.mean(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores)
        }
        self.generation_history.append(generation_stats)
        
        # Crear nueva población
        while len(new_population) < self.population_size:
            # Seleccionar padres
            parent1, parent2 = self._select_parents()
            
            # Cruce
            child1, child2 = self._crossover(parent1, parent2)
            
            # Mutación
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Mantener el tamaño de población
        self.population = new_population[:self.population_size]
        
        logger.debug(f"Generación {len(self.generation_history)}: Mejor fitness = {generation_stats['best_fitness']:.4f}")

    def evolutionary_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización evolutiva de pesos con Algoritmos Genéticos.")
        
        # Inicializar población
        self._initialize_population(model)
        
        # Evolución
        for generation in range(self.generations):
            self._evolve_generation(model, data_loader)
            
            # Log de progreso
            if generation % 10 == 0:
                best_fitness = max(self.fitness_scores)
                avg_fitness = np.mean(self.fitness_scores)
                logger.info(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, Promedio = {avg_fitness:.4f}")
        
        # Aplicar el mejor individuo al modelo
        best_idx = np.argmax(self.fitness_scores)
        best_individual = self.population[best_idx]
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in best_individual:
                param.data = best_individual[name]
        
        logger.info(f"Optimización evolutiva completada. Mejor fitness final = {max(self.fitness_scores):.4f}")
        return model

class DifferentialEvolutionWeightOptimizer(EvolutionaryWeightOptimizer):
    """
    Optimizador de pesos basado en Evolución Diferencial.
    Utiliza evolución diferencial para optimizar los pesos de la red neuronal.
    """
    def __init__(self, population_size: int = 30, generations: int = 100,
                 mutation_factor: float = 0.5, crossover_probability: float = 0.7,
                 scaling_factor: float = 0.8, config=None):
        super().__init__(config)
        self.population_size = self.config.get('population_size', population_size)
        self.generations = self.config.get('generations', generations)
        self.mutation_factor = self.config.get('mutation_factor', mutation_factor)
        self.crossover_probability = self.config.get('crossover_probability', crossover_probability)
        self.scaling_factor = self.config.get('scaling_factor', scaling_factor)
        self.population = []
        self.fitness_scores = []
        self.generation_history = []
        logger.info(f"DifferentialEvolutionWeightOptimizer inicializado: pop_size={self.population_size}, generations={self.generations}")

    def _create_individual(self, model: nn.Module) -> Dict:
        """
        Crea un individuo (conjunto de pesos) para la población.
        """
        individual = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Crear pesos aleatorios basados en la forma del parámetro
                individual[name] = torch.randn_like(param.data) * 0.1
        return individual

    def _initialize_population(self, model: nn.Module) -> None:
        """
        Inicializa la población con individuos aleatorios.
        """
        self.population = []
        for _ in range(self.population_size):
            individual = self._create_individual(model)
            self.population.append(individual)
        logger.info(f"Población inicializada con {len(self.population)} individuos.")

    def _evaluate_fitness(self, model: nn.Module, individual: Dict, data_loader=None) -> float:
        """
        Evalúa la aptitud (fitness) de un individuo.
        """
        # Guardar pesos originales
        original_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                original_weights[name] = param.data.clone()
        
        # Aplicar pesos del individuo
        for name, param in model.named_parameters():
            if param.requires_grad and name in individual:
                param.data = individual[name]
        
        # Evaluar modelo
        if data_loader is None:
            # Evaluación simplificada basada en normas de pesos
            total_norm = sum(torch.norm(param.data).item() for param in model.parameters() if param.requires_grad)
            fitness = 1.0 / (1.0 + total_norm / 1000.0)  # Normalizar
        else:
            model.eval()
            total_loss = 0.0
            with torch.no_grad():
                for inputs, targets in data_loader:
                    outputs = model(inputs)
                    loss = F.mse_loss(outputs, targets)
                    total_loss += loss.item()
            fitness = 1.0 / (1.0 + total_loss)  # Convertir pérdida a fitness
        
        # Restaurar pesos originales
        for name, param in model.named_parameters():
            if param.requires_grad and name in original_weights:
                param.data = original_weights[name]
        
        return fitness

    def _differential_mutation(self, target_idx: int) -> Dict:
        """
        Mutación diferencial.
        """
        # Seleccionar tres individuos diferentes al objetivo
        candidates = [i for i in range(self.population_size) if i != target_idx]
        a, b, c = random.sample(candidates, 3)
        
        # Crear vector mutado
        mutated_individual = {}
        for name in self.population[target_idx]:
            # Mutación diferencial: v = a + F * (b - c)
            difference = self.population[b][name] - self.population[c][name]
            mutated_individual[name] = self.population[a][name] + self.mutation_factor * difference
        
        return mutated_individual

    def _crossover(self, target: Dict, mutant: Dict) -> Dict:
        """
        Cruce entre individuo objetivo y mutante.
        """
        trial_individual = {}
        
        for name in target:
            if random.random() < self.crossover_probability:
                trial_individual[name] = mutant[name].clone()
            else:
                trial_individual[name] = target[name].clone()
        
        return trial_individual

    def _evolve_generation(self, model: nn.Module, data_loader=None) -> None:
        """
        Evoluciona una generación.
        """
        new_population = []
        new_fitness_scores = []
        
        # Evaluar fitness de la población actual
        self.fitness_scores = []
        for individual in self.population:
            fitness = self._evaluate_fitness(model, individual, data_loader)
            self.fitness_scores.append(fitness)
        
        # Guardar estadísticas de la generación
        generation_stats = {
            'generation': len(self.generation_history),
            'best_fitness': max(self.fitness_scores),
            'avg_fitness': np.mean(self.fitness_scores),
            'worst_fitness': min(self.fitness_scores)
        }
        self.generation_history.append(generation_stats)
        
        # Evolución diferencial
        for i in range(self.population_size):
            # Mutación diferencial
            mutant = self._differential_mutation(i)
            
            # Cruce
            trial = self._crossover(self.population[i], mutant)
            
            # Evaluar trial
            trial_fitness = self._evaluate_fitness(model, trial, data_loader)
            
            # Selección
            if trial_fitness > self.fitness_scores[i]:
                new_population.append(trial)
                new_fitness_scores.append(trial_fitness)
            else:
                new_population.append(self.population[i])
                new_fitness_scores.append(self.fitness_scores[i])
        
        self.population = new_population
        self.fitness_scores = new_fitness_scores
        
        logger.debug(f"Generación {len(self.generation_history)}: Mejor fitness = {generation_stats['best_fitness']:.4f}")

    def evolutionary_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización evolutiva de pesos con Evolución Diferencial.")
        
        # Inicializar población
        self._initialize_population(model)
        
        # Evolución
        for generation in range(self.generations):
            self._evolve_generation(model, data_loader)
            
            # Log de progreso
            if generation % 10 == 0:
                best_fitness = max(self.fitness_scores)
                avg_fitness = np.mean(self.fitness_scores)
                logger.info(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, Promedio = {avg_fitness:.4f}")
        
        # Aplicar el mejor individuo al modelo
        best_idx = np.argmax(self.fitness_scores)
        best_individual = self.population[best_idx]
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in best_individual:
                param.data = best_individual[name]
        
        logger.info(f"Optimización evolutiva completada. Mejor fitness final = {max(self.fitness_scores):.4f}")
        return model

class EvolutionaryWeightAnalyzer:
    """
    Analizador para evaluar el rendimiento de la optimización evolutiva de pesos.
    """
    def __init__(self):
        logger.info("EvolutionaryWeightAnalyzer inicializado.")

    def analyze_evolutionary_optimization(self, original_model: nn.Module, 
                                       optimized_model: nn.Module, 
                                       test_data_loader=None) -> Dict[str, float]:
        """
        Analiza el rendimiento de la optimización evolutiva.
        """
        analysis_results = {}
        
        # Evaluar rendimiento original
        original_performance = self._evaluate_model_performance(original_model, test_data_loader)
        
        # Evaluar rendimiento optimizado
        optimized_performance = self._evaluate_model_performance(optimized_model, test_data_loader)
        
        # Calcular mejora
        improvement = original_performance - optimized_performance
        improvement_percentage = (improvement / original_performance) * 100
        
        analysis_results['original_performance'] = original_performance
        analysis_results['optimized_performance'] = optimized_performance
        analysis_results['improvement'] = improvement
        analysis_results['improvement_percentage'] = improvement_percentage
        
        # Analizar características de optimización evolutiva
        analysis_results['evolutionary_fitness'] = self._analyze_evolutionary_fitness(optimized_model)
        analysis_results['population_diversity'] = self._analyze_population_diversity(optimized_model)
        
        logger.info(f"Análisis de optimización evolutiva: Mejora = {improvement_percentage:.2f}%")
        return analysis_results

    def _evaluate_model_performance(self, model: nn.Module, data_loader) -> float:
        """
        Evalúa el rendimiento del modelo.
        """
        if data_loader is None:
            return random.random()
        
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                total_loss += loss.item()
        
        return total_loss

    def _analyze_evolutionary_fitness(self, model: nn.Module) -> float:
        """
        Analiza la aptitud evolutiva del modelo.
        """
        # Simular aptitud evolutiva basándose en la estabilidad de los pesos
        weight_stability = 0.0
        for param in model.parameters():
            if param.requires_grad:
                weight_stability += torch.var(param.data).item()
        
        fitness = 1.0 / (1.0 + weight_stability / 1000.0)
        return fitness

    def _analyze_population_diversity(self, model: nn.Module) -> float:
        """
        Analiza la diversidad de población del modelo.
        """
        # Simular diversidad de población basándose en la variabilidad de los pesos
        weight_variability = 0.0
        for param in model.parameters():
            if param.requires_grad:
                weight_variability += torch.std(param.data).item()
        
        diversity = weight_variability / 1000.0  # Normalizar
        return diversity

def create_evolutionary_weight_optimizer(optimizer_type: str, **kwargs) -> EvolutionaryWeightOptimizer:
    """
    Factoría para crear diferentes tipos de optimizadores evolutivos de pesos.
    """
    if optimizer_type == "genetic_algorithm":
        return GeneticAlgorithmWeightOptimizer(**kwargs)
    elif optimizer_type == "differential_evolution":
        return DifferentialEvolutionWeightOptimizer(**kwargs)
    else:
        raise ValueError(f"Tipo de optimizador evolutivo no soportado: {optimizer_type}")

def evolutionary_optimize_model_weights(model: nn.Module, optimizer_type: str, 
                                      data_loader=None, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Función de conveniencia para aplicar optimización evolutiva a los pesos de un modelo.
    """
    optimizer = create_evolutionary_weight_optimizer(optimizer_type, **kwargs)
    original_model = copy.deepcopy(model)
    
    # Aplicar optimización evolutiva
    optimized_model = optimizer.evolutionary_optimize_weights(model, data_loader)
    
    # Analizar rendimiento
    analyzer = EvolutionaryWeightAnalyzer()
    analysis = analyzer.analyze_evolutionary_optimization(original_model, optimized_model, data_loader)
    
    return optimized_model, analysis

# Exportar clases y funciones principales
__all__ = [
    'EvolutionaryWeightOptimizer',
    'GeneticAlgorithmWeightOptimizer',
    'DifferentialEvolutionWeightOptimizer',
    'EvolutionaryWeightAnalyzer',
    'create_evolutionary_weight_optimizer',
    'evolutionary_optimize_model_weights'
]

logger.info("RFEN7_RN_2 - Optimización Evolutiva de Pesos cargada correctamente")
