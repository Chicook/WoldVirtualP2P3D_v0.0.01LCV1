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
import queue
import concurrent.futures

# Configuración del logger
logger = logging.getLogger(__name__)

class PopulationBasedWeightOptimizer(ABC):
    """
    Clase base abstracta para optimizadores de pesos basados en población.
    Define la interfaz común para todas las estrategias de optimización basadas en población.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("PopulationBasedWeightOptimizer base inicializado.")

    @abstractmethod
    def population_based_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        """
        Método abstracto para optimizar los pesos usando optimización basada en población.
        Debe ser implementado por las subclases.
        """
        pass

class PopulationBasedTrainingOptimizer(PopulationBasedWeightOptimizer):
    """
    Optimizador de pesos basado en Population Based Training (PBT).
    Utiliza entrenamiento basado en población para optimizar los pesos de la red neuronal.
    """
    def __init__(self, population_size: int = 20, exploit_frequency: int = 10,
                 explore_fraction: float = 0.2, truncation_threshold: float = 0.2,
                 mutation_factor: float = 0.2, config=None):
        super().__init__(config)
        self.population_size = self.config.get('population_size', population_size)
        self.exploit_frequency = self.config.get('exploit_frequency', exploit_frequency)
        self.explore_fraction = self.config.get('explore_fraction', explore_fraction)
        self.truncation_threshold = self.config.get('truncation_threshold', truncation_threshold)
        self.mutation_factor = self.config.get('mutation_factor', mutation_factor)
        self.population = []
        self.fitness_scores = []
        self.hyperparameters = []
        self.training_history = []
        logger.info(f"PopulationBasedTrainingOptimizer inicializado: pop_size={self.population_size}, exploit_freq={self.exploit_frequency}")

    def _create_individual(self, model: nn.Module) -> Tuple[Dict, Dict]:
        """
        Crea un individuo (modelo + hiperparámetros) para la población.
        """
        # Crear copia del modelo
        model_copy = copy.deepcopy(model)
        
        # Crear hiperparámetros aleatorios
        hyperparameters = {
            'learning_rate': random.uniform(1e-5, 1e-2),
            'weight_decay': random.uniform(1e-6, 1e-3),
            'momentum': random.uniform(0.5, 0.99),
            'batch_size': random.choice([16, 32, 64, 128])
        }
        
        return model_copy, hyperparameters

    def _initialize_population(self, model: nn.Module) -> None:
        """
        Inicializa la población con individuos aleatorios.
        """
        self.population = []
        self.hyperparameters = []
        
        for _ in range(self.population_size):
            individual_model, hyperparams = self._create_individual(model)
            self.population.append(individual_model)
            self.hyperparameters.append(hyperparams)
        
        logger.info(f"Población inicializada con {len(self.population)} individuos.")

    def _evaluate_fitness(self, model: nn.Module, hyperparameters: Dict, data_loader=None) -> float:
        """
        Evalúa la aptitud (fitness) de un individuo.
        """
        # Crear optimizador con hiperparámetros
        optimizer = torch.optim.SGD(model.parameters(),
                                  lr=hyperparameters['learning_rate'],
                                  weight_decay=hyperparameters['weight_decay'],
                                  momentum=hyperparameters['momentum'])
        
        # Entrenamiento simplificado
        model.train()
        total_loss = 0.0
        epochs = 5  # Entrenamiento corto para evaluación
        
        for epoch in range(epochs):
            if data_loader is None:
                # Simulación de entrenamiento
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        param.data += torch.randn_like(param.data) * 0.01
                total_loss += random.random()
            else:
                for inputs, targets in data_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = F.mse_loss(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
        
        # Calcular fitness basado en pérdida
        fitness = 1.0 / (1.0 + total_loss / epochs)
        return fitness

    def _exploit_step(self) -> None:
        """
        Paso de explotación: reemplazar individuos de bajo rendimiento.
        """
        # Evaluar fitness de todos los individuos
        self.fitness_scores = []
        for i, (model, hyperparams) in enumerate(zip(self.population, self.hyperparameters)):
            fitness = self._evaluate_fitness(model, hyperparams)
            self.fitness_scores.append(fitness)
        
        # Ordenar por fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]  # Descendente
        
        # Reemplazar individuos de bajo rendimiento
        num_to_replace = int(self.population_size * self.truncation_threshold)
        
        for i in range(num_to_replace):
            # Seleccionar individuo de bajo rendimiento
            worst_idx = sorted_indices[-(i+1)]
            
            # Seleccionar individuo de alto rendimiento
            best_idx = sorted_indices[i % num_to_replace]
            
            # Copiar modelo y hiperparámetros del mejor individuo
            self.population[worst_idx] = copy.deepcopy(self.population[best_idx])
            self.hyperparameters[worst_idx] = copy.deepcopy(self.hyperparameters[best_idx])
            
            logger.debug(f"Individuo {worst_idx} reemplazado por individuo {best_idx}")

    def _explore_step(self) -> None:
        """
        Paso de exploración: mutar hiperparámetros.
        """
        num_to_explore = int(self.population_size * self.explore_fraction)
        explore_indices = random.sample(range(self.population_size), num_to_explore)
        
        for idx in explore_indices:
            # Mutar hiperparámetros
            hyperparams = self.hyperparameters[idx]
            
            for param_name in hyperparams:
                if random.random() < 0.5:  # 50% de probabilidad de mutar cada parámetro
                    if param_name == 'learning_rate':
                        hyperparams[param_name] *= random.uniform(0.8, 1.2)
                        hyperparams[param_name] = max(1e-6, min(1e-1, hyperparams[param_name]))
                    elif param_name == 'weight_decay':
                        hyperparams[param_name] *= random.uniform(0.8, 1.2)
                        hyperparams[param_name] = max(1e-7, min(1e-2, hyperparams[param_name]))
                    elif param_name == 'momentum':
                        hyperparams[param_name] += random.uniform(-0.1, 0.1)
                        hyperparams[param_name] = max(0.0, min(0.99, hyperparams[param_name]))
                    elif param_name == 'batch_size':
                        hyperparams[param_name] = random.choice([16, 32, 64, 128])
            
            logger.debug(f"Individuo {idx} mutado con nuevos hiperparámetros: {hyperparams}")

    def _train_population(self, data_loader=None) -> None:
        """
        Entrena la población de modelos.
        """
        for i, (model, hyperparams) in enumerate(zip(self.population, self.hyperparameters)):
            # Crear optimizador con hiperparámetros del individuo
            optimizer = torch.optim.SGD(model.parameters(),
                                      lr=hyperparams['learning_rate'],
                                      weight_decay=hyperparams['weight_decay'],
                                      momentum=hyperparams['momentum'])
            
            # Entrenamiento
            model.train()
            for epoch in range(3):  # Entrenamiento corto
                if data_loader is None:
                    # Simulación de entrenamiento
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            param.data += torch.randn_like(param.data) * 0.01
                else:
                    for inputs, targets in data_loader:
                        optimizer.zero_grad()
                        outputs = model(inputs)
                        loss = F.mse_loss(outputs, targets)
                        loss.backward()
                        optimizer.step()

    def population_based_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización basada en población con Population Based Training.")
        
        # Inicializar población
        self._initialize_population(model)
        
        # Entrenamiento basado en población
        for step in range(20):  # 20 pasos de PBT
            # Entrenar población
            self._train_population(data_loader)
            
            # Paso de explotación
            if step % self.exploit_frequency == 0:
                self._exploit_step()
            
            # Paso de exploración
            self._explore_step()
            
            # Guardar estadísticas
            best_fitness = max(self.fitness_scores) if self.fitness_scores else 0
            avg_fitness = np.mean(self.fitness_scores) if self.fitness_scores else 0
            
            step_stats = {
                'step': step,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness
            }
            self.training_history.append(step_stats)
            
            logger.debug(f"Paso {step}: Mejor fitness = {best_fitness:.4f}, Promedio = {avg_fitness:.4f}")
        
        # Seleccionar el mejor individuo
        best_idx = np.argmax(self.fitness_scores) if self.fitness_scores else 0
        best_model = self.population[best_idx]
        best_hyperparams = self.hyperparameters[best_idx]
        
        # Copiar pesos del mejor modelo al modelo original
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = best_model.state_dict()[name].clone()
        
        logger.info(f"Optimización basada en población completada. Mejor fitness final = {max(self.fitness_scores):.4f}")
        logger.info(f"Mejores hiperparámetros: {best_hyperparams}")
        return model

class MultiPopulationWeightOptimizer(PopulationBasedWeightOptimizer):
    """
    Optimizador de pesos basado en múltiples poblaciones.
    Utiliza múltiples poblaciones para explorar diferentes regiones del espacio de búsqueda.
    """
    def __init__(self, num_populations: int = 5, population_size: int = 10,
                 migration_frequency: int = 10, migration_rate: float = 0.1,
                 diversity_threshold: float = 0.5, config=None):
        super().__init__(config)
        self.num_populations = self.config.get('num_populations', num_populations)
        self.population_size = self.config.get('population_size', population_size)
        self.migration_frequency = self.config.get('migration_frequency', migration_frequency)
        self.migration_rate = self.config.get('migration_rate', migration_rate)
        self.diversity_threshold = self.config.get('diversity_threshold', diversity_threshold)
        self.populations = []
        self.population_fitness = []
        self.migration_history = []
        logger.info(f"MultiPopulationWeightOptimizer inicializado: num_pops={self.num_populations}, pop_size={self.population_size}")

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

    def _initialize_populations(self, model: nn.Module) -> None:
        """
        Inicializa múltiples poblaciones con individuos aleatorios.
        """
        self.populations = []
        self.population_fitness = []
        
        for pop_idx in range(self.num_populations):
            population = []
            fitness_scores = []
            
            for _ in range(self.population_size):
                individual = self._create_individual(model)
                population.append(individual)
                
                # Evaluar fitness inicial
                fitness = self._evaluate_fitness(model, individual)
                fitness_scores.append(fitness)
            
            self.populations.append(population)
            self.population_fitness.append(fitness_scores)
        
        logger.info(f"Múltiples poblaciones inicializadas: {len(self.populations)} poblaciones con {self.population_size} individuos cada una.")

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

    def _calculate_population_diversity(self, population: List[Dict]) -> float:
        """
        Calcula la diversidad de una población.
        """
        if len(population) < 2:
            return 0.0
        
        # Calcular distancia promedio entre individuos
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = 0.0
                for name in population[i]:
                    if name in population[j]:
                        diff = population[i][name] - population[j][name]
                        distance += torch.norm(diff).item()
                total_distance += distance
                num_pairs += 1
        
        avg_distance = total_distance / num_pairs if num_pairs > 0 else 0.0
        return avg_distance

    def _migrate_individuals(self) -> None:
        """
        Migra individuos entre poblaciones.
        """
        migration_events = []
        
        for pop_idx in range(self.num_populations):
            # Seleccionar población de destino
            target_pop_idx = (pop_idx + 1) % self.num_populations
            
            # Calcular diversidad de ambas poblaciones
            source_diversity = self._calculate_population_diversity(self.populations[pop_idx])
            target_diversity = self._calculate_population_diversity(self.populations[target_pop_idx])
            
            # Migrar si la diversidad es baja
            if source_diversity < self.diversity_threshold:
                # Seleccionar mejor individuo de la población fuente
                best_idx = np.argmax(self.population_fitness[pop_idx])
                best_individual = self.populations[pop_idx][best_idx]
                
                # Reemplazar peor individuo de la población destino
                worst_idx = np.argmin(self.population_fitness[target_pop_idx])
                self.populations[target_pop_idx][worst_idx] = copy.deepcopy(best_individual)
                
                # Recalcular fitness
                self.population_fitness[target_pop_idx][worst_idx] = self.population_fitness[pop_idx][best_idx]
                
                migration_events.append({
                    'from_population': pop_idx,
                    'to_population': target_pop_idx,
                    'individual_idx': best_idx,
                    'fitness': self.population_fitness[pop_idx][best_idx]
                })
        
        # Guardar historial de migración
        migration_stats = {
            'step': len(self.migration_history),
            'migration_events': migration_events,
            'total_migrations': len(migration_events)
        }
        self.migration_history.append(migration_stats)
        
        logger.debug(f"Migración completada: {len(migration_events)} eventos de migración")

    def _evolve_populations(self, model: nn.Module, data_loader=None) -> None:
        """
        Evoluciona todas las poblaciones.
        """
        for pop_idx in range(self.num_populations):
            population = self.populations[pop_idx]
            fitness_scores = self.population_fitness[pop_idx]
            
            # Evolución simple: reemplazar peor individuo con mutación del mejor
            best_idx = np.argmax(fitness_scores)
            worst_idx = np.argmin(fitness_scores)
            
            # Crear mutación del mejor individuo
            best_individual = population[best_idx]
            mutated_individual = {}
            
            for name, weights in best_individual.items():
                # Mutación gaussiana
                mutation_noise = torch.randn_like(weights) * 0.05
                mutated_individual[name] = weights + mutation_noise
            
            # Reemplazar peor individuo
            population[worst_idx] = mutated_individual
            fitness_scores[worst_idx] = self._evaluate_fitness(model, mutated_individual, data_loader)
            
            self.population_fitness[pop_idx] = fitness_scores

    def population_based_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización basada en población con múltiples poblaciones.")
        
        # Inicializar múltiples poblaciones
        self._initialize_populations(model)
        
        # Optimización con múltiples poblaciones
        for step in range(30):  # 30 pasos de optimización
            # Evolucionar poblaciones
            self._evolve_populations(model, data_loader)
            
            # Migración entre poblaciones
            if step % self.migration_frequency == 0:
                self._migrate_individuals()
            
            # Log de progreso
            if step % 5 == 0:
                best_fitnesses = [max(fitness_scores) for fitness_scores in self.population_fitness]
                avg_best_fitness = np.mean(best_fitnesses)
                logger.info(f"Paso {step}: Mejor fitness promedio = {avg_best_fitness:.4f}")
        
        # Seleccionar el mejor individuo de todas las poblaciones
        best_fitness = -float('inf')
        best_individual = None
        
        for pop_idx, fitness_scores in enumerate(self.population_fitness):
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_individual = self.populations[pop_idx][best_idx]
        
        # Aplicar el mejor individuo al modelo
        if best_individual:
            for name, param in model.named_parameters():
                if param.requires_grad and name in best_individual:
                    param.data = best_individual[name]
        
        logger.info(f"Optimización basada en población completada. Mejor fitness final = {best_fitness:.4f}")
        return model

class PopulationBasedWeightAnalyzer:
    """
    Analizador para evaluar el rendimiento de la optimización basada en población.
    """
    def __init__(self):
        logger.info("PopulationBasedWeightAnalyzer inicializado.")

    def analyze_population_based_optimization(self, original_model: nn.Module, 
                                            optimized_model: nn.Module, 
                                            test_data_loader=None) -> Dict[str, float]:
        """
        Analiza el rendimiento de la optimización basada en población.
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
        
        # Analizar características de optimización basada en población
        analysis_results['population_diversity'] = self._analyze_population_diversity(optimized_model)
        analysis_results['optimization_efficiency'] = self._analyze_optimization_efficiency(optimized_model)
        
        logger.info(f"Análisis de optimización basada en población: Mejora = {improvement_percentage:.2f}%")
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

    def _analyze_optimization_efficiency(self, model: nn.Module) -> float:
        """
        Analiza la eficiencia de optimización del modelo.
        """
        # Simular eficiencia basándose en la magnitud de los pesos
        total_efficiency = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total_efficiency += torch.norm(param.data).item()
        
        return total_efficiency / 1000.0  # Normalizar

def create_population_based_weight_optimizer(optimizer_type: str, **kwargs) -> PopulationBasedWeightOptimizer:
    """
    Factoría para crear diferentes tipos de optimizadores basados en población.
    """
    if optimizer_type == "population_based_training":
        return PopulationBasedTrainingOptimizer(**kwargs)
    elif optimizer_type == "multi_population":
        return MultiPopulationWeightOptimizer(**kwargs)
    else:
        raise ValueError(f"Tipo de optimizador basado en población no soportado: {optimizer_type}")

def population_based_optimize_model_weights(model: nn.Module, optimizer_type: str, 
                                          data_loader=None, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Función de conveniencia para aplicar optimización basada en población a los pesos de un modelo.
    """
    optimizer = create_population_based_weight_optimizer(optimizer_type, **kwargs)
    original_model = copy.deepcopy(model)
    
    # Aplicar optimización basada en población
    optimized_model = optimizer.population_based_optimize_weights(model, data_loader)
    
    # Analizar rendimiento
    analyzer = PopulationBasedWeightAnalyzer()
    analysis = analyzer.analyze_population_based_optimization(original_model, optimized_model, data_loader)
    
    return optimized_model, analysis

# Exportar clases y funciones principales
__all__ = [
    'PopulationBasedWeightOptimizer',
    'PopulationBasedTrainingOptimizer',
    'MultiPopulationWeightOptimizer',
    'PopulationBasedWeightAnalyzer',
    'create_population_based_weight_optimizer',
    'population_based_optimize_model_weights'
]

logger.info("RFEN7_RN_3 - Optimización Basada en Población cargada correctamente")
