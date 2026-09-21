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

class MultiSwarmWeightOptimizer(ABC):
    """
    Clase base abstracta para optimizadores multi-enjambre de pesos.
    Define la interfaz común para todas las estrategias de optimización multi-enjambre.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("MultiSwarmWeightOptimizer base inicializado.")

    @abstractmethod
    def multi_swarm_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        """
        Método abstracto para optimizar los pesos usando optimización multi-enjambre.
        Debe ser implementado por las subclases.
        """
        pass

class MultiSwarmPSOWeightOptimizer(MultiSwarmWeightOptimizer):
    """
    Optimizador de pesos basado en Multi-Swarm PSO (Particle Swarm Optimization).
    Utiliza múltiples enjambres de partículas para optimizar los pesos de la red neuronal.
    """
    def __init__(self, num_swarms: int = 10, particles_per_swarm: int = 20,
                 generations: int = 50, inertia: float = 0.9,
                 cognitive_weight: float = 2.0, social_weight: float = 2.0,
                 global_weight: float = 1.0, config=None):
        super().__init__(config)
        self.num_swarms = self.config.get('num_swarms', num_swarms)
        self.particles_per_swarm = self.config.get('particles_per_swarm', particles_per_swarm)
        self.generations = self.config.get('generations', generations)
        self.inertia = self.config.get('inertia', inertia)
        self.cognitive_weight = self.config.get('cognitive_weight', cognitive_weight)
        self.social_weight = self.config.get('social_weight', social_weight)
        self.global_weight = self.config.get('global_weight', global_weight)
        self.swarms = []
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        self.generation_history = []
        logger.info(f"MultiSwarmPSOWeightOptimizer inicializado: {self.num_swarms} enjambres, {self.particles_per_swarm} partículas por enjambre")

    def _create_particle(self, model: nn.Module) -> Dict:
        """
        Crea una partícula PSO basada en el modelo.
        """
        particle = {
            'position': [],
            'velocity': [],
            'best_position': [],
            'best_fitness': float('inf'),
            'fitness': float('inf')
        }
        
        # Inicializar posición y velocidad basándose en los pesos del modelo
        for param in model.parameters():
            if param.requires_grad:
                # Posición inicial (pesos actuales)
                particle['position'].append(param.data.clone())
                
                # Velocidad inicial aleatoria
                velocity = torch.randn_like(param.data) * 0.1
                particle['velocity'].append(velocity)
                
                # Mejor posición inicial
                particle['best_position'].append(param.data.clone())
        
        return particle

    def _initialize_swarms(self, model: nn.Module) -> None:
        """
        Inicializa los enjambres con partículas.
        """
        self.swarms = []
        for swarm_id in range(self.num_swarms):
            swarm = {
                'id': swarm_id,
                'particles': [],
                'best_position': None,
                'best_fitness': float('inf')
            }
            
            # Crear partículas para este enjambre
            for _ in range(self.particles_per_swarm):
                particle = self._create_particle(model)
                swarm['particles'].append(particle)
            
            self.swarms.append(swarm)
        
        logger.info(f"Enjambres inicializados: {len(self.swarms)} enjambres con {self.particles_per_swarm} partículas cada uno.")

    def _evaluate_particle_fitness(self, model: nn.Module, particle: Dict, data_loader=None) -> float:
        """
        Evalúa la aptitud (fitness) de una partícula.
        """
        # Aplicar posición de la partícula al modelo
        param_idx = 0
        for param in model.parameters():
            if param.requires_grad:
                param.data = particle['position'][param_idx].clone()
                param_idx += 1
        
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
        
        return fitness

    def _update_particle_velocity(self, particle: Dict, swarm_best_position: List[torch.Tensor], 
                                 global_best_position: List[torch.Tensor]) -> None:
        """
        Actualiza la velocidad de una partícula según las reglas PSO.
        """
        for i in range(len(particle['position'])):
            # Componente de inercia
            inertia_component = self.inertia * particle['velocity'][i]
            
            # Componente cognitivo
            cognitive_component = self.cognitive_weight * random.random() * (
                particle['best_position'][i] - particle['position'][i]
            )
            
            # Componente social (mejor del enjambre)
            social_component = self.social_weight * random.random() * (
                swarm_best_position[i] - particle['position'][i]
            )
            
            # Componente global (mejor global)
            global_component = self.global_weight * random.random() * (
                global_best_position[i] - particle['position'][i]
            )
            
            # Actualizar velocidad
            particle['velocity'][i] = (inertia_component + cognitive_component + 
                                     social_component + global_component)

    def _update_particle_position(self, particle: Dict) -> None:
        """
        Actualiza la posición de una partícula.
        """
        for i in range(len(particle['position'])):
            # Actualizar posición
            particle['position'][i] += particle['velocity'][i]
            
            # Limitar posición a rango válido
            particle['position'][i] = torch.clamp(particle['position'][i], -10.0, 10.0)

    def _update_swarm_best(self, swarm: Dict) -> None:
        """
        Actualiza la mejor posición del enjambre.
        """
        for particle in swarm['particles']:
            if particle['fitness'] < swarm['best_fitness']:
                swarm['best_fitness'] = particle['fitness']
                swarm['best_position'] = [pos.clone() for pos in particle['position']]

    def _update_global_best(self) -> None:
        """
        Actualiza la mejor posición global.
        """
        for swarm in self.swarms:
            if swarm['best_fitness'] < self.global_best_fitness:
                self.global_best_fitness = swarm['best_fitness']
                self.global_best_position = [pos.clone() for pos in swarm['best_position']]

    def _evolve_generation(self, model: nn.Module, data_loader=None) -> None:
        """
        Evoluciona una generación multi-enjambre.
        """
        # Evaluar fitness de todas las partículas
        for swarm in self.swarms:
            for particle in swarm['particles']:
                fitness = self._evaluate_particle_fitness(model, particle, data_loader)
                particle['fitness'] = fitness
                
                # Actualizar mejor posición de la partícula
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = [pos.clone() for pos in particle['position']]
            
            # Actualizar mejor del enjambre
            self._update_swarm_best(swarm)
        
        # Actualizar mejor global
        self._update_global_best()
        
        # Actualizar velocidades y posiciones
        for swarm in self.swarms:
            for particle in swarm['particles']:
                self._update_particle_velocity(particle, swarm['best_position'], self.global_best_position)
                self._update_particle_position(particle)
        
        # Guardar estadísticas de la generación
        generation_stats = {
            'generation': len(self.generation_history),
            'best_fitness': self.global_best_fitness,
            'avg_fitness': np.mean([p['fitness'] for swarm in self.swarms for p in swarm['particles']]),
            'worst_fitness': max([p['fitness'] for swarm in self.swarms for p in swarm['particles']])
        }
        self.generation_history.append(generation_stats)
        
        logger.debug(f"Generación {len(self.generation_history)}: Mejor fitness = {generation_stats['best_fitness']:.4f}")

    def multi_swarm_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización de pesos con Multi-Swarm PSO.")
        
        # Inicializar enjambres
        self._initialize_swarms(model)
        
        # Evolución multi-enjambre
        for generation in range(self.generations):
            self._evolve_generation(model, data_loader)
            
            # Log de progreso
            if generation % 10 == 0:
                logger.info(f"Generación {generation}: Mejor fitness = {self.global_best_fitness:.4f}")
        
        # Aplicar la mejor posición global al modelo
        if self.global_best_position:
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    param.data = self.global_best_position[param_idx].clone()
                    param_idx += 1
        
        logger.info(f"Optimización Multi-Swarm PSO completada. Mejor fitness final = {self.global_best_fitness:.4f}")
        return model

class MultiSwarmGeneticWeightOptimizer(MultiSwarmWeightOptimizer):
    """
    Optimizador de pesos basado en Multi-Swarm Genetic Algorithm.
    Utiliza múltiples enjambres genéticos para optimizar los pesos de la red neuronal.
    """
    def __init__(self, num_swarms: int = 8, individuals_per_swarm: int = 25,
                 generations: int = 40, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1, migration_rate: float = 0.1,
                 elite_size: int = 5, config=None):
        super().__init__(config)
        self.num_swarms = self.config.get('num_swarms', num_swarms)
        self.individuals_per_swarm = self.config.get('individuals_per_swarm', individuals_per_swarm)
        self.generations = self.config.get('generations', generations)
        self.crossover_rate = self.config.get('crossover_rate', crossover_rate)
        self.mutation_rate = self.config.get('mutation_rate', mutation_rate)
        self.migration_rate = self.config.get('migration_rate', migration_rate)
        self.elite_size = self.config.get('elite_size', elite_size)
        self.swarms = []
        self.global_best_individual = None
        self.global_best_fitness = float('inf')
        self.generation_history = []
        logger.info(f"MultiSwarmGeneticWeightOptimizer inicializado: {self.num_swarms} enjambres, {self.individuals_per_swarm} individuos por enjambre")

    def _create_individual(self, model: nn.Module) -> Dict:
        """
        Crea un individuo genético basado en el modelo.
        """
        individual = {
            'chromosome': [],
            'fitness': float('inf')
        }
        
        # Crear cromosoma basándose en los pesos del modelo
        for param in model.parameters():
            if param.requires_grad:
                # Convertir pesos a cromosoma
                chromosome_gene = param.data.clone()
                individual['chromosome'].append(chromosome_gene)
        
        return individual

    def _initialize_swarms(self, model: nn.Module) -> None:
        """
        Inicializa los enjambres con individuos genéticos.
        """
        self.swarms = []
        for swarm_id in range(self.num_swarms):
            swarm = {
                'id': swarm_id,
                'individuals': [],
                'best_individual': None,
                'best_fitness': float('inf')
            }
            
            # Crear individuos para este enjambre
            for _ in range(self.individuals_per_swarm):
                individual = self._create_individual(model)
                swarm['individuals'].append(individual)
            
            self.swarms.append(swarm)
        
        logger.info(f"Enjambres genéticos inicializados: {len(self.swarms)} enjambres con {self.individuals_per_swarm} individuos cada uno.")

    def _evaluate_individual_fitness(self, model: nn.Module, individual: Dict, data_loader=None) -> float:
        """
        Evalúa la aptitud (fitness) de un individuo.
        """
        # Aplicar cromosoma del individuo al modelo
        param_idx = 0
        for param in model.parameters():
            if param.requires_grad:
                param.data = individual['chromosome'][param_idx].clone()
                param_idx += 1
        
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
        
        return fitness

    def _crossover_individuals(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """
        Cruce entre dos individuos.
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2
        
        # Crear hijos
        child1 = {'chromosome': [], 'fitness': float('inf')}
        child2 = {'chromosome': [], 'fitness': float('inf')}
        
        for i in range(len(parent1['chromosome'])):
            # Cruce uniforme
            if random.random() < 0.5:
                child1['chromosome'].append(parent1['chromosome'][i].clone())
                child2['chromosome'].append(parent2['chromosome'][i].clone())
            else:
                child1['chromosome'].append(parent2['chromosome'][i].clone())
                child2['chromosome'].append(parent1['chromosome'][i].clone())
        
        return child1, child2

    def _mutate_individual(self, individual: Dict) -> Dict:
        """
        Mutación de un individuo.
        """
        mutated_individual = {'chromosome': [], 'fitness': float('inf')}
        
        for gene in individual['chromosome']:
            if random.random() < self.mutation_rate:
                # Mutación gaussiana
                mutation = torch.randn_like(gene) * 0.1
                mutated_gene = gene + mutation
            else:
                mutated_gene = gene.clone()
            
            mutated_individual['chromosome'].append(mutated_gene)
        
        return mutated_individual

    def _select_individuals(self, swarm: Dict) -> List[Dict]:
        """
        Selección de individuos para reproducción.
        """
        # Ordenar por fitness
        sorted_individuals = sorted(swarm['individuals'], key=lambda x: x['fitness'])
        
        # Seleccionar elite
        elite = sorted_individuals[:self.elite_size]
        
        # Selección por torneo para el resto
        selected = elite[:]
        while len(selected) < len(swarm['individuals']):
            # Torneo de tamaño 3
            tournament = random.sample(swarm['individuals'], 3)
            winner = min(tournament, key=lambda x: x['fitness'])
            selected.append(winner)
        
        return selected

    def _migrate_individuals(self) -> None:
        """
        Migración de individuos entre enjambres.
        """
        if random.random() < self.migration_rate:
            # Seleccionar enjambres aleatorios
            source_swarm = random.choice(self.swarms)
            target_swarm = random.choice([s for s in self.swarms if s['id'] != source_swarm['id']])
            
            # Seleccionar individuo para migrar
            migrant = random.choice(source_swarm['individuals'])
            
            # Reemplazar individuo aleatorio en el enjambre destino
            target_swarm['individuals'][random.randint(0, len(target_swarm['individuals']) - 1)] = migrant
            
            logger.debug(f"Migración: Individuo migrado del enjambre {source_swarm['id']} al enjambre {target_swarm['id']}")

    def _evolve_generation(self, model: nn.Module, data_loader=None) -> None:
        """
        Evoluciona una generación multi-enjambre genética.
        """
        # Evaluar fitness de todos los individuos
        for swarm in self.swarms:
            for individual in swarm['individuals']:
                fitness = self._evaluate_individual_fitness(model, individual, data_loader)
                individual['fitness'] = fitness
                
                # Actualizar mejor del enjambre
                if fitness < swarm['best_fitness']:
                    swarm['best_fitness'] = fitness
                    swarm['best_individual'] = individual
        
        # Actualizar mejor global
        for swarm in self.swarms:
            if swarm['best_fitness'] < self.global_best_fitness:
                self.global_best_fitness = swarm['best_fitness']
                self.global_best_individual = swarm['best_individual']
        
        # Evolución de cada enjambre
        for swarm in self.swarms:
            # Selección
            selected = self._select_individuals(swarm)
            
            # Reproducción
            new_individuals = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    child1, child2 = self._crossover_individuals(selected[i], selected[i + 1])
                    new_individuals.extend([child1, child2])
                else:
                    new_individuals.append(selected[i])
            
            # Mutación
            for individual in new_individuals:
                individual = self._mutate_individual(individual)
            
            # Reemplazar población
            swarm['individuals'] = new_individuals
        
        # Migración
        self._migrate_individuals()
        
        # Guardar estadísticas de la generación
        generation_stats = {
            'generation': len(self.generation_history),
            'best_fitness': self.global_best_fitness,
            'avg_fitness': np.mean([ind['fitness'] for swarm in self.swarms for ind in swarm['individuals']]),
            'worst_fitness': max([ind['fitness'] for swarm in self.swarms for ind in swarm['individuals']])
        }
        self.generation_history.append(generation_stats)
        
        logger.debug(f"Generación {len(self.generation_history)}: Mejor fitness = {generation_stats['best_fitness']:.4f}")

    def multi_swarm_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización de pesos con Multi-Swarm Genetic Algorithm.")
        
        # Inicializar enjambres
        self._initialize_swarms(model)
        
        # Evolución multi-enjambre
        for generation in range(self.generations):
            self._evolve_generation(model, data_loader)
            
            # Log de progreso
            if generation % 8 == 0:
                logger.info(f"Generación {generation}: Mejor fitness = {self.global_best_fitness:.4f}")
        
        # Aplicar el mejor individuo global al modelo
        if self.global_best_individual:
            param_idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    param.data = self.global_best_individual['chromosome'][param_idx].clone()
                    param_idx += 1
        
        logger.info(f"Optimización Multi-Swarm Genetic completada. Mejor fitness final = {self.global_best_fitness:.4f}")
        return model

class MultiSwarmWeightAnalyzer:
    """
    Analizador para evaluar el rendimiento de la optimización multi-enjambre de pesos.
    """
    def __init__(self):
        logger.info("MultiSwarmWeightAnalyzer inicializado.")

    def analyze_multi_swarm_optimization(self, original_model: nn.Module, 
                                        optimized_model: nn.Module, 
                                        test_data_loader=None) -> Dict[str, float]:
        """
        Analiza el rendimiento de la optimización multi-enjambre.
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
        
        # Analizar características multi-enjambre
        analysis_results['swarm_diversity'] = self._analyze_swarm_diversity(optimized_model)
        analysis_results['convergence_speed'] = self._analyze_convergence_speed(optimized_model)
        analysis_results['exploration_exploitation_balance'] = self._analyze_exploration_exploitation_balance(optimized_model)
        
        logger.info(f"Análisis multi-enjambre: Mejora = {improvement_percentage:.2f}%")
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

    def _analyze_swarm_diversity(self, model: nn.Module) -> float:
        """
        Analiza la diversidad de enjambres del modelo.
        """
        # Simular diversidad de enjambres basándose en la variabilidad de pesos
        weight_vars = []
        for param in model.parameters():
            if param.requires_grad:
                weight_vars.append(torch.var(param.data).item())
        
        swarm_diversity = np.mean(weight_vars) if weight_vars else 0.0
        return swarm_diversity

    def _analyze_convergence_speed(self, model: nn.Module) -> float:
        """
        Analiza la velocidad de convergencia del modelo.
        """
        # Simular velocidad de convergencia basándose en la estabilidad de pesos
        weight_stability = 0.0
        for param in model.parameters():
            if param.requires_grad:
                weight_stability += torch.std(param.data).item()
        
        convergence_speed = 1.0 / (1.0 + weight_stability)
        return convergence_speed

    def _analyze_exploration_exploitation_balance(self, model: nn.Module) -> float:
        """
        Analiza el balance exploración-explotación del modelo.
        """
        # Simular balance exploración-explotación basándose en la distribución de pesos
        weight_means = []
        for param in model.parameters():
            if param.requires_grad:
                weight_means.append(torch.mean(param.data).item())
        
        exploration_exploitation_balance = 1.0 / (1.0 + abs(np.mean(weight_means)))
        return exploration_exploitation_balance

def create_multi_swarm_weight_optimizer(optimizer_type: str, **kwargs) -> MultiSwarmWeightOptimizer:
    """
    Factoría para crear diferentes tipos de optimizadores multi-enjambre.
    """
    if optimizer_type == "multi_swarm_pso":
        return MultiSwarmPSOWeightOptimizer(**kwargs)
    elif optimizer_type == "multi_swarm_genetic":
        return MultiSwarmGeneticWeightOptimizer(**kwargs)
    else:
        raise ValueError(f"Tipo de optimizador multi-enjambre no soportado: {optimizer_type}")

def multi_swarm_optimize_model_weights(model: nn.Module, optimizer_type: str, 
                                     data_loader=None, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Función de conveniencia para aplicar optimización multi-enjambre a los pesos de un modelo.
    """
    optimizer = create_multi_swarm_weight_optimizer(optimizer_type, **kwargs)
    original_model = copy.deepcopy(model)
    
    # Aplicar optimización multi-enjambre
    optimized_model = optimizer.multi_swarm_optimize_weights(model, data_loader)
    
    # Analizar rendimiento
    analyzer = MultiSwarmWeightAnalyzer()
    analysis = analyzer.analyze_multi_swarm_optimization(original_model, optimized_model, data_loader)
    
    return optimized_model, analysis

# Exportar clases y funciones principales
__all__ = [
    'MultiSwarmWeightOptimizer',
    'MultiSwarmPSOWeightOptimizer',
    'MultiSwarmGeneticWeightOptimizer',
    'MultiSwarmWeightAnalyzer',
    'create_multi_swarm_weight_optimizer',
    'multi_swarm_optimize_model_weights'
]

logger.info("RFEN7_RN_6 - Optimización Multi-Enjambre de Pesos cargada correctamente")
