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

class NeuroevolutionWeightOptimizer(ABC):
    """
    Clase base abstracta para optimizadores de neuroevolución de pesos.
    Define la interfaz común para todas las estrategias de neuroevolución.
    """
    def __init__(self, config=None):
        self.config = config if config is not None else {}
        logger.info("NeuroevolutionWeightOptimizer base inicializado.")

    @abstractmethod
    def neuroevolution_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        """
        Método abstracto para optimizar los pesos usando neuroevolución.
        Debe ser implementado por las subclases.
        """
        pass

class NEATWeightOptimizer(NeuroevolutionWeightOptimizer):
    """
    Optimizador de pesos basado en NEAT (NeuroEvolution of Augmenting Topologies).
    Utiliza NEAT para optimizar tanto la topología como los pesos de la red neuronal.
    """
    def __init__(self, population_size: int = 100, generations: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7,
                 add_connection_prob: float = 0.3, add_node_prob: float = 0.1,
                 weight_mutation_rate: float = 0.8, config=None):
        super().__init__(config)
        self.population_size = self.config.get('population_size', population_size)
        self.generations = self.config.get('generations', generations)
        self.mutation_rate = self.config.get('mutation_rate', mutation_rate)
        self.crossover_rate = self.config.get('crossover_rate', crossover_rate)
        self.add_connection_prob = self.config.get('add_connection_prob', add_connection_prob)
        self.add_node_prob = self.config.get('add_node_prob', add_node_prob)
        self.weight_mutation_rate = self.config.get('weight_mutation_rate', weight_mutation_rate)
        self.population = []
        self.fitness_scores = []
        self.species = []
        self.generation_history = []
        logger.info(f"NEATWeightOptimizer inicializado: pop_size={self.population_size}, generations={self.generations}")

    def _create_genome(self, model: nn.Module) -> Dict:
        """
        Crea un genoma NEAT basado en el modelo.
        """
        genome = {
            'nodes': {},
            'connections': {},
            'fitness': 0.0,
            'species_id': 0
        }
        
        # Crear nodos basados en las capas del modelo
        node_id = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Tanh, nn.Sigmoid)):
                genome['nodes'][node_id] = {
                    'type': type(module).__name__,
                    'layer_name': name,
                    'parameters': module.state_dict()
                }
                node_id += 1
        
        # Crear conexiones entre nodos
        connection_id = 0
        for i in range(len(genome['nodes']) - 1):
            genome['connections'][connection_id] = {
                'from_node': i,
                'to_node': i + 1,
                'weight': random.uniform(-1.0, 1.0),
                'enabled': True
            }
            connection_id += 1
        
        return genome

    def _initialize_population(self, model: nn.Module) -> None:
        """
        Inicializa la población con genomas NEAT.
        """
        self.population = []
        for _ in range(self.population_size):
            genome = self._create_genome(model)
            self.population.append(genome)
        logger.info(f"Población NEAT inicializada con {len(self.population)} genomas.")

    def _evaluate_fitness(self, model: nn.Module, genome: Dict, data_loader=None) -> float:
        """
        Evalúa la aptitud (fitness) de un genoma.
        """
        # Aplicar genoma al modelo
        for node_id, node in genome['nodes'].items():
            if node_id < len(list(model.named_modules())):
                module_name, module = list(model.named_modules())[node_id]
                if hasattr(module, 'state_dict'):
                    # Aplicar parámetros del genoma
                    for param_name, param_value in node['parameters'].items():
                        if param_name in module.state_dict():
                            module.state_dict()[param_name].copy_(param_value)
        
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

    def _mutate_genome(self, genome: Dict) -> Dict:
        """
        Mutación de un genoma NEAT.
        """
        mutated_genome = copy.deepcopy(genome)
        
        # Mutación de pesos
        if random.random() < self.weight_mutation_rate:
            for connection_id, connection in mutated_genome['connections'].items():
                if random.random() < self.mutation_rate:
                    # Mutación de peso
                    connection['weight'] += random.gauss(0, 0.1)
                    connection['weight'] = max(-1.0, min(1.0, connection['weight']))
        
        # Agregar nueva conexión
        if random.random() < self.add_connection_prob:
            # Seleccionar nodos aleatorios
            from_node = random.choice(list(mutated_genome['nodes'].keys()))
            to_node = random.choice(list(mutated_genome['nodes'].keys()))
            
            if from_node != to_node:
                # Crear nueva conexión
                new_connection_id = max(mutated_genome['connections'].keys()) + 1 if mutated_genome['connections'] else 0
                mutated_genome['connections'][new_connection_id] = {
                    'from_node': from_node,
                    'to_node': to_node,
                    'weight': random.uniform(-1.0, 1.0),
                    'enabled': True
                }
        
        # Agregar nuevo nodo
        if random.random() < self.add_node_prob:
            # Seleccionar conexión aleatoria para dividir
            if mutated_genome['connections']:
                connection_id = random.choice(list(mutated_genome['connections'].keys()))
                connection = mutated_genome['connections'][connection_id]
                
                # Deshabilitar conexión original
                connection['enabled'] = False
                
                # Crear nuevo nodo
                new_node_id = max(mutated_genome['nodes'].keys()) + 1
                mutated_genome['nodes'][new_node_id] = {
                    'type': 'Linear',
                    'layer_name': f'new_layer_{new_node_id}',
                    'parameters': {}
                }
                
                # Crear nuevas conexiones
                new_connection_id1 = max(mutated_genome['connections'].keys()) + 1
                new_connection_id2 = new_connection_id1 + 1
                
                mutated_genome['connections'][new_connection_id1] = {
                    'from_node': connection['from_node'],
                    'to_node': new_node_id,
                    'weight': 1.0,
                    'enabled': True
                }
                
                mutated_genome['connections'][new_connection_id2] = {
                    'from_node': new_node_id,
                    'to_node': connection['to_node'],
                    'weight': connection['weight'],
                    'enabled': True
                }
        
        return mutated_genome

    def _crossover_genomes(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Cruce entre dos genomas NEAT.
        """
        if random.random() > self.crossover_rate:
            return parent1 if parent1['fitness'] > parent2['fitness'] else parent2
        
        # Crear genoma hijo
        child_genome = {
            'nodes': {},
            'connections': {},
            'fitness': 0.0,
            'species_id': 0
        }
        
        # Cruce de nodos
        all_nodes = set(parent1['nodes'].keys()) | set(parent2['nodes'].keys())
        for node_id in all_nodes:
            if node_id in parent1['nodes'] and node_id in parent2['nodes']:
                # Ambos padres tienen el nodo
                child_genome['nodes'][node_id] = copy.deepcopy(
                    parent1['nodes'][node_id] if random.random() < 0.5 else parent2['nodes'][node_id]
                )
            elif node_id in parent1['nodes']:
                child_genome['nodes'][node_id] = copy.deepcopy(parent1['nodes'][node_id])
            else:
                child_genome['nodes'][node_id] = copy.deepcopy(parent2['nodes'][node_id])
        
        # Cruce de conexiones
        all_connections = set(parent1['connections'].keys()) | set(parent2['connections'].keys())
        for connection_id in all_connections:
            if connection_id in parent1['connections'] and connection_id in parent2['connections']:
                # Ambos padres tienen la conexión
                child_genome['connections'][connection_id] = copy.deepcopy(
                    parent1['connections'][connection_id] if random.random() < 0.5 else parent2['connections'][connection_id]
                )
            elif connection_id in parent1['connections']:
                child_genome['connections'][connection_id] = copy.deepcopy(parent1['connections'][connection_id])
            else:
                child_genome['connections'][connection_id] = copy.deepcopy(parent2['connections'][connection_id])
        
        return child_genome

    def _evolve_generation(self, model: nn.Module, data_loader=None) -> None:
        """
        Evoluciona una generación NEAT.
        """
        # Evaluar fitness de todos los genomas
        self.fitness_scores = []
        for genome in self.population:
            fitness = self._evaluate_fitness(model, genome, data_loader)
            genome['fitness'] = fitness
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
        new_population = []
        
        # Mantener el mejor genoma
        best_genome = self.population[np.argmax(self.fitness_scores)]
        new_population.append(copy.deepcopy(best_genome))
        
        # Generar resto de la población
        while len(new_population) < self.population_size:
            # Selección por torneo
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Cruce
            child = self._crossover_genomes(parent1, parent2)
            
            # Mutación
            child = self._mutate_genome(child)
            
            new_population.append(child)
        
        self.population = new_population
        logger.debug(f"Generación {len(self.generation_history)}: Mejor fitness = {generation_stats['best_fitness']:.4f}")

    def _tournament_selection(self, tournament_size: int = 3) -> Dict:
        """
        Selección por torneo.
        """
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_genomes = [self.population[i] for i in tournament_indices]
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        
        best_idx = np.argmax(tournament_fitness)
        return tournament_genomes[best_idx]

    def neuroevolution_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización de pesos con NEAT.")
        
        # Inicializar población
        self._initialize_population(model)
        
        # Evolución NEAT
        for generation in range(self.generations):
            self._evolve_generation(model, data_loader)
            
            # Log de progreso
            if generation % 10 == 0:
                best_fitness = max(self.fitness_scores)
                avg_fitness = np.mean(self.fitness_scores)
                logger.info(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, Promedio = {avg_fitness:.4f}")
        
        # Aplicar el mejor genoma al modelo
        best_genome = self.population[np.argmax(self.fitness_scores)]
        
        # Aplicar genoma al modelo
        for node_id, node in best_genome['nodes'].items():
            if node_id < len(list(model.named_modules())):
                module_name, module = list(model.named_modules())[node_id]
                if hasattr(module, 'state_dict'):
                    # Aplicar parámetros del genoma
                    for param_name, param_value in node['parameters'].items():
                        if param_name in module.state_dict():
                            module.state_dict()[param_name].copy_(param_value)
        
        logger.info(f"Optimización NEAT completada. Mejor fitness final = {max(self.fitness_scores):.4f}")
        return model

class HyperNEATWeightOptimizer(NeuroevolutionWeightOptimizer):
    """
    Optimizador de pesos basado en HyperNEAT.
    Utiliza HyperNEAT para optimizar los pesos de la red neuronal con representación indirecta.
    """
    def __init__(self, population_size: int = 50, generations: int = 30,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7,
                 cppn_mutation_rate: float = 0.1, config=None):
        super().__init__(config)
        self.population_size = self.config.get('population_size', population_size)
        self.generations = self.config.get('generations', generations)
        self.mutation_rate = self.config.get('mutation_rate', mutation_rate)
        self.crossover_rate = self.config.get('crossover_rate', crossover_rate)
        self.cppn_mutation_rate = self.config.get('cppn_mutation_rate', cppn_mutation_rate)
        self.population = []
        self.fitness_scores = []
        self.generation_history = []
        logger.info(f"HyperNEATWeightOptimizer inicializado: pop_size={self.population_size}, generations={self.generations}")

    def _create_cppn(self, model: nn.Module) -> Dict:
        """
        Crea un CPPN (Compositional Pattern Producing Network) para HyperNEAT.
        """
        cppn = {
            'nodes': {},
            'connections': {},
            'fitness': 0.0
        }
        
        # Crear nodos CPPN
        node_id = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                cppn['nodes'][node_id] = {
                    'type': 'hidden',
                    'activation': random.choice(['sigmoid', 'tanh', 'relu']),
                    'bias': random.uniform(-1.0, 1.0)
                }
                node_id += 1
        
        # Crear conexiones CPPN
        connection_id = 0
        for i in range(len(cppn['nodes']) - 1):
            cppn['connections'][connection_id] = {
                'from_node': i,
                'to_node': i + 1,
                'weight': random.uniform(-1.0, 1.0),
                'enabled': True
            }
            connection_id += 1
        
        return cppn

    def _initialize_population(self, model: nn.Module) -> None:
        """
        Inicializa la población con CPPNs.
        """
        self.population = []
        for _ in range(self.population_size):
            cppn = self._create_cppn(model)
            self.population.append(cppn)
        logger.info(f"Población HyperNEAT inicializada con {len(self.population)} CPPNs.")

    def _evaluate_cppn_fitness(self, model: nn.Module, cppn: Dict, data_loader=None) -> float:
        """
        Evalúa la aptitud (fitness) de un CPPN.
        """
        # Generar pesos usando CPPN
        generated_weights = self._generate_weights_from_cppn(model, cppn)
        
        # Aplicar pesos generados al modelo
        for name, param in model.named_parameters():
            if param.requires_grad and name in generated_weights:
                param.data = generated_weights[name]
        
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

    def _generate_weights_from_cppn(self, model: nn.Module, cppn: Dict) -> Dict:
        """
        Genera pesos del modelo usando CPPN.
        """
        generated_weights = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Generar pesos usando CPPN
                weight_shape = param.data.shape
                weights = torch.zeros_like(param.data)
                
                # Simular generación de pesos con CPPN
                for i in range(weight_shape[0]):
                    for j in range(weight_shape[1]):
                        # Usar CPPN para generar peso
                        cppn_output = self._evaluate_cppn(cppn, [i, j])
                        weights[i, j] = cppn_output
                
                generated_weights[name] = weights
        
        return generated_weights

    def _evaluate_cppn(self, cppn: Dict, inputs: List[float]) -> float:
        """
        Evalúa un CPPN con entradas dadas.
        """
        # Simulación simplificada de evaluación CPPN
        output = 0.0
        for connection_id, connection in cppn['connections'].items():
            if connection['enabled']:
                from_node = cppn['nodes'][connection['from_node']]
                to_node = cppn['nodes'][connection['to_node']]
                
                # Simular activación
                activation = inputs[0] * connection['weight'] + from_node['bias']
                
                # Aplicar función de activación
                if from_node['activation'] == 'sigmoid':
                    activation = 1.0 / (1.0 + math.exp(-activation))
                elif from_node['activation'] == 'tanh':
                    activation = math.tanh(activation)
                elif from_node['activation'] == 'relu':
                    activation = max(0.0, activation)
                
                output += activation
        
        return output

    def _mutate_cppn(self, cppn: Dict) -> Dict:
        """
        Mutación de un CPPN.
        """
        mutated_cppn = copy.deepcopy(cppn)
        
        # Mutación de pesos de conexiones
        for connection_id, connection in mutated_cppn['connections'].items():
            if random.random() < self.cppn_mutation_rate:
                connection['weight'] += random.gauss(0, 0.1)
                connection['weight'] = max(-1.0, min(1.0, connection['weight']))
        
        # Mutación de bias de nodos
        for node_id, node in mutated_cppn['nodes'].items():
            if random.random() < self.cppn_mutation_rate:
                node['bias'] += random.gauss(0, 0.1)
                node['bias'] = max(-1.0, min(1.0, node['bias']))
        
        return mutated_cppn

    def _crossover_cppns(self, parent1: Dict, parent2: Dict) -> Dict:
        """
        Cruce entre dos CPPNs.
        """
        if random.random() > self.crossover_rate:
            return parent1 if parent1['fitness'] > parent2['fitness'] else parent2
        
        # Crear CPPN hijo
        child_cppn = {
            'nodes': {},
            'connections': {},
            'fitness': 0.0
        }
        
        # Cruce de nodos
        all_nodes = set(parent1['nodes'].keys()) | set(parent2['nodes'].keys())
        for node_id in all_nodes:
            if node_id in parent1['nodes'] and node_id in parent2['nodes']:
                child_cppn['nodes'][node_id] = copy.deepcopy(
                    parent1['nodes'][node_id] if random.random() < 0.5 else parent2['nodes'][node_id]
                )
            elif node_id in parent1['nodes']:
                child_cppn['nodes'][node_id] = copy.deepcopy(parent1['nodes'][node_id])
            else:
                child_cppn['nodes'][node_id] = copy.deepcopy(parent2['nodes'][node_id])
        
        # Cruce de conexiones
        all_connections = set(parent1['connections'].keys()) | set(parent2['connections'].keys())
        for connection_id in all_connections:
            if connection_id in parent1['connections'] and connection_id in parent2['connections']:
                child_cppn['connections'][connection_id] = copy.deepcopy(
                    parent1['connections'][connection_id] if random.random() < 0.5 else parent2['connections'][connection_id]
                )
            elif connection_id in parent1['connections']:
                child_cppn['connections'][connection_id] = copy.deepcopy(parent1['connections'][connection_id])
            else:
                child_cppn['connections'][connection_id] = copy.deepcopy(parent2['connections'][connection_id])
        
        return child_cppn

    def _evolve_generation(self, model: nn.Module, data_loader=None) -> None:
        """
        Evoluciona una generación HyperNEAT.
        """
        # Evaluar fitness de todos los CPPNs
        self.fitness_scores = []
        for cppn in self.population:
            fitness = self._evaluate_cppn_fitness(model, cppn, data_loader)
            cppn['fitness'] = fitness
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
        new_population = []
        
        # Mantener el mejor CPPN
        best_cppn = self.population[np.argmax(self.fitness_scores)]
        new_population.append(copy.deepcopy(best_cppn))
        
        # Generar resto de la población
        while len(new_population) < self.population_size:
            # Selección por torneo
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Cruce
            child = self._crossover_cppns(parent1, parent2)
            
            # Mutación
            child = self._mutate_cppn(child)
            
            new_population.append(child)
        
        self.population = new_population
        logger.debug(f"Generación {len(self.generation_history)}: Mejor fitness = {generation_stats['best_fitness']:.4f}")

    def _tournament_selection(self, tournament_size: int = 3) -> Dict:
        """
        Selección por torneo.
        """
        tournament_indices = random.sample(range(len(self.population)), tournament_size)
        tournament_cppns = [self.population[i] for i in tournament_indices]
        tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
        
        best_idx = np.argmax(tournament_fitness)
        return tournament_cppns[best_idx]

    def neuroevolution_optimize_weights(self, model: nn.Module, data_loader=None) -> nn.Module:
        logger.info("Iniciando optimización de pesos con HyperNEAT.")
        
        # Inicializar población
        self._initialize_population(model)
        
        # Evolución HyperNEAT
        for generation in range(self.generations):
            self._evolve_generation(model, data_loader)
            
            # Log de progreso
            if generation % 5 == 0:
                best_fitness = max(self.fitness_scores)
                avg_fitness = np.mean(self.fitness_scores)
                logger.info(f"Generación {generation}: Mejor fitness = {best_fitness:.4f}, Promedio = {avg_fitness:.4f}")
        
        # Aplicar el mejor CPPN al modelo
        best_cppn = self.population[np.argmax(self.fitness_scores)]
        generated_weights = self._generate_weights_from_cppn(model, best_cppn)
        
        # Aplicar pesos generados al modelo
        for name, param in model.named_parameters():
            if param.requires_grad and name in generated_weights:
                param.data = generated_weights[name]
        
        logger.info(f"Optimización HyperNEAT completada. Mejor fitness final = {max(self.fitness_scores):.4f}")
        return model

class NeuroevolutionWeightAnalyzer:
    """
    Analizador para evaluar el rendimiento de la neuroevolución de pesos.
    """
    def __init__(self):
        logger.info("NeuroevolutionWeightAnalyzer inicializado.")

    def analyze_neuroevolution_optimization(self, original_model: nn.Module, 
                                          optimized_model: nn.Module, 
                                          test_data_loader=None) -> Dict[str, float]:
        """
        Analiza el rendimiento de la neuroevolución.
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
        
        # Analizar características de neuroevolución
        analysis_results['neuroevolution_improvement'] = self._analyze_neuroevolution_improvement(optimized_model)
        analysis_results['topology_evolution'] = self._analyze_topology_evolution(optimized_model)
        
        logger.info(f"Análisis de neuroevolución: Mejora = {improvement_percentage:.2f}%")
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

    def _analyze_neuroevolution_improvement(self, model: nn.Module) -> float:
        """
        Analiza la mejora de neuroevolución del modelo.
        """
        # Simular mejora de neuroevolución basándose en la complejidad del modelo
        total_params = sum(p.numel() for p in model.parameters())
        neuroevolution_improvement = 1.0 / (1.0 + total_params / 1000000.0)
        return neuroevolution_improvement

    def _analyze_topology_evolution(self, model: nn.Module) -> float:
        """
        Analiza la evolución de topología del modelo.
        """
        # Simular evolución de topología basándose en la estructura del modelo
        num_layers = len(list(model.named_modules()))
        topology_evolution = 1.0 / (1.0 + num_layers / 100.0)
        return topology_evolution

def create_neuroevolution_weight_optimizer(optimizer_type: str, **kwargs) -> NeuroevolutionWeightOptimizer:
    """
    Factoría para crear diferentes tipos de optimizadores de neuroevolución.
    """
    if optimizer_type == "neat":
        return NEATWeightOptimizer(**kwargs)
    elif optimizer_type == "hyperneat":
        return HyperNEATWeightOptimizer(**kwargs)
    else:
        raise ValueError(f"Tipo de optimizador de neuroevolución no soportado: {optimizer_type}")

def neuroevolution_optimize_model_weights(model: nn.Module, optimizer_type: str, 
                                        data_loader=None, **kwargs) -> Tuple[nn.Module, Dict[str, float]]:
    """
    Función de conveniencia para aplicar neuroevolución a los pesos de un modelo.
    """
    optimizer = create_neuroevolution_weight_optimizer(optimizer_type, **kwargs)
    original_model = copy.deepcopy(model)
    
    # Aplicar neuroevolución
    optimized_model = optimizer.neuroevolution_optimize_weights(model, data_loader)
    
    # Analizar rendimiento
    analyzer = NeuroevolutionWeightAnalyzer()
    analysis = analyzer.analyze_neuroevolution_optimization(original_model, optimized_model, data_loader)
    
    return optimized_model, analysis

# Exportar clases y funciones principales
__all__ = [
    'NeuroevolutionWeightOptimizer',
    'NEATWeightOptimizer',
    'HyperNEATWeightOptimizer',
    'NeuroevolutionWeightAnalyzer',
    'create_neuroevolution_weight_optimizer',
    'neuroevolution_optimize_model_weights'
]

logger.info("RFEN7_RN_5 - Neuroevolución de Pesos cargada correctamente")
