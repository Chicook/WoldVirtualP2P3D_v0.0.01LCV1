"""
RF_RFENRN1_4_5.py - Gestor de Meta-Optimización y AutoML
========================================================

Implementa técnicas de meta-aprendizaje y AutoML para optimización automática
de hiperparámetros en redes neuronales de aprendizaje por refuerzo.

Características:
- Meta-aprendizaje para hiperparámetros
- AutoML con búsqueda bayesiana
- Optimización de arquitectura neuronal (NAS)
- Transferencia de conocimiento entre tareas
- Búsqueda evolutiva de hiperparámetros
- Multi-task learning para optimización
- Few-shot learning para configuración rápida
- Neural Architecture Search (NAS)

Autor: LucIA Development Team
Versión: 4.5.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
import random
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_5')


@dataclass
class MetaOptimizationConfig:
    """Configuración para meta-optimización"""
    # Meta-aprendizaje
    meta_learning_rate: float = 1e-3
    meta_batch_size: int = 32
    meta_epochs: int = 100
    meta_inner_steps: int = 5

    # Búsqueda bayesiana
    bayesian_n_iter: int = 100
    bayesian_n_initial: int = 10
    bayesian_acquisition: str = 'EI'  # 'EI', 'PI', 'UCB'

    # NAS
    nas_search_space: Dict = field(default_factory=dict)
    nas_population_size: int = 50
    nas_generations: int = 100
    nas_mutation_rate: float = 0.1

    # Transferencia
    transfer_enabled: bool = True
    transfer_weight: float = 0.1

    # Few-shot learning
    few_shot_support_size: int = 5
    few_shot_query_size: int = 15


class MetaLearner:
    """Meta-aprendizaje para optimización de hiperparámetros"""

    def __init__(self, config: MetaOptimizationConfig):
        self.config = config
        self.meta_network = self._build_meta_network()
        self.meta_optimizer = torch.optim.Adam(self.meta_network.parameters(), lr=config.meta_learning_rate)
        self.task_history = []

    def _build_meta_network(self):
        """Construye la red meta-aprendizaje"""
        return nn.Sequential(
            nn.Linear(10, 64),  # Entrada: características de la tarea
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5)    # Salida: hiperparámetros optimizados
        )

    def meta_train(self, tasks: List[Dict]) -> Dict[str, float]:
        """Entrenamiento meta-aprendizaje"""
        meta_losses = []

        for task in tasks:
            # Forward pass meta
            task_features = self._extract_task_features(task)
            predicted_params = self.meta_network(task_features)

            # Inner loop: entrenar con parámetros predichos
            inner_loss = self._inner_loop(task, predicted_params)

            # Meta loss
            meta_loss = inner_loss
            meta_losses.append(meta_loss)

        # Meta update
        total_meta_loss = torch.stack(meta_losses).mean()
        self.meta_optimizer.zero_grad()
        total_meta_loss.backward()
        self.meta_optimizer.step()

        return {'meta_loss': total_meta_loss.item()}

    def _extract_task_features(self, task: Dict) -> torch.Tensor:
        """Extrae características de la tarea"""
        features = torch.tensor([
            task.get('input_dim', 0),
            task.get('output_dim', 0),
            task.get('num_samples', 0),
            task.get('complexity', 0),
            task.get('noise_level', 0),
            task.get('sparsity', 0),
            task.get('correlation', 0),
            task.get('nonlinearity', 0),
            task.get('dimensionality', 0),
            task.get('task_type', 0)
        ], dtype=torch.float32)

        return features.unsqueeze(0)

    def _inner_loop(self, task: Dict, params: torch.Tensor) -> torch.Tensor:
        """Loop interno de entrenamiento"""
        # Simular entrenamiento con parámetros dados
        # En implementación real, esto entrenaría el modelo
        simulated_loss = torch.randn(1, requires_grad=True)
        return simulated_loss


class BayesianOptimizer:
    """Optimización bayesiana para hiperparámetros"""

    def __init__(self, config: MetaOptimizationConfig):
        self.config = config
        self.gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=1.0, nu=2.5),
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=5
        )
        self.X_observed = []
        self.y_observed = []
        self.bounds = {
            'learning_rate': (1e-5, 1e-1),
            'weight_decay': (1e-6, 1e-2),
            'batch_size': (16, 256),
            'dropout_rate': (0.0, 0.5),
            'momentum': (0.0, 0.99)
        }

    def optimize(self, objective_function: Callable, n_iter: int = None) -> Dict[str, float]:
        """Optimización bayesiana"""
        n_iter = n_iter or self.config.bayesian_n_iter

        # Inicialización aleatoria
        for _ in range(self.config.bayesian_n_initial):
            params = self._random_sample()
            score = objective_function(params)
            self.X_observed.append(params)
            self.y_observed.append(score)

        # Optimización bayesiana
        for i in range(n_iter):
            # Entrenar GP
            X = np.array(self.X_observed)
            y = np.array(self.y_observed)
            self.gp.fit(X, y)

            # Encontrar siguiente punto
            next_params = self._acquisition_optimization()
            next_score = objective_function(next_params)

            # Actualizar observaciones
            self.X_observed.append(next_params)
            self.y_observed.append(next_score)

        # Retornar mejores parámetros
        best_idx = np.argmax(self.y_observed)
        return self._params_to_dict(self.X_observed[best_idx])

    def _random_sample(self) -> List[float]:
        """Muestra aleatoria de parámetros"""
        params = []
        for param_name, (low, high) in self.bounds.items():
            if param_name in ['learning_rate', 'weight_decay']:
                # Log scale para learning rate y weight decay
                value = np.exp(np.random.uniform(np.log(low), np.log(high)))
            else:
                value = np.random.uniform(low, high)
            params.append(value)
        return params

    def _acquisition_optimization(self) -> List[float]:
        """Optimización de función de adquisición"""
        best_params = None
        best_acquisition = -np.inf

        # Búsqueda aleatoria para optimización de adquisición
        for _ in range(1000):
            params = self._random_sample()
            acquisition_value = self._expected_improvement(params)

            if acquisition_value > best_acquisition:
                best_acquisition = acquisition_value
                best_params = params

        return best_params

    def _expected_improvement(self, params: List[float]) -> float:
        """Expected Improvement acquisition function"""
        X = np.array([params])
        mu, sigma = self.gp.predict(X, return_std=True)

        if len(self.y_observed) == 0:
            return 0.0

        best_y = max(self.y_observed)
        improvement = mu[0] - best_y

        if sigma[0] == 0:
            return 0.0

        z = improvement / sigma[0]
        ei = improvement * self._normal_cdf(z) + sigma[0] * self._normal_pdf(z)

        return ei

    def _normal_cdf(self, x: float) -> float:
        """CDF normal estándar"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _normal_pdf(self, x: float) -> float:
        """PDF normal estándar"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    def _params_to_dict(self, params: List[float]) -> Dict[str, float]:
        """Convierte lista de parámetros a diccionario"""
        param_dict = {}
        param_names = list(self.bounds.keys())
        for i, name in enumerate(param_names):
            param_dict[name] = params[i]
        return param_dict


class NeuralArchitectureSearch:
    """Neural Architecture Search (NAS)"""

    def __init__(self, config: MetaOptimizationConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_architecture = None
        self.best_score = -np.inf

    def search(self, search_space: Dict, objective_function: Callable) -> Dict[str, Any]:
        """Búsqueda de arquitectura neuronal"""
        # Inicializar población
        self._initialize_population(search_space)

        # Evolución
        for gen in range(self.config.nas_generations):
            # Evaluar población
            scores = []
            for arch in self.population:
                score = objective_function(arch)
                scores.append(score)

            # Seleccionar mejores
            sorted_indices = np.argsort(scores)[::-1]
            elite_size = self.config.nas_population_size // 4
            elite = [self.population[i] for i in sorted_indices[:elite_size]]

            # Actualizar mejor arquitectura
            if scores[sorted_indices[0]] > self.best_score:
                self.best_score = scores[sorted_indices[0]]
                self.best_architecture = self.population[sorted_indices[0]].copy()

            # Generar nueva población
            new_population = elite.copy()
            while len(new_population) < self.config.nas_population_size:
                # Selección de padres
                parent1 = self._tournament_selection(scores)
                parent2 = self._tournament_selection(scores)

                # Cruzamiento
                child = self._crossover(parent1, parent2)

                # Mutación
                child = self._mutate(child, search_space)

                new_population.append(child)

            self.population = new_population
            self.generation += 1

        return self.best_architecture

    def _initialize_population(self, search_space: Dict):
        """Inicializa población aleatoria"""
        self.population = []
        for _ in range(self.config.nas_population_size):
            arch = {}
            for param_name, param_values in search_space.items():
                arch[param_name] = random.choice(param_values)
            self.population.append(arch)

    def _tournament_selection(self, scores: List[float], tournament_size: int = 3) -> Dict[str, Any]:
        """Selección por torneo"""
        tournament_indices = random.sample(range(len(scores)), tournament_size)
        tournament_scores = [scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_scores)]
        return self.population[winner_idx]

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict[str, Any]:
        """Cruzamiento de arquitecturas"""
        child = {}
        for param_name in parent1.keys():
            if random.random() < 0.5:
                child[param_name] = parent1[param_name]
            else:
                child[param_name] = parent2[param_name]
        return child

    def _mutate(self, arch: Dict, search_space: Dict) -> Dict[str, Any]:
        """Mutación de arquitectura"""
        mutated = arch.copy()
        for param_name in arch.keys():
            if random.random() < self.config.nas_mutation_rate:
                mutated[param_name] = random.choice(search_space[param_name])
        return mutated


class MetaOptimizationManager:
    """
    Gestor de meta-optimización y AutoML para redes de refuerzo.

    Proporciona técnicas avanzadas de meta-aprendizaje y optimización
    automática de hiperparámetros con transferencia de conocimiento.
    """

    def __init__(self, config: Optional[MetaOptimizationConfig] = None):
        """
        Inicializa el gestor de meta-optimización.

        Args:
            config: Configuración de meta-optimización (opcional)
        """
        self.config = config or MetaOptimizationConfig()
        self.meta_learner = MetaLearner(self.config)
        self.bayesian_optimizer = BayesianOptimizer(self.config)
        self.nas = NeuralArchitectureSearch(self.config)

        self.optimization_history = []
        self.metrics = {
            'total_optimizations': 0,
            'best_score': -np.inf,
            'average_score': 0.0,
            'optimization_time': 0.0,
            'transfer_success_rate': 0.0
        }

        logger.info("MetaOptimizationManager inicializado")

    def optimize_hyperparameters(self, objective_function: Callable,
                                 method: str = 'bayesian') -> Dict[str, float]:
        """
        Optimiza hiperparámetros usando el método especificado.

        Args:
            objective_function: Función objetivo a optimizar
            method: Método de optimización ('bayesian', 'meta_learning', 'random')

        Returns:
            Mejores hiperparámetros encontrados
        """
        start_time = time.time()

        if method == 'bayesian':
            best_params = self.bayesian_optimizer.optimize(objective_function)
        elif method == 'meta_learning':
            best_params = self._meta_learning_optimization(objective_function)
        elif method == 'random':
            best_params = self._random_search(objective_function)
        else:
            raise ValueError(f"Método de optimización no soportado: {method}")

        optimization_time = time.time() - start_time

        # Actualizar métricas
        self.metrics['total_optimizations'] += 1
        self.metrics['optimization_time'] = optimization_time

        # Registrar en historial
        optimization_record = {
            'method': method,
            'best_params': best_params,
            'optimization_time': optimization_time,
            'timestamp': time.time()
        }
        self.optimization_history.append(optimization_record)

        logger.info(f"Optimización {method} completada en {optimization_time:.2f}s")

        return best_params

    def _meta_learning_optimization(self, objective_function: Callable) -> Dict[str, float]:
        """Optimización usando meta-aprendizaje"""
        # Simular tareas para meta-entrenamiento
        tasks = self._generate_synthetic_tasks()

        # Meta-entrenamiento
        meta_losses = []
        for epoch in range(self.config.meta_epochs):
            meta_loss = self.meta_learner.meta_train(tasks)
            meta_losses.append(meta_loss['meta_loss'])

        # Usar meta-red para predecir parámetros
        task_features = self._extract_task_features(objective_function)
        predicted_params = self.meta_learner.meta_network(task_features)

        # Convertir a diccionario
        param_names = ['learning_rate', 'weight_decay', 'batch_size', 'dropout_rate', 'momentum']
        best_params = {}
        for i, name in enumerate(param_names):
            best_params[name] = predicted_params[0][i].item()

        return best_params

    def _random_search(self, objective_function: Callable, n_trials: int = 100) -> Dict[str, float]:
        """Búsqueda aleatoria de hiperparámetros"""
        best_score = -np.inf
        best_params = None

        bounds = {
            'learning_rate': (1e-5, 1e-1),
            'weight_decay': (1e-6, 1e-2),
            'batch_size': (16, 256),
            'dropout_rate': (0.0, 0.5),
            'momentum': (0.0, 0.99)
        }

        for _ in range(n_trials):
            params = {}
            for param_name, (low, high) in bounds.items():
                if param_name in ['learning_rate', 'weight_decay']:
                    params[param_name] = np.exp(np.random.uniform(np.log(low), np.log(high)))
                else:
                    params[param_name] = np.random.uniform(low, high)

            score = objective_function(params)
            if score > best_score:
                best_score = score
                best_params = params

        return best_params

    def _generate_synthetic_tasks(self) -> List[Dict]:
        """Genera tareas sintéticas para meta-entrenamiento"""
        tasks = []
        for _ in range(self.config.meta_batch_size):
            task = {
                'input_dim': np.random.randint(10, 100),
                'output_dim': np.random.randint(1, 10),
                'num_samples': np.random.randint(100, 1000),
                'complexity': np.random.uniform(0, 1),
                'noise_level': np.random.uniform(0, 0.5),
                'sparsity': np.random.uniform(0, 1),
                'correlation': np.random.uniform(0, 1),
                'nonlinearity': np.random.uniform(0, 1),
                'dimensionality': np.random.uniform(0, 1),
                'task_type': np.random.randint(0, 3)
            }
            tasks.append(task)
        return tasks

    def _extract_task_features(self, objective_function: Callable) -> torch.Tensor:
        """Extrae características de la función objetivo"""
        # En implementación real, esto analizaría la función objetivo
        features = torch.tensor([50, 5, 500, 0.5, 0.1, 0.3, 0.7, 0.8, 0.6, 1], dtype=torch.float32)
        return features.unsqueeze(0)

    def search_architecture(self, search_space: Dict, objective_function: Callable) -> Dict[str, Any]:
        """
        Busca la mejor arquitectura neuronal.

        Args:
            search_space: Espacio de búsqueda de arquitecturas
            objective_function: Función objetivo

        Returns:
            Mejor arquitectura encontrada
        """
        start_time = time.time()

        best_architecture = self.nas.search(search_space, objective_function)

        optimization_time = time.time() - start_time

        # Actualizar métricas
        self.metrics['total_optimizations'] += 1
        self.metrics['optimization_time'] = optimization_time

        logger.info(f"Búsqueda de arquitectura completada en {optimization_time:.2f}s")

        return best_architecture

    def transfer_knowledge(self, source_tasks: List[Dict], target_task: Dict) -> Dict[str, float]:
        """
        Transfiere conocimiento de tareas fuente a tarea objetivo.

        Args:
            source_tasks: Lista de tareas fuente
            target_task: Tarea objetivo

        Returns:
            Parámetros transferidos
        """
        if not self.config.transfer_enabled:
            return self._random_search(lambda x: 0.0)

        # Entrenar meta-red con tareas fuente
        for _ in range(10):  # Pocas iteraciones para transferencia rápida
            self.meta_learner.meta_train(source_tasks)

        # Predecir parámetros para tarea objetivo
        task_features = self._extract_task_features_from_task(target_task)
        predicted_params = self.meta_learner.meta_network(task_features)

        # Convertir a diccionario
        param_names = ['learning_rate', 'weight_decay', 'batch_size', 'dropout_rate', 'momentum']
        transferred_params = {}
        for i, name in enumerate(param_names):
            transferred_params[name] = predicted_params[0][i].item()

        logger.info("Conocimiento transferido exitosamente")

        return transferred_params

    def _extract_task_features_from_task(self, task: Dict) -> torch.Tensor:
        """Extrae características de una tarea específica"""
        features = torch.tensor([
            task.get('input_dim', 50),
            task.get('output_dim', 5),
            task.get('num_samples', 500),
            task.get('complexity', 0.5),
            task.get('noise_level', 0.1),
            task.get('sparsity', 0.3),
            task.get('correlation', 0.7),
            task.get('nonlinearity', 0.8),
            task.get('dimensionality', 0.6),
            task.get('task_type', 1)
        ], dtype=torch.float32)

        return features.unsqueeze(0)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la meta-optimización"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'total_optimizations': self.metrics['total_optimizations'],
            'average_optimization_time': self.metrics['optimization_time'] / max(1, self.metrics['total_optimizations']),
            'history_length': len(self.optimization_history)
        }

    def save_state(self, path: str) -> None:
        """Guarda el estado de meta-optimización"""
        torch.save({
            'config': self.config,
            'metrics': self.metrics,
            'optimization_history': self.optimization_history,
            'meta_learner_state': self.meta_learner.meta_network.state_dict(),
            'bayesian_optimizer': {
                'X_observed': self.bayesian_optimizer.X_observed,
                'y_observed': self.bayesian_optimizer.y_observed
            }
        }, path)
        logger.info(f"Estado de meta-optimización guardado en {path}")

    def load_state(self, path: str) -> None:
        """Carga el estado de meta-optimización"""
        checkpoint = torch.load(path)

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.optimization_history = checkpoint.get('optimization_history', [])

        # Cargar estado de meta-red
        meta_learner_state = checkpoint.get('meta_learner_state')
        if meta_learner_state:
            self.meta_learner.meta_network.load_state_dict(meta_learner_state)

        # Cargar estado de optimizador bayesiano
        bayesian_state = checkpoint.get('bayesian_optimizer')
        if bayesian_state:
            self.bayesian_optimizer.X_observed = bayesian_state['X_observed']
            self.bayesian_optimizer.y_observed = bayesian_state['y_observed']

        logger.info(f"Estado de meta-optimización cargado desde {path}")

    def reset_state(self) -> None:
        """Reinicia el estado de meta-optimización"""
        self.metrics = {
            'total_optimizations': 0,
            'best_score': -np.inf,
            'average_score': 0.0,
            'optimization_time': 0.0,
            'transfer_success_rate': 0.0
        }

        self.optimization_history.clear()
        self.bayesian_optimizer.X_observed.clear()
        self.bayesian_optimizer.y_observed.clear()

        logger.info("Estado de meta-optimización reiniciado")
