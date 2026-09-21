"""
RFEN3_RN_8 - Sistema de Ajuste Automático de Hiperparámetros
Implementación de técnicas avanzadas para optimización automática de hiperparámetros
Incluye: Bayesian Optimization, Grid Search, Random Search, y métodos evolutivos
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
import json
import itertools
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel

logger = logging.getLogger(__name__)


@dataclass
class HyperparameterSpace:
    """Espacio de búsqueda de hiperparámetros"""
    learning_rate: Tuple[float, float] = (1e-5, 1e-1)
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64, 128])
    weight_decay: Tuple[float, float] = (1e-6, 1e-2)
    dropout_rate: Tuple[float, float] = (0.0, 0.8)
    hidden_size: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    num_layers: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    optimizer: List[str] = field(default_factory=lambda: ['Adam', 'AdamW', 'SGD', 'RMSprop'])
    scheduler: List[str] = field(default_factory=lambda: ['cosine', 'step', 'plateau', 'exponential'])


@dataclass
class HyperparameterTuningConfig:
    """Configuración para ajuste de hiperparámetros"""
    method: str = "bayesian"  # bayesian, grid, random, evolutionary
    max_trials: int = 100
    max_evaluations: int = 50
    early_stopping_patience: int = 5
    cv_folds: int = 3
    objective_metric: str = "accuracy"  # accuracy, loss, f1_score, custom
    minimize_objective: bool = False
    parallel_trials: int = 1
    random_seed: int = 42
    budget_per_trial: int = 10  # épocas por trial
    acquisition_function: str = "ei"  # ei (expected improvement), ucb, poi


class BaseHyperparameterOptimizer(ABC):
    """Clase base abstracta para optimizadores de hiperparámetros"""

    def __init__(self, config: HyperparameterTuningConfig, space: HyperparameterSpace):
        self.config = config
        self.space = space
        self.trial_history = []
        self.best_params = None
        self.best_score = float('inf') if config.minimize_objective else float('-inf')

    @abstractmethod
    def suggest_parameters(self) -> Dict[str, Union[float, int, str]]:
        """Sugiere nuevos parámetros para evaluar"""
        pass

    @abstractmethod
    def update(self, params: Dict[str, Union[float, int, str]], score: float) -> None:
        """Actualiza el optimizador con los resultados de un trial"""
        pass

    def evaluate_parameters(self, params: Dict[str, Union[float, int, str]],
                            model_fn: Callable, data_loader: torch.utils.data.DataLoader) -> float:
        """Evalúa un conjunto de parámetros"""

        # Crear modelo con parámetros dados
        model = model_fn(params)

        # Configurar optimizador
        optimizer = self._create_optimizer(model, params)

        # Entrenar modelo
        score = self._train_and_evaluate(model, optimizer, data_loader, params)

        # Registrar trial
        self.trial_history.append({
            'params': params.copy(),
            'score': score,
            'timestamp': time.time()
        })

        # Actualizar mejor resultado
        if self._is_better_score(score):
            self.best_score = score
            self.best_params = params.copy()

        return score

    def _create_optimizer(self, model: nn.Module, params: Dict[str, Union[float, int, str]]) -> optim.Optimizer:
        """Crea optimizador basado en parámetros"""

        optimizer_name = params.get('optimizer', 'Adam')
        lr = params.get('learning_rate', 0.001)
        weight_decay = params.get('weight_decay', 0.0)

        if optimizer_name == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'SGD':
            momentum = params.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_name == 'RMSprop':
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    def _train_and_evaluate(self, model: nn.Module, optimizer: optim.Optimizer,
                            data_loader: torch.utils.data.DataLoader,
                            params: Dict[str, Union[float, int, str]]) -> float:
        """Entrena y evalúa el modelo"""

        epochs = self.config.budget_per_trial
        criterion = nn.CrossEntropyLoss()

        # Entrenar modelo
        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(data_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Evaluar modelo
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                output = model(data)
                predicted = torch.argmax(output, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        accuracy = correct / total

        # Retornar métrica según configuración
        if self.config.objective_metric == "accuracy":
            return accuracy if not self.config.minimize_objective else 1 - accuracy
        elif self.config.objective_metric == "loss":
            # Calcular pérdida de validación
            model.eval()
            total_loss = 0
            with torch.no_grad():
                for data, target in data_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    total_loss += loss.item()
            return total_loss / len(data_loader)
        else:
            return accuracy

    def _is_better_score(self, score: float) -> bool:
        """Determina si un score es mejor que el actual"""
        if self.config.minimize_objective:
            return score < self.best_score
        else:
            return score > self.best_score


class BayesianOptimizer(BaseHyperparameterOptimizer):
    """
    Optimizador bayesiano usando Gaussian Process
    """

    def __init__(self, config: HyperparameterTuningConfig, space: HyperparameterSpace):
        super().__init__(config, space)
        self.gp_model = None
        self.param_history = []
        self.score_history = []
        self.kernel = Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=1e-6)

    def suggest_parameters(self) -> Dict[str, Union[float, int, str]]:
        """Sugiere parámetros usando adquisición bayesiana"""

        if len(self.param_history) < 5:
            # Usar muestreo aleatorio para los primeros trials
            return self._random_sample()

        # Entrenar modelo GP
        self._fit_gp_model()

        # Optimizar función de adquisición
        best_params = self._optimize_acquisition()

        return best_params

    def update(self, params: Dict[str, Union[float, int, str]], score: float) -> None:
        """Actualiza el modelo GP con nuevos datos"""

        # Convertir parámetros a vector numérico
        param_vector = self._params_to_vector(params)

        self.param_history.append(param_vector)
        self.score_history.append(score)

    def _random_sample(self) -> Dict[str, Union[float, int, str]]:
        """Muestra parámetros aleatoriamente"""

        params = {}

        # Learning rate (log scale)
        params['learning_rate'] = 10 ** np.random.uniform(
            np.log10(self.space.learning_rate[0]),
            np.log10(self.space.learning_rate[1])
        )

        # Batch size
        params['batch_size'] = random.choice(self.space.batch_size)

        # Weight decay (log scale)
        params['weight_decay'] = 10 ** np.random.uniform(
            np.log10(self.space.weight_decay[0]),
            np.log10(self.space.weight_decay[1])
        )

        # Dropout rate
        params['dropout_rate'] = np.random.uniform(
            self.space.dropout_rate[0],
            self.space.dropout_rate[1]
        )

        # Hidden size
        params['hidden_size'] = random.choice(self.space.hidden_size)

        # Number of layers
        params['num_layers'] = random.choice(self.space.num_layers)

        # Optimizer
        params['optimizer'] = random.choice(self.space.optimizer)

        # Scheduler
        params['scheduler'] = random.choice(self.space.scheduler)

        return params

    def _params_to_vector(self, params: Dict[str, Union[float, int, str]]) -> np.ndarray:
        """Convierte parámetros a vector numérico"""

        vector = []

        # Learning rate (log scale)
        vector.append(np.log10(params['learning_rate']))

        # Batch size (normalizado)
        batch_size_idx = self.space.batch_size.index(params['batch_size'])
        vector.append(batch_size_idx / len(self.space.batch_size))

        # Weight decay (log scale)
        vector.append(np.log10(params['weight_decay']))

        # Dropout rate
        vector.append(params['dropout_rate'])

        # Hidden size (normalizado)
        hidden_size_idx = self.space.hidden_size.index(params['hidden_size'])
        vector.append(hidden_size_idx / len(self.space.hidden_size))

        # Number of layers (normalizado)
        num_layers_idx = self.space.num_layers.index(params['num_layers'])
        vector.append(num_layers_idx / len(self.space.num_layers))

        # Optimizer (one-hot)
        optimizer_idx = self.space.optimizer.index(params['optimizer'])
        optimizer_onehot = [0] * len(self.space.optimizer)
        optimizer_onehot[optimizer_idx] = 1
        vector.extend(optimizer_onehot)

        # Scheduler (one-hot)
        scheduler_idx = self.space.scheduler.index(params['scheduler'])
        scheduler_onehot = [0] * len(self.space.scheduler)
        scheduler_onehot[scheduler_idx] = 1
        vector.extend(scheduler_onehot)

        return np.array(vector)

    def _vector_to_params(self, vector: np.ndarray) -> Dict[str, Union[float, int, str]]:
        """Convierte vector numérico a parámetros"""

        params = {}
        idx = 0

        # Learning rate
        params['learning_rate'] = 10 ** vector[idx]
        idx += 1

        # Batch size
        batch_size_idx = int(vector[idx] * len(self.space.batch_size))
        batch_size_idx = min(batch_size_idx, len(self.space.batch_size) - 1)
        params['batch_size'] = self.space.batch_size[batch_size_idx]
        idx += 1

        # Weight decay
        params['weight_decay'] = 10 ** vector[idx]
        idx += 1

        # Dropout rate
        params['dropout_rate'] = vector[idx]
        idx += 1

        # Hidden size
        hidden_size_idx = int(vector[idx] * len(self.space.hidden_size))
        hidden_size_idx = min(hidden_size_idx, len(self.space.hidden_size) - 1)
        params['hidden_size'] = self.space.hidden_size[hidden_size_idx]
        idx += 1

        # Number of layers
        num_layers_idx = int(vector[idx] * len(self.space.num_layers))
        num_layers_idx = min(num_layers_idx, len(self.space.num_layers) - 1)
        params['num_layers'] = self.space.num_layers[num_layers_idx]
        idx += 1

        # Optimizer
        optimizer_onehot = vector[idx:idx + len(self.space.optimizer)]
        optimizer_idx = np.argmax(optimizer_onehot)
        params['optimizer'] = self.space.optimizer[optimizer_idx]
        idx += len(self.space.optimizer)

        # Scheduler
        scheduler_onehot = vector[idx:idx + len(self.space.scheduler)]
        scheduler_idx = np.argmax(scheduler_onehot)
        params['scheduler'] = self.space.scheduler[scheduler_idx]

        return params

    def _fit_gp_model(self) -> None:
        """Entrena el modelo Gaussian Process"""

        X = np.array(self.param_history)
        y = np.array(self.score_history)

        self.gp_model = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=10
        )

        self.gp_model.fit(X, y)

    def _optimize_acquisition(self) -> Dict[str, Union[float, int, str]]:
        """Optimiza la función de adquisición"""

        def acquisition_function(x):
            x = x.reshape(1, -1)
            mean, std = self.gp_model.predict(x, return_std=True)

            if self.config.acquisition_function == "ei":
                # Expected Improvement
                best_score = max(self.score_history) if not self.config.minimize_objective else min(self.score_history)
                z = (mean - best_score) / (std + 1e-9)
                ei = (mean - best_score) * self._normal_cdf(z) + std * self._normal_pdf(z)
                return -ei[0]  # Minimizar para maximizar EI

            elif self.config.acquisition_function == "ucb":
                # Upper Confidence Bound
                beta = 2.0
                ucb = mean + beta * std
                return -ucb[0] if not self.config.minimize_objective else ucb[0]

            else:
                return -mean[0] if not self.config.minimize_objective else mean[0]

        # Optimización
        bounds = self._get_bounds()
        result = minimize(
            acquisition_function,
            x0=self.param_history[-1],
            bounds=bounds,
            method='L-BFGS-B'
        )

        return self._vector_to_params(result.x)

    def _get_bounds(self) -> List[Tuple[float, float]]:
        """Obtiene límites para la optimización"""

        bounds = []

        # Learning rate bounds (log scale)
        bounds.append((np.log10(self.space.learning_rate[0]), np.log10(self.space.learning_rate[1])))

        # Batch size bounds (normalizado)
        bounds.append((0, 1))

        # Weight decay bounds (log scale)
        bounds.append((np.log10(self.space.weight_decay[0]), np.log10(self.space.weight_decay[1])))

        # Dropout rate bounds
        bounds.append(self.space.dropout_rate)

        # Hidden size bounds (normalizado)
        bounds.append((0, 1))

        # Number of layers bounds (normalizado)
        bounds.append((0, 1))

        # Optimizer bounds (one-hot)
        for _ in range(len(self.space.optimizer)):
            bounds.append((0, 1))

        # Scheduler bounds (one-hot)
        for _ in range(len(self.space.scheduler)):
            bounds.append((0, 1))

        return bounds

    def _normal_cdf(self, x):
        """Función de distribución acumulativa normal"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _normal_pdf(self, x):
        """Función de densidad de probabilidad normal"""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)


class GridSearchOptimizer(BaseHyperparameterOptimizer):
    """
    Optimizador de búsqueda en cuadrícula
    """

    def __init__(self, config: HyperparameterTuningConfig, space: HyperparameterSpace):
        super().__init__(config, space)
        self.param_combinations = list(itertools.product(
            self.space.batch_size,
            self.space.hidden_size,
            self.space.num_layers,
            self.space.optimizer,
            self.space.scheduler
        ))
        self.current_index = 0

    def suggest_parameters(self) -> Dict[str, Union[float, int, str]]:
        """Sugiere parámetros de la cuadrícula"""

        if self.current_index >= len(self.param_combinations):
            return None

        batch_size, hidden_size, num_layers, optimizer, scheduler = self.param_combinations[self.current_index]

        params = {
            'learning_rate': np.random.uniform(self.space.learning_rate[0], self.space.learning_rate[1]),
            'batch_size': batch_size,
            'weight_decay': np.random.uniform(self.space.weight_decay[0], self.space.weight_decay[1]),
            'dropout_rate': np.random.uniform(self.space.dropout_rate[0], self.space.dropout_rate[1]),
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'optimizer': optimizer,
            'scheduler': scheduler
        }

        self.current_index += 1
        return params

    def update(self, params: Dict[str, Union[float, int, str]], score: float) -> None:
        """Actualiza el historial"""
        self.trial_history.append({
            'params': params.copy(),
            'score': score,
            'timestamp': time.time()
        })


class RandomSearchOptimizer(BaseHyperparameterOptimizer):
    """
    Optimizador de búsqueda aleatoria
    """

    def suggest_parameters(self) -> Dict[str, Union[float, int, str]]:
        """Sugiere parámetros aleatoriamente"""

        params = {}

        # Learning rate (log scale)
        params['learning_rate'] = 10 ** np.random.uniform(
            np.log10(self.space.learning_rate[0]),
            np.log10(self.space.learning_rate[1])
        )

        # Batch size
        params['batch_size'] = random.choice(self.space.batch_size)

        # Weight decay (log scale)
        params['weight_decay'] = 10 ** np.random.uniform(
            np.log10(self.space.weight_decay[0]),
            np.log10(self.space.weight_decay[1])
        )

        # Dropout rate
        params['dropout_rate'] = np.random.uniform(
            self.space.dropout_rate[0],
            self.space.dropout_rate[1]
        )

        # Hidden size
        params['hidden_size'] = random.choice(self.space.hidden_size)

        # Number of layers
        params['num_layers'] = random.choice(self.space.num_layers)

        # Optimizer
        params['optimizer'] = random.choice(self.space.optimizer)

        # Scheduler
        params['scheduler'] = random.choice(self.space.scheduler)

        return params

    def update(self, params: Dict[str, Union[float, int, str]], score: float) -> None:
        """Actualiza el historial"""
        self.trial_history.append({
            'params': params.copy(),
            'score': score,
            'timestamp': time.time()
        })


class HyperparameterTuningManager:
    """
    Gestor principal de ajuste de hiperparámetros
    Coordina todas las técnicas de optimización
    """

    def __init__(self, config: HyperparameterTuningConfig, space: HyperparameterSpace):
        self.config = config
        self.space = space

        # Crear optimizador según método
        if config.method == "bayesian":
            self.optimizer = BayesianOptimizer(config, space)
        elif config.method == "grid":
            self.optimizer = GridSearchOptimizer(config, space)
        elif config.method == "random":
            self.optimizer = RandomSearchOptimizer(config, space)
        else:
            raise ValueError(f"Método de optimización no soportado: {config.method}")

        self.tuning_history = []
        self.best_model = None

    def optimize(self, model_fn: Callable, data_loader: torch.utils.data.DataLoader) -> Dict:
        """Realiza optimización de hiperparámetros"""

        logger.info(f"Iniciando optimización de hiperparámetros usando {self.config.method}")

        start_time = time.time()

        for trial in range(self.config.max_trials):
            # Sugerir parámetros
            params = self.optimizer.suggest_parameters()

            if params is None:
                break

            logger.info(f"Trial {trial + 1}: Evaluando parámetros {params}")

            # Evaluar parámetros
            score = self.optimizer.evaluate_parameters(params, model_fn, data_loader)

            # Actualizar optimizador
            self.optimizer.update(params, score)

            # Registrar en historial
            self.tuning_history.append({
                'trial': trial + 1,
                'params': params.copy(),
                'score': score,
                'timestamp': time.time()
            })

            logger.info(f"Trial {trial + 1}: Score = {score:.4f}")

            # Early stopping
            if self._should_early_stop():
                logger.info("Early stopping activado")
                break

        end_time = time.time()

        # Crear modelo con mejores parámetros
        if self.optimizer.best_params:
            self.best_model = model_fn(self.optimizer.best_params)

        # Generar resumen
        summary = self._generate_summary(end_time - start_time)

        logger.info(f"Optimización completada. Mejor score: {self.optimizer.best_score:.4f}")

        return summary

    def _should_early_stop(self) -> bool:
        """Determina si se debe aplicar early stopping"""

        if len(self.tuning_history) < self.config.early_stopping_patience:
            return False

        # Verificar si no hay mejora en los últimos trials
        recent_scores = [trial['score'] for trial in self.tuning_history[-self.config.early_stopping_patience:]]

        if self.config.minimize_objective:
            return all(recent_scores[i] >= recent_scores[i-1] for i in range(1, len(recent_scores)))
        else:
            return all(recent_scores[i] <= recent_scores[i-1] for i in range(1, len(recent_scores)))

    def _generate_summary(self, total_time: float) -> Dict:
        """Genera resumen de la optimización"""

        return {
            'best_params': self.optimizer.best_params,
            'best_score': self.optimizer.best_score,
            'total_trials': len(self.tuning_history),
            'total_time': total_time,
            'avg_time_per_trial': total_time / len(self.tuning_history) if self.tuning_history else 0,
            'method': self.config.method,
            'objective_metric': self.config.objective_metric,
            'tuning_history': self.tuning_history,
            'convergence_analysis': self._analyze_convergence()
        }

    def _analyze_convergence(self) -> Dict:
        """Analiza la convergencia de la optimización"""

        if len(self.tuning_history) < 5:
            return {'status': 'insufficient_data'}

        scores = [trial['score'] for trial in self.tuning_history]

        # Calcular tendencia
        if len(scores) >= 10:
            recent_trend = np.polyfit(range(len(scores[-10:])), scores[-10:], 1)[0]
        else:
            recent_trend = np.polyfit(range(len(scores)), scores, 1)[0]

        # Calcular estabilidad
        score_variance = np.var(scores[-10:]) if len(scores) >= 10 else np.var(scores)

        return {
            'recent_trend': recent_trend,
            'score_variance': score_variance,
            'is_converging': abs(recent_trend) < 0.001,
            'stability': 1.0 / (1.0 + score_variance)
        }

    def get_tuning_summary(self) -> Dict:
        """Obtiene resumen del estado de tuning"""
        return {
            'config': {
                'method': self.config.method,
                'max_trials': self.config.max_trials,
                'objective_metric': self.config.objective_metric,
                'minimize_objective': self.config.minimize_objective
            },
            'current_state': {
                'completed_trials': len(self.tuning_history),
                'best_score': self.optimizer.best_score,
                'best_params': self.optimizer.best_params
            },
            'space_definition': {
                'learning_rate_range': self.space.learning_rate,
                'batch_size_options': self.space.batch_size,
                'weight_decay_range': self.space.weight_decay,
                'dropout_rate_range': self.space.dropout_rate,
                'hidden_size_options': self.space.hidden_size,
                'num_layers_options': self.space.num_layers,
                'optimizer_options': self.space.optimizer,
                'scheduler_options': self.space.scheduler
            }
        }

# Funciones de utilidad


def create_hyperparameter_space(learning_rate_range: Tuple[float, float] = (1e-5, 1e-1),
                                batch_sizes: List[int] = None,
                                weight_decay_range: Tuple[float, float] = (1e-6, 1e-2)) -> HyperparameterSpace:
    """Crea espacio de hiperparámetros personalizado"""

    if batch_sizes is None:
        batch_sizes = [16, 32, 64, 128]

    return HyperparameterSpace(
        learning_rate=learning_rate_range,
        batch_size=batch_sizes,
        weight_decay=weight_decay_range
    )


def optimize_hyperparameters(model_fn: Callable, data_loader: torch.utils.data.DataLoader,
                             method: str = "bayesian", max_trials: int = 50) -> Dict:
    """Función de conveniencia para optimizar hiperparámetros"""

    config = HyperparameterTuningConfig(method=method, max_trials=max_trials)
    space = HyperparameterSpace()

    manager = HyperparameterTuningManager(config, space)
    return manager.optimize(model_fn, data_loader)


def compare_optimization_methods(model_fn: Callable, data_loader: torch.utils.data.DataLoader,
                                 methods: List[str] = None, max_trials: int = 20) -> Dict[str, Dict]:
    """Compara diferentes métodos de optimización"""

    if methods is None:
        methods = ['bayesian', 'random', 'grid']

    results = {}
    space = HyperparameterSpace()

    for method in methods:
        logger.info(f"Probando método: {method}")

        config = HyperparameterTuningConfig(method=method, max_trials=max_trials)
        manager = HyperparameterTuningManager(config, space)

        result = manager.optimize(model_fn, data_loader)
        results[method] = result

    return results


# Exportar clases y funciones principales
__all__ = [
    'HyperparameterSpace',
    'HyperparameterTuningConfig',
    'BaseHyperparameterOptimizer',
    'BayesianOptimizer',
    'GridSearchOptimizer',
    'RandomSearchOptimizer',
    'HyperparameterTuningManager',
    'create_hyperparameter_space',
    'optimize_hyperparameters',
    'compare_optimization_methods'
]

logger.info("RFEN3_RN_8 - Sistema de Ajuste Automático de Hiperparámetros cargado correctamente")
