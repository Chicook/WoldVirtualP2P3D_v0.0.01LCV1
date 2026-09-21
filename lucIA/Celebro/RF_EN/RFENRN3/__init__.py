"""
RFENRN3 - Módulo de Optimización Avanzada de Pesos para Redes Neuronales
Versión: 2025.1.0
Autor: Sistema de Red Neuronal Modular
Descripción: Implementación de técnicas avanzadas de optimización de pesos para redes neuronales
             incluyendo optimizadores modernos, aprendizaje adaptativo y consolidación de memoria.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
import json
import time
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración global del módulo
MODULE_CONFIG = {
    "version": "2025.1.0",
    "max_file_lines": 300,
    "optimization_algorithms": [
        "Lion", "AdaBelief", "RAdam", "AdamW", "RMSprop", "SGD"
    ],
    "learning_rate_schedules": [
        "cosine", "exponential", "polynomial", "step", "plateau"
    ],
    "regularization_methods": [
        "dropout", "batch_norm", "layer_norm", "weight_decay", "gradient_clipping"
    ],
    "memory_consolidation": True,
    "transfer_learning": True,
    "hyperparameter_tuning": True,
    "performance_monitoring": True
}


@dataclass
class OptimizationConfig:
    """Configuración para optimización de pesos"""
    algorithm: str = "AdamW"
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    momentum: float = 0.9
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_norm: float = 1.0
    use_scheduler: bool = True
    scheduler_type: str = "cosine"
    warmup_epochs: int = 5


@dataclass
class PerformanceMetrics:
    """Métricas de rendimiento del modelo"""
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    weight_norm: float = 0.0
    epoch: int = 0
    timestamp: float = 0.0


class BaseOptimizer(ABC):
    """Clase base abstracta para optimizadores personalizados"""

    def __init__(self, params: List[torch.Tensor], config: OptimizationConfig):
        self.params = params
        self.config = config
        self.state = {}
        self.step_count = 0

    @abstractmethod
    def step(self, closure: Optional[Callable] = None) -> None:
        """Realiza un paso de optimización"""
        pass

    @abstractmethod
    def zero_grad(self) -> None:
        """Limpia los gradientes"""
        pass


class WeightOptimizationManager:
    """Gestor principal de optimización de pesos"""

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.optimizer = None
        self.scheduler = None
        self.metrics_history = []
        self.best_weights = None
        self.best_score = float('inf')
        self.patience_counter = 0

    def initialize_optimizer(self, model: nn.Module) -> None:
        """Inicializa el optimizador según la configuración"""
        try:
            if self.config.algorithm == "AdamW":
                self.optimizer = optim.AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    betas=(self.config.beta1, self.config.beta2),
                    eps=self.config.epsilon
                )
            elif self.config.algorithm == "RMSprop":
                self.optimizer = optim.RMSprop(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    momentum=self.config.momentum,
                    eps=self.config.epsilon
                )
            elif self.config.algorithm == "SGD":
                self.optimizer = optim.SGD(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay,
                    momentum=self.config.momentum
                )
            else:
                raise ValueError(f"Optimizador no soportado: {self.config.algorithm}")

            logger.info(f"Optimizador {self.config.algorithm} inicializado correctamente")

        except Exception as e:
            logger.error(f"Error al inicializar optimizador: {e}")
            raise

    def initialize_scheduler(self) -> None:
        """Inicializa el scheduler de tasa de aprendizaje"""
        if not self.config.use_scheduler or not self.optimizer:
            return

        try:
            if self.config.scheduler_type == "cosine":
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.epochs,
                    eta_min=self.config.learning_rate * 0.01
                )
            elif self.config.scheduler_type == "exponential":
                self.scheduler = optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=0.95
                )
            elif self.config.scheduler_type == "step":
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=20,
                    gamma=0.5
                )
            elif self.config.scheduler_type == "plateau":
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                )

            logger.info(f"Scheduler {self.config.scheduler_type} inicializado")

        except Exception as e:
            logger.error(f"Error al inicializar scheduler: {e}")
            raise

    def update_metrics(self, metrics: PerformanceMetrics) -> None:
        """Actualiza el historial de métricas"""
        self.metrics_history.append(metrics)

        # Verificar si es la mejor puntuación
        if metrics.val_loss < self.best_score:
            self.best_score = metrics.val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

    def should_early_stop(self) -> bool:
        """Verifica si se debe aplicar early stopping"""
        return self.patience_counter >= self.config.early_stopping_patience

    def get_current_lr(self) -> float:
        """Obtiene la tasa de aprendizaje actual"""
        if self.optimizer:
            return self.optimizer.param_groups[0]['lr']
        return self.config.learning_rate

    def save_best_weights(self, model: nn.Module) -> None:
        """Guarda los mejores pesos del modelo"""
        self.best_weights = {name: param.clone() for name, param in model.named_parameters()}
        logger.info("Mejores pesos guardados")

    def load_best_weights(self, model: nn.Module) -> None:
        """Carga los mejores pesos en el modelo"""
        if self.best_weights:
            for name, param in model.named_parameters():
                if name in self.best_weights:
                    param.data = self.best_weights[name].data
            logger.info("Mejores pesos cargados")

    def get_optimization_summary(self) -> Dict:
        """Obtiene un resumen del estado de optimización"""
        return {
            "algorithm": self.config.algorithm,
            "current_lr": self.get_current_lr(),
            "best_score": self.best_score,
            "patience_counter": self.patience_counter,
            "total_epochs": len(self.metrics_history),
            "should_stop": self.should_early_stop()
        }

# Funciones de utilidad para el módulo


def calculate_gradient_norm(model: nn.Module) -> float:
    """Calcula la norma de los gradientes"""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def calculate_weight_norm(model: nn.Module) -> float:
    """Calcula la norma de los pesos"""
    total_norm = 0.0
    for param in model.parameters():
        param_norm = param.data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** (1. / 2)


def apply_gradient_clipping(model: nn.Module, max_norm: float) -> None:
    """Aplica recorte de gradientes"""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def initialize_weights_xavier(model: nn.Module) -> None:
    """Inicializa pesos usando Xavier/Glorot"""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def initialize_weights_he(model: nn.Module) -> None:
    """Inicializa pesos usando He/Kaiming"""
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def save_config(config: OptimizationConfig, filepath: str) -> None:
    """Guarda la configuración en un archivo JSON"""
    config_dict = {
        "algorithm": config.algorithm,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "beta1": config.beta1,
        "beta2": config.beta2,
        "epsilon": config.epsilon,
        "momentum": config.momentum,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "gradient_clip_norm": config.gradient_clip_norm,
        "use_scheduler": config.use_scheduler,
        "scheduler_type": config.scheduler_type,
        "warmup_epochs": config.warmup_epochs
    }

    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

    logger.info(f"Configuración guardada en {filepath}")


def load_config(filepath: str) -> OptimizationConfig:
    """Carga la configuración desde un archivo JSON"""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)

    return OptimizationConfig(**config_dict)


# Exportar clases y funciones principales
__all__ = [
    'OptimizationConfig',
    'PerformanceMetrics',
    'BaseOptimizer',
    'WeightOptimizationManager',
    'calculate_gradient_norm',
    'calculate_weight_norm',
    'apply_gradient_clipping',
    'initialize_weights_xavier',
    'initialize_weights_he',
    'save_config',
    'load_config',
    'MODULE_CONFIG'
]

logger.info("Módulo RFENRN3 - Optimización Avanzada de Pesos inicializado correctamente")
