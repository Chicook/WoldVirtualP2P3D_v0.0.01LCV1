"""
RF_RFENRN1_4_1.py - Gestor de Optimizadores Clásicos Avanzados
===============================================================

Implementa optimizadores clásicos mejorados y sus variantes modernas para
optimización de pesos en redes neuronales de aprendizaje por refuerzo.
Incluye SGD, Adam, RMSprop y sus extensiones con técnicas de estabilización.

Características:
- SGD con momentum, Nesterov y variantes adaptativas
- Adam con correcciones de sesgo y variantes mejoradas
- RMSprop con momentum y adaptaciones modernas
- Análisis de convergencia y estabilidad
- Configuración automática de hiperparámetros
- Soporte para múltiples frameworks

Autor: LucIA Development Team
Versión: 4.1.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_1')


@dataclass
class ClassicalOptimizerConfig:
    """Configuración para optimizadores clásicos"""
    optimizer_type: str = 'Adam'  # 'SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    nesterov: bool = False
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    amsgrad: bool = False
    alpha: float = 0.99  # Para RMSprop
    centered: bool = False
    lr_decay: float = 0.0
    initial_accumulator_value: float = 0.1
    rho: float = 0.9  # Para Adadelta
    eps_decay: float = 1e-6
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    use_warmup: bool = False
    warmup_steps: int = 1000
    use_cosine_annealing: bool = False
    T_max: int = 1000
    eta_min: float = 1e-6


class ConvergenceAnalyzer:
    """Analizador de convergencia para optimizadores"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)
        self.weight_changes = deque(maxlen=window_size)
        self.convergence_metrics = {}

    def update(self, loss: float, grad_norm: float, weight_change: float):
        """Actualiza métricas de convergencia"""
        self.loss_history.append(loss)
        self.gradient_norms.append(grad_norm)
        self.weight_changes.append(weight_change)

    def is_converged(self, threshold: float = 1e-6) -> bool:
        """Verifica si el optimizador ha convergido"""
        if len(self.loss_history) < self.window_size:
            return False

        recent_losses = list(self.loss_history)[-self.window_size//2:]
        older_losses = list(self.loss_history)[-self.window_size:-self.window_size//2]

        if len(recent_losses) == 0 or len(older_losses) == 0:
            return False

        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)

        improvement = abs(older_avg - recent_avg) / abs(older_avg + 1e-8)

        return improvement < threshold

    def get_convergence_rate(self) -> float:
        """Calcula la tasa de convergencia"""
        if len(self.loss_history) < 2:
            return 0.0

        losses = list(self.loss_history)
        if len(losses) < 10:
            return 0.0

        recent_losses = losses[-10:]
        older_losses = losses[-20:-10] if len(losses) >= 20 else losses[:-10]

        if len(older_losses) == 0:
            return 0.0

        recent_avg = np.mean(recent_losses)
        older_avg = np.mean(older_losses)

        if older_avg == 0:
            return 0.0

        return (older_avg - recent_avg) / abs(older_avg)

    def get_stability_score(self) -> float:
        """Calcula el score de estabilidad"""
        if len(self.loss_history) < 10:
            return 0.0

        losses = list(self.loss_history)[-10:]
        return 1.0 / (1.0 + np.std(losses))


class LearningRateScheduler:
    """Programador de learning rate avanzado"""

    def __init__(self, config: ClassicalOptimizerConfig):
        self.config = config
        self.step_count = 0
        self.base_lr = config.learning_rate

    def get_lr(self) -> float:
        """Obtiene el learning rate actual"""
        if self.config.use_warmup and self.step_count < self.config.warmup_steps:
            # Warmup linear
            return self.base_lr * (self.step_count + 1) / self.config.warmup_steps

        if self.config.use_cosine_annealing:
            # Cosine annealing
            if self.step_count < self.config.warmup_steps:
                return self.base_lr * (self.step_count + 1) / self.config.warmup_steps

            progress = (self.step_count - self.config.warmup_steps) / self.config.T_max
            progress = min(progress, 1.0)

            return self.config.eta_min + (self.base_lr - self.config.eta_min) * \
                (1 + math.cos(math.pi * progress)) / 2

        # Decay exponencial
        if self.config.lr_decay > 0:
            return self.base_lr * math.exp(-self.config.lr_decay * self.step_count)

        return self.base_lr

    def step(self):
        """Incrementa el contador de pasos"""
        self.step_count += 1


class ClassicalOptimizerManager:
    """
    Gestor de optimizadores clásicos mejorados para redes de refuerzo.

    Proporciona implementaciones optimizadas de SGD, Adam, RMSprop y sus
    variantes con análisis automático de convergencia y configuración adaptativa.
    """

    def __init__(self, config: Optional[ClassicalOptimizerConfig] = None):
        """
        Inicializa el gestor de optimizadores clásicos.

        Args:
            config: Configuración del optimizador (opcional)
        """
        self.config = config or ClassicalOptimizerConfig()
        self.optimizer = None
        self.scheduler = LearningRateScheduler(self.config)
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.optimization_history = []
        self.metrics = {
            'total_steps': 0,
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'final_loss': float('inf'),
            'best_loss': float('inf'),
            'gradient_norm': 0.0,
            'weight_update_magnitude': 0.0
        }

        logger.info(f"ClassicalOptimizerManager inicializado con {self.config.optimizer_type}")

    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """
        Crea un optimizador clásico para el modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Optimizador configurado
        """
        parameters = model.parameters()

        if self.config.optimizer_type == 'SGD':
            optimizer = optim.SGD(
                parameters,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=self.config.nesterov
            )

        elif self.config.optimizer_type == 'Adam':
            optimizer = optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
                amsgrad=self.config.amsgrad
            )

        elif self.config.optimizer_type == 'RMSprop':
            optimizer = optim.RMSprop(
                parameters,
                lr=self.config.learning_rate,
                alpha=self.config.alpha,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum,
                centered=self.config.centered
            )

        elif self.config.optimizer_type == 'Adagrad':
            optimizer = optim.Adagrad(
                parameters,
                lr=self.config.learning_rate,
                lr_decay=self.config.lr_decay,
                weight_decay=self.config.weight_decay,
                initial_accumulator_value=self.config.initial_accumulator_value,
                eps=self.config.eps
            )

        elif self.config.optimizer_type == 'Adadelta':
            optimizer = optim.Adadelta(
                parameters,
                lr=self.config.learning_rate,
                rho=self.config.rho,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )

        else:
            logger.warning(f"Optimizador {self.config.optimizer_type} no soportado, usando Adam")
            optimizer = optim.Adam(
                parameters,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )

        self.optimizer = optimizer
        logger.info(f"Optimizador {self.config.optimizer_type} creado con lr={self.config.learning_rate}")

        return optimizer

    def optimize_step(self, model: nn.Module, loss: torch.Tensor,
                      previous_weights: Optional[Dict] = None) -> Dict[str, float]:
        """
        Ejecuta un paso de optimización.

        Args:
            model: Modelo PyTorch
            loss: Tensor de pérdida
            previous_weights: Pesos anteriores para análisis

        Returns:
            Métricas del paso de optimización
        """
        if self.optimizer is None:
            raise ValueError("Optimizador no inicializado. Llame a create_optimizer() primero.")

        # Calcular gradientes
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping si está habilitado
        grad_norm = 0.0
        if self.config.use_gradient_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.max_grad_norm
            )
        else:
            grad_norm = self._calculate_gradient_norm(model)

        # Actualizar learning rate
        current_lr = self.scheduler.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = current_lr

        # Paso de optimización
        self.optimizer.step()
        self.scheduler.step()

        # Calcular métricas
        weight_change = 0.0
        if previous_weights is not None:
            weight_change = self._calculate_weight_change(model, previous_weights)

        # Actualizar análisis de convergencia
        loss_value = loss.item()
        self.convergence_analyzer.update(loss_value, grad_norm, weight_change)

        # Actualizar métricas
        self.metrics['total_steps'] += 1
        self.metrics['final_loss'] = loss_value
        self.metrics['gradient_norm'] = grad_norm
        self.metrics['weight_update_magnitude'] = weight_change

        if loss_value < self.metrics['best_loss']:
            self.metrics['best_loss'] = loss_value

        # Calcular métricas de convergencia
        self.metrics['convergence_rate'] = self.convergence_analyzer.get_convergence_rate()
        self.metrics['stability_score'] = self.convergence_analyzer.get_stability_score()

        # Registrar en historial
        step_metrics = {
            'step': self.metrics['total_steps'],
            'loss': loss_value,
            'gradient_norm': grad_norm,
            'weight_change': weight_change,
            'learning_rate': current_lr,
            'convergence_rate': self.metrics['convergence_rate'],
            'stability_score': self.metrics['stability_score']
        }
        self.optimization_history.append(step_metrics)

        return step_metrics

    def _calculate_gradient_norm(self, model: nn.Module) -> float:
        """Calcula la norma del gradiente"""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)

    def _calculate_weight_change(self, model: nn.Module, previous_weights: Dict) -> float:
        """Calcula la magnitud del cambio de pesos"""
        total_change = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if name in previous_weights:
                weight_change = torch.norm(param.data - previous_weights[name]).item()
                total_change += weight_change
                param_count += 1

        return total_change / param_count if param_count > 0 else 0.0

    def is_converged(self, threshold: float = 1e-6) -> bool:
        """Verifica si el optimizador ha convergido"""
        return self.convergence_analyzer.is_converged(threshold)

    def get_current_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Obtiene los pesos actuales del modelo"""
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data.clone()
        return weights

    def adaptive_learning_rate_adjustment(self, model: nn.Module) -> None:
        """
        Ajusta el learning rate de forma adaptativa basándose en métricas.

        Args:
            model: Modelo PyTorch
        """
        convergence_rate = self.metrics['convergence_rate']
        stability_score = self.metrics['stability_score']

        # Ajustar learning rate basándose en convergencia
        if convergence_rate < 0.01:  # Convergencia lenta
            new_lr = self.config.learning_rate * 1.1
        elif convergence_rate > 0.1:  # Convergencia muy rápida (posible inestabilidad)
            new_lr = self.config.learning_rate * 0.9
        else:
            new_lr = self.config.learning_rate

        # Ajustar basándose en estabilidad
        if stability_score < 0.5:  # Baja estabilidad
            new_lr *= 0.95
        elif stability_score > 0.8:  # Alta estabilidad
            new_lr *= 1.05

        # Aplicar límites
        new_lr = max(1e-6, min(new_lr, 1e-1))

        if abs(new_lr - self.config.learning_rate) > 1e-6:
            self.config.learning_rate = new_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            logger.info(f"Learning rate ajustado a {new_lr:.6f}")

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la optimización"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'convergence_analysis': {
                'is_converged': self.is_converged(),
                'convergence_rate': self.metrics['convergence_rate'],
                'stability_score': self.metrics['stability_score'],
                'total_steps': self.metrics['total_steps']
            },
            'learning_rate_info': {
                'current_lr': self.scheduler.get_lr(),
                'base_lr': self.scheduler.base_lr,
                'step_count': self.scheduler.step_count
            },
            'history_length': len(self.optimization_history)
        }

    def save_optimizer_state(self, path: str) -> None:
        """Guarda el estado del optimizador"""
        if self.optimizer is not None:
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'metrics': self.metrics,
                'optimization_history': self.optimization_history,
                'scheduler_step_count': self.scheduler.step_count
            }, path)
            logger.info(f"Estado del optimizador guardado en {path}")

    def load_optimizer_state(self, path: str) -> None:
        """Carga el estado del optimizador"""
        checkpoint = torch.load(path)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.optimization_history = checkpoint.get('optimization_history', [])
        self.scheduler.step_count = checkpoint.get('scheduler_step_count', 0)

        logger.info(f"Estado del optimizador cargado desde {path}")

    def reset_optimizer(self) -> None:
        """Reinicia el optimizador manteniendo la configuración"""
        if self.optimizer is not None:
            self.optimizer.zero_grad()

        self.metrics = {
            'total_steps': 0,
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'final_loss': float('inf'),
            'best_loss': float('inf'),
            'gradient_norm': 0.0,
            'weight_update_magnitude': 0.0
        }

        self.optimization_history.clear()
        self.convergence_analyzer = ConvergenceAnalyzer()
        self.scheduler.step_count = 0

        logger.info("Optimizador reiniciado")
