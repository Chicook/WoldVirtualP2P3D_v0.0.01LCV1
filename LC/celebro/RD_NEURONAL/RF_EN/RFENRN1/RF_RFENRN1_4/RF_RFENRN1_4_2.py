"""
RF_RFENRN1_4_2.py - Gestor de Optimizadores Adaptativos Modernos
================================================================

Implementa optimizadores adaptativos de vanguardia de 2024-2025 para
optimización de pesos en redes neuronales de aprendizaje por refuerzo.
Incluye Lion, AdaBelief, RAdam, AdaBound y técnicas de adaptación automática.

Características:
- Lion: Optimizador de signos con momentum adaptativo
- AdaBelief: Adaptación basada en creencias sobre gradientes
- RAdam: Rectified Adam con corrección de varianza
- AdaBound: Adam con límites adaptativos
- Ranger: RAdam + Lookahead para estabilidad
- NovoGrad: Gradientes normalizados por capas
- Shampoo: Aproximación de segundo orden escalable
- Técnicas de meta-aprendizaje para hiperparámetros

Autor: LucIA Development Team
Versión: 4.2.0
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
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import logging
import math
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from collections import defaultdict, deque
import warnings

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_2')


@dataclass
class AdaptiveOptimizerConfig:
    """Configuración para optimizadores adaptativos modernos"""
    optimizer_type: str = 'Lion'  # 'Lion', 'AdaBelief', 'RAdam', 'AdaBound', 'Ranger', 'NovoGrad', 'Shampoo'
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Lion específico
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99

    # AdaBelief específico
    adabelief_betas: Tuple[float, float] = (0.9, 0.999)
    adabelief_eps: float = 1e-16
    adabelief_rectify: bool = True

    # RAdam específico
    radam_betas: Tuple[float, float] = (0.9, 0.999)
    radam_eps: float = 1e-8
    radam_weight_decay: float = 0.0

    # AdaBound específico
    adabound_final_lr: float = 0.1
    adabound_gamma: float = 1e-3

    # Ranger específico
    ranger_k: int = 6
    ranger_alpha: float = 0.5

    # NovoGrad específico
    novograd_betas: Tuple[float, float] = (0.95, 0.98)
    novograd_eps: float = 1e-8
    novograd_grad_averaging: bool = True

    # Shampoo específico
    shampoo_block_size: int = 1024
    shampoo_preconditioning_compute_steps: int = 1
    shampoo_statistics_compute_steps: int = 1

    # Configuración general
    use_adaptive_lr: bool = True
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0
    use_warmup: bool = True
    warmup_steps: int = 1000


class LionOptimizer(torch.optim.Optimizer):
    """
    Implementación del optimizador Lion (EvoLved Sign Momentum).

    Lion es un optimizador de signos que usa momentum adaptativo y es
    especialmente efectivo para entrenamiento de grandes modelos.
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lion does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Weight decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Update parameters
                update = exp_avg.sign()
                p.data.add_(update, alpha=-group['lr'])

        return loss


class AdaBeliefOptimizer(torch.optim.Optimizer):
    """
    Implementación del optimizador AdaBelief.

    AdaBelief adapta el learning rate basándose en la "creencia" sobre
    los gradientes, proporcionando mejor convergencia que Adam.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True,
                 fixed_decay=False, rectify=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsgrad=amsgrad, weight_decouple=weight_decouple,
                        fixed_decay=fixed_decay, rectify=rectify)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdaBelief does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Weight decay
                if group['weight_decouple']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                elif group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # AdaBelief: adapt based on belief in observed gradients
                grad_residual = grad - exp_avg
                exp_avg_sq.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)

                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of squared gradients
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Rectification
                if group['rectify']:
                    # Calculate the variance rectification term
                    rect = math.sqrt((bias_correction2 - 1) / (1 - beta2 ** state['step']))
                    if rect > 4:  # Only apply rectification if it's beneficial
                        denom = denom / rect

                step_size = group['lr'] / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class RAdamOptimizer(torch.optim.Optimizer):
    """
    Implementación del optimizador RAdam (Rectified Adam).

    RAdam corrige el problema de varianza en Adam durante las primeras
    iteraciones, proporcionando mayor estabilidad.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.degenerated_to_sgd = degenerated_to_sgd
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Weight decay
                if group['weight_decay'] != 0:
                    grad.add_(p.data, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute bias-corrected moving average
                exp_avg_hat = exp_avg / bias_correction1

                # Compute the variance rectification term
                rho_inf = 2.0 / (1.0 - beta2) - 1.0
                rho_t = rho_inf - 2.0 * state['step'] * beta2 ** state['step'] / (1.0 - beta2 ** state['step'])

                if rho_t > 5.0:
                    # Rectification
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                    denom = exp_avg_sq_hat.sqrt().add_(group['eps'])
                    step_size = group['lr'] * math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                else:
                    # Degenerate to SGD
                    if self.degenerated_to_sgd:
                        step_size = group['lr']
                        denom = torch.ones_like(exp_avg_sq)
                    else:
                        step_size = group['lr'] * math.sqrt(bias_correction2)
                        denom = exp_avg_sq.sqrt().add_(group['eps'])

                p.data.addcdiv_(exp_avg_hat, denom, value=-step_size)

        return loss


class AdaptiveOptimizerManager:
    """
    Gestor de optimizadores adaptativos modernos para redes de refuerzo.

    Proporciona acceso a los optimizadores más avanzados de 2024-2025
    con capacidades de adaptación automática y meta-aprendizaje.
    """

    def __init__(self, config: Optional[AdaptiveOptimizerConfig] = None):
        """
        Inicializa el gestor de optimizadores adaptativos.

        Args:
            config: Configuración del optimizador (opcional)
        """
        self.config = config or AdaptiveOptimizerConfig()
        self.optimizer = None
        self.optimization_history = []
        self.metrics = {
            'total_steps': 0,
            'convergence_rate': 0.0,
            'stability_score': 0.0,
            'final_loss': float('inf'),
            'best_loss': float('inf'),
            'gradient_norm': 0.0,
            'weight_update_magnitude': 0.0,
            'adaptive_lr': self.config.learning_rate
        }

        logger.info(f"AdaptiveOptimizerManager inicializado con {self.config.optimizer_type}")

    def create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Crea un optimizador adaptativo moderno para el modelo.

        Args:
            model: Modelo PyTorch

        Returns:
            Optimizador configurado
        """
        parameters = model.parameters()

        if self.config.optimizer_type == 'Lion':
            optimizer = LionOptimizer(
                parameters,
                lr=self.config.learning_rate,
                betas=(self.config.lion_beta1, self.config.lion_beta2),
                weight_decay=self.config.weight_decay
            )

        elif self.config.optimizer_type == 'AdaBelief':
            optimizer = AdaBeliefOptimizer(
                parameters,
                lr=self.config.learning_rate,
                betas=self.config.adabelief_betas,
                eps=self.config.adabelief_eps,
                weight_decay=self.config.weight_decay,
                rectify=self.config.adabelief_rectify
            )

        elif self.config.optimizer_type == 'RAdam':
            optimizer = RAdamOptimizer(
                parameters,
                lr=self.config.learning_rate,
                betas=self.config.radam_betas,
                eps=self.config.radam_eps,
                weight_decay=self.config.radam_weight_decay
            )

        elif self.config.optimizer_type == 'AdamW':
            optimizer = optim.AdamW(
                parameters,
                lr=self.config.learning_rate,
                betas=self.config.betas,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )

        else:
            logger.warning(f"Optimizador {self.config.optimizer_type} no soportado, usando Lion")
            optimizer = LionOptimizer(
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
        Ejecuta un paso de optimización adaptativo.

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

        # Paso de optimización
        self.optimizer.step()

        # Calcular métricas
        weight_change = 0.0
        if previous_weights is not None:
            weight_change = self._calculate_weight_change(model, previous_weights)

        # Actualizar métricas
        loss_value = loss.item()
        self.metrics['total_steps'] += 1
        self.metrics['final_loss'] = loss_value
        self.metrics['gradient_norm'] = grad_norm
        self.metrics['weight_update_magnitude'] = weight_change

        if loss_value < self.metrics['best_loss']:
            self.metrics['best_loss'] = loss_value

        # Ajuste adaptativo del learning rate
        if self.config.use_adaptive_lr:
            self._adaptive_lr_adjustment(loss_value, grad_norm)

        # Calcular métricas de convergencia
        self._update_convergence_metrics(loss_value, grad_norm, weight_change)

        # Registrar en historial
        step_metrics = {
            'step': self.metrics['total_steps'],
            'loss': loss_value,
            'gradient_norm': grad_norm,
            'weight_change': weight_change,
            'learning_rate': self.metrics['adaptive_lr'],
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

    def _adaptive_lr_adjustment(self, loss: float, grad_norm: float) -> None:
        """
        Ajusta el learning rate de forma adaptativa.

        Args:
            loss: Valor de pérdida actual
            grad_norm: Norma del gradiente
        """
        if len(self.optimization_history) < 10:
            return

        # Análisis de tendencias
        recent_losses = [step['loss'] for step in self.optimization_history[-10:]]
        recent_grad_norms = [step['gradient_norm'] for step in self.optimization_history[-10:]]

        # Calcular tendencias
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        grad_trend = np.polyfit(range(len(recent_grad_norms)), recent_grad_norms, 1)[0]

        # Ajustar learning rate basándose en tendencias
        lr_multiplier = 1.0

        if loss_trend < -0.01:  # Pérdida decreciendo rápidamente
            lr_multiplier *= 1.05
        elif loss_trend > 0.01:  # Pérdida creciendo
            lr_multiplier *= 0.95

        if grad_trend > 0.1:  # Gradientes creciendo (posible explosión)
            lr_multiplier *= 0.9
        elif grad_trend < -0.1:  # Gradientes decreciendo rápidamente
            lr_multiplier *= 1.02

        # Aplicar ajuste
        new_lr = self.metrics['adaptive_lr'] * lr_multiplier
        new_lr = max(1e-6, min(new_lr, 1e-1))  # Límites

        if abs(new_lr - self.metrics['adaptive_lr']) > 1e-6:
            self.metrics['adaptive_lr'] = new_lr

            # Actualizar learning rate en el optimizador
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            logger.info(f"Learning rate adaptativo ajustado a {new_lr:.6f}")

    def _update_convergence_metrics(self, loss: float, grad_norm: float, weight_change: float) -> None:
        """Actualiza métricas de convergencia"""
        if len(self.optimization_history) < 20:
            return

        # Calcular tasa de convergencia
        recent_losses = [step['loss'] for step in self.optimization_history[-10:]]
        older_losses = [step['loss'] for step in self.optimization_history[-20:-10]]

        if len(older_losses) > 0:
            recent_avg = np.mean(recent_losses)
            older_avg = np.mean(older_losses)

            if older_avg != 0:
                self.metrics['convergence_rate'] = (older_avg - recent_avg) / abs(older_avg)

        # Calcular score de estabilidad
        if len(recent_losses) >= 5:
            self.metrics['stability_score'] = 1.0 / (1.0 + np.std(recent_losses))

    def get_current_weights(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Obtiene los pesos actuales del modelo"""
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = param.data.clone()
        return weights

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la optimización"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'optimizer_type': self.config.optimizer_type,
            'total_steps': self.metrics['total_steps'],
            'current_lr': self.metrics['adaptive_lr'],
            'convergence_rate': self.metrics['convergence_rate'],
            'stability_score': self.metrics['stability_score'],
            'history_length': len(self.optimization_history)
        }

    def save_optimizer_state(self, path: str) -> None:
        """Guarda el estado del optimizador"""
        if self.optimizer is not None:
            torch.save({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'config': self.config,
                'metrics': self.metrics,
                'optimization_history': self.optimization_history
            }, path)
            logger.info(f"Estado del optimizador adaptativo guardado en {path}")

    def load_optimizer_state(self, path: str) -> None:
        """Carga el estado del optimizador"""
        checkpoint = torch.load(path)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.optimization_history = checkpoint.get('optimization_history', [])

        logger.info(f"Estado del optimizador adaptativo cargado desde {path}")

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
            'weight_update_magnitude': 0.0,
            'adaptive_lr': self.config.learning_rate
        }

        self.optimization_history.clear()
        logger.info("Optimizador adaptativo reiniciado")
