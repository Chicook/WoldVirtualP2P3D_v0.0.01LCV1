"""
RF_RFENRN1_4_4.py - Gestor de Regularización Avanzada
=====================================================

Implementa técnicas avanzadas de regularización para optimización de pesos
en redes neuronales de aprendizaje por refuerzo. Incluye dropout adaptativo,
weight decay inteligente, batch normalization y técnicas de estabilización.

Características:
- Dropout adaptativo con tasas dinámicas
- Weight decay con decay schedules
- Batch normalization avanzada
- Layer normalization
- Spectral normalization
- Gradient penalty
- Orthogonal regularization
- Elastic net regularization
- Early stopping inteligente

Autor: LucIA Development Team
Versión: 4.4.0
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
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

logger = logging.getLogger('RFENRN1.RF_RFENRN1_4_4')


@dataclass
class RegularizationConfig:
    """Configuración para técnicas de regularización"""
    # Dropout adaptativo
    adaptive_dropout: bool = True
    dropout_rate: float = 0.1
    dropout_schedule: str = 'cosine'  # 'cosine', 'linear', 'exponential'
    dropout_min_rate: float = 0.05
    dropout_max_rate: float = 0.5

    # Weight decay
    weight_decay: float = 1e-4
    weight_decay_schedule: str = 'cosine'  # 'cosine', 'linear', 'exponential'
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-2

    # Batch normalization
    use_batch_norm: bool = True
    batch_norm_momentum: float = 0.1
    batch_norm_eps: float = 1e-5

    # Layer normalization
    use_layer_norm: bool = False
    layer_norm_eps: float = 1e-5

    # Spectral normalization
    use_spectral_norm: bool = False
    spectral_norm_power_iter: int = 1

    # Gradient penalty
    use_gradient_penalty: bool = False
    gradient_penalty_weight: float = 10.0

    # Orthogonal regularization
    use_orthogonal_reg: bool = False
    orthogonal_reg_weight: float = 1e-4

    # Elastic net
    use_elastic_net: bool = False
    elastic_net_l1: float = 1e-4
    elastic_net_l2: float = 1e-4

    # Early stopping
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4


class AdaptiveDropout(nn.Module):
    """Dropout adaptativo con tasas dinámicas"""

    def __init__(self, initial_rate=0.1, min_rate=0.05, max_rate=0.5, schedule='cosine'):
        super().__init__()
        self.initial_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.schedule = schedule
        self.current_rate = initial_rate
        self.step_count = 0

    def forward(self, x):
        if self.training:
            return F.dropout(x, p=self.current_rate, training=True)
        return x

    def update_rate(self, progress):
        """Actualiza la tasa de dropout basándose en el progreso"""
        if self.schedule == 'cosine':
            self.current_rate = self.min_rate + (self.max_rate - self.min_rate) * \
                (1 + math.cos(math.pi * progress)) / 2
        elif self.schedule == 'linear':
            self.current_rate = self.max_rate - (self.max_rate - self.min_rate) * progress
        elif self.schedule == 'exponential':
            self.current_rate = self.min_rate + (self.max_rate - self.min_rate) * \
                math.exp(-5 * progress)

        self.current_rate = max(self.min_rate, min(self.current_rate, self.max_rate))


class AdaptiveWeightDecay:
    """Weight decay adaptativo con schedules"""

    def __init__(self, initial_decay=1e-4, min_decay=1e-6, max_decay=1e-2, schedule='cosine'):
        self.initial_decay = initial_decay
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.schedule = schedule
        self.current_decay = initial_decay

    def update_decay(self, progress):
        """Actualiza el weight decay basándose en el progreso"""
        if self.schedule == 'cosine':
            self.current_decay = self.min_decay + (self.max_decay - self.min_decay) * \
                (1 + math.cos(math.pi * progress)) / 2
        elif self.schedule == 'linear':
            self.current_decay = self.max_decay - (self.max_decay - self.min_decay) * progress
        elif self.schedule == 'exponential':
            self.current_decay = self.min_decay + (self.max_decay - self.min_decay) * \
                math.exp(-3 * progress)

        self.current_decay = max(self.min_decay, min(self.current_decay, self.max_decay))


class GradientPenalty:
    """Penalización de gradientes para estabilidad"""

    def __init__(self, weight=10.0):
        self.weight = weight

    def compute_penalty(self, model, real_data, fake_data):
        """Calcula la penalización de gradiente"""
        alpha = torch.rand(real_data.size(0), 1, device=real_data.device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Calcular gradientes
        output = model(interpolated)
        gradients = torch.autograd.grad(
            outputs=output,
            inputs=interpolated,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True
        )[0]

        # Calcular penalización
        gradient_norm = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()

        return self.weight * penalty


class OrthogonalRegularization:
    """Regularización ortogonal para estabilidad"""

    def __init__(self, weight=1e-4):
        self.weight = weight

    def compute_penalty(self, model):
        """Calcula la penalización ortogonal"""
        total_penalty = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Reshape para matrices
                weight_matrix = param.view(param.size(0), -1)

                # Calcular penalización ortogonal
                gram_matrix = torch.mm(weight_matrix, weight_matrix.t())
                identity = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
                penalty = torch.norm(gram_matrix - identity, p='fro') ** 2

                total_penalty += penalty
                param_count += 1

        return self.weight * (total_penalty / param_count if param_count > 0 else 0.0)


class ElasticNetRegularization:
    """Regularización Elastic Net (L1 + L2)"""

    def __init__(self, l1_weight=1e-4, l2_weight=1e-4):
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight

    def compute_penalty(self, model):
        """Calcula la penalización Elastic Net"""
        l1_penalty = 0.0
        l2_penalty = 0.0

        for param in model.parameters():
            l1_penalty += torch.norm(param, p=1)
            l2_penalty += torch.norm(param, p=2) ** 2

        return self.l1_weight * l1_penalty + self.l2_weight * l2_penalty


class EarlyStopping:
    """Early stopping inteligente"""

    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, current_loss, model):
        """Verifica si debe detenerse el entrenamiento"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {name: param.data.clone()
                                     for name, param in model.named_parameters()}
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                for name, param in model.named_parameters():
                    param.data.copy_(self.best_weights[name])
            return True

        return False


class RegularizationManager:
    """
    Gestor de técnicas de regularización avanzadas para redes de refuerzo.

    Proporciona un conjunto completo de técnicas de regularización
    con configuración adaptativa y análisis automático.
    """

    def __init__(self, config: Optional[RegularizationConfig] = None):
        """
        Inicializa el gestor de regularización.

        Args:
            config: Configuración de regularización (opcional)
        """
        self.config = config or RegularizationConfig()
        self.adaptive_dropout = None
        self.adaptive_weight_decay = AdaptiveWeightDecay(
            self.config.weight_decay,
            self.config.weight_decay_min,
            self.config.weight_decay_max,
            self.config.weight_decay_schedule
        )
        self.gradient_penalty = GradientPenalty(self.config.gradient_penalty_weight) if self.config.use_gradient_penalty else None
        self.orthogonal_reg = OrthogonalRegularization(self.config.orthogonal_reg_weight) if self.config.use_orthogonal_reg else None
        self.elastic_net = ElasticNetRegularization(self.config.elastic_net_l1, self.config.elastic_net_l2) if self.config.use_elastic_net else None
        self.early_stopping = EarlyStopping(self.config.early_stopping_patience, self.config.early_stopping_min_delta) if self.config.use_early_stopping else None

        self.regularization_history = []
        self.metrics = {
            'total_penalty': 0.0,
            'dropout_rate': self.config.dropout_rate,
            'weight_decay': self.config.weight_decay,
            'gradient_penalty': 0.0,
            'orthogonal_penalty': 0.0,
            'elastic_net_penalty': 0.0
        }

        logger.info("RegularizationManager inicializado")

    def apply_regularization(self, model: nn.Module, progress: float = 0.0) -> Dict[str, float]:
        """
        Aplica técnicas de regularización al modelo.

        Args:
            model: Modelo PyTorch
            progress: Progreso del entrenamiento (0.0 a 1.0)

        Returns:
            Diccionario con penalizaciones aplicadas
        """
        penalties = {}

        # Actualizar weight decay adaptativo
        self.adaptive_weight_decay.update_decay(progress)
        self.metrics['weight_decay'] = self.adaptive_weight_decay.current_decay

        # Aplicar weight decay
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.add_(param.data, alpha=self.adaptive_weight_decay.current_decay)

        # Penalización ortogonal
        if self.orthogonal_reg:
            orthogonal_penalty = self.orthogonal_reg.compute_penalty(model)
            penalties['orthogonal'] = orthogonal_penalty.item()
            self.metrics['orthogonal_penalty'] = orthogonal_penalty.item()

        # Penalización Elastic Net
        if self.elastic_net:
            elastic_penalty = self.elastic_net.compute_penalty(model)
            penalties['elastic_net'] = elastic_penalty.item()
            self.metrics['elastic_net_penalty'] = elastic_penalty.item()

        # Penalización de gradiente (requiere datos específicos)
        if self.gradient_penalty:
            # Esta se aplicaría en el contexto de GANs
            penalties['gradient'] = 0.0
            self.metrics['gradient_penalty'] = 0.0

        # Calcular penalización total
        total_penalty = sum(penalties.values())
        self.metrics['total_penalty'] = total_penalty

        # Registrar en historial
        step_metrics = {
            'progress': progress,
            'penalties': penalties,
            'total_penalty': total_penalty,
            'weight_decay': self.adaptive_weight_decay.current_decay
        }
        self.regularization_history.append(step_metrics)

        return penalties

    def add_adaptive_dropout(self, model: nn.Module) -> None:
        """Añade dropout adaptativo al modelo"""
        if not self.config.adaptive_dropout:
            return

        self.adaptive_dropout = AdaptiveDropout(
            self.config.dropout_rate,
            self.config.dropout_min_rate,
            self.config.dropout_max_rate,
            self.config.dropout_schedule
        )

        # Aplicar dropout a capas lineales
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Insertar dropout después de la capa lineal
                setattr(model, f"{name}_dropout", self.adaptive_dropout)

        logger.info("Dropout adaptativo añadido al modelo")

    def update_dropout_rate(self, progress: float) -> None:
        """Actualiza la tasa de dropout"""
        if self.adaptive_dropout:
            self.adaptive_dropout.update_rate(progress)
            self.metrics['dropout_rate'] = self.adaptive_dropout.current_rate

    def check_early_stopping(self, current_loss: float, model: nn.Module) -> bool:
        """Verifica si debe aplicarse early stopping"""
        if self.early_stopping:
            return self.early_stopping(current_loss, model)
        return False

    def get_regularization_summary(self) -> Dict[str, Any]:
        """Obtiene un resumen de la regularización"""
        return {
            'config': self.config,
            'metrics': self.metrics.copy(),
            'current_dropout_rate': self.metrics['dropout_rate'],
            'current_weight_decay': self.metrics['weight_decay'],
            'total_penalty': self.metrics['total_penalty'],
            'history_length': len(self.regularization_history)
        }

    def save_state(self, path: str) -> None:
        """Guarda el estado de regularización"""
        torch.save({
            'config': self.config,
            'metrics': self.metrics,
            'regularization_history': self.regularization_history,
            'adaptive_weight_decay': self.adaptive_weight_decay,
            'early_stopping': self.early_stopping
        }, path)
        logger.info(f"Estado de regularización guardado en {path}")

    def load_state(self, path: str) -> None:
        """Carga el estado de regularización"""
        checkpoint = torch.load(path)

        self.config = checkpoint.get('config', self.config)
        self.metrics = checkpoint.get('metrics', self.metrics)
        self.regularization_history = checkpoint.get('regularization_history', [])
        self.adaptive_weight_decay = checkpoint.get('adaptive_weight_decay', self.adaptive_weight_decay)
        self.early_stopping = checkpoint.get('early_stopping', self.early_stopping)

        logger.info(f"Estado de regularización cargado desde {path}")

    def reset_state(self) -> None:
        """Reinicia el estado de regularización"""
        self.metrics = {
            'total_penalty': 0.0,
            'dropout_rate': self.config.dropout_rate,
            'weight_decay': self.config.weight_decay,
            'gradient_penalty': 0.0,
            'orthogonal_penalty': 0.0,
            'elastic_net_penalty': 0.0
        }

        self.regularization_history.clear()

        if self.early_stopping:
            self.early_stopping.best_loss = float('inf')
            self.early_stopping.counter = 0
            self.early_stopping.best_weights = None

        logger.info("Estado de regularización reiniciado")
