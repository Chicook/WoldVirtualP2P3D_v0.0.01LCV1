"""
RFEN3_RN_2 - Sistema de Aprendizaje Adaptativo y Tasas Dinámicas
Implementación de técnicas avanzadas de ajuste dinámico de tasas de aprendizaje
Incluye: schedulers inteligentes, aprendizaje adaptativo por capas, y meta-aprendizaje
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import time
from . import OptimizationConfig, PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class AdaptiveLearningConfig:
    """Configuración para aprendizaje adaptativo"""
    base_lr: float = 0.001
    min_lr: float = 1e-6
    max_lr: float = 0.1
    warmup_epochs: int = 5
    cooldown_epochs: int = 10
    patience: int = 5
    factor: float = 0.5
    threshold: float = 1e-4
    mode: str = "min"  # min, max, auto
    adaptive_by_layer: bool = True
    meta_learning: bool = False


class BaseScheduler(ABC):
    """Clase base abstracta para schedulers de tasa de aprendizaje"""

    def __init__(self, optimizer: optim.Optimizer, config: AdaptiveLearningConfig):
        self.optimizer = optimizer
        self.config = config
        self.epoch = 0
        self.best_score = float('inf') if config.mode == 'min' else float('-inf')
        self.patience_counter = 0

    @abstractmethod
    def step(self, metrics: Optional[PerformanceMetrics] = None) -> float:
        """Actualiza la tasa de aprendizaje"""
        pass

    def get_current_lr(self) -> float:
        """Obtiene la tasa de aprendizaje actual"""
        return self.optimizer.param_groups[0]['lr']

    def get_lr_history(self) -> List[float]:
        """Obtiene el historial de tasas de aprendizaje"""
        return getattr(self, '_lr_history', [])


class CosineAnnealingWarmRestartsScheduler(BaseScheduler):
    """
    Scheduler de Cosine Annealing con Warm Restarts
    Basado en: "SGDR: Stochastic Gradient Descent with Warm Restarts"
    """

    def __init__(self, optimizer: optim.Optimizer, config: AdaptiveLearningConfig,
                 T_0: int = 10, T_mult: int = 2, eta_min: float = 0):
        super().__init__(optimizer, config)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min or config.min_lr
        self.T_cur = 0
        self.T_i = T_0
        self._lr_history = []

    def step(self, metrics: Optional[PerformanceMetrics] = None) -> float:
        """Actualiza la tasa de aprendizaje con cosine annealing"""
        if self.T_cur == self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult

        # Calcular nueva tasa de aprendizaje
        lr = self.eta_min + (self.config.base_lr - self.eta_min) * \
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2

        # Actualizar tasa de aprendizaje
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.T_cur += 1
        self.epoch += 1
        self._lr_history.append(lr)

        logger.debug(f"Epoch {self.epoch}: LR = {lr:.6f}")
        return lr


class OneCycleScheduler(BaseScheduler):
    """
    Scheduler One Cycle Learning Rate
    Basado en: "Super-Convergence: Very Fast Training of Neural Networks"
    """

    def __init__(self, optimizer: optim.Optimizer, config: AdaptiveLearningConfig,
                 max_lr: float = None, total_steps: int = 1000,
                 pct_start: float = 0.3, div_factor: float = 25.0):
        super().__init__(optimizer, config)
        self.max_lr = max_lr or config.max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.div_factor = div_factor
        self.step_count = 0
        self._lr_history = []

        # Calcular tasas de aprendizaje
        self.start_lr = self.max_lr / self.div_factor
        self.end_lr = self.start_lr / 1e4

    def step(self, metrics: Optional[PerformanceMetrics] = None) -> float:
        """Actualiza la tasa de aprendizaje con one cycle"""
        if self.step_count >= self.total_steps:
            return self.get_current_lr()

        # Calcular progreso
        progress = self.step_count / self.total_steps

        if progress < self.pct_start:
            # Fase de aumento
            phase_progress = progress / self.pct_start
            lr = self.start_lr + (self.max_lr - self.start_lr) * phase_progress
        else:
            # Fase de disminución
            phase_progress = (progress - self.pct_start) / (1 - self.pct_start)
            lr = self.max_lr - (self.max_lr - self.end_lr) * phase_progress

        # Actualizar tasa de aprendizaje
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.step_count += 1
        self.epoch += 1
        self._lr_history.append(lr)

        logger.debug(f"Step {self.step_count}: LR = {lr:.6f}")
        return lr


class AdaptivePlateauScheduler(BaseScheduler):
    """
    Scheduler adaptativo basado en plateau con mejoras inteligentes
    """

    def __init__(self, optimizer: optim.Optimizer, config: AdaptiveLearningConfig):
        super().__init__(optimizer, config)
        self.patience_counter = 0
        self.cooldown_counter = 0
        self._lr_history = []
        self._score_history = []

    def step(self, metrics: Optional[PerformanceMetrics] = None) -> float:
        """Actualiza la tasa de aprendizaje basado en plateau"""
        if metrics is None:
            return self.get_current_lr()

        current_score = metrics.val_loss if self.config.mode == 'min' else metrics.val_accuracy
        self._score_history.append(current_score)

        # Verificar si hay mejora
        if self.config.mode == 'min':
            is_better = current_score < self.best_score - self.config.threshold
        else:
            is_better = current_score > self.best_score + self.config.threshold

        if is_better:
            self.best_score = current_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Reducir tasa de aprendizaje si no hay mejora
        if self.patience_counter >= self.config.patience and self.cooldown_counter == 0:
            current_lr = self.get_current_lr()
            new_lr = max(current_lr * self.config.factor, self.config.min_lr)

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

            self.patience_counter = 0
            self.cooldown_counter = self.config.cooldown_epochs

            logger.info(f"Reduciendo LR de {current_lr:.6f} a {new_lr:.6f}")

        # Cooldown
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        self.epoch += 1
        self._lr_history.append(self.get_current_lr())

        return self.get_current_lr()


class LayerAdaptiveLearningManager:
    """
    Gestor de aprendizaje adaptativo por capas
    Permite diferentes tasas de aprendizaje para diferentes capas
    """

    def __init__(self, model: nn.Module, config: AdaptiveLearningConfig):
        self.model = model
        self.config = config
        self.layer_groups = self._create_layer_groups()
        self.layer_schedulers = {}
        self.layer_metrics = {}

    def _create_layer_groups(self) -> Dict[str, List[nn.Parameter]]:
        """Crea grupos de parámetros por tipo de capa"""
        groups = {
            'conv_layers': [],
            'linear_layers': [],
            'batch_norm_layers': [],
            'embedding_layers': [],
            'other_layers': []
        }

        for name, param in self.model.named_parameters():
            layer_name = name.split('.')[0] if '.' in name else name

            if 'conv' in layer_name.lower():
                groups['conv_layers'].append(param)
            elif 'linear' in layer_name.lower() or 'fc' in layer_name.lower():
                groups['linear_layers'].append(param)
            elif 'bn' in layer_name.lower() or 'norm' in layer_name.lower():
                groups['batch_norm_layers'].append(param)
            elif 'embed' in layer_name.lower():
                groups['embedding_layers'].append(param)
            else:
                groups['other_layers'].append(param)

        return groups

    def create_layer_optimizers(self, base_config: OptimizationConfig) -> Dict[str, optim.Optimizer]:
        """Crea optimizadores específicos para cada grupo de capas"""
        optimizers = {}

        # Tasas de aprendizaje específicas por tipo de capa
        lr_multipliers = {
            'conv_layers': 1.0,
            'linear_layers': 0.5,
            'batch_norm_layers': 0.1,
            'embedding_layers': 2.0,
            'other_layers': 1.0
        }

        for group_name, params in self.layer_groups.items():
            if not params:
                continue

            lr = base_config.learning_rate * lr_multipliers[group_name]

            if base_config.algorithm == "AdamW":
                optimizers[group_name] = optim.AdamW(
                    params, lr=lr, weight_decay=base_config.weight_decay
                )
            elif base_config.algorithm == "SGD":
                optimizers[group_name] = optim.SGD(
                    params, lr=lr, weight_decay=base_config.weight_decay, momentum=base_config.momentum
                )
            else:
                optimizers[group_name] = optim.Adam(
                    params, lr=lr, weight_decay=base_config.weight_decay
                )

        return optimizers

    def update_layer_learning_rates(self, epoch: int, total_epochs: int) -> Dict[str, float]:
        """Actualiza las tasas de aprendizaje por capas"""
        current_lrs = {}

        for group_name, optimizer in self.layer_optimizers.items():
            # Aplicar warmup
            if epoch < self.config.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.warmup_epochs
                base_lr = optimizer.param_groups[0]['lr']
                new_lr = base_lr * warmup_factor
            else:
                # Aplicar decay específico por capa
                decay_factor = self._get_layer_decay_factor(group_name, epoch, total_epochs)
                base_lr = optimizer.param_groups[0]['lr']
                new_lr = base_lr * decay_factor

            # Actualizar tasa de aprendizaje
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

            current_lrs[group_name] = new_lr

        return current_lrs

    def _get_layer_decay_factor(self, group_name: str, epoch: int, total_epochs: int) -> float:
        """Calcula el factor de decay específico para cada tipo de capa"""
        progress = epoch / total_epochs

        # Factores de decay específicos por tipo de capa
        decay_factors = {
            'conv_layers': 0.95,      # Decay más lento para capas convolucionales
            'linear_layers': 0.9,     # Decay moderado para capas lineales
            'batch_norm_layers': 0.98,  # Decay muy lento para batch norm
            'embedding_layers': 0.85,  # Decay más rápido para embeddings
            'other_layers': 0.9       # Decay estándar
        }

        factor = decay_factors.get(group_name, 0.9)
        return factor ** (progress * 10)  # Aplicar decay gradual


class MetaLearningScheduler(BaseScheduler):
    """
    Scheduler de meta-aprendizaje que ajusta la tasa de aprendizaje
    basado en el rendimiento histórico y patrones de convergencia
    """

    def __init__(self, optimizer: optim.Optimizer, config: AdaptiveLearningConfig):
        super().__init__(optimizer, config)
        self.performance_history = []
        self.lr_history = []
        self.convergence_patterns = []
        self.meta_model = self._initialize_meta_model()

    def _initialize_meta_model(self) -> Dict:
        """Inicializa el modelo meta para predicción de tasas de aprendizaje"""
        return {
            'convergence_speed': 0.0,
            'stability_score': 0.0,
            'optimal_lr_range': (self.config.min_lr, self.config.max_lr),
            'adaptation_rate': 0.1
        }

    def step(self, metrics: Optional[PerformanceMetrics] = None) -> float:
        """Actualiza la tasa de aprendizaje usando meta-aprendizaje"""
        if metrics is None:
            return self.get_current_lr()

        # Actualizar historial de rendimiento
        self.performance_history.append({
            'loss': metrics.val_loss,
            'accuracy': metrics.val_accuracy,
            'gradient_norm': metrics.gradient_norm,
            'epoch': metrics.epoch
        })

        # Analizar patrones de convergencia
        convergence_analysis = self._analyze_convergence_patterns()

        # Predecir tasa de aprendizaje óptima
        predicted_lr = self._predict_optimal_lr(convergence_analysis)

        # Aplicar tasa de aprendizaje con suavizado
        current_lr = self.get_current_lr()
        new_lr = current_lr * 0.9 + predicted_lr * 0.1

        # Limitar dentro del rango permitido
        new_lr = max(min(new_lr, self.config.max_lr), self.config.min_lr)

        # Actualizar tasa de aprendizaje
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        self.epoch += 1
        self.lr_history.append(new_lr)

        logger.debug(f"Meta-Learning Epoch {self.epoch}: LR = {new_lr:.6f}")
        return new_lr

    def _analyze_convergence_patterns(self) -> Dict:
        """Analiza patrones de convergencia del modelo"""
        if len(self.performance_history) < 5:
            return {'pattern': 'insufficient_data', 'confidence': 0.0}

        recent_losses = [p['loss'] for p in self.performance_history[-5:]]
        recent_gradients = [p['gradient_norm'] for p in self.performance_history[-5:]]

        # Calcular velocidad de convergencia
        loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
        gradient_trend = np.polyfit(range(len(recent_gradients)), recent_gradients, 1)[0]

        # Calcular estabilidad
        loss_variance = np.var(recent_losses)
        gradient_variance = np.var(recent_gradients)

        stability_score = 1.0 / (1.0 + loss_variance + gradient_variance)

        return {
            'loss_trend': loss_trend,
            'gradient_trend': gradient_trend,
            'stability_score': stability_score,
            'convergence_speed': abs(loss_trend),
            'pattern': 'converging' if loss_trend < 0 else 'diverging'
        }

    def _predict_optimal_lr(self, analysis: Dict) -> float:
        """Predice la tasa de aprendizaje óptima basada en el análisis"""
        base_lr = self.config.base_lr

        if analysis['pattern'] == 'converging':
            # Si está convergiendo, mantener o reducir ligeramente
            if analysis['stability_score'] > 0.8:
                return base_lr * 0.95  # Reducir ligeramente
            else:
                return base_lr  # Mantener
        else:
            # Si está divergiendo, reducir significativamente
            return base_lr * 0.5


class AdaptiveLearningManager:
    """
    Gestor principal de aprendizaje adaptativo
    Coordina todos los componentes del sistema adaptativo
    """

    def __init__(self, model: nn.Module, config: AdaptiveLearningConfig):
        self.model = model
        self.config = config
        self.scheduler = None
        self.layer_manager = None
        self.meta_scheduler = None
        self.learning_history = []

        # Inicializar componentes según configuración
        if self.config.adaptive_by_layer:
            self.layer_manager = LayerAdaptiveLearningManager(model, config)

        if self.config.meta_learning:
            self.meta_scheduler = MetaLearningScheduler(None, config)

    def initialize_scheduler(self, optimizer: optim.Optimizer,
                             scheduler_type: str = "cosine") -> BaseScheduler:
        """Inicializa el scheduler principal"""

        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingWarmRestartsScheduler(optimizer, self.config)
        elif scheduler_type == "onecycle":
            self.scheduler = OneCycleScheduler(optimizer, self.config)
        elif scheduler_type == "plateau":
            self.scheduler = AdaptivePlateauScheduler(optimizer, self.config)
        else:
            raise ValueError(f"Tipo de scheduler no soportado: {scheduler_type}")

        return self.scheduler

    def step(self, metrics: Optional[PerformanceMetrics] = None) -> Dict[str, float]:
        """Realiza un paso de actualización de tasas de aprendizaje"""
        results = {}

        # Actualizar scheduler principal
        if self.scheduler:
            main_lr = self.scheduler.step(metrics)
            results['main_lr'] = main_lr

        # Actualizar tasas por capas
        if self.layer_manager:
            layer_lrs = self.layer_manager.update_layer_learning_rates(
                metrics.epoch if metrics else 0, 100
            )
            results.update(layer_lrs)

        # Actualizar meta-scheduler
        if self.meta_scheduler:
            meta_lr = self.meta_scheduler.step(metrics)
            results['meta_lr'] = meta_lr

        # Guardar historial
        self.learning_history.append({
            'epoch': metrics.epoch if metrics else 0,
            'timestamp': time.time(),
            'learning_rates': results.copy()
        })

        return results

    def get_learning_summary(self) -> Dict:
        """Obtiene un resumen del estado del aprendizaje adaptativo"""
        return {
            'total_epochs': len(self.learning_history),
            'current_lr': self.scheduler.get_current_lr() if self.scheduler else 0.0,
            'layer_adaptive': self.config.adaptive_by_layer,
            'meta_learning': self.config.meta_learning,
            'best_performance': min([h.get('val_loss', float('inf'))
                                     for h in self.learning_history]) if self.learning_history else 0.0
        }

# Funciones de utilidad


def create_adaptive_scheduler(optimizer: optim.Optimizer,
                              scheduler_type: str = "cosine",
                              config: Optional[AdaptiveLearningConfig] = None) -> BaseScheduler:
    """Factory para crear schedulers adaptativos"""

    if config is None:
        config = AdaptiveLearningConfig()

    if scheduler_type == "cosine":
        return CosineAnnealingWarmRestartsScheduler(optimizer, config)
    elif scheduler_type == "onecycle":
        return OneCycleScheduler(optimizer, config)
    elif scheduler_type == "plateau":
        return AdaptivePlateauScheduler(optimizer, config)
    else:
        raise ValueError(f"Tipo de scheduler no soportado: {scheduler_type}")


def analyze_learning_curves(history: List[Dict]) -> Dict[str, float]:
    """Analiza las curvas de aprendizaje para detectar patrones"""
    if len(history) < 10:
        return {'status': 'insufficient_data'}

    losses = [h.get('val_loss', 0) for h in history]
    lrs = [h.get('learning_rates', {}).get('main_lr', 0) for h in history]

    # Calcular métricas de análisis
    loss_trend = np.polyfit(range(len(losses)), losses, 1)[0]
    lr_variance = np.var(lrs)
    convergence_rate = abs(loss_trend)

    return {
        'loss_trend': loss_trend,
        'lr_variance': lr_variance,
        'convergence_rate': convergence_rate,
        'is_converging': loss_trend < 0,
        'lr_stability': 1.0 / (1.0 + lr_variance)
    }


# Exportar clases y funciones principales
__all__ = [
    'AdaptiveLearningConfig',
    'BaseScheduler',
    'CosineAnnealingWarmRestartsScheduler',
    'OneCycleScheduler',
    'AdaptivePlateauScheduler',
    'LayerAdaptiveLearningManager',
    'MetaLearningScheduler',
    'AdaptiveLearningManager',
    'create_adaptive_scheduler',
    'analyze_learning_curves'
]

logger.info("RFEN3_RN_2 - Sistema de Aprendizaje Adaptativo cargado correctamente")
