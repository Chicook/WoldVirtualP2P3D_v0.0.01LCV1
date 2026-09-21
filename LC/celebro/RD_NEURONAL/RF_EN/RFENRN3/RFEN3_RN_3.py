"""
RFEN3_RN_3 - Técnicas de Regularización Avanzadas para Redes Neuronales
Implementación de métodos modernos de regularización para prevenir sobreajuste
Incluye: Dropout adaptativo, BatchNorm mejorado, LayerNorm, Weight Decay inteligente
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    pass  # dependencia pesada opcional
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random

logger = logging.getLogger(__name__)


@dataclass
class RegularizationConfig:
    """Configuración para técnicas de regularización"""
    dropout_rate: float = 0.5
    adaptive_dropout: bool = True
    batch_norm_momentum: float = 0.1
    batch_norm_eps: float = 1e-5
    layer_norm_eps: float = 1e-5
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    stochastic_depth: float = 0.1
    drop_path_rate: float = 0.1


class AdaptiveDropout(nn.Module):
    """
    Dropout adaptativo que ajusta la tasa según el rendimiento del modelo
    """

    def __init__(self, p: float = 0.5, adaptive: bool = True, min_p: float = 0.1, max_p: float = 0.8):
        super().__init__()
        self.p = p
        self.adaptive = adaptive
        self.min_p = min_p
        self.max_p = max_p
        self.current_p = p
        self.performance_history = []
        self.adjustment_factor = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con dropout adaptativo"""
        if self.training:
            if self.adaptive:
                self._update_dropout_rate()

            return F.dropout(x, p=self.current_p, training=True)
        return x

    def _update_dropout_rate(self) -> None:
        """Actualiza la tasa de dropout basado en el rendimiento"""
        if len(self.performance_history) < 5:
            return

        # Calcular tendencia de rendimiento
        recent_performance = self.performance_history[-5:]
        trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]

        # Ajustar tasa de dropout
        if trend < 0:  # Rendimiento mejorando
            self.current_p = max(self.current_p - self.adjustment_factor, self.min_p)
        else:  # Rendimiento empeorando
            self.current_p = min(self.current_p + self.adjustment_factor, self.max_p)

    def update_performance(self, performance: float) -> None:
        """Actualiza el historial de rendimiento"""
        self.performance_history.append(performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)


class ImprovedBatchNorm(nn.Module):
    """
    Batch Normalization mejorado con características adicionales
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_parameter('num_batches_tracked', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con BatchNorm mejorado"""
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

        # Calcular estadísticas
        if self.training:
            mean = x.mean(dim=[0, 2, 3] if x.dim() == 4 else [0])
            var = x.var(dim=[0, 2, 3] if x.dim() == 4 else [0], unbiased=False)

            # Actualizar running stats
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * mean + \
                        (1 - exponential_average_factor) * self.running_mean
                    self.running_var = exponential_average_factor * var + \
                        (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalizar
        x_normalized = (x - mean.view(1, -1, 1, 1) if x.dim() == 4 else x - mean) / \
            torch.sqrt(var.view(1, -1, 1, 1) + self.eps if x.dim() == 4 else var + self.eps)

        # Aplicar transformación afín
        if self.affine:
            x_normalized = x_normalized * self.weight.view(1, -1, 1, 1) + \
                self.bias.view(1, -1, 1, 1) if x.dim() == 4 else \
                x_normalized * self.weight + self.bias

        return x_normalized


class LayerNormImproved(nn.Module):
    """
    Layer Normalization mejorado con características adicionales
    """

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-5,
                 elementwise_affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = [normalized_shape]
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con LayerNorm mejorado"""
        # Calcular dimensiones para normalización
        dims = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))

        # Calcular media y varianza
        mean = x.mean(dim=dims, keepdim=True)
        var = x.var(dim=dims, keepdim=True, unbiased=False)

        # Normalizar
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)

        # Aplicar transformación afín
        if self.elementwise_affine:
            x_normalized = x_normalized * self.weight + self.bias

        return x_normalized


class StochasticDepth(nn.Module):
    """
    Stochastic Depth para regularización durante el entrenamiento
    """

    def __init__(self, drop_rate: float = 0.1, mode: str = 'batch'):
        super().__init__()
        self.drop_rate = drop_rate
        self.mode = mode
        self.scale_by_keep = True

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward pass con Stochastic Depth"""
        if not self.training or self.drop_rate == 0.0:
            return x + residual

        # Generar máscara de dropout
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        keep_prob = 1.0 - self.drop_rate

        if self.scale_by_keep:
            random_tensor.div_(keep_prob)

        # Aplicar máscara
        output = x + residual
        return output * random_tensor


class DropPath(nn.Module):
    """
    Drop Path (Stochastic Depth) para regularización
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass con Drop Path"""
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.div_(keep_prob)

        return x * random_tensor


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss con Label Smoothing
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass con Label Smoothing"""
        log_prob = F.log_softmax(x, dim=-1)
        weight = x.new_ones(x.size()) * self.smoothing / (x.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss


class MixUpAugmentation:
    """
    MixUp Data Augmentation para regularización
    """

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Aplica MixUp augmentation"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam


class CutMixAugmentation:
    """
    CutMix Data Augmentation para regularización
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Aplica CutMix augmentation"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Calcular región de corte
        W = x.size(2)
        H = x.size(3)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Posición aleatoria
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        # Aplicar corte
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

        # Ajustar lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam


class WeightDecayRegularizer:
    """
    Regularizador de Weight Decay inteligente
    """

    def __init__(self, model: nn.Module, weight_decay: float = 0.01,
                 exclude_bias: bool = True, exclude_norm: bool = True):
        self.model = model
        self.weight_decay = weight_decay
        self.exclude_bias = exclude_bias
        self.exclude_norm = exclude_norm

        # Identificar parámetros a regularizar
        self.params_to_regularize = []
        self.params_to_exclude = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if self._should_exclude_param(name):
                    self.params_to_exclude.append(param)
                else:
                    self.params_to_regularize.append(param)

    def _should_exclude_param(self, name: str) -> bool:
        """Determina si un parámetro debe ser excluido de la regularización"""
        if self.exclude_bias and 'bias' in name:
            return True
        if self.exclude_norm and any(norm in name.lower() for norm in ['norm', 'bn', 'ln']):
            return True
        return False

    def apply_weight_decay(self) -> None:
        """Aplica weight decay a los parámetros seleccionados"""
        with torch.no_grad():
            for param in self.params_to_regularize:
                param.data.mul_(1 - self.weight_decay)


class GradientClippingRegularizer:
    """
    Regularizador de Gradient Clipping adaptativo
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0,
                 adaptive: bool = True):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.adaptive = adaptive
        self.gradient_history = []
        self.adaptation_rate = 0.1

    def clip_gradients(self, model: nn.Module) -> float:
        """Aplica gradient clipping"""
        if self.adaptive:
            self._update_clipping_norm()

        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, norm_type=self.norm_type
        )

        self.gradient_history.append(total_norm.item())
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)

        return total_norm.item()

    def _update_clipping_norm(self) -> None:
        """Actualiza la norma de clipping basado en el historial"""
        if len(self.gradient_history) < 10:
            return

        recent_norms = self.gradient_history[-10:]
        avg_norm = np.mean(recent_norms)

        # Ajustar norma de clipping
        if avg_norm > self.max_norm * 1.5:
            self.max_norm *= (1 + self.adaptation_rate)
        elif avg_norm < self.max_norm * 0.5:
            self.max_norm *= (1 - self.adaptation_rate)


class RegularizationManager:
    """
    Gestor principal de técnicas de regularización
    Coordina todas las técnicas de regularización del modelo
    """

    def __init__(self, model: nn.Module, config: RegularizationConfig):
        self.model = model
        self.config = config
        self.regularizers = {}
        self.augmentations = {}

        # Inicializar regularizadores
        self._initialize_regularizers()
        self._initialize_augmentations()

    def _initialize_regularizers(self) -> None:
        """Inicializa todos los regularizadores"""

        # Weight Decay
        self.regularizers['weight_decay'] = WeightDecayRegularizer(
            self.model, self.config.weight_decay
        )

        # Gradient Clipping
        self.regularizers['gradient_clipping'] = GradientClippingRegularizer(
            self.config.gradient_clipping, adaptive=True
        )

        # Label Smoothing Loss
        self.regularizers['label_smoothing'] = LabelSmoothingCrossEntropy(
            self.config.label_smoothing
        )

    def _initialize_augmentations(self) -> None:
        """Inicializa las técnicas de data augmentation"""

        # MixUp
        if self.config.mixup_alpha > 0:
            self.augmentations['mixup'] = MixUpAugmentation(self.config.mixup_alpha)

        # CutMix
        if self.config.cutmix_alpha > 0:
            self.augmentations['cutmix'] = CutMixAugmentation(self.config.cutmix_alpha)

    def apply_regularization(self, loss: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Aplica todas las técnicas de regularización"""

        # Aplicar weight decay
        self.regularizers['weight_decay'].apply_weight_decay()

        # Aplicar gradient clipping
        grad_norm = self.regularizers['gradient_clipping'].clip_gradients(model)

        return loss

    def apply_augmentation(self, x: torch.Tensor, y: torch.Tensor,
                           augmentation_type: str = 'mixup') -> Tuple[torch.Tensor, torch.Tensor]:
        """Aplica data augmentation"""

        if augmentation_type in self.augmentations:
            aug = self.augmentations[augmentation_type]
            if augmentation_type == 'mixup':
                mixed_x, y_a, y_b, lam = aug(x, y)
                return mixed_x, (y_a, y_b, lam)
            elif augmentation_type == 'cutmix':
                mixed_x, y_a, y_b, lam = aug(x, y)
                return mixed_x, (y_a, y_b, lam)

        return x, y

    def get_regularization_summary(self) -> Dict:
        """Obtiene un resumen del estado de regularización"""
        return {
            'weight_decay_rate': self.config.weight_decay,
            'gradient_clipping_norm': self.config.gradient_clipping,
            'label_smoothing': self.config.label_smoothing,
            'mixup_alpha': self.config.mixup_alpha,
            'cutmix_alpha': self.config.cutmix_alpha,
            'active_regularizers': list(self.regularizers.keys()),
            'active_augmentations': list(self.augmentations.keys())
        }

# Funciones de utilidad


def create_regularization_modules(config: RegularizationConfig) -> Dict[str, nn.Module]:
    """Factory para crear módulos de regularización"""
    modules = {}

    if config.adaptive_dropout:
        modules['adaptive_dropout'] = AdaptiveDropout(config.dropout_rate)

    modules['improved_batch_norm'] = ImprovedBatchNorm(
        num_features=128,  # Se ajustará según el modelo
        eps=config.batch_norm_eps,
        momentum=config.batch_norm_momentum
    )

    modules['layer_norm'] = LayerNormImproved(
        normalized_shape=128,  # Se ajustará según el modelo
        eps=config.layer_norm_eps
    )

    if config.stochastic_depth > 0:
        modules['stochastic_depth'] = StochasticDepth(config.stochastic_depth)

    if config.drop_path_rate > 0:
        modules['drop_path'] = DropPath(config.drop_path_rate)

    return modules


def analyze_regularization_effectiveness(model: nn.Module,
                                         train_losses: List[float],
                                         val_losses: List[float]) -> Dict[str, float]:
    """Analiza la efectividad de las técnicas de regularización"""

    if len(train_losses) < 10 or len(val_losses) < 10:
        return {'status': 'insufficient_data'}

    # Calcular gap de generalización
    generalization_gap = np.mean(val_losses[-10:]) - np.mean(train_losses[-10:])

    # Calcular estabilidad de entrenamiento
    train_stability = 1.0 / (1.0 + np.var(train_losses[-10:]))
    val_stability = 1.0 / (1.0 + np.var(val_losses[-10:]))

    # Calcular tendencia de convergencia
    train_trend = np.polyfit(range(len(train_losses[-10:])), train_losses[-10:], 1)[0]
    val_trend = np.polyfit(range(len(val_losses[-10:])), val_losses[-10:], 1)[0]

    return {
        'generalization_gap': generalization_gap,
        'train_stability': train_stability,
        'val_stability': val_stability,
        'train_convergence': abs(train_trend),
        'val_convergence': abs(val_trend),
        'overfitting_risk': 'high' if generalization_gap > 0.1 else 'low'
    }


# Exportar clases y funciones principales
__all__ = [
    'RegularizationConfig',
    'AdaptiveDropout',
    'ImprovedBatchNorm',
    'LayerNormImproved',
    'StochasticDepth',
    'DropPath',
    'LabelSmoothingCrossEntropy',
    'MixUpAugmentation',
    'CutMixAugmentation',
    'WeightDecayRegularizer',
    'GradientClippingRegularizer',
    'RegularizationManager',
    'create_regularization_modules',
    'analyze_regularization_effectiveness'
]

logger.info("RFEN3_RN_3 - Técnicas de Regularización Avanzadas cargadas correctamente")
